import math
import os
from typing import Optional, Union

import torch
import triton
import triton.language as tl
from einops import rearrange

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.fla.chunk import chunk_gated_delta_rule
from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.layers.attention.fla.fused_gdn_gating import fused_gdn_gating
from sglang.srt.layers.attention.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule_update,
)
from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)
from sglang.srt.layers.attention.fla.kda import (
    chunk_kda,
    fused_kda_gate,
    fused_recurrent_kda,
)
from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
    PAD_SLOT_ID,
    causal_conv1d_fn,
    causal_conv1d_update,
)
from sglang.srt.layers.attention.mamba.mamba import MambaMixer2
from sglang.srt.layers.attention.mamba.mamba2_metadata import (
    ForwardMetadata,
    Mamba2Metadata,
)

# Import Simple GLA from fla if available
try:
    from fla.ops.simple_gla import chunk_simple_gla
    from fla.ops.simple_gla.fused_recurrent import fused_recurrent_simple_gla
    SIMPLE_GLA_AVAILABLE = True
except ImportError:
    SIMPLE_GLA_AVAILABLE = False
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, MambaPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput
from sglang.srt.speculative.spec_info import SpecInput
from sglang.srt.utils import is_cuda, is_npu


if is_cuda():
    from sglang.srt.layers.attention.mamba.causal_conv1d import (
        causal_conv1d_fn as causal_conv1d_fn_cuda,
    )

    causal_conv1d_fn = causal_conv1d_fn_cuda
elif is_npu():
    from sgl_kernel_npu.fla.chunk import chunk_gated_delta_rule_npu
    from sgl_kernel_npu.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update_npu,
    )
    from sgl_kernel_npu.mamba.causal_conv1d import (
        causal_conv1d_fn_npu,
        causal_conv1d_update_npu,
    )

    chunk_gated_delta_rule = chunk_gated_delta_rule_npu
    fused_sigmoid_gating_delta_rule_update = fused_sigmoid_gating_delta_rule_update_npu
    causal_conv1d_fn = causal_conv1d_fn_npu
    causal_conv1d_update = causal_conv1d_update_npu


def _build_slope_tensor(nheads: int) -> torch.Tensor:
    """Build ALiBi slope tensor - matches MiniCPM implementation.

    This function computes the Attention with Linear Biases (ALiBi) slopes
    used in Simple GLA for decay calculation. The slopes are computed using
    a geometric progression with power-of-2 optimization.

    Args:
        nheads: Number of attention heads

    Returns:
        slopes: Tensor of shape (nheads,) containing ALiBi slopes
    """
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
            )

    slopes = torch.tensor(get_slopes(nheads))
    return slopes


# Kernel to track mamba states if needed based on track mask
@triton.jit
def track_mamba_state_if_needed_kernel(
    conv_states_ptr,
    ssm_states_ptr,
    cache_indices_ptr,
    mamba_track_mask_ptr,
    mamba_track_indices_ptr,
    conv_state_stride_0,  # stride for first dimension (batch/pool index)
    ssm_state_stride_0,  # stride for first dimension (batch/pool index)
    conv_state_numel_per_row: tl.constexpr,  # total elements per row
    ssm_state_numel_per_row: tl.constexpr,  # total elements per row
    BLOCK_SIZE: tl.constexpr,
):
    """
    Track conv_states and ssm_states rows based on track mask.

    This kernel replaces a Python loop that copies state tensors for mamba attention.
    For each batch element, if the track mask is True, it copies the entire row from
    the source index (cache_indices[i]) to the destination index (mamba_track_indices[i]).

    Grid: (batch_size,)
    Each block handles one batch element, using multiple threads to copy data in parallel.
    """
    batch_idx = tl.program_id(0)

    # Load the copy mask for this batch element
    track_mask = tl.load(mamba_track_mask_ptr + batch_idx)

    # Early exit if we don't need to track
    if not track_mask:
        return

    # Load source and destination indices
    src_idx = tl.load(cache_indices_ptr + batch_idx)
    dst_idx = tl.load(mamba_track_indices_ptr + batch_idx)

    # Copy conv_states
    # Each thread handles BLOCK_SIZE elements
    for offset in range(0, conv_state_numel_per_row, BLOCK_SIZE):
        element_indices = offset + tl.arange(0, BLOCK_SIZE)
        mask = element_indices < conv_state_numel_per_row

        src_ptr = conv_states_ptr + src_idx * conv_state_stride_0 + element_indices
        dst_ptr = conv_states_ptr + dst_idx * conv_state_stride_0 + element_indices

        data = tl.load(src_ptr, mask=mask, other=0.0)
        tl.store(dst_ptr, data, mask=mask)

    # Copy ssm_states
    for offset in range(0, ssm_state_numel_per_row, BLOCK_SIZE):
        element_indices = offset + tl.arange(0, BLOCK_SIZE)
        mask = element_indices < ssm_state_numel_per_row

        src_ptr = ssm_states_ptr + src_idx * ssm_state_stride_0 + element_indices
        dst_ptr = ssm_states_ptr + dst_idx * ssm_state_stride_0 + element_indices

        data = tl.load(src_ptr, mask=mask, other=0.0)
        tl.store(dst_ptr, data, mask=mask)


def track_mamba_states_if_needed(
    conv_states: torch.Tensor,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    mamba_track_mask: torch.Tensor,
    mamba_track_indices: torch.Tensor,
    batch_size: int,
):
    """
    Track mamba states using Triton kernel for better performance.

    Args:
        conv_states: Convolution states tensor [pool_size, ...]
        ssm_states: SSM states tensor [pool_size, ...]
        cache_indices: Source indices for each batch element [batch_size]
        mamba_track_mask: Boolean mask indicating which elements to track [batch_size]
        mamba_track_indices: Indices to track for each batch element [batch_size]
        batch_size: Number of batch elements
    """
    conv_state_numel_per_row = conv_states[0].numel()
    ssm_state_numel_per_row = ssm_states[0].numel()

    # Choose BLOCK_SIZE based on the size of the data
    BLOCK_SIZE = 1024

    # Launch kernel with batch_size blocks
    grid = (batch_size,)
    track_mamba_state_if_needed_kernel[grid](
        conv_states,
        ssm_states,
        cache_indices,
        mamba_track_mask,
        mamba_track_indices,
        conv_states.stride(0),
        ssm_states.stride(0),
        conv_state_numel_per_row,
        ssm_state_numel_per_row,
        BLOCK_SIZE,
    )


class MambaAttnBackendBase(AttentionBackend):
    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        self.pad_slot_id = PAD_SLOT_ID
        self.device = model_runner.device
        self.req_to_token_pool: HybridReqToTokenPool = model_runner.req_to_token_pool
        self.forward_metadata: ForwardMetadata = None
        self.state_indices_list = []
        self.query_start_loc_list = []
        self.retrieve_next_token_list = []
        self.retrieve_next_sibling_list = []
        self.retrieve_parent_token_list = []
        self.cached_cuda_graph_decode_query_start_loc: torch.Tensor = None
        self.cached_cuda_graph_verify_query_start_loc: torch.Tensor = None
        self.conv_states_shape: tuple[int, int] = None

    def _forward_metadata(self, forward_batch: ForwardBatch):
        bs = forward_batch.batch_size

        retrieve_next_token = None
        retrieve_next_sibling = None
        retrieve_parent_token = None
        track_conv_indices = None
        track_ssm_h_src = None
        track_ssm_h_dst = None
        track_ssm_final_src = None
        track_ssm_final_dst = None

        mamba_cache_indices = self.req_to_token_pool.get_mamba_indices(
            forward_batch.req_pool_indices
        )

        if forward_batch.forward_mode.is_decode_or_idle():
            query_start_loc = torch.arange(
                0, bs + 1, dtype=torch.int32, device=self.device
            )
        elif forward_batch.forward_mode.is_extend():
            if forward_batch.forward_mode.is_target_verify():
                query_start_loc = torch.arange(
                    0,
                    forward_batch.input_ids.shape[0] + 1,
                    step=forward_batch.spec_info.draft_token_num,
                    dtype=torch.int32,
                    device=forward_batch.input_ids.device,
                )

                if forward_batch.spec_info.topk > 1:
                    retrieve_next_token = forward_batch.spec_info.retrive_next_token
                    retrieve_next_sibling = forward_batch.spec_info.retrive_next_sibling
                    # retrieve_next_token is None during dummy run so skip tensor creation
                    if retrieve_next_token is not None:
                        retrieve_parent_token = torch.empty_like(retrieve_next_token)
            else:
                query_start_loc = torch.empty(
                    (bs + 1,), dtype=torch.int32, device=self.device
                )
                query_start_loc[:bs] = forward_batch.extend_start_loc
                query_start_loc[bs] = (
                    forward_batch.extend_start_loc[-1]
                    + forward_batch.extend_seq_lens[-1]
                )
                if (
                    forward_batch.mamba_track_mask is not None
                    and forward_batch.mamba_track_mask.any()
                ):
                    track_conv_indices = self._init_track_conv_indices(
                        query_start_loc, forward_batch
                    )

                    (
                        track_ssm_h_src,
                        track_ssm_h_dst,
                        track_ssm_final_src,
                        track_ssm_final_dst,
                    ) = self._init_track_ssm_indices(mamba_cache_indices, forward_batch)
        else:
            raise ValueError(f"Invalid forward mode: {forward_batch.forward_mode=}")

        return ForwardMetadata(
            query_start_loc=query_start_loc,
            mamba_cache_indices=mamba_cache_indices,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            retrieve_parent_token=retrieve_parent_token,
            track_conv_indices=track_conv_indices,
            track_ssm_h_src=track_ssm_h_src,
            track_ssm_h_dst=track_ssm_h_dst,
            track_ssm_final_src=track_ssm_final_src,
            track_ssm_final_dst=track_ssm_final_dst,
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        self.forward_metadata = self._forward_metadata(forward_batch)

    def _get_effective_extend_seq_lens(self, forward_batch: ForwardBatch) -> torch.Tensor:
        if forward_batch.extend_seq_lens is not None:
            return forward_batch.extend_seq_lens

        if forward_batch.forward_mode.is_target_verify() and forward_batch.spec_info is not None:
            return torch.full(
                (forward_batch.batch_size,),
                forward_batch.spec_info.draft_token_num,
                dtype=torch.int32,
                device=self.device,
            )

        raise RuntimeError(
            "extend_seq_lens is None but cannot infer an effective value for the hybrid attention backend."
        )

    def _get_effective_extend_prefix_lens(
        self, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        if forward_batch.extend_prefix_lens is not None:
            return forward_batch.extend_prefix_lens

        if forward_batch.forward_mode.is_target_verify():
            return forward_batch.seq_lens

        raise RuntimeError(
            "extend_prefix_lens is None but cannot infer an effective value for the hybrid attention backend."
        )

    def _init_track_conv_indices(
        self, query_start_loc: torch.Tensor, forward_batch: ForwardBatch
    ):
        """
        Compute indices for extracting conv states from the input sequence during extend.

        In Mamba models, the conv layer maintains a sliding window of recent inputs.
        After processing a prefill chunk, we need to save the last `conv_state_len` tokens
        of the processed region for prefix caching.

        The key insight is that FLA (Flash Linear Attention) processes sequences in chunks
        of FLA_CHUNK_SIZE. We only track the conv state up to the last complete chunk boundary
        (aligned_len).

        start_indices is the starting token index of the conv state to track in this extend batch.
        indices include all pos to track in this extend batch, conv_state_len for each req that
        needs to be tracked (i.e. mamba_track_mask is True)

        Returns:
            indices: Tensor of shape [num_tracked_requests, conv_state_len] containing
                     flattened positions into the packed input tensor.
        """
        conv_state_len = self.conv_states_shape[-1]

        # Calculate the end position of the last aligned chunk
        lens_to_track = (
            forward_batch.mamba_track_seqlens - forward_batch.extend_prefix_lens
        )
        aligned_len = (lens_to_track // FLA_CHUNK_SIZE) * FLA_CHUNK_SIZE
        start_indices = query_start_loc[:-1] + aligned_len - conv_state_len
        start_indices = start_indices[forward_batch.mamba_track_mask]

        # Create indices: [batch_size, conv_state_len]
        indices = start_indices.unsqueeze(-1) + torch.arange(
            conv_state_len,
            device=self.device,
            dtype=start_indices.dtype,
        )

        return indices.clamp(0, query_start_loc[-1] - 1)

    def _init_track_ssm_indices(
        self, mamba_cache_indices: torch.Tensor, forward_batch: ForwardBatch
    ):
        """
        Compute source and destination indices for tracking SSM states for prefix caching.

        After processing a prefill, we need to save the SSM recurrent state for prefix caching.
        The FLA kernel outputs intermediate hidden states `h` at each chunk boundary,
        plus a `last_recurrent_state` at the end of the chunked prefill size.

        The challenge is that sequences may or may not end on a chunk boundary:
          - Aligned case (len % FLA_CHUNK_SIZE == 0): In this case, FLA will store the to-cache
            state in the last_recurrent_state.
          - Unaligned case (len % FLA_CHUNK_SIZE != 0): The last_recurrent_state includes the
            unaligned position, but we only want state up to the last chunk boundary.
            We must extract from the intermediate `h` tensor at the appropriate chunk index.

        We compute the src and dst indices for all requests that need to be cached
        (i.e. mamba_track_mask is True) based on the rule above.

        For example:
        1. If chunked prefill length is < 64, then only final state has value. In this case we
           cache `final` state.
        2. if chunked prefill length == 64, then only final state has value. In this case we
           cache pos 64, from `final` state
        3. if chunked prefill length >64 and < 128, then both h and final state have value.
           We cache pos 64 from `h` state
        4. if chunked prefill length ==128, then both h and final state have value. We cache
           pos 128 from `final` state. Note `h` doesn't include the pos 128.

        Returns:
            track_ssm_h_src: Source indices into the packed `h` tensor (for unaligned seqs)
            track_ssm_h_dst: Destination cache slot indices (for unaligned seqs)
            track_ssm_final_src: Source indices into last_recurrent_state buffer (for aligned seqs)
            track_ssm_final_dst: Destination cache slot indices (for aligned seqs)
        """
        # Move to CPU to avoid kernel launches for masking operations
        mamba_track_mask = forward_batch.mamba_track_mask.cpu()
        extend_seq_lens = forward_batch.extend_seq_lens.cpu()
        mamba_track_indices = forward_batch.mamba_track_indices.cpu()
        mamba_cache_indices = mamba_cache_indices.cpu()
        mamba_track_seqlens = forward_batch.mamba_track_seqlens.cpu()
        prefix_lens = forward_batch.extend_prefix_lens.cpu()

        # Calculate the number of hidden states per request
        num_h_states = (extend_seq_lens - 1) // FLA_CHUNK_SIZE + 1

        # Calculate the starting offset for each sequence in the packed batch
        track_ssm_src_offset = torch.zeros_like(num_h_states)
        track_ssm_src_offset[1:] = torch.cumsum(num_h_states[:-1], dim=0)

        # Filter variables by track mask
        lens_to_track = mamba_track_seqlens - prefix_lens
        lens_masked = lens_to_track[mamba_track_mask]
        offset_masked = track_ssm_src_offset[mamba_track_mask]
        dst_masked = mamba_track_indices[mamba_track_mask]

        # Determine if the sequence ends at a chunk boundary
        is_aligned = (lens_masked % FLA_CHUNK_SIZE) == 0

        # Case 1: Aligned. Use last_recurrent_state from ssm_states.
        track_ssm_final_src = mamba_cache_indices[mamba_track_mask][is_aligned]
        track_ssm_final_dst = dst_masked[is_aligned]

        # Case 2: Unaligned. Use intermediate state from h.
        # TODO: if support FLA_CHUNK_SIZE % page size != 0, then need to modify this
        not_aligned = ~is_aligned
        track_ssm_h_src = offset_masked[not_aligned] + (
            lens_masked[not_aligned] // FLA_CHUNK_SIZE
        )
        track_ssm_h_dst = dst_masked[not_aligned]

        # Move back to GPU
        return (
            track_ssm_h_src.to(self.device, non_blocking=True),
            track_ssm_h_dst.to(self.device, non_blocking=True),
            track_ssm_final_src.to(self.device, non_blocking=True),
            track_ssm_final_dst.to(self.device, non_blocking=True),
        )

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        self.forward_metadata = self._capture_metadata(
            bs, req_pool_indices, forward_mode, spec_info
        )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
        seq_lens_cpu: Optional[torch.Tensor],
        forward_batch: Optional[ForwardBatch] = None,
    ):
        self.forward_metadata = self._replay_metadata(
            bs, req_pool_indices, forward_mode, spec_info, seq_lens_cpu
        )

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        assert (
            max_num_tokens % max_bs == 0
        ), f"max_num_tokens={max_num_tokens} must be divisible by max_bs={max_bs}"
        draft_token_num = max_num_tokens // max_bs
        for i in range(max_bs):
            self.state_indices_list.append(
                torch.full(
                    (i + 1,), self.pad_slot_id, dtype=torch.int32, device=self.device
                )
            )
            self.query_start_loc_list.append(
                torch.zeros((i + 2,), dtype=torch.int32, device=self.device)
            )
            self.retrieve_next_token_list.append(
                torch.zeros(
                    (i + 1, draft_token_num), dtype=torch.int32, device=self.device
                )
            )
            self.retrieve_next_sibling_list.append(
                torch.zeros(
                    (i + 1, draft_token_num), dtype=torch.int32, device=self.device
                )
            )
            self.retrieve_parent_token_list.append(
                torch.zeros(
                    (i + 1, draft_token_num), dtype=torch.int32, device=self.device
                )
            )
        self.cached_cuda_graph_decode_query_start_loc = torch.arange(
            0, max_bs + 1, dtype=torch.int32, device=self.device
        )
        self.cached_cuda_graph_verify_query_start_loc = torch.arange(
            0,
            max_bs * draft_token_num + 1,
            step=draft_token_num,
            dtype=torch.int32,
            device=self.device,
        )

        # Pre-allocate buffers for _forward_target_verify CUDA graph compatibility.
        # These buffers eliminate dynamic allocations during graph capture/replay.
        # q/k/v step buffers and output buffer are lazily allocated on first
        # capture because num_heads and head_dim are not known at this point.
        self._cg_verify_q_step = None
        self._cg_verify_k_step = None
        self._cg_verify_v_step = None
        self._cg_verify_out = None
        self._cg_verify_max_bs = max_bs
        self._cg_verify_draft_token_num = draft_token_num

    def _capture_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        if forward_mode.is_decode_or_idle():
            self.query_start_loc_list[bs - 1].copy_(
                self.cached_cuda_graph_decode_query_start_loc[: bs + 1]
            )
        elif forward_mode.is_target_verify():
            self.query_start_loc_list[bs - 1].copy_(
                self.cached_cuda_graph_verify_query_start_loc[: bs + 1]
            )
        else:
            raise ValueError(f"Invalid forward mode: {forward_mode=}")
        mamba_indices = self.req_to_token_pool.get_mamba_indices(req_pool_indices)
        self.state_indices_list[bs - 1][: len(mamba_indices)].copy_(mamba_indices)

        # If topk > 1, we need to use retrieve_next_token and retrieve_next_sibling to handle the eagle tree custom attention mask
        if forward_mode.is_target_verify() and spec_info.topk > 1:
            # They are None during cuda graph capture so skip the copy_...
            # self.retrieve_next_token_list[bs - 1].copy_(spec_info.retrive_next_token)
            # self.retrieve_next_sibling_list[bs - 1].copy_(spec_info.retrive_next_sibling)
            return ForwardMetadata(
                query_start_loc=self.query_start_loc_list[bs - 1],
                mamba_cache_indices=self.state_indices_list[bs - 1],
                retrieve_next_token=self.retrieve_next_token_list[bs - 1],
                retrieve_next_sibling=self.retrieve_next_sibling_list[bs - 1],
                retrieve_parent_token=self.retrieve_parent_token_list[bs - 1],
            )
        else:
            return ForwardMetadata(
                query_start_loc=self.query_start_loc_list[bs - 1],
                mamba_cache_indices=self.state_indices_list[bs - 1],
            )

    def _replay_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        num_padding = torch.count_nonzero(
            seq_lens_cpu == self.get_cuda_graph_seq_len_fill_value()
        )
        # Make sure forward metadata is correctly handled for padding reqs
        req_pool_indices[bs - num_padding :] = 0
        mamba_indices = self.req_to_token_pool.get_mamba_indices(req_pool_indices)
        mamba_indices[bs - num_padding :] = -1
        self.state_indices_list[bs - 1][: len(mamba_indices)].copy_(mamba_indices)
        if forward_mode.is_decode_or_idle():
            if num_padding == 0:
                self.query_start_loc_list[bs - 1].copy_(
                    self.cached_cuda_graph_decode_query_start_loc[: bs + 1]
                )
            else:
                self.query_start_loc_list[bs - 1][: bs - num_padding].copy_(
                    self.cached_cuda_graph_decode_query_start_loc[: bs - num_padding]
                )
                self.query_start_loc_list[bs - 1][bs - num_padding :].copy_(
                    bs - num_padding
                )
        elif forward_mode.is_target_verify():
            if num_padding == 0:
                self.query_start_loc_list[bs - 1].copy_(
                    self.cached_cuda_graph_verify_query_start_loc[: bs + 1]
                )
            else:
                self.query_start_loc_list[bs - 1][: bs - num_padding].copy_(
                    self.cached_cuda_graph_verify_query_start_loc[: bs - num_padding]
                )
                self.query_start_loc_list[bs - 1][bs - num_padding :].copy_(
                    (bs - num_padding) * spec_info.draft_token_num
                )
        else:
            raise ValueError(f"Invalid forward mode: {forward_mode=}")

        # If topk > 1, we need to use retrieve_next_token and retrieve_next_sibling to handle the eagle tree custom attention mask
        if forward_mode.is_target_verify() and spec_info.topk > 1:
            bs_without_pad = spec_info.retrive_next_token.shape[0]
            self.retrieve_next_token_list[bs - 1][:bs_without_pad].copy_(
                spec_info.retrive_next_token
            )
            self.retrieve_next_sibling_list[bs - 1][:bs_without_pad].copy_(
                spec_info.retrive_next_sibling
            )
            return ForwardMetadata(
                query_start_loc=self.query_start_loc_list[bs - 1],
                mamba_cache_indices=self.state_indices_list[bs - 1],
                retrieve_next_token=self.retrieve_next_token_list[bs - 1],
                retrieve_next_sibling=self.retrieve_next_sibling_list[bs - 1],
                retrieve_parent_token=self.retrieve_parent_token_list[bs - 1],
            )
        else:
            return ForwardMetadata(
                query_start_loc=self.query_start_loc_list[bs - 1],
                mamba_cache_indices=self.state_indices_list[bs - 1],
            )

    def get_cuda_graph_seq_len_fill_value(self):
        return 1  # Mamba attn does not use seq lens to index kv cache

    def _track_mamba_state_decode(
        self,
        forward_batch: ForwardBatch,
        conv_states: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
    ):
        """
        Track and copy Mamba conv/SSM states during decode for prefix caching.

        During decode, each token update modifies conv_states and ssm_states in-place
        at positions indexed by cache_indices (the working slots). For prefix caching,
        we need to copy these updated states to persistent cache slots (mamba_track_indices)
        so they can be prefix cached.

        This delegates to `track_mamba_states_if_needed`, which performs:
            conv_states[mamba_track_indices[i]] = conv_states[cache_indices[i]]
            ssm_states[mamba_track_indices[i]] = ssm_states[cache_indices[i]]
        for all requests where mamba_track_mask[i] is True.
        """
        if forward_batch.mamba_track_mask is not None:
            track_mamba_states_if_needed(
                conv_states,
                ssm_states,
                cache_indices,
                forward_batch.mamba_track_mask,
                forward_batch.mamba_track_indices,
                forward_batch.batch_size,
            )

    def _track_mamba_state_extend(
        self,
        forward_batch: ForwardBatch,
        h: torch.Tensor,
        ssm_states: torch.Tensor,
        forward_metadata: ForwardMetadata,
    ):
        """
        Track and copy SSM states during extend for prefix caching.

        After the FLA chunked prefill kernel runs, we need to save the SSM recurrent
        state at the last chunk boundary so it can be reused for prefix caching.
        The source of the state depends on whether the sequence length is aligned
        to FLA_CHUNK_SIZE. See `_init_track_ssm_indices` for more details on how
        the source and destination indices are computed.

        Note: Conv state tracking for extend is handled separately via gather operations
        using indices computed by `_init_track_conv_indices`.
        """
        if (
            forward_batch.mamba_track_mask is not None
            and forward_batch.mamba_track_mask.any()
        ):
            h = h.squeeze(0)

            if forward_metadata.track_ssm_h_src.numel() > 0:
                ssm_states[forward_metadata.track_ssm_h_dst] = h[
                    forward_metadata.track_ssm_h_src
                ].to(ssm_states.dtype, copy=False)
            if forward_metadata.track_ssm_final_src.numel() > 0:
                ssm_states[forward_metadata.track_ssm_final_dst] = ssm_states[
                    forward_metadata.track_ssm_final_src
                ]


class KimiLinearAttnBackend(MambaAttnBackendBase):
    """Attention backend using Mamba kernel."""

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        q_proj_states = kwargs["q_proj_states"]
        k_proj_states = kwargs["k_proj_states"]
        v_proj_states = kwargs["v_proj_states"]
        q_conv_weights = kwargs["q_conv_weights"]
        k_conv_weights = kwargs["k_conv_weights"]
        v_conv_weights = kwargs["v_conv_weights"]

        q_conv_bias = kwargs["q_conv_bias"]
        k_conv_bias = kwargs["k_conv_bias"]
        v_conv_bias = kwargs["v_conv_bias"]

        A_log = kwargs["A_log"]
        dt_bias = kwargs["dt_bias"]
        b_proj = kwargs["b_proj"]
        f_a_proj = kwargs["f_a_proj"]
        f_b_proj = kwargs["f_b_proj"]
        hidden_states = kwargs["hidden_states"]
        head_dim = kwargs["head_dim"]
        layer_id = kwargs["layer_id"]

        layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer_id)
        q_conv_state, k_conv_state, v_conv_state = layer_cache.conv
        ssm_states = layer_cache.temporal
        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices

        q_conv_state = q_conv_state.transpose(-1, -2)
        k_conv_state = k_conv_state.transpose(-1, -2)
        v_conv_state = v_conv_state.transpose(-1, -2)

        q = causal_conv1d_update(
            q_proj_states,
            q_conv_state,
            q_conv_weights,
            q_conv_bias,
            activation="silu",
            conv_state_indices=cache_indices,
        )
        k = causal_conv1d_update(
            k_proj_states,
            k_conv_state,
            k_conv_weights,
            k_conv_bias,
            activation="silu",
            conv_state_indices=cache_indices,
        )
        v = causal_conv1d_update(
            v_proj_states,
            v_conv_state,
            v_conv_weights,
            v_conv_bias,
            activation="silu",
            conv_state_indices=cache_indices,
        )

        q, k, v = map(
            lambda x: rearrange(x, "n (h d) -> 1 n h d", d=head_dim), (q, k, v)
        )

        beta = b_proj(hidden_states)[0].float().sigmoid()

        g = f_b_proj(f_a_proj(hidden_states)[0])[0]
        g = fused_kda_gate(g, A_log, head_dim, g_bias=dt_bias)

        beta = beta.unsqueeze(0)
        g = g.unsqueeze(0)

        initial_state = ssm_states[cache_indices].contiguous()
        (
            core_attn_out,
            last_recurrent_state,
        ) = fused_recurrent_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=query_start_loc,
        )
        ssm_states[cache_indices] = last_recurrent_state
        return core_attn_out

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
            causal_conv1d_fn,
        )

        q_proj_states = kwargs["q_proj_states"]
        k_proj_states = kwargs["k_proj_states"]
        v_proj_states = kwargs["v_proj_states"]
        q_conv_weights = kwargs["q_conv_weights"]
        k_conv_weights = kwargs["k_conv_weights"]
        v_conv_weights = kwargs["v_conv_weights"]

        q_conv_bias = kwargs["q_conv_bias"]
        k_conv_bias = kwargs["k_conv_bias"]
        v_conv_bias = kwargs["v_conv_bias"]

        A_log = kwargs["A_log"]
        dt_bias = kwargs["dt_bias"]
        b_proj = kwargs["b_proj"]
        f_a_proj = kwargs["f_a_proj"]
        f_b_proj = kwargs["f_b_proj"]
        hidden_states = kwargs["hidden_states"]
        head_dim = kwargs["head_dim"]
        layer_id = kwargs["layer_id"]

        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices

        mamba_cache_params = self.req_to_token_pool.mamba2_layer_cache(layer_id)
        conv_state_q, conv_state_k, conv_state_v = mamba_cache_params.conv
        # deal with strides
        conv_state_q = conv_state_q.transpose(-1, -2)
        conv_state_k = conv_state_k.transpose(-1, -2)
        conv_state_v = conv_state_v.transpose(-1, -2)

        ssm_states = mamba_cache_params.temporal

        has_initial_state = forward_batch.extend_prefix_lens > 0

        q_proj_states = q_proj_states.transpose(0, 1)
        k_proj_states = k_proj_states.transpose(0, 1)
        v_proj_states = v_proj_states.transpose(0, 1)

        q = causal_conv1d_fn(
            q_proj_states,
            q_conv_weights,
            q_conv_bias,
            activation="silu",
            conv_states=conv_state_q,
            has_initial_state=has_initial_state,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
        ).transpose(0, 1)

        k = causal_conv1d_fn(
            k_proj_states,
            k_conv_weights,
            k_conv_bias,
            activation="silu",
            conv_states=conv_state_k,
            has_initial_state=has_initial_state,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
        ).transpose(0, 1)

        v = causal_conv1d_fn(
            v_proj_states,
            v_conv_weights,
            v_conv_bias,
            activation="silu",
            conv_states=conv_state_v,
            has_initial_state=has_initial_state,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
        ).transpose(0, 1)

        q, k, v = map(
            lambda x: rearrange(x, "n (h d) -> 1 n h d", d=head_dim), (q, k, v)
        )

        beta = b_proj(hidden_states)[0].float().sigmoid()

        g = f_b_proj(f_a_proj(hidden_states)[0])[0]
        g = fused_kda_gate(g, A_log, head_dim, g_bias=dt_bias)

        beta = beta.unsqueeze(0)
        g = g.unsqueeze(0)

        core_attn_out = chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=ssm_states,
            initial_state_indices=cache_indices,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=query_start_loc,
        )

        return core_attn_out


class GDNAttnBackend(MambaAttnBackendBase):
    """Attention backend using Mamba kernel."""

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        self.conv_states_shape = (
            model_runner.req_to_token_pool.mamba_pool.mamba_cache.conv[0].shape
        )
        assert (
            self.conv_states_shape[-1] < FLA_CHUNK_SIZE
        ), f"{self.conv_states_shape[-1]=} should be less than {FLA_CHUNK_SIZE}"

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        mixed_qkv = kwargs["mixed_qkv"]
        conv_weights = kwargs["conv_weights"]
        bias = kwargs["bias"]
        activation = kwargs["activation"]
        key_dim = kwargs["key_dim"]
        value_dim = kwargs["value_dim"]
        attn_tp_size = kwargs["attention_tp_size"]
        head_k_dim = kwargs["head_k_dim"]
        head_v_dim = kwargs["head_v_dim"]
        a = kwargs["a"]
        b = kwargs["b"]
        A_log = kwargs["A_log"]
        dt_bias = kwargs["dt_bias"]
        layer_id = kwargs["layer_id"]

        layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer_id)
        conv_states = layer_cache.conv[0]
        ssm_states = layer_cache.temporal
        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices

        mixed_qkv = causal_conv1d_update(
            mixed_qkv,
            conv_states,
            conv_weights,
            bias,
            activation,
            conv_state_indices=cache_indices,
        )

        query, key, value = torch.split(
            mixed_qkv,
            [
                key_dim // attn_tp_size,
                key_dim // attn_tp_size,
                value_dim // attn_tp_size,
            ],
            dim=-1,
        )
        # Reshape from [l, h*d] to [1, l, h, d]
        seq_len = query.shape[0]
        num_heads = query.shape[1] // head_k_dim
        query = query.view(1, seq_len, num_heads, head_k_dim)
        key = key.view(1, seq_len, num_heads, head_k_dim)
        value = value.view(1, seq_len, value.shape[1] // head_v_dim, head_v_dim)

        core_attn_out = fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            dt_bias=dt_bias,
            q=query,
            k=key,
            v=value,
            a=a,
            b=b,
            initial_state_source=ssm_states,
            initial_state_indices=cache_indices,
            cu_seqlens=query_start_loc,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
        )

        self._track_mamba_state_decode(
            forward_batch, conv_states, ssm_states, cache_indices
        )

        return core_attn_out

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        mixed_qkv = kwargs["mixed_qkv"]
        conv_weights = kwargs["conv_weights"]
        bias = kwargs["bias"]
        activation = kwargs["activation"]
        key_dim = kwargs["key_dim"]
        value_dim = kwargs["value_dim"]
        attn_tp_size = kwargs["attention_tp_size"]
        head_k_dim = kwargs["head_k_dim"]
        head_v_dim = kwargs["head_v_dim"]
        a = kwargs["a"]
        b = kwargs["b"]
        A_log = kwargs["A_log"]
        dt_bias = kwargs["dt_bias"]
        layer_id = kwargs["layer_id"]
        seq_len = kwargs["seq_len"]

        is_target_verify = forward_batch.forward_mode.is_target_verify()
        forward_metadata = self.forward_metadata

        query_start_loc = forward_metadata.query_start_loc
        cache_indices = forward_metadata.mamba_cache_indices
        retrieve_next_token = forward_metadata.retrieve_next_token
        retrieve_next_sibling = forward_metadata.retrieve_next_sibling
        retrieve_parent_token = forward_metadata.retrieve_parent_token

        mamba_cache_params = self.req_to_token_pool.mamba2_layer_cache(layer_id)
        conv_states = mamba_cache_params.conv[0]
        ssm_states = mamba_cache_params.temporal
        if is_target_verify:
            assert isinstance(mamba_cache_params, MambaPool.SpeculativeState)
            intermediate_state_cache = mamba_cache_params.intermediate_ssm
            intermediate_conv_window_cache = (
                mamba_cache_params.intermediate_conv_window[0]
            )
            has_initial_states = torch.ones(
                seq_len // forward_batch.spec_info.draft_token_num,
                dtype=torch.bool,
                device=forward_batch.input_ids.device,
            )
            intermediate_state_indices = torch.arange(
                cache_indices.shape[0], dtype=torch.int32, device=cache_indices.device
            )
        else:
            has_initial_states = forward_batch.extend_prefix_lens > 0

        if is_target_verify:
            batch_size = seq_len // forward_batch.spec_info.draft_token_num
            draft_token_num = forward_batch.spec_info.draft_token_num

            # Diagnostic: optionally disable GDN tree routing to isolate bug source
            # Uses file flag /tmp/SGLANG_DISABLE_GDN_TREE (survives tmux)
            import os
            _diag_disable_gdn_tree = os.path.exists('/tmp/SGLANG_DISABLE_GDN_TREE')
            if _diag_disable_gdn_tree and retrieve_next_token is not None:
                retrieve_next_token = None
                retrieve_next_sibling = None
                retrieve_parent_token = None

            mixed_qkv_reshaped = mixed_qkv.view(
                batch_size, draft_token_num, -1
            ).transpose(1, 2)
            mixed_qkv_processed = causal_conv1d_update(
                mixed_qkv_reshaped,
                conv_states,
                conv_weights,
                bias,
                activation,
                conv_state_indices=cache_indices[:batch_size],
                intermediate_conv_window=intermediate_conv_window_cache,
                intermediate_state_indices=intermediate_state_indices[:batch_size],
                retrieve_next_token=retrieve_next_token,
                retrieve_next_sibling=retrieve_next_sibling,
                retrieve_parent_token=retrieve_parent_token,
            )
            mixed_qkv = mixed_qkv_processed.transpose(1, 2).view(seq_len, -1)
        else:
            mixed_qkv = mixed_qkv.transpose(0, 1)
            if (
                forward_batch.mamba_track_mask is not None
                and forward_batch.mamba_track_mask.any()
            ):
                conv_dst = forward_batch.mamba_track_indices
                # Gather all slices at once: [:, track_conv_indices] -> [d, num_masked, slice_len]
                # track_conv_indices is already filtered and clamped in _init_track_conv_indices
                mixed_qkv_to_track = mixed_qkv[
                    :, forward_metadata.track_conv_indices
                ].transpose(0, 1)
                # Apply mask and assign to destinations
                mask_indices = forward_batch.mamba_track_mask.nonzero(as_tuple=True)[0]
                conv_states[conv_dst[mask_indices]] = mixed_qkv_to_track

            mixed_qkv = causal_conv1d_fn(
                mixed_qkv,
                conv_weights,
                bias,
                activation=activation,
                conv_states=conv_states,
                has_initial_state=has_initial_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
            ).transpose(0, 1)[:seq_len]

        key_split_dim = key_dim // attn_tp_size
        value_split_dim = value_dim // attn_tp_size

        query, key, value = torch.split(
            mixed_qkv,
            [key_split_dim, key_split_dim, value_split_dim],
            dim=-1,
        )

        actual_seq_len = query.shape[0]
        num_heads = query.shape[1] // head_k_dim
        num_value_heads = value.shape[1] // head_v_dim

        query = query.view(1, actual_seq_len, num_heads, head_k_dim)
        key = key.view(1, actual_seq_len, num_heads, head_k_dim)
        value = value.view(1, actual_seq_len, num_value_heads, head_v_dim)

        g, beta = fused_gdn_gating(A_log, a, b, dt_bias)

        if is_target_verify:
            # When GDN tree is disabled, also disable for SSM path
            _gdn_rpt = retrieve_parent_token
            if _diag_disable_gdn_tree:
                _gdn_rpt = None
            core_attn_out = fused_recurrent_gated_delta_rule_update(
                q=query,
                k=key,
                v=value,
                g=g,
                beta=beta,
                initial_state_source=ssm_states,
                initial_state_indices=cache_indices,
                cu_seqlens=query_start_loc,
                use_qk_l2norm_in_kernel=True,
                disable_state_update=True,
                intermediate_states_buffer=intermediate_state_cache,
                intermediate_state_indices=intermediate_state_indices,
                cache_steps=forward_batch.spec_info.draft_token_num,
                retrieve_parent_token=_gdn_rpt,
            )
        else:
            # Only cuda env uses fuse ssm_states update
            recurrent_state = ssm_states
            recurrent_state_indices_args = {"initial_state_indices": cache_indices}
            if is_npu():
                recurrent_state = ssm_states[cache_indices]
                recurrent_state_indices_args = {}
            core_attn_out, last_recurrent_state, h = chunk_gated_delta_rule(
                q=query,
                k=key,
                v=value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                cu_seqlens=query_start_loc,
                head_first=False,
                use_qk_l2norm_in_kernel=True,
                **recurrent_state_indices_args,
            )
            if is_npu():
                last_recurrent_state = last_recurrent_state.to(
                    ssm_states.dtype, copy=False
                )
                ssm_states[cache_indices] = last_recurrent_state

            self._track_mamba_state_extend(
                forward_batch, h, ssm_states, forward_metadata
            )

        return core_attn_out


class Mamba2AttnBackend(MambaAttnBackendBase):
    """Attention backend wrapper for Mamba2Mixer kernels."""

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        config = model_runner.mamba2_config
        assert config is not None
        self.mamba_chunk_size = config.mamba_chunk_size

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        metadata = self._forward_metadata(forward_batch)
        self.forward_metadata = Mamba2Metadata.prepare_mixed(
            metadata,
            self.mamba_chunk_size,
            forward_batch,
        )

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        metadata = self._capture_metadata(bs, req_pool_indices, forward_mode, spec_info)
        draft_token_num = spec_info.draft_token_num if spec_info is not None else 1
        self.forward_metadata = Mamba2Metadata.prepare_decode(
            metadata,
            seq_lens,
            is_target_verify=forward_mode.is_target_verify(),
            draft_token_num=draft_token_num,
        )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
        seq_lens_cpu: Optional[torch.Tensor],
        forward_batch: Optional[ForwardBatch] = None,
    ):
        metadata = self._replay_metadata(
            bs, req_pool_indices, forward_mode, spec_info, seq_lens_cpu
        )
        draft_token_num = spec_info.draft_token_num if spec_info is not None else 1
        self.forward_metadata = Mamba2Metadata.prepare_decode(
            metadata,
            seq_lens,
            is_target_verify=forward_mode.is_target_verify(),
            draft_token_num=draft_token_num,
        )

    def forward(
        self,
        mixer: MambaMixer2,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
        layer_id: int,
        mup_vector: Optional[torch.Tensor] = None,
        use_triton_causal_conv: bool = False,
    ):
        assert isinstance(self.forward_metadata, Mamba2Metadata)
        layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer_id)
        return mixer.forward(
            hidden_states=hidden_states,
            output=output,
            layer_cache=layer_cache,
            metadata=self.forward_metadata,
            mup_vector=mup_vector,
            use_triton_causal_conv=use_triton_causal_conv,
        )

    def forward_decode(self, *args, **kwargs):
        raise NotImplementedError(
            "Mamba2AttnBackend's forward is called directly instead of through HybridLinearAttnBackend, as it supports mixed prefill and decode"
        )

    def forward_extend(self, *args, **kwargs):
        raise NotImplementedError(
            "Mamba2AttnBackend's forward is called directly instead of through HybridLinearAttnBackend, as it supports mixed prefill and decode"
        )


class HybridLinearAttnBackend(AttentionBackend):
    """Manages a full and linear attention backend"""

    def __init__(
        self,
        full_attn_backend: AttentionBackend,
        linear_attn_backend: MambaAttnBackendBase,
        full_attn_layers: list[int],
    ):
        self.full_attn_layers = full_attn_layers
        self.full_attn_backend = full_attn_backend
        self.linear_attn_backend = linear_attn_backend
        self.attn_backend_list = [full_attn_backend, linear_attn_backend]

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        for attn_backend in self.attn_backend_list:
            attn_backend.init_forward_metadata(forward_batch)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        for attn_backend in self.attn_backend_list:
            attn_backend.init_cuda_graph_state(max_bs, max_num_tokens)

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ):
        for attn_backend in self.attn_backend_list:
            attn_backend.init_forward_metadata_capture_cuda_graph(
                bs,
                num_tokens,
                req_pool_indices,
                seq_lens,
                encoder_lens,
                forward_mode,
                spec_info,
            )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
        forward_batch: Optional[ForwardBatch] = None,
    ):
        for attn_backend in self.attn_backend_list:
            attn_backend.init_forward_metadata_replay_cuda_graph(
                bs,
                req_pool_indices,
                seq_lens,
                seq_lens_sum,
                encoder_lens,
                forward_mode,
                spec_info,
                seq_lens_cpu,
                forward_batch=forward_batch,
            )

    def get_cuda_graph_seq_len_fill_value(self):
        return self.full_attn_backend.get_cuda_graph_seq_len_fill_value()

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        layer_id = layer.layer_id if layer else kwargs["layer_id"]
        if layer_id in self.full_attn_layers:
            return self.full_attn_backend.forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )
        return self.linear_attn_backend.forward_decode(
            q, k, v, layer, forward_batch, save_kv_cache, **kwargs
        )

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        layer_id = layer.layer_id if layer else kwargs["layer_id"]
        if layer_id in self.full_attn_layers:
            return self.full_attn_backend.forward_extend(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )
        return self.linear_attn_backend.forward_extend(
            q, k, v, layer, forward_batch, save_kv_cache, **kwargs
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        """Run forward on an attention layer."""
        if forward_batch.forward_mode.is_idle():
            if layer is None:
                return torch.empty_like(kwargs["z"])
            return q.new_empty(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
        elif forward_batch.forward_mode.is_decode():
            return self.forward_decode(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
                **kwargs,
            )
        else:
            return self.forward_extend(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
                **kwargs,
            )

    def update_mamba_state_after_mtp_verify(
        self,
        accepted_steps: torch.Tensor,
        mamba_track_indices: Optional[torch.Tensor] = None,
        mamba_steps_to_track: Optional[torch.Tensor] = None,
        model=None,
    ):
        """Update recurrent states after speculative verify based on accepted tokens.

        Supports two calling conventions:
        - 2-arg: (accepted_steps, model) -- from multi_layer_eagle_worker
        - 4-arg: (accepted_steps, mamba_track_indices, mamba_steps_to_track, model) -- from eagle_worker
        """
        # Handle 2-arg calling convention from multi_layer_eagle_worker:
        # update_mamba_state_after_mtp_verify(steps, model_obj) where model_obj
        # is passed as the positional arg 'mamba_track_indices'.
        if model is None and mamba_steps_to_track is None and mamba_track_indices is not None:
            if not isinstance(mamba_track_indices, torch.Tensor):
                # It's actually the model object, not a tensor
                model = mamba_track_indices
                mamba_track_indices = None


        request_number = accepted_steps.shape[0]

        state_indices_tensor = (
            self.linear_attn_backend.forward_metadata.mamba_cache_indices[
                :request_number
            ]
        )
        intermediate_state_indices = torch.arange(
            request_number, dtype=torch.int32, device=state_indices_tensor.device
        )

        mamba_caches = (
            self.linear_attn_backend.req_to_token_pool.get_speculative_mamba2_params_all_layers()
        )

        has_conv = mamba_caches.conv and len(mamba_caches.conv) > 0
        if has_conv:
            conv_states = mamba_caches.conv[0]
            intermediate_conv_window_cache = mamba_caches.intermediate_conv_window[0]
        ssm_states = mamba_caches.temporal
        intermediate_state_cache = mamba_caches.intermediate_ssm

        # Compute common indices once to avoid duplication
        valid_mask = accepted_steps >= 0
        dst_state_indices = state_indices_tensor[valid_mask].to(torch.int64)  # [N]
        src_state_indices = intermediate_state_indices[valid_mask].to(
            torch.int64
        )  # [N]
        last_steps = accepted_steps[valid_mask].to(torch.int64)  # [N]

        # [DIAG-H4-STATE] Log state recovery details
        import os
        if os.environ.get('SGLANG_DIAG_STATE2', '0') == '1':
            if not hasattr(self, '_diag_state_count'):
                self._diag_state_count = 0
            if self._diag_state_count < 20:
                restored = intermediate_state_cache[:, src_state_indices, last_steps]
                print(f"[DIAG-STATE] layer=mamba_update "
                      f"dst={dst_state_indices.tolist()} src={src_state_indices.tolist()} "
                      f"last_steps={last_steps.tolist()} "
                      f"restored_hash={restored.sum().item():.6f} "
                      f"ssm_before_hash={ssm_states[:, dst_state_indices].sum().item():.6f}",
                      flush=True)
                self._diag_state_count += 1

        # scatter into ssm_states at the chosen cache lines
        ssm_states[:, dst_state_indices, :] = intermediate_state_cache[
            :, src_state_indices, last_steps
        ].to(ssm_states.dtype, copy=False)

        # Scatter into conv_states at the chosen cache lines (if conv states exist)
        if has_conv:
            conv_states[:, dst_state_indices, :] = intermediate_conv_window_cache[
                :, src_state_indices, last_steps
            ].to(conv_states.dtype, copy=False)

        # Track indices used for tracking mamba states for prefix cache
        if mamba_track_indices is not None:
            assert mamba_steps_to_track is not None
            track_mask = mamba_steps_to_track >= 0
            track_steps = mamba_steps_to_track[track_mask].to(torch.int64)  # [N]
            if track_steps.numel() == 0:
                # No track indices to update
                return
            dst_track_indices = mamba_track_indices[track_mask].to(torch.int64)
            src_track_indices = intermediate_state_indices[track_mask].to(torch.int64)

            # scatter into ssm_states at the chosen track states
            ssm_states[:, dst_track_indices, :] = intermediate_state_cache[
                :, src_track_indices, track_steps
            ].to(ssm_states.dtype, copy=False)

            # scatter into conv_states at the chosen track states (if conv states exist)
            if has_conv:
                conv_states[:, dst_track_indices, :] = intermediate_conv_window_cache[
                    :, src_track_indices, track_steps
                ].to(conv_states.dtype, copy=False)


class SimpleGLAAttnBackend(MambaAttnBackendBase):
    """Attention backend for MiniCPM hybrid models using the SimpleGLA CUDA kernels.

    This backend assumes the model's ``mixer_types`` includes ``"lightning-attn"`` and
    that the optional ``fla`` package is installed. It does **not** perform any
    convolution or Mamba‑style processing; instead it forwards the query, key and
    value tensors directly to ``parallel_simple_gla``.

    If the ``fla`` package is missing an ``ImportError`` is raised during
    construction, allowing the caller to fall back to an alternative backend such
    as ``Mamba2AttnBackend``.
    """

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        minicpm_config = getattr(model_runner, 'minicpm_hybrid_config', None)
        assert minicpm_config is not None, "minicpm_hybrid_config is required for SimpleGLA backend"

        self.conv_states_shape = None

        tp_size = get_tensor_model_parallel_world_size()
        total_num_heads = minicpm_config.lightning_nkv or 16
        # Must divide by tp_size, same as OLD implementation
        assert total_num_heads % tp_size == 0, f"lightning_nkv ({total_num_heads}) must be divisible by tp_size ({tp_size})"
        num_heads = total_num_heads // tp_size
        self.num_heads = num_heads

        self.g_gamma = (
            _build_slope_tensor(num_heads).to(dtype=torch.float32, device=self.device)
            * (-1.0)
        )  # (h)

        head_dim = getattr(minicpm_config, 'lightning_head_dim', 128)
        scale_config = getattr(minicpm_config, 'lightning_scale', '1/sqrt(d)')
        if scale_config == '1/sqrt(d)':
            self.scale = head_dim ** (-0.5)
        elif scale_config == '1/d':
            self.scale = head_dim ** (-1.0)
        else:
            self.scale = 1.0

        if not SIMPLE_GLA_AVAILABLE:
            raise ImportError(
                "Simple GLA backend requested but the 'fla' package is not installed. "
                "Install it or configure the model to use a supported attention backend (e.g., Mamba2)."
            )

        # Pre-allocated staging buffers for _forward_target_verify.
        # Lazily allocated on first call (need H, D from actual tensors).
        # Initialized here so eager mode (no CUDA graph) also works.
        self._cg_verify_q_step = None
        self._cg_verify_k_step = None
        self._cg_verify_v_step = None
        self._cg_verify_out = None
        self._cg_verify_max_bs = 0
        self._cg_verify_draft_token_num = 0

    def _get_mamba_indices(self, forward_batch: ForwardBatch) -> torch.Tensor:
        """Get mamba cache indices with fallback logic.

        First tries to get from forward_metadata.mamba_cache_indices.
        If not available, falls back to req_to_token_pool.get_mamba_indices().

        Args:
            forward_batch: Forward batch containing forward metadata

        Returns:
            mamba_indices: Tensor of cache indices for each batch element

        Raises:
            RuntimeError: If both sources fail to provide indices
        """
        if (hasattr(self, 'forward_metadata')
            and self.forward_metadata is not None
            and hasattr(self.forward_metadata, 'mamba_cache_indices')
            and self.forward_metadata.mamba_cache_indices is not None):
            return self.forward_metadata.mamba_cache_indices
        else:
            return self.req_to_token_pool.get_mamba_indices(
                forward_batch.req_pool_indices
            )

    def _init_track_conv_indices(
        self, query_start_loc: torch.Tensor, forward_batch: ForwardBatch
    ):
        """
        Simple GLA doesn't use convolution states, so return None.
        This overrides the parent class method that would fail without conv_states_shape.
        """
        return None

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        metadata = self._forward_metadata(forward_batch)
        self.forward_metadata = metadata

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ):
        metadata = self._capture_metadata(bs, req_pool_indices, forward_mode, spec_info)
        self.forward_metadata = metadata

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
        forward_batch: Optional[ForwardBatch] = None,
    ):
        metadata = self._replay_metadata(bs, req_pool_indices, forward_mode, spec_info, seq_lens_cpu)
        self.forward_metadata = metadata

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        output_attentions: bool = False,
    ) -> torch.Tensor:

        num_heads = q.shape[2]
        head_dim = q.shape[3]
        effective_extend_seq_lens = None
        effective_extend_prefix_lens = None
        is_target_verify = forward_batch.forward_mode.is_target_verify()

        if forward_batch.forward_mode.is_decode() or is_target_verify:
            seq_len = 1
        else:
            effective_extend_seq_lens = self._get_effective_extend_seq_lens(
                forward_batch
            )
            seq_len = torch.max(effective_extend_seq_lens)

        mamba_indices = self._get_mamba_indices(forward_batch)
        initial_state = None
        cache_idx = self.req_to_token_pool.mamba_map.get(layer_id)
        if cache_idx is None:
            raise RuntimeError(
                f"SimpleGLAAttnBackend layer {layer_id} is missing from mamba_map. "
                f"This indicates a misconfiguration - lightning layers must be registered in cache_params.layers. "
                f"Available layers: {list(self.req_to_token_pool.mamba_map.keys())}"
            )
        layer_cache = self.req_to_token_pool.mamba_pool.mamba2_layer_cache(cache_idx)

        if forward_batch.forward_mode.is_decode() or is_target_verify:
            # decode and target_verify always need initial_state; skip .any() which
            # would cause a CPU-sync forbidden during CUDA graph capture.
            has_initial_state = True
        else:
            effective_extend_prefix_lens = self._get_effective_extend_prefix_lens(
                forward_batch
            )
            has_initial_state = effective_extend_prefix_lens > 0
        if forward_batch.forward_mode.is_decode() or is_target_verify or (hasattr(has_initial_state, 'any') and has_initial_state.any()):
            initial_state = layer_cache.temporal[mamba_indices, :].contiguous()

        scale = self.scale
        g_gamma = self.g_gamma

        if is_target_verify:
            # TARGET_VERIFY: process draft tokens step-by-step and cache
            # intermediate states so update_mamba_state_after_mtp_verify can
            # pick the state corresponding to the last accepted token.
            o = self._forward_target_verify(
                q, k, v, forward_batch, layer_id, layer_cache,
                cache_idx, mamba_indices, initial_state, scale, g_gamma,
            )
        else:
            mode = "fused_recurrent" if seq_len < 64 else "chunk"
            if forward_batch.forward_mode.is_decode() or mode == "fused_recurrent":
                o, final_state = fused_recurrent_simple_gla(
                    q=q,
                    k=k,
                    v=v,
                    g_gamma=g_gamma,
                    scale=scale,
                    initial_state=initial_state,
                    output_final_state=True,
                    cu_seqlens=self.forward_metadata.query_start_loc,
                )
            else:
                o, final_state = chunk_simple_gla(
                    q=q,
                    k=k,
                    v=v,
                    g_gamma=g_gamma,
                    initial_state=initial_state,
                    output_final_state=True,
                    scale=scale,
                    cu_seqlens=self.forward_metadata.query_start_loc,
                )

            if final_state is not None:
                layer_cache.temporal[mamba_indices, :] = final_state

        o = o.reshape(-1, num_heads * head_dim)

        return o

    def _forward_target_verify(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        layer_cache,
        cache_idx: int,
        mamba_indices: torch.Tensor,
        initial_state: torch.Tensor,
        scale: float,
        g_gamma: torch.Tensor,
    ) -> torch.Tensor:
        """Handle TARGET_VERIFY mode for SimpleGLA with intermediate state caching.

        CUDA-graph-compatible version: all dynamic allocations removed.

        During speculative verify, multiple draft tokens are processed at once.
        We need to cache the GLA state after each token so that
        update_mamba_state_after_mtp_verify can later pick the state
        corresponding to the last accepted token (rolling back rejected ones).

        Strategy: process tokens one-by-one using the SAME fused kernel
        (fused_recurrent_simple_gla) to guarantee bit-exact state parity
        with normal decode. Each step yields both the correct output vector
        and the exact final state that the kernel would produce, avoiding
        any precision mismatch from a hand-written Python recurrence.

        CUDA graph compatibility:
        - No torch.arange() — reuse pre-allocated cu_seqlens buffer
        - No outputs=[] + torch.cat() — write into pre-allocated output buffer
        - No .transpose().contiguous() — copy_ into pre-allocated staging buffers
        - intermediate_ssm indexed with slice instead of arange
        """
        draft_token_num = forward_batch.spec_info.draft_token_num
        batch_size = forward_batch.batch_size
        H = q.shape[2]
        D = q.shape[3]

        intermediate_ssm = layer_cache.intermediate_ssm

        # Reuse the pre-allocated cu_seqlens buffer from init_cuda_graph_state().
        # cached_cuda_graph_decode_query_start_loc is [0, 1, 2, ..., max_bs],
        # which is exactly what single-token-per-sequence verify needs.
        # In eager mode (CUDA graph disabled), the buffer may not exist yet,
        # so fall back to creating one (this allocation is fine outside capture).
        if self.cached_cuda_graph_decode_query_start_loc is not None:
            single_cu = self.cached_cuda_graph_decode_query_start_loc[:batch_size + 1]
        else:
            single_cu = torch.arange(
                batch_size + 1, dtype=torch.int32, device=q.device
            )

        # Reshape q/k/v from [1, B*N, H, D] -> [B, N, H, D]
        q_4d = q.squeeze(0).reshape(batch_size, draft_token_num, H, D)
        k_4d = k.squeeze(0).reshape(batch_size, draft_token_num, H, D)
        v_4d = v.squeeze(0).reshape(batch_size, draft_token_num, H, D)

        # Lazy initialization of pre-allocated staging buffers.
        # We cannot allocate in init_cuda_graph_state() because num_heads (H)
        # and head_dim (D) are not known at that point.
        # Once allocated, these buffers are reused across all subsequent calls
        # (both eager and CUDA graph replay).
        if self._cg_verify_q_step is None or self._cg_verify_q_step.shape[1] < batch_size:
            max_bs = max(batch_size, getattr(self, '_cg_verify_max_bs', batch_size))
            max_draft = max(draft_token_num, getattr(self, '_cg_verify_draft_token_num', draft_token_num))
            self._cg_verify_q_step = torch.empty(1, max_bs, H, D, dtype=q.dtype, device=q.device)
            self._cg_verify_k_step = torch.empty(1, max_bs, H, D, dtype=q.dtype, device=q.device)
            self._cg_verify_v_step = torch.empty(1, max_bs, H, D, dtype=q.dtype, device=q.device)
            self._cg_verify_out = torch.empty(max_bs, max_draft, H, D, dtype=q.dtype, device=q.device)

        # Slice pre-allocated buffers to the current batch/draft size.
        q_step_buf = self._cg_verify_q_step[:, :batch_size]
        k_step_buf = self._cg_verify_k_step[:, :batch_size]
        v_step_buf = self._cg_verify_v_step[:, :batch_size]
        out_buf = self._cg_verify_out[:batch_size, :draft_token_num]

        current_state = initial_state  # [B, H, K, V]

        # [DIAG-H4-QKV] Compare root token's QKV between topk configs
        import os
        _diag_qkv = os.environ.get('SGLANG_DIAG_QKV', '0') == '1'
        if _diag_qkv:
            if not hasattr(self, '_diag_qkv_count'):
                self._diag_qkv_count = 0
            if self._diag_qkv_count < 50:
                # q_4d shape: [B, dtn, H, D], root token is [:, 0, :, :]
                q_root = q_4d[:, 0]  # [B, H, D]
                k_root = k_4d[:, 0]
                v_root = v_4d[:, 0]
                topk = getattr(forward_batch.spec_info, 'topk', 1)
                print(f"[DIAG-QKV] layer={layer_id} topk={topk} dtn={draft_token_num} "
                      f"q_root_hash={q_root.sum().item():.6f} "
                      f"k_root_hash={k_root.sum().item():.6f} "
                      f"v_root_hash={v_root.sum().item():.6f} "
                      f"q_shape={list(q_4d.shape)} "
                      f"initial_state_hash={initial_state.sum().item():.6f}",
                      flush=True)
                self._diag_qkv_count += 1

        # Tree-aware state routing for topk>1: derive parent mapping so
        # sibling tokens use their parent's state, not the preceding sibling's.
        spec_info = forward_batch.spec_info
        parent_step = None
        if hasattr(spec_info, 'topk') and spec_info.topk > 1 and spec_info.retrive_next_token is not None:
            rnt = spec_info.retrive_next_token[0].tolist()
            rns = spec_info.retrive_next_sibling[0].tolist()
            parent_step = [-1] * draft_token_num
            for i in range(draft_token_num):
                child = rnt[i]
                if 0 <= child < draft_token_num:
                    parent_step[child] = i
                sib = rns[i]
                if 0 <= sib < draft_token_num:
                    parent_step[sib] = parent_step[i]

        # Diagnostic: compare parent_step with retrieve_parent_token from GDN conv1d
        import os
        _diag = os.path.exists('/tmp/SGLANG_DIAG_TREE_VERIFY')
        if _diag and parent_step is not None and layer_id == 1:
            rpt = getattr(self, 'forward_metadata', None)
            if rpt is not None:
                rpt_tensor = getattr(rpt, 'retrieve_parent_token', None)
                if rpt_tensor is not None and rpt_tensor.numel() > 0:
                    rpt_list = rpt_tensor[0, :draft_token_num].tolist()
                    if not hasattr(self, '_diag_count'):
                        self._diag_count = 0
                    if self._diag_count < 3:
                        import logging
                        logging.getLogger(__name__).warning(
                            f"[DIAG] layer={layer_id} topk={spec_info.topk} dtn={draft_token_num} "
                            f"parent_step={parent_step} retrieve_parent_token={rpt_list}"
                        )
                        self._diag_count += 1

        # Keep full-precision state cache (no dtype conversion) for tree routing
        state_cache = {}  # step -> state tensor (same dtype as initial_state)

        for step in range(draft_token_num):
            # Tree routing: load parent state for siblings
            if parent_step is not None and step > 0:
                p = parent_step[step]
                if p >= 0 and p in state_cache:
                    current_state = state_cache[p]
                else:
                    current_state = initial_state

            # Copy into pre-allocated contiguous staging buffers instead of
            # doing .transpose(0,1).contiguous() which allocates a new tensor.
            # q_4d[:, step] is [B, H, D]; copy into q_step_buf[0] which is [B, H, D].
            q_step_buf[0].copy_(q_4d[:, step])
            k_step_buf[0].copy_(k_4d[:, step])
            v_step_buf[0].copy_(v_4d[:, step])

            o_step, next_state = fused_recurrent_simple_gla(
                q=q_step_buf,
                k=k_step_buf,
                v=v_step_buf,
                g_gamma=g_gamma,
                scale=scale,
                initial_state=current_state,
                output_final_state=True,
                cu_seqlens=single_cu,
            )
            # o_step: [1, B, H, D], next_state: [B, H, K, V]

            # Write directly into the output buffer slice instead of
            # appending to a list and later doing torch.cat().
            out_buf[:, step].copy_(o_step[0])

            # Cache intermediate state for rollback.
            # Use slice [:batch_size] instead of torch.arange-based indexing.
            intermediate_ssm[:batch_size, step] = next_state.to(
                intermediate_ssm.dtype
            )

            # [DIAG-H4-VERIFY] Log per-step state in _forward_target_verify
            import os
            if os.environ.get('SGLANG_DIAG_STATE2', '0') == '1':
                if not hasattr(self, '_diag_verify_count'):
                    self._diag_verify_count = 0
                if self._diag_verify_count < 30:
                    ps_info = f"parent={parent_step[step]}" if parent_step and step > 0 else "root"
                    print(f"[DIAG-VERIFY] layer={layer_id} step={step}/{draft_token_num} "
                          f"topk={getattr(spec_info,'topk',1)} {ps_info} "
                          f"input_state_hash={current_state.sum().item():.6f} "
                          f"next_state_hash={next_state.sum().item():.6f} "
                          f"o_hash={o_step.sum().item():.6f} "
                          f"cached_hash={intermediate_ssm[0,step].sum().item():.6f}",
                          flush=True)
                    self._diag_verify_count += 1

            # Save full-precision state for tree routing
            if parent_step is not None:
                state_cache[step] = next_state
            current_state = next_state

        # Output layout:
        # out_buf is [B, N, H, D] with tokens grouped by sequence
        #   i.e. [seq0_tok0, seq0_tok1, ..., seq1_tok0, seq1_tok1, ...]
        # reshape to [1, B*N, H, D] preserves sequence-major ordering,
        # which matches the old code's transpose(1,2) re-interleave result.
        # Do NOT update the main state here -- update_mamba_state_after_mtp_verify
        # will pick the correct intermediate state based on accepted tokens.
        return out_buf.reshape(1, batch_size * draft_token_num, H, D)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        return self.forward(q, k, v, forward_batch, layer_id, output_attentions)

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        return self.forward(q, k, v, forward_batch, layer_id, output_attentions)
