"""Fused BF16→FP8_E5M2 quantization + indexed KV cache write kernel.

Replaces the 4-step Python path in MHATokenToKVPool.set_kv_buffer:
  1. cache_k.div_(k_scale)           → in-place scale
  2. cache_k = cache_k.to(fp8_e5m2)  → dtype conversion (allocates new tensor)
  3. cache_k = cache_k.view(uint8)    → view cast
  4. k_buffer[layer_id][loc] = cache_k → scatter write

With a single Triton kernel that does scale + quantize + scatter write in one pass.
"""

import logging
from typing import Optional

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _fused_fp8e5m2_kv_write_kernel(
    # Input K, V in BF16/FP16
    k_ptr,
    v_ptr,
    # Output KV cache buffers (stored as uint8 = bitcast of fp8_e5m2)
    k_cache_ptr,
    v_cache_ptr,
    # Scatter indices
    cache_loc_ptr,  # [num_tokens] int32
    # Inverse scales (on GPU, 0-D tensors)
    inv_k_scale_ptr,
    inv_v_scale_ptr,
    use_provided_scale: tl.constexpr,
    # Dimensions
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    # Input strides [num_tokens, num_kv_heads, head_dim]
    k_stride_token,
    k_stride_head,
    k_stride_dim,
    v_stride_token,
    v_stride_head,
    v_stride_dim,
    # Cache strides [total_slots, num_kv_heads, head_dim]
    k_cache_stride_slot,
    k_cache_stride_head,
    k_cache_stride_dim,
    v_cache_stride_slot,
    v_cache_stride_head,
    v_cache_stride_dim,
    # Block sizes
    BLOCK_HEAD: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    """
    Fused FP8_E5M2 quantization + scatter KV cache write.

    Grid: (num_tokens, num_head_blocks, 2)
      - dim0: token index
      - dim1: head block index
      - dim2: 0=K, 1=V
    """
    token_id = tl.program_id(0)
    head_block_id = tl.program_id(1)
    kv_idx = tl.program_id(2)  # 0=K, 1=V

    # Load scatter target slot for this token
    cache_loc = tl.load(cache_loc_ptr + token_id)

    # Select K or V pointers and strides
    if kv_idx == 0:
        input_ptr = k_ptr
        cache_ptr = k_cache_ptr
        in_stride_token = k_stride_token
        in_stride_head = k_stride_head
        in_stride_dim = k_stride_dim
        cache_stride_slot = k_cache_stride_slot
        cache_stride_head = k_cache_stride_head
        cache_stride_dim = k_cache_stride_dim
        if use_provided_scale:
            inv_scale = tl.load(inv_k_scale_ptr)
        else:
            inv_scale = 1.0
    else:
        input_ptr = v_ptr
        cache_ptr = v_cache_ptr
        in_stride_token = v_stride_token
        in_stride_head = v_stride_head
        in_stride_dim = v_stride_dim
        cache_stride_slot = v_cache_stride_slot
        cache_stride_head = v_cache_stride_head
        cache_stride_dim = v_cache_stride_dim
        if use_provided_scale:
            inv_scale = tl.load(inv_v_scale_ptr)
        else:
            inv_scale = 1.0

    # Process heads in this block
    head_start = head_block_id * BLOCK_HEAD
    head_offsets = head_start + tl.arange(0, BLOCK_HEAD)
    head_mask = head_offsets < num_kv_heads

    # Process head_dim in blocks
    for dim_start in range(0, head_dim, BLOCK_DIM):
        dim_offsets = dim_start + tl.arange(0, BLOCK_DIM)
        dim_mask = dim_offsets < head_dim

        mask = head_mask[:, None] & dim_mask[None, :]

        # Load from input [token_id, head, dim]
        in_offsets = (
            token_id * in_stride_token
            + head_offsets[:, None] * in_stride_head
            + dim_offsets[None, :] * in_stride_dim
        )
        block = tl.load(input_ptr + in_offsets, mask=mask, other=0.0)

        # Scale + quantize to FP8_E5M2, then bitcast to uint8 for storage
        if use_provided_scale:
            block_fp8 = (block * inv_scale).to(tl.float8e5)
        else:
            block_fp8 = block.to(tl.float8e5)
        block_u8 = block_fp8.to(tl.uint8, bitcast=True)

        # Write to cache at [cache_loc, head, dim]
        cache_offsets = (
            cache_loc * cache_stride_slot
            + head_offsets[:, None] * cache_stride_head
            + dim_offsets[None, :] * cache_stride_dim
        )
        tl.store(cache_ptr + cache_offsets, block_u8, mask=mask)


def fused_fp8e5m2_set_kv_buffer(
    k: torch.Tensor,  # [num_tokens, num_kv_heads, head_dim] BF16
    v: torch.Tensor,  # [num_tokens, num_kv_heads, head_dim] BF16
    k_cache: torch.Tensor,  # [total_slots, num_kv_heads, head_dim] uint8
    v_cache: torch.Tensor,  # [total_slots, num_kv_heads, head_dim] uint8
    cache_loc: torch.Tensor,  # [num_tokens] int32
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
):
    """
    Fused BF16→FP8_E5M2 quantization + scatter write to KV cache.

    Replaces:
        cache_k.div_(k_scale)
        cache_k = cache_k.to(torch.float8_e5m2)
        cache_k = cache_k.view(torch.uint8)
        k_buffer[loc] = cache_k
    """
    assert k.is_contiguous() and v.is_contiguous(), "K, V must be contiguous"
    assert k.ndim == 3 and v.ndim == 3, "Expected [num_tokens, num_kv_heads, head_dim]"

    num_tokens, num_kv_heads, head_dim = k.shape

    if num_tokens == 0:
        return

    # Compute inverse scales on GPU (CUDA graph safe)
    use_provided_scale = k_scale is not None or v_scale is not None
    if use_provided_scale:
        inv_k_scale = torch.tensor(
            1.0 / k_scale if k_scale else 1.0, dtype=torch.float32, device=k.device
        )
        inv_v_scale = torch.tensor(
            1.0 / v_scale if v_scale else 1.0, dtype=torch.float32, device=k.device
        )
    else:
        # Dummy tensors (won't be read)
        inv_k_scale = torch.empty(1, dtype=torch.float32, device=k.device)
        inv_v_scale = inv_k_scale

    # Block sizes
    BLOCK_HEAD = min(num_kv_heads, 8)
    BLOCK_DIM = min(head_dim, 128)
    num_head_blocks = (num_kv_heads + BLOCK_HEAD - 1) // BLOCK_HEAD

    # Grid: (num_tokens, num_head_blocks, 2=K/V)
    grid = (num_tokens, num_head_blocks, 2)

    _fused_fp8e5m2_kv_write_kernel[grid](
        k, v,
        k_cache, v_cache,
        cache_loc,
        inv_k_scale, inv_v_scale,
        use_provided_scale,
        num_kv_heads, head_dim,
        # K input strides
        k.stride(0), k.stride(1), k.stride(2),
        # V input strides
        v.stride(0), v.stride(1), v.stride(2),
        # K cache strides
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        # V cache strides
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        BLOCK_HEAD=BLOCK_HEAD,
        BLOCK_DIM=BLOCK_DIM,
    )
