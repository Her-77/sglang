"""
Dual-weight loader: Load Marlin WNA16 weights onto an NVFP4 model.

After the primary NVFP4 model is loaded, this module loads a secondary
Marlin WNA16 model's weights and registers them on each linear layer.
The ModelOptFp4LinearMethod.apply() then dispatches:
  - extend (prefill) → FP4 GEMM (fast for large M)
  - decode → Marlin WNA16 GEMM (fast for small M)

Supports W4A16 (default) and W8A16 via SGLANG_DUAL_WEIGHT_NUM_BITS env var.

Usage:
  SGLANG_DUAL_WEIGHT_MARLIN_PATH=/path/to/marlin/model \
  SGLANG_DUAL_WEIGHT_NUM_BITS=8 \
  python -m sglang.launch_server --model /path/to/nvfp4/model --quantization modelopt_fp4 ...
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import torch
from torch.nn.parameter import Parameter

logger = logging.getLogger(__name__)

# Configurable num_bits: 4 (W4A16, default) or 8 (W8A16)
DUAL_WEIGHT_NUM_BITS = int(os.environ.get("SGLANG_DUAL_WEIGHT_NUM_BITS", "4"))
DUAL_WEIGHT_PACK_FACTOR = 32 // DUAL_WEIGHT_NUM_BITS  # 8 for 4-bit, 4 for 8-bit

# Mapping: sglang fused layer name → list of separate safetensors names
FUSED_MAPPING = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}


def load_marlin_weights_onto_fp4_model(
    model: torch.nn.Module,
    marlin_model_path: str,
    device: torch.device,
) -> int:
    """Load Marlin WNA16 weights from a secondary model onto NVFP4 layers.

    Supports W4A16 (num_bits=4, default) and W8A16 (num_bits=8) via
    SGLANG_DUAL_WEIGHT_NUM_BITS environment variable.

    Args:
        model: The loaded NVFP4 model.
        marlin_model_path: Path to the Marlin model directory.
        device: Target device (cuda:X).

    Returns:
        Number of layers that received Marlin weights.
    """
    from safetensors import safe_open
    from sgl_kernel import gptq_marlin_repack

    from sglang.srt.layers.quantization.marlin_utils import (
        marlin_make_empty_g_idx,
        marlin_make_workspace,
        marlin_permute_scales,
    )

    # 1. Load safetensors index
    index_path = os.path.join(marlin_model_path, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    # 2. Open safetensors files lazily
    file_cache: Dict[str, object] = {}

    def _get_tensor(key: str) -> torch.Tensor:
        filename = weight_map[key]
        if filename not in file_cache:
            path = os.path.join(marlin_model_path, filename)
            file_cache[filename] = safe_open(path, framework="pt")
        return file_cache[filename].get_tensor(key)

    # 3. Shared Marlin auxiliary tensors (created once per device)
    empty_g_idx = marlin_make_empty_g_idx(device)
    workspace = marlin_make_workspace(device)

    # 4. Iterate through all linear layers
    count = 0
    total_bytes = 0

    for name, module in model.named_modules():
        quant_method = getattr(module, "quant_method", None)
        if quant_method is None:
            continue
        # Accept both NVFP4 layers and unquantized layers (e.g., minicpm4 attention
        # layers protected by NVFP4 ignore list). Both benefit from Marlin decode.
        _ACCEPTED_METHODS = {"ModelOptFp4LinearMethod", "UnquantizedLinearMethod"}
        if quant_method.__class__.__name__ not in _ACCEPTED_METHODS:
            continue

        # Determine safetensors key prefix
        parts = name.split(".")
        layer_suffix = parts[-1]  # e.g., "qkv_proj", "o_proj", "down_proj"

        try:
            if layer_suffix in FUSED_MAPPING:
                w_packed, w_scale, K, N = _load_fused_weights(
                    name, layer_suffix, _get_tensor
                )
            else:
                w_packed, w_scale, K, N = _load_direct_weights(name, _get_tensor)
        except KeyError as e:
            logger.warning(f"Dual-weight: skipping {name}, key not found: {e}")
            continue

        # 5. Repack to Marlin format
        # Compressed-tensors stores [N, K/pack_factor] int32
        # Marlin needs [K/pack_factor, N] after repack
        w_t = w_packed.t().contiguous().to(device)
        w_repacked = gptq_marlin_repack(
            w_t, perm=empty_g_idx, size_k=K, size_n=N, num_bits=DUAL_WEIGHT_NUM_BITS
        )

        # Scales: [N, K/group_size] → transpose → marlin_permute_scales
        s_t = w_scale.t().contiguous().to(device)
        # Determine group_size from dimensions
        num_groups = s_t.shape[0]
        group_size = K // num_groups if num_groups > 0 else 128
        s_permuted = marlin_permute_scales(s_t, size_k=K, size_n=N, group_size=group_size)

        # 6. Register on layer
        module.marlin_qweight = Parameter(w_repacked, requires_grad=False)
        module.marlin_scales = Parameter(s_permuted, requires_grad=False)
        module.marlin_g_idx = empty_g_idx
        module.marlin_g_idx_sort_indices = empty_g_idx
        module.marlin_zp = empty_g_idx  # symmetric quant → empty zero points
        module.marlin_workspace = workspace
        module.marlin_output_size = N
        module.marlin_input_size = K
        module.has_marlin_weights = True

        layer_bytes = (
            w_repacked.nelement() * w_repacked.element_size()
            + s_permuted.nelement() * s_permuted.element_size()
        )
        total_bytes += layer_bytes
        count += 1
        logger.info(
            f"Dual-weight: {name} [{K}x{N}] → Marlin, {layer_bytes / 1024**2:.1f} MB"
        )

    # Cleanup file handles
    file_cache.clear()

    # Install forward hooks on nn.Linear layers that got Marlin weights
    # (these are unquantized layers that bypass SGLang's LinearBase dispatch)
    hook_count = _install_marlin_decode_hooks(model)

    total_mb = total_bytes / 1024**2
    logger.info(
        f"Dual-weight: loaded Marlin W{DUAL_WEIGHT_NUM_BITS}A16 weights for {count} layers, "
        f"total {total_mb:.1f} MB ({total_mb / 1024:.2f} GB)"
    )
    if hook_count > 0:
        logger.info(
            f"Dual-weight: installed Marlin decode hooks on {hook_count} nn.Linear layers"
        )
    return count


def _install_marlin_decode_hooks(model: torch.nn.Module) -> int:
    """Monkey-patch nn.Linear.forward on layers with Marlin weights.

    For layers loaded via trust_remote_code (using raw nn.Linear instead of
    SGLang's LinearBase), the normal quant_method.apply() dispatch is bypassed.
    This replaces forward() to dispatch to Marlin GEMM during decode mode.
    """
    import types

    from sglang.srt.layers.quantization.marlin_utils import apply_gptq_marlin_linear

    try:
        from sgl_kernel.scalar_type import scalar_types
    except ImportError:
        return 0

    _wtype = (
        scalar_types.uint8b128
        if DUAL_WEIGHT_NUM_BITS == 8
        else scalar_types.uint4b8
    )

    hook_count = 0
    # Debug: count candidates
    candidates = []
    for n, m in model.named_modules():
        if not getattr(m, "has_marlin_weights", False):
            continue
        qm = getattr(m, "quant_method", None)
        qm_name = qm.__class__.__name__ if qm else "None"
        if qm_name != "ModelOptFp4LinearMethod":
            candidates.append((n, type(m).__name__, qm_name))
    logger.info(f"Dual-weight hooks: {len(candidates)} candidate layers for forward patch")
    for n, cls, qm in candidates[:5]:
        logger.info(f"  candidate: {n} ({cls}, quant={qm})")

    for name, module in model.named_modules():
        if not getattr(module, "has_marlin_weights", False):
            continue
        # Skip if already handled by ModelOptFp4LinearMethod dispatch
        qm = getattr(module, "quant_method", None)
        if qm is not None and qm.__class__.__name__ == "ModelOptFp4LinearMethod":
            continue
        # Only patch modules with a forward that does F.linear or equivalent
        if not hasattr(module, "forward"):
            continue

        # Save original forward
        module._original_forward = module.forward

        def _dual_forward(self, x):
            """Dispatch to Marlin GEMM in decode mode, original BF16 in extend."""
            import sglang.srt.layers.quantization.modelopt_quant as _mqm

            if _mqm._current_forward_mode == 1:
                if not getattr(_dual_forward, '_logged', False):
                    logger.info("Dual-weight: Marlin decode hook FIRED (first call)")
                    _dual_forward._logged = True
                bias = self.bias if not getattr(self, "skip_bias_add", False) else None
                out = apply_gptq_marlin_linear(
                    input=x,
                    weight=self.marlin_qweight,
                    weight_scale=self.marlin_scales,
                    weight_zp=self.marlin_zp,
                    g_idx=self.marlin_g_idx,
                    g_idx_sort_indices=self.marlin_g_idx_sort_indices,
                    workspace=self.marlin_workspace,
                    wtype=_wtype,
                    output_size_per_partition=self.marlin_output_size,
                    input_size_per_partition=self.marlin_input_size,
                    is_k_full=True,
                    bias=bias,
                )
                # SGLang LinearBase.forward() returns (output, output_bias) tuple
                output_bias = self.bias if getattr(self, "skip_bias_add", False) else None
                return out, output_bias
            return self._original_forward(x)

        module.forward = types.MethodType(_dual_forward, module)
        hook_count += 1
        logger.debug(f"Dual-weight: patched forward on {name}")

    return hook_count


def _load_fused_weights(
    module_name: str,
    layer_suffix: str,
    get_tensor,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """Load and fuse separate weights for a fused layer (qkv_proj, gate_up_proj)."""
    separate_names = FUSED_MAPPING[layer_suffix]
    prefix = module_name.rsplit(".", 1)[0]  # e.g., "model.layers.0.self_attn"

    w_packed_list = []
    w_scale_list = []

    for sep_name in separate_names:
        sep_prefix = f"{prefix}.{sep_name}"
        w_packed_list.append(get_tensor(f"{sep_prefix}.weight_packed"))
        w_scale_list.append(get_tensor(f"{sep_prefix}.weight_scale"))

    # Concatenate along output dimension (dim=0)
    w_packed = torch.cat(w_packed_list, dim=0)
    w_scale = torch.cat(w_scale_list, dim=0)

    N = w_packed.shape[0]
    K = w_packed.shape[1] * DUAL_WEIGHT_PACK_FACTOR

    return w_packed, w_scale, K, N


def _load_direct_weights(
    module_name: str,
    get_tensor,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """Load weights for a non-fused layer (o_proj, o_gate, z_proj, down_proj)."""
    w_packed = get_tensor(f"{module_name}.weight_packed")
    w_scale = get_tensor(f"{module_name}.weight_scale")

    N = w_packed.shape[0]
    K = w_packed.shape[1] * DUAL_WEIGHT_PACK_FACTOR

    return w_packed, w_scale, K, N
