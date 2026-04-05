"""
EAGLE3 draft model for MiniCPM-SALA (inference-only).

Adapted from llama_eagle3.py. This module implements the EAGLE3 speculative
draft head for the hybrid MiniCPM-SALA architecture (24 GLA + 8 full-attention).

Key design decisions:
  - The draft head uses ONLY full-attention (standard RadixAttention), not GLA/lightning.
    EAGLE3 draft operates on a single decoder layer with its own KV cache.
    GLA is a recurrent mechanism in the target model; the draft head does NOT
    re-run GLA layers -- it consumes hidden states already produced by the target.
  - Input: 3 captured aux hidden states from target (concatenated to hidden_size*3),
    projected through fc -> hidden_size, then processed by 1 decoder layer.
  - The decoder layer uses MiniCPM-style MLP (gate_up_proj + down_proj + SiLU)
    but standard full-attention QKV (not GLA).
  - lm_head can be shared from target or have its own (for FRSpec / vocab-pruned).
  - Residual scaling uses scale_depth / sqrt(num_hidden_layers) to match target.

Architecture of the draft head (parameter count with hidden_size=4096):
  - embed_tokens: vocab_size * hidden_size  (shared from target, 0 new params)
  - fc: (hidden_size*3, hidden_size)        = 4096*3 * 4096 = 50.3M params
  - midlayer:
    - input_layernorm: hidden_size          = 4096
    - hidden_norm: hidden_size              = 4096
    - qkv_proj: (2*hidden_size) -> heads    ~ 2*4096 * (32+2+2)*128 = ~37.7M
    - o_proj: (heads*head_dim, hidden_size) = 32*128 * 4096 = 16.8M
    - post_attention_layernorm: hidden_size = 4096
    - gate_up_proj: hidden_size -> 2*inter  = 4096 * 2*16384 = 134.2M
    - down_proj: inter -> hidden_size       = 16384 * 4096 = 67.1M
    - mlp subtotal                          ~ 201.3M
  - norm: hidden_size                       = 4096
  - lm_head: (hidden_size, draft_vocab)     = shared or ~4096 * vocab_size

  Total new params (excluding shared embed/lm_head): ~290M = ~580MB in bf16
  With lm_head (73448 vocab): +300M = ~600MB -> total ~1.16GB in bf16
  FRSpec vocab-pruned lm_head could reduce to <800MB.
"""

import copy
from typing import Iterable, Optional, Tuple

import torch
from torch import nn

from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import add_prefix


# --------------------------------------------------------------------------- #
#  MLP — matches MiniCPM target (SiLU gate + up + down)
# --------------------------------------------------------------------------- #
class MiniCPMDraftMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


# --------------------------------------------------------------------------- #
#  Attention — standard full attention (RadixAttention) for the draft head
#  Uses 2*hidden_size input dim for EAGLE3's [embeds || hidden_states] concat
# --------------------------------------------------------------------------- #
class MiniCPMDraftAttention(nn.Module):
    def __init__(
        self,
        config,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        from sglang.srt.distributed import get_tensor_model_parallel_world_size

        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        # EAGLE3: QKV input is 2 * hidden_size (embed concat hidden)
        self.qkv_proj = QKVParallelLinear(
            2 * self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=getattr(config, "max_position_embeddings", 524288),
            base=getattr(config, "rope_theta", 10000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


# --------------------------------------------------------------------------- #
#  Decoder Layer — EAGLE3 style: takes (positions, embeds, hidden_states, ...)
#  Applies input_layernorm to embeds, hidden_norm to hidden_states,
#  concatenates them for QKV, then MLP on the result.
# --------------------------------------------------------------------------- #
class MiniCPMEagle3DecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.config = config

        self.self_attn = MiniCPMDraftAttention(
            config, layer_id, quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.mlp = MiniCPMDraftMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states

        # EAGLE3: norm embeds and hidden_states separately, then concat for attn
        embeds = self.input_layernorm(embeds)
        hidden_states = self.hidden_norm(hidden_states)
        hidden_states = torch.cat([embeds, hidden_states], dim=-1)

        # Self Attention (full attention, not GLA)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        # Post-attention residual + layernorm
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        # MLP
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


# --------------------------------------------------------------------------- #
#  Model — EAGLE3 draft backbone
# --------------------------------------------------------------------------- #
class MiniCPMEagle3Model(nn.Module):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=add_prefix("embed_tokens", prefix),
        )

        # EAGLE3: fc projects concatenated aux hidden states (N * target_hidden_size)
        # down to draft hidden_size. N = number of aux layer IDs from eagle_config.
        if hasattr(config, "target_hidden_size"):
            self.hidden_size_in = config.target_hidden_size
        else:
            self.hidden_size_in = config.hidden_size

        eagle_cfg = getattr(config, "eagle_config", None)
        if eagle_cfg is not None:
            aux_ids = eagle_cfg.get("eagle_aux_hidden_state_layer_ids", [0, 0, 0])
            num_aux = len(aux_ids)
        else:
            num_aux = 3  # legacy default

        self.fc = ColumnParallelLinear(
            self.hidden_size_in * num_aux,
            config.hidden_size,
            bias=getattr(config, "bias", False),
            quant_config=quant_config,
            prefix=add_prefix("fc", prefix),
        )

        # Decoder layers — supports 1 or more layers
        # When num_hidden_layers == 1, a single "midlayer" alias is created for
        # backward-compatible weight loading. For N > 1, layers are stored in a
        # ModuleList named "midlayers" with keys midlayers.0 … midlayers.N-1.
        n_layers = getattr(config, "num_hidden_layers", 1)
        if n_layers == 1:
            # Keep single-layer naming for backward compatibility with existing checkpoints
            self.midlayer = MiniCPMEagle3DecoderLayer(
                config, 0, quant_config, prefix=add_prefix("midlayer", prefix)
            )
            self.midlayers = None
        else:
            self.midlayer = None
            self.midlayers = nn.ModuleList([
                MiniCPMEagle3DecoderLayer(
                    config, i, quant_config,
                    prefix=add_prefix(f"midlayers.{i}", prefix),
                )
                for i in range(n_layers)
            ])

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            embeds = self.embed_tokens(input_ids)
        else:
            embeds = input_embeds

        # Get hidden states from target model (3 * hidden_size, concatenated aux states)
        hidden_states = forward_batch.spec_info.hidden_states
        if hidden_states.shape[-1] != embeds.shape[-1]:
            hidden_states, _ = self.fc(hidden_states)

        # idle batch
        if hidden_states.shape[0] == 0:
            return hidden_states, [hidden_states]

        residual = None
        layers = [self.midlayer] if self.midlayer is not None else self.midlayers
        for layer in layers:
            hidden_states, residual = layer(
                positions,
                embeds,
                hidden_states,
                forward_batch,
                residual,
            )

        hidden_states_to_logits, hidden_states_to_aux = self.norm(
            hidden_states, residual
        )

        # For draft decode, capture the hidden state before norm
        return hidden_states_to_logits, [hidden_states_to_aux]


# --------------------------------------------------------------------------- #
#  Top-level CausalLM — EAGLE3 draft for MiniCPM-SALA
# --------------------------------------------------------------------------- #
class MiniCPMSALAForCausalLMEagle3(nn.Module):
    """
    EAGLE3 speculative draft model for MiniCPM-SALA.

    This class mirrors LlamaForCausalLMEagle3 but is designed for the
    MiniCPM-SALA architecture. The draft head uses standard full-attention
    (not GLA/lightning) since it only has 1 layer with its own KV cache.

    The target model's GLA state management is handled separately by the
    hybrid_linear_attn_backend during verify. The draft head does NOT need
    to manage GLA states.
    """

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.quant_config = quant_config
        self.pp_group = get_pp_group()

        self.model = MiniCPMEagle3Model(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )

        # lm_head handling: shared from target or independent
        self.load_lm_head_from_target = False
        if self.config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            draft_vocab_size = getattr(config, "draft_vocab_size", None)
            if draft_vocab_size is None:
                self.load_lm_head_from_target = True
                draft_vocab_size = config.vocab_size
            self.lm_head = ParallelLMHead(
                draft_vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )

        config_ = copy.deepcopy(config)
        config_.vocab_size = getattr(config_, "draft_vocab_size", config_.vocab_size)
        self.logits_processor = LogitsProcessor(config_)

        self.capture_aux_hidden_states = True
        self.hot_token_id = None

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)

        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states

        return self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            forward_batch,
            aux_hidden_states,
        )

    # ---- Embed / Head sharing interface (required by MultiLayerEagleWorker) ----

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def get_embed(self):
        return self.model.embed_tokens.weight

    def set_embed(self, embed):
        if (
            hasattr(self.config, "target_hidden_size")
            and self.config.target_hidden_size != self.config.hidden_size
        ):
            return
        del self.model.embed_tokens.weight
        self.model.embed_tokens.weight = embed
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def get_hot_token_id(self):
        return self.hot_token_id

    # ---- Weight loading ----

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        params_dict = dict(self.named_parameters())
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        for name, loaded_weight in weights:
            if "d2t" in name:
                self.hot_token_id = loaded_weight + torch.arange(
                    loaded_weight.shape[0]
                )
                import logging
                logging.getLogger(__name__).info(
                    f"[MiniCPM-EAGLE3] Loaded d2t mapping: shape={loaded_weight.shape}, "
                    f"hot_token_id range=[{self.hot_token_id.min()}, {self.hot_token_id.max()}]"
                )
                continue
            if "t2d" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param_name = f"model.{name}" if name not in params_dict else name
                if param_name in params_dict:
                    param = params_dict[param_name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param_name = name if name in params_dict else f"model.{name}"
                if param_name in params_dict:
                    param = params_dict[param_name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)

    # ---- EAGLE3 layer capture (not needed for draft, but interface compat) ----

    def set_eagle3_layers_to_capture(self, layer_ids=None):
        pass  # Draft model doesn't capture aux states from itself


EntryClass = [MiniCPMSALAForCausalLMEagle3]
