"""StateKV carried-state selection for InternVL3."""

from __future__ import annotations

import torch

from lmms_eval.api.registry import register_model
from lmms_eval.models.chat.internvl3_streaming import PromptCacheState, StreamingInternVL3


@register_model("statekv_internvl3")
class StateKVInternVL3(StreamingInternVL3):
    method_name = "StateKV"

    def __init__(
        self,
        cstate_size: int = 4096,
        cache_attn_implementation: str = "triton",
        **kwargs,
    ) -> None:
        if int(cstate_size) <= 0:
            raise ValueError("cstate_size must be a positive token count")
        self.cstate_size = int(cstate_size)
        super().__init__(
            cache_attn_implementation=cache_attn_implementation,
            **kwargs,
        )

    def _carried_state_size(self, tokens_per_frame: int) -> int:
        return self.cstate_size

    def _prune_carried_state(
        self,
        state: PromptCacheState,
        key_scores: list[torch.Tensor],
        max_tokens: int,
    ) -> PromptCacheState:
        """Select the most-attended keys independently in each layer and KV head."""
        kept_tokens = None
        batch_size = None
        for layer_index, layer in enumerate(state.past_key_values.layers):
            batch_size, num_kv_heads, key_length, key_dim = layer.keys.shape
            scores = key_scores[layer_index]
            if scores.shape[0] != batch_size or scores.shape[-1] != key_length:
                raise RuntimeError(f"Invalid key scores for layer {layer_index}")

            num_attention_heads = scores.shape[1]
            if num_attention_heads != num_kv_heads:
                if num_attention_heads % num_kv_heads:
                    raise RuntimeError("Attention heads must be divisible by KV heads")
                group_size = num_attention_heads // num_kv_heads
                scores = scores.view(batch_size, num_kv_heads, group_size, -1).sum(dim=2)

            kept_tokens = min(max_tokens, key_length)
            keep_indices = torch.topk(scores, k=kept_tokens, dim=-1, largest=True, sorted=False).indices
            keep_indices = torch.sort(keep_indices, dim=-1, stable=True).values
            key_indices = keep_indices.unsqueeze(-1).expand(batch_size, num_kv_heads, kept_tokens, key_dim)
            value_indices = keep_indices.unsqueeze(-1).expand(
                batch_size, num_kv_heads, kept_tokens, layer.values.shape[-1]
            )
            layer.keys = torch.gather(layer.keys, dim=-2, index=key_indices)
            layer.values = torch.gather(layer.values, dim=-2, index=value_indices)

        if kept_tokens is None or batch_size is None:
            raise RuntimeError("The model returned an empty KV cache")
        if state.attn_mask is not None and not bool((state.attn_mask == 1).all()):
            raise RuntimeError("StateKV only supports unpadded carried states")

        state.attn_mask = torch.ones(
            (batch_size, kept_tokens),
            dtype=torch.bool,
            device=state.past_key_values.layers[0].keys.device,
        )
        state.seq_len = kept_tokens
        return state
