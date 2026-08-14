"""Recency-window (ReKV) baseline for InternVL3."""

from __future__ import annotations

import torch

from lmms_eval.api.registry import register_model
from lmms_eval.models.chat.internvl3_streaming import PromptCacheState, StreamingInternVL3


@register_model("internvl3_rekv")
class ReKVInternVL3(StreamingInternVL3):
    method_name = "ReKV"

    def __init__(self, retrieved_frames: int = 16, **kwargs) -> None:
        if int(retrieved_frames) <= 0:
            raise ValueError("retrieved_frames must be positive")
        self.retrieved_frames = int(retrieved_frames)
        super().__init__(**kwargs)

    def _carried_state_size(self, tokens_per_frame: int) -> int:
        return tokens_per_frame * self.retrieved_frames

    def _prune_carried_state(
        self,
        state: PromptCacheState,
        attention_weights: list[torch.Tensor | None],
        max_tokens: int,
    ) -> PromptCacheState:
        """Retain the newest tokens in every layer of the carried cache."""
        del attention_weights
        kept_tokens = None
        batch_size = None
        for layer in state.past_key_values.layers:
            batch_size = layer.keys.shape[0]
            kept_tokens = min(max_tokens, layer.keys.shape[-2])
            layer.keys = layer.keys[:, :, -kept_tokens:, :]
            layer.values = layer.values[:, :, -kept_tokens:, :]

        if kept_tokens is None or batch_size is None:
            raise RuntimeError("The model returned an empty KV cache")
        if state.attn_mask is not None and not bool((state.attn_mask == 1).all()):
            raise RuntimeError("ReKV only supports unpadded carried states")

        state.attn_mask = torch.ones(
            (batch_size, kept_tokens),
            dtype=torch.bool,
            device=state.past_key_values.layers[0].keys.device,
        )
        state.seq_len = kept_tokens
        return state
