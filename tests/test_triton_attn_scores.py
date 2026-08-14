"""Numerical tests for the StateKV Triton attention-score kernel."""

from __future__ import annotations

import math

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for Triton kernels",
)


def _eager_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    causal: bool,
    cache_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_query_heads = query.shape[1]
    num_kv_heads = key.shape[1]
    groups = num_query_heads // num_kv_heads
    query = query.float()
    key = key.float().repeat_interleave(groups, dim=1)
    value = value.float().repeat_interleave(groups, dim=1)
    scores = query @ key.transpose(-2, -1)
    scores *= 1.0 / math.sqrt(query.shape[-1])

    if causal:
        query_positions = torch.arange(
            query.shape[-2],
            device=query.device,
        ).view(1, 1, -1, 1)
        key_positions = torch.arange(
            key.shape[-2],
            device=key.device,
        ).view(1, 1, 1, -1)
        scores.masked_fill_(
            key_positions > cache_length + query_positions,
            float("-inf"),
        )

    weights = scores.softmax(dim=-1)
    return weights @ value, weights.sum(dim=2)


def _random_tensor(*shape: int, dtype: torch.dtype) -> torch.Tensor:
    return torch.randn(*shape, device="cuda", dtype=dtype) * 0.1


@pytest.mark.parametrize(
    ("query_heads", "kv_heads", "query_length", "cache_length", "head_dim", "dtype"),
    [
        (4, 4, 37, 0, 32, torch.float16),
        (4, 4, 64, 0, 64, torch.float16),
        (8, 2, 32, 96, 128, torch.bfloat16),
        (8, 2, 259, 4096, 128, torch.bfloat16),
    ],
)
def test_causal_output_and_key_scores_match_eager(
    query_heads,
    kv_heads,
    query_length,
    cache_length,
    head_dim,
    dtype,
):
    from lmms_eval.models.model_utils.triton_attn_scores import (
        flash_attn_with_key_scores,
    )

    torch.manual_seed(1234)
    key_length = cache_length + query_length
    query = _random_tensor(
        1,
        query_heads,
        query_length,
        head_dim,
        dtype=dtype,
    )
    key = _random_tensor(
        1,
        kv_heads,
        key_length,
        head_dim,
        dtype=dtype,
    )
    value = _random_tensor(
        1,
        kv_heads,
        key_length,
        head_dim,
        dtype=dtype,
    )

    output, key_scores = flash_attn_with_key_scores(
        query,
        key,
        value,
        causal=True,
        cache_len=cache_length,
    )
    reference_output, reference_scores = _eager_reference(
        query,
        key,
        value,
        causal=True,
        cache_length=cache_length,
    )

    tolerance = 2e-2 if dtype is torch.bfloat16 else 1e-2
    torch.testing.assert_close(
        output,
        reference_output,
        atol=tolerance,
        rtol=tolerance,
    )
    torch.testing.assert_close(
        key_scores,
        reference_scores,
        atol=tolerance,
        rtol=tolerance,
    )
    torch.testing.assert_close(
        key_scores.sum(dim=-1),
        torch.full_like(key_scores.sum(dim=-1), query_length),
        atol=0.1,
        rtol=0.01,
    )


def test_noncausal_gqa_matches_eager():
    from lmms_eval.models.model_utils.triton_attn_scores import (
        flash_attn_with_key_scores,
    )

    torch.manual_seed(1234)
    query = _random_tensor(1, 8, 48, 128, dtype=torch.float16)
    key = _random_tensor(1, 2, 96, 128, dtype=torch.float16)
    value = _random_tensor(1, 2, 96, 128, dtype=torch.float16)
    output, key_scores = flash_attn_with_key_scores(
        query,
        key,
        value,
        causal=False,
    )
    reference_output, reference_scores = _eager_reference(
        query,
        key,
        value,
        causal=False,
        cache_length=0,
    )
    torch.testing.assert_close(output, reference_output, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        key_scores,
        reference_scores,
        atol=1e-2,
        rtol=1e-2,
    )


def test_streaming_integration_matches_eager_output_layout():
    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Model

    from lmms_eval.models.chat.internvl3_streaming import StreamingInternVL3

    torch.manual_seed(1234)
    config = Qwen2Config(
        vocab_size=32,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    model = Qwen2Model(config).to(device="cuda", dtype=torch.bfloat16).eval()
    inputs_embeds = torch.randn(
        1,
        19,
        config.hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    attention_mask = torch.ones(1, 19, device="cuda", dtype=torch.long)
    cache_position = torch.arange(19, device="cuda")

    eager_runner = object.__new__(StreamingInternVL3)
    eager_runner.cache_attn_implementation = "eager"
    triton_runner = object.__new__(StreamingInternVL3)
    triton_runner.cache_attn_implementation = "triton"

    eager_result = eager_runner.run_with_attention(
        model,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        use_cache=False,
        cache_position=cache_position,
    )
    triton_result = triton_runner.run_with_attention(
        model,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        use_cache=False,
        cache_position=cache_position,
    )

    torch.testing.assert_close(
        triton_result["last_hidden_state"],
        eager_result["last_hidden_state"],
        atol=2e-2,
        rtol=2e-2,
    )
    torch.testing.assert_close(
        triton_result["key_scores_layers"][0],
        eager_result["key_scores_layers"][0],
        atol=2e-2,
        rtol=2e-2,
    )
