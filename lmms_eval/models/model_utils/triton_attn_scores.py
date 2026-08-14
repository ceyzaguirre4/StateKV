"""Flash attention with key-score accumulation via two Triton passes.

Pass 1 computes the attention output and per-query log-sum-exp. Pass 2 reuses
that normalizer to accumulate each key's attention mass without materializing
the full ``[batch, heads, queries, keys]`` attention matrix.

The kernels support grouped-query attention and prefix-causal masking for
incremental KV-cache inference.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _flash_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    output_ptr,
    lse_ptr,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    stride_lb,
    stride_lh,
    stride_lm,
    num_query_heads,
    num_kv_heads,
    query_length,
    key_length,
    softmax_scale,
    cache_length,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch_index = batch_head // num_query_heads
    query_head = batch_head % num_query_heads
    kv_head = query_head // (num_query_heads // num_kv_heads)

    query_start = query_block * BLOCK_M
    query_offsets = query_start + tl.arange(0, BLOCK_M)
    dimension_offsets = tl.arange(0, BLOCK_D)
    query_mask = query_offsets < query_length

    query_base = q_ptr + batch_index * stride_qb + query_head * stride_qh
    query = tl.load(
        query_base
        + query_offsets[:, None] * stride_qm
        + dimension_offsets[None, :] * stride_qd,
        mask=query_mask[:, None],
        other=0.0,
    ).to(tl.float32)

    key_base = k_ptr + batch_index * stride_kb + kv_head * stride_kh
    value_base = v_ptr + batch_index * stride_vb + kv_head * stride_vh
    row_max = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    row_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    if IS_CAUSAL:
        key_bound = tl.minimum(
            cache_length + query_start + BLOCK_M,
            key_length,
        )
    else:
        key_bound = key_length

    for key_start in range(0, key_bound, BLOCK_N):
        key_offsets = key_start + tl.arange(0, BLOCK_N)
        key_mask = key_offsets < key_length
        key = tl.load(
            key_base
            + key_offsets[:, None] * stride_kn
            + dimension_offsets[None, :] * stride_kd,
            mask=key_mask[:, None],
            other=0.0,
        ).to(tl.float32)

        scores = tl.dot(query, tl.trans(key)) * softmax_scale
        scores = tl.where(key_mask[None, :], scores, float("-inf"))
        if IS_CAUSAL:
            causal_mask = key_offsets[None, :] <= (
                cache_length + query_offsets[:, None]
            )
            scores = tl.where(causal_mask, scores, float("-inf"))

        new_row_max = tl.maximum(row_max, tl.max(scores, axis=1))
        safe_difference = tl.where(
            row_max == float("-inf"),
            0.0,
            row_max - new_row_max,
        )
        correction = tl.exp(safe_difference)
        probabilities = tl.exp(scores - new_row_max[:, None])
        row_sum = row_sum * correction + tl.sum(probabilities, axis=1)

        value = tl.load(
            value_base
            + key_offsets[:, None] * stride_vn
            + dimension_offsets[None, :] * stride_vd,
            mask=key_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        accumulator = (
            accumulator * correction[:, None]
            + tl.dot(probabilities, value)
        )
        row_max = new_row_max

    safe_row_sum = tl.where(row_sum == 0.0, 1.0, row_sum)
    accumulator = accumulator / safe_row_sum[:, None]
    lse = tl.where(
        row_sum == 0.0,
        float("-inf"),
        row_max + tl.log(row_sum),
    )

    output_base = (
        output_ptr + batch_index * stride_ob + query_head * stride_oh
    )
    tl.store(
        output_base
        + query_offsets[:, None] * stride_om
        + dimension_offsets[None, :] * stride_od,
        accumulator,
        mask=query_mask[:, None],
    )
    lse_base = lse_ptr + batch_index * stride_lb + query_head * stride_lh
    tl.store(
        lse_base + query_offsets * stride_lm,
        lse,
        mask=query_mask,
    )


@triton.jit
def _key_score_kernel(
    q_ptr,
    k_ptr,
    lse_ptr,
    output_ptr,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_lb,
    stride_lh,
    stride_lm,
    stride_ob,
    stride_oh,
    stride_on,
    num_query_heads,
    num_kv_heads,
    query_length,
    key_length,
    softmax_scale,
    cache_length,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    key_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch_index = batch_head // num_query_heads
    query_head = batch_head % num_query_heads
    kv_head = query_head // (num_query_heads // num_kv_heads)

    key_start = key_block * BLOCK_N
    key_offsets = key_start + tl.arange(0, BLOCK_N)
    dimension_offsets = tl.arange(0, BLOCK_D)
    key_mask = key_offsets < key_length

    key_base = k_ptr + batch_index * stride_kb + kv_head * stride_kh
    key = tl.load(
        key_base
        + key_offsets[:, None] * stride_kn
        + dimension_offsets[None, :] * stride_kd,
        mask=key_mask[:, None],
        other=0.0,
    ).to(tl.float32)

    accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)
    query_base = q_ptr + batch_index * stride_qb + query_head * stride_qh
    lse_base = lse_ptr + batch_index * stride_lb + query_head * stride_lh

    if IS_CAUSAL:
        first_query = key_start - cache_length
        if first_query < 0:
            first_query = 0
        query_lower_bound = (first_query // BLOCK_M) * BLOCK_M
    else:
        query_lower_bound = 0

    for query_start in range(query_lower_bound, query_length, BLOCK_M):
        query_offsets = query_start + tl.arange(0, BLOCK_M)
        query_mask = query_offsets < query_length
        query = tl.load(
            query_base
            + query_offsets[:, None] * stride_qm
            + dimension_offsets[None, :] * stride_qd,
            mask=query_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        lse = tl.load(
            lse_base + query_offsets * stride_lm,
            mask=query_mask,
            other=0.0,
        )

        scores = tl.dot(query, tl.trans(key)) * softmax_scale
        probabilities = tl.exp(scores - lse[:, None])
        probabilities = tl.where(
            query_mask[:, None],
            probabilities,
            0.0,
        )
        if IS_CAUSAL:
            causal_mask = key_offsets[None, :] <= (
                cache_length + query_offsets[:, None]
            )
            probabilities = tl.where(
                causal_mask,
                probabilities,
                0.0,
            )
        accumulator += tl.sum(probabilities, axis=0)

    accumulator = tl.where(key_mask, accumulator, 0.0)
    output_base = (
        output_ptr + batch_index * stride_ob + query_head * stride_oh
    )
    tl.store(
        output_base + key_offsets * stride_on,
        accumulator,
        mask=key_mask,
    )


_SUPPORTED_HEAD_DIMS = (32, 64, 128, 256)
_BLOCK_MN_FOR_D: dict[int, tuple[int, int]] = {
    32: (32, 32),
    64: (64, 64),
    128: (32, 32),
    256: (16, 16),
}


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _padded_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor | None,
    block_dimension: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    head_dimension = q.shape[-1]
    if head_dimension != block_dimension:
        padding = block_dimension - head_dimension
        q = torch.nn.functional.pad(q, (0, padding))
        k = torch.nn.functional.pad(k, (0, padding))
        if v is not None:
            v = torch.nn.functional.pad(v, (0, padding))
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    if v is not None:
        v = v.to(torch.float32)
    return q, k, v


def _run_flash_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    causal: bool,
    cache_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, num_query_heads, query_length, head_dimension = q.shape
    _, num_kv_heads, key_length, _ = k.shape
    block_dimension = _next_power_of_2(head_dimension)
    if block_dimension not in _SUPPORTED_HEAD_DIMS:
        raise ValueError(
            f"head_dim={head_dimension} pads to unsupported {block_dimension}"
        )
    block_m, block_n = _BLOCK_MN_FOR_D[block_dimension]

    output = torch.empty(
        batch_size,
        num_query_heads,
        query_length,
        head_dimension,
        device=q.device,
        dtype=torch.float32,
    )
    lse = torch.empty(
        batch_size,
        num_query_heads,
        query_length,
        device=q.device,
        dtype=torch.float32,
    )
    q_padded, k_padded, v_padded = _padded_inputs(
        q,
        k,
        v,
        block_dimension,
    )
    if head_dimension != block_dimension:
        padded_output = torch.empty(
            batch_size,
            num_query_heads,
            query_length,
            block_dimension,
            device=q.device,
            dtype=torch.float32,
        )
    else:
        padded_output = output

    grid = (triton.cdiv(query_length, block_m), batch_size * num_query_heads)
    _flash_fwd_kernel[grid](
        q_padded,
        k_padded,
        v_padded,
        padded_output,
        lse,
        *q_padded.stride(),
        *k_padded.stride(),
        *v_padded.stride(),
        *padded_output.stride(),
        *lse.stride(),
        num_query_heads,
        num_kv_heads,
        query_length,
        key_length,
        softmax_scale,
        cache_length,
        IS_CAUSAL=causal,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_dimension,
    )
    if head_dimension != block_dimension:
        output.copy_(padded_output[..., :head_dimension])
    return output, lse


def _run_key_score_accumulation(
    q: torch.Tensor,
    k: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: float,
    causal: bool,
    cache_length: int,
) -> torch.Tensor:
    batch_size, num_query_heads, query_length, head_dimension = q.shape
    _, num_kv_heads, key_length, _ = k.shape
    block_dimension = _next_power_of_2(head_dimension)
    if block_dimension not in _SUPPORTED_HEAD_DIMS:
        raise ValueError(
            f"head_dim={head_dimension} pads to unsupported {block_dimension}"
        )
    block_m, block_n = _BLOCK_MN_FOR_D[block_dimension]
    key_scores = torch.zeros(
        batch_size,
        num_query_heads,
        key_length,
        device=q.device,
        dtype=torch.float32,
    )
    q_padded, k_padded, _ = _padded_inputs(q, k, None, block_dimension)

    grid = (triton.cdiv(key_length, block_n), batch_size * num_query_heads)
    _key_score_kernel[grid](
        q_padded,
        k_padded,
        lse,
        key_scores,
        *q_padded.stride(),
        *k_padded.stride(),
        *lse.stride(),
        *key_scores.stride(),
        num_query_heads,
        num_kv_heads,
        query_length,
        key_length,
        softmax_scale,
        cache_length,
        IS_CAUSAL=causal,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_dimension,
    )
    return key_scores


def flash_attn_with_key_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: Optional[float] = None,
    causal: bool = True,
    cache_len: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return flash-attention output and attention mass summed per key.

    ``q`` has shape ``[B, H_q, S_q, D]`` and ``k``/``v`` have shape
    ``[B, H_kv, S_k, D]``. For prefix-causal cache inference,
    ``S_k == cache_len + S_q``.
    """
    if not q.is_cuda:
        raise ValueError("flash_attn_with_key_scores requires CUDA tensors")
    batch_size, num_query_heads, query_length, head_dimension = q.shape
    del batch_size
    _, num_kv_heads, key_length, key_dimension = k.shape
    if num_query_heads % num_kv_heads:
        raise ValueError("Query heads must be divisible by KV heads")
    if k.shape != v.shape:
        raise ValueError("K and V must have the same shape")
    if key_dimension != head_dimension:
        raise ValueError("Q and K must have the same head dimension")
    if q.device != k.device or q.device != v.device:
        raise ValueError("Q, K, and V must be on the same device")
    if causal and key_length != cache_len + query_length:
        raise ValueError(
            "Prefix-causal attention requires key_length == cache_len + query_length"
        )

    scale = sm_scale if sm_scale is not None else 1.0 / math.sqrt(head_dimension)
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    output, lse = _run_flash_forward(q, k, v, scale, causal, cache_len)
    key_scores = _run_key_score_accumulation(
        q,
        k,
        lse,
        scale,
        causal,
        cache_len,
    )
    return output, key_scores
