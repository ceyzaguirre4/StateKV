"""Shared frame-at-a-time InternVL3 cache construction.

Concrete release models provide only the carried-state budget and pruning rule.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from transformers import AutoConfig, AutoModel, AutoTokenizer, DynamicCache, GenerationConfig, StoppingCriteriaList
from transformers.models.qwen2.modeling_qwen2 import (
    apply_rotary_pos_emb,
    create_causal_mask,
    create_sliding_window_causal_mask,
    eager_attention_forward,
)
from transformers.utils import is_flash_attn_2_available

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.models.chat.internvl3 import StopOnStrings, load_image, load_video
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.model_utils.reasoning_model_utils import parse_reasoning_model_answer
from lmms_eval.models.model_utils.video_prefetch_mixin import VideoPrefetchMixin
from lmms_eval.protocol import ChatMessages


@dataclass
class PromptCacheState:
    past_key_values: DynamicCache | None = None
    attn_mask: torch.Tensor | None = None
    seq_len: int = 0  # Actual kept cache length after pruning.
    virtual_seq_len: int = 0  # Total tokens seen, including pruned tokens.


class StreamingInternVL3(VideoPrefetchMixin, lmms):
    is_simple = False
    method_name = "streaming InternVL3"

    def __init__(
        self,
        pretrained: str = "OpenGVLab/InternVL3-8B",
        device: str = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Union[int, str] = 1,
        max_num_frames: int = 32,
        max_fps: Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs:
            raise ValueError(f"Unexpected model arguments: {sorted(kwargs)}")
        if int(batch_size) != 1:
            raise ValueError(f"{self.method_name} requires batch_size=1 per process")
        if int(max_num_frames) <= 0:
            raise ValueError("max_num_frames must be positive")
        if max_fps is not None and float(max_fps) <= 0:
            raise ValueError("max_fps must be positive")

        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map or device

        config = AutoConfig.from_pretrained(pretrained, trust_remote_code=True)
        setattr(config.llm_config, "_attn_implementation_autoset", False)
        setattr(config.llm_config, "attn_implementation", "eager")
        setattr(config.llm_config, "_attn_implementation", "eager")
        self._model = AutoModel.from_pretrained(
            pretrained,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map=self.device_map,
            trust_remote_code=True,
            attn_implementation="eager",
        ).eval()
        setattr(self._model.language_model.model.config, "_attn_implementation", "eager")

        self.max_num_frames = int(max_num_frames)
        self.max_fps = float(max_fps) if max_fps is not None else None
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)
        self.model.img_context_token_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        self._config = self.model.config
        self._max_length = 2048
        self.batch_size_per_gpu = 1
        self.generation_attn_implementation = (
            "flash_attention_2" if is_flash_attn_2_available() else "sdpa"
        )

        if accelerator.num_processes > 1:
            if accelerator.distributed_type not in [DistributedType.FSDP, DistributedType.MULTI_GPU]:
                raise RuntimeError(f"Unsupported distributed type: {accelerator.distributed_type}")
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            if accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = accelerator.local_process_index
            self._world_size = accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    def _carried_state_size(self, tokens_per_frame: int) -> int:
        raise NotImplementedError

    def _prune_carried_state(
        self,
        state: PromptCacheState,
        attention_weights: list[torch.Tensor | None],
        max_tokens: int,
    ) -> PromptCacheState:
        raise NotImplementedError

    # ------------------- properties -------------------
    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError(f"Loglikelihood is not implemented for {self.method_name}.")

    # ------------------- multimodal prep -------------------
    def process_vision_info(
        self,
        chat_messages: List[ChatMessages],
        *,
        image_input_size: int = 448,
        image_max_num: int = 12,
        video_max_num: int = 1,
        video_num_segments: Optional[int] = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        all_tensors: List[torch.Tensor] = []
        num_patches_matrix: List[List[int]] = []
        num_segments = video_num_segments or self.max_num_frames
        for cm in chat_messages:
            patch_counts_for_prompt: List[int] = []
            imgs, vids, _ = cm.extract_media()
            for img_path in imgs:
                pv = load_image(img_path, input_size=image_input_size, max_num=image_max_num)
                patch_counts_for_prompt.append(pv.size(0))
                all_tensors.append(pv)
            for vid_path in vids:
                pv, npl = load_video(
                    vid_path,
                    input_size=image_input_size,
                    max_num=video_max_num,
                    num_segments=num_segments,
                    max_fps=self.max_fps,
                )
                if pv.numel() > 0:
                    patch_counts_for_prompt.extend(npl)
                    all_tensors.append(pv)
            num_patches_matrix.append(patch_counts_for_prompt)
        if len(all_tensors) > 0:
            pixel_values = torch.cat(all_tensors, dim=0).to(dtype).to(self.device)
        else:
            pixel_values = torch.empty(0, 3, image_input_size, image_input_size, dtype=dtype, device=self.device)
        return pixel_values, num_patches_matrix

    # ------------------- cache builder -------------------
    @torch.no_grad()
    def run_with_attention(
        self,
        model,
        inputs_embeds: torch.FloatTensor,              # (B, T, D)
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,                          # DynamicCache | None
        use_cache: bool = False,
        cache_position: torch.LongTensor | None = None,
    ):
        """
        Mirrors lmms_model.model.language_model.model.forward + layers/self_attn.forward with full attention.
        Returns dict(last_hidden_state, past_key_values).
        """

        # ---- code from lmms_model.model.language_model.model.forward ----
        # initialize cache like upstream
        if use_cache and past_key_values is None:
            if DynamicCache is None:
                raise RuntimeError("transformers.cache_utils.DynamicCache not available")
            past_key_values = DynamicCache(config=model.config)

        if cache_position is None:
            if past_key_values is not None:
                raise ValueError("cache_position is required when a compressed cache is supplied")
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across decoder layers
        # (cos, sin) tuple exactly as upstream computes
        position_embeddings = model.rotary_emb(hidden_states, position_ids)
        # ----------------------------------------------------------------

        # Restore upstream mask creation (needed when applying attention)
        # It may already have been prepared by e.g. `generate`; here we mirror upstream behavior.
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": model.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            if getattr(model, "has_sliding_layers", False):
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        # ---- iterate decoder layers as in lmms_model.model.language_model.model.forward ----
        attn_weights_layers = []    # keep track of attention weights per layer for pruning later

        for layer in model.layers[: model.config.num_hidden_layers]:
            # ---- code from lmms_model.model.language_model.model.layers[0].forward ----
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)

            # ---- "self_attn.forward" (apply full attention exactly like upstream) ----
            attn = layer.self_attn
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, attn.head_dim)

            # q/k/v projections
            query_states = attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            key_states   = attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            value_states = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            # rotary positions on q,k
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            # cache update as upstream
            if past_key_values is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_values.update(
                    key_states, value_states, attn.layer_idx, cache_kwargs
                )

            # select attention implementation and compute attention with mask
            if model.config._attn_implementation != "eager":
                raise RuntimeError(f"{self.method_name} cache construction requires eager attention")

            attn_output, attn_weights = eager_attention_forward(
                attn,
                query_states,
                key_states,
                value_states,
                causal_mask_mapping[layer.attention_type],
                dropout=0.0 if not model.training else attn.attention_dropout,
                scaling=attn.scaling,
                sliding_window=attn.sliding_window,
            )
            attn_weights_layers.append(attn_weights)

            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = attn.o_proj(attn_output)

            hidden_states = residual + attn_output

            # Fully Connected path as upstream
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states
            # ---- end layer.forward ----

        # final norm as in lmms_model.model.language_model.model.forward
        hidden_states = model.norm(hidden_states)

        # return same shapes/objects as BaseModelOutputWithPast would carry
        result = {
            "last_hidden_state": hidden_states,
            "past_key_values": past_key_values if use_cache else None,
            "attn_weights_layers": attn_weights_layers,
        }
        return result

    @torch.no_grad()
    def _build_prompt_cache(
        self,
        pixel_values: torch.Tensor,
        num_patches_matrix: List[List[int]],
        prompt_text: str = "Describe the video in detail.",
        IMG_START_TOKEN: str = "<img>",
        IMG_END_TOKEN: str = "</img>",
        IMG_CONTEXT_TOKEN: str = "<IMG_CONTEXT>",
        max_new_tokens: int = 1024,
    ):
        """
        Build the detailed and carried video caches from prefetched frames.

        Args:
            pixel_values: Tensor of shape [M, 3, H, W] containing video frames
            num_patches_matrix: List of patch counts per frame
            prompt_text: Text prompt used to size the RoPE range
            max_new_tokens: Maximum new tokens for generation

        Returns:
            Tuple of (cache_kv, cache_mask, cache_pos, old_inv_freq, old_max_seq_len, old_original_max_seq_len)
        """
        if pixel_values is None or pixel_values.numel() == 0:
            raise ValueError(f"{self.method_name} requires a video input")
        patch_counts = [int(v) for v in num_patches_matrix[0]]  # assumes either batch_size=1 or every element in the batch has the same patch structure.
        assert pixel_values.dim() == 4 and pixel_values.shape[1] == 3, "pixel_values must be [M,3,H,W]"
        assert len(num_patches_matrix) == 1 or all(
            num_patches_matrix[0] == npm for npm in num_patches_matrix
        ), "All elements in the batch must have the same patch structure."

        IMG_CTX_PER_PATCH = self.model.num_image_token
        assert type(IMG_CTX_PER_PATCH) == int
        multi_img_block = ""
        for npatches in patch_counts:
            multi_img_block += (
                IMG_START_TOKEN
                + (IMG_CONTEXT_TOKEN * (IMG_CTX_PER_PATCH * npatches))
                + IMG_END_TOKEN
                + "\n"
            )
        if not patch_counts or len(set(patch_counts)) != 1:
            raise ValueError(f"{self.method_name} requires the same number of visual patches per video frame")
        NUM_TOKENS_PER_IMAGE = (IMG_CTX_PER_PATCH * patch_counts[0]) + 3
        cstate_size = self._carried_state_size(NUM_TOKENS_PER_IMAGE)

        model_inputs = self.tokenizer([multi_img_block], return_tensors="pt", padding=False)
        input_ids = model_inputs["input_ids"].to(self.device)            # (1, T_now)
        attention_mask = model_inputs["attention_mask"].to(self.device)  # (1, T_now)
        vit_embeds = self.model.extract_feature(pixel_values)  # (#ctx, D)
        input_embeds = self.model.language_model.get_input_embeddings()(input_ids)  # (1, T_now, D)

        flat_ids = input_ids.view(-1)
        flat_embeds = input_embeds.view(-1, input_embeds.shape[-1])
        sel = (flat_ids == self.model.img_context_token_id)
        n_slots = int(sel.sum().item())
        vi = vit_embeds.reshape(-1, vit_embeds.shape[-1]).to(flat_embeds.device)
        assert vi.shape[0] == n_slots, f"ViT/context mismatch: {vi.shape[0]} vs {n_slots}"
        flat_embeds[sel] = vi
        input_embeds = flat_embeds.view_as(input_embeds)

        cstate = PromptCacheState() # the compressed state
        cstate.past_key_values = DynamicCache(config=self.model.language_model.model.config)
        dstate = PromptCacheState() # the no-interframe attention state
        dstate.past_key_values = DynamicCache(config=self.model.language_model.model.config)
        F = len(patch_counts)
        frames_per_step = 1

        # Manual RoPE scaling for long videos
        # Save old RoPE state for restoration after processing
        rope_module = self.model.language_model.model.rotary_emb
        old_inv_freq = rope_module.inv_freq.clone()
        old_max_seq_len_cached = rope_module.max_seq_len_cached
        old_original_max_seq_len = rope_module.original_max_seq_len

        # Calculate expected maximum sequence length
        video_tokens = F * NUM_TOKENS_PER_IMAGE
        prompt_ids = self.tokenizer([prompt_text], return_tensors="pt", padding=False)["input_ids"]
        prompt_tokens = prompt_ids.shape[1]
        expected_max_seq_len = video_tokens + prompt_tokens + max_new_tokens + 128  # +128 buffer for safety

        eval_logger.debug(
            f"Token count breakdown: "
            f"F={F} frames, "
            f"video_tokens={video_tokens} ({F} * {NUM_TOKENS_PER_IMAGE}), "
            f"input_embeds.shape[1]={input_embeds.shape[1]}, "
            f"prompt_tokens={prompt_tokens}, "
            f"max_new_tokens={max_new_tokens}, "
            f"expected_max_seq_len={expected_max_seq_len}"
        )

        config = self.model.language_model.model.config
        current_max = config.max_position_embeddings

        # Log RoPE configuration for debugging
        rope_config = getattr(config, "rope_scaling", None)
        eval_logger.debug(
            f"RoPE config: type={rope_config.get('rope_type', 'unknown') if rope_config else 'none'}, "
            f"factor={rope_config.get('factor', 'N/A') if rope_config else 'N/A'}, "
            f"rope_theta={getattr(config, 'rope_theta', 'N/A')}, "
            f"max_position_embeddings={current_max}"
        )
        eval_logger.debug(f"RoPE module before scaling: inv_freq.shape={rope_module.inv_freq.shape}, max_seq_len_cached={rope_module.max_seq_len_cached}")

        # If expected sequence exceeds training length, manually compute scaled RoPE
        if expected_max_seq_len > current_max:
            # Manually compute dynamic NTK scaled inv_freq
            # Based on _compute_dynamic_ntk_parameters in modeling_rope_utils.py
            base = config.rope_theta
            partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
            head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
            dim = int(head_dim * partial_rotary_factor)
            factor = config.rope_scaling["factor"]
            effective_factor = (factor * expected_max_seq_len / current_max) - (factor - 1)

            eval_logger.debug(
                f"Video: {F} frames ({video_tokens} tokens), Prompt: {prompt_tokens} tokens, "
                f"Generation: {max_new_tokens} tokens. Expected max: {expected_max_seq_len} > {current_max}. "
                f"Scaling factor: {factor}, effective factor: {effective_factor:.3f}. "
                f"Manually computing scaled RoPE..."
            )

            # Dynamic NTK scaling formula: base * ((factor * seq_len / max_pos) - (factor - 1)) ** (dim / (dim - 2))
            scaled_base = base * ((factor * expected_max_seq_len / current_max) - (factor - 1)) ** (dim / (dim - 2))
            inv_freq = 1.0 / (
                scaled_base ** (torch.arange(0, dim, 2, dtype=torch.int64, device=self.device).float() / dim)
            )
            attention_scaling = 1.0  # Unused in dynamic NTK

            # Set new RoPE parameters
            rope_module.register_buffer("inv_freq", inv_freq, persistent=False)
            rope_module.attention_scaling = attention_scaling
            rope_module.max_seq_len_cached = expected_max_seq_len
            rope_module.original_max_seq_len = expected_max_seq_len  # Prevent reset

            eval_logger.debug(
                f"RoPE manually scaled: max_seq_len_cached={rope_module.max_seq_len_cached}, "
                f"inv_freq[0]={inv_freq[0].item():.6f}, inv_freq[-1]={inv_freq[-1].item():.6f}"
            )

            # Store values for override detection
            manual_inv_freq_first = inv_freq[0].item()
            manual_inv_freq_last = inv_freq[-1].item()
        else:
            manual_inv_freq_first = None
            manual_inv_freq_last = None
            eval_logger.debug(
                f"Video: {F} frames ({video_tokens} tokens), expected max: {expected_max_seq_len} <= {current_max}. "
                f"Using original RoPE."
            )

        eval_logger.debug(f"Building {self.method_name} cache from {F} frames")
        for frame_index in range(F):
            start = frame_index * NUM_TOKENS_PER_IMAGE
            stop = (frame_index + frames_per_step) * NUM_TOKENS_PER_IMAGE
            cstate, dstate = self._build_prompt_cache_incremental(
                input_embeds=input_embeds[:, start:stop, :],
                attention_mask=attention_mask[:, start:stop],
                cstate=cstate,
                dstate=dstate,
                cstate_size=cstate_size,
            )
            eval_logger.debug(
                f"Encoded frame {frame_index + 1}/{F}; "
                f"cstate={cstate.seq_len} tokens, dstate={dstate.seq_len} tokens"
            )

        # Detect if RoPE was overridden during forward passes
        if manual_inv_freq_first is not None:
            current_inv_freq_first = rope_module.inv_freq[0].item()
            current_inv_freq_last = rope_module.inv_freq[-1].item()
            if abs(current_inv_freq_first - manual_inv_freq_first) > 1e-6 or abs(current_inv_freq_last - manual_inv_freq_last) > 1e-6:
                eval_logger.warning(
                    f"⚠️  RoPE WAS OVERRIDDEN! "
                    f"Manual: inv_freq[0]={manual_inv_freq_first:.6f}, inv_freq[-1]={manual_inv_freq_last:.6f}. "
                    f"Current: inv_freq[0]={current_inv_freq_first:.6f}, inv_freq[-1]={current_inv_freq_last:.6f}. "
                    f"max_seq_len_cached={rope_module.max_seq_len_cached}"
                )
            else:
                eval_logger.debug(
                    f"✓ RoPE parameters preserved through forward passes. "
                    f"inv_freq[0]={current_inv_freq_first:.6f}, max_seq_len_cached={rope_module.max_seq_len_cached}"
                )

        # Keep the scaled RoPE active through answer generation.
        # (RoPE must remain scaled during generation to match cache positions)
        return (
            dstate.past_key_values,
            dstate.attn_mask,
            dstate.virtual_seq_len,
            old_inv_freq,
            old_max_seq_len_cached,
            old_original_max_seq_len,
        )

    @torch.no_grad()
    def _build_prompt_cache_incremental(
        self,
        input_embeds: torch.FloatTensor,    # (1, T_now, D)
        attention_mask: torch.Tensor,      # (1, T_now)
        cstate: PromptCacheState | None = None,
        dstate: PromptCacheState | None = None,
        cstate_size: int = 4096,
    ):
        T_now = int(input_embeds.shape[1])

        # Prepare mask for cstate
        if cstate.attn_mask is not None:
            acc_mask_cstate = torch.cat([cstate.attn_mask, attention_mask], dim=-1)
        else:
            acc_mask_cstate = attention_mask

        # Cache position based on virtual sequence length (not actual cache length after compression)
        cache_position = torch.arange(
            cstate.virtual_seq_len,
            cstate.virtual_seq_len + T_now,
            device=input_embeds.device
        )

        # Forward with the carried state and retain attention for StateKV selection.
        out_cstate = self.run_with_attention(
            self.model.language_model.model,
            inputs_embeds=input_embeds,
            attention_mask=acc_mask_cstate,
            use_cache=True,
            past_key_values=cstate.past_key_values,
            cache_position=cache_position,
        )
        attn_layers = out_cstate["attn_weights_layers"]    # video-to-video attention weights
        cstate.past_key_values = out_cstate["past_key_values"]

        # Update dstate with the same key/value states that were added to cstate
        # Extract the newly added tokens (last T_now tokens) from cstate and add to dstate
        n_layers = len(cstate.past_key_values.layers)
        for li in range(n_layers):
            cstate_layer = cstate.past_key_values.layers[li]
            # Extract the last T_now tokens that were just added
            new_keys = cstate_layer.keys[:, :, -T_now:, :]      # (B, H_kv, T_now, D)
            new_values = cstate_layer.values[:, :, -T_now:, :]  # (B, H_kv, T_now, D)

            # Update dstate with these new key/value states
            # Note: DynamicCache.update expects (B, H, T, D) tensors and handles concatenation internally
            dstate.past_key_values.update(
                new_keys, new_values, li,
                {"cache_position": cache_position}
            )

        # Select the next carried state using video-to-video attention only.
        cstate = self._prune_carried_state(
            state=cstate,
            attention_weights=attn_layers,
            max_tokens=cstate_size,
        )

        cstate.virtual_seq_len += T_now
        dstate.attn_mask = torch.cat([dstate.attn_mask, attention_mask], dim=-1) if dstate.attn_mask is not None else attention_mask
        dstate.seq_len = cstate.virtual_seq_len
        dstate.virtual_seq_len = cstate.virtual_seq_len
        return cstate, dstate

    @torch.no_grad()
    def _only_language_model(self, *, tokenizer, texts: List[Union[str, List[Dict[str, Any]]]], generation_config: Optional[GenerationConfig] = None, return_text: bool = True, skip_special_tokens: bool = True, **generate_kwargs):
        def _apply_stopping(kwargs):
            if "until" in kwargs:
                stop_strings = kwargs.pop("until")
                if isinstance(stop_strings, str):
                    stop_strings = [stop_strings]
                stopping = StoppingCriteriaList([StopOnStrings(stop_strings, tokenizer)])
                kwargs.setdefault("stopping_criteria", stopping)
        gen_kwargs = {"max_new_tokens": 1024, "do_sample": True, "temperature": 0.7, "top_p": 0.9, "num_beams": 1}
        gen_kwargs.update({k: v for k, v in generate_kwargs.items() if v is not None})
        _apply_stopping(gen_kwargs)
        tokenizer.padding_side = "right"    # moved to right since we're receiving an equally sized kvcache per row in batch
        encoded = tokenizer(texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in encoded.items()}
        if "past_key_values" in gen_kwargs:
            assert "pkv_mask" in gen_kwargs, "If passing a kvcache you must also pass the attention mask used to build it."
            new_ids = inputs["input_ids"]
            _, L2 = new_ids.shape
            assert L2 > 0

            # build mask + positions
            old_mask = gen_kwargs.pop("pkv_mask").to(new_ids.device)
            inputs["attention_mask"] = torch.cat([old_mask, inputs["attention_mask"]], dim=-1)

            # >>> use provided virtual position instead of cache length since cache size doesn't correspond to actual positioning
            pos_base = int(gen_kwargs.pop("pkv_pos_base"))
            inputs["cache_position"] = torch.arange(pos_base, pos_base + L2, device=new_ids.device)

        attn_implementation_backup = self.model.language_model.model.config._attn_implementation
        setattr(
            self.model.language_model.model.config,
            "_attn_implementation",
            self.generation_attn_implementation,
        )
        try:
            outputs = self.model.language_model.generate(
                **inputs,
                generation_config=generation_config,
                use_cache=True,
                **gen_kwargs,
            )
        finally:
            setattr(self.model.language_model.model.config, "_attn_implementation", attn_implementation_backup)

        if return_text:
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(encoded.input_ids, outputs)]
            return tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
        return outputs

    # ------------------- main entry: generate_until -------------------
    def generate_until(self, requests: List[Instance]) -> List[str]:
        """Main entry point for generation with video prefetching."""

        def load_vision(chat_messages_raw, chat_messages):
            """Load vision data (videos/images) for a batch."""
            return self.process_vision_info(
                chat_messages,
                image_input_size=448,
                image_max_num=12,
                video_max_num=1,
                video_num_segments=self.max_num_frames,
            )

        def process_chunk(chunk_data, vision_data):
            """Process one chunk: build cache, run inference, collect results."""
            chunk, chat_messages_raw, chat_messages = chunk_data
            pixel_values, num_patches_matrix = vision_data

            # Extract chunk components
            ctx, doc_to_messages, all_gen_kwargs, doc_id, task, split = zip(*chunk)

            # Prepare generation kwargs
            current_gen_kwargs = dict(all_gen_kwargs[0]) if all_gen_kwargs else {}
            default_gen = {
                "max_new_tokens": 1024,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
                "num_beams": 1,
            }
            for k, v in default_gen.items():
                current_gen_kwargs.setdefault(k, v)

            # Prepare text prompts
            batched_messages = [chat_message.to_hf_messages() for chat_message in chat_messages]
            batched_messages_textonly = [
                [
                    dict(
                        role=m["role"],
                        content="".join(
                            c["text"] for c in m["content"] if c["type"] == "text"
                        ),
                    )
                    for m in group
                ]
                for group in batched_messages
            ]
            texts = [
                self.tokenizer.apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=True
                )
                for msg in batched_messages_textonly
            ]

            if len(texts) != 1:
                raise ValueError(f"{self.method_name} requires one prompt per process")
            prompt_text = texts[0]

            # Build cache from prefetched vision data
            start_time = time.time()
            (
                prompt_cache_kv,
                prompt_cache_mask,
                prompt_cache_pos,
                old_inv_freq,
                old_max_seq_len_cached,
                old_original_max_seq_len,
            ) = self._build_prompt_cache(
                pixel_values,
                num_patches_matrix,
                prompt_text=prompt_text,
                max_new_tokens=current_gen_kwargs.get("max_new_tokens", 1024),
            )

            rope_module = self.model.language_model.model.rotary_emb
            try:
                answers = self._only_language_model(
                    tokenizer=self.tokenizer,
                    texts=texts,
                    generation_config=None,
                    return_text=True,
                    skip_special_tokens=True,
                    past_key_values=prompt_cache_kv,
                    pkv_mask=prompt_cache_mask,
                    pkv_pos_base=prompt_cache_pos,
                    **current_gen_kwargs,
                )
            finally:
                rope_module.register_buffer("inv_freq", old_inv_freq, persistent=False)
                rope_module.max_seq_len_cached = old_max_seq_len_cached
                rope_module.original_max_seq_len = old_original_max_seq_len
                eval_logger.debug("RoPE restored to original state after generation")

            end_time = time.time()

            # Collect results
            chunk_res = []
            chunk_raw = []
            for ans, context in zip(answers, texts):
                clean_ans = parse_reasoning_model_answer(ans)
                chunk_res.append(clean_ans)
                chunk_raw.append(ans)
                self.cache_hook.add_partial(
                    "generate_until", (context, current_gen_kwargs), clean_ans
                )

                eval_logger.debug(f"Question: {context}")
                eval_logger.debug(f"Model Raw Response: {ans}")
                eval_logger.debug(f"Model Clean Response: {clean_ans}")

            # Clean up
            del prompt_cache_kv
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            metrics = {
                "latency": end_time - start_time,
                "tokens": 0,
            }
            return chunk_res, chunk_raw, metrics

        # Use prefetching mixin to process all requests
        res, _, e2e_latency, total_tokens = self._generate_until_with_prefetch(
            requests, load_vision, process_chunk
        )

        # Log timing metrics
        avg_speed = total_tokens / e2e_latency if e2e_latency > 0 else 0
        metric_dict = {
            "total_tokens": total_tokens,
            "e2e_latency": e2e_latency,
            "avg_speed": avg_speed,
            "additional_metrics": {"rank": self.rank},
        }
        log_metrics(**metric_dict)

        return res

    def generate_until_multi_round(self, *args, **kwargs):
        raise NotImplementedError()
