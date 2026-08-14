"""Full-self-attention InternVL3 integration used for the VideoMME reference runs."""

import time
from functools import lru_cache
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torchvision.transforms as T
from accelerate import Accelerator, DistributedType
from decord import VideoReader, cpu
from loguru import logger as eval_logger
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoConfig, AutoModel, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.model_utils.reasoning_model_utils import parse_reasoning_model_answer
from lmms_eval.models.model_utils.video_prefetch_mixin import VideoPrefetchMixin
from lmms_eval.protocol import ChatMessages

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# ---- Video helpers (adapted from reference snippet) ----
def get_index(bound, fps, max_frame, first_idx=0, num_segments=32, max_fps=None):
    if fps <= 0:
        raise ValueError("Video FPS must be positive")
    if num_segments <= 0:
        raise ValueError("max_num_frames must be positive")
    if max_fps is not None and max_fps <= 0:
        raise ValueError("max_fps must be positive")
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)

    # Calculate the number of frames to sample
    if max_fps is not None:
        # Use max_fps as primary criteria
        duration = (end_idx - start_idx + 1) / fps
        frames_at_max_fps = max(1, int(duration * max_fps))
        # Cap at num_segments to avoid memory issues
        actual_num_segments = min(frames_at_max_fps, num_segments)
    else:
        # Use num_segments directly when max_fps not specified
        actual_num_segments = num_segments

    seg_size = float(end_idx - start_idx) / actual_num_segments if actual_num_segments > 0 else 1
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(actual_num_segments)
    ]) if actual_num_segments > 0 else np.array([])
    return frame_indices

@lru_cache(maxsize=1)
def _load_video_cached(
    video_path: str,
    bound: Optional[Tuple],
    input_size: int,
    max_num: int,
    num_segments: int,
    max_fps: Optional[float],
):
    """Cached video loading implementation."""
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())
    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(
        bound, fps, max_frame, first_idx=0, num_segments=num_segments, max_fps=max_fps
    )
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
        tiles = dynamic_preprocess(
            img, image_size=input_size, use_thumbnail=True, max_num=max_num
        )
        frame_tensor_list = [transform(tile) for tile in tiles]
        frame_tensor = torch.stack(frame_tensor_list)
        num_patches_list.append(frame_tensor.shape[0])
        pixel_values_list.append(frame_tensor)
    if len(pixel_values_list) == 0:
        return torch.empty(0), []
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list


def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=8, max_fps=None):
    """Return (pixel_values, num_patches_list) where pixel_values is concatenated tiles of sampled frames."""
    # Convert bound to tuple for caching (lists are not hashable)
    bound_tuple = tuple(bound) if bound is not None else None
    pixel_values, num_patches_list = _load_video_cached(
        video_path, bound_tuple, input_size, max_num, num_segments, max_fps
    )
    # Clone to prevent cache corruption from in-place modifications
    return (
        pixel_values.clone() if pixel_values.numel() > 0 else pixel_values,
        num_patches_list.copy(),
    )

class StopOnStrings(StoppingCriteria):
    def __init__(self, stop_strings: Sequence[str], tokenizer):
        """
        stop_ids: List[List[int]]  (each inner list = tokenized stop string)
        """
        super().__init__()
        self.stop_tensors = [torch.tensor(tokenizer.encode(p), dtype=torch.long) for p in stop_strings]

    def _ends_with(self, seq_ids: torch.Tensor, pat: torch.Tensor) -> bool:
        L = pat.numel()
        if L == 0 or seq_ids.size(0) < L:
            return False
        return torch.equal(seq_ids[-L:], pat.to(seq_ids.device))

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs) -> torch.BoolTensor:
        # Ensure buffers live on the same device
        finished = torch.zeros(input_ids.size(0), dtype=torch.bool, device=input_ids.device)
        stop_tensors = [p.to(input_ids.device) for p in self.stop_tensors]

        for i in range(input_ids.size(0)):
            if not finished[i]:
                seq = input_ids[i]
                for pat in stop_tensors:
                    if self._ends_with(seq, pat):
                        finished[i] = True
                        break

        # Return per-sequence flags (shape: [batch])
        return finished.clone()


@register_model("internvl3")
class InternVL3(VideoPrefetchMixin, lmms):
    is_simple = False

    def __init__(
        self,
        pretrained: str = "OpenGVLab/InternVL3-8B",
        device: str = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Union[int, str] = 1,
        attn_implementation: Optional[str] = None,
        max_num_frames: int = 32,
        max_fps: Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs:
            raise ValueError(f"Unexpected model arguments: {sorted(kwargs)}")
        if int(batch_size) != 1:
            raise ValueError("InternVL3 currently requires batch_size=1 per process")
        if attn_implementation not in [None, "flash_attention_2", "sdpa", "eager"]:
            raise ValueError(f"Unsupported attention implementation: {attn_implementation}")
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

        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": self.device_map,
            "trust_remote_code": True,
        }
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        config = AutoConfig.from_pretrained(pretrained, trust_remote_code=True)
        if attn_implementation is not None:
            setattr(config.llm_config, "_attn_implementation_autoset", False)
            setattr(config.llm_config, "attn_implementation", attn_implementation)
            setattr(config.llm_config, "_attn_implementation", attn_implementation)
        self._model = AutoModel.from_pretrained(pretrained, config=config, **model_kwargs).eval()

        self.max_num_frames = int(max_num_frames)
        self.max_fps = float(max_fps) if max_fps is not None else None
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)
        self.model.img_context_token_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        self._config = self.model.config
        self._max_length = 2048
        self.batch_size_per_gpu = 1

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

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
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
        raise NotImplementedError("Loglikelihood is not implemented for InternVL3.")

    def process_vision_info(
        self,
        chat_messages: List[ChatMessages],
        *,
        image_input_size: int = 448,
        image_max_num: int = 12,
        video_max_num: int = 1,
        video_num_segments: Optional[int] = None,  # default = self.max_num_frames
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        For each ChatMessages in the batch, collect images/videos, load them into tiles/frames,
        and return a single stacked tensor of pixel_values plus a per-prompt per-media patch matrix.

        Returns:
            pixel_values: torch.Tensor on self.device with shape [total_tiles, 3, H, W] (or empty tensor)
            num_patches_matrix: List[List[int]], one row per prompt; each entry is the tile count for an image
                                OR per-frame tile counts for a video (one entry per sampled frame).
        """
        all_tensors: List[torch.Tensor] = []
        num_patches_matrix: List[List[int]] = []

        num_segments = video_num_segments or self.max_num_frames

        for cm in chat_messages:
            patch_counts_for_prompt: List[int] = []

            imgs, vids, _ = cm.extract_media()  # (images, videos, audio)
            # Images
            for img_path in imgs:
                pv = load_image(img_path, input_size=image_input_size, max_num=image_max_num)
                patch_counts_for_prompt.append(pv.size(0))
                all_tensors.append(pv)

            # Videos (append one count per sampled frame)
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


    @torch.no_grad()
    def generate_with_media(
        self,
        prompts: Union[str, List[str]],
        *,
        pixel_values: Optional[torch.FloatTensor],
        num_patches_matrix: List[List[int]],
        **generate_kwargs,
    ) -> torch.Tensor:
        """Expand InternVL image placeholders and invoke the upstream generator."""
        is_batch = isinstance(prompts, (list, tuple))
        prompts = list(prompts) if is_batch else [prompts]
        if len(prompts) != len(num_patches_matrix):
            raise ValueError("Each prompt must have one row in num_patches_matrix")

        finalized_texts = []
        for prompt, patch_counts in zip(prompts, num_patches_matrix):
            missing_images = len(patch_counts) - prompt.count("<image>")
            if pixel_values is not None and missing_images > 0:
                prefix = "\n".join(["<image>"] * missing_images)
                prompt = f"{prefix}\n{prompt}" if prompt else prefix
            for patch_count in patch_counts:
                image_block = (
                    "<img>"
                    + "<IMG_CONTEXT>" * (self.model.num_image_token * patch_count)
                    + "</img>"
                )
                prompt = prompt.replace("<image>", image_block, 1)
            finalized_texts.append(prompt)

        self.tokenizer.padding_side = "left"
        model_inputs = self.tokenizer(finalized_texts, return_tensors="pt", padding=True)
        input_ids = model_inputs["input_ids"].to(self.device)
        attention_mask = model_inputs["attention_mask"].to(self.device)

        if "until" in generate_kwargs:
            stop_strings = generate_kwargs.pop("until")
            if isinstance(stop_strings, str):
                stop_strings = [stop_strings]
            generate_kwargs.setdefault(
                "stopping_criteria",
                StoppingCriteriaList([StopOnStrings(stop_strings, self.tokenizer)]),
            )

        return self.model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generate_kwargs,
        )


    def generate_until(self, requests: List[Instance]) -> List[str]:
        """Main entry point for generation with video prefetching."""

        def load_vision(chat_messages_raw, chat_messages):
            """Load vision data (images/videos) for a batch."""
            return self.process_vision_info(
                chat_messages,
                image_input_size=448,
                image_max_num=12,
                video_max_num=1,
                video_num_segments=self.max_num_frames,
            )

        def process_chunk(chunk_data, vision_data):
            """Process one chunk: run inference and collect results."""
            chunk, chat_messages_raw, chat_messages = chunk_data
            pixel_values, num_patches_matrix = vision_data

            # Extract chunk components
            ctx, doc_to_messages, all_gen_kwargs, doc_id, task, split = zip(*chunk)

            # Prepare generation kwargs
            current_gen_kwargs = dict(all_gen_kwargs[0]) if all_gen_kwargs else {}
            default_gen = {
                "max_new_tokens": 32,
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

            # Run inference
            start_time = time.time()
            cont = self.generate_with_media(
                prompts=texts,
                pixel_values=None if pixel_values.numel() == 0 else pixel_values,
                num_patches_matrix=num_patches_matrix,
                **current_gen_kwargs,
            )
            end_time = time.time()

            # Decode and collect results
            generated_ids_trimmed = cont
            answers = self.tokenizer.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True
            )

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

            metrics = {
                "latency": end_time - start_time,
                "tokens": sum(len(ids) for ids in generated_ids_trimmed),
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
