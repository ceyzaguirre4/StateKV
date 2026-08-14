# StateKV VideoMME reproduction

This release is based on
[`lmms-eval` commit `bb1ebe76`](https://github.com/EvolvingLMMs-Lab/lmms-eval/commit/bb1ebe76e7a942386c25c4664f902e0e59e8a401)
and contains only the InternVL3 code needed to reproduce the paper's VideoMME
experiments. It supports the `OpenGVLab/InternVL3-1B`,
`OpenGVLab/InternVL3-2B`, and `OpenGVLab/InternVL3-8B` checkpoints through
three lmms-eval model IDs:

| Model ID | Evaluation path | Paper setting |
| --- | --- | --- |
| `internvl3` | Full self-attention reference | 512 frames |
| `internvl3_rekv` | Recency-window baseline | 16 retrieved frames |
| `statekv_internvl3` | StateKV with Triton attention | 4,096-token carried state |

All three paths sample at no more than 1 FPS, cap each video at 512 frames,
use batch size 1, retain the full per-frame KV cache for answer generation,
and evaluate VideoMME without subtitles.

## Environment

The reported runs used Python 3.10, PyTorch 2.6.0, torchvision 0.21.0,
Transformers 4.57.1, Accelerate 1.10.1, Decord 0.6.0, FlashAttention 2.8.3,
and Triton 3.2.0. The provided Conda environment targets Linux with CUDA 12.4,
matching the validated setup:

```bash
conda env create -f environment.yml
conda activate statekv
python -m pip install flash-attn==2.8.3 --no-build-isolation
```

The editable repository install and the remaining Python dependencies are part
of `environment.yml`. Building FlashAttention separately is required because
its build isolation must be disabled; this step needs a CUDA 12.4-compatible
toolkit and compiler. On a different CUDA version, adjust the PyTorch wheel
index and build FlashAttention against that PyTorch installation.

StateKV uses the paper's two-pass Triton kernel during frame-by-frame cache
construction. The kernel computes the attention output and per-key attention
mass without materializing the full attention matrix. Set
`cache_attn_implementation=eager` to use the numerical reference path instead.
StateKV and ReKV use FlashAttention 2 for answer generation when it is
installed, and otherwise fall back to PyTorch SDPA.

Set `HF_TOKEN` if the local Hugging Face configuration requires it for the
VideoMME dataset. The first run downloads both the checkpoint and dataset.

## Run

The helper accepts a model size (`1B`, `2B`, or `8B`) and a method (`full`,
`rekv`, or `statekv`):

```bash
NUM_PROCESSES=1 examples/models/statekv_videomme.sh 1B statekv
NUM_PROCESSES=1 examples/models/statekv_videomme.sh 2B statekv
NUM_PROCESSES=1 examples/models/statekv_videomme.sh 8B statekv
```

Increase `NUM_PROCESSES` to shard the VideoMME examples across GPUs. Each
process loads a complete model replica, so this changes throughput rather than
the evaluation configuration. Set `OUTPUT_ROOT` to choose the results folder.

The equivalent direct StateKV command is:

```bash
accelerate launch --num_processes 1 --mixed_precision bf16 -m lmms_eval \
  --model statekv_internvl3 \
  --model_args pretrained=OpenGVLab/InternVL3-1B,max_num_frames=512,max_fps=1,cstate_size=4096,cache_attn_implementation=triton \
  --tasks videomme \
  --batch_size 1 \
  --log_samples \
  --output_path logs/videomme/internvl3-1b/statekv
```

Change `--model` to `internvl3` for full self-attention. For the recency
baseline, use `--model internvl3_rekv` and replace `cstate_size=4096` with
`retrieved_frames=16`.

## Reference scores

The paper reports VideoMME accuracy in the subtitles-free setting:

| Checkpoint | Full SA | ReKV (R=16) | StateKV (B=4096) |
| --- | ---: | ---: | ---: |
| InternVL3-1B | 46.19 | 37.11 | 45.80 |
| InternVL3-2B | 55.81 | 31.78 | 54.15 |
| InternVL3-8B | 64.19 | 54.56 | 62.52 |

The exposed release hyperparameters are intentionally narrow:

- `max_num_frames` caps uniformly sampled frames.
- `max_fps` caps sampling rate before the frame-count cap is applied.
- `cstate_size` controls StateKV's fixed carried-state token budget.
- `cache_attn_implementation` selects `triton` (default) or the `eager`
  numerical reference for StateKV cache construction.
- `retrieved_frames` controls the recency baseline's window size.

StateKV selects the carried state exclusively from video-to-video attention.
No language-query selection or other experimental cache policies are included.
