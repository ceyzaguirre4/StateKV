# StateKV

## Linear Scaling Video VLMs for Long Video Understanding

Turning pretrained video VLMs into linearly-scaling streaming systems without
retraining.

[Cristobal Eyzaguirre](https://ceyzaguirre4.github.io/),
[Jiajun Wu](https://jiajunwu.com/), and
[Juan Carlos Niebles](https://www.niebles.net/)<br>
Stanford University

[Project page](https://ceyzaguirre4.github.io/StateKV/) ·
[Paper](https://arxiv.org/abs/2605.31598) ·
[Reproduction guide](STATEKV.md) ·
[Upstream `lmms-eval`](https://github.com/EvolvingLMMs-Lab/lmms-eval)

![StateKV maintaining and updating a compact state across video frames](https://ceyzaguirre4.github.io/StateKV/assets/statekv-animation.gif)

## The problem

The biggest problem in video understanding today isn't the models. It's that
we can barely run them.

Video VLMs have finally gotten good; even open-source models post surprisingly
strong benchmark numbers. But these models are increasingly vital for
long-horizon and streaming tasks such as autonomous driving and embodied
robotics, where a system must integrate evidence over minutes or hours, often
in real time. And that's exactly where today's architectures break down.

The central challenge is that computational cost grows quadratically with the
length of the video. Because dominant architectures let each frame attend over
all previous video tokens, the per-frame cost rises as the video grows, and the
overall cost of processing the entire video scales quadratically with its
length.

This creates a significant bottleneck for practical deployment and rules out
real-time streaming applications entirely: a car that has driven for an hour
is, in principle, harder to query than one that just pulled out of the
driveway, even when both need an answer right now. But a real-time system needs
its response time to stay flat as it keeps running. Linear overall complexity,
or equivalently constant per-frame cost, is therefore central to scalable
long-video understanding, and a critical prerequisite for streaming.

![Long-video cost scaling for full attention and efficient alternatives](https://ceyzaguirre4.github.io/StateKV/assets/figure-1.svg)

*Long-video inference becomes expensive when each frame carries the full
history forward: full self-attention (gray) scales quadratically, while
StateKV (red) keeps a linear cost.*

Our goal was simple: turn a pretrained video VLM into a linearly-scaling system
without retraining it. To do that, we started by asking a simple question: does
every frame really need access to the entire history?

## Why this should work

### Observation 1: Most of the past does not matter

When a video model processes a frame, it technically has access to every token
from every previous frame. But does it actually use all of them?

When we analyzed attention patterns in pretrained video VLMs, we found that
most cross-frame attention is concentrated on a surprisingly small subset of
tokens. In many cases, a few thousand carefully chosen tokens capture the
overwhelming majority of the attention mass.

If we somehow knew which tokens those were, we could approximate the effect of
full attention without paying the full quadratic cost.

![Attention concentration analysis for pretrained video VLMs](https://ceyzaguirre4.github.io/StateKV/assets/observation-1-concentration.png)

![Summary statistics showing how top-B tokens capture historical attention mass](https://ceyzaguirre4.github.io/StateKV/assets/stats-assumption1.svg)

*Attention concentrates on a small subset of past tokens.*

### Observation 2: The important tokens change slowly

Finding the important tokens for frame *n* is only useful if we can also find
them for frame *n+1*. Fortunately, the set of useful tokens evolves gradually.

The tokens that matter for one frame tend to remain useful for the next frame,
with only a small number of additions and removals over time. This suggests
that instead of recomputing an optimal memory from scratch, we can maintain and
update a compact state as the video progresses.

![Weighted recall analysis for selected tokens across adjacent frames](https://ceyzaguirre4.github.io/StateKV/assets/observation-1-recall.png)

*Useful tokens remain predictive across nearby frames.*

![Token churn showing that important token sets change gradually](https://ceyzaguirre4.github.io/StateKV/assets/observation-2-churn.png)

*Important tokens evolve gradually over time.*

## The method

StateKV maintains two memories:

- A **detailed state** stores every token so that we preserve full information
  for final decoding.
- A **compressed state** stores only the most important tokens and is used to
  carry information between frames.

After processing each frame, we update this compressed state using attention
scores and carry it forward to the next frame. The result is a recurrent
architecture with fixed compute per frame, built entirely from a frozen
pretrained transformer.

![StateKV method diagram showing detailed and compressed states](https://ceyzaguirre4.github.io/StateKV/assets/method-statekv.png)

*StateKV keeps full information for decoding while carrying a compact
recurrent memory between frames.*

## Results

### Better compute-quality tradeoffs

StateKV changes the Pareto frontier in two ways. First, it lets us
significantly reduce computation while maintaining, or mostly maintaining,
performance. Second, the compute reduction is large enough that we can run
larger models at similar cost to a smaller full self-attention model.

| Lower compute at similar quality | Larger models at similar cost |
| --- | --- |
| ![StateKV moving left on the Pareto frontier](https://ceyzaguirre4.github.io/StateKV/assets/pareto-transition.gif) | ![StateKV enabling a larger model at similar cost](https://ceyzaguirre4.github.io/StateKV/assets/pareto-larger-models.gif) |

### Why linear scaling matters more over time

The same compute gap becomes much starker as videos get longer. Extrapolating
from 512 frames to 1,024 frames and then to 3,600 frames, roughly an hour of
video, the difference between O(N) and O(N²) scaling dominates the practical
cost of inference.

![Frame-scaling extrapolation from 512 to 3,600 frames](https://ceyzaguirre4.github.io/StateKV/assets/frame-scaling.gif)

### Across model families, parameter scales, and datasets

The same carried-memory intervention transfers across seven video VLMs with
different parameter counts and model families, evaluated on three video
benchmarks. StateKV closely approximates full self-attention while consistently
outperforming prior streaming work, including ReKV.

For a fair efficiency comparison, sliding-window retrieval with 16 frames and
StateKV with cache budget *B* = 4,096 are compute-matched.

![Results comparing full self-attention, sliding window, ReKV, and StateKV](https://ceyzaguirre4.github.io/StateKV/assets/results-table.png)

## A Triton kernel for practical wall time

StateKV needs attention-derived importance scores while building the cache. A
naive eager implementation can expose those scores, but it materializes the
full attention matrix. To make the cache-building path practical, we implement
a custom Triton kernel inspired by FlashAttention-style tiling: it computes
attention without forming the full matrix while accumulating the statistics
needed for token selection.

This matters in wall-clock time. At the main cache budget we evaluate,
*B* = 4,096 tokens, StateKV is faster per frame than full self-attention with
FlashAttention-2, even for the largest models in our comparison. The bounded
cache turns the algorithmic scaling advantage into an actual runtime
advantage.

![Wall-time comparison of StateKV with Triton and full self-attention with FlashAttention-2](https://ceyzaguirre4.github.io/StateKV/assets/triton-walltime.png)

*Measured per-frame wall time on an NVIDIA L40S.*

## Code release

This repository is a deliberately minimal release for reproducing the paper's
subtitles-free VideoMME experiments with InternVL3. It supports the
`OpenGVLab/InternVL3-1B`, `OpenGVLab/InternVL3-2B`, and
`OpenGVLab/InternVL3-8B` checkpoints through three evaluation paths:

| Method | `lmms-eval` model ID | Configuration |
| --- | --- | --- |
| Full self-attention | `internvl3` | 512 frames |
| ReKV baseline | `internvl3_rekv` | 16 retrieved frames |
| StateKV | `statekv_internvl3` | 4,096-token carried state |

The StateKV path defaults to the two-pass Triton kernel. An eager PyTorch
implementation remains available as a numerical reference. Unused
experimental policies and unrelated model-family implementations are
intentionally excluded.

### Installation

The provided Conda environment matches the validated Linux and CUDA 12.4
setup:

```bash
git clone https://github.com/ceyzaguirre4/StateKV.git
cd StateKV
conda env create -f environment.yml
conda activate statekv
python -m pip install flash-attn==2.8.3 --no-build-isolation
```

The environment uses Python 3.10, PyTorch 2.6.0, Transformers 4.57.1,
FlashAttention 2.8.3, and Triton 3.2.0. FlashAttention must be installed after
the environment is created because its build isolation needs to be disabled.
See [the reproduction guide](STATEKV.md#environment) for CUDA and Hugging Face
setup notes.

### Reproduce VideoMME

Run StateKV by choosing an InternVL3 checkpoint size:

```bash
NUM_PROCESSES=1 examples/models/statekv_videomme.sh 1B statekv
NUM_PROCESSES=1 examples/models/statekv_videomme.sh 2B statekv
NUM_PROCESSES=1 examples/models/statekv_videomme.sh 8B statekv
```

The helper also accepts `full` and `rekv` in place of `statekv`. All paper
configurations use at most 1 FPS, cap videos at 512 frames, use batch size 1,
and evaluate VideoMME without subtitles. Checkpoints and the dataset are
downloaded from Hugging Face on first use.

For direct commands, multi-GPU evaluation, output configuration, and the
`cache_attn_implementation=eager` reference path, see
[STATEKV.md](STATEKV.md#run).

### InternVL3 VideoMME scores

| Checkpoint | Full SA | ReKV (`R=16`) | StateKV (`B=4096`) |
| --- | ---: | ---: | ---: |
| InternVL3-1B | 46.19 | 37.11 | 45.80 |
| InternVL3-2B | 55.81 | 31.78 | 54.15 |
| InternVL3-8B | 64.19 | 54.56 | 62.52 |

The released Triton path was validated end-to-end with InternVL3-1B and 512
frames on six NVIDIA L40S GPUs: it processed all 2,700 questions with no
invalid answers and scored 46.89. Answer generation uses sampling, so small
run-to-run differences from the paper table are expected.

## Relationship to `lmms-eval`

StateKV is built on
[`lmms-eval` at commit `bb1ebe76`](https://github.com/EvolvingLMMs-Lab/lmms-eval/commit/bb1ebe76e7a942386c25c4664f902e0e59e8a401).
This repository retains the evaluation framework and adds the minimal
InternVL3 streaming, StateKV, ReKV, and Triton components needed for the paper
reproduction. For the full evaluation framework, its supported models and
tasks, and general documentation, use the
[upstream `lmms-eval` repository](https://github.com/EvolvingLMMs-Lab/lmms-eval).

## Citation

```bibtex
@misc{eyzaguirre2026linearscalingvideovlms,
  title         = {Linear Scaling Video VLMs for Long Video Understanding},
  author        = {Cristobal Eyzaguirre and Jiajun Wu and Juan Carlos Niebles},
  year          = {2026},
  eprint        = {2605.31598},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2605.31598}
}
```

## License

StateKV retains the licenses and attribution of the upstream evaluation
framework. See [LICENSE](LICENSE) for details.
