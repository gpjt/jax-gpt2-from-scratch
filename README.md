# GPT-2 small from scratch, in JAX

A from-scratch implementation of the GPT-2 architecture, built and trained in
JAX (with [Flax NNX](https://flax.readthedocs.io/en/stable/),
[Optax](https://optax.readthedocs.io/en/latest/) and
[Orbax](https://orbax.readthedocs.io/en/latest/)) on a single RTX 3090.

This is the capstone project of my
"[Writing an LLM from scratch](https://www.gilesthomas.com/llm-from-scratch)"
blog series: a GPT-2 small (163M parameters), written from my notes alone -- no
reference to books or to the PyTorch code I'd written before -- and trained on
3.26B tokens of FineWeb. The final model scored 3.42 loss on my held-back test
set, a little better than the original OpenAI GPT-2 small weights (3.50) on the
same data.

**Status: complete.** This repo is the finished endpoint of that series and
I'm not planning further development here -- new experiments (RoPE, MoE, and
whatever comes next) will happen in new repos. Issues and PRs are unlikely to
get much attention, but the code should keep working as a reference
implementation.

Two posts describe how it was all put together:

* [Part 34a -- building a JAX training loop for an LLM training run](https://www.gilesthomas.com/2026/06/llm-from-scratch-34a-building-a-jax-training-loop-for-an-llm-training-run):
  the training harness -- gradient accumulation via `optax.MultiSteps`, warmup +
  cosine-decay learning rate scheduling, gradient clipping by global norm,
  non-finite gradient handling, and checkpoint save/restore -- all tested against
  a trivial "A-to-A" model (embeddings straight into an output head, trained to
  map its input to itself).
* [Part 34b -- from bigrams to GPT-2, one component at a time](https://www.gilesthomas.com/2026/07/llm-from-scratch-34b-building-and-training-gpt-2-small-in-jax):
  converting that A-to-A model into a next-token predictor (essentially a
  bigram model), then adding LayerNorm, single-head attention, shortcut
  connections, position embeddings, multi-head attention, the FFN, multiple
  layers, pre-norm, and dropout -- with a training run at each step to show how
  each component changed the loss.

## What's in the repo

* `gpt.py` -- the model: `LayerNorm`, `MultiHeadAttention`, `TransformersLayer`
  and `GPTModel`, all as Flax NNX modules. Hand-rolled attention and LayerNorm
  rather than the built-in versions, because that was rather the point.
* `train.py` -- the training script. Downloads a pre-tokenised dataset from
  Hugging Face, loads it into (committed!) CPU RAM, and runs a JITted training
  loop with gradient accumulation, LR scheduling, clipping, checkpointing, and
  xkcd-style loss/learning-rate charts.
* `checkpointing.py` -- saves model weights as Safetensors and optimiser state
  via Orbax, with `best` and `latest` symlinks plus per-checkpoint metadata for
  the charts.
* `test_generation.py` -- smoke test: greedy-decodes a continuation of "Every
  effort moves you" from a saved model.
* `test_a_to_a.py` -- smoke test for the A-to-A stage: feeds in the first verse
  of *The Rime of the Ancient Mariner* and checks what comes back out.
* `convert_model_to_pytorch.py` -- converts a JAX Safetensors checkpoint into
  the key/shape conventions used by my earlier
  [PyTorch model](https://github.com/gpjt/ddp-base-model-from-scratch), so both
  can be run through the same evals.
* `check_jax.py` -- prints the JAX version, backend and devices, to confirm the
  install is sane.
* `array_speed_test.py` -- a little benchmark from when I was debugging why
  fetching batches from a CPU-resident array was slow
  ([committed vs uncommitted arrays](https://www.gilesthomas.com/2026/06/jax-commitment-issues)).
* `runs/` -- one directory per training run, each with a `model.json` (architecture)
  and `train.json` (hyperparameters). These trace the incremental build-up from
  the two posts: `a-to-a`, `a-to-a-next-token`, `layer-norm`,
  `sha-only-transformers-layer`, `mha-single-layer`,
  `multiple-full-layers-with-pre-norm`, and so on up to
  `full-llm-full-train-with-mha-output-bias`, the final 3.26B-token GPT-2 small
  run. Checkpoints themselves aren't in the repo (they're ~650MB each).

## Installation

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.13. First,
install the basic dependencies:

```bash
uv sync
```

If you have an NVIDIA GPU, add CUDA support:

```bash
uv sync --extra cuda13
```

Then check that JAX can see your hardware:

```bash
uv run check_jax.py
```

## Training

`train.py` takes the name of a run directory (under `runs/`) and a path to a
directory to store datasets in:

```bash
uv run train.py full-llm-full-train ~/datasets
```

The dataset named in the run's `train.json` -- for the runs in this repo,
[`gpjt/fineweb-gpt2-tokens`](https://huggingface.co/datasets/gpjt/fineweb-gpt2-tokens),
a pre-tokenised slice of FineWeb -- is downloaded from Hugging Face on first use.

To resume an interrupted run from a checkpoint, pass the checkpoint directory
name as a third argument (`latest` is usually what you want):

```bash
uv run train.py full-llm-full-train ~/datasets latest
```

One tip from bitter experience: JAX pre-allocates 75% of VRAM by default and
never asks for more. For the full model on a 24GiB card you'll want to raise
that, e.g.:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run train.py full-llm-full-train ~/datasets
```

Each run writes checkpoints to `runs/<name>/checkpoints/`, along with
`loss-chart.png` and `learning-rate-chart.png` so you can keep an eye on how
things are going.

## Trying out a trained model

```bash
uv run test_generation.py runs/full-llm-full-train/checkpoints/best/model.safetensors
```

...will greedily generate 20 tokens following "Every effort moves you". After
the full training run, mine produced:

```
Every effort moves you closer to your goals, but if you are unsure of what it takes, you don't
```

Not bad for a consumer GPU and 37 hours!

## Licence

Apache 2.0 -- see [LICENSE](LICENSE).
