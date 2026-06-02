# JAX GPT-2 from scratch

An implemention of the GPT-2 architecture using JAX.


## Installation

First install the basic dependencies:

```bash
uv sync
```

Next, if you have CUDA, add it like this:

```bash
uv sync --extra cuda13
```

Next, check it's all OK:

```bash
uv run check_jax.py
```
