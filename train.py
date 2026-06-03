from pathlib import Path

from flax import nnx

from checkpointing import save_checkpoint
from gpt import GPTModel


def main():
    rngs = nnx.Rngs(42)
    model = GPTModel(
        vocab_size=1024,
        context_length=1024,
        emb_dim=768,
        n_heads=12, n_layers=12,
        qkv_bias=False,
        drop_rate=0.1,
        rngs=rngs,
    )
    save_checkpoint(Path("/tmp"), "checkpoint-test", model)


if __name__ == "__main__":
    main()
