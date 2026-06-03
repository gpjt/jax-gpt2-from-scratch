import json
from pathlib import Path

import click

from flax import nnx

from checkpointing import save_checkpoint
from gpt import GPTModel


@click.command()
@click.argument("run")
def main(run):
    run_dir = Path(__file__).resolve().parent / "runs" / run
    if not run_dir.is_dir():
        raise Exception(f"Could not find run dir {run_dir}")

    model_conf_file = run_dir / "model.json"
    if not model_conf_file.is_file():
        raise Exception(f"Could not find model config in {model_conf_file}")
    with open(model_conf_file, "r") as f:
        model_conf = json.load(f)

    train_conf_file = run_dir / "train.json"
    if not train_conf_file.is_file():
        raise Exception(f"Could not find train config in {train_conf_file}")
    with open(train_conf_file, "r") as f:
        train_conf = json.load(f)

    rngs = nnx.Rngs(42)
    model = GPTModel(
        vocab_size=model_conf["vocab_size"],
        context_length=model_conf["context_length"],
        emb_dim=model_conf["emb_dim"],
        n_heads=model_conf["n_heads"],
        n_layers=model_conf["n_layers"],
        qkv_bias=model_conf["qkv_bias"],
        drop_rate=train_conf["drop_rate"],
        rngs=rngs,
    )
    save_checkpoint(run_dir, "checkpoint-test", model)


if __name__ == "__main__":
    main()
