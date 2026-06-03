import json
from pathlib import Path

import click

from huggingface_hub import snapshot_download

from flax import nnx
import optax

from checkpointing import save_checkpoint
from gpt import GPTModel


def download_dataset(dataset_dir, dataset_name):
    snapshot_download(
        f"{dataset_name}",
        repo_type="dataset",
        local_dir=dataset_dir,
        allow_patterns="*"
    )


def train(run_dir, model, optimizer):
    save_checkpoint(run_dir, "checkpoint-test", model)


@click.command()
@click.argument("run")
@click.argument("datasets_dir_path")
def main(run, datasets_dir_path):
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

    datasets_dir = Path(datasets_dir_path)
    dataset_name = train_conf["dataset"]
    dataset_dir = datasets_dir / dataset_name
    if not datasets_dir.exists():
        datasets_dir.mkdir()
    if not datasets_dir.is_dir():
        raise Exception(f"{datasets_dir_path} is not a directory")
    download_dataset(dataset_dir, dataset_name)

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

    optimizer = nnx.Optimizer(
        model,
        optax.adamw(
            learning_rate=train_conf["learning_rate"],
            weight_decay=train_conf["weight_decay"],
        ),
        wrt=nnx.Param
    )

    train(run_dir, model, optimizer)


if __name__ == "__main__":
    main()
