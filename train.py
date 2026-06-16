import json
from datetime import datetime
from pathlib import Path

import click
from tqdm import tqdm

from huggingface_hub import snapshot_download

import jax
import optax
from flax import nnx
from safetensors.flax import load_file

from checkpointing import save_checkpoint
from gpt import GPTModel


def log(s):
    print(f"{datetime.now()} {s}")


def download_dataset(dataset_dir, dataset_name):
    snapshot_download(
        f"{dataset_name}",
        repo_type="dataset",
        local_dir=dataset_dir,
        allow_patterns="*"
    )
    # snapshot_download messes with the stdout a bit
    print("\n\n")


class BigTrainDataset:

    def __init__(self, all_tokens, seq_length, microbatch_size):
        self.xs = all_tokens[:-1].reshape(-1, microbatch_size, seq_length)
        self.ys = all_tokens[:-1].reshape(-1, microbatch_size, seq_length)

    def __getitem__(self, ix):
        return self.xs[ix], self.ys[ix]

    def __len__(self):
        return self.xs.shape[0]


def load_dataset(
    dataset_dir, split,
    min_tokens, start_token,
    world_size, microbatch_size,
    gradient_accumulation_steps,
    seq_length
):
    cpu0 = jax.devices("cpu")[0]
    with jax.default_device(cpu0):
        full_dataset = load_file(dataset_dir / f"{split}.safetensors")["tokens"]
    full_dataset = jax.device_put(full_dataset, cpu0)
    if start_token > len(full_dataset):
        raise Exception(f"start_token {start_token} is past the end of the dataset")

    one_full_batch_tokens = world_size * microbatch_size * gradient_accumulation_steps * seq_length

    if min_tokens == -1:
        available_tokens = len(full_dataset) - start_token
        available_batches = (available_tokens // one_full_batch_tokens)
        tokens_needed = available_batches * one_full_batch_tokens
    else:
        if min_tokens % one_full_batch_tokens == 0:
            tokens_needed = min_tokens
        else:
            batches_for_just_over_min = (min_tokens // one_full_batch_tokens) + 1
            tokens_needed = batches_for_just_over_min * one_full_batch_tokens

    # Note that we need one extra token for our Ys.
    tokens_needed += 1

    if len(full_dataset) < start_token + tokens_needed:
        raise Exception(f"Not enough tokens (wanted {start_token + tokens_needed}, got {len(full_dataset)})")

    return BigTrainDataset(
        full_dataset[start_token:start_token + tokens_needed].block_until_ready(),
        seq_length, microbatch_size,
    )



def train(run_dir, model, optimizer, train_dataset, rank, world_size, gradient_accumulation_steps, start_global_step):
    save_checkpoint(run_dir, "checkpoint-test", model)

    total_global_steps = (len(train_dataset) // world_size) // gradient_accumulation_steps

    progress_bar = tqdm(
        range(start_global_step, total_global_steps),
        disable=(rank != 0)
    )
    for global_step in progress_bar:
        for accumulation_step in range(gradient_accumulation_steps):
            inputs, targets = train_dataset[((global_step * gradient_accumulation_steps) + accumulation_step) * world_size + rank]




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

    log("Downloading dataset")
    datasets_dir = Path(datasets_dir_path)
    dataset_name = train_conf["dataset"]
    dataset_dir = datasets_dir / dataset_name
    if not datasets_dir.exists():
        datasets_dir.mkdir()
    if not datasets_dir.is_dir():
        raise Exception(f"{datasets_dir_path} is not a directory")
    download_dataset(dataset_dir, dataset_name)

    log("Loading dataset into RAM")
    world_size = 1  ## DDP
    train_dataset = load_dataset(
        dataset_dir, "train",
        train_conf["min_train_tokens"], train_conf["start_train_token"],
        world_size, train_conf["microbatch_size"],
        train_conf.get("gradient_accumulation_steps"),
        model_conf["context_length"]
    )

    log("Creating model")
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

    log("Creating optimizer")
    optimizer = nnx.Optimizer(
        model,
        optax.adamw(
            learning_rate=train_conf["learning_rate"],
            weight_decay=train_conf["weight_decay"],
        ),
        wrt=nnx.Param
    )

    log("Start train")
    start_global_step = 0  ## checkpointing
    rank = 0  ## DDP
    train(
        run_dir,
        model, optimizer,
        train_dataset,
        rank, world_size,
        train_conf["gradient_accumulation_steps"],
        start_global_step
    )
    log("Done")


if __name__ == "__main__":
    main()
