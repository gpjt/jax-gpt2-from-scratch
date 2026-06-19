import json
import time
from datetime import datetime
from pathlib import Path

import click
from tqdm import tqdm

from huggingface_hub import snapshot_download

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator

import jax
import optax
from flax import nnx
from safetensors.flax import load_file

from checkpointing import get_checkpoints_dir, save_checkpoint
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
    print("")


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


def get_training_data(run_dir):
    checkpoints_dir = get_checkpoints_dir(run_dir)

    def sanitize(val, cap=1_000_000):
        if val == float("inf"):
            return cap
        if val == float("-inf"):
            return -cap
        return val

    learning_rates = []
    min_train_losses = []
    max_train_losses = []
    avg_train_losses = []
    max_grad_norms = []
    avg_grad_norms = []
    frac_clipped = []
    best_global_step = None
    for item in checkpoints_dir.iterdir():
        if item.name == "latest":
            continue

        meta = json.loads((item / "meta.json").read_text())
        if item.name == "best":
            best_global_step = meta["global_step"]
            continue

        if meta.get("learning_rate") is not None:
            learning_rates.append((meta["global_step"], meta["learning_rate"]))
        min_train_losses.append((meta["global_step"], meta["min_train_loss"]))
        max_train_losses.append((meta["global_step"], meta["max_train_loss"]))
        avg_train_losses.append((meta["global_step"], meta["avg_train_loss"]))

    learning_rates.sort(key=lambda x: x[0])
    min_train_losses.sort(key=lambda x: x[0])
    max_train_losses.sort(key=lambda x: x[0])
    avg_train_losses.sort(key=lambda x: x[0])
    max_grad_norms.sort(key=lambda x: x[0])
    avg_grad_norms.sort(key=lambda x: x[0])
    frac_clipped.sort(key=lambda x: x[0])

    return (
        learning_rates,
        min_train_losses, max_train_losses, avg_train_losses,
        best_global_step
    )


def generate_training_charts(run_dir, clipping_max_norm=None):
    (
        learning_rates,
        min_train_points, max_train_points, avg_train_points,
        best_global_step
    ) = get_training_data(run_dir)

    plt.xkcd()

    font_family = None
    for f in font_manager.fontManager.ttflist:
        if "xkcd" in f.name.lower():
            font_family = f.name
            break
    if font_family is not None:
        plt.rcParams['font.family'] = font_family

    # --- Chart 1: Loss ---

    fig, ax_loss = plt.subplots(figsize=(8, 6), dpi=100)

    train_steps, min_train_losses = zip(*min_train_points)
    _, max_train_losses = zip(*max_train_points)
    _, avg_train_losses = zip(*avg_train_points)

    ax_loss.fill_between(
        train_steps,
        min_train_losses,
        max_train_losses,
        color="lightblue",
        alpha=0.25,
        label="MIN–MAX LOSS",
    )
    ax_loss.plot(
        train_steps,
        avg_train_losses,
        color="blue",
        label="AVG LOSS",
        marker="o",
        linestyle="-",
    )

    ax_loss.set_title("TRAINING RUN: LOSS")
    ax_loss.set_xlabel("GLOBAL STEP")
    ax_loss.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_loss.set_ylabel("LOSS (LOG)")
    ax_loss.set_yscale("log")

    if best_global_step is not None:
        ax_loss.axvline(
            best_global_step,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label="BEST STEP",
        )

    ax_loss.legend(
        loc="upper right",
        handlelength=2.0,
        handletextpad=0.6,
    )

    fig.tight_layout(rect=(0, 0.12, 1, 1))
    image_file = run_dir / "loss-chart.png"
    fig.savefig(image_file, bbox_inches="tight")
    plt.close(fig)

    # --- Chart 2: Learning Rate ---

    if learning_rates:
        fig_lr, ax_lr = plt.subplots(figsize=(8, 6), dpi=100)

        lr_steps, lr_values = zip(*learning_rates)

        ax_lr.plot(
            lr_steps,
            lr_values,
            color="purple",
            marker="o",
            linestyle="-",
        )

        ax_lr.set_title("TRAINING RUN: LEARNING RATE")
        ax_lr.set_xlabel("GLOBAL STEP")
        ax_lr.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax_lr.set_ylabel("LEARNING RATE")

        fig_lr.tight_layout(rect=(0, 0.15, 1, 1))
        lr_image_file = run_dir / "learning-rate-chart.png"
        fig_lr.savefig(lr_image_file, bbox_inches="tight")
        plt.close(fig_lr)


def calculate_loss(model, inputs, targets):
    logits = model(inputs)
    loss = optax.losses.softmax_cross_entropy_with_integer_labels(
        logits, targets
    ).mean()
    return loss


@nnx.jit
def train_step(model, optimizer, inputs, targets):
    loss, grads = nnx.value_and_grad(calculate_loss)(model, inputs, targets)
    optimizer.update(model, grads)
    return loss


def train(
    run_dir,
    model, optimizer,
    get_learning_rate,
    train_dataset,
    rank, world_size,
    gradient_accumulation_steps,
    start_global_step,
    checkpoint_interval,
):
    model_device = jax.devices()[0]

    total_global_steps = (len(train_dataset) // world_size) // gradient_accumulation_steps

    progress_bar = tqdm(
        range(start_global_step, total_global_steps),
        disable=(rank != 0)
    )

    best_loss = None
    train_losses = []
    tokens_seen_this_rank = 0
    start_time = time.time()

    for global_step in progress_bar:
        for accumulation_step in range(gradient_accumulation_steps):
            inputs, targets = train_dataset[((global_step * gradient_accumulation_steps) + accumulation_step) * world_size + rank]
            inputs = jax.device_put(inputs, model_device)
            targets = jax.device_put(targets, model_device)

            train_loss = train_step(model, optimizer, inputs, targets)
            train_losses.append(train_loss.item())

            microbatch_size, sequence_length = inputs.shape
            tokens_seen_this_rank += microbatch_size * sequence_length

        is_checkpoint_iter = (
            (global_step % checkpoint_interval == 0)
            or (global_step == total_global_steps - 1)
        )
        if is_checkpoint_iter:
            if rank == 0:
                log("Saving checkpoint")
                min_train_loss = min(train_losses)
                max_train_loss = max(train_losses)
                avg_train_loss = sum(train_losses) / len(train_losses)
                train_losses = []

                if best_loss is None or avg_train_loss < best_loss:
                    is_best = True
                    best_loss = avg_train_loss
                else:
                    is_best = False

                current_learning_rate = get_learning_rate()
                save_checkpoint(
                    run_dir,
                    f"iteration-{global_step}",
                    model,
                    current_learning_rate,
                    min_train_loss, max_train_loss, avg_train_loss,
                    global_step,
                    is_best,
                )
                generate_training_charts(run_dir)

        if rank == 0:
            elapsed_time = time.time() - start_time
            tokens_per_sec = (tokens_seen_this_rank * world_size) / elapsed_time
            progress_bar.set_postfix(
                loss=f"{train_loss.item():.3f}",
                tps=f"{tokens_per_sec:,.0f}"
            )

    end_time = time.time()
    elapsed_time = end_time - start_time

    if rank == 0:
        log(f"\n\n\nTraining complete in {elapsed_time:,.3f} seconds")
        total_tokens_seen = tokens_seen_this_rank * world_size
        log(f"Tokens seen: {total_tokens_seen:,.0f}")
        log(f"Throughput: {total_tokens_seen / elapsed_time:,.0f} tokens/second")
        log(f"Final train loss: {train_loss.item():.3f}")



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

    gradient_accumulation_steps = train_conf.get("gradient_accumulation_steps")

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
    rank = 0  ## DDP
    train_dataset = load_dataset(
        dataset_dir, "train",
        train_conf["min_train_tokens"], train_conf["start_train_token"],
        world_size, train_conf["microbatch_size"],
        gradient_accumulation_steps,
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
    total_steps = (len(train_dataset) // world_size) // gradient_accumulation_steps
    warmup_steps = (total_steps * train_conf["warmup_period_percent"]) // 100
    learning_rate = train_conf["learning_rate"]
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=learning_rate * 0.00001,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,  # !!
        end_value=learning_rate / 10,
    )

    optax_optimizer = optax.chain(
        optax.clip_by_global_norm(train_conf["clipping_max_norm"]),
        optax.inject_hyperparams(optax.adamw)(
            learning_rate=schedule,
            weight_decay=train_conf["weight_decay"],
        )
    )
    optimizer = nnx.Optimizer(
        model,
        optax.MultiSteps(
            optax_optimizer,
            every_k_schedule=gradient_accumulation_steps
        ),
        wrt=nnx.Param
    )

    def get_learning_rate():
        return (
            optimizer
            .opt_state
            .inner_opt_state[1]
            .hyperparams["learning_rate"]
            .get_value()
            .item()
        )

    log("Start train")
    start_global_step = 0  ## checkpointing
    train(
        run_dir,
        model, optimizer,
        get_learning_rate,
        train_dataset,
        rank, world_size,
        gradient_accumulation_steps,
        start_global_step,
        train_conf["checkpoint_interval"],
    )
    log("Done")


if __name__ == "__main__":
    main()
