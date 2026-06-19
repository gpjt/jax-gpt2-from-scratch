import datetime
import json
from pathlib import Path

from flax import nnx

from orbax.checkpoint import v1 as ocp
from safetensors.flax import load_file, save_file


def get_checkpoints_dir(run_dir):
    return run_dir / "checkpoints"


def save_checkpoint(
    run_dir,
    name,
    model, optimizer,
    learning_rate,
    min_train_loss, max_train_loss, avg_train_loss,
    global_step,
    is_best,
):
    checkpoints_dir = get_checkpoints_dir(run_dir)

    if not checkpoints_dir.exists():
        checkpoints_dir.mkdir()

    now = datetime.datetime.now(datetime.UTC)
    checkpoint_name = f"{now:%Y%m%dZ%H%M%S}-{name}"
    checkpoint_dir = checkpoints_dir / checkpoint_name
    checkpoint_dir.mkdir()

    model_state = nnx.state(model)
    flat_state = nnx.to_flat_state(model_state)
    simple_dict = {}
    for tuple_key, array in flat_state:
        key = ".".join(str(key) for key in tuple_key)
        simple_dict[key] = array

    save_file(simple_dict, checkpoint_dir / "model.safetensors")

    ocp.save_pytree(checkpoint_dir / "optimizer", optimizer.opt_state)

    with open(checkpoint_dir / "meta.json", "w") as f:
        json.dump(
            dict(
                learning_rate=learning_rate,
                min_train_loss=min_train_loss,
                max_train_loss=max_train_loss,
                avg_train_loss=avg_train_loss,
                global_step=global_step,
                is_best=is_best,
            ),
            f,
            indent=2,
        )

    symlink_target = Path(".") / checkpoint_dir.name
    if is_best:
        best_path = checkpoints_dir / "best"
        best_path.unlink(missing_ok=True)
        best_path.symlink_to(symlink_target, target_is_directory=True)

    latest_path = checkpoints_dir / "latest"
    latest_path.unlink(missing_ok=True)
    latest_path.symlink_to(symlink_target, target_is_directory=True)


def load_model(model, file):
    model_state_simple_dict = load_file(file)
    dict_flat_state = {}
    for key, array in model_state_simple_dict.items():
        elements = key.split(".")
        list_key = []
        for element in elements:
            try:
                list_key.append(int(element))
            except ValueError:
                list_key.append(element)
        dict_flat_state[tuple(list_key)] = array

    new_flat_state = nnx.from_flat_state(dict_flat_state)
    nnx.update(model, new_flat_state)


def load_checkpoint(run_dir, checkpoint, model, optimizer=None):
    checkpoints_dir = get_checkpoints_dir(run_dir)
    checkpoint_dir = checkpoints_dir / checkpoint

    load_model(model, checkpoint_dir / "model.safetensors")

    if optimizer is not None:
        optimizer.opt_state = ocp.load_pytree(checkpoint_dir / "optimizer", optimizer.opt_state)

    with open(checkpoint_dir / "meta.json", "r") as f:
        meta = json.load(f)
        restart_global_step = meta["global_step"] + 1

    with open(checkpoints_dir / "best" / "meta.json") as f:
        best_loss = json.load(f)["avg_train_loss"]

    return restart_global_step, best_loss

