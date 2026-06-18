import datetime
import json
from pathlib import Path

from flax import nnx

from safetensors.flax import save_file as st_save_file


def get_checkpoints_dir(run_dir):
    return run_dir / "checkpoints"


def save_checkpoint(
    run_dir,
    name,
    model,
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

    st_save_file(simple_dict, checkpoint_dir / "model.safetensors")

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
