import datetime

from flax import nnx

from safetensors.flax import save_file as st_save_file


def get_checkpoints_dir(run_dir):
    return run_dir / "checkpoints"


def save_checkpoint(
    run_dir,
    name,
    model,
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
        key = ".".join(list(tuple_key))
        simple_dict[key] = array

    st_save_file(simple_dict, checkpoint_dir / "model.safetensors")
