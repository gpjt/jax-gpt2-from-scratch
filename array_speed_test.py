from datetime import datetime
import time

import click

import jax


@click.command()
@click.option("--commit", is_flag=True)
def main(commit):
    key = jax.random.key(42)

    cpu0 = jax.devices("cpu")[0]
    with jax.default_device(cpu0):
        base_array = jax.random.randint(
            key,
            (3_260_252_160,),
            0, 50_000,
            dtype=jax.numpy.uint16
        )
    if commit:
        base_array = jax.device_put(base_array, cpu0)

    print(f"{datetime.now()}: {base_array.shape=}")
    print(f"{datetime.now()}: {base_array.device=}")
    print(f"{datetime.now()}: {base_array.committed=}")

    reshaped = base_array.reshape(-1, 6, 1024)
    print(f"{datetime.now()}: {reshaped.shape=}")
    print(f"{datetime.now()}: {reshaped.device=}")
    print(f"{datetime.now()}: {reshaped.committed=}")

    start = time.time()
    item = reshaped[0]
    end = time.time()

    print(f"{datetime.now()}: Getting item took {end - start}s")

    print(f"{datetime.now()}: {item.shape=}")
    print(f"{datetime.now()}: {item.device=}")
    print(f"{datetime.now()}: {item.committed=}")


if __name__ == "__main__":
    main()
