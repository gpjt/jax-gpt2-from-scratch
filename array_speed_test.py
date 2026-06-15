from datetime import datetime
import time

import click

import jax


@click.command()
@click.option("--commit", is_flag=True)
@click.option("--put_items_to_gpu", is_flag=True)
def main(commit, put_items_to_gpu):
    key = jax.random.key(42)

    cpu0 = jax.devices("cpu")[0]
    cuda0 = jax.devices("cuda")[0]
    with jax.default_device(cpu0):
        array = jax.random.randint(
            key,
            (530640, 6, 1024),
            0, 50_000,
            dtype=jax.numpy.uint16
        )
    if commit:
        array = jax.device_put(array, cpu0)

    print(f"{datetime.now()}: {array.shape=}")
    print(f"{datetime.now()}: {array.device=}")
    print(f"{datetime.now()}: {array.committed=}")

    for ii in range(10):
        start = time.time()
        item = array[ii]
        end = time.time()

        print(f"{datetime.now()}: Getting item {ii} took {end - start}s")
        print(f"{datetime.now()}: {item.shape=}")
        print(f"{datetime.now()}: {item.device=}")
        print(f"{datetime.now()}: {item.committed=}")

        if put_items_to_gpu:
            start = time.time()
            item = jax.device_put(item, cuda0)
            end = time.time()

            print(f"{datetime.now()}: Putting item {ii} to GPU took {end - start}s")
            print(f"{datetime.now()}: {item.shape=}")
            print(f"{datetime.now()}: {item.device=}")
            print(f"{datetime.now()}: {item.committed=}")


if __name__ == "__main__":
    main()
