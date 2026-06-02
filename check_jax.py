import jax

print("JAX version:", jax.__version__)
print("Backend:", jax.default_backend())
print("Devices:")
for device in jax.devices():
    print(" ", device)
