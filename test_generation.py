from functools import partial

import click
import tiktoken

import jax
from flax import nnx
from jax import numpy as jnp

from checkpointing import load_model
from gpt import GPTModel


def generate_token(model, context):
    logits = model(context)
    last_token_logits = logits[:, -1, :]
    predicted = jnp.argmax(last_token_logits, axis=-1)
    expanded_predicted = jnp.expand_dims(predicted, axis=0)
    context = jnp.append(context, expanded_predicted, axis=1)
    return context


@partial(jax.jit, static_argnums=(2,))
def generate(model, seed_tokens, tokens_to_generate):
    context = jnp.array([seed_tokens])
    for ii in range(tokens_to_generate):
        context = generate_token(model, context)

    return jnp.asarray(context.squeeze())


@click.command()
@click.argument("model_safetensors_file")
def main(model_safetensors_file):
    tokenizer = tiktoken.get_encoding("gpt2")

    rngs = nnx.Rngs(43)
    model = GPTModel(
        vocab_size=tokenizer.n_vocab,
        context_length=1024,
        d_emb=768,
        n_heads=12, d_qk=64, d_v=64,
        n_layers=12,
        qkv_bias=False,
        rngs=rngs,
        drop_rate=None,
    )
    load_model(model, model_safetensors_file)

    seed_text = "Every effort moves you"
    seed_tokens = tokenizer.encode(seed_text)
    result = generate(model, seed_tokens, 20)

    print(tokenizer.decode(result))



if __name__ == "__main__":
    main()
