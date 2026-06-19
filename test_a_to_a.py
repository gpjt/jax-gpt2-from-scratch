from functools import partial
from textwrap import dedent

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
        emb_dim=768,
        n_heads=12, n_layers=12,
        qkv_bias=False,
        drop_rate=0.1,
        rngs=rngs,
    )
    load_model(model, model_safetensors_file)

    test_text = dedent("""
        It is an ancient Mariner,
        And he stoppeth one of three.
        'By thy long grey beard and glittering eye,
        Now wherefore stopp'st thou me?
    """)
    print(f"Input:\n---\n{test_text}\n---\n")

    test_tokens = tokenizer.encode(test_text)

    batched_tokens = jnp.array([test_tokens])

    logits = model(batched_tokens)

    unbatched_logits = jnp.squeeze(logits, axis=0)
    predicted = jnp.argmax(unbatched_logits, axis=-1)

    output = tokenizer.decode(predicted)
    print(f"Output:\n---\n{output}\n---\n")



if __name__ == "__main__":
    main()
