from functools import partial

import tiktoken

import jax
from flax import nnx
from jax import numpy as jnp

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


def main():
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

    seed_text = "Every effort moves you"
    seed_tokens = tokenizer.encode(seed_text)
    result = generate(model, seed_tokens, 20)

    print(tokenizer.decode(result))



if __name__ == "__main__":
    main()
