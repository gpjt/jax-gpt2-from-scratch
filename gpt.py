import jax.numpy as jnp
from flax import nnx


class LayerNorm(nnx.Module):

    def __init__(self, emb_dim):
        self.scale = nnx.Param(jnp.ones((1, 1, emb_dim)))
        self.bias = nnx.Param(jnp.zeros((1, 1, emb_dim)))

    def __call__(self, xs):
        means = xs.mean(axis=-1, keepdims=True)
        stds = xs.std(axis=-1, keepdims=True)

        normalized = (xs - means) / (stds + 1e-5)

        scaled_and_biased = (normalized * self.scale) + self.bias
        return scaled_and_biased


class GPTModel(nnx.Module):

    def __init__(
        self,
        vocab_size, context_length,
        emb_dim,
        n_heads, n_layers,
        qkv_bias,
        drop_rate,
        rngs,
    ):
        self.token_embedding = nnx.Embed(
            num_embeddings=vocab_size,
            features=emb_dim,
            rngs=rngs,
        )

        self.output_norm = LayerNorm(emb_dim)

        self.output_head = nnx.Linear(
            in_features=emb_dim,
            out_features=vocab_size,
            use_bias=False,
            rngs=rngs,
        )


    def __call__(self, xs):
        input_embeddings = self.token_embedding(xs)

        normalised = self.output_norm(input_embeddings)

        return self.output_head(normalised)
