import jax
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



class Attention(nnx.Module):

    def __init__(self, emb_dim, qkv_bias, rngs):
        self.d_qk = emb_dim

        self.W_q = nnx.Linear(emb_dim, self.d_qk, use_bias=qkv_bias, rngs=rngs)
        self.W_k = nnx.Linear(emb_dim, self.d_qk, use_bias=qkv_bias, rngs=rngs)
        self.W_v = nnx.Linear(emb_dim, self.d_qk, use_bias=qkv_bias, rngs=rngs)


    def __call__(self, xs):
        Q = self.W_q(xs)
        K = self.W_k(xs)
        V = self.W_v(xs)

        omega = Q @ jnp.transpose(K, axes=(0, 2, 1))

        omega /= jnp.sqrt(self.d_qk)

        causal_mask = jnp.ones_like(omega, dtype=bool)
        causal_mask = jnp.tril(causal_mask)

        causal_omega = jnp.where(causal_mask, omega, -jnp.inf)

        attention_weights = jax.nn.softmax(causal_omega, axis=-1)

        return attention_weights @ V



class TransformersLayer(nnx.Module):

    def __init__(self, emb_dim, qkv_bias, rngs):
        self.pre_attention_norm = LayerNorm(emb_dim)
        self.attention = Attention(emb_dim, qkv_bias, rngs)


    def __call__(self, xs):
        shortcut = xs
        pre_attention_normed = self.pre_attention_norm(xs)
        att = self.attention(pre_attention_normed)
        return shortcut + att



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

        self.transformers_layer = TransformersLayer(emb_dim, qkv_bias, rngs)

        self.output_norm = LayerNorm(emb_dim)

        self.output_head = nnx.Linear(
            in_features=emb_dim,
            out_features=vocab_size,
            use_bias=False,
            rngs=rngs,
        )


    def __call__(self, xs):
        input_embeddings = self.token_embedding(xs)

        transformed = self.transformers_layer(input_embeddings)

        normalized = self.output_norm(transformed)

        return self.output_head(normalized)
