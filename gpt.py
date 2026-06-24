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



class MultiHeadAttention(nnx.Module):

    def __init__(self, d_emb, n_heads, d_qk, d_v, qkv_bias, rngs):
        self.n_heads = n_heads
        self.d_qk = d_qk
        self.d_v = d_v

        self.W_q = nnx.Linear(d_emb, self.d_qk * n_heads, use_bias=qkv_bias, rngs=rngs)
        self.W_k = nnx.Linear(d_emb, self.d_qk * n_heads, use_bias=qkv_bias, rngs=rngs)
        self.W_v = nnx.Linear(d_emb, self.d_v * n_heads, use_bias=qkv_bias, rngs=rngs)

        self.output_projection = nnx.Linear(self.d_v * n_heads, d_emb, use_bias=False, rngs=rngs)


    def __call__(self, xs):
        batch_size, len_sequence, d_emb = xs.shape

        # For each of the below:
        # * The initial linear layer projects them to
        #   (batch_size, len_sequence, d_X * n_heads)
        #   where X is qk or v as appropriate.
        # * The reshape makes them (batch_size, len_sequence, n_heads, d_X)
        # * The transpose makes them (batch_size, n_heads, len_sequence, d_X)
        Q = jnp.transpose(
            self.W_q(xs).reshape(
                (batch_size, len_sequence, self.n_heads, self.d_qk)
            ),
            (0, 2, 1, 3)
        )
        K = jnp.transpose(
            self.W_k(xs).reshape(
                (batch_size, len_sequence, self.n_heads, self.d_qk)
            ),
            (0, 2, 1, 3)
        )
        V = jnp.transpose(
            self.W_v(xs).reshape(
                (batch_size, len_sequence, self.n_heads, self.d_v)
            ),
            (0, 2, 1, 3)
        )

        # Q and K are (batch_size, n_heads, len_sequence, d_qk) per above
        # We need K to be (batch_size, n_heads, d_qk, len_sequence)
        # and then we get omega (batch_size, n_heads, len_sequence, len_sequence)
        omega = Q @ jnp.transpose(K, axes=(0, 1, 3, 2))

        omega /= jnp.sqrt(self.d_qk)

        causal_mask = jnp.ones_like(omega, dtype=bool)
        # tril treats all but the last two axes as batches so we're OK here.
        causal_mask = jnp.tril(causal_mask)

        causal_omega = jnp.where(causal_mask, omega, -jnp.inf)

        # last axis is still OK.
        attention_weights = jax.nn.softmax(causal_omega, axis=-1)

        # attention_weights is (batch_size, n_heads, len_sequence, len_sequence)
        # V is (batch_size, n_heads, len_sequence, d_v)
        # So this will come out as (batch_size, n_heads, len_sequence, d_v)
        weighted = attention_weights @ V

        # Transpose to (batch_size, len_sequence, n_heads, d_v),
        # then reshape to (batch_size, len_sequence, n_heads * d_v)
        striped_output = jnp.transpose(
            weighted,
            (0, 2, 1, 3)
        ).reshape(
            batch_size, len_sequence, self.n_heads * self.d_v
        )

        # Final linear layer to combine
        return self.output_projection(striped_output)



class TransformersLayer(nnx.Module):

    def __init__(self, d_emb, n_heads, d_qk, d_v, qkv_bias, rngs):
        self.attention = MultiHeadAttention(d_emb, n_heads, d_qk, d_v, qkv_bias, rngs)
        self.ffn = nnx.Sequential(
            nnx.Linear(
                in_features=d_emb,
                out_features=d_emb * 4,
                use_bias=True,
                rngs=rngs
            ),
            jax.nn.gelu,
            nnx.Linear(
                in_features=d_emb * 4,
                out_features=d_emb,
                use_bias=True,
                rngs=rngs
            ),
        )


    def __call__(self, xs):
        shortcut = xs
        att = self.attention(xs)
        post_attention = shortcut + att
        fed_forward = self.ffn(post_attention)
        return fed_forward + post_attention



class GPTModel(nnx.Module):

    def __init__(
        self,
        vocab_size, context_length,
        d_emb,
        n_heads, d_qk, d_v,
        n_layers,
        qkv_bias,
        drop_rate,
        rngs,
    ):
        self.token_embedding = nnx.Embed(
            num_embeddings=vocab_size,
            features=d_emb,
            rngs=rngs,
        )
        self.position_embedding = nnx.Embed(
            num_embeddings=context_length,
            features=d_emb,
            rngs=rngs,
        )

        self.transformers_layers = nnx.Sequential(
            *(
                TransformersLayer(
                    d_emb, n_heads, d_qk, d_v, qkv_bias, rngs
                )
                for _ in range(n_layers)
            )
        )

        self.output_norm = LayerNorm(d_emb)

        self.output_head = nnx.Linear(
            in_features=d_emb,
            out_features=vocab_size,
            use_bias=False,
            rngs=rngs,
        )


    def __call__(self, xs):
        token_embeddings = self.token_embedding(xs)
        b, n = xs.shape
        position_embeddings = self.position_embedding(jnp.arange(n))
        input_embeddings = token_embeddings + position_embeddings

        transformed = self.transformers_layers(input_embeddings)

        normalized = self.output_norm(transformed)

        return self.output_head(normalized)
