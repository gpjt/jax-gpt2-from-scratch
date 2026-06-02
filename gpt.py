from flax import nnx


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

        self.output_head = nnx.Linear(
            in_features=emb_dim,
            out_features=vocab_size,
            use_bias=False,
            rngs=rngs,
        )


    def __call__(self, xs):
        input_embeddings = self.token_embedding(xs)

        return self.output_head(input_embeddings)
