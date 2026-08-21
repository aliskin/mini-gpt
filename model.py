import jax
import jax.numpy as jnp
import flax.nnx as nnx


class TransformerBlock(nnx.Module):

    def __init__(self, embed_dim, num_heads, ff_dim, *, rngs):
        self.norm1 = nnx.LayerNorm(embed_dim, rngs=rngs)

        self.attention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=embed_dim,
            qkv_features=embed_dim,
            out_features=embed_dim,
            decode=False,
            rngs=rngs
        )

        self.norm2 = nnx.LayerNorm(embed_dim, rngs=rngs)

        self.ff1 = nnx.Linear(embed_dim, ff_dim, rngs=rngs)
        self.ff2 = nnx.Linear(ff_dim, embed_dim, rngs=rngs)

    def __call__(self, x, mask=None):
        attn_out = self.attention(self.norm1(x), mask=mask)
        x = x + attn_out

        # Feed-forward sub-layer
        ff_out = self.ff1(self.norm2(x))
        ff_out = jax.nn.gelu(ff_out)
        ff_out = self.ff2(ff_out)
        x = x + ff_out
        return x


class TokenAndPositionEmbedding(nnx.Module):
    def __init__(self, maxlen, vocab_size, embed_dim, *, rngs):
        self.token_emb = nnx.Embed(vocab_size, embed_dim, rngs=rngs)
        self.pos_emb = nnx.Embed(maxlen, embed_dim, rngs=rngs)

    def __call__(self, x):
        seq_len = x.shape[1]
        positions = jnp.arange(seq_len)[None, :]
        return self.token_emb(x) + self.pos_emb(positions)


class MiniGPT(nnx.Module):

    def __init__(self,
                 maxlen, vocab_size, embed_dim, num_heads,
                 feed_forward_dim, num_transformer_blocks, *, rngs=nnx.Rngs(0)):
        self.maxlen = maxlen

        self.embedding = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim, rngs=rngs)

        self.transformer_blocks = [
            TransformerBlock(embed_dim, num_heads, feed_forward_dim, rngs=rngs)
            for _ in range(num_transformer_blocks)
        ]

        self.output_layer = nnx.Linear(embed_dim, vocab_size, use_bias=False, rngs=rngs)

    def causal_attention_mask(self, seq_len):
        return jnp.tril(
            jnp.ones((seq_len, seq_len), dtype=jnp.bool_)
        )

    def __call__(self, token_ids):
        seq_len = token_ids.shape[1]
        mask = self.causal_attention_mask(seq_len)

        x = self.embedding(token_ids)

        for block in self.transformer_blocks:
            x = block(x, mask=mask)

        logits = self.output_layer(x)

        return logits