import jax
import jax.numpy as jnp
import flax.nnx as nnx


class TransformerBlock(nnx.Module):

    def __init__(self, embed_dim, num_heads, ff_dim, *, rngs,
                 use_pre_norm=True, use_ffn=True, attention_norm_name="norm1"):
        self.use_pre_norm = use_pre_norm
        self.use_ffn = use_ffn
        self.attention_norm_name = attention_norm_name
        if use_pre_norm:
            setattr(self, attention_norm_name, nnx.LayerNorm(embed_dim, rngs=rngs))

        self.attention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=embed_dim,
            qkv_features=embed_dim,
            out_features=embed_dim,
            decode=False,
            rngs=rngs
        )

        if use_ffn:
            if use_pre_norm:
                self.norm2 = nnx.LayerNorm(embed_dim, rngs=rngs)
            self.ff1 = nnx.Linear(embed_dim, ff_dim, rngs=rngs)
            self.ff2 = nnx.Linear(ff_dim, embed_dim, rngs=rngs)

    def __call__(self, x, mask=None):
        attn_input = getattr(self, self.attention_norm_name)(x) if self.use_pre_norm else x
        attn_out = self.attention(attn_input, mask=mask)
        x = x + attn_out

        # Feed-forward sub-layer
        if self.use_ffn:
            ff_input = self.norm2(x) if self.use_pre_norm else x
            ff_out = self.ff1(ff_input)
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
                 feed_forward_dim, num_transformer_blocks, *, rngs=None,
                 use_pre_norm=True, use_ffn=True, use_final_norm=False,
                 attention_norm_name="norm1"):
        if rngs is None:
            rngs = nnx.Rngs(0)
        self.maxlen = maxlen

        self.embedding = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim, rngs=rngs)

        self.transformer_blocks = [
            TransformerBlock(
                embed_dim, num_heads, feed_forward_dim, rngs=rngs,
                use_pre_norm=use_pre_norm, use_ffn=use_ffn,
                attention_norm_name=attention_norm_name,
            )
            for _ in range(num_transformer_blocks)
        ]

        self.output_layer = nnx.Linear(embed_dim, vocab_size, use_bias=False, rngs=rngs)

        self.use_final_norm = use_final_norm
        if use_final_norm:
            self.final_norm = nnx.LayerNorm(embed_dim, rngs=rngs)

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

        if self.use_final_norm:
            x = self.final_norm(x)
        logits = self.output_layer(x)

        return logits

ARCHITECTURE_DEFAULTS = {
    "use_pre_norm": True,
    "use_ffn": True,
    "use_final_norm": False,
    "attention_norm_name": "norm1",
}


def model_from_config(config):
    """Shared construction path for training and checkpoint evaluation."""
    architecture = {key: config.get(key, value) for key, value in ARCHITECTURE_DEFAULTS.items()}
    return MiniGPT(
        maxlen=config["maxlen"], vocab_size=config["vocab_size"],
        embed_dim=config["embed_dim"], num_heads=config["num_heads"],
        feed_forward_dim=config["ff_dim"], num_transformer_blocks=config["num_blocks"],
        rngs=nnx.Rngs(config.get("seed", 42)), **architecture,
    )
