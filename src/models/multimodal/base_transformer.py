import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.hidden_dim = config.hidden_size

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.scale = math.sqrt(self.head_dim)

        # Attention cache for memory efficiency
        self.cache_enabled = hasattr(config, 'use_cache') and config.use_cache
        self.key_cache = {}
        self.value_cache = {}

    def forward(self, x, attention_mask=None, layer_id=None):
        batch_size = x.shape[0]

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        if self.cache_enabled and layer_id is not None:
            cache_key = f"layer_{layer_id}"
            if cache_key in self.key_cache:
                k = torch.cat([self.key_cache[cache_key], k], dim=2)
                v = torch.cat([self.value_cache[cache_key], v], dim=2)
            self.key_cache[cache_key] = k.detach()
            self.value_cache[cache_key] = v.detach()

        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask == 0, float('-inf'))

        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context = torch.matmul(attention_probs, v)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, self.hidden_dim)

        return self.o_proj(context)

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),  # Standard MLP dimension is 4x hidden size
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size * 4, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob)
        )
        self.attention_norm = nn.LayerNorm(config.hidden_size)
        self.feed_forward_norm = nn.LayerNorm(config.hidden_size)
        self.gradient_checkpointing = False

    def forward(self, x, attention_mask=None, layer_id=None, use_cache=True):
        # Make forward compatible with gradient checkpointing
        def attention_module(x):
            return self.attention(self.attention_norm(x), attention_mask, layer_id if use_cache else None)

        def feed_forward_module(x):
            return self.feed_forward(self.feed_forward_norm(x))

        if self.gradient_checkpointing and self.training:
            attention_output = torch.utils.checkpoint.checkpoint(attention_module, x)
            x = x + attention_output
            feed_forward_output = torch.utils.checkpoint.checkpoint(feed_forward_module, x)
        else:
            attention_output = attention_module(x)
            x = x + attention_output
            feed_forward_output = feed_forward_module(x)

        return x + feed_forward_output

class BaseTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_hidden_layers)
        ])

        self.final_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.gradient_checkpointing = False

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        batch_size, seq_length = input_ids.shape

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        embeddings = self.embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)

        x = embeddings + position_embeddings
        x = self.dropout(x)

        for i, layer in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer,
                    x,
                    attention_mask,
                    i
                )
            else:
                x = layer(x, attention_mask, layer_id=i)

        return self.final_norm(x)

    def clear_cache(self):
        for layer in self.layers:
            if hasattr(layer.attention, 'key_cache'):
                layer.attention.key_cache.clear()
                layer.attention.value_cache.clear()
