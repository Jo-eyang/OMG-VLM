# Copyright (c) Alibaba Cloud.
# OMG-VLM target-aware text aggregation modules.

from typing import List, Optional, Sequence, Tuple, Union

import torch
from torch import nn


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(tensor: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (tensor * cos) + (_rotate_half(tensor) * sin)


class SimpleRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().unsqueeze(0).unsqueeze(1)  # [1,1,seq,dim]
        sin = emb.sin().unsqueeze(0).unsqueeze(1)
        cos = cos.to(dtype=dtype)
        sin = sin.to(dtype=dtype)
        return cos, sin


class RotarySelfAttentionBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_size),
            nn.Dropout(dropout),
        )

        self.rotary_emb = SimpleRotaryEmbedding(self.head_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        x = self.norm1(hidden_states)
        batch, seq_len, _ = x.size()

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        cos, sin = self.rotary_emb(seq_len, device=x.device, dtype=x.dtype)
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, -1)
        attn_output = self.proj(attn_output)
        attn_output = self.dropout(attn_output)

        x = residual + attn_output
        x = x + self.mlp(self.norm2(x))
        return x


class TargetAwareTextualAggregation(nn.Module):
    """Compress target text into a query vector and retrieve neighbor context."""

    def __init__(
        self,
        embedding_layer: Optional[nn.Embedding],
        hidden_size: int = 4096,
        num_attention_heads: int = 16,
        pool_num_layers: int = 1,
        pool_mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        query_token_count: int = 16,
    ):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.hidden_size = hidden_size
        self.query_token_count = query_token_count

        assert hidden_size % num_attention_heads == 0, "hidden_size must be divisible by num_attention_heads"

        self.pool_tokens = nn.Parameter(torch.randn(query_token_count, hidden_size))
        self.pool_layers = nn.ModuleList(
            [
                RotarySelfAttentionBlock(
                    hidden_size=hidden_size,
                    num_heads=num_attention_heads,
                    mlp_ratio=pool_mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(max(pool_num_layers, 0))
            ]
        )
        self.pool_norm = nn.LayerNorm(hidden_size)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(hidden_size)
        self.cross_dropout = nn.Dropout(dropout)
        self.neighbor_separator = nn.Parameter(torch.randn(hidden_size))

    def set_embedding_layer(self, embedding_layer: nn.Embedding):
        self.embedding_layer = embedding_layer

    def _ensure_tensor(self, token_ids: Union[torch.Tensor, List[int]]) -> torch.Tensor:
        if isinstance(token_ids, torch.Tensor):
            tensor = token_ids
        else:
            device = self.embedding_layer.weight.device
            tensor = torch.tensor(token_ids, device=device, dtype=torch.long)
        return tensor

    def _embed_sequence(self, token_ids: Union[torch.Tensor, List[int]]) -> torch.Tensor:
        ids = self._ensure_tensor(token_ids)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        return self.embedding_layer(ids)

    def _prepare_neighbors(
        self,
        neighbor_ids: Optional[Sequence[Union[torch.Tensor, List[int]]]],
        fallback: torch.Tensor,
        batch_size: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if neighbor_ids is None or len(neighbor_ids) == 0:
            return fallback

        neighbor_embeddings = []
        total_neighbors = len(neighbor_ids)
        for idx, seq in enumerate(neighbor_ids):
            if seq is None:
                continue
            emb = self._embed_sequence(seq).to(dtype)
            neighbor_embeddings.append(emb)
            if idx < total_neighbors - 1:
                sep = self.neighbor_separator.unsqueeze(0).unsqueeze(1)
                sep = sep.to(dtype=dtype, device=emb.device)
                sep = sep.expand(batch_size, 1, -1)
                neighbor_embeddings.append(sep)

        if not neighbor_embeddings:
            return fallback

        return torch.cat(neighbor_embeddings, dim=1)

    def _run_pool_encoder(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = hidden_states
        if not self.pool_layers:
            return output
        for layer in self.pool_layers:
            output = layer(output)
        return output

    def forward(
        self,
        target_ids: Union[torch.Tensor, List[int]],
        neighbor_ids: Optional[Sequence[Union[torch.Tensor, List[int]]]] = None,
    ) -> torch.Tensor:
        if self.embedding_layer is None:
            raise ValueError("TargetAwareTextualAggregation requires a valid embedding layer reference")

        target_embeddings = self._embed_sequence(target_ids)
        batch_size = target_embeddings.size(0)

        pool_tokens = self.pool_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        pool_tokens = pool_tokens.to(target_embeddings.dtype).to(target_embeddings.device)
        pooled_input = torch.cat([target_embeddings, pool_tokens], dim=1)
        pooled_output = self._run_pool_encoder(pooled_input)
        target_query = self.pool_norm(pooled_output[:, -self.query_token_count :, :])

        neighbor_embeddings = self._prepare_neighbors(
            neighbor_ids,
            fallback=target_embeddings,
            batch_size=batch_size,
            dtype=target_embeddings.dtype,
        )
        neighbor_embeddings = neighbor_embeddings.to(target_embeddings.dtype)

        attention_output, _ = self.cross_attention(
            query=target_query,  # [B, Q, H]
            key=neighbor_embeddings,
            value=neighbor_embeddings,
        )
        attention_output = self.cross_dropout(attention_output)
        fused_context = self.cross_norm(attention_output)

        return fused_context


__all__ = ["TargetAwareTextualAggregation"]
