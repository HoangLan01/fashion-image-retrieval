from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class MultiHeadAdditiveAttention(nn.Module):
    def __init__(self, embedding_dim: int = 768, num_heads: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.hidden_projection = nn.Linear(embedding_dim, embedding_dim)
        self.context_weight = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.context_weight)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_attention: bool = False,
        context_intervention: str = "none",
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if context_intervention not in {"none", "shuffled", "uniform"}:
            raise ValueError(f"Unsupported context intervention: {context_intervention}")
        batch_size, num_tokens, _ = x.shape
        hidden = self.hidden_projection(x)
        hidden_heads = hidden.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (hidden_heads * self.context_weight.view(1, self.num_heads, 1, self.head_dim)).sum(-1)
        scores = scores / math.sqrt(self.head_dim)
        if attention_mask is not None:
            mask = attention_mask[:, None, :].to(dtype=torch.bool, device=scores.device)
            scores = scores.masked_fill(~mask, -torch.inf)

        if context_intervention == "uniform":
            if attention_mask is None:
                valid = torch.ones(
                    batch_size,
                    1,
                    num_tokens,
                    dtype=scores.dtype,
                    device=scores.device,
                )
            else:
                valid = attention_mask[:, None, :].to(dtype=scores.dtype, device=scores.device)
            weights = valid / valid.sum(dim=-1, keepdim=True).clamp_min(1.0)
            weights = weights.expand(-1, self.num_heads, -1)
        else:
            weights = torch.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        context = torch.sum(weights.unsqueeze(-1) * hidden_heads, dim=2)
        if context_intervention == "shuffled":
            if batch_size < 2:
                raise ValueError("Shuffled context requires an inference batch with at least 2 samples.")
            context = torch.roll(context, shifts=1, dims=0)
        modulated = context.unsqueeze(2) * hidden_heads
        modulated = modulated.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.embedding_dim)
        output = hidden + self.dropout(self.output_projection(modulated))
        if return_attention:
            return output, weights
        return output


class AdditiveAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 768,
        num_heads: int = 8,
        ffn_multiplier: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.attention = MultiHeadAdditiveAttention(embedding_dim, num_heads, dropout)
        self.linear = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embedding_dim)
        hidden_dim = embedding_dim * ffn_multiplier
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_attention: bool = False,
        context_intervention: str = "none",
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        attention_output = self.attention(
            self.norm1(x),
            attention_mask=attention_mask,
            return_attention=return_attention,
            context_intervention=context_intervention,
        )
        if return_attention:
            attention_output, weights = attention_output
        attended = self.linear(attention_output)
        x = x + self.dropout(attended)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        if return_attention:
            return x, weights
        return x


class AACLCompositionModule(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 768,
        num_blocks: int = 3,
        num_heads: int = 8,
        ffn_multiplier: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                AdditiveAttentionBlock(embedding_dim, num_heads, ffn_multiplier, dropout)
                for _ in range(num_blocks)
            ]
        )
        self.output_norm = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_attention_mask: torch.Tensor | None = None,
        return_attention: bool = False,
        context_intervention: str = "none",
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch_size, image_token_count, _ = image_tokens.shape
        tokens = torch.cat([image_tokens, text_tokens], dim=1)

        attention_mask = None
        if text_attention_mask is not None:
            image_mask = torch.ones(
                batch_size,
                image_token_count,
                dtype=text_attention_mask.dtype,
                device=text_attention_mask.device,
            )
            attention_mask = torch.cat([image_mask, text_attention_mask], dim=1)

        attention_weights = []
        for block in self.blocks:
            if return_attention:
                tokens, block_weights = block(
                    tokens,
                    attention_mask=attention_mask,
                    return_attention=True,
                    context_intervention=context_intervention,
                )
                attention_weights.append(block_weights)
            else:
                tokens = block(
                    tokens,
                    attention_mask=attention_mask,
                    context_intervention=context_intervention,
                )

        image_output = self.output_norm(tokens[:, :image_token_count])
        embedding = F.normalize(image_output.mean(dim=1), dim=-1)
        if return_attention:
            # [batch, block, head, image+text token]
            return embedding, torch.stack(attention_weights, dim=1)
        return embedding
