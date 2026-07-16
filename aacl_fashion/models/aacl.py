from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from aacl_fashion.models.composition_module import AACLCompositionModule
from aacl_fashion.models.image_encoder import build_image_encoder
from aacl_fashion.models.text_encoder import TextFeatures, build_text_encoder


@dataclass
class QueryAttentionOutput:
    embedding: torch.Tensor
    attention_weights: torch.Tensor
    image_token_count: int
    text_features: TextFeatures


class AACLModel(nn.Module):
    def __init__(
        self,
        image_encoder: nn.Module,
        text_encoder: nn.Module,
        composition: AACLCompositionModule,
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.composition = composition

    def encode_image_tokens(self, images: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(images)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        image_tokens = self.encode_image_tokens(images)
        return F.normalize(image_tokens.mean(dim=1), dim=-1)

    def encode_text(self, captions: list[str] | tuple[str, ...]) -> TextFeatures:
        return self.text_encoder(captions)

    def encode_query(self, query_images: torch.Tensor, captions: list[str] | tuple[str, ...]) -> torch.Tensor:
        image_tokens = self.encode_image_tokens(query_images)
        text_features = self.encode_text(captions)
        return self.composition(
            image_tokens,
            text_features.tokens,
            text_attention_mask=text_features.attention_mask,
        )

    def encode_query_with_context_intervention(
        self,
        query_images: torch.Tensor,
        captions: list[str] | tuple[str, ...],
        intervention: str,
    ) -> torch.Tensor:
        """Encode queries after an inference-only intervention on global context.

        ``shuffled`` cyclically moves each block's learned context vector between
        samples in the batch. ``uniform`` replaces learned token attention with a
        uniform distribution over valid image and text tokens. The default
        ``encode_query`` and training paths are unaffected.
        """
        image_tokens = self.encode_image_tokens(query_images)
        text_features = self.encode_text(captions)
        return self.composition(
            image_tokens,
            text_features.tokens,
            text_attention_mask=text_features.attention_mask,
            context_intervention=intervention,
        )

    def encode_query_with_attention(
        self,
        query_images: torch.Tensor,
        captions: list[str] | tuple[str, ...],
    ) -> QueryAttentionOutput:
        """Encode a query and expose additive-attention weights for inference analysis.

        The regular ``encode_query`` and training ``forward`` paths remain unchanged and
        do not retain attention tensors.
        """
        image_tokens = self.encode_image_tokens(query_images)
        text_features = self.encode_text(captions)
        embedding, attention_weights = self.composition(
            image_tokens,
            text_features.tokens,
            text_attention_mask=text_features.attention_mask,
            return_attention=True,
        )
        return QueryAttentionOutput(
            embedding=embedding,
            attention_weights=attention_weights,
            image_token_count=int(image_tokens.shape[1]),
            text_features=text_features,
        )

    def forward(
        self,
        query_images: torch.Tensor,
        target_images: torch.Tensor,
        captions: list[str] | tuple[str, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_embeddings = self.encode_query(query_images, captions)
        target_embeddings = self.encode_image(target_images)
        return query_embeddings, target_embeddings


def build_model(config: dict) -> AACLModel:
    embedding_dim = int(config.get("embedding_dim", 768))

    image_config = dict(config.get("image_encoder", {}))
    image_config.setdefault("embedding_dim", embedding_dim)
    image_encoder = build_image_encoder(image_config)

    text_config = dict(config.get("text_encoder", {}))
    text_config.setdefault("embedding_dim", embedding_dim)
    text_encoder = build_text_encoder(text_config)

    composition_config = dict(config.get("composition", {}))
    composition = AACLCompositionModule(
        embedding_dim=embedding_dim,
        num_blocks=int(composition_config.get("num_blocks", 3)),
        num_heads=int(composition_config.get("num_heads", 8)),
        ffn_multiplier=int(composition_config.get("ffn_multiplier", 4)),
        dropout=float(composition_config.get("dropout", 0.1)),
    )
    return AACLModel(image_encoder, text_encoder, composition)
