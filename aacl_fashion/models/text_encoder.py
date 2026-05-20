from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class TextFeatures:
    tokens: torch.Tensor
    attention_mask: torch.Tensor | None = None


def _set_trainable(module: nn.Module, trainable: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = trainable


class DistilBertTextEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        max_length: int = 128,
        freeze: bool = False,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Install transformers to use DistilBertTextEncoder: pip install transformers"
            ) from exc

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.max_length = max_length
        if freeze:
            _set_trainable(self.model, False)

    @property
    def output_dim(self) -> int:
        return int(self.model.config.hidden_size)

    def forward(self, texts: list[str] | tuple[str, ...]) -> TextFeatures:
        device = next(self.model.parameters()).device
        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        output = self.model(**encoded)
        return TextFeatures(tokens=output.last_hidden_state, attention_mask=encoded.get("attention_mask"))


class SimpleTextEncoder(nn.Module):
    """Dependency-free tokenizer/encoder for tests and synthetic smoke runs."""

    def __init__(self, embedding_dim: int = 768, max_length: int = 32, vocab_size: int = 256) -> None:
        super().__init__()
        self.max_length = max_length
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.projection = nn.Linear(embedding_dim, embedding_dim)

    @property
    def output_dim(self) -> int:
        return self.embedding.embedding_dim

    def forward(self, texts: list[str] | tuple[str, ...]) -> TextFeatures:
        device = self.embedding.weight.device
        token_rows = []
        mask_rows = []
        for text in texts:
            raw = text.lower().encode("utf-8")[: self.max_length]
            ids = [byte for byte in raw]
            mask = [1] * len(ids)
            pad_count = self.max_length - len(ids)
            ids.extend([0] * pad_count)
            mask.extend([0] * pad_count)
            token_rows.append(ids)
            mask_rows.append(mask)

        token_ids = torch.tensor(token_rows, dtype=torch.long, device=device)
        attention_mask = torch.tensor(mask_rows, dtype=torch.long, device=device)
        return TextFeatures(tokens=self.projection(self.embedding(token_ids)), attention_mask=attention_mask)


def build_text_encoder(config: dict) -> nn.Module:
    encoder_type = config.get("type", "distilbert")
    embedding_dim = int(config.get("embedding_dim", 768))

    if encoder_type == "simple":
        return SimpleTextEncoder(
            embedding_dim=embedding_dim,
            max_length=int(config.get("max_length", 32)),
        )
    if encoder_type != "distilbert":
        raise ValueError(f"Unsupported text encoder type: {encoder_type}")

    return DistilBertTextEncoder(
        model_name=config.get("name", "distilbert-base-uncased"),
        max_length=int(config.get("max_length", 128)),
        freeze=bool(config.get("freeze", False)),
    )
