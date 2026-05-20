from __future__ import annotations

import tempfile
import unittest

import torch

from aacl_fashion.config import load_config
from aacl_fashion.data.fashioniq_dataset import FashionIQDataset
from aacl_fashion.losses import BatchSoftmaxLoss
from aacl_fashion.models import build_model
from aacl_fashion.models.composition_module import AACLCompositionModule
from aacl_fashion.utils.metrics import recall_at_ks


class AACLCoreTests(unittest.TestCase):
    def test_composition_shape(self) -> None:
        module = AACLCompositionModule(
            embedding_dim=64,
            num_blocks=1,
            num_heads=4,
            ffn_multiplier=2,
            dropout=0.0,
        )
        image_tokens = torch.randn(2, 98, 64)
        text_tokens = torch.randn(2, 6, 64)
        text_mask = torch.ones(2, 6, dtype=torch.long)
        output = module(image_tokens, text_tokens, text_mask)
        self.assertEqual(tuple(output.shape), (2, 64))
        self.assertTrue(torch.allclose(output.norm(dim=-1), torch.ones(2), atol=1e-5))

    def test_symmetric_loss(self) -> None:
        loss_fn = BatchSoftmaxLoss(temperature=0.07, symmetric=True, label_smoothing=0.1)
        query = torch.randn(4, 64, requires_grad=True)
        target = torch.randn(4, 64)
        loss = loss_fn(query, target)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertIsNotNone(query.grad)

    def test_mock_model_forward(self) -> None:
        config = load_config("configs/synthetic_smoke.yaml")
        model = build_model(config["model"])
        query_images = torch.randn(2, 3, 32, 32)
        target_images = torch.randn(2, 3, 32, 32)
        captions = ["make it blue", "add long sleeves"]
        query_embeddings, target_embeddings = model(query_images, target_images, captions)
        self.assertEqual(tuple(query_embeddings.shape), (2, 64))
        self.assertEqual(tuple(target_embeddings.shape), (2, 64))

    def test_recall_excludes_query_id(self) -> None:
        query = torch.tensor([[1.0, 0.0]])
        gallery = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        metrics = recall_at_ks(
            query_embeddings=query,
            gallery_embeddings=gallery,
            target_ids=["target"],
            gallery_ids=["query", "target"],
            ks=[1],
            query_ids=["query"],
            exclude_query=True,
        )
        self.assertEqual(metrics["R@1"], 100.0)

    def test_missing_fashioniq_root_has_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(FileNotFoundError, "FashionIQ data is incomplete"):
                FashionIQDataset(root=temp_dir, category="dress", split="train")


if __name__ == "__main__":
    unittest.main()
