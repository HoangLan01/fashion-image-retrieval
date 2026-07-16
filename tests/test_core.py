from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from aacl_fashion.attention import (
    compute_attention_flow,
    merge_wordpiece_scores,
    split_image_attention_maps,
)
from aacl_fashion.config import load_config
from aacl_fashion.data.fashioniq_dataset import FashionIQDataset
from aacl_fashion.data.lexical_holdout import (
    audit_records,
    filter_holdout_records,
    record_matches_holdout,
)
from aacl_fashion.evaluation import retrieval_details
from aacl_fashion.losses import BatchSoftmaxLoss
from aacl_fashion.models import build_model
from aacl_fashion.models.composition_module import AACLCompositionModule
from aacl_fashion.occlusion import (
    apply_patch_mask,
    paired_bootstrap_ci,
    paired_sign_flip_pvalue,
    select_patch_mask,
)
from aacl_fashion.training import resolve_run_output_dir, train_one_category
from aacl_fashion.utils.checkpoint import load_checkpoint, save_checkpoint
from aacl_fashion.utils.device import resolve_device
from aacl_fashion.utils.metrics import recall_at_ks
from scripts.run_context_intervention import _sign_flip_pvalue


class AACLCoreTests(unittest.TestCase):
    def test_resolve_device_adds_default_cuda_index(self) -> None:
        with patch("torch.cuda.set_device") as set_device:
            device = resolve_device("cuda")

        self.assertEqual(device, torch.device("cuda:0"))
        set_device.assert_called_once_with(0)

    def test_resolve_device_preserves_explicit_cuda_index(self) -> None:
        with patch("torch.cuda.set_device") as set_device:
            device = resolve_device("cuda:2")

        self.assertEqual(device, torch.device("cuda:2"))
        set_device.assert_called_once_with(2)

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

    def test_composition_optionally_returns_block_head_token_attention(self) -> None:
        module = AACLCompositionModule(
            embedding_dim=64,
            num_blocks=3,
            num_heads=4,
            ffn_multiplier=2,
            dropout=0.0,
        ).eval()
        image_tokens = torch.randn(2, 98, 64)
        text_tokens = torch.randn(2, 6, 64)
        text_mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]])
        legacy_output = module(image_tokens, text_tokens, text_mask)
        output, weights = module(
            image_tokens,
            text_tokens,
            text_mask,
            return_attention=True,
        )
        self.assertTrue(torch.allclose(legacy_output, output, atol=1e-6))
        self.assertEqual(tuple(weights.shape), (2, 3, 4, 104))
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(2, 3, 4), atol=1e-6))
        self.assertTrue(torch.equal(weights[0, :, :, -2:], torch.zeros(3, 4, 2)))

    def test_context_interventions_leave_default_path_unchanged(self) -> None:
        module = AACLCompositionModule(
            embedding_dim=32,
            num_blocks=2,
            num_heads=4,
            ffn_multiplier=2,
            dropout=0.0,
        ).eval()
        image_tokens = torch.randn(4, 8, 32)
        text_tokens = torch.randn(4, 5, 32)
        text_mask = torch.tensor(
            [[1, 1, 1, 1, 0], [1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [1, 1, 1, 1, 0]]
        )
        default = module(image_tokens, text_tokens, text_mask)
        explicit = module(
            image_tokens,
            text_tokens,
            text_mask,
            context_intervention="none",
        )
        shuffled = module(
            image_tokens,
            text_tokens,
            text_mask,
            context_intervention="shuffled",
        )
        uniform, weights = module(
            image_tokens,
            text_tokens,
            text_mask,
            return_attention=True,
            context_intervention="uniform",
        )
        self.assertTrue(torch.allclose(default, explicit, atol=1e-6))
        self.assertFalse(torch.allclose(default, shuffled))
        self.assertFalse(torch.allclose(default, uniform))
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(4, 2, 4), atol=1e-6))
        self.assertTrue(torch.equal(weights[0, :, :, -1], torch.zeros(2, 4)))

    def test_shuffled_context_rejects_singleton_batch(self) -> None:
        module = AACLCompositionModule(
            embedding_dim=16,
            num_blocks=1,
            num_heads=4,
            ffn_multiplier=2,
            dropout=0.0,
        ).eval()
        with self.assertRaisesRegex(ValueError, "at least 2 samples"):
            module(
                torch.randn(1, 4, 16),
                torch.randn(1, 3, 16),
                torch.ones(1, 3, dtype=torch.long),
                context_intervention="shuffled",
            )

    def test_attention_flow_splits_stage3_stage4_and_masks_padding(self) -> None:
        weights = torch.rand(2, 3, 4, 104)
        mask = torch.ones(2, 104, dtype=torch.long)
        mask[0, -2:] = 0
        weights = weights * mask[:, None, None, :]
        weights = weights / weights.sum(dim=-1, keepdim=True)
        flow = compute_attention_flow(weights, attention_mask=mask)
        self.assertEqual(tuple(flow.shape), (2, 4, 104))
        self.assertTrue(torch.equal(flow[0, :, -2:], torch.zeros(4, 2)))
        maps = split_image_attention_maps(flow, image_token_count=98, pool_size=7)
        self.assertEqual(tuple(maps["stage3_per_head"].shape), (2, 4, 7, 7))
        self.assertEqual(tuple(maps["stage4"].shape), (2, 7, 7))
        self.assertEqual(tuple(maps["average"].shape), (2, 7, 7))

    def test_wordpiece_scores_merge_subwords_and_drop_special_tokens(self) -> None:
        words = merge_wordpiece_scores(
            ["[CLS]", "long", "##er", "sleeves", "[SEP]", "[PAD]"],
            torch.tensor([0.0, 0.2, 0.6, 0.9, 0.0, 0.0]),
        )
        self.assertEqual([item["word"] for item in words], ["longer", "sleeves"])
        self.assertEqual(words[0]["score"], 0.0)
        self.assertEqual(words[1]["score"], 1.0)

    def test_high_low_occlusion_masks_have_equal_area(self) -> None:
        attention = torch.arange(49, dtype=torch.float32).reshape(7, 7)
        high = select_patch_mask(attention, ratio=0.2, highest=True)
        low = select_patch_mask(attention, ratio=0.2, highest=False)
        self.assertEqual(int(high.sum()), 10)
        self.assertEqual(int(low.sum()), 10)
        self.assertFalse(bool((high & low).any()))

        image = torch.ones(3, 14, 14)
        occluded = apply_patch_mask(image, high, fill_value=0.0)
        self.assertEqual(int((occluded[0] == 0).sum()), 40)
        self.assertTrue(torch.equal(occluded[0] == 0, occluded[1] == 0))

    def test_paired_bootstrap_and_exact_sign_flip_are_reproducible(self) -> None:
        first = paired_bootstrap_ci([1.0, 2.0, 3.0], samples=200, seed=42)
        second = paired_bootstrap_ci([1.0, 2.0, 3.0], samples=200, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(paired_sign_flip_pvalue([1.0, 1.0, 1.0]), 0.25)
        self.assertEqual(paired_sign_flip_pvalue([1.0, -1.0]), 1.0)

    def test_large_sign_flip_test_uses_reproducible_monte_carlo(self) -> None:
        differences = torch.linspace(-0.5, 1.0, 25).numpy()
        first = _sign_flip_pvalue(differences, samples=200, seed=42)
        second = _sign_flip_pvalue(differences, samples=200, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(first[1], "monte_carlo_200")

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

    def test_mock_model_attention_inference_api(self) -> None:
        config = load_config("configs/synthetic_smoke.yaml")
        model = build_model(config["model"]).eval()
        images = torch.randn(2, 3, 32, 32)
        output = model.encode_query_with_attention(images, ["make it blue", "add sleeves"])
        self.assertEqual(tuple(output.embedding.shape), (2, 64))
        self.assertEqual(tuple(output.attention_weights.shape), (2, 1, 4, 122))
        self.assertEqual(output.image_token_count, 98)
        self.assertIsNotNone(output.text_features.token_labels)

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

    def test_retrieval_details_export_target_rank_and_margin(self) -> None:
        rows = retrieval_details(
            query_embeddings=torch.tensor([[1.0, 0.0]]),
            gallery_embeddings=torch.tensor([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]]),
            query_ids=["query"],
            target_ids=["target"],
            gallery_ids=["query", "target", "other"],
            exclude_query=True,
        )
        self.assertEqual(rows[0]["target_rank"], 1)
        self.assertEqual(rows[0]["top1_id"], "target")
        self.assertAlmostEqual(float(rows[0]["top1_top2_margin"]), 0.8)

    def test_missing_fashioniq_root_has_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(FileNotFoundError, "FashionIQ data is incomplete"):
                FashionIQDataset(root=temp_dir, category="dress", split="train")

    def test_lexical_holdout_matches_surface_forms_without_substring_leakage(self) -> None:
        positives = ["T-shirt", "t shirts", "TSHIRT", "two tees"]
        negatives = ["shirt", "toptee", "teeth", "contest-shirt"]
        for caption in positives:
            self.assertTrue(record_matches_holdout({"captions": [caption]}), caption)
        for caption in negatives:
            self.assertFalse(record_matches_holdout({"captions": [caption]}), caption)

    def test_lexical_holdout_removes_entire_matching_record(self) -> None:
        records = [
            {"captions": ["long sleeves", "plain shirt"]},
            {"captions": ["make it a tee", "short sleeves"]},
        ]
        retained, removed = filter_holdout_records(records)
        self.assertEqual(retained, [records[0]])
        self.assertEqual(removed, [records[1]])
        self.assertEqual(audit_records(retained)["matching_records"], 0)

    def test_run_output_dir_separates_named_runs_and_categories(self) -> None:
        config = load_config("configs/synthetic_smoke.yaml")
        config["training"]["output_dir"] = "outputs/example"
        config["training"]["run_name"] = "seed_42"
        self.assertEqual(
            resolve_run_output_dir(config, "shirt"),
            Path("outputs/example/seed_42/shirt"),
        )

    def test_extended_checkpoint_round_trip(self) -> None:
        config = load_config("configs/synthetic_smoke.yaml")
        model = build_model(config["model"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        history = [{"epoch": 1, "train_loss": 1.25, "R@1": 50.0}]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "checkpoint.pt"
            save_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                epoch=1,
                metrics={"loss": 1.25, "R@1": 50.0},
                config=config,
                best_score=50.0,
                best_epoch=1,
                best_metrics={"loss": 1.25, "R@1": 50.0},
                history=history,
                category="dress",
            )
            checkpoint = load_checkpoint(path)
        self.assertEqual(checkpoint["epoch"], 1)
        self.assertEqual(checkpoint["category"], "dress")
        self.assertEqual(checkpoint["best_score"], 50.0)
        self.assertEqual(checkpoint["best_epoch"], 1)
        self.assertEqual(checkpoint["history"], history)
        self.assertIn("optimizer", checkpoint)
        self.assertIn("rng_state", checkpoint)

    def test_training_can_resume_with_history(self) -> None:
        config = load_config("configs/synthetic_smoke.yaml")
        with tempfile.TemporaryDirectory() as temp_dir:
            config["training"]["output_dir"] = temp_dir
            config["training"]["epochs"] = 1
            train_one_category(config, "dress", overwrite=True, device_name="cpu")

            config["training"]["epochs"] = 2
            train_one_category(config, "dress", resume="auto", device_name="cpu")

            output_dir = resolve_run_output_dir(config, "dress")
            checkpoint = load_checkpoint(output_dir / "latest.pt")
            metrics_csv = (output_dir / "metrics.csv").read_text(encoding="utf-8")

        self.assertEqual(checkpoint["epoch"], 2)
        self.assertEqual(checkpoint["best_epoch"], 1)
        self.assertEqual([row["epoch"] for row in checkpoint["history"]], [1, 2])
        self.assertIn("train_loss", metrics_csv)
        self.assertIn("duration_seconds", metrics_csv)


if __name__ == "__main__":
    unittest.main()
