from __future__ import annotations

import random
import unittest

from scripts.create_hallucination_probes import SCENARIOS, _select_unique_queries


class HallucinationWorkflowTests(unittest.TestCase):
    def test_probe_selection_is_deterministic_and_uses_unique_query_ids(self) -> None:
        records = [
            {"query_id": "a", "target_id": "x", "captions": ["one"]},
            {"query_id": "a", "target_id": "y", "captions": ["two"]},
            {"query_id": "b", "target_id": "z", "captions": ["three"]},
            {"query_id": "c", "target_id": "w", "captions": ["four"]},
        ]
        first = _select_unique_queries(records, 3, random.Random(42))
        second = _select_unique_queries(records, 3, random.Random(42))
        self.assertEqual(first, second)
        self.assertEqual(len({record["query_id"] for _, record in first}), 3)

    def test_probe_selection_applies_documented_exclusions(self) -> None:
        records = [
            {"query_id": "bad", "target_id": "x", "captions": ["shoe"]},
            {"query_id": "good", "target_id": "y", "captions": ["dress"]},
        ]
        selected = _select_unique_queries(
            records, 1, random.Random(1), excluded_query_ids={"bad"}
        )
        self.assertEqual(selected[0][1]["query_id"], "good")

    def test_probe_protocol_has_six_distinct_scenarios(self) -> None:
        names = [item["scenario"] for item in SCENARIOS]
        self.assertEqual(len(names), 6)
        self.assertEqual(len(set(names)), 6)
        self.assertIn("in_domain", names)
        self.assertIn("ood", names)

if __name__ == "__main__":
    unittest.main()
