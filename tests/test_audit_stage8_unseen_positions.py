"""Verify complete-state decontamination of Stage8 position reports."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.audit_stage8_unseen_positions import audit
from scripts.train_neural_go_readout import object_hash


class UnseenPositionAuditTests(unittest.TestCase):
    def test_excludes_matching_complete_state(self) -> None:
        report = {
            "checkpoint": {"hash": "checkpoint-hash"},
            "records": [
                {"state_sha256": "same", "game_id": "new-game", "teacher_match": True},
                {"state_sha256": "fresh", "game_id": "new-game", "teacher_match": False},
            ],
        }
        report["report_sha256"] = object_hash(report)
        cache_records = [{"hashes": {"state_sha256": "same"}, "game_id": "old-game"}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            with patch(
                "scripts.audit_stage8_unseen_positions.load_cache",
                return_value=(None, None, cache_records, {"dataset_hash": "dataset"}),
            ):
                result = audit(Path(directory), path)
        self.assertEqual(result["cached_state_overlap_count"], 1)
        self.assertEqual(result["independent_count"], 1)
        self.assertEqual(result["independent_matches"], 0)
        self.assertEqual(result["cached_game_id_overlap_count"], 0)

    def test_rejects_tampered_report(self) -> None:
        report = {
            "checkpoint": {"hash": "checkpoint-hash"},
            "records": [{"state_sha256": "fresh", "game_id": "new", "teacher_match": True}],
            "report_sha256": "not-the-hash",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            with patch(
                "scripts.audit_stage8_unseen_positions.load_cache",
                return_value=(None, None, [], {"dataset_hash": "dataset"}),
            ):
                with self.assertRaisesRegex(ValueError, "hash"):
                    audit(Path(directory), path)


if __name__ == "__main__":
    unittest.main()
