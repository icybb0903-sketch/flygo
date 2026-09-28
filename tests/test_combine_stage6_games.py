"""Statistics and validation tests for Stage 6 batch combination."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.combine_stage6_games import binomial_tail, load_batch
from scripts.train_neural_go_readout import object_hash


class Stage6CombinationTests(unittest.TestCase):
    def test_one_sided_fair_binomial_tail(self) -> None:
        self.assertAlmostEqual(binomial_tail(10, 10), 1 / 1024)
        self.assertAlmostEqual(binomial_tail(9, 10), 11 / 1024)
        self.assertAlmostEqual(binomial_tail(0, 10), 1.0)

    def test_invalid_binomial_inputs_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            binomial_tail(2, 1)
        with self.assertRaises(ValueError):
            binomial_tail(0, 1, 1.1)

    def test_legacy_v1_batch_remains_readable(self) -> None:
        payload = {
            "schema": "malecns-go-stage6-games-v1",
            "configuration": {"opponent": "uniform-legal-point-random-v1"},
            "games": [],
        }
        payload["report_sha256"] = object_hash(payload)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = load_batch(path)
        self.assertEqual(loaded["schema"], "malecns-go-stage6-games-v1")

    def test_legacy_v2_batch_remains_readable(self) -> None:
        payload = {
            "schema": "malecns-go-stage6-games-v2",
            "configuration": {
                "opponent": "uniform-legal-point-random-v1",
                "opponent_random_stream_version": "seed-only-v2",
            },
            "games": [],
        }
        payload["report_sha256"] = object_hash(payload)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy-v2.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = load_batch(path)
        self.assertEqual(loaded["schema"], "malecns-go-stage6-games-v2")

    def test_legacy_v3_batch_remains_readable(self) -> None:
        payload = {
            "schema": "malecns-go-stage6-games-v3",
            "configuration": {
                "opponent": "uniform-legal-point-random-v1",
                "opponent_random_stream_version": "seed-only-v2",
                "move_state_hash": {
                    "version": "go-state-to-dict-sha256-v1",
                    "algorithm": "sha256-canonical-json",
                    "payload": "GoState.to_dict()",
                },
            },
            "games": [],
        }
        payload["report_sha256"] = object_hash(payload)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy-v3.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = load_batch(path)
        self.assertEqual(loaded["schema"], "malecns-go-stage6-games-v3")


if __name__ == "__main__":
    unittest.main()
