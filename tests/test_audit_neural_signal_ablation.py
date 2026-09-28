"""Tests for the non-fabricated signal-dependence counterfactual."""

from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

import numpy as np

from scripts.audit_neural_signal_ablation import point_predictions
from scripts.evaluate_stage8_zero_signal_games import play_zero_game
from scripts.train_neural_go_readout import object_hash
from app.server import _load_hashed_report


class NeuralSignalAblationTests(unittest.TestCase):
    def test_point_predictions_respect_legal_mask_and_exclude_pass(self) -> None:
        features = np.array([[1.0, 0.0], [0.0, 1.0]])
        weights = np.zeros((82, 2))
        bias = np.zeros(82)
        weights[5, 0] = 3.0
        weights[7, 1] = 4.0
        bias[81] = 100.0
        masks = np.zeros((2, 82), dtype=np.uint8)
        masks[0, [5, 7, 81]] = 1
        masks[1, [5, 7, 81]] = 1
        self.assertEqual(
            point_predictions(features, weights, bias, masks).tolist(),
            [5, 7],
        )
        masks[0, 5] = 0
        self.assertEqual(
            point_predictions(features, weights, bias, masks).tolist(),
            [7, 7],
        )

    def test_rejects_state_without_legal_point(self) -> None:
        masks = np.zeros((1, 82), dtype=np.uint8)
        masks[0, 81] = 1
        with self.assertRaisesRegex(ValueError, "no legal point"):
            point_predictions(
                np.zeros((1, 2)), np.zeros((82, 2)), np.zeros(82),
                masks,
            )

    def test_zero_signal_game_is_deterministic_and_provenance_is_explicit(self) -> None:
        weights = np.zeros((82, 128), dtype=np.float32)
        bias = np.zeros(82, dtype=np.float32)
        bias[40] = 1.0
        first = play_zero_game(
            seed=17, max_moves=6, opening_plies=2, weights=weights, bias=bias
        )
        second = play_zero_game(
            seed=17, max_moves=6, opening_plies=2, weights=weights, bias=bias
        )
        self.assertEqual(first, second)
        self.assertFalse(first["neural_runtime_called"])
        self.assertEqual(first["evaluation_move_count"], 6)
        self.assertGreater(first["zero_readout_decision_count"], 0)
        self.assertTrue(all(
            item["source"] != "malecns-neural-policy" for item in first["decisions"]
        ))

    def test_ablation_report_loader_rejects_tampered_metrics(self) -> None:
        report = {"schema": "malecns-go-neural-signal-ablation-v1", "matches": 17}
        report["report_sha256"] = object_hash(report)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ablation.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            self.assertEqual(
                _load_hashed_report(path, report["schema"])["matches"], 17
            )
            report["matches"] = 101
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                _load_hashed_report(path, report["schema"])


if __name__ == "__main__":
    unittest.main()
