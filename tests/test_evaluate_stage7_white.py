"""Unit tests for the stricter Stage 7 production-role evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.evaluate_stage7_white import (
    FINAL_MINIMUM_GAMES,
    _reference_evidence,
    _parser,
    build_final_gates,
    play_random_white_baseline,
    semantic_core_hash,
    wilson_interval,
)
from scripts.evaluate_neural_go_games import play_game
from src.go_engine import WHITE, legal_moves


class Stage7WhiteEvaluationTests(unittest.TestCase):
    def test_wilson_interval_for_twenty_wins(self) -> None:
        lower, upper = wilson_interval(20, 20)
        self.assertAlmostEqual(lower, 0.8388748419471806)
        self.assertAlmostEqual(upper, 1.0)

    def test_wilson_interval_contains_observed_rate(self) -> None:
        lower, upper = wilson_interval(16, 20)
        self.assertLess(lower, 0.8)
        self.assertGreater(upper, 0.8)

    def test_invalid_inputs_fail_closed(self) -> None:
        for successes, trials in ((1, 0), (-1, 2), (3, 2)):
            with self.subTest(successes=successes, trials=trials):
                with self.assertRaises(ValueError):
                    wilson_interval(successes, trials)

    def test_default_twenty_game_run_is_development(self) -> None:
        args = _parser().parse_args([])
        self.assertEqual(args.seed_count, 20)
        self.assertEqual(args.evaluation_tier, "development")
        self.assertEqual(FINAL_MINIMUM_GAMES, 100)

    def test_twenty_games_can_never_pass_frozen_final_gate(self) -> None:
        gates = build_final_gates(
            game_count=20,
            win_rate=1.0,
            wilson_lower=0.84,
            natural_finish_rate=1.0,
            random_white_lift=0.5,
            overlap_is_assessed=True,
            seed_overlap_count=0,
            state_hash_overlap_count=0,
            unique_evaluation_seeds=True,
            runtime_guarantees_hold=True,
        )
        self.assertFalse(gates["minimum_game_count_100"])
        self.assertFalse(all(gates.values()))

    def test_frozen_final_gate_requires_every_audited_condition(self) -> None:
        gates = build_final_gates(
            game_count=100,
            win_rate=0.80,
            wilson_lower=0.70,
            natural_finish_rate=0.90,
            random_white_lift=0.15,
            overlap_is_assessed=True,
            seed_overlap_count=0,
            state_hash_overlap_count=0,
            unique_evaluation_seeds=True,
            runtime_guarantees_hold=True,
        )
        self.assertTrue(all(gates.values()))

    def test_semantic_hash_ignores_elapsed_and_paths(self) -> None:
        first = {
            "status": "development_completed",
            "elapsed_seconds": 1.0,
            "output_path": "C:/machine-a/report.json",
            "nested": {"checkpoint_path": "C:/a", "path": "C:/raw", "wins": 4},
        }
        second = {
            "status": "development_completed",
            "elapsed_seconds": 999.0,
            "output_path": "D:/machine-b/report.json",
            "nested": {"checkpoint_path": "D:/b", "path": "D:/raw", "wins": 4},
        }
        self.assertEqual(semantic_core_hash(first), semantic_core_hash(second))

    def test_random_white_baseline_is_deterministic_and_neural_free(self) -> None:
        first = play_random_white_baseline(seed=123, max_moves=8)
        second = play_random_white_baseline(seed=123, max_moves=8)
        self.assertEqual(first, second)
        self.assertFalse(first["neural_runtime_called"])
        self.assertEqual(first["white_policy"], "uniform-legal-point-random-v1")

    def test_random_white_baseline_shares_neural_games_opening_and_black_stream(self) -> None:
        def first_legal(state, _seed):
            return legal_moves(state, include_pass=False)[0], {}

        game = play_game(
            seed=123,
            neural_color=WHITE,
            max_moves=8,
            opening_plies=4,
            neural_selector=first_legal,
        )
        baseline = play_random_white_baseline(
            seed=123, max_moves=8, opening_plies=4
        )
        self.assertEqual(
            baseline["opening_final_board_hash"],
            game["opening"]["moves"][-1]["board_hash_after"],
        )
        self.assertEqual(
            baseline["opponent_random_stream_id"],
            game["opponent_random_stream"]["id"],
        )

    def test_reference_evidence_uses_neural_decision_states(self) -> None:
        payload = {
            "games": [
                {
                    "seed": 77,
                    "moves": [
                        {
                            "actor": "legal-random",
                            "board_hash_before": "random-state",
                        },
                        {
                            "actor": "malecns-neural-policy",
                            "board_hash_before": "decision-state",
                        },
                    ],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            seeds, hashes = _reference_evidence([path])
        self.assertEqual(seeds, {77})
        self.assertEqual(hashes, {"decision-state"})


if __name__ == "__main__":
    unittest.main()
