"""Fast tests for the Stage 6 paired-game evaluator."""

from __future__ import annotations

import random
import unittest

from scripts.evaluate_neural_go_games import (
    OPENING_RANDOM_STREAM_VERSION,
    OPPONENT_RANDOM_STREAM_VERSION,
    STATE_HASH_VERSION,
    opening_random_stream_id,
    opponent_random_stream_id,
    play_game,
    schedule,
    select_legal_random_move,
)
from scripts.train_neural_go_readout import object_hash
from src.go_engine import BLACK, PASS, WHITE, initial_state, legal_moves, step


class Stage6GameEvaluationTests(unittest.TestCase):
    def test_schedule_swaps_neural_colour_for_every_seed(self) -> None:
        self.assertEqual(
            schedule(100, 2),
            [(100, BLACK), (100, WHITE), (101, BLACK), (101, WHITE)],
        )

    def test_legal_random_policy_is_seeded_and_returns_a_legal_point(self) -> None:
        state = initial_state()
        first = select_legal_random_move(state, random.Random(17))
        second = select_legal_random_move(state, random.Random(17))
        self.assertEqual(first, second)
        self.assertIn(first, legal_moves(state, include_pass=False))

    def test_paired_colours_share_the_same_seed_only_random_stream(self) -> None:
        def first_legal(state, _seed):
            return legal_moves(state, include_pass=False)[0], {}

        black_game = play_game(
            seed=29,
            neural_color=BLACK,
            max_moves=4,
            neural_selector=first_legal,
        )
        white_game = play_game(
            seed=29,
            neural_color=WHITE,
            max_moves=4,
            neural_selector=first_legal,
        )
        black_random = next(move for move in black_game["moves"] if move["actor"] == "legal-random")
        white_random = next(move for move in white_game["moves"] if move["actor"] == "legal-random")
        expected_stream_id = opponent_random_stream_id(29)

        self.assertEqual(black_game["opponent_random_stream"], white_game["opponent_random_stream"])
        self.assertEqual(black_game["opponent_random_stream"]["id"], expected_stream_id)
        self.assertEqual(black_random["provenance"]["random_stream_id"], expected_stream_id)
        self.assertEqual(white_random["provenance"]["random_stream_id"], expected_stream_id)
        self.assertEqual(black_random["provenance"]["seed_scope"], "game-seed-only")
        self.assertEqual(white_random["provenance"]["seed_scope"], "game-seed-only")
        self.assertEqual(
            black_random["provenance"]["random_stream_version"],
            OPPONENT_RANDOM_STREAM_VERSION,
        )

    def test_seed_only_stream_replays_the_same_random_draw_sequence(self) -> None:
        first = random.Random(opponent_random_stream_id(31))
        second = random.Random(opponent_random_stream_id(31))
        self.assertEqual(
            [first.random() for _ in range(8)],
            [second.random() for _ in range(8)],
        )

    def test_game_records_both_policies_and_horizon_truthfully(self) -> None:
        def first_legal(state, seed):
            move = legal_moves(state, include_pass=False)[0]
            return move, {
                "test_seed": seed,
                "teacher_accessed_at_runtime": False,
                "baseline_controller_called": False,
                "fallback_used": False,
            }

        game = play_game(
            seed=21,
            neural_color=BLACK,
            max_moves=6,
            neural_selector=first_legal,
        )
        self.assertEqual(game["move_count"], 6)
        self.assertEqual(game["termination"], "fixed_horizon_adjudication")
        self.assertEqual([move["actor"] for move in game["moves"]][::2], ["malecns-neural-policy"] * 3)
        self.assertEqual([move["actor"] for move in game["moves"]][1::2], ["legal-random"] * 3)
        self.assertTrue(all(len(move["record_sha256"]) == 64 for move in game["moves"]))
        self.assertEqual(len(game["game_sha256"]), 64)

    def test_rule_pass_is_attributed_to_rule_controller(self) -> None:
        def rule_selector(_state, _seed):
            return PASS, {
                "controller_source": "go-rule-pass-gate",
                "controller_reason": "opponent-last-move-was-pass",
                "neural_readout_used": False,
            }

        game = play_game(
            seed=22,
            neural_color=BLACK,
            max_moves=4,
            neural_selector=rule_selector,
        )
        rule_moves = [move for move in game["moves"] if move["color"] == "black"]
        self.assertTrue(rule_moves)
        self.assertTrue(all(move["actor"] == "go-rule-pass-gate" for move in rule_moves))
        self.assertTrue(all(not move["provenance"]["neural_readout_used"] for move in rule_moves))

    def test_colour_swapped_games_support_full_state_leakage_audit(self) -> None:
        def first_legal(state, _seed):
            return legal_moves(state, include_pass=False)[0], {}

        games = [
            play_game(
                seed=37,
                neural_color=neural_color,
                max_moves=6,
                neural_selector=first_legal,
            )
            for neural_color in (BLACK, WHITE)
        ]
        initial_hash = object_hash(initial_state().to_dict())
        self.assertEqual(
            {game["moves"][0]["state_sha256_before"] for game in games},
            {initial_hash},
        )

        for game in games:
            state = initial_state()
            for record in game["moves"]:
                self.assertEqual(record["state_sha256_before"], object_hash(state.to_dict()))
                move = PASS if record["move"] == "pass" else tuple(record["move"])
                state = step(state, move)
                self.assertEqual(record["state_sha256_after"], object_hash(state.to_dict()))
            self.assertTrue(
                all(
                    previous["state_sha256_after"] == following["state_sha256_before"]
                    for previous, following in zip(game["moves"], game["moves"][1:])
                )
            )

        self.assertEqual(STATE_HASH_VERSION, "go-state-to-dict-sha256-v1")

    def test_seeded_openings_are_reproducible_and_independent_by_role(self) -> None:
        def first_legal(state, _seed):
            return legal_moves(state, include_pass=False)[0], {}

        black_first = play_game(
            seed=41,
            neural_color=BLACK,
            max_moves=4,
            opening_plies=3,
            neural_selector=first_legal,
        )
        black_replay = play_game(
            seed=41,
            neural_color=BLACK,
            max_moves=4,
            opening_plies=3,
            neural_selector=first_legal,
        )
        white_game = play_game(
            seed=41,
            neural_color=WHITE,
            max_moves=4,
            opening_plies=3,
            neural_selector=first_legal,
        )

        self.assertEqual(black_first["opening"], black_replay["opening"])
        self.assertEqual(
            black_first["opening"]["random_stream"]["id"],
            opening_random_stream_id(41, BLACK),
        )
        self.assertEqual(
            white_game["opening"]["random_stream"]["id"],
            opening_random_stream_id(41, WHITE),
        )
        self.assertNotEqual(
            black_first["opening"]["random_stream"]["id"],
            white_game["opening"]["random_stream"]["id"],
        )
        self.assertEqual(
            black_first["opening"]["random_stream"]["version"],
            OPENING_RANDOM_STREAM_VERSION,
        )

    def test_opening_is_auditable_and_excluded_from_evaluation_move_count(self) -> None:
        selector_state_plies = []

        def first_legal(state, _seed):
            selector_state_plies.append(state.move_number)
            return legal_moves(state, include_pass=False)[0], {}

        game = play_game(
            seed=43,
            neural_color=BLACK,
            max_moves=6,
            opening_plies=2,
            neural_selector=first_legal,
        )
        self.assertEqual(game["opening"]["requested_plies"], 2)
        self.assertEqual(game["opening"]["completed_plies"], 2)
        self.assertFalse(game["opening"]["neural_policy_called"])
        self.assertFalse(game["opening"]["attributed_to_evaluated_policy"])
        self.assertEqual(game["move_count"], 6)
        self.assertEqual(game["evaluation_move_count"], 6)
        self.assertEqual(game["total_move_count"], 8)
        self.assertTrue(selector_state_plies)
        self.assertGreaterEqual(min(selector_state_plies), 2)

        state = initial_state()
        for opening_record in game["opening"]["moves"]:
            self.assertEqual(
                opening_record["state_sha256_before"], object_hash(state.to_dict())
            )
            move = PASS if opening_record["move"] == "pass" else tuple(opening_record["move"])
            self.assertIn(move, legal_moves(state, include_pass=True))
            state = step(state, move)
            self.assertEqual(
                opening_record["state_sha256_after"], object_hash(state.to_dict())
            )
            self.assertFalse(opening_record["provenance"]["neural_policy_called"])
        self.assertEqual(
            game["moves"][0]["state_sha256_before"], object_hash(state.to_dict())
        )

    def test_invalid_schedule_and_game_inputs_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            schedule(0, 0)
        with self.assertRaises(ValueError):
            opponent_random_stream_id(-1)
        with self.assertRaises(ValueError):
            opening_random_stream_id(1, 3)
        with self.assertRaises(ValueError):
            play_game(seed=1, neural_color=3, max_moves=4, neural_selector=lambda *_: (None, {}))
        with self.assertRaises(ValueError):
            play_game(
                seed=1,
                neural_color=BLACK,
                max_moves=4,
                opening_plies=-1,
                neural_selector=lambda *_: (None, {}),
            )


if __name__ == "__main__":
    unittest.main()
