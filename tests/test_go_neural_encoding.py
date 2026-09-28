"""Tests for the deterministic Go-to-neural engineering adapter."""

from __future__ import annotations

import json
import unittest

from src.go_engine import BLACK, BOARD_POINTS, EMPTY, PASS, WHITE, GoState, initial_state, step
from src.go_neural_encoding import (
    ACTION_COUNT,
    PASS_ACTION_INDEX,
    VECTOR_LENGTH,
    action_to_move,
    board_hash,
    encode_go_state,
    legal_action_mask,
    move_to_action,
)


class GoNeuralEncodingTests(unittest.TestCase):
    def test_all_82_actions_round_trip(self) -> None:
        for action in range(ACTION_COUNT):
            self.assertEqual(move_to_action(action_to_move(action)), action)
        self.assertEqual(move_to_action("pass"), PASS_ACTION_INDEX)
        self.assertIsNone(action_to_move(PASS_ACTION_INDEX))

    def test_invalid_moves_and_actions_are_rejected(self) -> None:
        for invalid in (-1, ACTION_COUNT, True, 1.0, "0"):
            with self.subTest(action=invalid), self.assertRaises(ValueError):
                action_to_move(invalid)
        for invalid in ((-1, 0), (9, 0), (0, 9), (True, 0), (1.0, 2), "A1", [0]):
            with self.subTest(move=invalid), self.assertRaises(ValueError):
                move_to_action(invalid)

    def test_initial_mask_has_every_action_legal(self) -> None:
        mask = legal_action_mask(initial_state())
        self.assertEqual(len(mask), ACTION_COUNT)
        self.assertEqual(mask, [1] * ACTION_COUNT)

    def test_mask_tracks_occupied_points_and_game_over(self) -> None:
        state = step(initial_state(), (4, 4))
        mask = legal_action_mask(state)
        self.assertEqual(mask[move_to_action((4, 4))], 0)
        self.assertEqual(mask[PASS_ACTION_INDEX], 1)
        state = step(state, PASS)
        state = step(state, PASS)
        self.assertEqual(legal_action_mask(state), [0] * ACTION_COUNT)

    def test_encoding_is_fixed_length_json_safe_and_deterministic(self) -> None:
        state = initial_state()
        for move in ((0, 0), (1, 0), (0, 1)):
            state = step(state, move)
        first = encode_go_state(state)
        second = encode_go_state(state)
        self.assertEqual(first, second)
        self.assertEqual(first["vector_length"], VECTOR_LENGTH)
        self.assertEqual(len(first["vector"]), VECTOR_LENGTH)
        self.assertEqual(len(first["legal_mask"]), ACTION_COUNT)
        self.assertEqual(first["last_move_action"], move_to_action((0, 1)))
        self.assertFalse(first["male_cns_used"])
        self.assertIsInstance(json.dumps(first, sort_keys=True), str)
        self.assertEqual(first["encoding_hash"], second["encoding_hash"])

    def test_plane_semantics_are_relative_to_player_to_move(self) -> None:
        state = step(initial_state(), (2, 3))  # White is now to play.
        encoded = encode_go_state(state)
        point = move_to_action((2, 3))
        own_offset = 0
        opponent_offset = BOARD_POINTS
        empty_offset = 2 * BOARD_POINTS
        last_offset = 4 * BOARD_POINTS
        self.assertEqual(encoded["vector"][own_offset + point], 0.0)
        self.assertEqual(encoded["vector"][opponent_offset + point], 1.0)
        self.assertEqual(encoded["vector"][empty_offset + point], 0.0)
        self.assertEqual(encoded["vector"][last_offset + point], 1.0)

    def test_pass_is_encoded_without_a_fake_board_location(self) -> None:
        state = step(initial_state(), PASS)
        encoded = encode_go_state(state)
        last_plane = encoded["vector"][4 * BOARD_POINTS : 5 * BOARD_POINTS]
        scalar_offset = 5 * BOARD_POINTS
        self.assertEqual(encoded["last_move_action"], PASS_ACTION_INDEX)
        self.assertEqual(sum(last_plane), 0.0)
        self.assertEqual(encoded["vector"][scalar_offset + 5], 1.0)

    def test_board_hash_ignores_turn_but_encoding_hash_does_not(self) -> None:
        state = initial_state()
        same_board_white = GoState(
            board=state.board,
            to_play=WHITE,
            black_captures=0,
            white_captures=0,
            consecutive_passes=0,
            move_number=0,
            position_history=(state.board,),
            komi=state.komi,
        )
        self.assertEqual(board_hash(state), board_hash(same_board_white))
        self.assertNotEqual(
            encode_go_state(state)["encoding_hash"],
            encode_go_state(same_board_white)["encoding_hash"],
        )

    def test_capture_counts_and_pass_count_are_in_scalar_tail(self) -> None:
        board = (EMPTY,) * BOARD_POINTS
        state = GoState(
            board=board,
            to_play=BLACK,
            black_captures=3,
            white_captures=2,
            consecutive_passes=1,
            move_number=1,
            position_history=(board, board),
            komi=6.5,
        )
        tail = encode_go_state(state)["vector"][5 * BOARD_POINTS :]
        self.assertEqual(tail, [1.0, 0.0, 3.0, 2.0, 1.0, 1.0, 1.0, 0.0, 1.0, 6.5])

    def test_inconsistent_history_is_rejected_instead_of_guessed(self) -> None:
        before = (EMPTY,) * BOARD_POINTS
        after = list(before)
        after[0] = BLACK
        after[1] = BLACK
        state = GoState(
            board=tuple(after),
            to_play=WHITE,
            black_captures=0,
            white_captures=0,
            consecutive_passes=0,
            move_number=1,
            position_history=(before, tuple(after)),
        )
        with self.assertRaisesRegex(ValueError, "cannot derive one last move"):
            encode_go_state(state)

    def test_wrong_state_type_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            encode_go_state({})  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            legal_action_mask(None)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            board_hash("board")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
