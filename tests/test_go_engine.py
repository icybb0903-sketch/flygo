"""Offline tests for the deterministic 9x9 Go environment."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import unittest

from src.go_engine import (
    BLACK,
    BOARD_POINTS,
    EMPTY,
    PASS,
    WHITE,
    GoEngine,
    GoState,
    IllegalMoveError,
    deserialize_state,
    initial_state,
    legal_moves,
    score_area,
    serialize_state,
    step,
)


def state_with_stones(
    stones: dict[tuple[int, int], int], *, to_play: int = BLACK, komi: float = 7.5
) -> GoState:
    board = [EMPTY] * BOARD_POINTS
    for (row, column), color in stones.items():
        board[row * 9 + column] = color
    frozen_board = tuple(board)
    return GoState(
        board=frozen_board,
        to_play=to_play,
        black_captures=0,
        white_captures=0,
        consecutive_passes=0,
        move_number=0,
        position_history=(frozen_board,),
        komi=komi,
    )


class GoEngineTests(unittest.TestCase):
    def test_initial_legal_moves_are_deterministic_and_pass_is_last(self) -> None:
        state = initial_state()
        first = legal_moves(state)
        second = legal_moves(state)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 82)
        self.assertEqual(first[:3], ((0, 0), (0, 1), (0, 2)))
        self.assertEqual(first[-2:], ((8, 8), PASS))

    def test_turns_alternate_and_occupied_point_is_illegal(self) -> None:
        before = initial_state()
        after_black = step(before, (4, 4))
        self.assertEqual(before.stone_at(4, 4), EMPTY)
        self.assertEqual(after_black.stone_at(4, 4), BLACK)
        self.assertEqual(after_black.to_play, WHITE)
        with self.assertRaisesRegex(IllegalMoveError, "occupied") as raised:
            step(after_black, (4, 4))
        self.assertEqual(raised.exception.reason, "occupied")
        self.assertEqual(after_black.stone_at(4, 4), BLACK)
        self.assertEqual(after_black.move_number, 1)

    def test_surrounded_stone_is_captured(self) -> None:
        state = initial_state()
        for move in ((0, 1), (1, 1), (1, 0), PASS, (1, 2), PASS, (2, 1)):
            state = step(state, move)
        self.assertEqual(state.stone_at(1, 1), EMPTY)
        self.assertEqual(state.black_captures, 1)
        self.assertEqual(state.white_captures, 0)

    def test_suicide_is_rejected_without_changing_state(self) -> None:
        state = state_with_stones(
            {
                (0, 1): WHITE,
                (1, 0): WHITE,
                (1, 2): WHITE,
                (2, 1): WHITE,
            }
        )
        snapshot = serialize_state(state)
        with self.assertRaises(IllegalMoveError) as raised:
            step(state, (1, 1))
        self.assertEqual(raised.exception.reason, "suicide")
        self.assertEqual(serialize_state(state), snapshot)

    def test_capture_that_repeats_position_is_ko_repetition(self) -> None:
        original = state_with_stones(
            {
                (1, 2): BLACK,
                (2, 1): BLACK,
                (2, 2): WHITE,
                (2, 3): BLACK,
                (3, 1): WHITE,
                (3, 3): WHITE,
                (4, 2): WHITE,
            },
            to_play=BLACK,
        )
        captured = step(original, (3, 2))
        self.assertEqual(captured.stone_at(2, 2), EMPTY)
        self.assertEqual(captured.black_captures, 1)
        with self.assertRaises(IllegalMoveError) as raised:
            step(captured, (2, 2))
        self.assertEqual(raised.exception.reason, "repetition")
        self.assertEqual(captured.stone_at(3, 2), BLACK)

    def test_two_consecutive_passes_end_game(self) -> None:
        state = step(initial_state(), PASS)
        self.assertFalse(state.game_over)
        self.assertEqual(state.consecutive_passes, 1)
        state = step(state, "pass")
        self.assertTrue(state.game_over)
        self.assertEqual(legal_moves(state), ())
        with self.assertRaises(IllegalMoveError) as raised:
            step(state, (0, 0))
        self.assertEqual(raised.exception.reason, "game_over")

    def test_stone_move_resets_pass_count(self) -> None:
        state = step(initial_state(), PASS)
        state = step(state, (0, 0))
        self.assertEqual(state.consecutive_passes, 0)
        self.assertFalse(state.game_over)

    def test_chinese_area_scoring_counts_stones_and_enclosed_points(self) -> None:
        state = state_with_stones(
            {
                (1, 2): BLACK,
                (2, 1): BLACK,
                (2, 3): BLACK,
                (3, 2): BLACK,
                (5, 6): WHITE,
                (6, 5): WHITE,
                (6, 7): WHITE,
                (7, 6): WHITE,
            },
            komi=0.0,
        )
        result = score_area(state)
        self.assertEqual(result.black_stones, 4)
        self.assertEqual(result.white_stones, 4)
        self.assertEqual(result.black_territory, 1)
        self.assertEqual(result.white_territory, 1)
        self.assertEqual(result.black_total, 5.0)
        self.assertEqual(result.white_total, 5.0)
        self.assertEqual(result.winner, EMPTY)

    def test_state_serialization_round_trip_is_stable(self) -> None:
        state = initial_state()
        for move in ((0, 0), (1, 0), PASS, (0, 1)):
            state = step(state, move)
        payload = serialize_state(state)
        restored = deserialize_state(payload)
        self.assertEqual(restored, state)
        self.assertEqual(serialize_state(restored), payload)
        self.assertEqual(legal_moves(restored), legal_moves(state))

    def test_state_is_immutable(self) -> None:
        state = initial_state()
        with self.assertRaises(FrozenInstanceError):
            state.to_play = WHITE  # type: ignore[misc]
        with self.assertRaises(TypeError):
            state.board[0] = BLACK  # type: ignore[index]

    def test_stateful_engine_keeps_state_after_illegal_move(self) -> None:
        engine = GoEngine()
        first_state = engine.step((0, 0))
        self.assertIs(engine.state, first_state)
        with self.assertRaises(IllegalMoveError):
            engine.step((0, 0))
        self.assertIs(engine.state, first_state)
        self.assertEqual(engine.state.move_number, 1)


if __name__ == "__main__":
    unittest.main()
