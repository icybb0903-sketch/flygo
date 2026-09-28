"""Tests for the Stage 2 human-versus-baseline Go session."""

from __future__ import annotations

import json
import unittest

from src.go_engine import BLACK, BOARD_POINTS, EMPTY, WHITE, GoState, IllegalMoveError
from src.go_session import (
    GoSession,
    SessionTurnError,
    StaleSessionError,
    select_capture_first_move,
)


def state_with_stones(stones: dict[tuple[int, int], int], *, to_play: int = WHITE) -> GoState:
    board = [EMPTY] * BOARD_POINTS
    for (row, column), color in stones.items():
        board[row * 9 + column] = color
    frozen = tuple(board)
    return GoState(
        board=frozen,
        to_play=to_play,
        black_captures=0,
        white_captures=0,
        consecutive_passes=0,
        move_number=0,
        position_history=(frozen,),
    )


class GoSessionTests(unittest.TestCase):
    @staticmethod
    def neural_provenance() -> dict[str, object]:
        return {
            "controller": "malecns-neural-policy",
            "policy_version": "test-policy-v1",
            "checkpoint_hash": "c" * 64,
            "decision_hash": "d" * 64,
            "frame_id": "f" * 64,
            "board_hash": "b" * 64,
            "encoding_hash": "e" * 64,
            "baseline_controller_called": False,
            "fallback_used": False,
        }

    def test_human_black_then_bot_white_and_turn_returns_to_human(self) -> None:
        session = GoSession()
        after_human = session.human_move((4, 4))
        self.assertEqual(after_human["to_play"], "white")
        self.assertEqual(after_human["board"][4][4], BLACK)
        after_bot = session.bot_move()
        self.assertEqual(after_bot["to_play"], "black")
        self.assertEqual(after_bot["move_number"], 2)
        self.assertEqual(after_bot["history_length"], 2)
        self.assertEqual(after_bot["last_move"]["actor"], "bot")
        self.assertEqual(after_bot["last_move"]["color"], "white")

    def test_wrong_turn_and_illegal_move_do_not_change_state(self) -> None:
        session = GoSession()
        initial = session.snapshot()
        with self.assertRaises(SessionTurnError):
            session.bot_move()
        self.assertEqual(session.snapshot(), initial)

        session.human_move((0, 0))
        before_illegal = session.snapshot()
        with self.assertRaises(SessionTurnError):
            session.human_move((0, 1))
        self.assertEqual(session.snapshot(), before_illegal)

        session.bot_move()
        before_occupied = session.snapshot()
        with self.assertRaises(IllegalMoveError):
            session.human_move((0, 0))
        self.assertEqual(session.snapshot(), before_occupied)

    def test_baseline_is_deterministic_and_prefers_capture(self) -> None:
        quiet = state_with_stones({(4, 4): BLACK})
        self.assertEqual(select_capture_first_move(quiet), (3, 4))
        self.assertEqual(select_capture_first_move(quiet), (3, 4))

        capture = state_with_stones(
            {
                (1, 1): BLACK,
                (0, 1): WHITE,
                (1, 0): WHITE,
                (2, 1): WHITE,
                (4, 4): BLACK,
            }
        )
        self.assertEqual(select_capture_first_move(capture), (1, 2))

    def test_undo_completed_round_and_pending_human_move(self) -> None:
        session = GoSession()
        session.human_move((4, 4))
        session.bot_move()
        undone = session.undo_round()
        self.assertEqual(undone["move_number"], 0)
        self.assertEqual(undone["history_length"], 0)
        self.assertTrue(all(point == EMPTY for row in undone["board"] for point in row))

        session.human_move((2, 2))
        pending_undo = session.undo_round()
        self.assertEqual(pending_undo["move_number"], 0)
        self.assertEqual(pending_undo["to_play"], "black")
        self.assertIsNone(pending_undo["last_move"])

    def test_reset_clears_board_history_captures_and_passes(self) -> None:
        session = GoSession()
        session.human_move(None)
        session.bot_move()
        reset = session.reset()
        self.assertEqual(reset["move_number"], 0)
        self.assertEqual(reset["captures"], {"black": 0, "white": 0})
        self.assertEqual(reset["passes"], {"black": 0, "white": 0, "consecutive": 0})
        self.assertEqual(reset["history"], [])
        self.assertFalse(reset["game_over"])

    def test_human_pass_bot_pass_ends_game_and_finalizes_score(self) -> None:
        session = GoSession()
        after_human = session.human_move(None)
        self.assertEqual(after_human["passes"]["consecutive"], 1)
        final = session.bot_move()
        self.assertTrue(final["game_over"])
        self.assertTrue(final["score"]["final"])
        self.assertEqual(final["passes"]["consecutive"], 2)
        self.assertEqual(final["last_move"]["actor"], "bot")
        self.assertTrue(final["last_move"]["is_pass"])
        self.assertEqual(final["score"]["winner"], WHITE)
        self.assertEqual(final["score"]["white_total"], 7.5)
        self.assertEqual(final["end_reason"], "two_consecutive_passes")
        self.assertEqual(final["result"]["winner"], "white")
        self.assertEqual(final["result"]["margin"], 7.5)
        self.assertEqual(final["strategy"]["current_turn"], "game-over")
        with self.assertRaises(IllegalMoveError) as raised:
            session.bot_move()
        self.assertEqual(raised.exception.reason, "game_over")

    def test_human_resignation_is_terminal_auditable_and_undoable(self) -> None:
        session = GoSession()
        session.human_move((4, 4))
        session.bot_move()
        before = session.snapshot()

        resigned = session.human_resign()
        self.assertTrue(resigned["game_over"])
        self.assertEqual(resigned["end_reason"], "resignation")
        self.assertEqual(resigned["result"]["winner"], "white")
        self.assertIsNone(resigned["result"]["margin"])
        self.assertFalse(resigned["score"]["final"])
        self.assertEqual(resigned["termination"]["resigned_by"], "black")
        self.assertEqual(resigned["move_number"], before["move_number"])
        self.assertEqual(resigned["board"], before["board"])
        with self.assertRaises(IllegalMoveError) as raised:
            session.human_move((0, 0))
        self.assertEqual(raised.exception.reason, "game_over")

        restored = session.undo_round()
        self.assertFalse(restored["game_over"])
        self.assertIsNone(restored["termination"])
        self.assertEqual(restored["board"], before["board"])

    def test_reset_clears_resignation(self) -> None:
        session = GoSession()
        session.human_resign()
        reset = session.reset()
        self.assertFalse(reset["game_over"])
        self.assertIsNone(reset["end_reason"])
        self.assertIsNone(reset["termination"])

    def test_snapshot_is_json_safe_detached_and_truthfully_labels_policy(self) -> None:
        session = GoSession()
        session.human_move((4, 4))
        snapshot = session.snapshot()
        encoded = json.dumps(snapshot, ensure_ascii=False, sort_keys=True)
        decoded = json.loads(encoded)
        self.assertEqual(decoded, snapshot)
        self.assertEqual(len(snapshot["board"]), 9)
        self.assertTrue(all(len(row) == 9 for row in snapshot["board"]))
        self.assertFalse(snapshot["policy"]["male_cns_used"])
        self.assertEqual(snapshot["policy"]["controller"], "baseline")

        snapshot["board"][4][4] = EMPTY
        snapshot["history"].clear()
        fresh = session.snapshot()
        self.assertEqual(fresh["board"][4][4], BLACK)
        self.assertEqual(fresh["history_length"], 1)

    def test_external_policy_commit_records_truthful_provenance_and_undoes_round(self) -> None:
        session = GoSession()
        session.human_move((4, 4))
        expected = session.external_policy_state()
        snapshot = session.commit_external_policy_move(
            expected, (3, 4), self.neural_provenance()
        )
        self.assertEqual(snapshot["last_move"]["actor"], "neural_policy")
        self.assertEqual(snapshot["last_move"]["controller"], "malecns-neural-policy")
        self.assertTrue(snapshot["policy"]["male_cns_used"])
        self.assertFalse(snapshot["policy"]["baseline_controller_called"])
        self.assertFalse(snapshot["policy"]["fallback_used"])
        self.assertEqual(session.undo_round()["move_number"], 0)

    def test_external_neural_pass_after_human_pass_finishes_naturally(self) -> None:
        session = GoSession()
        session.human_move(None)
        expected = session.external_policy_state()
        provenance = self.neural_provenance()
        provenance["action_index"] = 81
        final = session.commit_external_policy_move(expected, None, provenance)

        self.assertTrue(final["game_over"])
        self.assertEqual(final["end_reason"], "two_consecutive_passes")
        self.assertEqual(final["last_move"]["actor"], "neural_policy")
        self.assertTrue(final["last_move"]["is_pass"])
        self.assertEqual(final["last_move"]["provenance"]["action_index"], 81)
        self.assertEqual(final["result"]["winner"], "white")

    def test_external_policy_stale_or_bad_provenance_leaves_board_unchanged(self) -> None:
        session = GoSession()
        session.human_move((4, 4))
        expected = session.external_policy_state()
        bad = self.neural_provenance()
        bad["fallback_used"] = True
        before = session.snapshot()
        with self.assertRaisesRegex(ValueError, "fail-closed"):
            session.commit_external_policy_move(expected, (3, 4), bad)
        self.assertEqual(session.snapshot(), before)

        session.reset()
        with self.assertRaises(StaleSessionError):
            session.commit_external_policy_move(
                expected, (3, 4), self.neural_provenance()
            )
        self.assertEqual(session.snapshot()["move_number"], 0)


if __name__ == "__main__":
    unittest.main()
