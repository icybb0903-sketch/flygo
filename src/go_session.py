"""Thread-safe human-versus-controller session for the 9x9 Go demo.

The Stage 2 capture-first controller remains available as an explicitly
labeled compatibility path.  Stage 4 can atomically commit a separately
computed neural-policy move with complete provenance and stale-state checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
import copy
from typing import Any, Final, Mapping

from .go_engine import (
    BLACK,
    BOARD_SIZE,
    PASS,
    WHITE,
    GoState,
    IllegalMoveError,
    Move,
    initial_state,
    legal_moves,
    score_area,
    step,
)


SESSION_SCHEMA: Final = "go-session-v1"
BASELINE_POLICY_ID: Final = "deterministic-capture-first-v1"


class SessionTurnError(ValueError):
    """Raised when a human or bot action is requested on the wrong turn."""


class StaleSessionError(RuntimeError):
    """Raised when an externally computed decision targets an old position."""


def _color_name(color: int) -> str:
    return "black" if color == BLACK else "white"


def _json_move(move: Move) -> list[int] | None:
    return None if move is PASS else [move[0], move[1]]


def _captures_made(before: GoState, after: GoState, color: int) -> int:
    if color == BLACK:
        return after.black_captures - before.black_captures
    return after.white_captures - before.white_captures


def select_capture_first_move(state: GoState) -> Move:
    """Choose White's move using the documented deterministic baseline.

    Legal point moves are ranked by most immediate captures, then squared
    distance from the 9x9 centre, then row and column.  Pass is selected only
    when there is no legal point move.
    """
    if not isinstance(state, GoState):
        raise TypeError("state must be a GoState")
    if state.game_over:
        raise IllegalMoveError("game_over", PASS, "the game has already ended")

    ranked: list[tuple[int, int, int, int, Move]] = []
    centre = (BOARD_SIZE - 1) // 2
    for move in legal_moves(state, include_pass=False):
        candidate = step(state, move)
        captured = _captures_made(state, candidate, state.to_play)
        row, column = move
        distance = (row - centre) ** 2 + (column - centre) ** 2
        ranked.append((-captured, distance, row, column, move))
    if not ranked:
        return PASS
    ranked.sort()
    return ranked[0][-1]


@dataclass(frozen=True, slots=True)
class _MoveRecord:
    actor: str
    color: int
    move: Move
    captures: int
    before: GoState
    after: GoState
    provenance: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "move_number": self.after.move_number,
            "actor": self.actor,
            "color": _color_name(self.color),
            "color_code": self.color,
            "move": _json_move(self.move),
            "is_pass": self.move is PASS,
            "captures": self.captures,
        }
        if self.provenance is not None:
            result["provenance"] = copy.deepcopy(self.provenance)
            result["controller"] = self.provenance.get("controller")
        elif self.actor == "human":
            result["controller"] = "human"
        return result


class GoSession:
    """A locked 9x9 session with Black human and auditable White controllers."""

    def __init__(self, *, komi: float = 7.5) -> None:
        self._lock = RLock()
        self._komi = float(komi)
        self._state = initial_state(komi=self._komi)
        self._history: list[_MoveRecord] = []
        self._resignation: dict[str, Any] | None = None

    @property
    def state(self) -> GoState:
        """Return the current immutable engine state."""
        with self._lock:
            return self._state

    def reset(self) -> dict[str, Any]:
        """Start a fresh game and discard the session move history."""
        with self._lock:
            self._state = initial_state(komi=self._komi)
            self._history.clear()
            self._resignation = None
            return self._snapshot_unlocked()

    def human_move(self, move: object) -> dict[str, Any]:
        """Apply one Black move, returning a JSON-safe session snapshot."""
        with self._lock:
            self._ensure_active_unlocked(move)
            if self._state.to_play != BLACK:
                raise SessionTurnError("it is not the human (Black) turn")
            return self._apply_unlocked("human", move)

    def human_resign(self) -> dict[str, Any]:
        """End the game by recording Black's resignation.

        Resignation is a terminal session event, not a Go board move: it does
        not change the board, move number, captures, or positional-superko
        history.  The canonical winner and end reason are exposed through the
        snapshot's ``result`` and ``termination`` objects.
        """
        with self._lock:
            self._ensure_active_unlocked("resign")
            self._resignation = {
                "reason": "resignation",
                "actor": "human",
                "resigned_by": "black",
                "resigned_by_code": BLACK,
                "winner": "white",
                "winner_code": WHITE,
            }
            return self._snapshot_unlocked()

    def bot_move(self) -> dict[str, Any]:
        """Apply one deterministic baseline move for White."""
        with self._lock:
            self._ensure_active_unlocked(PASS)
            if self._state.to_play != WHITE:
                raise SessionTurnError("it is not the baseline bot (White) turn")
            # A human pass is the standard signal to begin scoring.  White
            # replies with a pass so two consecutive passes end the game and
            # make the engine's Chinese area score final.
            move = (
                PASS
                if self._state.consecutive_passes == 1
                else select_capture_first_move(self._state)
            )
            return self._apply_unlocked(
                "bot",
                move,
                provenance={
                    "controller": "baseline",
                    "policy_id": BASELINE_POLICY_ID,
                    "male_cns_used": False,
                    "baseline_controller_called": True,
                },
            )

    def external_policy_state(self) -> GoState:
        """Return the immutable White-to-play state for external computation.

        The caller must later pass this exact position to
        :meth:`commit_external_policy_move`; the commit fails atomically if any
        request changed the session while the policy was running.
        """
        with self._lock:
            self._ensure_active_unlocked(PASS)
            if self._state.to_play != WHITE:
                raise SessionTurnError("it is not an external policy (White) turn")
            if self._state.game_over:
                raise IllegalMoveError("game_over", PASS, "the game has already ended")
            return self._state

    def commit_external_policy_move(
        self,
        expected_state: GoState,
        move: object,
        provenance: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Atomically commit a verified external White policy decision.

        ``expected_state`` is the immutable state used to build the policy
        input.  A reset, undo, human move, or other controller commit between
        computation and this method makes the decision stale and leaves the
        session unchanged.
        """
        if not isinstance(expected_state, GoState):
            raise TypeError("expected_state must be a GoState")
        if not isinstance(provenance, Mapping):
            raise TypeError("provenance must be a mapping")
        try:
            safe_provenance = json_round_trip(dict(provenance))
        except (TypeError, ValueError) as exc:
            raise ValueError("provenance must be finite JSON data") from exc
        required = {
            "controller",
            "policy_version",
            "checkpoint_hash",
            "decision_hash",
            "frame_id",
            "board_hash",
            "encoding_hash",
            "baseline_controller_called",
            "fallback_used",
        }
        if not required.issubset(safe_provenance):
            raise ValueError("external policy provenance is incomplete")
        if (
            safe_provenance["controller"] != "malecns-neural-policy"
            or safe_provenance["baseline_controller_called"] is not False
            or safe_provenance["fallback_used"] is not False
        ):
            raise ValueError("external policy provenance does not satisfy fail-closed contract")

        with self._lock:
            if self._state != expected_state:
                raise StaleSessionError("the Go position changed while the policy was running")
            self._ensure_active_unlocked(move)
            if self._state.to_play != WHITE:
                raise SessionTurnError("it is not an external policy (White) turn")
            return self._apply_unlocked(
                "neural_policy", move, provenance=safe_provenance
            )

    def undo_round(self) -> dict[str, Any]:
        """Undo the latest human/bot round, or a lone pending human move.

        A completed round ends in a bot record, so both that record and the
        immediately preceding human record are removed.  If only the human
        has moved, exactly that one move is removed.
        """
        with self._lock:
            if self._resignation is not None:
                self._resignation = None
                return self._snapshot_unlocked()
            if not self._history:
                return self._snapshot_unlocked()

            latest = self._history.pop()
            self._state = latest.before
            if latest.actor in {"bot", "neural_policy"} and self._history and self._history[-1].actor == "human":
                human = self._history.pop()
                self._state = human.before
            return self._snapshot_unlocked()

    def _ensure_active_unlocked(self, move: object) -> None:
        if self._state.game_over or self._resignation is not None:
            raise IllegalMoveError("game_over", move, "the game has already ended")

    def snapshot(self) -> dict[str, Any]:
        """Return a detached JSON-compatible view of the complete session."""
        with self._lock:
            return self._snapshot_unlocked()

    def _apply_unlocked(
        self,
        actor: str,
        move: object,
        *,
        provenance: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        before = self._state
        after = step(before, move)
        normalized: Move = PASS if move is PASS or move == "pass" else tuple(move)  # type: ignore[arg-type]
        record = _MoveRecord(
            actor=actor,
            color=before.to_play,
            move=normalized,
            captures=_captures_made(before, after, before.to_play),
            before=before,
            after=after,
            provenance=provenance,
        )
        self._state = after
        self._history.append(record)
        return self._snapshot_unlocked()

    def _snapshot_unlocked(self) -> dict[str, Any]:
        state = self._state
        score = score_area(state)
        game_over = state.game_over or self._resignation is not None
        if self._resignation is not None:
            termination = copy.deepcopy(self._resignation)
            result_winner = WHITE
            result_margin: float | None = None
            result_reason: str | None = "resignation"
        elif state.game_over:
            result_winner = score.winner
            result_margin = score.margin
            result_reason = "two_consecutive_passes"
            termination = {
                "reason": result_reason,
                "actor": None,
                "resigned_by": None,
                "resigned_by_code": None,
                "winner": {0: "draw", BLACK: "black", WHITE: "white"}[result_winner],
                "winner_code": result_winner,
            }
        else:
            result_winner = None
            result_margin = None
            result_reason = None
            termination = None
        black_passes = sum(record.move is PASS and record.color == BLACK for record in self._history)
        white_passes = sum(record.move is PASS and record.color == WHITE for record in self._history)
        history = [record.to_dict() for record in self._history]
        last_policy_record = next(
            (record for record in reversed(self._history) if record.provenance is not None),
            None,
        )
        last_decision = (
            copy.deepcopy(last_policy_record.provenance)
            if last_policy_record is not None
            else None
        )
        if last_decision is not None and last_decision.get("controller") == "malecns-neural-policy":
            policy = {
                "id": last_decision.get("policy_version"),
                "controller": "malecns-neural-policy",
                "display_name": "Trained MaleCNS-frame linear readout",
                "deterministic": True,
                "male_cns_used": True,
                "checkpoint_hash": last_decision.get("checkpoint_hash"),
                "teacher_accessed_at_runtime": False,
                "baseline_controller_called": False,
                "fallback_used": False,
            }
        else:
            policy = {
                "id": BASELINE_POLICY_ID,
                "controller": "baseline",
                "display_name": "Deterministic capture-first baseline",
                "deterministic": True,
                "male_cns_used": False,
                "selection_order": [
                    "reply_pass_after_human_pass",
                    "most_immediate_captures",
                    "nearest_board_center",
                    "lowest_row",
                    "lowest_column",
                    "pass_only_if_no_legal_point",
                ],
            }
        return {
            "schema": SESSION_SCHEMA,
            "board_size": BOARD_SIZE,
            "board": [
                list(state.board[row * BOARD_SIZE : (row + 1) * BOARD_SIZE])
                for row in range(BOARD_SIZE)
            ],
            "to_play": _color_name(state.to_play),
            "to_play_code": state.to_play,
            "move_number": state.move_number,
            "captures": {
                "black": state.black_captures,
                "white": state.white_captures,
            },
            "passes": {
                "black": black_passes,
                "white": white_passes,
                "consecutive": state.consecutive_passes,
            },
            "game_over": game_over,
            "end_reason": result_reason,
            "termination": termination,
            "result": {
                "final": game_over,
                "reason": result_reason,
                "winner": (
                    None
                    if result_winner is None
                    else {0: "draw", BLACK: "black", WHITE: "white"}[result_winner]
                ),
                "winner_code": result_winner,
                "margin": result_margin,
            },
            "score": {
                **score.to_dict(),
                # Area totals are decisive only after two passes.  On
                # resignation, ``result`` is final while this board estimate
                # remains explicitly non-final.
                "final": state.game_over,
            },
            "history_length": len(history),
            "history": history,
            "last_move": history[-1] if history else None,
            "strategy": {
                "current_turn": (
                    "game-over"
                    if game_over
                    else (
                        "human-black"
                        if state.to_play == BLACK
                        else "white-controller-pending"
                    )
                ),
                "last_controller": (
                    last_decision.get("controller") if last_decision is not None else None
                ),
                "last_decision": last_decision,
            },
            "policy": policy,
        }


def json_round_trip(value: object) -> Any:
    """Detach JSON provenance while rejecting NaN/Infinity and custom types."""
    import json

    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


__all__ = [
    "BASELINE_POLICY_ID",
    "GoSession",
    "SESSION_SCHEMA",
    "SessionTurnError",
    "StaleSessionError",
    "select_capture_first_move",
]
