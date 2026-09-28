"""Deterministic, dependency-free 9x9 Go rules for an external policy loop.

The functional API is deliberately small: create an :func:`initial_state`,
inspect :func:`legal_moves`, and apply one with :func:`step`.  ``GoState`` is
immutable, so trying an illegal move cannot partially modify a position.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from typing import Any, Iterable, Mapping, Sequence, TypeAlias


BOARD_SIZE = 9
BOARD_POINTS = BOARD_SIZE * BOARD_SIZE
EMPTY = 0
BLACK = 1
WHITE = 2
PASS = None
STATE_SCHEMA = "go-state-v1"
DEFAULT_KOMI = 7.5

Coordinate: TypeAlias = tuple[int, int]
Move: TypeAlias = Coordinate | None
Board: TypeAlias = tuple[int, ...]


class GoStateError(ValueError):
    """Raised when a serialized or directly constructed state is invalid."""


class IllegalMoveError(ValueError):
    """Raised when a move cannot legally be applied to a state."""

    def __init__(self, reason: str, move: object, message: str) -> None:
        super().__init__(message)
        self.reason = reason
        self.move = move


def _is_plain_int(value: object) -> bool:
    return type(value) is int


def _validate_board(board: object, *, name: str) -> Board:
    if not isinstance(board, tuple) or len(board) != BOARD_POINTS:
        raise GoStateError(f"{name} must be a tuple of {BOARD_POINTS} points")
    if any(not _is_plain_int(point) or point not in (EMPTY, BLACK, WHITE) for point in board):
        raise GoStateError(f"{name} may contain only EMPTY, BLACK, and WHITE")
    return board


@dataclass(frozen=True, slots=True)
class GoState:
    """An immutable and exactly replay-safe Go position.

    ``position_history`` stores full board tuples rather than hashes.  Stone
    moves are rejected when they recreate any earlier board, implementing
    positional superko without collision risk.  Passes may repeat a board.
    """

    board: Board
    to_play: int
    black_captures: int
    white_captures: int
    consecutive_passes: int
    move_number: int
    position_history: tuple[Board, ...]
    komi: float = DEFAULT_KOMI

    def __post_init__(self) -> None:
        _validate_board(self.board, name="board")
        if self.to_play not in (BLACK, WHITE) or not _is_plain_int(self.to_play):
            raise GoStateError("to_play must be BLACK or WHITE")
        for name, value in (
            ("black_captures", self.black_captures),
            ("white_captures", self.white_captures),
            ("move_number", self.move_number),
        ):
            if not _is_plain_int(value) or value < 0:
                raise GoStateError(f"{name} must be a nonnegative integer")
        if not _is_plain_int(self.consecutive_passes) or not 0 <= self.consecutive_passes <= 2:
            raise GoStateError("consecutive_passes must be 0, 1, or 2")
        if not isinstance(self.position_history, tuple) or not self.position_history:
            raise GoStateError("position_history must be a nonempty tuple")
        for index, historical_board in enumerate(self.position_history):
            _validate_board(historical_board, name=f"position_history[{index}]")
        if self.position_history[-1] != self.board:
            raise GoStateError("the last history board must equal board")
        if len(self.position_history) != self.move_number + 1:
            raise GoStateError("position_history length must equal move_number + 1")
        if isinstance(self.komi, bool) or not isinstance(self.komi, (int, float)):
            raise GoStateError("komi must be a finite number")
        if not math.isfinite(float(self.komi)):
            raise GoStateError("komi must be a finite number")

    @property
    def game_over(self) -> bool:
        """Whether two consecutive passes have ended the game."""
        return self.consecutive_passes == 2

    def stone_at(self, row: int, column: int) -> int:
        """Return the point value at a zero-based coordinate."""
        if not _valid_coordinate((row, column)):
            raise IndexError("board coordinate is outside the 9x9 board")
        return self.board[_index(row, column)]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible, versioned representation."""
        return {
            "schema": STATE_SCHEMA,
            "board_size": BOARD_SIZE,
            "board": list(self.board),
            "to_play": self.to_play,
            "black_captures": self.black_captures,
            "white_captures": self.white_captures,
            "consecutive_passes": self.consecutive_passes,
            "move_number": self.move_number,
            "position_history": [list(board) for board in self.position_history],
            "komi": float(self.komi),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GoState":
        """Validate and restore a state produced by :meth:`to_dict`."""
        if not isinstance(data, Mapping):
            raise GoStateError("serialized state must be an object")
        if data.get("schema") != STATE_SCHEMA:
            raise GoStateError(f"state schema must be {STATE_SCHEMA!r}")
        if data.get("board_size") != BOARD_SIZE:
            raise GoStateError(f"board_size must be {BOARD_SIZE}")
        required = (
            "board",
            "to_play",
            "black_captures",
            "white_captures",
            "consecutive_passes",
            "move_number",
            "position_history",
            "komi",
        )
        missing = [key for key in required if key not in data]
        if missing:
            raise GoStateError(f"serialized state is missing: {', '.join(missing)}")
        try:
            board = tuple(data["board"])
            history = tuple(tuple(item) for item in data["position_history"])
        except TypeError as exc:
            raise GoStateError("board and position_history must be sequences") from exc
        return cls(
            board=board,
            to_play=data["to_play"],
            black_captures=data["black_captures"],
            white_captures=data["white_captures"],
            consecutive_passes=data["consecutive_passes"],
            move_number=data["move_number"],
            position_history=history,
            komi=data["komi"],
        )


@dataclass(frozen=True, slots=True)
class AreaScore:
    """Chinese area score: living stones plus surrounded empty points."""

    black_stones: int
    white_stones: int
    black_territory: int
    white_territory: int
    komi: float

    @property
    def black_total(self) -> float:
        return float(self.black_stones + self.black_territory)

    @property
    def white_total(self) -> float:
        return float(self.white_stones + self.white_territory) + self.komi

    @property
    def winner(self) -> int:
        if self.black_total > self.white_total:
            return BLACK
        if self.white_total > self.black_total:
            return WHITE
        return EMPTY

    @property
    def margin(self) -> float:
        """Absolute winning margin, or zero for a draw."""
        return abs(self.black_total - self.white_total)

    def to_dict(self) -> dict[str, int | float]:
        return {
            "black_stones": self.black_stones,
            "white_stones": self.white_stones,
            "black_territory": self.black_territory,
            "white_territory": self.white_territory,
            "komi": self.komi,
            "black_total": self.black_total,
            "white_total": self.white_total,
            "winner": self.winner,
            "margin": self.margin,
        }


def initial_state(*, komi: float = DEFAULT_KOMI) -> GoState:
    """Create an empty 9x9 game with Black to play."""
    board = (EMPTY,) * BOARD_POINTS
    return GoState(
        board=board,
        to_play=BLACK,
        black_captures=0,
        white_captures=0,
        consecutive_passes=0,
        move_number=0,
        position_history=(board,),
        komi=komi,
    )


new_game = initial_state


def _index(row: int, column: int) -> int:
    return row * BOARD_SIZE + column


def _coordinate(index: int) -> Coordinate:
    return divmod(index, BOARD_SIZE)


def _valid_coordinate(move: object) -> bool:
    return (
        isinstance(move, tuple)
        and len(move) == 2
        and all(_is_plain_int(part) for part in move)
        and 0 <= move[0] < BOARD_SIZE
        and 0 <= move[1] < BOARD_SIZE
    )


def _neighbors(index: int) -> tuple[int, ...]:
    row, column = _coordinate(index)
    result: list[int] = []
    if row > 0:
        result.append(index - BOARD_SIZE)
    if column > 0:
        result.append(index - 1)
    if column + 1 < BOARD_SIZE:
        result.append(index + 1)
    if row + 1 < BOARD_SIZE:
        result.append(index + BOARD_SIZE)
    return tuple(result)


def _group_and_liberties(board: Sequence[int], start: int) -> tuple[set[int], set[int]]:
    color = board[start]
    group = {start}
    liberties: set[int] = set()
    pending = [start]
    while pending:
        point = pending.pop()
        for neighbor in _neighbors(point):
            value = board[neighbor]
            if value == EMPTY:
                liberties.add(neighbor)
            elif value == color and neighbor not in group:
                group.add(neighbor)
                pending.append(neighbor)
    return group, liberties


def _other(color: int) -> int:
    return WHITE if color == BLACK else BLACK


def _normalize_move(move: object) -> Move:
    if move is PASS or move == "pass":
        return PASS
    if isinstance(move, list):
        move = tuple(move)
    if not _valid_coordinate(move):
        raise IllegalMoveError(
            "invalid_move",
            move,
            "move must be None/'pass' or a zero-based (row, column) on the 9x9 board",
        )
    return move


def step(state: GoState, move: object) -> GoState:
    """Apply one move and return a new state.

    Ordinary moves use positional superko.  ``None`` and ``"pass"`` are pass
    moves and are exempt from repetition checking.  Any rejection occurs
    before a new state is returned, leaving the supplied state unchanged.
    """
    if not isinstance(state, GoState):
        raise TypeError("state must be a GoState")
    if state.game_over:
        raise IllegalMoveError("game_over", move, "the game has already ended")
    normalized = _normalize_move(move)
    next_player = _other(state.to_play)
    next_move_number = state.move_number + 1

    if normalized is PASS:
        return replace(
            state,
            to_play=next_player,
            consecutive_passes=state.consecutive_passes + 1,
            move_number=next_move_number,
            position_history=state.position_history + (state.board,),
        )

    row, column = normalized
    played_at = _index(row, column)
    if state.board[played_at] != EMPTY:
        raise IllegalMoveError("occupied", normalized, f"point {normalized!r} is occupied")

    board = list(state.board)
    board[played_at] = state.to_play
    opponent = _other(state.to_play)
    captured: set[int] = set()
    examined: set[int] = set()
    for neighbor in _neighbors(played_at):
        if board[neighbor] != opponent or neighbor in examined:
            continue
        group, liberties = _group_and_liberties(board, neighbor)
        examined.update(group)
        if not liberties:
            captured.update(group)
    for point in captured:
        board[point] = EMPTY

    _, own_liberties = _group_and_liberties(board, played_at)
    if not own_liberties:
        raise IllegalMoveError("suicide", normalized, f"point {normalized!r} is suicide")

    next_board = tuple(board)
    if next_board in state.position_history:
        raise IllegalMoveError(
            "repetition",
            normalized,
            f"point {normalized!r} would repeat an earlier board position",
        )

    black_captures = state.black_captures
    white_captures = state.white_captures
    if state.to_play == BLACK:
        black_captures += len(captured)
    else:
        white_captures += len(captured)
    return GoState(
        board=next_board,
        to_play=next_player,
        black_captures=black_captures,
        white_captures=white_captures,
        consecutive_passes=0,
        move_number=next_move_number,
        position_history=state.position_history + (next_board,),
        komi=state.komi,
    )


def is_legal(state: GoState, move: object) -> bool:
    """Return whether ``move`` can be applied, without changing ``state``."""
    try:
        step(state, move)
    except IllegalMoveError:
        return False
    return True


def legal_moves(state: GoState, *, include_pass: bool = True) -> tuple[Move, ...]:
    """Return legal moves in deterministic row-major order, with pass last."""
    if not isinstance(state, GoState):
        raise TypeError("state must be a GoState")
    if state.game_over:
        return ()
    moves: list[Move] = []
    for row in range(BOARD_SIZE):
        for column in range(BOARD_SIZE):
            move = (row, column)
            if state.board[_index(row, column)] == EMPTY and is_legal(state, move):
                moves.append(move)
    if include_pass:
        moves.append(PASS)
    return tuple(moves)


def score_area(state: GoState) -> AreaScore:
    """Score a position by Chinese area scoring, without dead-stone adjudication."""
    if not isinstance(state, GoState):
        raise TypeError("state must be a GoState")
    black_stones = state.board.count(BLACK)
    white_stones = state.board.count(WHITE)
    black_territory = 0
    white_territory = 0
    visited: set[int] = set()
    for start, value in enumerate(state.board):
        if value != EMPTY or start in visited:
            continue
        region = {start}
        border_colors: set[int] = set()
        pending = [start]
        visited.add(start)
        while pending:
            point = pending.pop()
            for neighbor in _neighbors(point):
                neighbor_value = state.board[neighbor]
                if neighbor_value == EMPTY and neighbor not in visited:
                    visited.add(neighbor)
                    region.add(neighbor)
                    pending.append(neighbor)
                elif neighbor_value in (BLACK, WHITE):
                    border_colors.add(neighbor_value)
        if border_colors == {BLACK}:
            black_territory += len(region)
        elif border_colors == {WHITE}:
            white_territory += len(region)
    return AreaScore(
        black_stones=black_stones,
        white_stones=white_stones,
        black_territory=black_territory,
        white_territory=white_territory,
        komi=float(state.komi),
    )


def serialize_state(state: GoState) -> str:
    """Serialize a state to stable, compact UTF-8 JSON text."""
    if not isinstance(state, GoState):
        raise TypeError("state must be a GoState")
    return json.dumps(state.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def deserialize_state(payload: str | bytes | bytearray) -> GoState:
    """Restore and validate JSON produced by :func:`serialize_state`."""
    try:
        data = json.loads(payload)
    except (json.JSONDecodeError, TypeError, UnicodeDecodeError) as exc:
        raise GoStateError("state payload is not valid JSON") from exc
    return GoState.from_dict(data)


class GoEngine:
    """Small stateful facade for callers that prefer an environment object."""

    def __init__(self, state: GoState | None = None, *, komi: float = DEFAULT_KOMI) -> None:
        self._state = initial_state(komi=komi) if state is None else state
        if not isinstance(self._state, GoState):
            raise TypeError("state must be a GoState")

    @property
    def state(self) -> GoState:
        return self._state

    def reset(self, *, komi: float | None = None) -> GoState:
        self._state = initial_state(komi=self._state.komi if komi is None else komi)
        return self._state

    def legal_moves(self, *, include_pass: bool = True) -> tuple[Move, ...]:
        return legal_moves(self._state, include_pass=include_pass)

    def step(self, move: object) -> GoState:
        next_state = step(self._state, move)
        self._state = next_state
        return next_state

    def score(self) -> AreaScore:
        return score_area(self._state)


__all__ = [
    "AreaScore",
    "BLACK",
    "BOARD_POINTS",
    "BOARD_SIZE",
    "DEFAULT_KOMI",
    "EMPTY",
    "GoEngine",
    "GoState",
    "GoStateError",
    "IllegalMoveError",
    "Move",
    "PASS",
    "STATE_SCHEMA",
    "WHITE",
    "deserialize_state",
    "initial_state",
    "is_legal",
    "legal_moves",
    "new_game",
    "score_area",
    "serialize_state",
    "step",
]
