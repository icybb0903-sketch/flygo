"""Deterministic engineering adapter from 9x9 Go to numeric features.

This module is an interface boundary for later neural experiments.  It is not
a biological sensory model and using it does not mean that MaleCNS has been
loaded, simulated, trained, or used to choose a move.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Final

from .go_engine import (
    BLACK,
    BOARD_POINTS,
    BOARD_SIZE,
    EMPTY,
    PASS,
    WHITE,
    GoState,
    Move,
    legal_moves,
)


ENCODING_SCHEMA: Final = "go-neural-encoding-v1"
ACTION_COUNT: Final = BOARD_POINTS + 1
PASS_ACTION_INDEX: Final = BOARD_POINTS
FEATURE_PLANES: Final = (
    "own_stone",
    "opponent_stone",
    "empty",
    "legal_point",
    "last_move",
)
SCALAR_FEATURES: Final = (
    "to_play_black",
    "to_play_white",
    "own_captures",
    "opponent_captures",
    "consecutive_passes",
    "last_move_was_pass",
    "pass_legal",
    "game_over",
    "move_number",
    "komi",
)
VECTOR_LENGTH: Final = len(FEATURE_PLANES) * BOARD_POINTS + len(SCALAR_FEATURES)
ADAPTER_LABEL: Final = "engineered_go_adapter_not_biological_sensory_system"


def _plain_int(value: object) -> bool:
    return type(value) is int


def move_to_action(move: object) -> int:
    """Map a zero-based board coordinate or pass to one of 82 actions."""
    if move is PASS or move == "pass":
        return PASS_ACTION_INDEX
    if isinstance(move, list):
        move = tuple(move)
    if (
        not isinstance(move, tuple)
        or len(move) != 2
        or not all(_plain_int(part) for part in move)
        or not all(0 <= part < BOARD_SIZE for part in move)
    ):
        raise ValueError("move must be None/'pass' or a zero-based coordinate on the 9x9 board")
    row, column = move
    return row * BOARD_SIZE + column


def action_to_move(action_index: object) -> Move:
    """Map an action index back to a board coordinate or pass."""
    if not _plain_int(action_index) or not 0 <= action_index < ACTION_COUNT:
        raise ValueError(f"action_index must be an integer from 0 through {ACTION_COUNT - 1}")
    if action_index == PASS_ACTION_INDEX:
        return PASS
    return divmod(action_index, BOARD_SIZE)


def legal_action_mask(state: GoState) -> list[int]:
    """Return an 82-element mask in row-major point order, followed by pass."""
    if not isinstance(state, GoState):
        raise TypeError("state must be a GoState")
    mask = [0] * ACTION_COUNT
    for move in legal_moves(state):
        mask[move_to_action(move)] = 1
    return mask


def _last_move_action(state: GoState) -> int | None:
    if state.move_number == 0:
        return None
    previous = state.position_history[-2]
    current = state.board
    if previous == current:
        return PASS_ACTION_INDEX

    previous_player = WHITE if state.to_play == BLACK else BLACK
    placed = [
        index
        for index, (before, after) in enumerate(zip(previous, current, strict=True))
        if before == EMPTY and after == previous_player
    ]
    if len(placed) != 1:
        raise ValueError(
            "cannot derive one last move from position_history; "
            "the final transition is not a valid single-stone placement or pass"
        )
    return placed[0]


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def board_hash(state: GoState) -> str:
    """Hash only the board geometry, independent of whose turn it is."""
    if not isinstance(state, GoState):
        raise TypeError("state must be a GoState")
    return _sha256({"board_size": BOARD_SIZE, "board": list(state.board)})


def encode_go_state(state: GoState) -> dict[str, Any]:
    """Encode ``state`` into a fixed-length, JSON-safe numeric contract.

    The five planes are flattened plane-major in row-major point order.  Own
    and opponent are relative to ``state.to_play``.  Capture values are also
    relative to the player to move; all other scalar meanings are absolute.
    """
    if not isinstance(state, GoState):
        raise TypeError("state must be a GoState")

    opponent = WHITE if state.to_play == BLACK else BLACK
    mask = legal_action_mask(state)
    last_action = _last_move_action(state)
    planes: list[list[float]] = [
        [1.0 if point == state.to_play else 0.0 for point in state.board],
        [1.0 if point == opponent else 0.0 for point in state.board],
        [1.0 if point == EMPTY else 0.0 for point in state.board],
        [float(value) for value in mask[:BOARD_POINTS]],
        [1.0 if last_action == index else 0.0 for index in range(BOARD_POINTS)],
    ]

    if state.to_play == BLACK:
        own_captures, opponent_captures = state.black_captures, state.white_captures
    else:
        own_captures, opponent_captures = state.white_captures, state.black_captures

    scalars = [
        1.0 if state.to_play == BLACK else 0.0,
        1.0 if state.to_play == WHITE else 0.0,
        float(own_captures),
        float(opponent_captures),
        float(state.consecutive_passes),
        1.0 if last_action == PASS_ACTION_INDEX else 0.0,
        float(mask[PASS_ACTION_INDEX]),
        1.0 if state.game_over else 0.0,
        float(state.move_number),
        float(state.komi),
    ]
    vector = [value for plane in planes for value in plane] + scalars
    if len(vector) != VECTOR_LENGTH:  # defensive guard for contract edits
        raise RuntimeError(f"encoding contract produced {len(vector)} values, expected {VECTOR_LENGTH}")

    payload: dict[str, Any] = {
        "schema": ENCODING_SCHEMA,
        "adapter": ADAPTER_LABEL,
        "male_cns_used": False,
        "board_size": BOARD_SIZE,
        "action_count": ACTION_COUNT,
        "pass_action_index": PASS_ACTION_INDEX,
        "feature_planes": list(FEATURE_PLANES),
        "scalar_features": list(SCALAR_FEATURES),
        "vector_length": VECTOR_LENGTH,
        "vector": vector,
        "legal_mask": mask,
        "last_move_action": last_action,
        "board_hash": board_hash(state),
    }
    payload["encoding_hash"] = _sha256(payload)
    return payload


__all__ = [
    "ACTION_COUNT",
    "ADAPTER_LABEL",
    "ENCODING_SCHEMA",
    "FEATURE_PLANES",
    "PASS_ACTION_INDEX",
    "SCALAR_FEATURES",
    "VECTOR_LENGTH",
    "action_to_move",
    "board_hash",
    "encode_go_state",
    "legal_action_mask",
    "move_to_action",
]
