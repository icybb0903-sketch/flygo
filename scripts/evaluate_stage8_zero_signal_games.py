"""Paired Go games with the trained readout's neural features set to zero.

This is a counterfactual control, not a MaleCNS simulation. It uses the same
checkpoint bias/weights, legal mask, rule-PASS gate, opponent RNG stream, game
seeds, and random opening states as an existing white-role neural report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_neural_signal_ablation import point_predictions
from scripts.evaluate_neural_go_games import (
    opening_random_stream_id, opponent_random_stream_id,
    select_legal_random_move,
)
from scripts.train_neural_go_readout import canonical_bytes, object_hash
from src.go_engine import BLACK, EMPTY, PASS, WHITE, initial_state, score_area, step
from src.go_neural_encoding import PASS_ACTION_INDEX, action_to_move, board_hash, encode_go_state
from src.malecns_policy import load_policy_checkpoint


SCHEMA = "malecns-go-stage8-zero-signal-games-v1"


def _reference(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    claimed = report.pop("report_sha256", None)
    if report.get("schema") != "malecns-go-stage7-production-white-v2":
        raise ValueError("reference is not a white-role evaluation report")
    if not isinstance(claimed, str) or object_hash(report) != claimed:
        raise ValueError("reference report hash mismatch")
    report["report_sha256"] = claimed
    return report


def play_zero_game(
    *, seed: int, max_moves: int, opening_plies: int,
    weights: np.ndarray, bias: np.ndarray,
) -> dict[str, Any]:
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(max_moves) is not int or max_moves <= 1:
        raise ValueError("max_moves must exceed one")
    if type(opening_plies) is not int or opening_plies < 0:
        raise ValueError("opening_plies must be nonnegative")
    opponent_rng = random.Random(opponent_random_stream_id(seed))
    opening_rng = random.Random(opening_random_stream_id(seed, WHITE))
    state = initial_state()
    for _ in range(opening_plies):
        state = step(state, select_legal_random_move(state, opening_rng))
        if state.game_over:
            break
    opening_hash = board_hash(state)
    completed_opening_plies = state.move_number
    decisions: list[dict[str, Any]] = []
    while not state.game_over and len(decisions) < max_moves:
        before = state
        if before.to_play == BLACK:
            move = select_legal_random_move(before, opponent_rng)
            source = "same-seed-legal-random-black"
        else:
            encoding = encode_go_state(before)
            mask = np.asarray(encoding["legal_mask"], dtype=np.uint8)
            point_legal = bool(np.any(mask[:PASS_ACTION_INDEX]))
            if encoding["last_move_action"] == PASS_ACTION_INDEX or not point_legal:
                move = PASS
                source = "go-rule-pass-gate"
            else:
                selected = int(point_predictions(
                    np.zeros((1, weights.shape[1]), dtype=np.float64),
                    weights, bias, mask[None, :],
                )[0])
                move = action_to_move(selected)
                source = "counterfactual-zero-neural-features"
        state = step(before, move)
        decisions.append({
            "ply": state.move_number,
            "source": source,
            "action": PASS_ACTION_INDEX if move is PASS else int(move[0]) * 9 + int(move[1]),
            "state_sha256_before": object_hash(before.to_dict()),
            "board_hash_after": board_hash(state),
        })
    score = score_area(state)
    result = {
        "seed": seed,
        "opening_plies": completed_opening_plies,
        "opening_final_board_hash": opening_hash,
        "opponent_random_stream_id": opponent_random_stream_id(seed),
        "neural_runtime_called": False,
        "feature_control": "all-128-malecns-output-features-zeroed",
        "same_trained_readout_and_bias": True,
        "termination": "two_passes" if state.game_over else "fixed_horizon_adjudication",
        "winner": "white" if score.winner == WHITE else "black" if score.winner == BLACK else "draw",
        "white_won": score.winner == WHITE,
        "white_score_difference": score.white_total - score.black_total,
        "evaluation_move_count": len(decisions),
        "zero_readout_decision_count": sum(
            item["source"] == "counterfactual-zero-neural-features" for item in decisions
        ),
        "rule_pass_decision_count": sum(
            item["source"] == "go-rule-pass-gate" for item in decisions
        ),
        "decisions": decisions,
    }
    result["game_sha256"] = object_hash(result)
    return result


def evaluate(reference_path: Path, checkpoint_path: Path) -> dict[str, Any]:
    reference = _reference(reference_path)
    checkpoint = load_policy_checkpoint(checkpoint_path)
    if reference["checkpoint"]["hash"] != checkpoint.checkpoint_hash:
        raise ValueError("reference checkpoint hash mismatch")
    if checkpoint.training_info.get("pass_control") != "rule-after-opponent-pass":
        raise ValueError("reference is not a rule-PASS hybrid checkpoint")
    config = reference["configuration"]
    if config.get("neural_color") != "white":
        raise ValueError("reference must evaluate White")
    games = []
    for expected in reference["games"]:
        game = play_zero_game(
            seed=int(expected["seed"]),
            max_moves=int(config["max_moves"]),
            opening_plies=int(config["opening_plies"]),
            weights=checkpoint.weights,
            bias=checkpoint.bias,
        )
        expected_opening = expected["opening"]["moves"]
        expected_hash = (
            expected_opening[-1]["board_hash_after"] if expected_opening
            else board_hash(initial_state())
        )
        if game["opening_final_board_hash"] != expected_hash:
            raise ValueError("zero-control opening differs from neural reference")
        games.append(game)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "scope": "paired same-seed game counterfactual; no MaleCNS runtime in control",
        "reference_report_sha256": reference["report_sha256"],
        "checkpoint_hash": checkpoint.checkpoint_hash,
        "configuration": {
            "max_moves": config["max_moves"],
            "opening_plies": config["opening_plies"],
            "same_seeds": True,
            "same_opening_board_hashes": True,
            "same_black_opponent_random_streams": True,
            "same_trained_point_readout_bias_and_weights": True,
            "rule_pass_gate_preserved": True,
            "neural_runtime_called_for_control": False,
        },
        "metrics": {
            "game_count": len(games),
            "neural_reference_white_wins": reference["metrics"]["wins"],
            "zero_control_white_wins": sum(game["white_won"] for game in games),
            "neural_reference_natural_finishes": reference["metrics"]["natural_finish_count"],
            "zero_control_natural_finishes": sum(
                game["termination"] == "two_passes" for game in games
            ),
            "zero_readout_decision_count": sum(
                game["zero_readout_decision_count"] for game in games
            ),
            "zero_rule_pass_decision_count": sum(
                game["rule_pass_decision_count"] for game in games
            ),
        },
        "games": games,
        "limitation": "Eight weak-opponent games are a development control, not final strength evidence or a biological-topology comparison.",
    }
    report["report_sha256"] = object_hash(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.reference, args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_bytes(report) + b"\n")
    print(json.dumps({
        "schema": report["schema"],
        "report_sha256": report["report_sha256"],
        "checkpoint_hash": report["checkpoint_hash"],
        "configuration": report["configuration"],
        "metrics": report["metrics"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
