"""Replay a white-role report's seeds with the *teacher rule*, not a neural policy.

This diagnoses whether capture-first imitation is an adequate whole-game
training target.  No result from this script may be described as fly behavior.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_neural_go_games import (
    opening_random_stream_id,
    opponent_random_stream_id,
    select_legal_random_move,
)
from scripts.train_neural_go_readout import canonical_bytes, object_hash
from src.go_engine import BLACK, PASS, WHITE, initial_state, score_area, step
from src.go_neural_encoding import board_hash
from src.go_session import select_capture_first_move


def evaluate(reference_path: Path) -> dict[str, object]:
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    claimed = reference.pop("report_sha256", None)
    if reference.get("schema") != "malecns-go-stage7-production-white-v2" or object_hash(reference) != claimed:
        raise ValueError("reference is not a verified white-role report")
    configuration = reference["configuration"]
    max_moves = int(configuration["max_moves"])
    opening_plies = int(configuration["opening_plies"])
    games = []
    for ref_game in reference["games"]:
        seed = int(ref_game["seed"])
        opening_rng = random.Random(opening_random_stream_id(seed, WHITE))
        black_rng = random.Random(opponent_random_stream_id(seed))
        state = initial_state()
        for _ in range(opening_plies):
            state = step(state, select_legal_random_move(state, opening_rng))
            if state.game_over:
                break
        expected_opening_hash = (
            ref_game["opening"]["moves"][-1]["board_hash_after"]
            if ref_game["opening"]["moves"] else board_hash(initial_state())
        )
        if board_hash(state) != expected_opening_hash:
            raise ValueError("opening state differs from neural reference")
        played = 0
        while not state.game_over and played < max_moves:
            if state.to_play == BLACK:
                move = select_legal_random_move(state, black_rng)
            elif state.consecutive_passes == 1:
                move = PASS
            else:
                move = select_capture_first_move(state)
            state = step(state, move)
            played += 1
        score = score_area(state)
        games.append({
            "seed": seed,
            "opening_board_hash": expected_opening_hash,
            "same_black_random_stream_id": opponent_random_stream_id(seed),
            "teacher_white_won": score.winner == WHITE,
            "winner": "white" if score.winner == WHITE else "black" if score.winner == BLACK else "draw",
            "termination": "two_passes" if state.game_over else "fixed_horizon_adjudication",
            "score_difference_white_minus_black": score.white_total - score.black_total,
            "evaluation_moves": played,
        })
    result = {
        "schema": "stage8-capture-first-teacher-white-baseline-v1",
        "scope": "rule teacher baseline; no MaleCNS graph or neural checkpoint called",
        "reference_report_sha256": claimed,
        "same_game_seeds": True,
        "same_opening_states": True,
        "same_black_random_streams": True,
        "games": games,
        "metrics": {
            "game_count": len(games),
            "teacher_white_wins": sum(game["teacher_white_won"] for game in games),
            "natural_finishes": sum(game["termination"] == "two_passes" for game in games),
        },
    }
    result["report_sha256"] = object_hash(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("teacher baseline output already exists")
    result = evaluate(args.reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_bytes(result) + b"\n")
    print(json.dumps({k: result[k] for k in ("schema", "metrics", "report_sha256")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
