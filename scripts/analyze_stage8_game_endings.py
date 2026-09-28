"""Replay verified white-role games to inspect why they stop or reach horizon."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_neural_go_readout import object_hash
from src.go_engine import BLACK, EMPTY, PASS, WHITE, initial_state, legal_moves, score_area, step
from src.go_neural_encoding import action_to_move, board_hash
from src.go_session import select_capture_first_move


def _max_next_black_capture(state) -> int:
    if state.to_play != BLACK or state.game_over:
        return 0
    return max(
        (step(state, move).black_captures - state.black_captures
         for move in legal_moves(state, include_pass=False)),
        default=0,
    )


def analyze(report_path: Path, *, tactical_risk: bool = False) -> dict[str, object]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    claimed = report.pop("report_sha256", None)
    if report.get("schema") != "malecns-go-stage7-production-white-v2" or object_hash(report) != claimed:
        raise ValueError("not a verified white-role game report")
    rows = []
    for game in report["games"]:
        state = initial_state()
        neural_point_count = 0
        teacher_agreement_count = 0
        next_black_capture_after_neural_move_count = 0
        next_black_large_capture_after_teacher_disagreement_count = 0
        preceding_neural_teacher_match: bool | None = None
        model_next_capture_risk_total = 0
        teacher_next_capture_risk_total = 0
        teacher_safer_move_count = 0
        model_safer_move_count = 0
        teacher_much_safer_examples = []
        for record in game["opening"]["moves"] + game["moves"]:
            if board_hash(state) != record["board_hash_before"]:
                raise ValueError("game move's previous board hash mismatch")
            if record["actor"] == "malecns-neural-policy":
                neural_point_count += 1
                teacher_move = select_capture_first_move(state)
                preceding_neural_teacher_match = (
                    action_to_move(int(record["action"])) == teacher_move
                )
                teacher_agreement_count += preceding_neural_teacher_match
                if tactical_risk:
                    teacher_state = step(state, teacher_move)
                    model_state = step(state, action_to_move(int(record["action"])))
                    teacher_risk = _max_next_black_capture(teacher_state)
                    model_risk = _max_next_black_capture(model_state)
                    teacher_next_capture_risk_total += teacher_risk
                    model_next_capture_risk_total += model_risk
                    teacher_safer_move_count += teacher_risk < model_risk
                    model_safer_move_count += model_risk < teacher_risk
                    if model_risk - teacher_risk >= 3:
                        teacher_much_safer_examples.append({
                            "ply": record["ply"],
                            "board_hash_before": record["board_hash_before"],
                            "model_action": record["action"],
                            "teacher_action": 81 if teacher_move is PASS else teacher_move[0] * 9 + teacher_move[1],
                            "model_max_next_black_capture": model_risk,
                            "teacher_max_next_black_capture": teacher_risk,
                        })
            black_captures_before = state.black_captures
            state = step(state, action_to_move(int(record["action"])))
            if record["actor"] == "legal-random" and record["color"] == "black":
                captured = state.black_captures - black_captures_before
                if captured >= 3 and preceding_neural_teacher_match is not None:
                    next_black_capture_after_neural_move_count += 1
                    if not preceding_neural_teacher_match:
                        next_black_large_capture_after_teacher_disagreement_count += 1
                preceding_neural_teacher_match = None
            if board_hash(state) != record["board_hash_after"]:
                raise ValueError("game move's resulting board hash mismatch")
        score = score_area(state)
        rows.append({
            "seed": game["seed"],
            "reported_termination": game["termination"],
            "reported_winner": game["winner"],
            "replayed_winner": "white" if score.winner == WHITE else "black" if score.winner == BLACK else "draw",
            "board_black_stones": sum(value == BLACK for value in state.board),
            "board_white_stones": sum(value == WHITE for value in state.board),
            "board_empty_points": sum(value == EMPTY for value in state.board),
            "black_captures": state.black_captures,
            "white_captures": state.white_captures,
            "legal_point_count_for_next_player": len(legal_moves(state, include_pass=False)) if not state.game_over else 0,
            "next_player": "white" if state.to_play == WHITE else "black",
            "consecutive_passes": state.consecutive_passes,
            "actual_game_over": state.game_over,
            "white_minus_black_area_score": score.white_total - score.black_total,
            "neural_point_move_count": neural_point_count,
            "neural_teacher_agreement_count_on_own_states": teacher_agreement_count,
            "next_black_capture_at_least_3_after_neural_move_count": next_black_capture_after_neural_move_count,
            "those_after_teacher_disagreement_count": next_black_large_capture_after_teacher_disagreement_count,
            "one_ply_tactical_risk": {
                "computed": tactical_risk,
                "model_max_next_black_capture_total": model_next_capture_risk_total if tactical_risk else None,
                "teacher_max_next_black_capture_total": teacher_next_capture_risk_total if tactical_risk else None,
                "teacher_safer_move_count": teacher_safer_move_count if tactical_risk else None,
                "model_safer_move_count": model_safer_move_count if tactical_risk else None,
                "teacher_much_safer_examples": teacher_much_safer_examples if tactical_risk else None,
            },
        })
    if any(row["reported_winner"] != row["replayed_winner"] for row in rows):
        raise ValueError("reported winner disagrees with replay")
    return {
        "schema": "stage8-game-ending-diagnostic-v1",
        "source_report_sha256": claimed,
        "scope": "verified replay, read-only; not a policy or new neural simulation",
        "games": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--tactical-risk", action="store_true")
    args = parser.parse_args()
    print(json.dumps(analyze(args.report, tactical_risk=args.tactical_risk), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
