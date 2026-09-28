"""Validate and combine Stage 6 game batches under predeclared gates."""

from __future__ import annotations

import argparse
from math import comb
import json
from pathlib import Path
import sys
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_neural_go_games import (
    SUPPORTED_REPORT_SCHEMAS,
)
from scripts.train_neural_go_readout import canonical_bytes, object_hash


COMBINED_SCHEMA = "malecns-go-stage6-tournament-v1"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "evaluation" / "stage6" / "report.json"


def binomial_tail(successes: int, trials: int, probability: float = 0.5) -> float:
    if type(successes) is not int or type(trials) is not int or not 0 <= successes <= trials:
        raise ValueError("successes and trials are invalid")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0,1]")
    return sum(
        comb(trials, count)
        * probability**count
        * (1.0 - probability) ** (trials - count)
        for count in range(successes, trials + 1)
    )


def load_batch(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") not in SUPPORTED_REPORT_SCHEMAS:
        raise ValueError(f"unsupported Stage 6 batch schema: {path}")
    claimed = payload.get("report_sha256")
    unsigned = dict(payload)
    unsigned.pop("report_sha256", None)
    if not isinstance(claimed, str) or object_hash(unsigned) != claimed:
        raise ValueError(f"Stage 6 batch hash mismatch: {path}")
    return payload


def combine_batches(
    paths: Sequence[Path],
    *,
    output_path: Path,
    minimum_games: int = 10,
    minimum_win_rate: float = 0.9,
    maximum_p_value: float = 0.05,
) -> dict[str, Any]:
    if len(paths) < 2:
        raise ValueError("at least two independently written batches are required")
    batches = [load_batch(path) for path in paths]
    first = batches[0]
    checkpoint_hash = first["checkpoint"]["hash"]
    graph_hash = first["graph"]["manifest_sha256"]
    max_moves = first["configuration"]["max_moves"]
    lif_seed = first["configuration"].get("lif_seed")
    opponent_random_stream_version = first["configuration"].get(
        "opponent_random_stream_version", "seed-and-neural-colour-v1"
    )
    move_state_hash = first["configuration"].get("move_state_hash")
    opening_plies = first["configuration"].get("opening_plies", 0)
    opening_random_stream_version = first["configuration"].get(
        "opening_random_stream_version", "none"
    )
    if type(lif_seed) is not int or lif_seed < 0:
        raise ValueError("batch does not declare a valid fixed LIF seed")
    for batch in batches[1:]:
        if batch["checkpoint"]["hash"] != checkpoint_hash:
            raise ValueError("batches use different checkpoints")
        if batch["graph"]["manifest_sha256"] != graph_hash:
            raise ValueError("batches use different graphs")
        if batch["configuration"]["max_moves"] != max_moves:
            raise ValueError("batches use different horizons")
        if batch["configuration"].get("lif_seed") != lif_seed:
            raise ValueError("batches use different LIF seeds")
        batch_stream_version = batch["configuration"].get(
            "opponent_random_stream_version", "seed-and-neural-colour-v1"
        )
        if batch_stream_version != opponent_random_stream_version:
            raise ValueError("batches use different opponent random stream versions")
        if batch["configuration"].get("move_state_hash") != move_state_hash:
            raise ValueError("batches use different move state hash semantics")
        if batch["configuration"].get("opening_plies", 0) != opening_plies:
            raise ValueError("batches use different opening ply counts")
        if (
            batch["configuration"].get("opening_random_stream_version", "none")
            != opening_random_stream_version
        ):
            raise ValueError("batches use different opening random stream versions")

    games = [game for batch in batches for game in batch["games"]]
    match_keys = [(game["seed"], game["neural_color"]) for game in games]
    if len(set(match_keys)) != len(match_keys):
        raise ValueError("duplicate seed/colour match found across batches")
    wins = sum(game["neural_won"] for game in games)
    draws = sum(game["draw"] for game in games)
    losses = len(games) - wins - draws
    differences = [float(game["neural_score_difference"]) for game in games]
    black_games = [game for game in games if game["neural_color"] == "black"]
    white_games = [game for game in games if game["neural_color"] == "white"]
    paired_seeds = {
        seed
        for seed in {game["seed"] for game in games}
        if {(game["neural_color"]) for game in games if game["seed"] == seed}
        == {"black", "white"}
    }
    p_value = binomial_tail(wins, len(games), 0.5)
    win_rate = wins / len(games)
    mean_difference = sum(differences) / len(differences)

    def role_evaluation(role_games: list[dict[str, Any]]) -> dict[str, Any]:
        role_wins = sum(game["neural_won"] for game in role_games)
        role_differences = [float(game["neural_score_difference"]) for game in role_games]
        role_p = binomial_tail(role_wins, len(role_games), 0.5)
        role_win_rate = role_wins / len(role_games)
        role_mean = sum(role_differences) / len(role_differences)
        role_gates = {
            "minimum_five_games": len(role_games) >= 5,
            "minimum_80_percent_win_rate": role_win_rate >= 0.8,
            "positive_mean_score_difference": role_mean > 0.0,
            "one_sided_binomial_significance": role_p <= 0.05,
        }
        return {
            "status": "passed" if all(role_gates.values()) else "failed",
            "game_count": len(role_games),
            "wins": role_wins,
            "losses": len(role_games) - role_wins,
            "win_rate": role_win_rate,
            "mean_neural_score_difference": role_mean,
            "one_sided_binomial_p_value_vs_half": role_p,
            "gates": role_gates,
        }
    gates = {
        "minimum_game_count": len(games) >= minimum_games,
        "all_seeds_have_colour_swap": len(paired_seeds) * 2 == len(games),
        "both_colours_have_minimum_five_games": len(black_games) >= 5 and len(white_games) >= 5,
        "minimum_win_rate": win_rate >= minimum_win_rate,
        "positive_mean_score_difference": mean_difference > 0.0,
        "one_sided_binomial_significance": p_value <= maximum_p_value,
        "all_runtime_guarantees_hold": all(
            batch["runtime_guarantees"]
            == {
                "same_select_action_path_as_web_app": True,
                "teacher_accessed_at_runtime": False,
                "baseline_controller_called": False,
                "fallback_used": False,
            }
            for batch in batches
        ),
    }
    report: dict[str, Any] = {
        "schema": COMBINED_SCHEMA,
        "scope": "10-game paired fixed-horizon tournament versus legal random",
        "status": "passed" if all(gates.values()) else "failed",
        "checkpoint": first["checkpoint"],
        "graph": first["graph"],
        "configuration": {
            "game_count": len(games),
            "unique_seed_count": len({game["seed"] for game in games}),
            "max_moves": max_moves,
            "lif_seed": lif_seed,
            "colour_swap_per_seed": True,
            "opponent": "uniform-legal-point-random-v1",
            "opponent_random_stream_version": opponent_random_stream_version,
            "move_state_hash": move_state_hash,
            "opening_plies": opening_plies,
            "opening_random_stream_version": opening_random_stream_version,
            "horizon_scoring": first["configuration"]["horizon_scoring"],
        },
        "metrics": {
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": win_rate,
            "black_games": len(black_games),
            "black_wins": sum(game["neural_won"] for game in black_games),
            "white_games": len(white_games),
            "white_wins": sum(game["neural_won"] for game in white_games),
            "mean_neural_score_difference": mean_difference,
            "minimum_neural_score_difference": min(differences),
            "maximum_neural_score_difference": max(differences),
            "one_sided_binomial_p_value_vs_half": p_value,
            "natural_finish_count": sum(game["termination"] == "two_passes" for game in games),
            "horizon_adjudication_count": sum(
                game["termination"] == "fixed_horizon_adjudication" for game in games
            ),
        },
        "thresholds": {
            "minimum_games": minimum_games,
            "minimum_win_rate": minimum_win_rate,
            "maximum_p_value": maximum_p_value,
        },
        "gates": gates,
        "role_evaluations": {
            "black": role_evaluation(black_games),
            "white": role_evaluation(white_games),
        },
        "input_batches": [
            {
                "file": str(path.resolve()),
                "report_sha256": batch["report_sha256"],
                "game_count": len(batch["games"]),
            }
            for path, batch in zip(paths, batches, strict=True)
        ],
        "limitations": first["limitations"],
        "runtime_guarantees": first["runtime_guarantees"],
        "games": games,
    }
    report["report_sha256"] = object_hash(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(canonical_bytes(report) + b"\n")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batches", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--minimum-games", type=int, default=10)
    parser.add_argument("--minimum-win-rate", type=float, default=0.9)
    parser.add_argument("--maximum-p-value", type=float, default=0.05)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = combine_batches(
        args.batches,
        output_path=args.output,
        minimum_games=args.minimum_games,
        minimum_win_rate=args.minimum_win_rate,
        maximum_p_value=args.maximum_p_value,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(args.output.resolve()),
                "report_sha256": report["report_sha256"],
                "metrics": report["metrics"],
                "gates": report["gates"],
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
