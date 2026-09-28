"""Stage 7 production-role evaluation for the MaleCNS Go controller.

The web application always assigns the MaleCNS controller to White.  This
evaluator therefore measures that exact role over new legal-random opponent
seeds while keeping the deployed LIF seed fixed.  It writes every move and
all neural provenance through the existing Stage 6 game recorder.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.combine_stage6_games import binomial_tail
from scripts.evaluate_neural_go_games import (
    opening_random_stream_id,
    opponent_random_stream_id,
    play_game,
    select_legal_random_move,
)
from scripts.train_neural_go_readout import (
    DEFAULT_CHECKPOINT,
    DEFAULT_GRAPH,
    canonical_bytes,
    object_hash,
)
from src.go_engine import (
    BLACK,
    EMPTY,
    PASS,
    WHITE,
    GoState,
    initial_state,
    legal_moves,
    score_area,
    step,
)
from src.go_neural_encoding import action_to_move, board_hash, encode_go_state
from src.malecns_dynamics import LIFParameters, load_graph_assets, simulate_go_encoding
from src.malecns_policy import load_policy_checkpoint, select_action


REPORT_SCHEMA = "malecns-go-stage7-production-white-v2"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "evaluation" / "stage7" / "white.json"

# These are the frozen Stage 7 acceptance criteria.  CLI threshold flags are
# retained for backwards compatibility, but only describe a development view;
# they cannot weaken this final gate.
FINAL_MINIMUM_GAMES = 100
FINAL_MINIMUM_WIN_RATE = 0.80
FINAL_MINIMUM_WILSON_LOWER = 0.70
FINAL_MINIMUM_NATURAL_FINISH_RATE = 0.90
FINAL_MINIMUM_RANDOM_WHITE_LIFT = 0.15
SEMANTIC_EXCLUDED_KEYS = frozenset(
    {
        "elapsed_seconds",
        "output",
        "path",
        "output_path",
        "report_sha256",
        "semantic_core_sha256",
    }
)


def semantic_core(value: Any) -> Any:
    """Return stable report content, excluding elapsed time and local paths."""
    if isinstance(value, Mapping):
        return {
            str(key): semantic_core(item)
            for key, item in value.items()
            if str(key) not in SEMANTIC_EXCLUDED_KEYS
            and not str(key).endswith("_path")
        }
    if isinstance(value, list):
        return [semantic_core(item) for item in value]
    if isinstance(value, tuple):
        return [semantic_core(item) for item in value]
    return value


def semantic_core_hash(report: Mapping[str, Any]) -> str:
    """Hash only semantic/core evidence, not timing or machine-local paths."""
    return object_hash(semantic_core(report))


def build_final_gates(
    *,
    game_count: int,
    win_rate: float,
    wilson_lower: float,
    natural_finish_rate: float,
    random_white_lift: float,
    overlap_is_assessed: bool,
    seed_overlap_count: int,
    state_hash_overlap_count: int,
    unique_evaluation_seeds: bool,
    runtime_guarantees_hold: bool,
) -> dict[str, bool]:
    """Evaluate the immutable final gate separately from development settings."""
    return {
        "minimum_game_count_100": game_count >= FINAL_MINIMUM_GAMES,
        "minimum_win_rate_0_80": win_rate >= FINAL_MINIMUM_WIN_RATE,
        "minimum_wilson_95_lower_bound_0_70": wilson_lower
        >= FINAL_MINIMUM_WILSON_LOWER,
        "minimum_natural_finish_rate_0_90": natural_finish_rate
        >= FINAL_MINIMUM_NATURAL_FINISH_RATE,
        "minimum_lift_over_random_white_0_15": random_white_lift
        >= FINAL_MINIMUM_RANDOM_WHITE_LIFT,
        "reference_overlap_audit_available": overlap_is_assessed,
        "zero_reference_seed_overlap": overlap_is_assessed and seed_overlap_count == 0,
        "zero_reference_decision_state_hash_overlap": overlap_is_assessed
        and state_hash_overlap_count == 0,
        "unique_evaluation_seeds": unique_evaluation_seeds,
        "all_runtime_guarantees_hold": runtime_guarantees_hold,
    }


def _random_move(state: GoState, rng: random.Random) -> object:
    points = legal_moves(state, include_pass=False)
    return points[rng.randrange(len(points))] if points else PASS


def play_random_white_baseline(
    *, seed: int, max_moves: int, opening_plies: int = 0
) -> dict[str, Any]:
    """Play same-seed random White vs random Black without invoking neural code."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(max_moves) is not int or max_moves <= 1:
        raise ValueError("max_moves must be greater than one")
    if type(opening_plies) is not int or opening_plies < 0:
        raise ValueError("opening_plies must be a nonnegative integer")
    # Black intentionally uses the exact stream binding used by play_game when
    # the neural controller is White. White receives an independent stream.
    black_rng = random.Random(opponent_random_stream_id(seed))
    white_rng = random.Random(object_hash(["stage7-random-white", seed]))
    opening_rng = random.Random(opening_random_stream_id(seed, WHITE))
    state = initial_state()
    for _ in range(opening_plies):
        state = step(state, select_legal_random_move(state, opening_rng))
        if state.game_over:
            break
    completed_opening_plies = state.move_number
    opening_final_board_hash = board_hash(state)
    state_hashes: list[str] = []
    while not state.game_over and state.move_number - completed_opening_plies < max_moves:
        state_hashes.append(board_hash(state))
        rng = black_rng if state.to_play == BLACK else white_rng
        state = step(state, _random_move(state, rng))
    score = score_area(state)
    white_won = score.winner == WHITE
    result = {
        "seed": seed,
        "white_policy": "uniform-legal-point-random-v1",
        "black_policy": "uniform-legal-point-random-v1",
        "neural_runtime_called": False,
        "move_count": state.move_number,
        "opening_plies": completed_opening_plies,
        "opening_final_board_hash": opening_final_board_hash,
        "evaluation_move_count": state.move_number - completed_opening_plies,
        "opponent_random_stream_id": opponent_random_stream_id(seed),
        "termination": (
            "two_passes" if state.game_over else "fixed_horizon_adjudication"
        ),
        "winner": (
            "white"
            if score.winner == WHITE
            else "black"
            if score.winner == BLACK
            else "draw"
        ),
        "white_won": white_won,
        "draw": score.winner == EMPTY,
        "white_score_difference": score.white_total - score.black_total,
        "final_board_hash": board_hash(state),
        "state_sequence_sha256": object_hash(state_hashes),
    }
    result["game_sha256"] = object_hash(result)
    return result


def _reference_evidence(paths: Sequence[Path]) -> tuple[set[int], set[str]]:
    """Extract prior game seeds and neural-decision state hashes from reports."""
    seeds: set[int] = set()
    hashes: set[str] = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for game in payload.get("games", []):
            seed = game.get("seed")
            if type(seed) is int:
                seeds.add(seed)
            for move in game.get("moves", []):
                if move.get("actor") == "malecns-neural-policy":
                    state_hash = move.get("board_hash_before")
                    if isinstance(state_hash, str):
                        hashes.add(state_hash)
    return seeds, hashes


def wilson_interval(
    successes: int,
    trials: int,
    *,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    """Return the two-sided Wilson score interval for a Bernoulli rate."""
    if (
        type(successes) is not int
        or type(trials) is not int
        or not 0 <= successes <= trials
    ):
        raise ValueError("successes and trials are invalid")
    if trials == 0:
        raise ValueError("trials must be positive")
    if not math.isfinite(z) or z <= 0:
        raise ValueError("z must be finite and positive")
    rate = successes / trials
    scale = 1.0 + z * z / trials
    centre = (rate + z * z / (2.0 * trials)) / scale
    half_width = z * math.sqrt(
        rate * (1.0 - rate) / trials + z * z / (4.0 * trials * trials)
    ) / scale
    return max(0.0, centre - half_width), min(1.0, centre + half_width)


def evaluate_white(
    *,
    graph_directory: Path,
    checkpoint_directory: Path,
    output_path: Path,
    base_seed: int,
    seed_count: int,
    max_moves: int,
    duration_ms: float,
    lif_seed: int,
    minimum_games: int,
    minimum_win_rate: float,
    minimum_wilson_lower: float,
    opening_plies: int = 0,
    evaluation_tier: str = "development",
    reference_reports: Sequence[Path] = (),
) -> dict[str, Any]:
    if type(base_seed) is not int or base_seed < 0:
        raise ValueError("base_seed must be a nonnegative integer")
    if type(seed_count) is not int or seed_count <= 0:
        raise ValueError("seed_count must be positive")
    if type(lif_seed) is not int or lif_seed < 0:
        raise ValueError("lif_seed must be a nonnegative integer")
    if type(minimum_games) is not int or minimum_games <= 0:
        raise ValueError("minimum_games must be positive")
    if type(opening_plies) is not int or opening_plies < 0:
        raise ValueError("opening_plies must be a nonnegative integer")
    if not 0.0 <= minimum_win_rate <= 1.0:
        raise ValueError("minimum_win_rate must be in [0, 1]")
    if not 0.0 <= minimum_wilson_lower <= 1.0:
        raise ValueError("minimum_wilson_lower must be in [0, 1]")
    if evaluation_tier not in {"development", "final"}:
        raise ValueError("evaluation_tier must be development or final")

    graph = load_graph_assets(graph_directory)
    checkpoint = load_policy_checkpoint(checkpoint_directory)
    if (
        checkpoint.model_id != graph.model_id
        or checkpoint.graph_manifest_sha256 != graph.manifest_sha256
    ):
        raise ValueError("checkpoint does not match the evaluation graph")
    parameters = LIFParameters(duration_ms=duration_ms)

    def production_selector(
        state: GoState,
        _game_seed: int,
    ) -> tuple[object, Mapping[str, Any]]:
        encoding = encode_go_state(state)
        frame = simulate_go_encoding(graph, encoding, parameters=parameters, seed=lif_seed)
        decision = select_action(frame, encoding, checkpoint)
        return action_to_move(decision["action_index"]), {
            "checkpoint_hash": decision["checkpoint_hash"],
            "encoding_hash": decision["encoding_hash"],
            "frame_id": decision["frame_id"],
            "decision_hash": decision["decision_hash"],
            "action_index": decision["action_index"],
            "output_features_hash": decision["output_features_hash"],
            "controller_source": decision["controller_source"],
            "controller_reason": decision["controller_reason"],
            "neural_readout_used": decision["neural_readout_used"],
            "pass_control": decision["pass_control"],
            "teacher_accessed_at_runtime": decision["teacher_accessed_at_runtime"],
            "baseline_controller_called": decision["baseline_controller_called"],
            "fallback_used": decision["fallback_used"],
        }

    games: list[dict[str, Any]] = []
    random_white_baseline_games: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index in range(seed_count):
        seed = base_seed + index
        game = play_game(
            seed=seed,
            neural_color=WHITE,
            max_moves=max_moves,
            neural_selector=production_selector,
            opening_plies=opening_plies,
        )
        games.append(game)
        random_white_baseline_games.append(
            play_random_white_baseline(
                seed=seed, max_moves=max_moves, opening_plies=opening_plies
            )
        )
        print(
            f"game {index + 1}/{seed_count}: seed={seed} neural=white "
            f"winner={game['winner']} diff={game['neural_score_difference']:+.1f} "
            f"termination={game['termination']}",
            flush=True,
        )

    wins = sum(bool(game["neural_won"]) for game in games)
    draws = sum(bool(game["draw"]) for game in games)
    losses = len(games) - wins - draws
    differences = [float(game["neural_score_difference"]) for game in games]
    lower, upper = wilson_interval(wins, len(games))
    win_rate = wins / len(games)
    mean_difference = sum(differences) / len(differences)
    p_value = binomial_tail(wins, len(games), 0.5)
    natural_finish_count = sum(game["termination"] == "two_passes" for game in games)
    natural_finish_rate = natural_finish_count / len(games)
    baseline_wins = sum(bool(game["white_won"]) for game in random_white_baseline_games)
    baseline_draws = sum(bool(game["draw"]) for game in random_white_baseline_games)
    baseline_losses = len(random_white_baseline_games) - baseline_wins - baseline_draws
    baseline_win_rate = baseline_wins / len(random_white_baseline_games)
    random_white_lift = win_rate - baseline_win_rate

    evaluation_seeds = {int(game["seed"]) for game in games}
    evaluation_state_hashes = {
        str(move["board_hash_before"])
        for game in games
        for move in game["moves"]
        if move["actor"] == "malecns-neural-policy"
    }
    reference_seeds, reference_state_hashes = _reference_evidence(reference_reports)
    seed_overlap = evaluation_seeds & reference_seeds
    state_hash_overlap = evaluation_state_hashes & reference_state_hashes
    overlap_is_assessed = bool(reference_reports)

    development_gates = {
        "minimum_game_count": len(games) >= minimum_games,
        "unique_unseen_opponent_seeds": (
            len({game["seed"] for game in games}) == len(games)
        ),
        "minimum_win_rate": win_rate >= minimum_win_rate,
        "minimum_wilson_95_lower_bound": lower >= minimum_wilson_lower,
        "positive_mean_score_difference": mean_difference > 0.0,
        "all_runtime_guarantees_hold": all(
            move["provenance"].get("teacher_accessed_at_runtime") is False
            and move["provenance"].get("baseline_controller_called") is False
            and move["provenance"].get("fallback_used") is False
            for game in games
            for move in game["moves"]
            if move["actor"] == "malecns-neural-policy"
        ),
    }
    final_gates = build_final_gates(
        game_count=len(games),
        win_rate=win_rate,
        wilson_lower=lower,
        natural_finish_rate=natural_finish_rate,
        random_white_lift=random_white_lift,
        overlap_is_assessed=overlap_is_assessed,
        seed_overlap_count=len(seed_overlap),
        state_hash_overlap_count=len(state_hash_overlap),
        unique_evaluation_seeds=len(evaluation_seeds) == len(games),
        runtime_guarantees_hold=development_gates["all_runtime_guarantees_hold"],
    )
    final_passed = all(final_gates.values())
    status = (
        "development_completed"
        if evaluation_tier == "development"
        else "passed" if final_passed else "failed"
    )
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "scope": "production-role MaleCNS White versus legal uniform-random Black",
        "evaluation_tier": evaluation_tier,
        "status": status,
        "claim": (
            "development evidence only; not a final acceptance result"
            if evaluation_tier == "development"
            else "final frozen-gate acceptance result"
        ),
        "checkpoint": {
            "hash": checkpoint.checkpoint_hash,
            "training_seed": checkpoint.training_info.get("seed"),
            "training_dataset_hash": checkpoint.training_info.get("dataset_hash"),
        },
        "graph": {
            "model_id": graph.model_id,
            "manifest_sha256": graph.manifest_sha256,
            "node_count": graph.node_count,
            "edge_count": graph.edge_count,
        },
        "configuration": {
            "base_seed": base_seed,
            "seed_count": seed_count,
            "game_count": len(games),
            "neural_color": "white",
            "random_color": "black",
            "max_moves": max_moves,
            "opening_plies": opening_plies,
            "opening_moves_attributed_to_evaluated_policy": False,
            "komi": 7.5,
            "opponent": "uniform-legal-point-random-v1",
            "lif_seed": lif_seed,
            "horizon_scoring": "Chinese area score without dead-stone adjudication",
            "lif_parameters": asdict(parameters),
            "reference_report_count": len(reference_reports),
        },
        "thresholds": {
            "development_configured": {
                "minimum_games": minimum_games,
                "minimum_win_rate": minimum_win_rate,
                "minimum_wilson_95_lower_bound": minimum_wilson_lower,
            },
            "final_frozen": {
                "minimum_games": FINAL_MINIMUM_GAMES,
                "minimum_win_rate": FINAL_MINIMUM_WIN_RATE,
                "minimum_wilson_95_lower_bound": FINAL_MINIMUM_WILSON_LOWER,
                "minimum_natural_finish_rate": FINAL_MINIMUM_NATURAL_FINISH_RATE,
                "minimum_win_rate_lift_over_random_white": FINAL_MINIMUM_RANDOM_WHITE_LIFT,
                "requires_new_seeds": True,
                "requires_zero_decision_state_hash_overlap": True,
            },
        },
        "metrics": {
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": win_rate,
            "wilson_95_interval": [lower, upper],
            "one_sided_binomial_p_value_vs_half": p_value,
            "one_sided_binomial_p_value_interpretation": (
                "descriptive only; not a Stage 7 primary acceptance gate"
            ),
            "mean_neural_score_difference": mean_difference,
            "minimum_neural_score_difference": min(differences),
            "maximum_neural_score_difference": max(differences),
            "natural_finish_count": natural_finish_count,
            "natural_finish_rate": natural_finish_rate,
            "horizon_adjudication_count": sum(
                game["termination"] == "fixed_horizon_adjudication" for game in games
            ),
            "random_white_baseline": {
                "wins": baseline_wins,
                "losses": baseline_losses,
                "draws": baseline_draws,
                "win_rate": baseline_win_rate,
                "same_seed_as_neural_games": True,
                "neural_runtime_called": False,
            },
            "win_rate_lift_over_random_white": random_white_lift,
        },
        "overlap_audit": {
            "assessed": overlap_is_assessed,
            "reference_report_count": len(reference_reports),
            "evaluation_seed_count": len(evaluation_seeds),
            "evaluation_decision_state_hash_count": len(evaluation_state_hashes),
            "reference_seed_count": len(reference_seeds),
            "reference_decision_state_hash_count": len(reference_state_hashes),
            "seed_overlap_count": len(seed_overlap),
            "decision_state_hash_overlap_count": len(state_hash_overlap),
            "overlapping_seeds": sorted(seed_overlap),
            "overlapping_decision_state_hashes": sorted(state_hash_overlap),
        },
        "gates": {
            "development_configured": development_gates,
            "final_frozen": final_gates,
            "final_passed": final_passed,
        },
        "limitations": [
            "The opponent is a weak legal random baseline and does not search.",
            "Horizon-adjudicated games are not natural two-pass finishes.",
            "A production-role result does not establish a general Black-and-White policy.",
            "The readout is capture-first imitation, not whole-game reinforcement learning.",
            "A 20-game run is development evidence and can never be called final Stage 7 acceptance.",
            "The one-sided p-value versus 0.5 is descriptive and is not a primary acceptance gate.",
        ],
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "games": games,
        "random_white_baseline_games": random_white_baseline_games,
    }
    report["semantic_core_sha256"] = semantic_core_hash(report)
    report["report_sha256"] = object_hash(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(canonical_bytes(report) + b"\n")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-seed", type=int, default=20270001)
    parser.add_argument("--seed-count", type=int, default=20)
    parser.add_argument("--max-moves", type=int, default=100)
    parser.add_argument("--opening-plies", type=int, default=0)
    parser.add_argument("--duration-ms", type=float, default=12.0)
    parser.add_argument("--lif-seed", type=int, default=0)
    parser.add_argument("--minimum-games", type=int, default=20)
    parser.add_argument("--minimum-win-rate", type=float, default=0.8)
    parser.add_argument("--minimum-wilson-lower", type=float, default=0.6)
    parser.add_argument(
        "--evaluation-tier",
        choices=("development", "final"),
        default="development",
        help="20-game/default runs are development; final uses immutable Stage 7 gates",
    )
    parser.add_argument(
        "--reference-report",
        type=Path,
        action="append",
        default=[],
        help="prior development/training report used to prove zero seed/state overlap",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = evaluate_white(
        graph_directory=args.graph,
        checkpoint_directory=args.checkpoint,
        output_path=args.output,
        base_seed=args.base_seed,
        seed_count=args.seed_count,
        max_moves=args.max_moves,
        opening_plies=args.opening_plies,
        duration_ms=args.duration_ms,
        lif_seed=args.lif_seed,
        minimum_games=args.minimum_games,
        minimum_win_rate=args.minimum_win_rate,
        minimum_wilson_lower=args.minimum_wilson_lower,
        evaluation_tier=args.evaluation_tier,
        reference_reports=args.reference_report,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(args.output.resolve()),
                "report_sha256": report["report_sha256"],
                "semantic_core_sha256": report["semantic_core_sha256"],
                "elapsed_seconds": report["elapsed_seconds"],
                "metrics": report["metrics"],
                "gates": report["gates"],
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0 if report["status"] in {"development_completed", "passed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
