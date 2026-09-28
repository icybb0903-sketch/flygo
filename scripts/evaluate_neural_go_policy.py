"""Independent Stage 5 evaluation for the trained MaleCNS Go readout.

This script creates new 9x9 positions from a seed that must differ from the
training seed, runs the same full-graph LIF and strict policy path used by the
web app, and compares its teacher agreement with uniform random legal play.

Passing this restricted imitation gate is evidence of reproducible signal on
unseen positions.  It is deliberately *not* reported as whole-game Go skill.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_neural_go_readout import (
    DEFAULT_CHECKPOINT,
    DEFAULT_GRAPH,
    canonical_bytes,
    generate_teacher_positions,
    object_hash,
)
from src.go_neural_encoding import encode_go_state
from src.malecns_dynamics import LIFParameters, load_graph_assets, simulate_go_encoding
from src.malecns_policy import load_policy_checkpoint, select_action


EVALUATION_SCHEMA = "malecns-go-stage5-evaluation-v1"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "evaluation" / "stage5" / "report.json"


def poisson_binomial_tail(probabilities: Sequence[float], observed: int) -> float:
    """Return P(X >= observed) for independent Bernoulli probabilities."""
    values = [float(value) for value in probabilities]
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in values):
        raise ValueError("probabilities must be finite values in [0,1]")
    if type(observed) is not int or observed < 0:
        raise ValueError("observed must be a nonnegative integer")
    if observed == 0:
        return 1.0
    if observed > len(values):
        return 0.0
    distribution = [0.0] * (len(values) + 1)
    distribution[0] = 1.0
    for index, probability in enumerate(values):
        for successes in range(index + 1, -1, -1):
            stay = distribution[successes] * (1.0 - probability)
            arrive = distribution[successes - 1] * probability if successes else 0.0
            distribution[successes] = stay + arrive
    return float(sum(distribution[observed:]))


def wilson_interval(successes: int, count: int, *, z: float = 1.959963984540054) -> tuple[float, float]:
    """Return a two-sided Wilson score interval for a binomial proportion."""
    if type(successes) is not int or type(count) is not int or not 0 <= successes <= count:
        raise ValueError("successes and count are invalid")
    if count == 0:
        raise ValueError("count must be positive")
    proportion = successes / count
    denominator = 1.0 + z * z / count
    centre = (proportion + z * z / (2.0 * count)) / denominator
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / count + z * z / (4.0 * count * count)
    ) / denominator
    return max(0.0, centre - radius), min(1.0, centre + radius)


def evaluate(
    *,
    graph_directory: Path,
    checkpoint_directory: Path,
    output_path: Path,
    game_count: int,
    max_moves: int,
    seed: int,
    lif_seed: int | None,
    duration_ms: float,
    minimum_samples: int,
    minimum_lift: float,
    maximum_p_value: float,
) -> dict[str, Any]:
    """Run an independent, production-path evaluation and persist its evidence."""
    checkpoint = load_policy_checkpoint(checkpoint_directory)
    training_seed = checkpoint.training_info.get("seed")
    if type(training_seed) is int and seed == training_seed:
        raise ValueError("evaluation seed must differ from the checkpoint training seed")

    graph = load_graph_assets(graph_directory)
    if (
        checkpoint.model_id != graph.model_id
        or checkpoint.graph_manifest_sha256 != graph.manifest_sha256
    ):
        raise ValueError("checkpoint does not match the evaluation graph")

    parameters = LIFParameters(duration_ms=duration_ms)
    resolved_lif_seed = seed if lif_seed is None else lif_seed
    if type(resolved_lif_seed) is not int or resolved_lif_seed < 0:
        raise ValueError("lif_seed must be a nonnegative integer")
    positions = generate_teacher_positions(
        game_count=game_count,
        max_moves=max_moves,
        seed=seed,
    )
    training_games = {
        game_id
        for values in checkpoint.training_info.get("split_games", {}).values()
        for game_id in values
    }
    evaluation_games = {position.game_id for position in positions}
    overlap = sorted(training_games & evaluation_games)
    if overlap:
        raise ValueError(f"evaluation games overlap training games: {overlap}")

    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for position in positions:
        encoding = encode_go_state(position.state)
        frame = simulate_go_encoding(
            graph,
            encoding,
            parameters=parameters,
            seed=resolved_lif_seed,
        )
        decision = select_action(frame, encoding, checkpoint)
        legal_count = int(sum(encoding["legal_mask"]))
        record = {
            "game_id": position.game_id,
            "ply": position.ply,
            "state_sha256": object_hash(position.state.to_dict()),
            "encoding_hash": encoding["encoding_hash"],
            "frame_id": frame["frame_id"],
            "decision_hash": decision["decision_hash"],
            "teacher_action": position.teacher_action,
            "selected_action": decision["action_index"],
            "teacher_match": decision["action_index"] == position.teacher_action,
            "legal_action_count": legal_count,
            "random_legal_match_probability": 1.0 / legal_count,
        }
        record["record_sha256"] = object_hash(record)
        records.append(record)

    sample_count = len(records)
    matches = sum(record["teacher_match"] for record in records)
    agreement = matches / sample_count
    random_probabilities = [
        record["random_legal_match_probability"] for record in records
    ]
    random_rate = float(np.mean(random_probabilities))
    confidence_low, confidence_high = wilson_interval(matches, sample_count)
    p_value = poisson_binomial_tail(random_probabilities, matches)
    gates = {
        "independent_seed": type(training_seed) is not int or seed != training_seed,
        "no_training_game_overlap": not overlap,
        "minimum_sample_count": sample_count >= minimum_samples,
        "minimum_absolute_lift_over_random": agreement - random_rate >= minimum_lift,
        "poisson_binomial_significance": p_value <= maximum_p_value,
    }

    report: dict[str, Any] = {
        "schema": EVALUATION_SCHEMA,
        "scope": "unseen-position imitation gate; not a whole-game strength claim",
        "status": "passed" if all(gates.values()) else "failed",
        "checkpoint": {
            "hash": checkpoint.checkpoint_hash,
            "policy_version": "malecns-go-linear-128x82-v1",
            "training_seed": training_seed,
            "training_dataset_hash": checkpoint.training_info.get("dataset_hash"),
        },
        "graph": {
            "model_id": graph.model_id,
            "manifest_sha256": graph.manifest_sha256,
            "node_count": graph.node_count,
            "edge_count": graph.edge_count,
        },
        "evaluation": {
            "seed": seed,
            "position_seed": seed,
            "lif_seed": resolved_lif_seed,
            "game_count": game_count,
            "max_moves": max_moves,
            "lif_parameters": asdict(parameters),
            "sample_count": sample_count,
            "game_ids": sorted(evaluation_games),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        },
        "metrics": {
            "teacher_matches": matches,
            "teacher_agreement_rate": agreement,
            "teacher_agreement_wilson_95": [confidence_low, confidence_high],
            "random_legal_expected_matches": float(sum(random_probabilities)),
            "random_legal_expected_rate": random_rate,
            "absolute_lift_over_random": agreement - random_rate,
            "poisson_binomial_p_value": p_value,
        },
        "thresholds": {
            "minimum_samples": minimum_samples,
            "minimum_absolute_lift": minimum_lift,
            "maximum_p_value": maximum_p_value,
        },
        "gates": gates,
        "runtime_guarantees": {
            "same_select_action_path_as_web_app": True,
            "teacher_used_only_for_offline_comparison": True,
            "teacher_accessed_by_policy": False,
            "baseline_fallback_used": False,
        },
        "records": records,
    }
    report["report_sha256"] = object_hash(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(canonical_bytes(report) + b"\n")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--max-moves", type=int, default=72)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument(
        "--lif-seed",
        type=int,
        help="LIF phase seed; omitted preserves the historical behavior of using --seed",
    )
    parser.add_argument("--duration-ms", type=float, default=12.0)
    parser.add_argument("--minimum-samples", type=int, default=100)
    parser.add_argument("--minimum-lift", type=float, default=0.10)
    parser.add_argument("--maximum-p-value", type=float, default=0.01)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = evaluate(
        graph_directory=args.graph,
        checkpoint_directory=args.checkpoint,
        output_path=args.output,
        game_count=args.games,
        max_moves=args.max_moves,
        seed=args.seed,
        lif_seed=args.lif_seed,
        duration_ms=args.duration_ms,
        minimum_samples=args.minimum_samples,
        minimum_lift=args.minimum_lift,
        maximum_p_value=args.maximum_p_value,
    )
    summary = {
        "status": report["status"],
        "output": str(args.output.resolve()),
        "report_sha256": report["report_sha256"],
        "metrics": report["metrics"],
        "gates": report["gates"],
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
