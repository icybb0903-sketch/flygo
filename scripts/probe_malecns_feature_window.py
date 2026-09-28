"""Small, read-only Stage 8 probe for MaleCNS feature-window information.

The probe selects a fixed, collision-enriched subset from an existing cache,
reruns only those Go states through the real MaleCNS runtime at several LIF
window lengths, and writes a new evidence JSON.  It does not train or write a
cache/checkpoint.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.go_engine import GoState
from src.go_neural_encoding import encode_go_state
from src.malecns_dynamics import LIFParameters, load_graph_assets, simulate_go_encoding


SCHEMA = "malecns-stage8-feature-window-probe-v1"
DEFAULT_GRAPH = (
    PROJECT_ROOT
    / "data"
    / "malecns"
    / "runtime"
    / "malecns-v1.0-w5-3acb6434e71160fc"
)
DEFAULT_SAMPLES = (
    PROJECT_ROOT
    / "data"
    / "training"
    / "stage8_role_symmetric_pilot"
    / "samples.jsonl"
)
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "evaluation" / "stage8" / "probes"


def _load_samples(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on {path}:{line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"sample on {path}:{line_number} is not an object")
            records.append(record)
    if not records:
        raise ValueError("source cache contains no samples")
    return records


def _sample_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(record.get("game_id")),
        int(record.get("ply", -1)),
        str(record.get("sample_id")),
    )


def select_probe_samples(
    records: list[dict[str, Any]], *, limit: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select collision members first, then deterministic ply-quantile controls."""
    if not 2 <= limit <= 12:
        raise ValueError("sample limit must be between 2 and 12")
    by_feature: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        feature_hash = record.get("hashes", {}).get("features_sha256")
        if not isinstance(feature_hash, str):
            raise ValueError("source sample is missing features_sha256")
        by_feature[feature_hash].append(record)
    collision_groups = [
        sorted(group, key=_sample_key)
        for _, group in sorted(by_feature.items())
        if len(group) > 1
    ]

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    collision_budget = min(limit, 8)
    for group in collision_groups:
        if len(selected) + len(group) > collision_budget:
            continue
        for record in group:
            sample_id = str(record["sample_id"])
            if sample_id not in selected_ids:
                selected.append(record)
                selected_ids.add(sample_id)
        if len(selected) >= collision_budget:
            break

    remaining = sorted(
        (record for record in records if str(record["sample_id"]) not in selected_ids),
        key=lambda record: (int(record.get("ply", -1)), _sample_key(record)),
    )
    needed = limit - len(selected)
    if needed:
        if len(remaining) < needed:
            raise ValueError("source cache is too small for requested probe")
        if needed == 1:
            control_indices = [len(remaining) // 2]
        else:
            control_indices = [
                round(index * (len(remaining) - 1) / (needed - 1))
                for index in range(needed)
            ]
        for index in control_indices:
            selected.append(remaining[index])

    selected.sort(key=_sample_key)
    selected_feature_counts = Counter(
        str(record["hashes"]["features_sha256"]) for record in selected
    )
    return selected, {
        "method": "up_to_8_members_from_12ms_collision_groups_then_ply_quantile_controls-v1",
        "collision_enriched": True,
        "warning": (
            "This bounded probe targets known 12 ms collisions and is not an estimate "
            "of collision frequency in the full dataset."
        ),
        "source_sample_count": len(records),
        "source_feature_unique_count": len(by_feature),
        "source_duplicate_feature_occurrence_count": len(records) - len(by_feature),
        "source_collision_group_count": len(collision_groups),
        "selected_sample_count": len(selected),
        "selected_12ms_collision_group_count": sum(
            count > 1 for count in selected_feature_counts.values()
        ),
        "selected_12ms_duplicate_occurrence_count": len(selected)
        - len(selected_feature_counts),
    }


def _pairwise_metrics(vectors: np.ndarray) -> dict[str, Any]:
    distances = [
        float(np.linalg.norm(vectors[left] - vectors[right]))
        for left in range(len(vectors))
        for right in range(left + 1, len(vectors))
    ]
    nonzero = [value for value in distances if value > 0.0]
    return {
        "pair_count": len(distances),
        "zero_distance_pair_count": len(distances) - len(nonzero),
        "minimum_nonzero_l2": min(nonzero) if nonzero else None,
        "median_l2": statistics.median(distances),
        "mean_l2": statistics.fmean(distances),
        "maximum_l2": max(distances),
    }


def _duration_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    vectors = np.asarray([row["features"] for row in rows], dtype=np.float64)
    hashes = [str(row["features_sha256"]) for row in rows]
    counts = Counter(hashes)
    runtimes = [float(row["runtime_seconds"]) for row in rows]
    return {
        "sample_count": len(rows),
        "unique_feature_count": len(counts),
        "duplicate_feature_occurrence_count": len(rows) - len(counts),
        "collision_group_count": sum(count > 1 for count in counts.values()),
        "all_zero_vector_count": int(np.sum(np.all(vectors == 0.0, axis=1))),
        "active_feature_count": int(np.sum(np.any(vectors != 0.0, axis=0))),
        "pairwise_l2": _pairwise_metrics(vectors),
        "runtime_seconds": {
            "total": sum(runtimes),
            "mean": statistics.fmean(runtimes),
            "median": statistics.median(runtimes),
            "maximum": max(runtimes),
        },
    }


def run_probe(
    *,
    graph_directory: Path,
    samples_path: Path,
    durations_ms: Iterable[float],
    sample_limit: int,
    lif_seed: int,
) -> dict[str, Any]:
    records = _load_samples(samples_path)
    selected, selection = select_probe_samples(records, limit=sample_limit)
    graph = load_graph_assets(graph_directory)
    durations = tuple(float(value) for value in durations_ms)
    if any(not math.isfinite(value) or value <= 0.0 for value in durations):
        raise ValueError("durations must be finite and positive")
    if len(set(durations)) != len(durations):
        raise ValueError("durations must be unique")

    runs: dict[str, list[dict[str, Any]]] = {}
    for duration_ms in durations:
        duration_key = f"{duration_ms:g}"
        parameters = LIFParameters(duration_ms=duration_ms)
        rows: list[dict[str, Any]] = []
        for index, record in enumerate(selected, 1):
            state = GoState.from_dict(record["state"])
            encoding = encode_go_state(state)
            started = time.perf_counter()
            frame = simulate_go_encoding(
                graph,
                encoding,
                parameters=parameters,
                seed=lif_seed,
            )
            elapsed = time.perf_counter() - started
            output = frame["output_pool"]
            rows.append(
                {
                    "sample_id": record["sample_id"],
                    "game_id": record["game_id"],
                    "ply": record["ply"],
                    "supervision_kind": record["supervision_provenance"]["kind"],
                    "encoding_sha256": encoding["encoding_hash"],
                    "source_12ms_features_sha256": record["hashes"]["features_sha256"],
                    "features_sha256": output["features_hash"],
                    "features": output["features"],
                    "total_spike_count": frame["total_spike_count"],
                    "synaptic_event_count": frame["synaptic_event_count"],
                    "runtime_seconds": elapsed,
                }
            )
            print(
                f"duration={duration_ms:g}ms sample={index}/{len(selected)} "
                f"runtime={elapsed:.3f}s",
                flush=True,
            )
        runs[duration_key] = rows

    summaries = {key: _duration_summary(rows) for key, rows in runs.items()}
    baseline_key = f"{durations[0]:g}"
    baseline_runtime = summaries[baseline_key]["runtime_seconds"]["mean"]
    for key, summary in summaries.items():
        summary["mean_runtime_ratio_vs_first_duration"] = (
            summary["runtime_seconds"]["mean"] / baseline_runtime
        )

    baseline_groups: dict[str, list[str]] = defaultdict(list)
    for row in runs[baseline_key]:
        baseline_groups[row["features_sha256"]].append(row["sample_id"])
    collided_groups = [ids for ids in baseline_groups.values() if len(ids) > 1]
    separation: dict[str, Any] = {}
    for key, rows in runs.items():
        hash_by_sample = {row["sample_id"]: row["features_sha256"] for row in rows}
        separated = sum(
            len({hash_by_sample[sample_id] for sample_id in group}) == len(group)
            for group in collided_groups
        )
        separation[key] = {
            "baseline_collision_group_count": len(collided_groups),
            "fully_separated_group_count": separated,
            "fully_separated_rate": (
                separated / len(collided_groups) if collided_groups else None
            ),
        }

    return {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "bounded feature-information probe only; no training, checkpoint, "
            "whole-game evaluation, or production mutation"
        ),
        "source_samples_path": str(samples_path.resolve()),
        "graph_directory": str(graph_directory.resolve()),
        "graph": {
            "model_id": graph.model_id,
            "manifest_sha256": graph.manifest_sha256,
            "node_count": graph.node_count,
            "edge_count": graph.edge_count,
        },
        "configuration": {
            "durations_ms": list(durations),
            "lif_seed": lif_seed,
            "sample_limit": sample_limit,
        },
        "selection": selection,
        "selected_samples": [
            {
                "sample_id": record["sample_id"],
                "game_id": record["game_id"],
                "ply": record["ply"],
                "supervision_kind": record["supervision_provenance"]["kind"],
                "source_12ms_features_sha256": record["hashes"]["features_sha256"],
            }
            for record in selected
        ],
        "summaries": summaries,
        "baseline_collision_separation": separation,
        "runs": runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--durations-ms", nargs="+", type=float, default=[12.0, 24.0, 36.0])
    parser.add_argument("--sample-limit", type=int, default=12)
    parser.add_argument("--lif-seed", type=int, default=0)
    args = parser.parse_args()
    if args.lif_seed < 0:
        parser.error("--lif-seed must be nonnegative")

    result = run_probe(
        graph_directory=args.graph,
        samples_path=args.samples,
        durations_ms=args.durations_ms,
        sample_limit=args.sample_limit,
        lif_seed=args.lif_seed,
    )
    args.output_directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = args.output_directory / f"feature_window_probe_{stamp}.json"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing probe: {output_path}")
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output_path), "summaries": result["summaries"], "baseline_collision_separation": result["baseline_collision_separation"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
