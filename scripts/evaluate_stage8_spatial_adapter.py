"""Pilot-only comparison of experimental adapters through the real MaleCNS graph.

Reuses frozen Stage 8 Go states, labels, and split. Writes diagnostic evidence,
not a deployable checkpoint. An output-feature collision across splits makes
the held-out score invalid and is reported rather than hidden.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.probe_stage8_adapter import _fit_and_score
from scripts.train_neural_go_readout import load_cache
from src.go_engine import GoState
from src.go_neural_encoding import PASS_ACTION_INDEX, encode_go_state
from src.malecns_dynamics import (
    LIFParameters,
    go_encoding_to_spatial_stimulus,
    go_encoding_to_stimulus,
    load_graph_assets,
    simulate_frame,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache", type=Path,
        default=PROJECT_ROOT / "data" / "training" / "stage8_role_symmetric_pilot",
    )
    parser.add_argument(
        "--graph", type=Path,
        default=PROJECT_ROOT / "data" / "malecns" / "runtime"
        / "malecns-v1.0-w5-3acb6434e71160fc",
    )
    parser.add_argument("--ridge", type=float, default=0.1)
    parser.add_argument("--stimulus-mode", choices=("spatial", "legacy"), default="spatial")
    parser.add_argument(
        "--output-pool-selection",
        choices=("random-v1", "input-downstream-v1"), default="random-v1",
    )
    parser.add_argument("--features-output", type=Path)
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "data" / "evaluation" / "stage8" / "spatial_pilot_probe.json",
    )
    args = parser.parse_args()
    old_features, labels, records, manifest = load_cache(args.cache)
    graph = load_graph_assets(args.graph)
    params = LIFParameters(**manifest["generator"]["lif_parameters"])
    new_features = np.empty_like(old_features)
    hashes: list[str] = []
    spike_total = 0
    event_total = 0
    for row, record in enumerate(records):
        encoding = encode_go_state(GoState.from_dict(record["state"]))
        stimulus = (
            go_encoding_to_spatial_stimulus(encoding)
            if args.stimulus_mode == "spatial"
            else go_encoding_to_stimulus(encoding)
        )
        frame = simulate_frame(
            graph, stimulus, parameters=params, seed=int(record["lif_seed"]),
            output_pool_selection=args.output_pool_selection,
        )
        new_features[row] = np.asarray(frame["output_pool"]["features"], dtype=np.float32)
        hashes.append(frame["output_pool"]["features_hash"])
        spike_total += int(frame["total_spike_count"])
        event_total += int(frame["synaptic_event_count"])
        if (row + 1) % 100 == 0 or row + 1 == len(records):
            print(f"experimental full-graph frames {row + 1}/{len(records)}", file=sys.stderr, flush=True)

    masks = np.asarray([record["legal_mask"] for record in records], dtype=bool)
    natural = np.asarray([
        record.get("supervision_provenance", {}).get("kind") == "natural-teacher-turn"
        and label != PASS_ACTION_INDEX
        for record, label in zip(records, labels, strict=True)
    ], dtype=bool)
    train = np.asarray([record["split"] == "train" for record in records]) & natural
    validation = np.asarray([record["split"] == "validation" for record in records]) & natural
    by_hash: dict[str, set[str]] = {}
    for record, feature_hash in zip(records, hashes, strict=True):
        by_hash.setdefault(feature_hash, set()).add(record["split"])
    cross_split = sorted(key for key, splits in by_hash.items() if len(splits) > 1)
    old_hashes = [record["hashes"]["features_sha256"] for record in records]
    centered = new_features - new_features.mean(axis=0)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    explained = np.cumsum(singular_values**2) / np.sum(singular_values**2)
    report = {
        "schema": "stage8-experimental-adapter-pilot-evaluation-v2",
        "scope": "experimental readout comparison, not deployable or final Go-strength evidence",
        "stimulus_mode": args.stimulus_mode,
        "output_pool_selection": args.output_pool_selection,
        "dataset_hash": manifest["dataset_hash"],
        "sample_count": len(records),
        "ridge": args.ridge,
        "old_unique_features": len(Counter(old_hashes)),
        "new_unique_features": len(Counter(hashes)),
        "new_components_for_95_percent_variance": int(np.searchsorted(explained, 0.95) + 1),
        "new_feature_cross_split_collision_count": len(cross_split),
        "new_feature_cross_split_collision_hashes": cross_split,
        "spike_total": spike_total,
        "synaptic_event_total": event_total,
        "old_readout": _fit_and_score(old_features, labels, masks, train, validation, args.ridge),
        "experimental_readout": None if cross_split else _fit_and_score(
            new_features, labels, masks, train, validation, args.ridge
        ),
        "heldout_score_valid": not cross_split,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.features_output is not None:
        args.features_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.features_output,
            features=new_features,
            sample_ids=np.asarray([record["sample_id"] for record in records]),
            feature_hashes=np.asarray(hashes),
            dataset_hash=np.asarray(manifest["dataset_hash"]),
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
