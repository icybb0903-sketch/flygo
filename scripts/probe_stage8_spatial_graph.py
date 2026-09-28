"""Bounded real-MaleCNS probe of the experimental spatial Go stimulus.

This does not train, evaluate Go strength, or change the live checkpoint.
The old feature hashes come from the immutable cache; the new feature vectors
are freshly simulated over the same full MaleCNS CSR graph and seed/window.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.probe_malecns_feature_window import select_probe_samples
from scripts.train_neural_go_readout import load_cache
from src.go_engine import GoState
from src.go_neural_encoding import encode_go_state
from src.malecns_dynamics import (
    LIFParameters,
    go_encoding_to_spatial_stimulus,
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
    parser.add_argument("--samples", type=int, default=12)
    args = parser.parse_args()

    _, _, records, manifest = load_cache(args.cache)
    selected, selection = select_probe_samples(records, limit=args.samples)
    params = LIFParameters(**manifest["generator"]["lif_parameters"])
    graph = load_graph_assets(args.graph)
    rows = []
    for index, record in enumerate(selected, 1):
        encoding = encode_go_state(GoState.from_dict(record["state"]))
        stimulus = go_encoding_to_spatial_stimulus(encoding)
        frame = simulate_frame(
            graph, stimulus, parameters=params, seed=int(record["lif_seed"]),
        )
        rows.append({
            "sample_id": record["sample_id"],
            "old_feature_hash": record["hashes"]["features_sha256"],
            "spatial_stimulus_hash": stimulus["stimulus_hash"],
            "new_feature_hash": frame["output_pool"]["features_hash"],
            "spike_count": frame["total_spike_count"],
            "synaptic_event_count": frame["synaptic_event_count"],
            "wall_time_ms": frame["wall_time_ms"],
        })
        print(f"simulated {index}/{len(selected)}", file=sys.stderr, flush=True)
    old_counts = Counter(row["old_feature_hash"] for row in rows)
    new_counts = Counter(row["new_feature_hash"] for row in rows)
    print(json.dumps({
        "schema": "stage8-spatial-full-graph-probe-v1",
        "scope": "bounded diagnostic; not a policy or Go-strength result",
        "dataset_hash": manifest["dataset_hash"],
        "selection": selection,
        "lif_parameters": asdict(params),
        "old_unique_features": len(old_counts),
        "new_unique_features": len(new_counts),
        "new_total_spikes": sum(row["spike_count"] for row in rows),
        "new_total_synaptic_events": sum(row["synaptic_event_count"] for row in rows),
        "rows": rows,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
