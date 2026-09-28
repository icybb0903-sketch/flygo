"""Bounded real-graph probe of label-blind, input-downstream readout pools.

The stimulus remains the current deployed hash adapter. Only the output pool
changes; this is not a trained Go policy or a strength result.
"""

from __future__ import annotations

import argparse
from collections import Counter
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
    LIFParameters, go_encoding_to_stimulus, load_graph_assets, simulate_frame,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=12)
    args = parser.parse_args()
    cache = PROJECT_ROOT / "data" / "training" / "stage8_role_symmetric_pilot"
    graph_path = (
        PROJECT_ROOT / "data" / "malecns" / "runtime"
        / "malecns-v1.0-w5-3acb6434e71160fc"
    )
    _, _, records, manifest = load_cache(cache)
    selected, selection = select_probe_samples(records, limit=args.samples)
    graph = load_graph_assets(graph_path)
    params = LIFParameters(**manifest["generator"]["lif_parameters"])
    rows = []
    for index, record in enumerate(selected, 1):
        stimulus = go_encoding_to_stimulus(
            encode_go_state(GoState.from_dict(record["state"]))
        )
        frame = simulate_frame(
            graph, stimulus, parameters=params, seed=int(record["lif_seed"]),
            output_pool_selection="input-downstream-v1",
        )
        rows.append({
            "sample_id": record["sample_id"],
            "old_feature_hash": record["hashes"]["features_sha256"],
            "new_feature_hash": frame["output_pool"]["features_hash"],
            "new_feature_vector": frame["output_pool"]["features"],
            "output_pool_version": frame["output_pool"]["version"],
            "unique_readout_neurons": frame["output_pool"]["unique_neuron_count"],
            "spike_count": frame["total_spike_count"],
            "synaptic_event_count": frame["synaptic_event_count"],
        })
        print(f"downstream-pool frames {index}/{len(selected)}", file=sys.stderr, flush=True)
    report = {
        "schema": "stage8-downstream-pool-full-graph-probe-v1",
        "scope": "bounded diagnostic only; no trained Go policy",
        "dataset_hash": manifest["dataset_hash"],
        "selection": selection,
        "old_unique_features": len(Counter(row["old_feature_hash"] for row in rows)),
        "new_unique_features": len(Counter(row["new_feature_hash"] for row in rows)),
        "unique_readout_neurons": rows[0]["unique_readout_neurons"],
        "rows": [
            {key: value for key, value in row.items() if key != "new_feature_vector"}
            for row in rows
        ],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
