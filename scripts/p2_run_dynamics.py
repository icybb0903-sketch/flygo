"""Run the deterministic P2 toy-dynamics acceptance suite offline."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.p1_graph import GraphDataError, canonical_bytes, load_verified_graph
from src.toy_dynamics import DynamicsError, run_experiment_suite


SCRIPT_VERSION = "1.0.0"
P1_SUBGRAPH = PROJECT_ROOT / "data" / "p1" / "subgraph.json"
P1_PROVENANCE = PROJECT_ROOT / "data" / "p1" / "provenance.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "p2" / "run.json"


def deterministic_bytes(result: dict[str, Any]) -> bytes:
    return canonical_bytes(result)


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def main() -> int:
    try:
        graph = load_verified_graph(P1_SUBGRAPH, P1_PROVENANCE)
        first = run_experiment_suite(graph)
        second = run_experiment_suite(graph)
        if deterministic_bytes(first) != deterministic_bytes(second):
            raise DynamicsError("repeated in-process experiment results differ")
        first["checks"]["repeat_call_identical"] = True
        encoded = deterministic_bytes(first)
        digest = hashlib.sha256(encoded).hexdigest()
        document = {
            "generated_at_utc": datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
            "result": first,
            "result_bytes": len(encoded),
            "result_sha256": digest,
            "schema_version": 1,
            "script_version": SCRIPT_VERSION,
        }
        _write_json_atomic(OUTPUT_PATH, document)
    except (GraphDataError, DynamicsError, OSError, ValueError) as exc:
        print(f"P2_FAIL error={type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    experiments = first["experiments"]
    print("P2_PASS")
    print(f"p1_sha256={graph.canonical_sha256}")
    print(f"result_sha256={digest}")
    print(f"result_bytes={len(encoded)}")
    print(
        "A_no_stimulus "
        f"source_spikes={experiments['A_no_stimulus']['summary']['source_spike_count']} "
        f"downstream_spikes={experiments['A_no_stimulus']['summary']['downstream_spike_count']} "
        f"downstream_peak={experiments['A_no_stimulus']['summary']['downstream_peak_potential']}"
    )
    print(
        "B_stimulated_real_edges "
        f"source_spikes={experiments['B_stimulated_real_edges']['summary']['source_spike_count']} "
        f"downstream_spikes={experiments['B_stimulated_real_edges']['summary']['downstream_spike_count']} "
        f"downstream_peak={experiments['B_stimulated_real_edges']['summary']['downstream_peak_potential']}"
    )
    print(
        "C_stimulated_edges_disconnected "
        f"source_spikes={experiments['C_stimulated_edges_disconnected']['summary']['source_spike_count']} "
        f"downstream_spikes={experiments['C_stimulated_edges_disconnected']['summary']['downstream_spike_count']} "
        f"downstream_peak={experiments['C_stimulated_edges_disconnected']['summary']['downstream_peak_potential']}"
    )
    print(f"all_states_bounded={first['checks']['all_states_bounded']}")
    print(f"observed_max_path_edges={first['topology']['observed_max_path_edges']}")
    print(f"output={OUTPUT_PATH.relative_to(PROJECT_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
