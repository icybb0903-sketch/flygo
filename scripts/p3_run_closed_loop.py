"""Run the deterministic P3 closed-loop suite offline."""

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

from src.closed_loop import ClosedLoopError, run_closed_loop_suite
from src.p1_graph import GraphDataError, canonical_bytes, load_verified_graph
from src.toy_dynamics import DynamicsError


SCRIPT_VERSION = "1.0.0"
P1_SUBGRAPH = PROJECT_ROOT / "data" / "p1" / "subgraph.json"
P1_PROVENANCE = PROJECT_ROOT / "data" / "p1" / "provenance.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "p3" / "run.json"


def deterministic_bytes(result: dict[str, Any]) -> bytes:
    """Canonical bytes intentionally exclude the outer generated timestamp."""
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
        first = run_closed_loop_suite(graph)
        second = run_closed_loop_suite(graph)
        if deterministic_bytes(first) != deterministic_bytes(second):
            raise ClosedLoopError("repeated in-process closed-loop results differ")
        first["checks"]["repeat_call_identical"] = True
        encoded = deterministic_bytes(first)
        digest = hashlib.sha256(encoded).hexdigest()
        document = {
            "generated_at_utc": datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
            "result": first,
            "result_bytes": len(encoded),
            "result_sha256_excluding_generated_at": digest,
            "schema_version": 1,
            "script_version": SCRIPT_VERSION,
        }
        _write_json_atomic(OUTPUT_PATH, document)
    except (GraphDataError, DynamicsError, ClosedLoopError, OSError, ValueError) as exc:
        print(f"P3_FAIL error={type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    summaries = first["summaries"]
    print(first["status"].replace("PASS", "P3_PASS").replace("FAIL", "P3_FAIL"))
    print(f"p1_sha256={graph.canonical_sha256}")
    print(f"result_sha256_excluding_generated_at={digest}")
    print(f"result_bytes={len(encoded)}")
    for condition in first["conditions"]:
        summary = summaries[condition]
        print(
            f"{condition} trials={summary['trial_count']} "
            f"correct={summary['correct_count']} abstain={summary['abstain_count']} "
            f"accuracy={summary['accuracy']} coverage={summary['decision_coverage']}"
        )
    print(f"checks_passed={sum(first['checks'].values())}/{len(first['checks'])}")
    print(f"output={OUTPUT_PATH.relative_to(PROJECT_ROOT).as_posix()}")
    return 0 if first["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
