"""Recompute new-seed imitation after excluding every cached complete-state overlap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_neural_go_readout import canonical_bytes, load_cache, object_hash


def audit(cache: Path, report_path: Path) -> dict[str, object]:
    _, _, cached_records, dataset = load_cache(cache)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    claimed_hash = report.pop("report_sha256", None)
    if object_hash(report) != claimed_hash:
        raise ValueError("evaluation report hash does not match contents")
    cached_states = {row["hashes"]["state_sha256"] for row in cached_records}
    cached_games = {row["game_id"] for row in cached_records}
    evaluated = report["records"]
    if not evaluated or any(type(row.get("teacher_match")) is not bool for row in evaluated):
        raise ValueError("evaluation records need explicit teacher_match booleans")
    overlap = [row for row in evaluated if row["state_sha256"] in cached_states]
    independent = [row for row in evaluated if row["state_sha256"] not in cached_states]
    game_overlap = sorted({row["game_id"] for row in evaluated} & cached_games)
    if not independent:
        raise ValueError("evaluation contains no independent positions")
    matches = sum(row["teacher_match"] for row in independent)
    result = {
        "schema": "stage8-unseen-position-overlap-audit-v1",
        "dataset_hash": dataset["dataset_hash"],
        "source_report_sha256": claimed_hash,
        "source_checkpoint_hash": report["checkpoint"]["hash"],
        "original_count": len(evaluated),
        "original_matches": sum(row["teacher_match"] for row in evaluated),
        "cached_state_overlap_count": len(overlap),
        "cached_state_overlap_matches": sum(row["teacher_match"] for row in overlap),
        "cached_game_id_overlap_count": len(game_overlap),
        "cached_game_id_overlaps": game_overlap,
        "independent_count": len(independent),
        "independent_matches": matches,
        "independent_agreement_rate": matches / len(independent),
        "exclusion_rule": "all complete-state SHA-256 hashes in the verified cache, including validation",
    }
    result["audit_sha256"] = object_hash(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.cache, args.report)
    if args.output is not None:
        if args.output.exists():
            raise ValueError("audit output path already exists")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(canonical_bytes(result) + b"\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
