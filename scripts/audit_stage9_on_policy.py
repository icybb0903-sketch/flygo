"""Audit Stage9 correction provenance and training-set tactical examples.

This is not a game-strength gate.  New-seed full games remain mandatory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_neural_signal_ablation import point_predictions
from scripts.train_neural_go_readout import canonical_bytes, load_cache, object_hash
from scripts.train_stage9_on_policy import load_on_policy_cache
from src.malecns_policy import load_policy_checkpoint


def audit(
    *, base_cache: Path, on_policy_cache: Path,
    source_checkpoint: Path, candidate_checkpoint: Path,
    game_report: Path | None = None,
) -> dict[str, object]:
    _, _, base_rows, base_manifest = load_cache(base_cache)
    features, labels, rows, on_manifest = load_on_policy_cache(
        on_policy_cache, base_dataset_hash=base_manifest["dataset_hash"]
    )
    source = load_policy_checkpoint(source_checkpoint)
    candidate = load_policy_checkpoint(candidate_checkpoint)
    combined_hash = object_hash({
        "schema": "stage9-combined-training-source-v1",
        "base_dataset_hash": base_manifest["dataset_hash"],
        "on_policy_dataset_hash": on_manifest["dataset_hash"],
    })
    validation_states = {
        row["hashes"]["state_sha256"] for row in base_rows if row["split"] == "validation"
    }
    validation_features = {
        row["hashes"]["features_sha256"] for row in base_rows if row["split"] == "validation"
    }
    overlap_states = {row["state_sha256"] for row in rows} & validation_states
    overlap_features = {row["features_hash"] for row in rows} & validation_features
    masks = np.asarray([row["legal_mask"] for row in rows], dtype=bool)
    old_moves = point_predictions(features, source.weights, source.bias, masks)
    new_moves = point_predictions(features, candidate.weights, candidate.bias, masks)
    high_risk = np.asarray([row["teacher_safer_by_at_least_3"] for row in rows], dtype=bool)
    checks = {
        "source_checkpoint_matches_game": source.checkpoint_hash == on_manifest["source_checkpoint_hash"],
        "candidate_composite_dataset_matches": candidate.training_info.get("dataset_hash") == combined_hash,
        "candidate_declares_both_sources": (
            candidate.training_info.get("training_sources", {}).get("base_cache_dataset_hash")
            == base_manifest["dataset_hash"]
            and candidate.training_info.get("training_sources", {}).get("on_policy_cache_dataset_hash")
            == on_manifest["dataset_hash"]
        ),
        "base_validation_state_overlap_zero": not overlap_states,
        "base_validation_feature_overlap_zero": not overlap_features,
        "on_policy_labels_legal": bool(np.all(masks[np.arange(len(rows)), labels])),
    }
    game_overlap = None
    if game_report is not None:
        game = json.loads(game_report.read_text(encoding="utf-8"))
        claimed_game_hash = game.pop("report_sha256", None)
        if game.get("schema") != "malecns-go-stage7-production-white-v2" or object_hash(game) != claimed_game_hash:
            raise ValueError("game report hash or schema mismatch")
        decision_rows = [
            move for played in game["games"] for move in played["moves"]
            if move["actor"] == "malecns-neural-policy"
        ]
        decision_states = {move["state_sha256_before"] for move in decision_rows}
        base_states = {row["hashes"]["state_sha256"] for row in base_rows}
        on_states = {row["state_sha256"] for row in rows}
        evaluated_seeds = {int(played["seed"]) for played in game["games"]}
        training_seeds = {int(row["game_seed"]) for row in rows}
        game_overlap = {
            "report_sha256": claimed_game_hash,
            "neural_decision_count": len(decision_rows),
            "base_cache_state_overlap_count": len(decision_states & base_states),
            "on_policy_training_state_overlap_count": len(decision_states & on_states),
            "on_policy_training_game_seed_overlap_count": len(evaluated_seeds & training_seeds),
        }
        checks.update({
            "game_checkpoint_matches_candidate": game["checkpoint"]["hash"] == candidate.checkpoint_hash,
            "new_game_base_cache_state_overlap_zero": game_overlap["base_cache_state_overlap_count"] == 0,
            "new_game_on_policy_state_overlap_zero": game_overlap["on_policy_training_state_overlap_count"] == 0,
            "new_game_training_seed_overlap_zero": game_overlap["on_policy_training_game_seed_overlap_count"] == 0,
            "new_game_runtime_guarantees": all(
                move["provenance"].get("teacher_accessed_at_runtime") is False
                and move["provenance"].get("baseline_controller_called") is False
                and move["provenance"].get("fallback_used") is False
                for move in decision_rows
            ),
        })
    result = {
        "schema": "stage9-on-policy-candidate-audit-v1",
        "scope": "provenance and training-set tactics only; no new-seed strength claim",
        "base_dataset_hash": base_manifest["dataset_hash"],
        "on_policy_dataset_hash": on_manifest["dataset_hash"],
        "composite_dataset_hash": combined_hash,
        "source_checkpoint_hash": source.checkpoint_hash,
        "candidate_checkpoint_hash": candidate.checkpoint_hash,
        "checks": checks,
        "integrity_passed": all(checks.values()),
        "on_policy_sample_count": len(rows),
        "source_on_policy_teacher_matches": int(np.sum(old_moves == labels)),
        "candidate_on_policy_teacher_matches": int(np.sum(new_moves == labels)),
        "high_tactical_risk_training_example_count": int(high_risk.sum()),
        "source_high_risk_teacher_matches": int(np.sum(old_moves[high_risk] == labels[high_risk])),
        "candidate_high_risk_teacher_matches": int(np.sum(new_moves[high_risk] == labels[high_risk])),
        "candidate_high_risk_original_blunders_repeated": int(np.sum(
            new_moves[high_risk]
            == np.asarray([row["model_action"] for row in rows])[high_risk]
        )),
        "base_validation_state_overlap_count": len(overlap_states),
        "base_validation_feature_overlap_count": len(overlap_features),
        "new_game_overlap": game_overlap,
    }
    result["report_sha256"] = object_hash(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-cache", type=Path, required=True)
    parser.add_argument("--on-policy-cache", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--game-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("audit output already exists")
    result = audit(
        base_cache=args.base_cache,
        on_policy_cache=args.on_policy_cache,
        source_checkpoint=args.source_checkpoint,
        candidate_checkpoint=args.candidate_checkpoint,
        game_report=args.game_report,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_bytes(result) + b"\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["integrity_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
