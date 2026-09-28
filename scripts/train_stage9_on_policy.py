"""Train an isolated DAgger-style correction readout on real MaleCNS frames.

This mixes the original *training split only* with verified model-visited
white-turn states from a separate failed game report.  The old validation
split is used for diagnostics, not fitting or hyperparameter selection.
The MaleCNS graph and LIF dynamics remain frozen.
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

from scripts.train_neural_go_readout import (
    evaluate_controlled_readout, file_hash, load_cache, object_hash,
)
from scripts.train_stage8_legal_softmax import fit_legal_softmax
from src.go_neural_encoding import PASS_ACTION_INDEX
from src.malecns_policy import load_policy_checkpoint, write_policy_checkpoint


METHOD = "on_policy_teacher_correction_weighted_legal_softmax_v1"


def load_on_policy_cache(directory: Path, *, base_dataset_hash: str) -> tuple[np.ndarray, np.ndarray, list[dict], dict]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    claimed = manifest.pop("dataset_hash", None)
    if object_hash(manifest) != claimed:
        raise ValueError("on-policy manifest hash mismatch")
    manifest["dataset_hash"] = claimed
    if manifest.get("schema") != "stage9-on-policy-male-cns-correction-cache-v1":
        raise ValueError("unsupported on-policy cache schema")
    if manifest["base_cache_dataset_hash"] != base_dataset_hash:
        raise ValueError("on-policy cache descends from another base dataset")
    for name, expected_hash in manifest["artifacts"].items():
        if file_hash(directory / name) != expected_hash:
            raise ValueError(f"on-policy artifact hash mismatch: {name}")
    features = np.load(directory / "features.npy", allow_pickle=False)
    labels = np.load(directory / "labels.npy", allow_pickle=False)
    rows = [json.loads(line) for line in (directory / "records.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if features.shape != (len(rows), 128) or labels.shape != (len(rows),) or len(rows) != manifest["sample_count"]:
        raise ValueError("on-policy cache row count or shape mismatch")
    if any(row["feature_row"] != index or row["teacher_action"] != int(labels[index]) for index, row in enumerate(rows)):
        raise ValueError("on-policy labels and records are not aligned")
    return features, labels, rows, manifest


def train(
    *,
    base_cache: Path,
    on_policy_cache: Path,
    source_checkpoint: Path,
    output_checkpoint: Path,
) -> dict[str, object]:
    if output_checkpoint.exists():
        raise ValueError("output checkpoint directory already exists")
    base_features, base_labels, base_rows, base_manifest = load_cache(base_cache)
    on_features, on_labels, on_rows, on_manifest = load_on_policy_cache(
        on_policy_cache, base_dataset_hash=base_manifest["dataset_hash"]
    )
    source = load_policy_checkpoint(source_checkpoint)
    if source.checkpoint_hash != on_manifest["source_checkpoint_hash"]:
        raise ValueError("source checkpoint does not match on-policy game report")
    if source.training_info.get("dataset_hash") != base_manifest["dataset_hash"]:
        raise ValueError("source checkpoint does not match base cache")
    validation_states = {
        row["hashes"]["state_sha256"] for row in base_rows if row["split"] == "validation"
    }
    validation_features = {
        row["hashes"]["features_sha256"] for row in base_rows if row["split"] == "validation"
    }
    if any(row["state_sha256"] in validation_states or row["features_hash"] in validation_features for row in on_rows):
        raise ValueError("on-policy sample overlaps base validation data")
    base_fit = np.asarray([
        row["split"] == "train"
        and row.get("supervision_provenance", {}).get("kind") == "natural-teacher-turn"
        and label != PASS_ACTION_INDEX
        for row, label in zip(base_rows, base_labels, strict=True)
    ], dtype=bool)
    base_validation = np.asarray([
        row["split"] == "validation"
        and row.get("supervision_provenance", {}).get("kind") == "natural-teacher-turn"
        and label != PASS_ACTION_INDEX
        for row, label in zip(base_rows, base_labels, strict=True)
    ], dtype=bool)
    features = np.vstack((base_features[base_fit], on_features))
    labels = np.concatenate((base_labels[base_fit], on_labels))
    legal_masks = np.asarray(
        [row["legal_mask"] for row, use in zip(base_rows, base_fit, strict=True) if use]
        + [row["legal_mask"] for row in on_rows],
        dtype=bool,
    )
    # Fixed exploratory correction recipe: 2x every model-visited state and
    # 8x only when the teacher avoids >=3 additional immediate black captures.
    # These weights do not come from old validation labels or game outcomes.
    on_weights = np.asarray([
        8.0 if row["teacher_safer_by_at_least_3"] else 2.0
        for row in on_rows
    ])
    sample_weights = np.concatenate((np.ones(int(base_fit.sum())), on_weights))
    weights, bias, optimizer = fit_legal_softmax(
        features, labels, legal_masks,
        steps=400, learning_rate=0.03, l2=0.5,
        sample_weights=sample_weights,
    )
    validation_records = [row for row, use in zip(base_rows, base_validation, strict=True) if use]
    validation_metrics = evaluate_controlled_readout(
        base_features[base_validation], base_labels[base_validation], validation_records,
        weights, bias, pass_control="rule-after-opponent-pass",
    )
    source_validation_metrics = evaluate_controlled_readout(
        base_features[base_validation], base_labels[base_validation], validation_records,
        source.weights, source.bias, pass_control="rule-after-opponent-pass",
    )
    on_predictions = np.argmax(
        np.where(
            np.asarray([row["legal_mask"][:PASS_ACTION_INDEX] for row in on_rows], dtype=bool),
            on_features @ weights[:PASS_ACTION_INDEX].T + bias[:PASS_ACTION_INDEX],
            -np.inf,
        ), axis=1,
    )
    composite_dataset_hash = object_hash({
        "schema": "stage9-combined-training-source-v1",
        "base_dataset_hash": base_manifest["dataset_hash"],
        "on_policy_dataset_hash": on_manifest["dataset_hash"],
    })
    training_info = dict(source.training_info)
    training_info.update({
        "schema": "stage9-on-policy-training-v1",
        "method": METHOD,
        "algorithm": METHOD,
        "dataset_id": composite_dataset_hash,
        "dataset_hash": composite_dataset_hash,
        "sample_count": base_manifest["sample_count"] + on_manifest["sample_count"],
        "fit_sample_count": len(features),
        "training_sources": {
            "base_cache_dataset_hash": base_manifest["dataset_hash"],
            "base_fit_natural_point_count": int(base_fit.sum()),
            "base_validation_count_not_used_for_fit": int(base_validation.sum()),
            "on_policy_cache_dataset_hash": on_manifest["dataset_hash"],
            "on_policy_source_game_report_sha256": on_manifest["source_game_report_sha256"],
            "on_policy_source_checkpoint_hash": on_manifest["source_checkpoint_hash"],
            "on_policy_fit_count": len(on_rows),
            "on_policy_base_validation_state_overlap_count": 0,
            "on_policy_base_validation_feature_overlap_count": 0,
        },
        "optimizer": {
            "name": "full_batch_adam",
            "steps": 400,
            "learning_rate": 0.03,
            "l2": 0.5,
            "base_sample_weight": 1.0,
            "on_policy_sample_weight": 2.0,
            "on_policy_high_tactical_risk_weight": 8.0,
            "high_tactical_risk_definition": "model_max_next_black_capture - teacher_max_next_black_capture >= 3",
            **optimizer,
        },
        "metrics": {
            "old_base_validation_natural_point": validation_metrics,
            "source_checkpoint_old_base_validation_natural_point": source_validation_metrics,
            "on_policy_training_teacher_agreement_rate": float(np.mean(on_predictions == on_labels)),
        },
        "model_selection_warning": (
            "Correction samples come from four previously inspected failed games; "
            "only wholly new game seeds can assess game improvement."
        ),
    })
    checkpoint_hash = write_policy_checkpoint(
        output_checkpoint,
        weights=weights,
        bias=bias,
        model_id=base_manifest["graph"]["model_id"],
        graph_manifest_sha256=base_manifest["graph"]["manifest_sha256"],
        output_pool_group_hash=base_manifest["output_pool"]["group_hash"],
        training_status="trained",
        training_info=training_info,
    )
    return {
        "schema": "stage9-on-policy-candidate-training-result-v1",
        "checkpoint": str(output_checkpoint.resolve()),
        "checkpoint_hash": checkpoint_hash,
        "composite_dataset_hash": composite_dataset_hash,
        "base_fit_count": int(base_fit.sum()),
        "on_policy_fit_count": len(on_rows),
        "high_tactical_risk_count": int(np.sum(on_weights == 8.0)),
        "old_base_validation_natural_point": validation_metrics,
        "source_old_base_validation_natural_point": source_validation_metrics,
        "on_policy_training_teacher_agreement_rate": float(np.mean(on_predictions == on_labels)),
        "promotion_status": "not_promoted_requires_new_seed_full_game_evaluation",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-cache", type=Path, required=True)
    parser.add_argument("--on-policy-cache", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(train(
        base_cache=args.base_cache,
        on_policy_cache=args.on_policy_cache,
        source_checkpoint=args.source_checkpoint,
        output_checkpoint=args.output_checkpoint,
    ), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
