"""Read-only nonlinear readout probe on frozen MaleCNS output features.

Fits a single predeclared RBF-kernel ridge model on natural non-PASS training
rows. The kernel scale comes only from training feature distances. This does
not write a deployable checkpoint or claim a fly learned Go.
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

from scripts.probe_stage8_adapter import _fit_and_score
from scripts.train_neural_go_readout import load_cache
from src.go_neural_encoding import PASS_ACTION_INDEX


def squared_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_sq = np.sum(left * left, axis=1)[:, None]
    right_sq = np.sum(right * right, axis=1)[None, :]
    return np.maximum(0.0, left_sq + right_sq - 2.0 * left @ right.T)


def probe(cache: Path, ridge: float) -> dict[str, object]:
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge must be finite and positive")
    features, labels, records, manifest = load_cache(cache)
    masks = np.asarray([record["legal_mask"] for record in records], dtype=bool)
    natural = np.asarray([
        record.get("supervision_provenance", {}).get("kind") == "natural-teacher-turn"
        and label != PASS_ACTION_INDEX
        for record, label in zip(records, labels, strict=True)
    ], dtype=bool)
    train = np.asarray([record["split"] == "train" for record in records]) & natural
    validation = np.asarray([record["split"] == "validation" for record in records]) & natural
    x_train = np.asarray(features[train], dtype=np.float64)
    x_validation = np.asarray(features[validation], dtype=np.float64)
    distances = squared_distances(x_train, x_train)
    nonzero = distances[np.triu_indices(len(x_train), 1)]
    nonzero = nonzero[nonzero > 1e-14]
    if not len(nonzero):
        raise ValueError("training features have no nonzero pairwise distance")
    median_sq_distance = float(np.median(nonzero))
    kernel_train = np.exp(-distances / median_sq_distance)
    targets = np.eye(PASS_ACTION_INDEX, dtype=np.float64)[labels[train]]
    coefficients = np.linalg.solve(
        kernel_train + ridge * np.eye(len(x_train)), targets
    )
    kernel_validation = np.exp(
        -squared_distances(x_validation, x_train) / median_sq_distance
    )
    scores = kernel_validation @ coefficients
    legal_scores = np.where(masks[validation, :PASS_ACTION_INDEX], scores, -np.inf)
    selected = np.argmax(legal_scores, axis=1)
    return {
        "schema": "stage8-nonlinear-readout-probe-v1",
        "scope": "read-only diagnostic; no deployable checkpoint or final Go-strength claim",
        "dataset_hash": manifest["dataset_hash"],
        "method": "RBF kernel ridge; bandwidth from median nonzero train pair distance",
        "ridge": ridge,
        "median_train_squared_distance": median_sq_distance,
        "train_natural_point_count": int(train.sum()),
        "validation_natural_point_count": int(validation.sum()),
        "linear_baseline": _fit_and_score(
            features, labels, masks, train, validation, ridge
        ),
        "rbf_validation_teacher_matches": int(np.sum(selected == labels[validation])),
        "rbf_validation_teacher_agreement_rate": float(
            np.mean(selected == labels[validation])
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--ridge", type=float, default=0.1)
    args = parser.parse_args()
    print(json.dumps(probe(args.cache, args.ridge), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
