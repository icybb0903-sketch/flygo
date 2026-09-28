"""Counterfactual signal-dependence audit for a trained MaleCNS Go readout.

The same frozen checkpoint is evaluated on the same held-out natural point
states with (a) real cached MaleCNS output, (b) zeroed output, and (c)
state-shuffled output. No control is represented as a real neural frame.
This is an offline diagnostic, not a claim about biological topology.
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

from scripts.train_neural_go_readout import canonical_bytes, load_cache, object_hash
from src.go_neural_encoding import PASS_ACTION_INDEX
from src.malecns_policy import load_policy_checkpoint


SCHEMA = "malecns-go-neural-signal-ablation-v1"


def point_predictions(
    features: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    legal_masks: np.ndarray,
) -> np.ndarray:
    """Apply the checkpoint's point logits without fabricating neural frames."""
    if features.ndim != 2 or features.shape[1] != weights.shape[1]:
        raise ValueError("feature shape does not match checkpoint")
    if legal_masks.shape != (features.shape[0], PASS_ACTION_INDEX + 1):
        raise ValueError("legal mask shape does not match feature rows")
    legal_points = legal_masks[:, :PASS_ACTION_INDEX].astype(bool)
    if not np.all(legal_points.any(axis=1)):
        raise ValueError("a selected state has no legal point action")
    scores = features.astype(np.float64) @ weights[:PASS_ACTION_INDEX].astype(np.float64).T
    scores += bias[:PASS_ACTION_INDEX].astype(np.float64)
    return np.argmax(np.where(legal_points, scores, -np.inf), axis=1)


def audit(
    cache: Path,
    checkpoint_directory: Path,
    *,
    permutation_seed: int = 20260923,
    permutation_count: int = 128,
) -> dict[str, object]:
    if type(permutation_seed) is not int or permutation_seed < 0:
        raise ValueError("permutation_seed must be a nonnegative integer")
    if type(permutation_count) is not int or permutation_count <= 0:
        raise ValueError("permutation_count must be positive")
    features, labels, records, manifest = load_cache(cache)
    checkpoint = load_policy_checkpoint(checkpoint_directory)
    if checkpoint.training_info.get("dataset_hash") != manifest["dataset_hash"]:
        raise ValueError("checkpoint dataset hash does not match cache")
    if checkpoint.training_info.get("pass_control") != "rule-after-opponent-pass":
        raise ValueError("this audit requires the point-readout + rule-PASS architecture")

    selected_rows = np.asarray([
        record["split"] == "validation"
        and record.get("supervision_provenance", {}).get("kind") == "natural-teacher-turn"
        and label != PASS_ACTION_INDEX
        for record, label in zip(records, labels, strict=True)
    ], dtype=bool)
    if not np.any(selected_rows):
        raise ValueError("cache has no held-out natural point examples")
    x = features[selected_rows]
    y = labels[selected_rows]
    masks = np.asarray(
        [record["legal_mask"] for record, keep in zip(records, selected_rows, strict=True) if keep],
        dtype=np.uint8,
    )
    intact = point_predictions(x, checkpoint.weights, checkpoint.bias, masks)
    zero = point_predictions(
        np.zeros_like(x), checkpoint.weights, checkpoint.bias, masks
    )
    intact_matches = int(np.sum(intact == y))
    zero_matches = int(np.sum(zero == y))

    rng = np.random.default_rng(permutation_seed)
    shuffled_matches: list[int] = []
    for _ in range(permutation_count):
        permuted = x[rng.permutation(len(x))]
        predictions = point_predictions(
            permuted, checkpoint.weights, checkpoint.bias, masks
        )
        shuffled_matches.append(int(np.sum(predictions == y)))
    count = int(len(y))
    report: dict[str, object] = {
        "schema": SCHEMA,
        "scope": "offline held-out natural point decisions; not a game or biological-topology result",
        "checkpoint_hash": checkpoint.checkpoint_hash,
        "dataset_hash": manifest["dataset_hash"],
        "selection": {
            "split": "validation",
            "supervision_kind": "natural-teacher-turn",
            "action_space": "legal point moves 0..80; rule PASS excluded",
            "sample_count": count,
        },
        "controls": {
            "intact": "same frozen readout with state-matched cached MaleCNS features",
            "zero": "same frozen readout and bias with 128 features set to zero; no neural frame fabricated",
            "shuffled": "same frozen readout with held-out feature rows permuted independently of state",
            "permutation_seed": permutation_seed,
            "permutation_count": permutation_count,
        },
        "results": {
            "intact_matches": intact_matches,
            "intact_agreement_rate": intact_matches / count,
            "zero_matches": zero_matches,
            "zero_agreement_rate": zero_matches / count,
            "intact_minus_zero_percentage_points": 100.0 * (intact_matches - zero_matches) / count,
            "shuffled_matches": shuffled_matches,
            "shuffled_mean_matches": float(np.mean(shuffled_matches)),
            "shuffled_max_matches": max(shuffled_matches),
            "shuffled_mean_agreement_rate": float(np.mean(shuffled_matches)) / count,
            "intact_minus_shuffled_mean_percentage_points": 100.0 * (
                intact_matches - float(np.mean(shuffled_matches))
            ) / count,
            "empirical_tail_probability_shuffled_ge_intact": (
                1 + sum(value >= intact_matches for value in shuffled_matches)
            ) / (1 + permutation_count),
        },
        "interpretation_limit": (
            "Signal dependence is not proof that biological MaleCNS wiring is better "
            "than a matched rewired or artificial network."
        ),
    }
    report["report_sha256"] = object_hash(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--permutation-seed", type=int, default=20260923)
    parser.add_argument("--permutation-count", type=int, default=128)
    args = parser.parse_args()
    report = audit(
        args.cache,
        args.checkpoint,
        permutation_seed=args.permutation_seed,
        permutation_count=args.permutation_count,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_bytes(report) + b"\n")
    print(json.dumps({key: value for key, value in report.items() if key != "results"} | {
        "results": {key: value for key, value in report["results"].items() if key != "shuffled_matches"}
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
