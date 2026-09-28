"""Train an isolated legal-point softmax readout on real cached MaleCNS features.

The graph, stimulation, and output pools are frozen.  Only the external linear
readout is trained.  Existing checkpoints and the live Go preview are untouched.
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
    _record_rule_pass_gate,
    evaluate_controlled_readout,
    load_cache,
    object_hash,
)
from src.go_neural_encoding import ACTION_COUNT, PASS_ACTION_INDEX
from src.malecns_policy import load_policy_checkpoint, write_policy_checkpoint


METHOD = "legal_masked_softmax_adam_natural_points_v1"


def fit_legal_softmax(
    features: np.ndarray,
    labels: np.ndarray,
    legal_masks: np.ndarray,
    *,
    steps: int = 400,
    learning_rate: float = 0.03,
    l2: float = 0.005,
    sample_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Fit a deterministic point-only classifier; return raw-feature weights."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    masks = np.asarray(legal_masks, dtype=bool)
    if x.ndim != 2 or x.shape[1] != 128 or y.shape != (len(x),):
        raise ValueError("expected [N,128] features and [N] labels")
    if masks.shape != (len(x), ACTION_COUNT) or not len(x):
        raise ValueError("legal masks must have shape [N,82]")
    if np.any((y < 0) | (y >= PASS_ACTION_INDEX)) or np.any(~masks[np.arange(len(x)), y]):
        raise ValueError("every training label must be a legal point move")
    if type(steps) is not int or steps <= 0 or not 0 < learning_rate <= 1 or not 0 <= l2 <= 1:
        raise ValueError("invalid optimizer parameters")
    example_weights = (
        np.ones(len(x), dtype=np.float64)
        if sample_weights is None else np.asarray(sample_weights, dtype=np.float64)
    )
    if (
        example_weights.shape != (len(x),)
        or not np.all(np.isfinite(example_weights))
        or np.any(example_weights <= 0)
    ):
        raise ValueError("sample_weights must be finite positive values for every row")
    total_weight = float(example_weights.sum())

    center = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 1e-5)
    z = (x - center) / scale
    point_masks = masks[:, :PASS_ACTION_INDEX]
    weights = np.zeros((PASS_ACTION_INDEX, 128), dtype=np.float64)
    bias = np.zeros(PASS_ACTION_INDEX, dtype=np.float64)
    first_moment_w = np.zeros_like(weights)
    second_moment_w = np.zeros_like(weights)
    first_moment_b = np.zeros_like(bias)
    second_moment_b = np.zeros_like(bias)
    loss = float("nan")

    for step in range(1, steps + 1):
        logits = np.where(point_masks, z @ weights.T + bias, -1e9)
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        loss = float(
            -np.dot(
                example_weights,
                np.log(np.maximum(probabilities[np.arange(len(x)), y], 1e-300)),
            ) / total_weight
            + 0.5 * l2 * np.sum(weights * weights)
        )
        gradient = probabilities
        gradient[np.arange(len(x)), y] -= 1.0
        gradient *= (example_weights / total_weight)[:, None]
        grad_w = gradient.T @ z + l2 * weights
        grad_b = gradient.sum(axis=0)
        first_moment_w = 0.9 * first_moment_w + 0.1 * grad_w
        second_moment_w = 0.999 * second_moment_w + 0.001 * (grad_w * grad_w)
        first_moment_b = 0.9 * first_moment_b + 0.1 * grad_b
        second_moment_b = 0.999 * second_moment_b + 0.001 * (grad_b * grad_b)
        correction_w = (first_moment_w / (1.0 - 0.9**step)) / (
            np.sqrt(second_moment_w / (1.0 - 0.999**step)) + 1e-8
        )
        correction_b = (first_moment_b / (1.0 - 0.9**step)) / (
            np.sqrt(second_moment_b / (1.0 - 0.999**step)) + 1e-8
        )
        weights -= learning_rate * correction_w
        bias -= learning_rate * correction_b

    raw_weights = weights / scale
    raw_bias = bias - raw_weights @ center
    full_weights = np.zeros((ACTION_COUNT, 128), dtype=np.float32)
    full_bias = np.zeros(ACTION_COUNT, dtype=np.float32)
    full_weights[:PASS_ACTION_INDEX] = raw_weights.astype(np.float32)
    full_bias[:PASS_ACTION_INDEX] = raw_bias.astype(np.float32)
    if not np.all(np.isfinite(full_weights)) or not np.all(np.isfinite(full_bias)):
        raise ValueError("optimizer produced nonfinite checkpoint parameters")
    return full_weights, full_bias, {"train_final_objective": loss}


def train_candidate(
    *,
    cache: Path,
    base_checkpoint: Path,
    checkpoint: Path,
    steps: int = 400,
    learning_rate: float = 0.03,
    l2: float = 0.005,
    cv_report: Path | None = None,
) -> dict[str, object]:
    if checkpoint.resolve() == base_checkpoint.resolve() or checkpoint.exists():
        raise ValueError("candidate output must be a new directory, not an existing checkpoint")
    features, labels, records, dataset = load_cache(cache)
    base = load_policy_checkpoint(base_checkpoint)
    if base.training_info.get("dataset_hash") != dataset["dataset_hash"]:
        raise ValueError("base checkpoint was not trained from this cache")
    if base.training_info.get("pass_control") != "rule-after-opponent-pass":
        raise ValueError("base checkpoint must declare the rule PASS gate")
    cv_metadata = None
    if cv_report is not None:
        cv = json.loads(cv_report.read_text(encoding="utf-8"))
        claimed_hash = cv.pop("report_sha256", None)
        if (
            object_hash(cv) != claimed_hash
            or cv.get("schema") not in (
                "stage8-softmax-train-only-group-cv-v1",
                "stage8-softmax-train-only-group-cv-v2",
            )
            or cv.get("dataset_hash") != dataset["dataset_hash"]
            or cv.get("selected_l2") != l2
            or cv.get("steps") != steps
        ):
            raise ValueError("CV report does not verify this dataset and training configuration")
        cv_metadata = {"path": str(cv_report.resolve()), "report_sha256": claimed_hash}
    train = np.asarray([row["split"] == "train" for row in records], dtype=bool)
    validation = ~train
    natural_point = np.asarray([
        row.get("supervision_provenance", {}).get("kind") == "natural-teacher-turn"
        and label != PASS_ACTION_INDEX
        for row, label in zip(records, labels, strict=True)
    ], dtype=bool)
    fit_rows = train & natural_point
    if not np.any(fit_rows) or not np.any(validation & natural_point):
        raise ValueError("cache has no natural point samples in one split")
    masks = np.asarray([row["legal_mask"] for row in records], dtype=bool)
    weights, bias, optimizer = fit_legal_softmax(
        features[fit_rows], labels[fit_rows], masks[fit_rows],
        steps=steps, learning_rate=learning_rate, l2=l2,
    )

    def score(rows: np.ndarray, candidate_weights: np.ndarray, candidate_bias: np.ndarray) -> dict[str, object]:
        selected_records = [record for record, selected in zip(records, rows, strict=True) if selected]
        return evaluate_controlled_readout(
            features[rows], labels[rows], selected_records,
            candidate_weights, candidate_bias,
            pass_control="rule-after-opponent-pass",
        )

    comparison = {
        "fit_natural_point": score(fit_rows, weights, bias),
        "validation_natural_point": score(validation & natural_point, weights, bias),
        "base_validation_natural_point": score(
            validation & natural_point, base.weights, base.bias
        ),
        "validation_all": score(validation, weights, bias),
        "rule_pass_validation_trigger_count": int(sum(
            _record_rule_pass_gate(row)
            for row, selected in zip(records, validation, strict=True) if selected
        )),
    }
    pass_response = np.asarray([
        row.get("supervision_provenance", {}).get("kind")
        == "constructed-opponent-pass-response"
        for row in records
    ], dtype=bool)
    natural = np.asarray([
        row.get("supervision_provenance", {}).get("kind")
        == "natural-teacher-turn"
        for row in records
    ], dtype=bool)
    candidate_metrics = {
        "train": score(train, weights, bias),
        "validation": score(validation, weights, bias),
        "stratified": {
            "train": {
                "natural": score(train & natural, weights, bias),
                "pass_response": score(train & pass_response, weights, bias),
            },
            "validation": {
                "natural": score(validation & natural, weights, bias),
                "pass_response": score(validation & pass_response, weights, bias),
            },
        },
        "model_selection_metric": "stratified.validation.natural.teacher_agreement_rate",
        "overall_metrics_include_constructed_pass_response": bool(np.any(pass_response)),
    }
    training_info = dict(base.training_info)
    training_info.pop("ridge", None)
    training_info.update({
        "method": METHOD,
        "algorithm": METHOD,
        "fit_sample_count": int(fit_rows.sum()),
        "metrics": candidate_metrics,
        "optimizer": {
            "name": "full_batch_adam",
            "steps": steps,
            "learning_rate": learning_rate,
            "l2": l2,
            "train_only_feature_standardization": True,
            "train_only_cv_selection": cv_metadata,
            **optimizer,
        },
        "development_comparison": comparison,
        "model_selection_warning": (
            "Historical validation splits have been inspected repeatedly; "
            "independent new-seed evaluation is required before promotion."
        ),
    })
    checkpoint_hash = write_policy_checkpoint(
        checkpoint,
        weights=weights,
        bias=bias,
        model_id=dataset["graph"]["model_id"],
        graph_manifest_sha256=dataset["graph"]["manifest_sha256"],
        output_pool_group_hash=dataset["output_pool"]["group_hash"],
        training_status="trained",
        training_info=training_info,
    )
    return {
        "schema": "stage8-legal-softmax-training-result-v1",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_hash": checkpoint_hash,
        "dataset_hash": dataset["dataset_hash"],
        "fit_sample_count": int(fit_rows.sum()),
        "comparison": comparison,
        "optimizer": training_info["optimizer"],
        "promotion_status": "not_promoted_requires_independent_evaluation",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--l2", type=float, default=0.005)
    parser.add_argument("--cv-report", type=Path)
    args = parser.parse_args()
    result = train_candidate(
        cache=args.cache,
        base_checkpoint=args.base_checkpoint,
        checkpoint=args.checkpoint,
        steps=args.steps,
        learning_rate=args.learning_rate,
        l2=args.l2,
        cv_report=args.cv_report,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
