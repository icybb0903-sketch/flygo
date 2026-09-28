"""Fast, read-only audit of a cached Stage 8 policy candidate.

This intentionally reuses cached neural features.  It never runs the MaleCNS
graph, fits a readout, or writes a checkpoint.  JSON is emitted to stdout so
callers may choose whether and where to persist the result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_neural_go_readout import load_cache
from src.go_engine import BLACK, WHITE
from src.go_neural_encoding import ACTION_COUNT, PASS_ACTION_INDEX
from src.malecns_policy import load_policy_checkpoint


AUDIT_SCHEMA = "malecns-go-stage8-candidate-audit-v1"
PASS_RESPONSE_KIND = "constructed-opponent-pass-response"
HYBRID_ARCHITECTURE = "hybrid-rule-after-opponent-pass-v1"
HYBRID_PASS_CONTROL = "rule-after-opponent-pass"
HYBRID_CONTROLLER_SOURCES = {
    "point_actions": "malecns-linear-readout",
    "pass_after_opponent_pass": "go-rule-pass-gate",
    "pass_when_only_legal": "go-rule-pass-gate",
}
MINIMUM_AGREEMENT_RATE = 0.25
MAXIMUM_TRAIN_VALIDATION_GAP = 0.10


def _colour(record: Mapping[str, Any]) -> str | None:
    value = record.get("teacher_colour")
    if value is None and isinstance(record.get("state"), Mapping):
        value = record["state"].get("to_play")
    if value == BLACK:
        return "black"
    if value == WHITE:
        return "white"
    return None


def _is_pass_response(record: Mapping[str, Any]) -> bool:
    provenance = record.get("supervision_provenance")
    return isinstance(provenance, Mapping) and provenance.get("kind") == PASS_RESPONSE_KIND


def _is_natural(record: Mapping[str, Any]) -> bool:
    provenance = record.get("supervision_provenance")
    if not isinstance(provenance, Mapping):
        # Legacy caches predate explicit provenance and contain natural states.
        return True
    return provenance.get("kind") == "natural-teacher-turn"


def _metric_group(
    rows: np.ndarray,
    labels: np.ndarray,
    legal_masks: np.ndarray,
    raw_predictions: np.ndarray,
    masked_predictions: np.ndarray,
) -> dict[str, Any]:
    count = int(np.count_nonzero(rows))
    pass_rows = rows & (labels == PASS_ACTION_INDEX)
    pass_count = int(np.count_nonzero(pass_rows))
    if count == 0:
        return {
            "sample_count": 0,
            "masked_teacher_agreement_rate": None,
            "legal_action_rate": None,
            "pass_teacher_sample_count": 0,
            "pass_teacher_masked_prediction_rate": None,
        }
    indices = np.flatnonzero(rows)
    legal_rate = np.mean(legal_masks[indices, raw_predictions[indices]])
    return {
        "sample_count": count,
        "masked_teacher_agreement_rate": float(
            np.mean(masked_predictions[rows] == labels[rows])
        ),
        "legal_action_rate": float(legal_rate),
        "pass_teacher_sample_count": pass_count,
        "pass_teacher_masked_prediction_rate": (
            float(np.mean(masked_predictions[pass_rows] == PASS_ACTION_INDEX))
            if pass_count
            else None
        ),
    }


def compute_grouped_metrics(
    features: np.ndarray,
    labels: np.ndarray,
    records: Sequence[Mapping[str, Any]],
    weights: np.ndarray,
    bias: np.ndarray,
) -> dict[str, Any]:
    """Evaluate cached rows, grouped by split, colour, and pass-response origin."""
    legal_masks = np.asarray([record["legal_mask"] for record in records], dtype=bool)
    if legal_masks.shape != (len(records), ACTION_COUNT):
        raise ValueError("sample legal masks must have shape [N, 82]")
    if np.any(~np.any(legal_masks, axis=1)):
        raise ValueError("every sample must expose at least one legal action")
    scores = (
        np.asarray(features, dtype=np.float64)
        @ np.asarray(weights, dtype=np.float64).T
        + np.asarray(bias, dtype=np.float64)
    )
    if scores.shape != legal_masks.shape or labels.shape != (len(records),):
        raise ValueError("cache and checkpoint arrays have incompatible shapes")
    raw_predictions = np.argmax(scores, axis=1)
    masked_predictions = np.argmax(np.where(legal_masks, scores, -np.inf), axis=1)
    colours = np.asarray([_colour(record) for record in records], dtype=object)
    pass_response = np.asarray([_is_pass_response(record) for record in records], dtype=bool)
    natural = np.asarray([_is_natural(record) for record in records], dtype=bool)
    splits = np.asarray([record.get("split") for record in records], dtype=object)
    unknown = sorted({str(item) for item in splits if item not in ("train", "validation")})
    if unknown:
        raise ValueError(f"unsupported sample splits: {unknown}")

    result: dict[str, Any] = {}
    for split in ("train", "validation"):
        split_rows = splits == split
        result[split] = {
            "overall": _metric_group(
                split_rows, labels, legal_masks, raw_predictions, masked_predictions
            ),
            "black": _metric_group(
                split_rows & (colours == "black"),
                labels,
                legal_masks,
                raw_predictions,
                masked_predictions,
            ),
            "white": _metric_group(
                split_rows & (colours == "white"),
                labels,
                legal_masks,
                raw_predictions,
                masked_predictions,
            ),
            "pass_response": _metric_group(
                split_rows & pass_response,
                labels,
                legal_masks,
                raw_predictions,
                masked_predictions,
            ),
            "natural": _metric_group(
                split_rows & natural,
                labels,
                legal_masks,
                raw_predictions,
                masked_predictions,
            ),
        }
    return result


def _rate(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _hybrid_group(
    rows: np.ndarray,
    labels: np.ndarray,
    selected: np.ndarray,
) -> dict[str, Any]:
    count = int(np.count_nonzero(rows))
    agreements = int(np.count_nonzero(rows & (selected == labels)))
    return {
        "sample_count": count,
        "agreement_count": agreements,
        "agreement_rate": _rate(agreements, count),
    }


def audit_hybrid_controller(
    features: np.ndarray,
    labels: np.ndarray,
    records: Sequence[Mapping[str, Any]],
    weights: np.ndarray,
    bias: np.ndarray,
    training_info: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Audit the exact public-state hybrid route, without invoking runtime code."""
    architecture = training_info.get("decision_architecture")
    pass_control = training_info.get("pass_control", "joint-neural")
    if architecture is None and pass_control != HYBRID_PASS_CONTROL:
        return None
    if architecture != HYBRID_ARCHITECTURE or pass_control != HYBRID_PASS_CONTROL:
        raise ValueError(
            "unknown or incomplete hybrid controller contract: "
            f"decision_architecture={architecture!r}, pass_control={pass_control!r}"
        )

    source_matches = training_info.get("controller_sources") == HYBRID_CONTROLLER_SOURCES
    legal_masks = np.asarray([record["legal_mask"] for record in records], dtype=bool)
    scores = (
        np.asarray(features, dtype=np.float64)
        @ np.asarray(weights, dtype=np.float64).T
        + np.asarray(bias, dtype=np.float64)
    )
    if legal_masks.shape != (len(records), ACTION_COUNT) or scores.shape != legal_masks.shape:
        raise ValueError("hybrid audit arrays have incompatible shapes")
    point_masks = legal_masks[:, :PASS_ACTION_INDEX]
    no_legal_point = ~np.any(point_masks, axis=1)
    after_opponent_pass = np.asarray(
        [
            isinstance(record.get("state"), Mapping)
            and record["state"].get("consecutive_passes") == 1
            for record in records
        ],
        dtype=bool,
    )
    rule_route = after_opponent_pass | no_legal_point
    neural_route = ~rule_route
    selected = np.full(len(records), PASS_ACTION_INDEX, dtype=np.int64)
    if np.any(neural_route):
        selected[neural_route] = np.argmax(
            np.where(
                point_masks[neural_route],
                scores[neural_route, :PASS_ACTION_INDEX],
                -np.inf,
            ),
            axis=1,
        )
    route_legal = legal_masks[np.arange(len(records)), selected]

    natural = np.asarray([_is_natural(record) for record in records], dtype=bool)
    constructed = np.asarray([_is_pass_response(record) for record in records], dtype=bool)
    point_target = labels != PASS_ACTION_INDEX
    natural_point = natural & point_target
    splits = np.asarray([record.get("split") for record in records], dtype=object)
    colours = np.asarray([_colour(record) for record in records], dtype=object)
    natural_false_pass = natural_point & rule_route
    train_natural_false_pass = natural_false_pass & (splits == "train")
    validation_natural_false_pass = natural_false_pass & (splits == "validation")
    constructed_gate_miss = constructed & ~rule_route
    constructed_count = int(np.count_nonzero(constructed))
    constructed_hits = int(np.count_nonzero(constructed & rule_route))

    primary: dict[str, Any] = {}
    for split in ("train", "validation"):
        split_natural = (splits == split) & natural_point & neural_route
        primary[split] = {
            "natural_point": _hybrid_group(split_natural, labels, selected),
            "black": _hybrid_group(
                split_natural & (colours == "black"), labels, selected
            ),
            "white": _hybrid_group(
                split_natural & (colours == "white"), labels, selected
            ),
        }
    train_rate = primary["train"]["natural_point"]["agreement_rate"]
    validation_rate = primary["validation"]["natural_point"]["agreement_rate"]
    gap = (
        max(0.0, float(train_rate) - float(validation_rate))
        if train_rate is not None and validation_rate is not None
        else None
    )
    routing = {
        "sample_count": len(records),
        "rule_route_count": int(np.count_nonzero(rule_route)),
        "neural_route_count": int(np.count_nonzero(neural_route)),
        "mutually_exclusive": bool(np.all(~(rule_route & neural_route))),
        "complete": bool(np.all(rule_route | neural_route)),
        "selected_action_legal_count": int(np.count_nonzero(route_legal)),
        "all_selected_actions_legal": bool(np.all(route_legal)),
        "neural_route_pass_count": int(
            np.count_nonzero(neural_route & (selected == PASS_ACTION_INDEX))
        ),
        "rule_route_non_pass_count": int(
            np.count_nonzero(rule_route & (selected != PASS_ACTION_INDEX))
        ),
    }
    validation_black = primary["validation"]["black"]["agreement_rate"]
    validation_white = primary["validation"]["white"]["agreement_rate"]
    failures: list[str] = []
    if not source_matches:
        failures.append("controller_source_declaration")
    if not routing["mutually_exclusive"] or not routing["complete"]:
        failures.append("routing_exclusivity_or_completeness")
    if (
        not routing["all_selected_actions_legal"]
        or routing["neural_route_pass_count"]
        or routing["rule_route_non_pass_count"]
    ):
        failures.append("routing_action_contract")
    # A teacher can choose another legal point after an opponent pass even
    # when passing would end a legal game. Report those training disagreements,
    # but reserve this screening gate for the held-out validation split.
    if np.any(validation_natural_false_pass):
        failures.append("validation_natural_false_pass")
    if constructed_count == 0:
        failures.append("constructed_rule_gate_unverified")
    elif np.any(constructed_gate_miss):
        failures.append("constructed_rule_gate_miss")
    if np.any(constructed & point_target):
        failures.append("constructed_pass_response_has_non_pass_target")
    if validation_rate is None or validation_rate < MINIMUM_AGREEMENT_RATE:
        failures.append("validation_natural_point_agreement_below_25pct")
    if validation_black is None or validation_black < MINIMUM_AGREEMENT_RATE:
        failures.append("validation_black_agreement_below_25pct")
    if validation_white is None or validation_white < MINIMUM_AGREEMENT_RATE:
        failures.append("validation_white_agreement_below_25pct")
    if gap is None or gap > MAXIMUM_TRAIN_VALIDATION_GAP:
        failures.append("train_validation_gap_above_10pp")

    return {
        "used_for_acceptance": True,
        "decision_architecture": architecture,
        "pass_control": pass_control,
        "controller_sources": {
            "declared": training_info.get("controller_sources"),
            "expected": HYBRID_CONTROLLER_SOURCES,
            "matches_contract": source_matches,
        },
        "routing": routing,
        "primary_metrics": primary,
        "train_validation_natural_point_gap": gap,
        "natural_false_pass": {
            "eligible_sample_count": int(np.count_nonzero(natural_point)),
            "count": int(np.count_nonzero(natural_false_pass)),
            "rate": _rate(
                int(np.count_nonzero(natural_false_pass)),
                int(np.count_nonzero(natural_point)),
            ),
            "train_count": int(np.count_nonzero(train_natural_false_pass)),
            "validation_count": int(np.count_nonzero(validation_natural_false_pass)),
        },
        "constructed_rule_gate": {
            "sample_count": constructed_count,
            "pass_target_count": int(np.count_nonzero(constructed & ~point_target)),
            "hit_count": constructed_hits,
            "miss_count": int(np.count_nonzero(constructed_gate_miss)),
            "hit_rate": _rate(constructed_hits, constructed_count),
        },
        "thresholds": {
            "minimum_validation_natural_point_agreement": MINIMUM_AGREEMENT_RATE,
            "minimum_validation_black_agreement": MINIMUM_AGREEMENT_RATE,
            "minimum_validation_white_agreement": MINIMUM_AGREEMENT_RATE,
            "maximum_train_validation_natural_point_gap": MAXIMUM_TRAIN_VALIDATION_GAP,
            "validation_natural_false_pass_count": 0,
            "constructed_rule_gate_miss_count": 0,
        },
        "metric_gate_failures": failures,
        "metric_gate_passed": not failures,
    }


def _neural_decisions(value: object) -> Iterable[Mapping[str, Any]]:
    """Yield only explicitly identified neural-policy decision records."""
    if isinstance(value, Mapping):
        if value.get("actor") == "malecns-neural-policy":
            yield value
        for child in value.values():
            yield from _neural_decisions(child)
    elif isinstance(value, list):
        for child in value:
            yield from _neural_decisions(child)


def audit_report_overlap(
    records: Sequence[Mapping[str, Any]], report_paths: Sequence[Path]
) -> dict[str, Any]:
    training_board_hashes = {
        record["board_hash"]
        for record in records
        if isinstance(record.get("board_hash"), str)
    }
    training_state_hashes = {
        record["hashes"]["state_sha256"]
        for record in records
        if isinstance(record.get("hashes"), Mapping)
        and isinstance(record["hashes"].get("state_sha256"), str)
    }
    decision_board_hashes: set[str] = set()
    decision_state_hashes: set[str] = set()
    decision_count = 0
    missing_state_hash_count = 0
    by_report: list[dict[str, Any]] = []
    for path in report_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        report_board_hashes: set[str] = set()
        report_state_hashes: set[str] = set()
        report_decisions = 0
        report_missing_state_hashes = 0
        for decision in _neural_decisions(payload):
            report_decisions += 1
            board_value = decision.get("board_hash_before")
            if not isinstance(board_value, str) or len(board_value) != 64:
                raise ValueError(
                    f"neural decision lacks a valid board_hash_before in {path}"
                )
            report_board_hashes.add(board_value)
            state_value = decision.get("state_sha256_before")
            if state_value is None:
                report_missing_state_hashes += 1
            elif not isinstance(state_value, str) or len(state_value) != 64:
                raise ValueError(
                    f"neural decision has an invalid state_sha256_before in {path}"
                )
            else:
                report_state_hashes.add(state_value)
        decision_count += report_decisions
        missing_state_hash_count += report_missing_state_hashes
        decision_board_hashes.update(report_board_hashes)
        decision_state_hashes.update(report_state_hashes)
        state_overlap = sorted(training_state_hashes & report_state_hashes)
        board_overlap = sorted(training_board_hashes & report_board_hashes)
        by_report.append(
            {
                "path": str(path.resolve()),
                "neural_decision_count": report_decisions,
                "missing_state_sha256_before_count": report_missing_state_hashes,
                "unique_neural_state_hash_count": len(report_state_hashes),
                "state_overlap_count": len(state_overlap),
                "state_overlap_hashes": state_overlap,
                "unique_neural_board_hash_count": len(report_board_hashes),
                "board_only_conservative_overlap_count": len(board_overlap),
                "board_only_conservative_overlap_hashes": board_overlap,
            }
        )
    state_overlap = sorted(training_state_hashes & decision_state_hashes)
    board_overlap = sorted(training_board_hashes & decision_board_hashes)
    return {
        "checked": bool(report_paths),
        "training_sample_count": len(records),
        "unique_training_state_hash_count": len(training_state_hashes),
        "unique_training_board_hash_count": len(training_board_hashes),
        "neural_decision_count": decision_count,
        "missing_state_sha256_before_count": missing_state_hash_count,
        "state_comparison_complete": bool(report_paths)
        and missing_state_hash_count == 0,
        "unique_neural_state_hash_count": len(decision_state_hashes),
        "state_overlap_count": len(state_overlap),
        "state_overlap_hashes": state_overlap,
        "unique_neural_board_hash_count": len(decision_board_hashes),
        "board_only_conservative_overlap_count": len(board_overlap),
        "board_only_conservative_overlap_hashes": board_overlap,
        "reports": by_report,
    }


def audit_cross_split_hashes(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Detect cached input/feature identity shared by train and validation."""
    result: dict[str, Any] = {}
    for output_name, record_name in (
        ("state", "state_sha256"),
        ("encoding", "encoding_sha256"),
        ("features", "features_sha256"),
    ):
        hashes = {
            split: {
                record["hashes"][record_name]
                for record in records
                if record.get("split") == split
                and isinstance(record.get("hashes"), Mapping)
                and isinstance(record["hashes"].get(record_name), str)
            }
            for split in ("train", "validation")
        }
        overlap = sorted(hashes["train"] & hashes["validation"])
        result[output_name] = {
            "train_unique_count": len(hashes["train"]),
            "validation_unique_count": len(hashes["validation"]),
            "overlap_count": len(overlap),
            "overlap_hashes": overlap,
        }
    result["zero_overlap"] = all(
        result[name]["overlap_count"] == 0
        for name in ("state", "encoding", "features")
    )
    return result


def audit_candidate(
    *,
    cache_directory: Path,
    checkpoint_directory: Path,
    game_reports: Sequence[Path] = (),
) -> dict[str, Any]:
    """Return a deterministic audit report without changing project state."""
    features, labels, records, dataset = load_cache(cache_directory)
    checkpoint = load_policy_checkpoint(checkpoint_directory)
    dataset_hash = dataset.get("dataset_hash")
    checkpoint_dataset_hash = checkpoint.training_info.get(
        "dataset_hash", checkpoint.training_info.get("dataset_id")
    )
    dataset_matches = (
        isinstance(dataset_hash, str)
        and isinstance(checkpoint_dataset_hash, str)
        and dataset_hash == checkpoint_dataset_hash
    )
    compatibility = {
        "training_status_is_trained": checkpoint.training_status == "trained",
        "dataset_hash_matches": dataset_matches,
        "sample_count_matches": checkpoint.training_info.get("sample_count")
        == len(records),
        "model_id_matches": checkpoint.model_id == dataset.get("graph", {}).get("model_id"),
        "graph_manifest_matches": checkpoint.graph_manifest_sha256
        == dataset.get("graph", {}).get("manifest_sha256"),
        "output_pool_matches": checkpoint.output_pool_group_hash
        == dataset.get("output_pool", {}).get("group_hash"),
    }
    failed = [name for name, passed in compatibility.items() if not passed]
    if failed:
        raise ValueError(f"checkpoint/cache compatibility failed: {failed}")
    legacy_metrics = compute_grouped_metrics(
        features,
        labels,
        records,
        checkpoint.weights,
        checkpoint.bias,
    )
    hybrid = audit_hybrid_controller(
        features,
        labels,
        records,
        checkpoint.weights,
        checkpoint.bias,
        checkpoint.training_info,
    )
    result = {
        "schema": AUDIT_SCHEMA,
        "read_only": True,
        "full_graph_simulation_run": False,
        "cache": {
            "path": str(cache_directory.resolve()),
            "dataset_hash": dataset_hash,
            "sample_count": len(records),
            "hash_verified": True,
        },
        "checkpoint": {
            "path": str(checkpoint_directory.resolve()),
            "checkpoint_hash": checkpoint.checkpoint_hash,
            "training_dataset_hash": checkpoint_dataset_hash,
            "hash_verified": True,
        },
        "compatibility": compatibility,
        "cross_split_hash_integrity": audit_cross_split_hashes(records),
        "evaluation_overlap": audit_report_overlap(records, game_reports),
    }
    if hybrid is None:
        # Preserve the original report shape for old joint-neural checkpoints.
        result["metrics"] = legacy_metrics
        result["decision_architecture"] = "legacy-joint-neural-82-class"
    else:
        result["decision_architecture"] = HYBRID_ARCHITECTURE
        result["metrics"] = hybrid["primary_metrics"]
        result["hybrid_acceptance"] = hybrid
        result["diagnostics"] = {
            "legacy_82_class": {
                "used_for_acceptance": False,
                "metrics": legacy_metrics,
            }
        }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--game-report",
        type=Path,
        action="append",
        default=[],
        help="optional full-game JSON report; may be supplied more than once",
    )
    parser.add_argument(
        "--strict-overlap",
        action="store_true",
        help=(
            "exit 1 on full-state evaluation overlap, missing v3 state hashes, "
            "or train/validation encoding-feature overlap"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = audit_candidate(
            cache_directory=args.cache,
            checkpoint_directory=args.checkpoint,
            game_reports=args.game_report,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(json.dumps({"schema": AUDIT_SCHEMA, "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    if args.strict_overlap:
        overlap = report["evaluation_overlap"]
        strict_failure = bool(
            overlap["state_overlap_count"] > 0
            or (overlap["checked"] and not overlap["state_comparison_complete"])
            or not report["cross_split_hash_integrity"]["zero_overlap"]
            or (
                "hybrid_acceptance" in report
                and not report["hybrid_acceptance"]["metric_gate_passed"]
            )
        )
        if strict_failure:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
