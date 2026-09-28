"""Fail-closed linear Go readout for an exact MaleCNS simulation frame.

This module contains no teacher, search or baseline controller.  It accepts a
hash-verified neural frame, a hash-verified public Go encoding and a strict
checkpoint, then applies an 82-action legal mask and deterministic argmax.
The checkpoint's training status and provenance remain explicit; an untrained
checkpoint is never silently treated as a usable Go policy.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import numpy as np

from .go_neural_encoding import (
    ACTION_COUNT,
    ENCODING_SCHEMA,
    PASS_ACTION_INDEX,
    action_to_move,
)
from .malecns_dynamics import (
    OUTPUT_FEATURE_COUNT,
    OUTPUT_POOL_VERSION,
    RUNTIME_SCHEMA,
)


CHECKPOINT_SCHEMA: Final = "malecns-go-linear-policy-checkpoint-v1"
POLICY_VERSION: Final = "malecns-go-linear-128x82-v1"
ACTION_MAPPING_VERSION: Final = "go-9x9-row-major-plus-pass-v1"
_ARTIFACTS: Final = ("weights", "bias")
PASS_CONTROL_MODES: Final = ("joint-neural", "rule-after-opponent-pass")


class PolicyError(RuntimeError):
    """Base class for a fail-closed policy rejection."""


class CheckpointValidationError(PolicyError):
    """Raised when a policy checkpoint does not match its declared contract."""


class PolicyInputError(PolicyError):
    """Raised when a frame or Go encoding cannot be authenticated."""


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def initialise_linear_parameters(
    *, seed: int = 0, scale: float = 0.01
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic *untrained* parameters for tooling/tests.

    This pure helper does not call a teacher and does not make the result a
    trained policy.  Training code must update the returned arrays and record
    its real provenance when writing a checkpoint.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("scale must be finite and nonnegative")
    rng = np.random.default_rng(seed)
    weights = rng.normal(
        0.0, scale, size=(ACTION_COUNT, OUTPUT_FEATURE_COUNT)
    ).astype(np.float32)
    bias = np.zeros(ACTION_COUNT, dtype=np.float32)
    return weights, bias


def build_checkpoint_manifest(
    *,
    model_id: str,
    graph_manifest_sha256: str,
    output_pool_group_hash: str,
    training_status: str,
    training_info: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the JSON-safe checkpoint contract without touching the filesystem."""
    if not isinstance(model_id, str) or not model_id:
        raise CheckpointValidationError("model_id must be a non-empty string")
    for name, value in (
        ("graph_manifest_sha256", graph_manifest_sha256),
        ("output_pool_group_hash", output_pool_group_hash),
    ):
        if not isinstance(value, str) or len(value) != 64:
            raise CheckpointValidationError(f"{name} must be a 64-character hash")
    if training_status not in ("untrained", "trained"):
        raise CheckpointValidationError("training_status must be untrained or trained")
    if not isinstance(training_info, Mapping):
        raise CheckpointValidationError("training_info must be an object")
    try:
        safe_training_info = json.loads(_canonical_json(dict(training_info)))
    except (TypeError, ValueError) as exc:
        raise CheckpointValidationError("training_info must be JSON-safe") from exc
    if training_status == "trained":
        required = ("method", "dataset_id", "sample_count", "completed_at")
        missing = [key for key in required if key not in safe_training_info]
        if missing:
            raise CheckpointValidationError(
                f"trained checkpoint is missing training_info fields: {missing}"
            )
        if type(safe_training_info["sample_count"]) is not int or safe_training_info["sample_count"] <= 0:
            raise CheckpointValidationError("trained sample_count must be a positive integer")
        for key in ("method", "dataset_id", "completed_at"):
            if not isinstance(safe_training_info[key], str) or not safe_training_info[key]:
                raise CheckpointValidationError(f"trained {key} must be a non-empty string")
        pass_control = safe_training_info.get("pass_control", "joint-neural")
        if pass_control not in PASS_CONTROL_MODES:
            raise CheckpointValidationError(
                f"trained pass_control must be one of {PASS_CONTROL_MODES}"
            )
        if pass_control == "rule-after-opponent-pass":
            expected_sources = {
                "point_actions": "malecns-linear-readout",
                "pass_after_opponent_pass": "go-rule-pass-gate",
                "pass_when_only_legal": "go-rule-pass-gate",
            }
            if safe_training_info.get("controller_sources") != expected_sources:
                raise CheckpointValidationError(
                    "rule pass-control checkpoint has invalid controller_sources"
                )
            fit_sample_count = safe_training_info.get("fit_sample_count")
            if type(fit_sample_count) is not int or fit_sample_count <= 0:
                raise CheckpointValidationError(
                    "rule pass-control checkpoint requires a positive fit_sample_count"
                )
            if (
                safe_training_info.get("decision_architecture")
                != "hybrid-rule-after-opponent-pass-v1"
                or safe_training_info.get("neural_action_space")
                != "point-actions-0..80"
                or safe_training_info.get("pass_selection_source")
                != "deterministic-go-rule"
            ):
                raise CheckpointValidationError(
                    "rule pass-control checkpoint has invalid decision architecture"
                )
    elif not isinstance(safe_training_info.get("reason"), str):
        raise CheckpointValidationError("untrained checkpoint requires training_info.reason")

    artifact_payload = {key: dict(value) for key, value in artifacts.items()}
    if set(artifact_payload) != set(_ARTIFACTS):
        raise CheckpointValidationError("artifacts must contain exactly weights and bias")
    expected_artifacts = {
        "weights": ("weights.npy", "float32", [ACTION_COUNT, OUTPUT_FEATURE_COUNT]),
        "bias": ("bias.npy", "float32", [ACTION_COUNT]),
    }
    for name, (filename, dtype, shape) in expected_artifacts.items():
        record = artifact_payload[name]
        if set(record) != {"file", "sha256", "bytes", "dtype", "shape"}:
            raise CheckpointValidationError(f"artifact {name!r} record fields mismatch")
        if (
            record.get("file") != filename
            or record.get("dtype") != dtype
            or record.get("shape") != shape
            or type(record.get("bytes")) is not int
            or record["bytes"] <= 0
            or not isinstance(record.get("sha256"), str)
            or len(record["sha256"]) != 64
        ):
            raise CheckpointValidationError(f"artifact {name!r} contract mismatch")
    return {
        "schema": CHECKPOINT_SCHEMA,
        "policy_version": POLICY_VERSION,
        "model_id": model_id,
        "graph_manifest_sha256": graph_manifest_sha256,
        "output_pool": {
            "version": OUTPUT_POOL_VERSION,
            "group_hash": output_pool_group_hash,
            "feature_count": OUTPUT_FEATURE_COUNT,
        },
        "action_mapping": {
            "version": ACTION_MAPPING_VERSION,
            "encoding_schema": ENCODING_SCHEMA,
            "action_count": ACTION_COUNT,
            "pass_action_index": PASS_ACTION_INDEX,
            "point_order": "zero-based row-major",
        },
        "training_status": training_status,
        "training_info": safe_training_info,
        "runtime_guarantees": {
            "teacher_access": False,
            "baseline_access": False,
            "fallback_controller": False,
        },
        "artifacts": artifact_payload,
    }


def _array_record(filename: str, array: np.ndarray, path: Path) -> dict[str, Any]:
    return {
        "file": filename,
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
        "dtype": str(array.dtype),
        "shape": list(array.shape),
    }


def write_policy_checkpoint(
    directory: str | Path,
    *,
    weights: np.ndarray,
    bias: np.ndarray,
    model_id: str,
    graph_manifest_sha256: str,
    output_pool_group_hash: str,
    training_status: str,
    training_info: Mapping[str, Any],
) -> str:
    """Write a strict checkpoint and return its manifest SHA-256.

    Passing ``training_status='trained'`` is an explicit provenance claim by
    the caller. This writer validates required metadata but performs no
    training and invents no training history.
    """
    weights_array = np.asarray(weights)
    bias_array = np.asarray(bias)
    if weights_array.dtype != np.dtype("float32") or weights_array.shape != (
        ACTION_COUNT,
        OUTPUT_FEATURE_COUNT,
    ):
        raise CheckpointValidationError(
            f"weights must be float32 with shape {(ACTION_COUNT, OUTPUT_FEATURE_COUNT)}"
        )
    if bias_array.dtype != np.dtype("float32") or bias_array.shape != (ACTION_COUNT,):
        raise CheckpointValidationError(
            f"bias must be float32 with shape {(ACTION_COUNT,)}"
        )
    if not np.all(np.isfinite(weights_array)) or not np.all(np.isfinite(bias_array)):
        raise CheckpointValidationError("weights and bias must be finite")

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    weights_path = root / "weights.npy"
    bias_path = root / "bias.npy"
    np.save(weights_path, weights_array, allow_pickle=False)
    np.save(bias_path, bias_array, allow_pickle=False)
    artifacts = {
        "weights": _array_record("weights.npy", weights_array, weights_path),
        "bias": _array_record("bias.npy", bias_array, bias_path),
    }
    manifest = build_checkpoint_manifest(
        model_id=model_id,
        graph_manifest_sha256=graph_manifest_sha256,
        output_pool_group_hash=output_pool_group_hash,
        training_status=training_status,
        training_info=training_info,
        artifacts=artifacts,
    )
    manifest_bytes = _canonical_json(manifest).encode("utf-8")
    (root / "manifest.json").write_bytes(manifest_bytes)
    return hashlib.sha256(manifest_bytes).hexdigest()


@dataclass(frozen=True)
class PolicyCheckpoint:
    directory: Path
    checkpoint_hash: str
    model_id: str
    graph_manifest_sha256: str
    output_pool_group_hash: str
    training_status: str
    training_info: Mapping[str, Any]
    weights: np.ndarray
    bias: np.ndarray


def load_policy_checkpoint(directory: str | Path) -> PolicyCheckpoint:
    """Hash-check and load a checkpoint; any mismatch fails closed."""
    root = Path(directory)
    manifest_path = root / "manifest.json"
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointValidationError(f"cannot read valid manifest.json: {exc}") from exc
    if not isinstance(manifest, Mapping) or manifest.get("schema") != CHECKPOINT_SCHEMA:
        raise CheckpointValidationError("unsupported checkpoint schema")
    expected_top_level = {
        "schema",
        "policy_version",
        "model_id",
        "graph_manifest_sha256",
        "output_pool",
        "action_mapping",
        "training_status",
        "training_info",
        "runtime_guarantees",
        "artifacts",
    }
    if set(manifest) != expected_top_level:
        raise CheckpointValidationError("checkpoint manifest fields mismatch")
    if manifest.get("policy_version") != POLICY_VERSION:
        raise CheckpointValidationError("unsupported policy version")
    action = manifest.get("action_mapping")
    if not isinstance(action, Mapping) or dict(action) != {
        "version": ACTION_MAPPING_VERSION,
        "encoding_schema": ENCODING_SCHEMA,
        "action_count": ACTION_COUNT,
        "pass_action_index": PASS_ACTION_INDEX,
        "point_order": "zero-based row-major",
    }:
        raise CheckpointValidationError("action mapping contract mismatch")
    output = manifest.get("output_pool")
    if not isinstance(output, Mapping):
        raise CheckpointValidationError("output_pool contract is missing")
    group_hash = output.get("group_hash")
    if (
        output.get("version") != OUTPUT_POOL_VERSION
        or output.get("feature_count") != OUTPUT_FEATURE_COUNT
        or not isinstance(group_hash, str)
        or len(group_hash) != 64
    ):
        raise CheckpointValidationError("output_pool contract mismatch")
    model_id = manifest.get("model_id")
    graph_hash = manifest.get("graph_manifest_sha256")
    if not isinstance(model_id, str) or not model_id:
        raise CheckpointValidationError("checkpoint model_id is missing")
    if not isinstance(graph_hash, str) or len(graph_hash) != 64:
        raise CheckpointValidationError("checkpoint graph manifest hash is invalid")
    training_status = manifest.get("training_status")
    training_info = manifest.get("training_info")
    # Reuse the pure validator for provenance and top-level contract fields.
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise CheckpointValidationError("checkpoint artifacts are missing")
    build_checkpoint_manifest(
        model_id=model_id,
        graph_manifest_sha256=graph_hash,
        output_pool_group_hash=group_hash,
        training_status=training_status,
        training_info=training_info,
        artifacts=artifacts,
    )
    if manifest.get("runtime_guarantees") != {
        "teacher_access": False,
        "baseline_access": False,
        "fallback_controller": False,
    }:
        raise CheckpointValidationError("runtime guarantee contract mismatch")

    arrays: dict[str, np.ndarray] = {}
    expected = {
        "weights": (np.dtype("float32"), (ACTION_COUNT, OUTPUT_FEATURE_COUNT)),
        "bias": (np.dtype("float32"), (ACTION_COUNT,)),
    }
    for name in _ARTIFACTS:
        record = artifacts.get(name)
        if not isinstance(record, Mapping):
            raise CheckpointValidationError(f"artifact {name!r} is missing")
        filename = record.get("file")
        digest = record.get("sha256")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise CheckpointValidationError(f"artifact {name!r} filename is unsafe")
        if not isinstance(digest, str) or len(digest) != 64:
            raise CheckpointValidationError(f"artifact {name!r} hash is invalid")
        path = root / filename
        try:
            if path.stat().st_size != record.get("bytes"):
                raise CheckpointValidationError(f"artifact {name!r} byte size mismatch")
            if _sha256_file(path) != digest.lower():
                raise CheckpointValidationError(f"artifact {name!r} SHA-256 mismatch")
            # Policy arrays are only ~42 KiB. Loading them eagerly avoids a
            # long-lived Windows mmap handle locking checkpoint files while
            # still retaining strict pre-load byte/hash verification.
            array = np.load(path, allow_pickle=False)
        except CheckpointValidationError:
            raise
        except (OSError, ValueError) as exc:
            raise CheckpointValidationError(f"cannot load artifact {name!r}: {exc}") from exc
        expected_dtype, expected_shape = expected[name]
        if (
            array.dtype != expected_dtype
            or array.shape != expected_shape
            or record.get("dtype") != str(expected_dtype)
            or record.get("shape") != list(expected_shape)
        ):
            raise CheckpointValidationError(f"artifact {name!r} dtype/shape mismatch")
        if not np.all(np.isfinite(array)):
            raise CheckpointValidationError(f"artifact {name!r} contains non-finite values")
        arrays[name] = array

    return PolicyCheckpoint(
        directory=root,
        checkpoint_hash=hashlib.sha256(manifest_bytes).hexdigest(),
        model_id=model_id,
        graph_manifest_sha256=graph_hash,
        output_pool_group_hash=group_hash,
        training_status=training_status,
        training_info=dict(training_info),
        weights=arrays["weights"],
        bias=arrays["bias"],
    )


def _verify_encoding(encoding: Mapping[str, Any]) -> tuple[str, str, list[int]]:
    if encoding.get("schema") != ENCODING_SCHEMA:
        raise PolicyInputError("unsupported encoding schema")
    supplied_hash = encoding.get("encoding_hash")
    unsigned = dict(encoding)
    unsigned.pop("encoding_hash", None)
    if not isinstance(supplied_hash, str) or _sha256_json(unsigned) != supplied_hash:
        raise PolicyInputError("encoding_hash does not match encoding payload")
    board_hash = encoding.get("board_hash")
    if not isinstance(board_hash, str) or len(board_hash) != 64:
        raise PolicyInputError("encoding board_hash is invalid")
    mask = encoding.get("legal_mask")
    if (
        not isinstance(mask, list)
        or len(mask) != ACTION_COUNT
        or any(type(value) is not int or value not in (0, 1) for value in mask)
    ):
        raise PolicyInputError("encoding legal_mask must contain 82 binary values")
    if not any(mask):
        raise PolicyInputError("cannot select an action from a finished game")
    return supplied_hash, board_hash, mask


def _verify_frame(frame: Mapping[str, Any]) -> tuple[list[float], Mapping[str, Any]]:
    if frame.get("schema") != RUNTIME_SCHEMA:
        raise PolicyInputError("unsupported neural frame schema")
    supplied_id = frame.get("frame_id")
    core = dict(frame)
    core.pop("frame_id", None)
    core.pop("wall_time_ms", None)
    # simulate_go_encoding attaches the already-hashed source stimulus after
    # simulate_frame has finalised its deterministic core.
    core.pop("stimulus", None)
    if not isinstance(supplied_id, str) or _sha256_json(core) != supplied_id:
        raise PolicyInputError("frame_id does not match deterministic frame payload")
    output = frame.get("output_pool")
    if not isinstance(output, Mapping):
        raise PolicyInputError("frame output_pool is missing")
    expected_output_fields = {
        "version",
        "model_id",
        "feature_count",
        "group_size",
        "unique_neuron_count",
        "reused_for_small_graph",
        "selection_rule",
        "feature_formula",
        "activity_source",
        "body_ids_by_pool",
        "group_hash",
        "features",
        "features_hash",
    }
    if set(output) != expected_output_fields:
        raise PolicyInputError("frame output_pool fields mismatch")
    features = output.get("features")
    if (
        output.get("version") != OUTPUT_POOL_VERSION
        or output.get("feature_count") != OUTPUT_FEATURE_COUNT
        or not isinstance(features, list)
        or len(features) != OUTPUT_FEATURE_COUNT
    ):
        raise PolicyInputError("frame output_pool contract mismatch")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
        for value in features
    ):
        raise PolicyInputError("frame output features must be finite values in [0,1]")
    group_hash = output.get("group_hash")
    feature_hash = output.get("features_hash")
    expected_feature_hash = _sha256_json(
        {
            "version": OUTPUT_POOL_VERSION,
            "group_hash": group_hash,
            "features": features,
        }
    )
    if not isinstance(group_hash, str) or len(group_hash) != 64:
        raise PolicyInputError("frame output_pool group_hash is invalid")
    definition = dict(output)
    definition.pop("group_hash")
    definition.pop("features")
    definition.pop("features_hash")
    if _sha256_json(definition) != group_hash:
        raise PolicyInputError("frame output_pool group_hash mismatch")
    body_ids = output.get("body_ids_by_pool")
    group_size = output.get("group_size")
    if (
        output.get("model_id") != frame.get("model_id")
        or type(group_size) is not int
        or group_size <= 0
        or not isinstance(body_ids, list)
        or len(body_ids) != OUTPUT_FEATURE_COUNT
        or any(
            not isinstance(pool, list)
            or len(pool) != group_size
            or any(type(body_id) is not int or body_id < 0 for body_id in pool)
            for pool in body_ids
        )
    ):
        raise PolicyInputError("frame output_pool model_id/body-ID contract mismatch")
    if feature_hash != expected_feature_hash:
        raise PolicyInputError("frame output_pool features_hash mismatch")
    return [float(value) for value in features], output


def select_action(
    frame: Mapping[str, Any],
    encoding: Mapping[str, Any],
    checkpoint: PolicyCheckpoint,
) -> dict[str, Any]:
    """Select one legal action from one exact frame, with no fallback path."""
    if not isinstance(checkpoint, PolicyCheckpoint):
        raise TypeError("checkpoint must be a loaded PolicyCheckpoint")
    if checkpoint.training_status != "trained":
        raise CheckpointValidationError("untrained checkpoint cannot select an action")
    encoding_hash, board_hash, legal_mask = _verify_encoding(encoding)
    features, output = _verify_frame(frame)
    if frame.get("encoding_hash") != encoding_hash or frame.get("board_hash") != board_hash:
        raise PolicyInputError("frame and encoding hashes do not refer to the same position")
    if frame.get("model_id") != checkpoint.model_id:
        raise PolicyInputError("frame model_id does not match checkpoint")
    if frame.get("manifest_sha256") != checkpoint.graph_manifest_sha256:
        raise PolicyInputError("frame graph manifest hash does not match checkpoint")
    if output.get("group_hash") != checkpoint.output_pool_group_hash:
        raise PolicyInputError("frame output pool does not match checkpoint")

    legal_indices = np.flatnonzero(np.asarray(legal_mask, dtype=np.uint8))
    if legal_indices.size == 0:
        raise PolicyInputError("cannot select an action from a finished game")
    pass_control = checkpoint.training_info.get("pass_control", "joint-neural")
    if pass_control not in PASS_CONTROL_MODES:
        raise CheckpointValidationError(
            f"checkpoint pass_control must be one of {PASS_CONTROL_MODES}"
        )
    only_pass_legal = (
        legal_indices.size == 1 and int(legal_indices[0]) == PASS_ACTION_INDEX
    )
    rule_pass_requested = pass_control == "rule-after-opponent-pass" and (
        encoding.get("last_move_action") == PASS_ACTION_INDEX or only_pass_legal
    )
    if rule_pass_requested and not legal_mask[PASS_ACTION_INDEX]:
        raise PolicyInputError("rule pass gate requested an illegal pass")
    rule_pass = rule_pass_requested
    if rule_pass:
        selected = PASS_ACTION_INDEX
        logits_payload: list[float] | None = None
        selected_logit: float | None = None
        controller_source = "go-rule-pass-gate"
        controller_reason = (
            "opponent-last-move-was-pass"
            if encoding.get("last_move_action") == PASS_ACTION_INDEX
            else "pass-is-only-legal-action"
        )
        neural_readout_used = False
    else:
        vector = np.asarray(features, dtype=np.float64)
        logits = (
            checkpoint.weights.astype(np.float64) @ vector
            + checkpoint.bias.astype(np.float64)
        )
        if not np.all(np.isfinite(logits)):
            raise PolicyInputError("policy produced non-finite logits")
        candidate_indices = legal_indices
        if pass_control == "rule-after-opponent-pass":
            candidate_indices = candidate_indices[
                candidate_indices != PASS_ACTION_INDEX
            ]
            if candidate_indices.size == 0:
                raise PolicyInputError(
                    "rule pass gate left no legal point action"
                )
        # np.argmax is deterministic and selects the first (lowest action
        # index) when two legal logits are exactly equal.
        selected = int(
            candidate_indices[int(np.argmax(logits[candidate_indices]))]
        )
        logits_payload = [float(value) for value in logits]
        selected_logit = float(logits[selected])
        controller_source = "malecns-linear-readout"
        controller_reason = (
            "highest-legal-point-logit"
            if pass_control == "rule-after-opponent-pass"
            else "highest-legal-action-logit"
        )
        neural_readout_used = True
    move = action_to_move(selected)
    payload: dict[str, Any] = {
        "policy_version": POLICY_VERSION,
        "checkpoint_hash": checkpoint.checkpoint_hash,
        "training_status": checkpoint.training_status,
        "model_id": checkpoint.model_id,
        "frame_id": frame["frame_id"],
        "board_hash": board_hash,
        "encoding_hash": encoding_hash,
        "output_pool_group_hash": checkpoint.output_pool_group_hash,
        "output_features_hash": output["features_hash"],
        "action_index": selected,
        "move": "pass" if move is None else [int(move[0]), int(move[1])],
        "pass_selected": selected == PASS_ACTION_INDEX,
        "selected_logit": selected_logit,
        "logits": logits_payload,
        "legal_mask": legal_mask,
        "pass_control": pass_control,
        "controller_source": controller_source,
        "controller_reason": controller_reason,
        "neural_readout_used": neural_readout_used,
        "teacher_accessed_at_runtime": False,
        "baseline_controller_called": False,
        "fallback_used": False,
    }
    payload["decision_hash"] = _sha256_json(payload)
    return payload


__all__ = [
    "ACTION_MAPPING_VERSION",
    "CHECKPOINT_SCHEMA",
    "CheckpointValidationError",
    "POLICY_VERSION",
    "PASS_CONTROL_MODES",
    "PolicyCheckpoint",
    "PolicyError",
    "PolicyInputError",
    "build_checkpoint_manifest",
    "initialise_linear_parameters",
    "load_policy_checkpoint",
    "select_action",
    "write_policy_checkpoint",
]
