"""Deterministic MaleCNS runtime graph assets.

This module builds and validates a compact, memory-mappable CSR graph from the
MaleCNS v1.0 Feather exports.  The graph and neurotransmitter codes are an
engineering representation; they are not a validated biological simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Mapping
import uuid

import numpy as np


BUILDER_ID = "malecns-runtime-graph-v1"
MANIFEST_SCHEMA = "malecns-runtime-manifest-v1"
UNKNOWN_NT = np.int8(-128)
NT_CODEBOOK: dict[str, int] = {
    "unknown": -128,
    "inhibitory": -1,
    "modulatory_or_zero_fast_current": 0,
    "excitatory": 1,
}
NT_NAME_TO_CODE: dict[str, int] = {
    "acetylcholine": 1,
    "gaba": -1,
    "glutamate": -1,
    "histamine": -1,
    "dopamine": 0,
    "octopamine": 0,
    "serotonin": 0,
    "unclear": -128,
}

SOURCE_FILES: dict[str, str] = {
    "annotations": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "neurotransmitters": "body-neurotransmitters-male-cns-v1.0.feather",
    "weights": "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
}

DEFAULT_SOURCE_SHA256: dict[str, str] = {
    "annotations": "2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2",
    "neurotransmitters": "95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621",
    "weights": "e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1",
}

ARTIFACT_SPECS: dict[str, tuple[str, tuple[int | None, ...]]] = {
    "node_ids": ("int64", (None,)),
    "soma_xyz": ("float32", (None, 3)),
    "indptr": ("uint64", (None,)),
    "indices": ("uint32", (None,)),
    "weights": ("float32", (None,)),
    "nt_code": ("int8", (None,)),
}


class AssetValidationError(RuntimeError):
    """Raised when a source or runtime asset fails closed validation."""


@dataclass(frozen=True)
class MaleCNSAssets:
    root: Path
    manifest: Mapping[str, Any]
    manifest_sha256: str
    node_ids: np.memmap
    soma_xyz: np.memmap
    indptr: np.memmap
    indices: np.memmap
    weights: np.memmap
    nt_code: np.memmap

    @property
    def model_id(self) -> str:
        return str(self.manifest["model_id"])

    @property
    def node_count(self) -> int:
        return int(self.node_ids.shape[0])

    @property
    def edge_count(self) -> int:
        return int(self.indices.shape[0])


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def derive_model_id(manifest_without_model_id: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(canonical_json_bytes(manifest_without_model_id)).hexdigest()
    threshold = int(manifest_without_model_id["selection"]["minimum_record_weight"])
    return f"malecns-v1.0-w{threshold}-{digest[:16]}"


def _source_records(
    source_dir: Path, expected_sha256: Mapping[str, str]
) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    paths: dict[str, Path] = {}
    records: dict[str, dict[str, Any]] = {}
    if set(expected_sha256) != set(SOURCE_FILES):
        raise AssetValidationError("expected source hashes must cover exactly three sources")
    for key, filename in SOURCE_FILES.items():
        path = source_dir / filename
        if not path.is_file():
            raise AssetValidationError(f"required source missing: {path}")
        actual = sha256_file(path)
        expected = str(expected_sha256[key]).lower()
        if actual != expected:
            raise AssetValidationError(
                f"source SHA-256 mismatch for {filename}: expected {expected}, got {actual}"
            )
        paths[key] = path
        records[key] = {"file": filename, "bytes": path.stat().st_size, "sha256": actual}
    return paths, records


def _arrow_reader(path: Path):
    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("building MaleCNS assets requires pyarrow") from exc
    source = pa.memory_map(str(path), "r")
    try:
        reader = ipc.open_file(source)
    except Exception:
        source.close()
        raise
    return source, reader


def _read_selected_nodes(path: Path) -> tuple[np.ndarray, np.ndarray, int, int]:
    source, reader = _arrow_reader(path)
    selected: list[tuple[int, tuple[float, float, float]]] = []
    total_rows = 0
    try:
        required = {"bodyId", "superclass", "somaLocation"}
        if not required.issubset(set(reader.schema.names)):
            raise AssetValidationError("annotations schema lacks required columns")
        id_i = reader.schema.get_field_index("bodyId")
        class_i = reader.schema.get_field_index("superclass")
        soma_i = reader.schema.get_field_index("somaLocation")
        for batch_index in range(reader.num_record_batches):
            batch = reader.get_batch(batch_index)
            total_rows += batch.num_rows
            ids = batch.column(id_i).to_pylist()
            classes = batch.column(class_i).to_pylist()
            somas = batch.column(soma_i).to_pylist()
            for body_id, superclass, soma in zip(ids, classes, somas, strict=True):
                if superclass is None or not str(superclass).strip():
                    continue
                if soma is None or len(soma) != 3:
                    continue
                xyz = tuple(float(value) for value in soma)
                if not all(math.isfinite(value) for value in xyz):
                    continue
                selected.append((int(body_id), xyz))
    finally:
        source.close()

    selected.sort(key=lambda item: item[0])
    node_ids = np.asarray([item[0] for item in selected], dtype=np.int64)
    if node_ids.size and np.any(node_ids[1:] <= node_ids[:-1]):
        raise AssetValidationError("selected bodyId values are not unique")
    soma_xyz = np.asarray([item[1] for item in selected], dtype=np.float32).reshape(-1, 3)
    return node_ids, soma_xyz, total_rows, len(selected)


def _membership_positions(sorted_ids: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positions = np.searchsorted(sorted_ids, values)
    valid = positions < sorted_ids.size
    if sorted_ids.size:
        clipped = np.minimum(positions, sorted_ids.size - 1)
        valid &= sorted_ids[clipped] == values
    return positions, valid


def _read_nt_codes(path: Path, node_ids: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    codes = np.full(node_ids.shape, UNKNOWN_NT, dtype=np.int8)
    seen = np.zeros(node_ids.shape, dtype=bool)
    total_rows = 0
    matched_rows = 0
    unrecognized_rows = 0
    source, reader = _arrow_reader(path)
    try:
        required = {"body", "consensus_nt"}
        if not required.issubset(set(reader.schema.names)):
            raise AssetValidationError("neurotransmitter schema lacks required columns")
        body_i = reader.schema.get_field_index("body")
        nt_i = reader.schema.get_field_index("consensus_nt")
        for batch_index in range(reader.num_record_batches):
            batch = reader.get_batch(batch_index)
            total_rows += batch.num_rows
            bodies = np.asarray(batch.column(body_i).to_numpy(zero_copy_only=False), dtype=np.int64)
            names = batch.column(nt_i).to_pylist()
            positions, valid = _membership_positions(node_ids, bodies)
            for row in np.flatnonzero(valid):
                position = int(positions[row])
                if seen[position]:
                    raise AssetValidationError(f"duplicate neurotransmitter body: {int(bodies[row])}")
                name = "" if names[row] is None else str(names[row]).strip().lower()
                code = NT_NAME_TO_CODE.get(name, -128)
                if name not in NT_NAME_TO_CODE:
                    unrecognized_rows += 1
                codes[position] = np.int8(code)
                seen[position] = True
                matched_rows += 1
    finally:
        source.close()

    stats = {
        "source_rows": total_rows,
        "matched_nodes": matched_rows,
        "missing_nodes": int(node_ids.size - matched_rows),
        "unrecognized_rows": unrecognized_rows,
        "unknown": int(np.count_nonzero(codes == -128)),
        "inhibitory": int(np.count_nonzero(codes == -1)),
        "modulatory_or_zero_fast_current": int(np.count_nonzero(codes == 0)),
        "excitatory": int(np.count_nonzero(codes == 1)),
    }
    return codes, stats


def _stream_filtered_edges(
    path: Path,
    node_ids: np.ndarray,
    minimum_weight: int,
    raw_path: Path,
) -> tuple[int, int]:
    record_dtype = np.dtype([("src", "<u4"), ("dst", "<u4"), ("weight", "<u8")])
    total_rows = 0
    filtered_rows = 0
    source, reader = _arrow_reader(path)
    try:
        required = {"body_pre", "body_post", "weight"}
        if not required.issubset(set(reader.schema.names)):
            raise AssetValidationError("weights schema lacks required columns")
        pre_i = reader.schema.get_field_index("body_pre")
        post_i = reader.schema.get_field_index("body_post")
        weight_i = reader.schema.get_field_index("weight")
        with raw_path.open("wb") as output:
            for batch_index in range(reader.num_record_batches):
                batch = reader.get_batch(batch_index)
                total_rows += batch.num_rows
                pre = np.asarray(batch.column(pre_i).to_numpy(zero_copy_only=False), dtype=np.int64)
                post = np.asarray(batch.column(post_i).to_numpy(zero_copy_only=False), dtype=np.int64)
                weight = np.asarray(batch.column(weight_i).to_numpy(zero_copy_only=False), dtype=np.int64)
                pre_pos, pre_ok = _membership_positions(node_ids, pre)
                post_pos, post_ok = _membership_positions(node_ids, post)
                keep = pre_ok & post_ok & (weight >= minimum_weight)
                kept = int(np.count_nonzero(keep))
                if not kept:
                    continue
                if np.any(weight[keep] < 0):
                    raise AssetValidationError("negative retained edge weight")
                records = np.empty(kept, dtype=record_dtype)
                records["src"] = pre_pos[keep].astype(np.uint32, copy=False)
                records["dst"] = post_pos[keep].astype(np.uint32, copy=False)
                records["weight"] = weight[keep].astype(np.uint64, copy=False)
                records.tofile(output)
                filtered_rows += kept
    finally:
        source.close()
    return total_rows, filtered_rows


def _sort_merge_to_csr(
    raw_path: Path, node_count: int, record_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    record_dtype = np.dtype([("src", "<u4"), ("dst", "<u4"), ("weight", "<u8")])
    if raw_path.stat().st_size != record_count * record_dtype.itemsize:
        raise AssetValidationError("temporary edge file size mismatch")
    if record_count == 0:
        return (
            np.zeros(node_count + 1, dtype=np.uint64),
            np.empty(0, dtype=np.uint32),
            np.empty(0, dtype=np.float32),
            0,
        )

    records = np.memmap(raw_path, dtype=record_dtype, mode="r+", shape=(record_count,))
    records.sort(order=("src", "dst"), kind="quicksort")
    records.flush()
    starts = np.empty(record_count, dtype=bool)
    starts[0] = True
    starts[1:] = (records["src"][1:] != records["src"][:-1]) | (
        records["dst"][1:] != records["dst"][:-1]
    )
    unique_positions = np.flatnonzero(starts)
    src = np.asarray(records["src"][unique_positions], dtype=np.uint32)
    indices = np.asarray(records["dst"][unique_positions], dtype=np.uint32)
    merged_uint64 = np.add.reduceat(records["weight"], unique_positions, dtype=np.uint64)
    weights = merged_uint64.astype(np.float32)
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise AssetValidationError("merged weights cannot be represented as positive finite float32")
    row_counts = np.bincount(src.astype(np.int64), minlength=node_count).astype(np.uint64)
    indptr = np.empty(node_count + 1, dtype=np.uint64)
    indptr[0] = 0
    np.cumsum(row_counts, out=indptr[1:])
    distinct_count = int(unique_positions.size)
    del records
    return indptr, indices, weights, distinct_count


def _artifact_record(path: Path, array: np.ndarray) -> dict[str, Any]:
    return {
        "file": path.name,
        "dtype": array.dtype.name,
        "shape": list(array.shape),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def build_runtime_assets(
    source_dir: str | Path,
    output_root: str | Path,
    *,
    expected_sha256: Mapping[str, str] | None = None,
    minimum_weight: int = 5,
) -> Path:
    """Build, validate, and atomically publish deterministic mmap assets."""
    if minimum_weight < 1:
        raise ValueError("minimum_weight must be >= 1")
    source_dir = Path(source_dir).resolve()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    hashes = DEFAULT_SOURCE_SHA256 if expected_sha256 is None else expected_sha256
    source_paths, source_records = _source_records(source_dir, hashes)
    staging = output_root / f".staging-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        node_ids, soma_xyz, annotation_rows, selected_nodes = _read_selected_nodes(
            source_paths["annotations"]
        )
        nt_code, nt_stats = _read_nt_codes(source_paths["neurotransmitters"], node_ids)
        raw_path = staging / "filtered_edges.bin"
        weight_rows, filtered_rows = _stream_filtered_edges(
            source_paths["weights"], node_ids, minimum_weight, raw_path
        )
        indptr, indices, weights, distinct_edges = _sort_merge_to_csr(
            raw_path, int(node_ids.size), filtered_rows
        )
        raw_path.unlink()

        arrays: dict[str, np.ndarray] = {
            "node_ids": node_ids,
            "soma_xyz": soma_xyz,
            "indptr": indptr,
            "indices": indices,
            "weights": weights,
            "nt_code": nt_code,
        }
        artifact_records: dict[str, dict[str, Any]] = {}
        for name, array in arrays.items():
            artifact_path = staging / f"{name}.npy"
            np.save(artifact_path, array, allow_pickle=False)
            artifact_records[name] = _artifact_record(artifact_path, array)

        base_manifest: dict[str, Any] = {
            "schema_version": MANIFEST_SCHEMA,
            "builder": {
                "id": BUILDER_ID,
                "warning": "engineering graph; not a validated biological simulation",
            },
            "source": source_records,
            "selection": {
                "node_rule": "superclass is non-empty AND somaLocation is exactly 3 finite numbers",
                "node_order": "bodyId ascending",
                "edge_rule": "both endpoints selected AND original record weight >= threshold",
                "edge_order": "pre dense index, then post dense index",
                "duplicate_rule": "sum integer weights for identical (pre,post) pairs",
                "minimum_record_weight": minimum_weight,
                "csr_direction": "pre_to_post",
                "nt_source_column": "consensus_nt",
            },
            "counts": {
                "annotation_rows": annotation_rows,
                "selected_nodes": selected_nodes,
                "weight_rows": weight_rows,
                "filtered_edge_records": filtered_rows,
                "distinct_edges": distinct_edges,
                "merged_duplicate_records": filtered_rows - distinct_edges,
                "neurotransmitters": nt_stats,
            },
            "nt_codebook": {
                "-128": "unknown: missing, unclear, or unrecognized",
                "-1": "inhibitory engineering assumption: gaba, glutamate, histamine",
                "0": "zero fast-current engineering assumption: dopamine, octopamine, serotonin",
                "1": "excitatory engineering assumption: acetylcholine",
            },
            "artifacts": artifact_records,
        }
        model_id = derive_model_id(base_manifest)
        manifest = {**base_manifest, "model_id": model_id}
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        # Validate the staged directory before it can become the published model.
        load_runtime_assets(staging, verify_hashes=True)
        target = output_root / model_id
        if target.exists():
            existing = load_runtime_assets(target, verify_hashes=True)
            if canonical_json_bytes(existing.manifest) != canonical_json_bytes(manifest):
                raise AssetValidationError(
                    f"validated output already exists with conflicting manifest: {target}"
                )
            shutil.rmtree(staging)
            return target
        os.replace(staging, target)
        return target
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _validate_manifest_structure(manifest: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "model_id",
        "builder",
        "source",
        "selection",
        "counts",
        "nt_codebook",
        "artifacts",
    }
    if set(manifest) != required:
        raise AssetValidationError("manifest top-level keys mismatch")
    if manifest["schema_version"] != MANIFEST_SCHEMA:
        raise AssetValidationError("unsupported manifest schema")
    base = dict(manifest)
    actual_model_id = str(base.pop("model_id"))
    expected_model_id = derive_model_id(base)
    if actual_model_id != expected_model_id:
        raise AssetValidationError("manifest model_id does not match canonical content")
    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(ARTIFACT_SPECS):
        raise AssetValidationError("manifest artifact set mismatch")


def load_runtime_assets(
    model_dir: str | Path, *, verify_hashes: bool = True
) -> MaleCNSAssets:
    """Load .npy arrays using mmap after strict, fail-closed validation."""
    root = Path(model_dir).resolve()
    manifest_path = root / "manifest.json"
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise AssetValidationError(f"cannot read manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise AssetValidationError("manifest must be a JSON object")
    _validate_manifest_structure(manifest)

    loaded: dict[str, np.memmap] = {}
    for name, (dtype_name, shape_pattern) in ARTIFACT_SPECS.items():
        record = manifest["artifacts"][name]
        if not isinstance(record, dict):
            raise AssetValidationError(f"invalid artifact record: {name}")
        expected_keys = {"file", "dtype", "shape", "bytes", "sha256"}
        if set(record) != expected_keys or record["file"] != f"{name}.npy":
            raise AssetValidationError(f"invalid artifact metadata: {name}")
        path = root / record["file"]
        if not path.is_file() or path.stat().st_size != int(record["bytes"]):
            raise AssetValidationError(f"artifact missing or byte size mismatch: {name}")
        if verify_hashes and sha256_file(path) != record["sha256"]:
            raise AssetValidationError(f"artifact SHA-256 mismatch: {name}")
        try:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
        except Exception as exc:
            raise AssetValidationError(f"cannot mmap artifact {name}: {exc}") from exc
        if not isinstance(array, np.memmap):
            raise AssetValidationError(f"artifact was not memory-mapped: {name}")
        expected_shape = tuple(int(value) for value in record["shape"])
        if array.dtype.name != dtype_name or record["dtype"] != dtype_name:
            raise AssetValidationError(f"dtype mismatch: {name}")
        if array.shape != expected_shape:
            raise AssetValidationError(f"shape mismatch: {name}")
        if len(expected_shape) != len(shape_pattern) or any(
            fixed is not None and actual != fixed
            for actual, fixed in zip(expected_shape, shape_pattern, strict=True)
        ):
            raise AssetValidationError(f"invalid shape contract: {name}")
        loaded[name] = array

    node_ids = loaded["node_ids"]
    soma_xyz = loaded["soma_xyz"]
    indptr = loaded["indptr"]
    indices = loaded["indices"]
    weights = loaded["weights"]
    nt_code = loaded["nt_code"]
    node_count = int(node_ids.size)
    edge_count = int(indices.size)
    if soma_xyz.shape != (node_count, 3) or nt_code.shape != (node_count,):
        raise AssetValidationError("node artifact shapes disagree")
    if indptr.shape != (node_count + 1,) or weights.shape != (edge_count,):
        raise AssetValidationError("CSR artifact shapes disagree")
    if node_count and np.any(node_ids[1:] <= node_ids[:-1]):
        raise AssetValidationError("node_ids must be strictly ascending")
    if not np.all(np.isfinite(soma_xyz)):
        raise AssetValidationError("soma coordinates must be finite")
    if indptr[0] != 0 or int(indptr[-1]) != edge_count or np.any(indptr[1:] < indptr[:-1]):
        raise AssetValidationError("invalid CSR indptr")
    if edge_count and int(indices.max()) >= node_count:
        raise AssetValidationError("CSR destination index out of range")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise AssetValidationError("edge weights must be positive and finite")
    if not np.all(np.isin(nt_code, np.asarray([-128, -1, 0, 1], dtype=np.int8))):
        raise AssetValidationError("invalid neurotransmitter code")
    counts = manifest["counts"]
    if int(counts["selected_nodes"]) != node_count or int(counts["distinct_edges"]) != edge_count:
        raise AssetValidationError("manifest counts disagree with artifacts")
    # Each CSR row must be strictly sorted, proving duplicate pairs were merged.
    for row in range(node_count):
        start, stop = int(indptr[row]), int(indptr[row + 1])
        if stop - start > 1 and np.any(indices[start + 1 : stop] <= indices[start : stop - 1]):
            raise AssetValidationError(f"CSR row {row} is not strictly sorted")

    return MaleCNSAssets(
        root=root,
        manifest=manifest,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        node_ids=node_ids,
        soma_xyz=soma_xyz,
        indptr=indptr,
        indices=indices,
        weights=weights,
        nt_code=nt_code,
    )
