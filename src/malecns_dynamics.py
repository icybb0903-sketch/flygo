"""Deterministic engineering LIF reference runtime for MaleCNS graph assets.

This module combines measured *structural* connectivity with an engineered,
unvalidated dynamics model.  Its frames are simulated values, not recorded
fly neural activity, and they are not evidence of fly cognition.  The module
does not choose Go moves and has no dependency on the project's baseline Go
controller.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Final, Mapping, Sequence

import numpy as np

from .go_neural_encoding import ENCODING_SCHEMA, VECTOR_LENGTH


RUNTIME_SCHEMA: Final = "malecns-engineered-lif-frame-v3"
STIMULUS_ADAPTER_VERSION: Final = "go415-to-32-hash-bucket-v1"
SPATIAL_STIMULUS_ADAPTER_VERSION: Final = "go415-to-32-spatial-tiles-v1"
INPUT_GROUP_VERSION: Final = "malecns-input-groups-sha256-v1"
OUTPUT_POOL_VERSION: Final = "malecns-output-pools-sha256-v1"
DOWNSTREAM_OUTPUT_POOL_VERSION: Final = "malecns-output-pools-input-downstream-v1"
DYNAMICS_VERSION: Final = "malecns-engineered-sparse-lif-v2"
DISCLAIMER: Final = (
    "Engineered LIF simulation over MaleCNS structural connectivity; not measured "
    "neural activity, not biologically calibrated, and not evidence of cognition."
)
CHANNEL_COUNT: Final = 32
OUTPUT_FEATURE_COUNT: Final = 128
OUTPUT_POOL_GROUP_SIZE: Final = 32
MAX_RATE_HZ: Final = 150.0
_ALLOWED_NT_CODES: Final = frozenset((-128, -1, 0, 1))
_ASSET_NAMES: Final = (
    "node_ids",
    "soma_xyz",
    "indptr",
    "indices",
    "weights",
    "nt_code",
)


class DynamicsError(RuntimeError):
    """Base class for a fail-closed runtime rejection."""


class AssetValidationError(DynamicsError):
    """Raised when graph assets do not match their declared contract."""


class EventLimitExceeded(DynamicsError):
    """Raised instead of returning a partial frame after a burst guard trips."""


class DynamicsCancelled(DynamicsError):
    """Raised when the caller cancels an in-flight simulation."""


class DynamicsTimeout(DynamicsError):
    """Raised when the configured wall-clock deadline is reached."""


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


@dataclass(frozen=True)
class GraphArrays:
    """Validated outgoing CSR arrays produced by the Phase 3 graph builder."""

    model_id: str
    manifest_sha256: str
    node_ids: np.ndarray
    soma_xyz: np.ndarray
    indptr: np.ndarray
    indices: np.ndarray
    weights: np.ndarray
    nt_code: np.ndarray

    @classmethod
    def from_arrays(
        cls,
        *,
        model_id: str,
        node_ids: np.ndarray,
        soma_xyz: np.ndarray,
        indptr: np.ndarray,
        indices: np.ndarray,
        weights: np.ndarray,
        nt_code: np.ndarray,
        manifest_sha256: str = "synthetic-fixture",
    ) -> "GraphArrays":
        graph = cls(
            model_id=model_id,
            manifest_sha256=manifest_sha256,
            node_ids=np.asarray(node_ids),
            soma_xyz=np.asarray(soma_xyz),
            indptr=np.asarray(indptr),
            indices=np.asarray(indices),
            weights=np.asarray(weights),
            nt_code=np.asarray(nt_code),
        )
        graph.validate()
        return graph

    @property
    def node_count(self) -> int:
        return int(self.node_ids.shape[0])

    @property
    def edge_count(self) -> int:
        return int(self.indices.shape[0])

    def validate(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id:
            raise AssetValidationError("model_id must be a non-empty string")
        expected_dtypes = {
            "node_ids": np.dtype("int64"),
            "soma_xyz": np.dtype("float32"),
            "indptr": np.dtype("uint64"),
            "indices": np.dtype("uint32"),
            "weights": np.dtype("float32"),
            "nt_code": np.dtype("int8"),
        }
        for name, dtype in expected_dtypes.items():
            value = getattr(self, name)
            if value.dtype != dtype:
                raise AssetValidationError(f"{name} dtype must be {dtype}, got {value.dtype}")
        n = self.node_count
        e = self.edge_count
        expected_shapes = {
            "node_ids": (n,),
            "soma_xyz": (n, 3),
            "indptr": (n + 1,),
            "indices": (e,),
            "weights": (e,),
            "nt_code": (n,),
        }
        for name, shape in expected_shapes.items():
            if getattr(self, name).shape != shape:
                raise AssetValidationError(
                    f"{name} shape must be {shape}, got {getattr(self, name).shape}"
                )
        if n == 0:
            raise AssetValidationError("graph must contain at least one node")
        if np.any(self.node_ids[1:] <= self.node_ids[:-1]):
            raise AssetValidationError("node_ids must be strictly increasing")
        if not np.all(np.isfinite(self.soma_xyz)):
            raise AssetValidationError("soma_xyz contains a non-finite value")
        if int(self.indptr[0]) != 0 or int(self.indptr[-1]) != e:
            raise AssetValidationError("indptr endpoints do not match edge count")
        if np.any(self.indptr[1:] < self.indptr[:-1]):
            raise AssetValidationError("indptr must be nondecreasing")
        if e and int(np.max(self.indices)) >= n:
            raise AssetValidationError("indices contains an out-of-range destination")
        if not np.all(np.isfinite(self.weights)) or np.any(self.weights < 0):
            raise AssetValidationError("weights must be finite and nonnegative")
        codes = {int(value) for value in np.unique(self.nt_code)}
        if not codes.issubset(_ALLOWED_NT_CODES):
            raise AssetValidationError(f"nt_code contains unsupported values: {sorted(codes)}")


def _artifact_mapping(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("artifacts", "assets", "files"):
        candidate = manifest.get(key)
        if isinstance(candidate, Mapping):
            return candidate
    raise AssetValidationError("manifest has no artifacts/assets/files mapping")


def load_graph_assets(directory: str | Path) -> GraphArrays:
    """Load and hash-verify the graph builder's mmap-friendly assets."""
    root = Path(directory)
    manifest_path = root / "manifest.json"
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise AssetValidationError(f"cannot read valid manifest.json: {exc}") from exc
    if not isinstance(manifest, Mapping):
        raise AssetValidationError("manifest root must be an object")
    model_id = manifest.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        raise AssetValidationError("manifest model_id is missing")
    artifacts = _artifact_mapping(manifest)
    arrays: dict[str, np.ndarray] = {}
    for name in _ASSET_NAMES:
        record = artifacts.get(name)
        if not isinstance(record, Mapping):
            raise AssetValidationError(f"manifest artifact {name!r} is missing")
        filename = record.get("file")
        expected_hash = record.get("sha256")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise AssetValidationError(f"artifact {name!r} has an unsafe filename")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise AssetValidationError(f"artifact {name!r} has no valid SHA-256")
        path = root / filename
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise AssetValidationError(f"artifact {name!r} is unavailable: {exc}") from exc
        if "bytes" in record and int(record["bytes"]) != size:
            raise AssetValidationError(f"artifact {name!r} byte size mismatch")
        if _sha256_file(path) != expected_hash.lower():
            raise AssetValidationError(f"artifact {name!r} SHA-256 mismatch")
        try:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise AssetValidationError(f"artifact {name!r} is not a valid .npy: {exc}") from exc
        if "dtype" in record and np.dtype(record["dtype"]) != array.dtype:
            raise AssetValidationError(f"artifact {name!r} dtype differs from manifest")
        if "shape" in record and tuple(int(v) for v in record["shape"]) != array.shape:
            raise AssetValidationError(f"artifact {name!r} shape differs from manifest")
        arrays[name] = array
    graph = GraphArrays(
        model_id=model_id,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        **arrays,
    )
    graph.validate()
    counts = manifest.get("counts")
    if isinstance(counts, Mapping):
        for keys, actual in (("nodes", graph.node_count), ("edges", graph.edge_count)):
            declared = counts.get(keys, counts.get(f"{keys[:-1]}_count"))
            if declared is not None and int(declared) != actual:
                raise AssetValidationError(f"manifest {keys} count mismatch")
    return graph


def _normalise_feature(index: int, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DynamicsError(f"encoding vector[{index}] is not numeric")
    number = float(value)
    if not math.isfinite(number):
        raise DynamicsError(f"encoding vector[{index}] is not finite")
    if index < 405:
        upper = 1.0
    else:
        upper = (1.0, 1.0, 81.0, 81.0, 2.0, 1.0, 1.0, 1.0, 162.0, 20.0)[
            index - 405
        ]
    return min(1.0, max(0.0, number / upper))


def go_encoding_to_stimulus(encoding: Mapping[str, Any]) -> dict[str, Any]:
    """Compress the public 415-value Go encoding into 32 fixed rate channels.

    Every source feature contributes to two deterministic buckets.  No teacher
    action, search score, baseline action or future state is accepted/read.
    """
    if encoding.get("schema") != ENCODING_SCHEMA:
        raise DynamicsError("unsupported Go encoding schema")
    vector = encoding.get("vector")
    if not isinstance(vector, Sequence) or isinstance(vector, (str, bytes)):
        raise DynamicsError("encoding vector is missing")
    if len(vector) != VECTOR_LENGTH or encoding.get("vector_length") != VECTOR_LENGTH:
        raise DynamicsError(f"encoding vector must have {VECTOR_LENGTH} values")
    supplied_hash = encoding.get("encoding_hash")
    unsigned = dict(encoding)
    unsigned.pop("encoding_hash", None)
    if not isinstance(supplied_hash, str) or _sha256_json(unsigned) != supplied_hash:
        raise DynamicsError("encoding_hash does not match encoding payload")
    board_hash = encoding.get("board_hash")
    if not isinstance(board_hash, str) or len(board_hash) != 64:
        raise DynamicsError("encoding has no valid board_hash")

    sums = np.zeros(CHANNEL_COUNT, dtype=np.float64)
    denominators = np.zeros(CHANNEL_COUNT, dtype=np.float64)
    for index, raw in enumerate(vector):
        value = _normalise_feature(index, raw)
        for bucket, weight in (((17 * index + 3) % 32, 1.0), ((29 * index + 11) % 32, 0.5)):
            sums[bucket] += value * weight
            denominators[bucket] += weight
    rates = np.divide(sums, denominators, out=np.zeros_like(sums), where=denominators > 0)
    rates = np.round(np.clip(rates * MAX_RATE_HZ, 0.0, MAX_RATE_HZ), 6)
    payload: dict[str, Any] = {
        "version": STIMULUS_ADAPTER_VERSION,
        "source_schema": ENCODING_SCHEMA,
        "source_vector_length": VECTOR_LENGTH,
        "channel_count": CHANNEL_COUNT,
        "max_rate_hz": MAX_RATE_HZ,
        "projection": {
            "primary_bucket": "(17 * feature_index + 3) mod 32; weight 1.0",
            "secondary_bucket": "(29 * feature_index + 11) mod 32; weight 0.5",
            "aggregation": "weighted mean after fixed per-feature clipping/normalisation",
        },
        "causal_inputs_only": True,
        "teacher_or_baseline_action_used": False,
        "board_hash": board_hash,
        "encoding_hash": supplied_hash,
        "rates_hz": rates.tolist(),
    }
    payload["stimulus_hash"] = _sha256_json(payload)
    return payload


def go_encoding_to_spatial_stimulus(encoding: Mapping[str, Any]) -> dict[str, Any]:
    """Experimental spatial adapter: 16 Go board tiles × own/opponent stones.

    This is an engineered input interface, not a biological visual system.
    The legacy hash adapter remains the default for all existing checkpoints.
    """
    validated = go_encoding_to_stimulus(encoding)
    planes = np.asarray(encoding["vector"][:162], dtype=np.float64).reshape(2, 9, 9)
    rates: list[float] = []
    bounds = (0, 2, 4, 6, 9)
    for colour_plane in planes:
        for tile_row in range(4):
            for tile_column in range(4):
                tile = colour_plane[
                    bounds[tile_row]:bounds[tile_row + 1],
                    bounds[tile_column]:bounds[tile_column + 1],
                ]
                rates.append(round(float(30.0 + 120.0 * tile.mean()), 6))
    payload = {
        **validated,
        "version": SPATIAL_STIMULUS_ADAPTER_VERSION,
        "projection": {
            "tile_boundaries": list(bounds),
            "channel_order": "own_stone_16_tiles_then_opponent_stone_16_tiles",
            "aggregation": "tile mean; rate = 30 + 120 * mean, Hz",
            "unused_encoding_features": "empty/legal/last-move planes and 10 scalars",
        },
        "rates_hz": rates,
    }
    payload.pop("stimulus_hash")
    payload["stimulus_hash"] = _sha256_json(payload)
    return payload


@dataclass(frozen=True)
class LIFParameters:
    """Versioned engineering constants; these are not biological calibration."""

    dt_ms: float = 0.2
    duration_ms: float = 40.0
    rest_mv: float = -52.0
    reset_mv: float = -52.0
    threshold_mv: float = -45.0
    floor_mv: float = -80.0
    membrane_tau_ms: float = 20.0
    synaptic_delay_ms: float = 1.8
    refractory_ms: float = 2.2
    # Engineering stability value for the weighted MaleCNS graph. This is not
    # a measured synaptic amplitude or biological calibration.
    mv_per_contact: float = 0.05
    stimulus_group_size: int = 8
    display_node_limit: int = 512
    display_edge_limit: int = 2_048
    max_synaptic_events: int = 5_000_000
    max_total_spikes: int = 500_000
    max_spikes_per_tick: int = 50_000
    max_recorded_spikes: int = 20_000

    def validate(self) -> None:
        finite_positive = (
            "dt_ms",
            "duration_ms",
            "membrane_tau_ms",
            "synaptic_delay_ms",
            "refractory_ms",
            "mv_per_contact",
        )
        for name in finite_positive:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise DynamicsError(f"{name} must be finite and positive")
        if not self.floor_mv <= self.rest_mv <= self.reset_mv < self.threshold_mv:
            raise DynamicsError("potential parameters have an invalid ordering")
        if self.duration_ms / self.dt_ms > 100_000:
            raise DynamicsError("simulation requests more than 100,000 ticks")
        for name in (
            "stimulus_group_size",
            "display_node_limit",
            "display_edge_limit",
            "max_synaptic_events",
            "max_total_spikes",
            "max_spikes_per_tick",
            "max_recorded_spikes",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise DynamicsError(f"{name} must be a positive integer")


def _display_subset(
    graph: GraphArrays,
    groups: tuple[tuple[int, ...], ...],
    spike_counts: np.ndarray,
    max_activity: np.ndarray,
    *,
    limit: int,
) -> tuple[np.ndarray, int, int]:
    """Choose activity plus real one-hop structural context deterministically.

    Up to half of the budget is reserved for actual stimulus sources. Activity
    candidates may then fill the subset to three quarters, leaving room for
    real one-hop destinations. This keeps silent or inhibitory paths visible
    without assigning synthetic activity to either endpoint.
    """
    active_indices = np.flatnonzero((spike_counts > 0) | (max_activity > 0))
    if active_indices.size:
        order = np.lexsort(
            (
                graph.node_ids[active_indices],
                -max_activity[active_indices],
                -spike_counts[active_indices].astype(np.int64),
            )
        )
        ranked_active = [int(value) for value in active_indices[order]]
    else:
        ranked_active = []

    selected: list[int] = []
    selected_set: set[int] = set()

    def add(index: int) -> bool:
        if len(selected) >= limit:
            return False
        if index not in selected_set:
            selected.append(index)
            selected_set.add(index)
        return len(selected) < limit

    group_sources: list[int] = []
    seen_sources: set[int] = set()
    for group in groups:
        for raw_index in group:
            index = int(raw_index)
            if index not in seen_sources:
                group_sources.append(index)
                seen_sources.add(index)

    source_budget = max(1, limit // 2)
    for index in group_sources[:source_budget]:
        add(index)
    activity_ceiling = max(len(selected), max(1, (3 * limit) // 4))
    for index in ranked_active:
        if len(selected) >= activity_ceiling:
            break
        add(index)

    context_sources: list[int] = []
    seen_context: set[int] = set()
    for index in [*selected, *group_sources]:
        if index not in seen_context:
            context_sources.append(index)
            seen_context.add(index)
    for source in context_sources:
        if not add(source):
            break
        start = int(graph.indptr[source])
        stop = int(graph.indptr[source + 1])
        for destination in graph.indices[start:stop]:
            destination_index = int(destination)
            if destination_index not in selected_set:
                add(destination_index)
                break
        if len(selected) >= limit:
            break

    if len(selected) < limit:
        for source in context_sources:
            start = int(graph.indptr[source])
            stop = int(graph.indptr[source + 1])
            for destination in graph.indices[start:stop]:
                add(int(destination))
                if len(selected) >= limit:
                    break
            if len(selected) >= limit:
                break
    if len(selected) < limit:
        for index in range(graph.node_count):
            add(index)
            if len(selected) >= limit:
                break
    displayed_active_count = sum(
        1
        for index in selected
        if int(spike_counts[index]) > 0 or float(max_activity[index]) > 0.0
    )
    return (
        np.asarray(selected, dtype=np.intp),
        int(active_indices.size),
        displayed_active_count,
    )


def _display_edges(
    graph: GraphArrays, display_indices: np.ndarray, *, limit: int
) -> tuple[list[dict[str, Any]], bool]:
    """Return only genuine CSR edges whose endpoints are both displayed."""
    displayed = {int(index) for index in display_indices}
    edges: list[dict[str, Any]] = []
    for source in sorted(displayed):
        start = int(graph.indptr[source])
        stop = int(graph.indptr[source + 1])
        sign = int(graph.nt_code[source])
        for edge_index in range(start, stop):
            destination = int(graph.indices[edge_index])
            if destination not in displayed:
                continue
            if len(edges) >= limit:
                return edges, True
            edges.append(
                {
                    "csr_edge_index": edge_index,
                    "source_model_index": source,
                    "target_model_index": destination,
                    "source_body_id": int(graph.node_ids[source]),
                    "target_body_id": int(graph.node_ids[destination]),
                    "weight": float(graph.weights[edge_index]),
                    "sign": sign,
                }
            )
    return edges, False


def select_input_groups(
    graph: GraphArrays, *, group_size: int = 8
) -> tuple[tuple[int, ...], ...]:
    """Select 32 deterministic source groups, returning model indices."""
    if type(group_size) is not int or group_size <= 0:
        raise DynamicsError("group_size must be a positive integer")
    out_degree = np.diff(graph.indptr)
    candidates = np.flatnonzero((out_degree > 0) & (graph.nt_code != -128))
    if candidates.size == 0:
        raise DynamicsError("graph has no usable outgoing source neuron")

    def rank(index: int) -> bytes:
        body_id = int(graph.node_ids[index])
        return hashlib.sha256(
            f"{INPUT_GROUP_VERSION}|{graph.model_id}|{body_id}".encode("ascii")
        ).digest()

    ranked = sorted((int(index) for index in candidates), key=lambda index: (rank(index), index))
    needed = CHANNEL_COUNT * group_size
    reused = len(ranked) < needed
    selected = [ranked[index % len(ranked)] for index in range(needed)] if reused else ranked[:needed]
    return tuple(
        tuple(selected[channel * group_size : (channel + 1) * group_size])
        for channel in range(CHANNEL_COUNT)
    )


def _cancelled(cancel: Callable[[], bool] | object | None) -> bool:
    if cancel is None:
        return False
    if callable(cancel):
        return bool(cancel())
    checker = getattr(cancel, "is_set", None)
    if callable(checker):
        return bool(checker())
    raise TypeError("cancel must be a callable, Event-like object, or None")


def _input_group_metadata(
    graph: GraphArrays, groups: tuple[tuple[int, ...], ...]
) -> dict[str, Any]:
    unique = {index for group in groups for index in group}
    return {
        "version": INPUT_GROUP_VERSION,
        "rule": "out_degree>0 and nt_code!=-128; SHA-256 rank(model_id, body_id); channel-major partition",
        "channel_count": CHANNEL_COUNT,
        "group_size": len(groups[0]),
        "unique_neuron_count": len(unique),
        "reused_for_small_graph": len(unique) < CHANNEL_COUNT * len(groups[0]),
        "body_ids_by_channel": [
            [int(graph.node_ids[index]) for index in group] for group in groups
        ],
    }


def select_output_pools(
    graph: GraphArrays, *, group_size: int = OUTPUT_POOL_GROUP_SIZE
) -> tuple[tuple[int, ...], ...]:
    """Select 128 deterministic readout pools, returning model indices.

    Selection depends only on the version, graph model ID and real body IDs.
    It intentionally does not inspect activity or the bounded display subset.
    Tiny synthetic graphs are cycled in ranked order and explicitly marked as
    reused by :func:`_output_pool_features`.
    """
    if type(group_size) is not int or group_size <= 0:
        raise DynamicsError("output pool group_size must be a positive integer")

    def rank(index: int) -> bytes:
        body_id = int(graph.node_ids[index])
        return hashlib.sha256(
            f"{OUTPUT_POOL_VERSION}|{graph.model_id}|{body_id}".encode("ascii")
        ).digest()

    ranked = sorted(range(graph.node_count), key=lambda index: (rank(index), index))
    needed = OUTPUT_FEATURE_COUNT * group_size
    selected = [ranked[index % len(ranked)] for index in range(needed)]
    return tuple(
        tuple(selected[pool * group_size : (pool + 1) * group_size])
        for pool in range(OUTPUT_FEATURE_COUNT)
    )


def select_downstream_output_pools(
    graph: GraphArrays,
    *,
    input_group_size: int = 8,
    output_group_size: int = OUTPUT_POOL_GROUP_SIZE,
) -> tuple[tuple[int, ...], ...]:
    """Select four label-blind, one-hop readout pools per input channel.

    Real edge weights rank downstream targets.  No activity, Go state, teacher
    action, or validation label participates in pool selection.
    """
    if type(output_group_size) is not int or output_group_size <= 0:
        raise DynamicsError("output_group_size must be a positive integer")
    input_groups = select_input_groups(graph, group_size=input_group_size)
    input_nodes = {node for group in input_groups for node in group}
    used: set[int] = set()
    pools: list[tuple[int, ...]] = []
    per_channel = (OUTPUT_FEATURE_COUNT // CHANNEL_COUNT) * output_group_size
    for channel, sources in enumerate(input_groups):
        scores: dict[int, float] = {}
        for source in sources:
            begin, end = int(graph.indptr[source]), int(graph.indptr[source + 1])
            for target, weight in zip(
                graph.indices[begin:end], graph.weights[begin:end], strict=True
            ):
                target_index = int(target)
                if target_index not in input_nodes:
                    scores[target_index] = scores.get(target_index, 0.0) + float(weight)
        ranked = sorted(
            scores,
            key=lambda node: (-scores[node], int(graph.node_ids[node])),
        )
        selected = [node for node in ranked if node not in used][:per_channel]
        if len(selected) < per_channel:
            selected.extend(
                node for node in ranked
                if node not in selected and len(selected) < per_channel
            )
        if len(selected) < per_channel:
            fallback = sorted(
                range(graph.node_count),
                key=lambda node: hashlib.sha256(
                    f"{DOWNSTREAM_OUTPUT_POOL_VERSION}|{graph.model_id}|{channel}|{int(graph.node_ids[node])}".encode("ascii")
                ).digest(),
            )
            for node in fallback:
                if len(selected) >= per_channel:
                    break
                if node not in selected and node not in input_nodes:
                    selected.append(node)
        if not selected:  # tiny synthetic fixtures only
            selected = [0]
        selected = [selected[index % len(selected)] for index in range(per_channel)]
        used.update(selected)
        for offset in range(0, per_channel, output_group_size):
            pools.append(tuple(selected[offset:offset + output_group_size]))
    return tuple(pools)


def _output_pool_features(
    graph: GraphArrays,
    spike_counts: np.ndarray,
    max_activity: np.ndarray,
    *,
    tick_count: int,
    group_size: int = OUTPUT_POOL_GROUP_SIZE,
    input_group_size: int = 8,
    selection_mode: str = "random-v1",
) -> dict[str, Any]:
    """Build bounded features from the complete arrays of this exact run."""
    if selection_mode == "random-v1":
        pools = select_output_pools(graph, group_size=group_size)
        version = OUTPUT_POOL_VERSION
        selection_rule = (
            "all controller nodes; SHA-256 rank(version, model_id, body_id); "
            "pool-major partition; cycle only when graph is smaller than requested"
        )
    elif selection_mode == "input-downstream-v1":
        pools = select_downstream_output_pools(
            graph, input_group_size=input_group_size, output_group_size=group_size
        )
        version = DOWNSTREAM_OUTPUT_POOL_VERSION
        selection_rule = (
            "four pools per input channel; one-hop targets ranked by summed real "
            "edge weight, independent of Go labels and simulated activity"
        )
    else:
        raise DynamicsError("unsupported output-pool selection mode")
    body_ids_by_pool = [
        [int(graph.node_ids[index]) for index in pool] for pool in pools
    ]
    definition: dict[str, Any] = {
        "version": version,
        "model_id": graph.model_id,
        "feature_count": OUTPUT_FEATURE_COUNT,
        "group_size": group_size,
        "unique_neuron_count": len({index for pool in pools for index in pool}),
        "reused_for_small_graph": graph.node_count < OUTPUT_FEATURE_COUNT * group_size,
        "selection_rule": selection_rule,
        "feature_formula": (
            "0.5*mean(clip(spike_count/tick_count,0,1)) + "
            "0.5*mean(clip(max_activity,0,1))"
        ),
        "activity_source": "same_simulate_frame_full_arrays",
        "body_ids_by_pool": body_ids_by_pool,
    }
    group_hash = _sha256_json(definition)
    features: list[float] = []
    divisor = float(max(1, tick_count))
    for pool in pools:
        indices = np.asarray(pool, dtype=np.intp)
        spike_component = float(
            np.mean(np.clip(spike_counts[indices].astype(np.float64) / divisor, 0.0, 1.0))
        )
        activity_component = float(
            np.mean(np.clip(max_activity[indices].astype(np.float64), 0.0, 1.0))
        )
        features.append(round(0.5 * spike_component + 0.5 * activity_component, 8))
    feature_record = {
        "version": version,
        "group_hash": group_hash,
        "features": features,
    }
    return {
        **definition,
        "group_hash": group_hash,
        "features": features,
        "features_hash": _sha256_json(feature_record),
    }


def simulate_frame(
    graph: GraphArrays,
    stimulus: Mapping[str, Any],
    *,
    parameters: LIFParameters | None = None,
    seed: int = 0,
    cancel: Callable[[], bool] | object | None = None,
    timeout_seconds: float | None = None,
    output_pool_selection: str = "random-v1",
) -> dict[str, Any]:
    """Run one bounded sparse LIF window and return a JSON-safe real frame.

    "Real" here means values come from this exact simulation over the supplied
    CSR graph.  It does not mean experimentally recorded or biologically valid.
    Any cancellation, timeout or guard violation raises without returning a
    partial frame; callers must withhold downstream decisions.
    """
    graph.validate()
    params = parameters or LIFParameters()
    params.validate()
    if type(seed) is not int or seed < 0:
        raise DynamicsError("seed must be a nonnegative integer")
    if timeout_seconds is not None and (
        not math.isfinite(timeout_seconds) or timeout_seconds <= 0
    ):
        raise DynamicsError("timeout_seconds must be finite and positive")
    if stimulus.get("version") not in (
        STIMULUS_ADAPTER_VERSION, SPATIAL_STIMULUS_ADAPTER_VERSION
    ):
        raise DynamicsError("unsupported stimulus adapter version")
    rates_raw = stimulus.get("rates_hz")
    if not isinstance(rates_raw, Sequence) or len(rates_raw) != CHANNEL_COUNT:
        raise DynamicsError("stimulus must contain 32 rates")
    rates = np.asarray(rates_raw, dtype=np.float64)
    if not np.all(np.isfinite(rates)) or np.any(rates < 0) or np.any(rates > MAX_RATE_HZ):
        raise DynamicsError("stimulus rates must be finite values in [0, 150]")
    unsigned_stimulus = dict(stimulus)
    supplied_stimulus_hash = unsigned_stimulus.pop("stimulus_hash", None)
    if not isinstance(supplied_stimulus_hash, str) or _sha256_json(unsigned_stimulus) != supplied_stimulus_hash:
        raise DynamicsError("stimulus_hash does not match stimulus payload")

    groups = select_input_groups(graph, group_size=params.stimulus_group_size)
    started = time.perf_counter()
    deadline = None if timeout_seconds is None else started + timeout_seconds
    tick_count = int(math.ceil(params.duration_ms / params.dt_ms))
    delay_ticks = max(1, int(round(params.synaptic_delay_ms / params.dt_ms)))
    refractory_ticks = max(1, int(round(params.refractory_ms / params.dt_ms)))
    decay = math.exp(-params.dt_ms / params.membrane_tau_ms)
    n = graph.node_count
    potential = np.full(n, params.rest_mv, dtype=np.float32)
    max_activity = np.zeros(n, dtype=np.float32)
    min_potential = np.full(n, params.rest_mv, dtype=np.float32)
    max_potential = np.full(n, params.rest_mv, dtype=np.float32)
    spike_counts = np.zeros(n, dtype=np.uint32)
    refractory_until = np.zeros(n, dtype=np.int32)
    ring: list[list[int]] = [[] for _ in range(delay_ticks + 1)]
    accumulators = np.zeros(CHANNEL_COUNT, dtype=np.float64)
    # A seed changes deterministic initial phases, not graph selection.
    for channel in range(CHANNEL_COUNT):
        digest = hashlib.sha256(f"{seed}|{channel}|{DYNAMICS_VERSION}".encode("ascii")).digest()
        accumulators[channel] = int.from_bytes(digest[:8], "little") / 2**64

    total_spikes = 0
    synaptic_events = 0
    recorded: list[dict[str, Any]] = []
    recorded_total = 0
    per_tick_counts: list[int] = []

    for tick in range(tick_count):
        if _cancelled(cancel):
            raise DynamicsCancelled("simulation cancelled; no frame was produced")
        if deadline is not None and time.perf_counter() >= deadline:
            raise DynamicsTimeout("simulation deadline reached; no frame was produced")

        refractory = refractory_until > tick
        potential[:] = params.rest_mv + (potential - params.rest_mv) * decay
        potential[refractory] = params.reset_mv

        arriving = ring[tick % len(ring)]
        ring[tick % len(ring)] = []
        for source in arriving:
            start = int(graph.indptr[source])
            stop = int(graph.indptr[source + 1])
            degree = stop - start
            synaptic_events += degree
            if synaptic_events > params.max_synaptic_events:
                raise EventLimitExceeded("max_synaptic_events exceeded; no frame was produced")
            sign = int(graph.nt_code[source])
            if degree and sign in (-1, 1):
                destinations = graph.indices[start:stop].astype(np.intp, copy=False)
                deltas = graph.weights[start:stop] * (params.mv_per_contact * sign)
                active = refractory_until[destinations] <= tick
                if np.any(active):
                    np.add.at(potential, destinations[active], deltas[active])
        np.maximum(potential, params.floor_mv, out=potential)
        np.minimum(min_potential, potential, out=min_potential)
        np.maximum(max_potential, potential, out=max_potential)

        threshold_spikes = np.flatnonzero((potential >= params.threshold_mv) & ~refractory)
        accumulators += rates * params.dt_ms / 1000.0
        firing_channels = np.flatnonzero(accumulators >= 1.0)
        accumulators[firing_channels] -= np.floor(accumulators[firing_channels])
        if firing_channels.size:
            forced = np.fromiter(
                (index for channel in firing_channels for index in groups[int(channel)]),
                dtype=np.intp,
            )
            spike_indices = np.unique(np.concatenate((threshold_spikes, forced)))
        else:
            spike_indices = threshold_spikes
        count = int(spike_indices.size)
        if count > params.max_spikes_per_tick:
            raise EventLimitExceeded("max_spikes_per_tick exceeded; no frame was produced")
        total_spikes += count
        if total_spikes > params.max_total_spikes:
            raise EventLimitExceeded("max_total_spikes exceeded; no frame was produced")
        per_tick_counts.append(count)
        if count:
            spike_counts[spike_indices] += 1
            potential[spike_indices] = params.reset_mv
            refractory_until[spike_indices] = tick + refractory_ticks
            ring[(tick + delay_ticks) % len(ring)].extend(int(v) for v in spike_indices)
            remaining = params.max_recorded_spikes - len(recorded)
            if remaining > 0:
                for index in spike_indices[:remaining]:
                    recorded.append(
                        {
                            "tick": tick,
                            "time_ms": round(tick * params.dt_ms, 6),
                            "model_index": int(index),
                            "body_id": int(graph.node_ids[index]),
                        }
                    )
            recorded_total += count
        activity = np.clip(
            (max_potential.astype(np.float64) - params.rest_mv)
            / (params.threshold_mv - params.rest_mv),
            0.0,
            1.0,
        )
        np.maximum(max_activity, activity.astype(np.float32), out=max_activity)

    # Readout features are computed from the full controller arrays before any
    # display downsampling. They therefore remain coupled to this exact run,
    # not to whichever nodes happen to fit in a browser payload.
    output_pool = _output_pool_features(
        graph,
        spike_counts,
        max_activity,
        tick_count=tick_count,
        input_group_size=params.stimulus_group_size,
        selection_mode=output_pool_selection,
    )
    display_indices, active_candidate_count, displayed_active_count = _display_subset(
        graph,
        groups,
        spike_counts,
        max_activity,
        limit=min(n, params.display_node_limit),
    )
    display_nodes = [
        {
            "model_index": int(index),
            "body_id": int(graph.node_ids[index]),
            "soma_xyz": [round(float(value), 6) for value in graph.soma_xyz[index]],
            "max_activity": round(float(max_activity[index]), 6),
            "spike_count": int(spike_counts[index]),
            "min_potential_mv": round(float(min_potential[index]), 6),
            "max_potential_mv": round(float(max_potential[index]), 6),
        }
        for index in display_indices
    ]
    display_edges, display_edges_truncated = _display_edges(
        graph, display_indices, limit=params.display_edge_limit
    )
    parameters_payload = asdict(params)
    parameters_payload.update(
        {
            "version": DYNAMICS_VERSION,
            "delay_ticks": delay_ticks,
            "refractory_ticks": refractory_ticks,
            "tick_count": tick_count,
            "nt_code_interpretation": {
                "1": "excitatory",
                "-1": "inhibitory",
                "0": "zero_fast_current_engineering_assumption",
                "-128": "unknown_zero_fast_current_fail_safe",
            },
        }
    )
    deterministic_core: dict[str, Any] = {
        "schema": RUNTIME_SCHEMA,
        "model_id": graph.model_id,
        "manifest_sha256": graph.manifest_sha256,
        "board_hash": stimulus.get("board_hash"),
        "encoding_hash": stimulus.get("encoding_hash"),
        "stimulus_hash": supplied_stimulus_hash,
        "parameters": parameters_payload,
        "parameters_hash": _sha256_json(parameters_payload),
        "seed": seed,
        "input_groups": _input_group_metadata(graph, groups),
        "controller_node_count": graph.node_count,
        "controller_edge_count": graph.edge_count,
        "simulated_ms": round(tick_count * params.dt_ms, 6),
        "tick_spike_counts": per_tick_counts,
        "total_spike_count": total_spikes,
        "synaptic_event_count": synaptic_events,
        "recorded_spikes": recorded,
        "recorded_spikes_truncated": recorded_total > len(recorded),
        "output_pool": output_pool,
        "display_nodes": display_nodes,
        "display_node_count": len(display_nodes),
        "display_node_limit": params.display_node_limit,
        "display_active_candidate_count": active_candidate_count,
        "display_active_node_count": displayed_active_count,
        "display_active_candidates_truncated": (
            active_candidate_count > displayed_active_count
        ),
        "display_nodes_truncated": graph.node_count > len(display_nodes),
        "display_edges": display_edges,
        "display_edge_count": len(display_edges),
        "display_edge_limit": params.display_edge_limit,
        "display_edges_truncated": display_edges_truncated,
        "display_selection": (
            "activity-ranked nodes plus actual stimulus sources and real one-hop "
            "CSR destinations; unused capacity uses a stable structural prefix"
        ),
        "activity_source": "same_controller_frame",
        "silenced": total_spikes == 0,
        "disclaimer": DISCLAIMER,
        "go_move_selected": False,
        "baseline_controller_called": False,
    }
    frame_id = _sha256_json(deterministic_core)
    frame = {"frame_id": frame_id, **deterministic_core}
    frame["wall_time_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
    return frame


def simulate_go_encoding(
    graph: GraphArrays,
    encoding: Mapping[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    """Convenience boundary: validate/compress a Go encoding, then simulate."""
    stimulus = go_encoding_to_stimulus(encoding)
    frame = simulate_frame(graph, stimulus, **kwargs)
    frame["stimulus"] = stimulus
    return frame


__all__ = [
    "AssetValidationError",
    "CHANNEL_COUNT",
    "DISCLAIMER",
    "DYNAMICS_VERSION",
    "DynamicsCancelled",
    "DynamicsError",
    "DynamicsTimeout",
    "EventLimitExceeded",
    "GraphArrays",
    "INPUT_GROUP_VERSION",
    "LIFParameters",
    "MAX_RATE_HZ",
    "OUTPUT_FEATURE_COUNT",
    "OUTPUT_POOL_GROUP_SIZE",
    "OUTPUT_POOL_VERSION",
    "DOWNSTREAM_OUTPUT_POOL_VERSION",
    "RUNTIME_SCHEMA",
    "STIMULUS_ADAPTER_VERSION",
    "SPATIAL_STIMULUS_ADAPTER_VERSION",
    "go_encoding_to_stimulus",
    "go_encoding_to_spatial_stimulus",
    "load_graph_assets",
    "select_input_groups",
    "select_output_pools",
    "select_downstream_output_pools",
    "simulate_frame",
    "simulate_go_encoding",
]
