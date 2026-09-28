"""Load and verify the fixed P1 MaleCNS subgraph before any computation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EXPECTED_DATASET = "male-cns:v1.0"
EXPECTED_COLUMNS = ["pre", "post", "weight", "pre_type", "post_type"]
EXPECTED_EDGE_COUNT = 10
MAX_INPUT_BYTES = 100 * 1024


class GraphDataError(RuntimeError):
    """Raised when P1 data or provenance is invalid."""


@dataclass(frozen=True, order=True)
class Edge:
    pre: int
    post: int
    weight: int
    pre_type: str
    post_type: str


@dataclass(frozen=True)
class VerifiedGraph:
    edges: tuple[Edge, ...]
    nodes: tuple[int, ...]
    node_types: tuple[tuple[int, str], ...]
    canonical_sha256: str

    @property
    def source_nodes(self) -> tuple[int, ...]:
        return tuple(sorted({edge.pre for edge in self.edges}))

    @property
    def downstream_nodes(self) -> tuple[int, ...]:
        return tuple(sorted({edge.post for edge in self.edges}))

    def nodes_of_type(self, neuron_type: str) -> tuple[int, ...]:
        return tuple(node for node, kind in self.node_types if kind == neuron_type)


def canonical_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _read_json(path: Path) -> Any:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise GraphDataError(f"cannot stat required file: {path}") from exc
    if size <= 0 or size > MAX_INPUT_BYTES:
        raise GraphDataError(f"file must contain 1..{MAX_INPUT_BYTES} bytes: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GraphDataError(f"cannot read valid UTF-8 JSON: {path}") from exc


def _positive_int(value: Any, field: str, row_index: int) -> int:
    if type(value) is not int or value <= 0:
        raise GraphDataError(f"row {row_index} field {field} must be a positive integer")
    return value


def validate_subgraph(payload: Any) -> tuple[dict[str, Any], tuple[Edge, ...]]:
    if not isinstance(payload, dict):
        raise GraphDataError("subgraph root must be an object")
    if payload.get("columns") != EXPECTED_COLUMNS:
        raise GraphDataError(f"columns must exactly equal {EXPECTED_COLUMNS!r}")
    rows = payload.get("data")
    if not isinstance(rows, list) or len(rows) != EXPECTED_EDGE_COUNT:
        raise GraphDataError(f"subgraph must contain exactly {EXPECTED_EDGE_COUNT} rows")

    normalized: list[list[Any]] = []
    edges: list[Edge] = []
    seen_pairs: set[tuple[int, int]] = set()
    node_types: dict[int, str] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != len(EXPECTED_COLUMNS):
            raise GraphDataError(f"row {index} must contain exactly five values")
        pre = _positive_int(row[0], "pre", index)
        post = _positive_int(row[1], "post", index)
        weight = _positive_int(row[2], "weight", index)
        pre_type, post_type = row[3], row[4]
        if not isinstance(pre_type, str) or not pre_type:
            raise GraphDataError(f"row {index} pre_type must be a non-empty string")
        if not isinstance(post_type, str) or not post_type:
            raise GraphDataError(f"row {index} post_type must be a non-empty string")
        pair = (pre, post)
        if pair in seen_pairs:
            raise GraphDataError(f"duplicate edge pair at row {index}: {pair}")
        seen_pairs.add(pair)
        for node, kind in ((pre, pre_type), (post, post_type)):
            previous = node_types.setdefault(node, kind)
            if previous != kind:
                raise GraphDataError(f"node {node} has conflicting type labels")
        normalized.append([pre, post, weight, pre_type, post_type])
        edges.append(Edge(pre, post, weight, pre_type, post_type))

    canonical = {"columns": EXPECTED_COLUMNS.copy(), "data": normalized}
    return canonical, tuple(edges)


def load_verified_graph(subgraph_path: Path, provenance_path: Path) -> VerifiedGraph:
    payload = _read_json(subgraph_path)
    provenance = _read_json(provenance_path)
    canonical, edges = validate_subgraph(payload)
    if not isinstance(provenance, dict):
        raise GraphDataError("provenance root must be an object")
    if provenance.get("dataset") != EXPECTED_DATASET:
        raise GraphDataError(f"provenance dataset must be {EXPECTED_DATASET}")

    encoded = canonical_bytes(canonical)
    actual_hash = hashlib.sha256(encoded).hexdigest()
    expected_hash = provenance.get("canonical_sha256")
    if not isinstance(expected_hash, str) or actual_hash != expected_hash:
        raise GraphDataError("P1 canonical SHA-256 does not match provenance")
    if provenance.get("canonical_bytes") != len(encoded):
        raise GraphDataError("P1 canonical byte count does not match provenance")

    node_type_map: dict[int, str] = {}
    for edge in edges:
        node_type_map[edge.pre] = edge.pre_type
        node_type_map[edge.post] = edge.post_type
    nodes = tuple(sorted(node_type_map))
    return VerifiedGraph(
        edges=edges,
        nodes=nodes,
        node_types=tuple(sorted(node_type_map.items())),
        canonical_sha256=actual_hash,
    )
