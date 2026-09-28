"""Fetch one bounded, real MaleCNS v1.0 subgraph using only Python stdlib."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


SCRIPT_VERSION = "1.0.0"
SOURCE_URL = "https://neuprint.janelia.org/api/custom/custom"
EXPECTED_HOST = "neuprint.janelia.org"
EXPECTED_PATH = "/api/custom/custom"
DATASET = "male-cns:v1.0"
QUERY = """MATCH (n:Neuron)-[e:ConnectsTo]->(m:Neuron)
WHERE n.type = 'DNge104'
RETURN n.bodyId AS pre, m.bodyId AS post, e.weight AS weight,
       n.type AS pre_type, m.type AS post_type
ORDER BY weight DESC
LIMIT 10"""
EXPECTED_COLUMNS = ["pre", "post", "weight", "pre_type", "post_type"]
EXPECTED_ROW_COUNT = 10
MAX_RESPONSE_BYTES = 100 * 1024
TIMEOUT_SECONDS = 20
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "p1"


class P1Error(RuntimeError):
    """Raised when a P1 network or validation boundary is violated."""


def _require_positive_int(value: Any, field: str, row_index: int) -> int:
    if type(value) is not int or value <= 0:
        raise P1Error(f"row {row_index} field {field} must be a positive integer")
    return value


def validate_payload(payload: Any) -> dict[str, list[Any]]:
    """Return the canonical columns/data subset after strict schema validation."""
    if not isinstance(payload, dict):
        raise P1Error("JSON root must be an object")

    columns = payload.get("columns")
    if columns != EXPECTED_COLUMNS:
        raise P1Error(f"columns must exactly equal {EXPECTED_COLUMNS!r}")

    data = payload.get("data")
    if not isinstance(data, list) or len(data) != EXPECTED_ROW_COUNT:
        raise P1Error(f"data must contain exactly {EXPECTED_ROW_COUNT} rows")

    normalized_rows: list[list[Any]] = []
    for index, row in enumerate(data):
        if not isinstance(row, list) or len(row) != len(EXPECTED_COLUMNS):
            raise P1Error(f"row {index} must contain exactly {len(EXPECTED_COLUMNS)} values")

        pre = _require_positive_int(row[0], "pre", index)
        post = _require_positive_int(row[1], "post", index)
        weight = _require_positive_int(row[2], "weight", index)
        pre_type, post_type = row[3], row[4]
        if not isinstance(pre_type, str) or not pre_type:
            raise P1Error(f"row {index} field pre_type must be a non-empty string")
        if not isinstance(post_type, str) or not post_type:
            raise P1Error(f"row {index} field post_type must be a non-empty string")
        normalized_rows.append([pre, post, weight, pre_type, post_type])

    return {"columns": EXPECTED_COLUMNS.copy(), "data": normalized_rows}


def canonical_bytes(payload: dict[str, list[Any]]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _content_type(headers: Any) -> str:
    value = headers.get("Content-Type", "")
    return value.split(";", 1)[0].strip().lower()


def _validate_final_url(url: str) -> None:
    parts = urlsplit(url)
    if (
        parts.scheme != "https"
        or parts.hostname != EXPECTED_HOST
        or parts.port not in (None, 443)
        or parts.path != EXPECTED_PATH
        or parts.query
        or parts.fragment
        or parts.username
        or parts.password
    ):
        raise P1Error("response final URL is outside the fixed HTTPS neuPrint endpoint")


def read_and_validate_response(response: Any) -> tuple[dict[str, list[Any]], int, str]:
    """Validate HTTP metadata, enforce the byte cap, and parse the response."""
    status = response.getcode()
    if status != 200:
        raise P1Error(f"HTTP status must be 200, got {status!r}")

    final_url = response.geturl()
    _validate_final_url(final_url)

    if _content_type(response.headers) != "application/json":
        raise P1Error("Content-Type must be application/json")

    declared_length = response.headers.get("Content-Length")
    if declared_length is not None:
        try:
            declared_bytes = int(declared_length)
        except (TypeError, ValueError) as exc:
            raise P1Error("Content-Length must be an integer when present") from exc
        if declared_bytes < 0 or declared_bytes > MAX_RESPONSE_BYTES:
            raise P1Error(f"declared response exceeds {MAX_RESPONSE_BYTES} bytes")

    body = response.read(MAX_RESPONSE_BYTES + 1)
    if len(body) > MAX_RESPONSE_BYTES:
        raise P1Error(f"response exceeds {MAX_RESPONSE_BYTES} bytes")

    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise P1Error("response is not valid UTF-8 JSON") from exc

    return validate_payload(payload), len(body), final_url


def fetch_live() -> tuple[dict[str, list[Any]], int, str]:
    """POST the fixed query without reading, printing, or sending credentials."""
    _validate_final_url(SOURCE_URL)
    request_body = json.dumps(
        {"cypher": QUERY, "dataset": DATASET},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    request = Request(
        SOURCE_URL,
        data=request_body,
        method="POST",
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": f"malecns-p1/{SCRIPT_VERSION}",
        },
    )
    with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        return read_and_validate_response(response)


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def persist_result(
    normalized: dict[str, list[Any]], response_bytes: int, final_url: str
) -> tuple[str, int, Path, Path]:
    encoded = canonical_bytes(normalized)
    digest = hashlib.sha256(encoded).hexdigest()
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    subgraph_path = OUTPUT_DIR / "subgraph.json"
    provenance_path = OUTPUT_DIR / "provenance.json"

    provenance = {
        "canonical_bytes": len(encoded),
        "canonical_sha256": digest,
        "dataset": DATASET,
        "fetched_at_utc": fetched_at,
        "query": QUERY,
        "response_bytes": response_bytes,
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "source_url": final_url,
    }
    _write_json_atomic(subgraph_path, normalized)
    _write_json_atomic(provenance_path, provenance)
    return digest, len(encoded), subgraph_path, provenance_path


def main() -> int:
    try:
        normalized, response_bytes, final_url = fetch_live()
        digest, encoded_size, subgraph_path, provenance_path = persist_result(
            normalized, response_bytes, final_url
        )
    except (P1Error, HTTPError, URLError, TimeoutError, OSError) as exc:
        print(f"P1_FAIL error={type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    rows = normalized["data"]
    nodes = {row[0] for row in rows} | {row[1] for row in rows}
    pre_bodies = sorted({row[0] for row in rows})
    print("P1_PASS")
    print(f"dataset={DATASET}")
    print(f"source={final_url}")
    print(f"response_bytes={response_bytes}")
    print(f"canonical_bytes={encoded_size}")
    print(f"rows={len(rows)}")
    print(f"nodes={len(nodes)}")
    print(f"pre_bodies={','.join(map(str, pre_bodies))}")
    print(f"max_weight={max(row[2] for row in rows)}")
    print(f"sha256={digest}")
    print(f"subgraph={subgraph_path.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"provenance={provenance_path.relative_to(PROJECT_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
