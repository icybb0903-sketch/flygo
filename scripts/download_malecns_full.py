"""Reliably download the three official MaleCNS v1.0 flat-connectome tables.

The CLI has no arbitrary URL option.  Sources, sizes, and GCS MD5 values are
pinned below so a typo, redirect, or unexpected object replacement fails closed.
Only Python's standard library is used.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import http.client
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


SCRIPT_VERSION = "1.0.0"
DATASET = "male-cns:v1.0"
OFFICIAL_HOST = "storage.googleapis.com"
OFFICIAL_PREFIX = "/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "malecns" / "v1.0"
MANIFEST_FILENAME = "manifest.json"
CHUNK_BYTES = 8 * 1024 * 1024
TIMEOUT_SECONDS = 60


class DownloadError(RuntimeError):
    """Raised when a source, response, path, or downloaded file is unsafe."""


class IntegrityError(DownloadError):
    """Raised when bytes do not match the pinned size or MD5 checksum."""


@dataclass(frozen=True)
class DownloadSpec:
    key: str
    filename: str
    url: str
    expected_bytes: int
    expected_md5_base64: str


@dataclass(frozen=True)
class RemoteMetadata:
    content_length: int
    etag: str
    md5_base64: str
    last_modified: str | None


OFFICIAL_FILES = (
    DownloadSpec(
        key="weights",
        filename="connectome-weights-male-cns-v1.0-minconf-0.5.feather",
        url=(
            "https://storage.googleapis.com/flyem-male-cns/v1.0/"
            "connectome-data/flat-connectome/"
            "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
        ),
        expected_bytes=1_051_241_946,
        expected_md5_base64="8w6dzKJc/QIb8eez2XVZng==",
    ),
    DownloadSpec(
        key="annotations",
        filename="body-annotations-male-cns-v1.0-minconf-0.5.feather",
        url=(
            "https://storage.googleapis.com/flyem-male-cns/v1.0/"
            "connectome-data/flat-connectome/"
            "body-annotations-male-cns-v1.0-minconf-0.5.feather"
        ),
        expected_bytes=14_483_314,
        expected_md5_base64="UKdxh3DFciDxYLpPQxq4ng==",
    ),
    DownloadSpec(
        key="neurotransmitters",
        filename="body-neurotransmitters-male-cns-v1.0.feather",
        url=(
            "https://storage.googleapis.com/flyem-male-cns/v1.0/"
            "connectome-data/flat-connectome/"
            "body-neurotransmitters-male-cns-v1.0.feather"
        ),
        expected_bytes=43_282_834,
        expected_md5_base64="PYQrEv5cSe763lKNfdJKHw==",
    ),
)
SPEC_BY_KEY = {spec.key: spec for spec in OFFICIAL_FILES}
OFFICIAL_URLS = frozenset(spec.url for spec in OFFICIAL_FILES)

OpenUrl = Callable[..., Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _decode_md5(value: str) -> bytes:
    try:
        decoded = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise DownloadError("MD5 metadata is not valid base64") from exc
    if len(decoded) != hashlib.md5().digest_size:
        raise DownloadError("MD5 metadata must decode to 16 bytes")
    return decoded


def _validate_official_url(url: str) -> None:
    if url not in OFFICIAL_URLS:
        raise DownloadError(f"URL is not in the fixed MaleCNS v1.0 allowlist: {url!r}")
    parts = urlsplit(url)
    if (
        parts.scheme != "https"
        or parts.hostname != OFFICIAL_HOST
        or parts.port not in (None, 443)
        or not parts.path.startswith(OFFICIAL_PREFIX)
        or parts.query
        or parts.fragment
        or parts.username
        or parts.password
    ):
        raise DownloadError(f"unsafe official source URL: {url!r}")


def validate_official_specs() -> None:
    """Fail closed if an edited source definition is ambiguous or unsafe."""
    keys: set[str] = set()
    filenames: set[str] = set()
    for spec in OFFICIAL_FILES:
        _validate_official_url(spec.url)
        _validate_filename(spec.filename)
        if spec.key in keys or spec.filename in filenames:
            raise DownloadError("official file keys and filenames must be unique")
        if spec.expected_bytes <= 0:
            raise DownloadError(f"invalid pinned byte count for {spec.key}")
        _decode_md5(spec.expected_md5_base64)
        if urlsplit(spec.url).path.rsplit("/", 1)[-1] != spec.filename:
            raise DownloadError(f"URL filename mismatch for {spec.key}")
        keys.add(spec.key)
        filenames.add(spec.filename)


def _validate_filename(filename: str) -> None:
    if (
        not filename
        or filename in {".", ".."}
        or filename.startswith(".")
        or "/" in filename
        or "\\" in filename
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", filename)
    ):
        raise DownloadError(f"unsafe destination filename: {filename!r}")


def _safe_child(directory: Path, filename: str) -> Path:
    """Return a single direct child, rejecting traversal and existing symlinks."""
    _validate_filename(filename)
    base = directory.resolve()
    candidate = (base / filename).resolve()
    if candidate.parent != base:
        raise DownloadError(f"destination escapes output directory: {filename!r}")
    return candidate


def resolve_output_dir(value: str | os.PathLike[str]) -> Path:
    """Resolve CLI output paths; relative ``..`` traversal is intentionally refused."""
    raw = Path(value).expanduser()
    if not raw.is_absolute():
        if ".." in raw.parts:
            raise DownloadError("relative output directory may not contain '..'")
        raw = PROJECT_ROOT / raw
    return raw.resolve()


def _status(response: Any) -> int:
    status = getattr(response, "status", None)
    if status is None:
        status = response.getcode()
    return int(status)


def _validate_final_url(response: Any, expected_url: str) -> None:
    if response.geturl() != expected_url:
        raise DownloadError(
            f"redirect or unexpected final URL refused: {response.geturl()!r}"
        )


def _header_md5(headers: Any) -> str | None:
    get_all = getattr(headers, "get_all", None)
    values = get_all("X-Goog-Hash") if callable(get_all) else None
    if not values:
        single = headers.get("X-Goog-Hash")
        values = [single] if single else []
    for value in values:
        for item in value.split(","):
            algorithm, separator, digest = item.strip().partition("=")
            if separator and algorithm.lower() == "md5":
                _decode_md5(digest)
                return digest
    return None


def _parse_content_length(headers: Any, *, field: str = "Content-Length") -> int:
    raw = headers.get(field)
    if raw is None:
        raise DownloadError(f"{field} header is required")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise DownloadError(f"{field} must be an integer") from exc
    if value < 0:
        raise DownloadError(f"{field} must not be negative")
    return value


def fetch_remote_metadata(
    spec: DownloadSpec,
    *,
    timeout: int = TIMEOUT_SECONDS,
    opener: OpenUrl = urlopen,
    allowed_urls: frozenset[str] | None = None,
) -> RemoteMetadata:
    """Read and validate GCS object metadata without downloading its body."""
    if allowed_urls is None:
        _validate_official_url(spec.url)
    elif spec.url not in allowed_urls:
        raise DownloadError(f"URL is not in the supplied test allowlist: {spec.url!r}")

    request = Request(
        spec.url,
        method="HEAD",
        headers={"Accept-Encoding": "identity", "User-Agent": f"malecns-full/{SCRIPT_VERSION}"},
    )
    try:
        with opener(request, timeout=timeout) as response:
            _validate_final_url(response, spec.url)
            if _status(response) != 200:
                raise DownloadError(f"HEAD returned HTTP {_status(response)}, expected 200")
            content_length = _parse_content_length(response.headers)
            etag = response.headers.get("ETag")
            md5_base64 = _header_md5(response.headers)
            last_modified = response.headers.get("Last-Modified")
    except DownloadError:
        raise
    except (HTTPError, URLError, TimeoutError, OSError, http.client.HTTPException) as exc:
        raise DownloadError(f"HEAD request failed for {spec.key}: {exc}") from exc

    if not etag:
        raise DownloadError("ETag header is required")
    if not md5_base64:
        raise DownloadError("X-Goog-Hash md5 metadata is required")
    if content_length != spec.expected_bytes:
        raise DownloadError(
            f"remote size changed for {spec.key}: {content_length} != {spec.expected_bytes}"
        )
    if md5_base64 != spec.expected_md5_base64:
        raise DownloadError(
            f"remote MD5 changed for {spec.key}: {md5_base64!r}"
        )
    return RemoteMetadata(content_length, etag, md5_base64, last_modified)


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _hash_file(path: Path) -> dict[str, str | int]:
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            byte_count += len(chunk)
            md5.update(chunk)
            sha256.update(chunk)
    return {
        "bytes": byte_count,
        "md5_base64": base64.b64encode(md5.digest()).decode("ascii"),
        "md5_hex": md5.hexdigest(),
        "sha256": sha256.hexdigest(),
    }


def _verify_file(path: Path, spec: DownloadSpec) -> dict[str, str | int]:
    actual = _hash_file(path)
    if actual["bytes"] != spec.expected_bytes:
        raise IntegrityError(
            f"size mismatch for {spec.filename}: {actual['bytes']} != {spec.expected_bytes}"
        )
    if actual["md5_base64"] != spec.expected_md5_base64:
        raise IntegrityError(f"MD5 mismatch for {spec.filename}")
    return actual


def _partial_metadata(spec: DownloadSpec, remote: RemoteMetadata) -> dict[str, Any]:
    return {
        "content_length": remote.content_length,
        "etag": remote.etag,
        "md5_base64": remote.md5_base64,
        "schema_version": 1,
        "source_url": spec.url,
    }


def _discard_partial(part_path: Path, metadata_path: Path) -> None:
    for path in (part_path, metadata_path):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _load_partial_offset(
    part_path: Path,
    metadata_path: Path,
    expected_metadata: dict[str, Any],
    expected_bytes: int,
) -> int:
    if not part_path.exists():
        _discard_partial(part_path, metadata_path)
        return 0
    try:
        if metadata_path.stat().st_size > 64 * 1024:
            raise ValueError("partial metadata is too large")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata != expected_metadata:
            raise ValueError("partial metadata does not match remote object")
        size = part_path.stat().st_size
        if not 0 <= size <= expected_bytes:
            raise ValueError("partial byte count is outside expected range")
        return size
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        _discard_partial(part_path, metadata_path)
        return 0


_CONTENT_RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+)")


def _validate_download_response(
    response: Any, spec: DownloadSpec, requested_offset: int
) -> tuple[str, int]:
    _validate_final_url(response, spec.url)
    status = _status(response)
    if requested_offset and status == 206:
        raw_range = response.headers.get("Content-Range", "")
        match = _CONTENT_RANGE.fullmatch(raw_range.strip())
        if not match:
            raise DownloadError("206 response has invalid Content-Range")
        start, end, total = map(int, match.groups())
        if start != requested_offset or total != spec.expected_bytes or end < start:
            raise DownloadError("206 Content-Range does not match requested object")
        declared = _parse_content_length(response.headers)
        if declared != end - start + 1:
            raise DownloadError("206 Content-Length does not match Content-Range")
        return "ab", requested_offset
    if status == 200:
        declared = _parse_content_length(response.headers)
        if declared != spec.expected_bytes:
            raise DownloadError("200 Content-Length does not match pinned object size")
        return "wb", 0
    raise DownloadError(f"GET returned HTTP {status}, expected 200 or valid 206")


def download_one(
    spec: DownloadSpec,
    output_dir: Path,
    *,
    timeout: int = TIMEOUT_SECONDS,
    opener: OpenUrl = urlopen,
    allowed_urls: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Download and atomically publish one verified file.

    ``allowed_urls`` exists only to permit loopback HTTP servers in offline tests.
    The CLI never supplies it and therefore always uses the official allowlist.
    """
    if allowed_urls is None:
        _validate_official_url(spec.url)
    elif spec.url not in allowed_urls:
        raise DownloadError(f"URL is not in the supplied test allowlist: {spec.url!r}")
    _decode_md5(spec.expected_md5_base64)

    final_path = _safe_child(output_dir, spec.filename)
    part_path = _safe_child(output_dir, spec.filename + ".part")
    metadata_path = _safe_child(output_dir, spec.filename + ".part.meta.json")
    output_dir.mkdir(parents=True, exist_ok=True)

    remote = fetch_remote_metadata(
        spec, timeout=timeout, opener=opener, allowed_urls=allowed_urls
    )
    if final_path.exists():
        try:
            actual = _verify_file(final_path, spec)
        except IntegrityError:
            pass
        else:
            return _manifest_entry(spec, remote, actual, "verified-existing", 0)

    expected_partial_metadata = _partial_metadata(spec, remote)
    offset = _load_partial_offset(
        part_path, metadata_path, expected_partial_metadata, spec.expected_bytes
    )
    _write_json_atomic(metadata_path, expected_partial_metadata)

    if offset == spec.expected_bytes:
        try:
            actual = _verify_file(part_path, spec)
        except IntegrityError:
            _discard_partial(part_path, metadata_path)
            offset = 0
            _write_json_atomic(metadata_path, expected_partial_metadata)
        else:
            os.replace(part_path, final_path)
            metadata_path.unlink(missing_ok=True)
            return _manifest_entry(spec, remote, actual, "resumed", offset)

    headers = {
        "Accept-Encoding": "identity",
        "User-Agent": f"malecns-full/{SCRIPT_VERSION}",
    }
    if offset:
        headers["Range"] = f"bytes={offset}-"
        headers["If-Range"] = remote.etag
    request = Request(spec.url, method="GET", headers=headers)

    try:
        with opener(request, timeout=timeout) as response:
            mode, retained_bytes = _validate_download_response(response, spec, offset)
            written = retained_bytes
            with part_path.open(mode) as handle:
                while chunk := response.read(CHUNK_BYTES):
                    written += len(chunk)
                    if written > spec.expected_bytes:
                        raise IntegrityError("response exceeded the pinned object size")
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
    except IntegrityError:
        _discard_partial(part_path, metadata_path)
        raise
    except DownloadError:
        _discard_partial(part_path, metadata_path)
        raise
    except (HTTPError, URLError, TimeoutError, OSError, http.client.HTTPException) as exc:
        # A transport interruption keeps the .part and its matching metadata so
        # the next run can resume from the exact verified remote object.
        raise DownloadError(f"GET request failed for {spec.key}: {exc}") from exc

    if part_path.stat().st_size != spec.expected_bytes:
        raise DownloadError(
            f"download incomplete for {spec.key}: "
            f"{part_path.stat().st_size}/{spec.expected_bytes} bytes"
        )
    try:
        actual = _verify_file(part_path, spec)
    except IntegrityError:
        _discard_partial(part_path, metadata_path)
        raise

    os.replace(part_path, final_path)
    metadata_path.unlink(missing_ok=True)
    status = "resumed" if offset else "downloaded"
    return _manifest_entry(spec, remote, actual, status, offset)


def _manifest_entry(
    spec: DownloadSpec,
    remote: RemoteMetadata,
    actual: dict[str, str | int],
    status: str,
    resumed_bytes: int,
) -> dict[str, Any]:
    return {
        "bytes": actual["bytes"],
        "etag": remote.etag,
        "filename": spec.filename,
        "last_modified": remote.last_modified,
        "md5_base64": actual["md5_base64"],
        "md5_hex": actual["md5_hex"],
        "pinned_bytes": spec.expected_bytes,
        "pinned_md5_base64": spec.expected_md5_base64,
        "resumed_bytes": resumed_bytes,
        "sha256": actual["sha256"],
        "source_url": spec.url,
        "status": status,
        "verified_at_utc": _utc_now(),
    }


def download_specs(
    specs: Sequence[DownloadSpec],
    output_dir: Path,
    *,
    timeout: int = TIMEOUT_SECONDS,
    opener: OpenUrl = urlopen,
    allowed_urls: frozenset[str] | None = None,
) -> tuple[dict[str, Any], Path]:
    """Download selected specs and atomically write their verification manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    entries: dict[str, Any] = {}
    for spec in specs:
        entries[spec.key] = download_one(
            spec,
            output_dir,
            timeout=timeout,
            opener=opener,
            allowed_urls=allowed_urls,
        )
    manifest = {
        "complete_official_set": {spec.key for spec in specs} == set(SPEC_BY_KEY),
        "dataset": DATASET,
        "files": entries,
        "generated_at_utc": _utc_now(),
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "selected": [spec.key for spec in specs],
    }
    manifest_path = _safe_child(output_dir, MANIFEST_FILENAME)
    _write_json_atomic(manifest_path, manifest)
    return manifest, manifest_path


def _selected_specs(keys: Iterable[str] | None) -> tuple[DownloadSpec, ...]:
    if not keys:
        return OFFICIAL_FILES
    selected: list[DownloadSpec] = []
    seen: set[str] = set()
    for key in keys:
        if key not in SPEC_BY_KEY:
            raise DownloadError(f"unknown --only value: {key!r}")
        if key not in seen:
            selected.append(SPEC_BY_KEY[key])
            seen.add(key)
    return tuple(selected)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and verify the official MaleCNS v1.0 flat-connectome files."
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"destination directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--only",
        action="append",
        choices=tuple(SPEC_BY_KEY),
        help="download one named table; repeat to select more than one",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the fixed plan without network access or filesystem writes",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_official_specs()
        output_dir = resolve_output_dir(args.output_dir)
        specs = _selected_specs(args.only)
        if args.dry_run:
            print(f"DRY_RUN dataset={DATASET} output={output_dir}")
            for spec in specs:
                print(
                    f"{spec.key}: bytes={spec.expected_bytes} "
                    f"md5={spec.expected_md5_base64} url={spec.url}"
                )
            return 0

        print(f"Downloading {len(specs)} MaleCNS file(s) to {output_dir}")
        manifest, manifest_path = download_specs(specs, output_dir)
    except DownloadError as exc:
        print(f"MALECNS_DOWNLOAD_FAIL: {exc}", file=sys.stderr)
        return 1

    for key, entry in manifest["files"].items():
        print(
            f"{key}: {entry['status']} bytes={entry['bytes']} "
            f"sha256={entry['sha256']}"
        )
    print(f"manifest={manifest_path}")
    print("MALECNS_DOWNLOAD_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
