"""Offline integration tests for the resumable MaleCNS downloader."""

from __future__ import annotations

import base64
import hashlib
import json
import tempfile
import threading
import unittest
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator

from scripts.download_malecns_full import (
    DownloadError,
    DownloadSpec,
    IntegrityError,
    RemoteMetadata,
    _partial_metadata,
    _safe_child,
    _write_json_atomic,
    download_one,
    download_specs,
    resolve_output_dir,
)


PAYLOAD = (b"male-cns-test-fixture-" * 4096) + b"done"
ETAG = '"offline-test-etag"'


def _md5_base64(payload: bytes) -> str:
    return base64.b64encode(hashlib.md5(payload).digest()).decode("ascii")


class _FixtureHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _headers(self, length: int) -> None:
        server = self.server
        self.send_header("Content-Length", str(length))
        self.send_header("ETag", ETAG)
        self.send_header("X-Goog-Hash", "crc32c=AAAAAA==")
        self.send_header("X-Goog-Hash", f"md5={server.advertised_md5}")
        self.send_header("Last-Modified", "Tue, 01 Sep 2026 00:00:00 GMT")
        self.send_header("Connection", "close")

    def do_HEAD(self) -> None:  # noqa: N802 - stdlib handler API
        if self.path != "/fixture.feather":
            self.send_error(404)
            return
        self.send_response(200)
        self._headers(len(self.server.payload))
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        if self.path != "/fixture.feather":
            self.send_error(404)
            return
        payload = self.server.served_payload
        range_header = self.headers.get("Range")
        self.server.seen_ranges.append(range_header)
        if range_header:
            prefix = "bytes="
            if not range_header.startswith(prefix) or not range_header.endswith("-"):
                self.send_error(400)
                return
            start = int(range_header[len(prefix) : -1])
            chunk = payload[start:]
            self.send_response(206)
            self.send_header(
                "Content-Range", f"bytes {start}-{len(payload) - 1}/{len(payload)}"
            )
            self._headers(len(chunk))
            self.end_headers()
            self.wfile.write(chunk)
            return
        self.send_response(200)
        self._headers(len(payload))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        return


@contextmanager
def fixture_server(
    payload: bytes = PAYLOAD, *, served_payload: bytes | None = None
) -> Iterator[tuple[str, ThreadingHTTPServer]]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FixtureHandler)
    server.payload = payload
    server.served_payload = payload if served_payload is None else served_payload
    server.advertised_md5 = _md5_base64(payload)
    server.seen_ranges = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}/fixture.feather", server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def fixture_spec(url: str, payload: bytes = PAYLOAD) -> DownloadSpec:
    return DownloadSpec(
        key="fixture",
        filename="fixture.feather",
        url=url,
        expected_bytes=len(payload),
        expected_md5_base64=_md5_base64(payload),
    )


class DownloaderIntegrationTests(unittest.TestCase):
    def test_download_and_atomic_manifest(self) -> None:
        with fixture_server() as (url, server), tempfile.TemporaryDirectory() as raw_dir:
            output_dir = Path(raw_dir)
            spec = fixture_spec(url)
            manifest, manifest_path = download_specs(
                (spec,), output_dir, allowed_urls=frozenset({url})
            )

            self.assertEqual((output_dir / spec.filename).read_bytes(), PAYLOAD)
            self.assertEqual(server.seen_ranges, [None])
            self.assertEqual(manifest["files"]["fixture"]["status"], "downloaded")
            self.assertEqual(
                manifest["files"]["fixture"]["sha256"],
                hashlib.sha256(PAYLOAD).hexdigest(),
            )
            on_disk = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(on_disk, manifest)
            self.assertFalse(list(output_dir.glob("*.tmp")))

    def test_matching_partial_file_is_resumed_with_range(self) -> None:
        split = len(PAYLOAD) // 3
        with fixture_server() as (url, server), tempfile.TemporaryDirectory() as raw_dir:
            output_dir = Path(raw_dir)
            output_dir.mkdir(exist_ok=True)
            spec = fixture_spec(url)
            part = _safe_child(output_dir, spec.filename + ".part")
            sidecar = _safe_child(output_dir, spec.filename + ".part.meta.json")
            part.write_bytes(PAYLOAD[:split])
            remote = RemoteMetadata(len(PAYLOAD), ETAG, _md5_base64(PAYLOAD), "test")
            _write_json_atomic(sidecar, _partial_metadata(spec, remote))

            entry = download_one(spec, output_dir, allowed_urls=frozenset({url}))

            self.assertEqual(entry["status"], "resumed")
            self.assertEqual(entry["resumed_bytes"], split)
            self.assertEqual(server.seen_ranges, [f"bytes={split}-"])
            self.assertEqual((output_dir / spec.filename).read_bytes(), PAYLOAD)
            self.assertFalse(part.exists())
            self.assertFalse(sidecar.exists())

    def test_checksum_failure_removes_untrusted_temporary_files(self) -> None:
        corrupt = b"X" * len(PAYLOAD)
        with fixture_server(served_payload=corrupt) as (url, _server), tempfile.TemporaryDirectory() as raw_dir:
            output_dir = Path(raw_dir)
            spec = fixture_spec(url)
            with self.assertRaisesRegex(IntegrityError, "MD5 mismatch"):
                download_one(spec, output_dir, allowed_urls=frozenset({url}))

            self.assertFalse((output_dir / spec.filename).exists())
            self.assertFalse((output_dir / (spec.filename + ".part")).exists())
            self.assertFalse((output_dir / (spec.filename + ".part.meta.json")).exists())

    def test_unknown_url_is_rejected_before_network_access(self) -> None:
        spec = fixture_spec("https://example.com/fixture.feather")
        with tempfile.TemporaryDirectory() as raw_dir:
            with self.assertRaisesRegex(DownloadError, "fixed MaleCNS"):
                download_one(spec, Path(raw_dir))

    def test_destination_traversal_is_rejected(self) -> None:
        with fixture_server() as (url, _server), tempfile.TemporaryDirectory() as raw_dir:
            spec = DownloadSpec(
                key="fixture",
                filename="../escape.feather",
                url=url,
                expected_bytes=len(PAYLOAD),
                expected_md5_base64=_md5_base64(PAYLOAD),
            )
            with self.assertRaisesRegex(DownloadError, "unsafe destination"):
                download_one(
                    spec, Path(raw_dir), allowed_urls=frozenset({url})
                )

    def test_relative_output_directory_traversal_is_rejected(self) -> None:
        with self.assertRaisesRegex(DownloadError, "may not contain"):
            resolve_output_dir("../outside")


if __name__ == "__main__":
    unittest.main()
