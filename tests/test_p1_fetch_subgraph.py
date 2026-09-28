"""Offline tests for the bounded P1 MaleCNS response parser."""

from __future__ import annotations

import copy
import hashlib
import json
import unittest

from scripts.p1_fetch_subgraph import (
    EXPECTED_COLUMNS,
    MAX_RESPONSE_BYTES,
    P1Error,
    SOURCE_URL,
    canonical_bytes,
    read_and_validate_response,
    validate_payload,
)


EXPECTED_CANONICAL_SHA256 = "ae456a6013e9c4af9612a9caf527ba1630799dbb70518dc2a313fbc32af5333b"


def valid_payload() -> dict[str, list]:
    return {
        "columns": EXPECTED_COLUMNS.copy(),
        "data": [
            [12781, 522822, 288, "DNge104", "AN17A076"],
            [556329, 15884, 238, "DNge104", "AN17A076"],
            [12781, 513052, 150, "DNge104", "DNg15"],
            [556329, 10141, 116, "DNge104", "DNg15"],
            [12781, 162471, 82, "DNge104", "AN09B020"],
            [556329, 25047, 79, "DNge104", "AN09B020"],
            [556329, 805181, 70, "DNge104", "SNta20"],
            [556329, 800721, 69, "DNge104", "IN09B014"],
            [12781, 22728, 66, "DNge104", "AN17A004"],
            [12781, 800632, 61, "DNge104", "IN09B014"],
        ],
    }


class FakeResponse:
    def __init__(
        self,
        body: bytes,
        *,
        status: int = 200,
        url: str = SOURCE_URL,
        content_type: str = "application/json; charset=UTF-8",
        content_length: str | None = None,
    ) -> None:
        self._body = body
        self._status = status
        self._url = url
        self.headers = {"Content-Type": content_type}
        if content_length is not None:
            self.headers["Content-Length"] = content_length

    def getcode(self) -> int:
        return self._status

    def geturl(self) -> str:
        return self._url

    def read(self, limit: int) -> bytes:
        return self._body[:limit]


class PayloadValidationTests(unittest.TestCase):
    def test_normal_payload_and_canonical_hash(self) -> None:
        normalized = validate_payload(valid_payload())
        digest = hashlib.sha256(canonical_bytes(normalized)).hexdigest()
        self.assertEqual(digest, EXPECTED_CANONICAL_SHA256)

    def test_columns_must_match_exactly(self) -> None:
        payload = valid_payload()
        payload["columns"][0] = "source"
        with self.assertRaisesRegex(P1Error, "columns"):
            validate_payload(payload)

    def test_row_count_must_be_exactly_ten(self) -> None:
        payload = valid_payload()
        payload["data"].pop()
        with self.assertRaisesRegex(P1Error, "exactly 10"):
            validate_payload(payload)

    def test_negative_pre_body_is_rejected(self) -> None:
        payload = valid_payload()
        payload["data"][0][0] = -1
        with self.assertRaisesRegex(P1Error, "field pre"):
            validate_payload(payload)

    def test_float_pre_body_is_rejected(self) -> None:
        payload = valid_payload()
        payload["data"][0][0] = 12781.0
        with self.assertRaisesRegex(P1Error, "field pre"):
            validate_payload(payload)

    def test_negative_post_body_is_rejected(self) -> None:
        payload = valid_payload()
        payload["data"][0][1] = -1
        with self.assertRaisesRegex(P1Error, "field post"):
            validate_payload(payload)

    def test_float_post_body_is_rejected(self) -> None:
        payload = valid_payload()
        payload["data"][0][1] = 522822.0
        with self.assertRaisesRegex(P1Error, "field post"):
            validate_payload(payload)

    def test_negative_weight_is_rejected(self) -> None:
        payload = valid_payload()
        payload["data"][0][2] = -1
        with self.assertRaisesRegex(P1Error, "field weight"):
            validate_payload(payload)

    def test_float_weight_is_rejected(self) -> None:
        payload = valid_payload()
        payload["data"][0][2] = 288.0
        with self.assertRaisesRegex(P1Error, "field weight"):
            validate_payload(payload)

    def test_boolean_integer_field_is_rejected(self) -> None:
        payload = valid_payload()
        payload["data"][0][2] = True
        with self.assertRaisesRegex(P1Error, "field weight"):
            validate_payload(payload)


class HttpBoundaryTests(unittest.TestCase):
    @staticmethod
    def _body() -> bytes:
        payload = copy.deepcopy(valid_payload())
        payload["debug"] = "ignored upstream debug field"
        return json.dumps(payload).encode("utf-8")

    def test_normal_http_response(self) -> None:
        body = self._body()
        normalized, byte_count, final_url = read_and_validate_response(
            FakeResponse(body, content_length=str(len(body)))
        )
        self.assertEqual(normalized, valid_payload())
        self.assertEqual(byte_count, len(body))
        self.assertEqual(final_url, SOURCE_URL)

    def test_oversized_actual_response_is_rejected(self) -> None:
        response = FakeResponse(b"x" * (MAX_RESPONSE_BYTES + 1))
        with self.assertRaisesRegex(P1Error, "response exceeds"):
            read_and_validate_response(response)

    def test_oversized_declared_response_is_rejected(self) -> None:
        response = FakeResponse(b"{}", content_length=str(MAX_RESPONSE_BYTES + 1))
        with self.assertRaisesRegex(P1Error, "declared response exceeds"):
            read_and_validate_response(response)

    def test_wrong_final_host_is_rejected(self) -> None:
        response = FakeResponse(self._body(), url="https://example.com/api/custom/custom")
        with self.assertRaisesRegex(P1Error, "final URL"):
            read_and_validate_response(response)

    def test_wrong_status_is_rejected(self) -> None:
        response = FakeResponse(self._body(), status=206)
        with self.assertRaisesRegex(P1Error, "status"):
            read_and_validate_response(response)

    def test_wrong_content_type_is_rejected(self) -> None:
        response = FakeResponse(self._body(), content_type="text/html")
        with self.assertRaisesRegex(P1Error, "Content-Type"):
            read_and_validate_response(response)


if __name__ == "__main__":
    unittest.main()
