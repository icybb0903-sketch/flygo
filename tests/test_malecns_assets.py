from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

try:
    import pyarrow as pa
    import pyarrow.feather as feather
except ImportError:  # System Python intentionally does not carry phase-3 deps.
    pa = None
    feather = None

from src.malecns_assets import (
    AssetValidationError,
    SOURCE_FILES,
    build_runtime_assets,
    load_runtime_assets,
    sha256_file,
)


@unittest.skipIf(pa is None, "phase-3 fixture tests require project .venv pyarrow")
class MaleCNSAssetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        annotations = pa.table(
            {
                "bodyId": pa.array([1, 2, 3, 4, 5], type=pa.int64()),
                "superclass": pa.array(["central", "motor", "", "sensory", "central"]),
                "somaLocation": pa.array(
                    [[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2], [10, 11, 12]],
                    type=pa.list_(pa.int64()),
                ),
            }
        )
        nts = pa.table(
            {
                "body": pa.array([1, 2, 3, 4], type=pa.int64()),
                "consensus_nt": pa.array(["acetylcholine", "gaba", "dopamine", "mystery"]),
            }
        )
        weights = pa.table(
            {
                "body_pre": pa.array([1, 1, 1, 2, 2, 4, 5], type=pa.int64()),
                "body_post": pa.array([2, 2, 5, 5, 1, 1, 1], type=pa.int64()),
                "weight": pa.array([5, 7, 4, 6, 4, 9, 8], type=pa.int64()),
            }
        )
        feather.write_feather(annotations, self.source / SOURCE_FILES["annotations"])
        feather.write_feather(nts, self.source / SOURCE_FILES["neurotransmitters"])
        feather.write_feather(weights, self.source / SOURCE_FILES["weights"])
        self.hashes = {key: sha256_file(self.source / name) for key, name in SOURCE_FILES.items()}

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_filter_threshold_duplicate_merge_missing_nt_and_mmap(self) -> None:
        output = build_runtime_assets(
            self.source, self.root / "out", expected_sha256=self.hashes, minimum_weight=5
        )
        assets = load_runtime_assets(output)
        self.assertIsInstance(assets.node_ids, np.memmap)
        np.testing.assert_array_equal(assets.node_ids, [1, 2, 5])
        np.testing.assert_array_equal(assets.nt_code, [1, -1, -128])
        np.testing.assert_array_equal(assets.indptr, [0, 1, 2, 3])
        np.testing.assert_array_equal(assets.indices, [1, 2, 0])
        np.testing.assert_array_equal(assets.weights, [12.0, 6.0, 8.0])
        counts = assets.manifest["counts"]
        self.assertEqual(counts["filtered_edge_records"], 4)
        self.assertEqual(counts["distinct_edges"], 3)
        self.assertEqual(counts["merged_duplicate_records"], 1)
        self.assertEqual(counts["neurotransmitters"]["missing_nodes"], 1)

    def test_repeated_build_is_byte_deterministic(self) -> None:
        first = build_runtime_assets(
            self.source, self.root / "a", expected_sha256=self.hashes
        )
        second = build_runtime_assets(
            self.source, self.root / "b", expected_sha256=self.hashes
        )
        self.assertEqual(first.name, second.name)
        for filename in ["manifest.json", "node_ids.npy", "soma_xyz.npy", "indptr.npy", "indices.npy", "weights.npy", "nt_code.npy"]:
            self.assertEqual(sha256_file(first / filename), sha256_file(second / filename))

    def test_source_hash_mismatch_fails_closed(self) -> None:
        bad = dict(self.hashes)
        bad["weights"] = "0" * 64
        with self.assertRaisesRegex(AssetValidationError, "source SHA-256 mismatch"):
            build_runtime_assets(self.source, self.root / "bad-source", expected_sha256=bad)

    def test_corrupt_artifact_hash_is_rejected(self) -> None:
        output = build_runtime_assets(
            self.source, self.root / "corrupt", expected_sha256=self.hashes
        )
        path = output / "weights.npy"
        content = bytearray(path.read_bytes())
        content[-1] ^= 0x01
        path.write_bytes(content)
        with self.assertRaisesRegex(AssetValidationError, "SHA-256 mismatch"):
            load_runtime_assets(output)


if __name__ == "__main__":
    unittest.main()
