"""Tests for the cached, read-only Stage 8 candidate audit."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.audit_stage8_candidate import (
    audit_candidate,
    audit_cross_split_hashes,
    audit_hybrid_controller,
    main,
)
from scripts.train_neural_go_readout import canonical_bytes, file_hash, load_cache, object_hash
from src.malecns_policy import CheckpointValidationError, load_policy_checkpoint, write_policy_checkpoint


class Stage8CandidateAuditTests(unittest.TestCase):
    def _fixture(
        self, root: Path, *, hybrid: bool = False
    ) -> tuple[Path, Path, list[dict[str, object]]]:
        cache = root / "cache"
        checkpoint = root / "checkpoint"
        cache.mkdir()
        rows = (
            ("train", 1, "natural-teacher-turn", 0),
            ("train", 2, "natural-teacher-turn", 2),
            ("train", 1, "constructed-opponent-pass-response", 81),
            ("train", 2, "constructed-opponent-pass-response", 81),
            ("validation", 1, "natural-teacher-turn", 1),
            ("validation", 2, "natural-teacher-turn", 3),
            ("validation", 1, "constructed-opponent-pass-response", 81),
            ("validation", 2, "constructed-opponent-pass-response", 81),
        )
        features = np.zeros((len(rows), 128), dtype=np.float32)
        for row in range(len(rows)):
            features[row, row] = 1.0
        labels = np.asarray([item[3] for item in rows], dtype=np.int64)
        records: list[dict[str, object]] = []
        for row, (split, colour, kind, _label) in enumerate(rows):
            state = {
                "row": row,
                "to_play": colour,
                "consecutive_passes": (
                    1 if kind == "constructed-opponent-pass-response" else 0
                ),
            }
            legal_mask = [True] * 82
            record: dict[str, object] = {
                "schema": "malecns-go-stage4-sample-v1",
                "sample_id": object_hash(["sample", row]),
                "game_id": f"game-{row}",
                "split": split,
                "feature_row": row,
                "board_hash": object_hash(["board", row]),
                "state": state,
                "teacher_colour": colour,
                "supervision_provenance": {"kind": kind},
                "label_action": int(labels[row]),
                "label_teacher": "teacher-v1",
                "legal_mask": legal_mask,
                "hashes": {
                    "state_sha256": object_hash(state),
                    "encoding_sha256": object_hash(["encoding", row]),
                    "features_sha256": object_hash(["features", row]),
                    "label_sha256": object_hash(
                        {"teacher": "teacher-v1", "action": int(labels[row])}
                    ),
                    "cached_feature_row_sha256": object_hash(
                        {
                            "dtype": "float32",
                            "values": [float(value) for value in features[row]],
                        }
                    ),
                },
            }
            record["record_sha256"] = object_hash(record)
            records.append(record)
        samples_path = cache / "samples.jsonl"
        samples_path.write_bytes(
            b"".join(canonical_bytes(record) + b"\n" for record in records)
        )
        np.save(cache / "features.npy", features, allow_pickle=False)
        np.save(cache / "labels.npy", labels, allow_pickle=False)
        manifest: dict[str, object] = {
            "schema": "malecns-go-stage4-dataset-v1",
            "graph": {"model_id": "model-v1", "manifest_sha256": "a" * 64},
            "output_pool": {"group_hash": "b" * 64},
            "artifacts": {
                "samples.jsonl": {"sha256": file_hash(samples_path)},
                "features.npy": {"sha256": file_hash(cache / "features.npy")},
                "labels.npy": {"sha256": file_hash(cache / "labels.npy")},
            },
        }
        manifest["dataset_hash"] = object_hash(manifest)
        (cache / "manifest.json").write_bytes(canonical_bytes(manifest) + b"\n")

        weights = np.zeros((82, 128), dtype=np.float32)
        for row, label in enumerate(labels):
            weights[label, row] = 10.0
        training_info: dict[str, object] = {
            "method": "test",
            "dataset_id": manifest["dataset_hash"],
            "dataset_hash": manifest["dataset_hash"],
            "sample_count": len(rows),
            "completed_at": "deterministic-test",
        }
        if hybrid:
            training_info.update(
                {
                    "decision_architecture": "hybrid-rule-after-opponent-pass-v1",
                    "pass_control": "rule-after-opponent-pass",
                    "neural_action_space": "point-actions-0..80",
                    "pass_selection_source": "deterministic-go-rule",
                    "controller_sources": {
                        "point_actions": "malecns-linear-readout",
                        "pass_after_opponent_pass": "go-rule-pass-gate",
                        "pass_when_only_legal": "go-rule-pass-gate",
                    },
                    "fit_sample_count": 4,
                }
            )
        write_policy_checkpoint(
            checkpoint,
            weights=weights,
            bias=np.zeros(82, dtype=np.float32),
            model_id="model-v1",
            graph_manifest_sha256="a" * 64,
            output_pool_group_hash="b" * 64,
            training_status="trained",
            training_info=training_info,
        )
        return cache, checkpoint, records

    def test_audit_reports_grouped_metrics_and_hash_compatibility(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache, checkpoint, _ = self._fixture(Path(temporary))
            report = audit_candidate(
                cache_directory=cache, checkpoint_directory=checkpoint
            )
            self.assertTrue(report["read_only"])
            self.assertFalse(report["full_graph_simulation_run"])
            self.assertTrue(all(report["compatibility"].values()))
            self.assertTrue(report["cross_split_hash_integrity"]["zero_overlap"])
            for split in ("train", "validation"):
                metrics = report["metrics"][split]
                self.assertEqual(metrics["overall"]["masked_teacher_agreement_rate"], 1.0)
                self.assertEqual(metrics["black"]["sample_count"], 2)
                self.assertEqual(metrics["white"]["sample_count"], 2)
                self.assertEqual(metrics["natural"]["sample_count"], 2)
                self.assertEqual(metrics["pass_response"]["sample_count"], 2)
                self.assertEqual(
                    metrics["pass_response"]["pass_teacher_masked_prediction_rate"],
                    1.0,
                )

    def test_dataset_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache, checkpoint, _ = self._fixture(Path(temporary))
            manifest_path = checkpoint / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            # Rewriting through the checkpoint writer is unnecessary here: the
            # strict loader authenticates this modified manifest and the audit
            # then rejects its dataset provenance.
            manifest["training_info"]["dataset_hash"] = "c" * 64
            manifest_path.write_bytes(canonical_bytes(manifest))
            with self.assertRaisesRegex(ValueError, "dataset_hash_matches"):
                audit_candidate(cache_directory=cache, checkpoint_directory=checkpoint)

    def test_v3_state_overlap_and_board_only_overlap_are_separate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache, checkpoint, records = self._fixture(root)
            report_path = root / "games.json"
            report_path.write_text(
                json.dumps(
                    {
                        "games": [
                            {
                                "moves": [
                                    {
                                        "actor": "malecns-neural-policy",
                                        "state_sha256_before": records[2]["hashes"][
                                            "state_sha256"
                                        ],
                                        "board_hash_before": records[1]["board_hash"],
                                    },
                                    {
                                        "actor": "legal-random",
                                        "state_sha256_before": records[0]["hashes"][
                                            "state_sha256"
                                        ],
                                        "board_hash_before": records[0]["board_hash"],
                                    },
                                ]
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            audit = audit_candidate(
                cache_directory=cache,
                checkpoint_directory=checkpoint,
                game_reports=[report_path],
            )
            overlap = audit["evaluation_overlap"]
            self.assertTrue(overlap["state_comparison_complete"])
            self.assertEqual(overlap["state_overlap_count"], 1)
            self.assertEqual(overlap["board_only_conservative_overlap_count"], 1)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(
                        [
                            "--cache",
                            str(cache),
                            "--checkpoint",
                            str(checkpoint),
                            "--game-report",
                            str(report_path),
                            "--strict-overlap",
                        ]
                    ),
                    1,
                )

    def test_cross_split_encoding_and_feature_overlap_is_detected(self) -> None:
        records = [
            {
                "split": "train",
                "hashes": {"encoding_sha256": "a", "features_sha256": "b"},
            },
            {
                "split": "validation",
                "hashes": {"encoding_sha256": "a", "features_sha256": "b"},
            },
        ]
        audit = audit_cross_split_hashes(records)
        self.assertFalse(audit["zero_overlap"])
        self.assertEqual(audit["encoding"]["overlap_count"], 1)
        self.assertEqual(audit["features"]["overlap_count"], 1)

    def test_hybrid_primary_gate_uses_rule_pass_and_neural_points(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache, checkpoint, _ = self._fixture(Path(temporary), hybrid=True)
            report = audit_candidate(
                cache_directory=cache, checkpoint_directory=checkpoint
            )
            self.assertEqual(
                report["decision_architecture"],
                "hybrid-rule-after-opponent-pass-v1",
            )
            hybrid = report["hybrid_acceptance"]
            self.assertTrue(hybrid["metric_gate_passed"])
            self.assertEqual(hybrid["routing"]["rule_route_count"], 4)
            self.assertEqual(hybrid["routing"]["neural_route_count"], 4)
            self.assertTrue(hybrid["routing"]["mutually_exclusive"])
            self.assertTrue(hybrid["routing"]["complete"])
            self.assertEqual(hybrid["natural_false_pass"]["count"], 0)
            self.assertEqual(hybrid["constructed_rule_gate"]["hit_rate"], 1.0)
            self.assertEqual(
                hybrid["primary_metrics"]["validation"]["natural_point"][
                    "agreement_rate"
                ],
                1.0,
            )
            self.assertEqual(
                hybrid["primary_metrics"]["validation"]["black"]["agreement_rate"],
                1.0,
            )
            self.assertEqual(
                hybrid["primary_metrics"]["validation"]["white"]["agreement_rate"],
                1.0,
            )
            self.assertFalse(
                report["diagnostics"]["legacy_82_class"]["used_for_acceptance"]
            )
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(
                        [
                            "--cache",
                            str(cache),
                            "--checkpoint",
                            str(checkpoint),
                            "--strict-overlap",
                        ]
                    ),
                    0,
                )

    def test_hybrid_discloses_training_teacher_conflict_without_hiding_validation_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache, checkpoint_dir, _ = self._fixture(Path(temporary), hybrid=True)
            features, labels, records, _ = load_cache(cache)
            records[0]["state"]["consecutive_passes"] = 1
            checkpoint = load_policy_checkpoint(checkpoint_dir)
            report = audit_hybrid_controller(
                features,
                labels,
                records,
                checkpoint.weights,
                checkpoint.bias,
                checkpoint.training_info,
            )
            self.assertIsNotNone(report)
            self.assertEqual(report["natural_false_pass"]["train_count"], 1)
            self.assertEqual(report["natural_false_pass"]["validation_count"], 0)
            self.assertTrue(report["metric_gate_passed"])

    def test_unknown_or_incomplete_hybrid_contract_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache, checkpoint, _ = self._fixture(root, hybrid=True)
            manifest_path = checkpoint / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["training_info"]["decision_architecture"] = "hybrid-unknown-v9"
            manifest_path.write_bytes(canonical_bytes(manifest))
            with self.assertRaisesRegex(CheckpointValidationError, "decision architecture"):
                audit_candidate(cache_directory=cache, checkpoint_directory=checkpoint)

    def test_strict_mode_rejects_report_missing_v3_state_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache, checkpoint, records = self._fixture(root)
            report_path = root / "legacy.json"
            report_path.write_text(
                json.dumps(
                    {
                        "moves": [
                            {
                                "actor": "malecns-neural-policy",
                                "board_hash_before": records[0]["board_hash"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(
                        [
                            "--cache",
                            str(cache),
                            "--checkpoint",
                            str(checkpoint),
                            "--game-report",
                            str(report_path),
                            "--strict-overlap",
                        ]
                    ),
                    1,
                )


if __name__ == "__main__":
    unittest.main()
