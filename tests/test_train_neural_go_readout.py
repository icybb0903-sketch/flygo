"""Fast synthetic tests for the Stage 4 offline training pipeline."""

from __future__ import annotations

import gc
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts.train_neural_go_readout import (
    FEATURE_COUNT,
    PASS_ACTION_INDEX,
    canonical_bytes,
    deduplicate_positions_by_state,
    derive_lif_seed,
    evaluate_readout,
    feature_equivalence_components,
    file_hash,
    fit_ridge_readout,
    generate_cache,
    generate_teacher_positions,
    load_cache,
    object_hash,
    split_game_ids,
    train_from_cache,
    Position,
)
from src.go_engine import BLACK, WHITE, initial_state
from src.malecns_policy import (
    CheckpointValidationError,
    load_policy_checkpoint,
    write_policy_checkpoint,
)


class Stage4TrainingTests(unittest.TestCase):
    def test_pass_response_augmentation_is_explicit_and_non_terminal(self) -> None:
        positions = generate_teacher_positions(
            game_count=2,
            max_moves=12,
            seed=23,
            trajectory_mode="role-symmetric",
            pass_response_samples_per_trajectory=1,
        )
        augmented = [
            position
            for position in positions
            if position.supervision_kind == "constructed-opponent-pass-response"
        ]
        self.assertEqual(len(augmented), 4)
        for position in augmented:
            self.assertEqual(position.teacher_action, PASS_ACTION_INDEX)
            self.assertEqual(position.label_teacher, "pass-response-augmentation-v1")
            self.assertEqual(position.state.consecutive_passes, 1)
            self.assertFalse(position.state.game_over)
            self.assertEqual(position.state.to_play, position.teacher_colour)
            self.assertIsNotNone(position.source_state_sha256)
            self.assertIsNotNone(position.source_ply)
            self.assertGreaterEqual(position.source_ply or 0, 9)

        disabled = generate_teacher_positions(
            game_count=2,
            max_moves=12,
            seed=23,
            trajectory_mode="role-symmetric",
        )
        self.assertFalse(
            any(item.supervision_kind != "natural-teacher-turn" for item in disabled)
        )
        with self.assertRaisesRegex(ValueError, "only in role-symmetric"):
            generate_teacher_positions(
                game_count=2,
                max_moves=12,
                seed=23,
                trajectory_mode="legacy",
                pass_response_samples_per_trajectory=1,
            )

    def test_role_symmetric_mode_only_labels_the_controlled_teacher(self) -> None:
        positions = generate_teacher_positions(
            game_count=2,
            max_moves=10,
            seed=29,
            trajectory_mode="role-symmetric",
        )
        self.assertEqual({position.teacher_colour for position in positions}, {BLACK, WHITE})
        self.assertTrue(
            all(position.state.to_play == position.teacher_colour for position in positions)
        )
        self.assertEqual(
            {position.trajectory_kind for position in positions},
            {
                "teacher-black-vs-seeded-random",
                "teacher-white-vs-seeded-random",
            },
        )
        by_group: dict[str, list[Position]] = {}
        for position in positions:
            by_group.setdefault(position.split_group_id or "", []).append(position)
        self.assertEqual(len(by_group), 2)
        for paired_positions in by_group.values():
            self.assertEqual(
                len({position.random_stream_id for position in paired_positions}), 1
            )
            self.assertEqual(
                {position.teacher_colour for position in paired_positions}, {BLACK, WHITE}
            )

    def test_role_symmetric_mode_supports_one_hundred_move_horizon(self) -> None:
        positions = generate_teacher_positions(
            game_count=2,
            max_moves=100,
            seed=31,
            trajectory_mode="role-symmetric",
        )
        self.assertTrue(positions)
        self.assertLess(max(position.ply for position in positions), 100)
        self.assertTrue(any(position.ply >= 72 for position in positions))

    def test_state_deduplication_is_stable(self) -> None:
        state = initial_state()
        first = Position("first", 0, state, 0, 0, BLACK)
        duplicate = Position("second", 0, state, 1, 0, BLACK)
        unique, removed = deduplicate_positions_by_state([first, duplicate])
        self.assertEqual(unique, [first])
        self.assertEqual(removed, 1)

    def test_feature_equivalence_keeps_connected_paired_groups_atomic(self) -> None:
        state = initial_state()
        positions = [
            Position("a-black", 0, state, 0, 0, split_group_id="a"),
            Position("b-white", 0, state, 0, 0, split_group_id="b"),
            Position("c-black", 0, state, 0, 0, split_group_id="c"),
        ]
        by_group, components = feature_equivalence_components(
            positions, ["1" * 64, "1" * 64, "2" * 64]
        )
        self.assertEqual(by_group["a"], by_group["b"])
        self.assertNotEqual(by_group["a"], by_group["c"])
        self.assertEqual(len(components), 2)
        self.assertIn(["a", "b"], components.values())

    def test_role_symmetric_cache_reports_zero_state_hash_overlap(self) -> None:
        fake_graph = SimpleNamespace(model_id="fake-graph", manifest_sha256="a" * 64)

        def fake_encoding(state):
            encoding_basis = (
                ["synthetic-pass-response-collision"]
                if state.consecutive_passes == 1
                else [state.board, state.to_play, state.consecutive_passes]
            )
            return {
                "board_hash": object_hash(state.board),
                "encoding_hash": object_hash(encoding_basis),
                "legal_mask": [True] * 82,
            }

        def fake_frame(graph, encoding, *, parameters, seed):
            digest_values = [
                int(encoding["encoding_hash"][index : index + 2], 16) / 255.0
                for index in range(0, 64, 2)
            ]
            vector = (digest_values * 4)[:FEATURE_COUNT]
            return {
                "frame_id": object_hash([encoding["encoding_hash"], seed]),
                "model_id": graph.model_id,
                "manifest_sha256": graph.manifest_sha256,
                "parameters_hash": object_hash({"duration_ms": parameters.duration_ms}),
                "output_pool": {
                    "version": "test-v1",
                    "group_hash": "b" * 64,
                    "features": vector,
                    "features_hash": object_hash(
                        {
                            "version": "test-v1",
                            "group_hash": "b" * 64,
                            "features": vector,
                        }
                    ),
                },
            }

        with tempfile.TemporaryDirectory() as temporary, patch(
            "scripts.train_neural_go_readout.load_graph_assets",
            return_value=fake_graph,
        ), patch(
            "scripts.train_neural_go_readout.encode_go_state",
            side_effect=fake_encoding,
        ), patch(
            "scripts.train_neural_go_readout.simulate_go_encoding",
            side_effect=fake_frame,
        ):
            manifest = generate_cache(
                graph_directory=Path(temporary) / "graph",
                cache_directory=Path(temporary) / "cache",
                game_count=4,
                max_moves=8,
                seed=37,
                validation_fraction=0.25,
                duration_ms=1.0,
                trajectory_mode="role-symmetric",
                pass_response_samples_per_trajectory=1,
            )
            audit = manifest["state_hash_integrity"]
            self.assertEqual(audit["train_validation_overlap_count"], 0)
            self.assertEqual(audit["retained_unique_state_count"], manifest["sample_count"])
            input_audit = manifest["model_input_integrity"]
            self.assertGreater(input_audit["encoding_duplicate_count_removed"], 0)
            self.assertEqual(input_audit["encoding_label_conflict_count"], 0)
            self.assertEqual(input_audit["train_validation_encoding_overlap_count"], 0)
            self.assertEqual(input_audit["train_validation_feature_overlap_count"], 0)
            self.assertEqual(
                manifest["generator"]["random_stream_policy"],
                "shared_per_game_pair_independent_of_teacher_colour-v1",
            )
            pass_audit = manifest["pass_response_augmentation"]
            self.assertTrue(pass_audit["enabled"])
            self.assertEqual(pass_audit["sample_count"], 1)
            self.assertFalse(pass_audit["natural_trajectory_claim"])
            self.assertFalse(pass_audit["constructed_states_are_terminal"])
            self.assertEqual(set(pass_audit["trajectory_sample_counts"].values()), {1})
            records = [
                json.loads(line)
                for line in (Path(temporary) / "cache" / "samples.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertTrue(
                all(record["state"]["to_play"] == record["teacher_colour"] for record in records)
            )
            augmented = [
                record
                for record in records
                if record["supervision_provenance"]["kind"]
                == "constructed-opponent-pass-response"
            ]
            self.assertTrue(augmented)
            self.assertTrue(
                all(
                    record["label_action"] == PASS_ACTION_INDEX
                    and record["state"]["consecutive_passes"] == 1
                    and not record["supervision_provenance"]["is_natural_trajectory_state"]
                    and record["supervision_provenance"]["source_game_id"]
                    == record["game_id"]
                    for record in augmented
                )
            )
            group_splits: dict[str, set[str]] = {}
            for record in records:
                group_splits.setdefault(record["split_group_id"], set()).add(record["split"])
            self.assertTrue(all(len(splits) == 1 for splits in group_splits.values()))
            result = train_from_cache(
                cache_directory=Path(temporary) / "cache",
                checkpoint_directory=Path(temporary) / "checkpoint",
                ridge=0.1,
            )
            metrics = result["metrics"]
            self.assertEqual(
                metrics["model_selection_metric"],
                "stratified.validation.natural.teacher_agreement_rate",
            )
            self.assertTrue(metrics["overall_metrics_include_constructed_pass_response"])
            for split in ("train", "validation"):
                self.assertEqual(
                    metrics["stratified"][split]["natural"]["sample_count"]
                    + metrics["stratified"][split]["pass_response"]["sample_count"],
                    metrics[split]["sample_count"],
                )
            hybrid = train_from_cache(
                cache_directory=Path(temporary) / "cache",
                checkpoint_directory=Path(temporary) / "hybrid-checkpoint",
                ridge=0.1,
                pass_control="rule-after-opponent-pass",
            )
            hybrid_info = hybrid["checkpoint_manifest"]["training_info"]
            self.assertEqual(
                hybrid_info["method"],
                "closed_form_ridge_natural_point_only_plus_rule_pass_gate_v1",
            )
            self.assertEqual(
                hybrid_info["decision_architecture"],
                "hybrid-rule-after-opponent-pass-v1",
            )
            self.assertEqual(hybrid_info["neural_action_space"], "point-actions-0..80")
            self.assertEqual(
                hybrid_info["pass_selection_source"], "deterministic-go-rule"
            )
            expected_fit_count = sum(
                record["split"] == "train"
                and record["supervision_provenance"]["kind"]
                == "natural-teacher-turn"
                and record["label_action"] != PASS_ACTION_INDEX
                for record in records
            )
            self.assertEqual(hybrid_info["fit_sample_count"], expected_fit_count)
            gate_metrics = hybrid["metrics"]["pass_control"]
            self.assertEqual(
                sum(
                    gate_metrics[split]["rule_pass_gate"]["hit_count"]
                    for split in ("train", "validation")
                ),
                len(augmented),
            )
            self.assertEqual(
                sum(
                    gate_metrics[split]["rule_pass_gate"]["false_pass_count"]
                    for split in ("train", "validation")
                ),
                0,
            )

    def test_role_symmetric_cache_fails_on_cross_split_feature_collision(self) -> None:
        fake_graph = SimpleNamespace(model_id="fake-graph", manifest_sha256="a" * 64)
        vector = [0.0] * FEATURE_COUNT
        feature_hash = object_hash(
            {"version": "test-v1", "group_hash": "b" * 64, "features": vector}
        )

        def constant_frame(graph, encoding, *, parameters, seed):
            return {
                "frame_id": object_hash([encoding["encoding_hash"], seed]),
                "model_id": graph.model_id,
                "manifest_sha256": graph.manifest_sha256,
                "parameters_hash": object_hash({"duration_ms": parameters.duration_ms}),
                "output_pool": {
                    "version": "test-v1",
                    "group_hash": "b" * 64,
                    "features": vector,
                    "features_hash": feature_hash,
                },
            }

        with tempfile.TemporaryDirectory() as temporary, patch(
            "scripts.train_neural_go_readout.load_graph_assets",
            return_value=fake_graph,
        ), patch(
            "scripts.train_neural_go_readout.simulate_go_encoding",
            side_effect=constant_frame,
        ):
            with self.assertRaisesRegex(RuntimeError, "feature equivalence connects"):
                generate_cache(
                    graph_directory=Path(temporary) / "graph",
                    cache_directory=Path(temporary) / "cache",
                    game_count=2,
                    max_moves=6,
                    seed=41,
                    validation_fraction=0.5,
                    duration_ms=1.0,
                    trajectory_mode="role-symmetric",
                )

    def test_both_colour_mode_labels_every_generated_ply(self) -> None:
        white_only = generate_teacher_positions(
            game_count=2,
            max_moves=4,
            seed=9,
            retained_colours="white-only",
        )
        both = generate_teacher_positions(
            game_count=2,
            max_moves=4,
            seed=9,
            retained_colours="both",
        )
        self.assertEqual(len(white_only), 4)
        self.assertEqual(len(both), 8)
        self.assertEqual({position.state.to_play for position in both}, {1, 2})

    def test_per_position_lif_seeds_are_deterministic_and_not_label_dependent(self) -> None:
        base = Position("game-a", 3, initial_state(), 7, 0)
        relabelled = Position("game-a", 3, initial_state(), 22, 4)
        other = Position("game-a", 5, initial_state(), 7, 0)
        self.assertEqual(
            derive_lif_seed(base, base_seed=19, mode="per-position"),
            derive_lif_seed(relabelled, base_seed=19, mode="per-position"),
        )
        self.assertNotEqual(
            derive_lif_seed(base, base_seed=19, mode="per-position"),
            derive_lif_seed(other, base_seed=19, mode="per-position"),
        )
        self.assertEqual(derive_lif_seed(base, base_seed=19, mode="fixed"), 19)
        with self.assertRaises(ValueError):
            derive_lif_seed(base, base_seed=19, mode="invented")

    def test_whole_game_split_is_deterministic_and_disjoint(self) -> None:
        games = [f"game-{index}" for index in range(10)]
        first = split_game_ids(games, validation_fraction=0.3, seed=17)
        second = split_game_ids(reversed(games), validation_fraction=0.3, seed=17)
        self.assertEqual(first, second)
        train = {game for game, split in first.items() if split == "train"}
        validation = {game for game, split in first.items() if split == "validation"}
        self.assertFalse(train & validation)
        self.assertEqual(train | validation, set(games))
        self.assertEqual(len(validation), 3)

    def test_ridge_training_and_metrics_are_deterministic(self) -> None:
        rng = np.random.default_rng(123)
        features = rng.random((24, FEATURE_COUNT), dtype=np.float32)
        labels = np.asarray([index % 6 for index in range(24)], dtype=np.int64)
        first = fit_ridge_readout(features, labels, ridge=0.1)
        second = fit_ridge_readout(features, labels, ridge=0.1)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        masks = np.ones((24, 82), dtype=bool)
        metrics = evaluate_readout(features, labels, masks, *first)
        self.assertEqual(metrics["sample_count"], 24)
        self.assertEqual(metrics["legal_action_rate"], 1.0)
        for name in ("top1_accuracy", "top5_accuracy", "teacher_agreement_rate"):
            self.assertGreaterEqual(metrics[name], 0.0)
            self.assertLessEqual(metrics[name], 1.0)

    def _write_synthetic_cache(self, root: Path) -> None:
        features = np.zeros((2, FEATURE_COUNT), dtype=np.float32)
        labels = np.asarray([0, 1], dtype=np.int64)
        records = []
        for row, split in enumerate(("train", "validation")):
            state = {"synthetic": True, "row": row}
            record = {
                "schema": "malecns-go-stage4-sample-v1",
                "sample_id": object_hash([row]),
                "game_id": f"game-{row}",
                "split": split,
                "feature_row": row,
                "state": state,
                "label_teacher": "deterministic-capture-first-v1",
                "hashes": {
                    "state_sha256": object_hash(state),
                    "label_sha256": object_hash(
                        {"teacher": "deterministic-capture-first-v1", "action": row}
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
        records_path = root / "samples.jsonl"
        records_path.write_bytes(b"".join(canonical_bytes(item) + b"\n" for item in records))
        np.save(root / "features.npy", features, allow_pickle=False)
        np.save(root / "labels.npy", labels, allow_pickle=False)
        manifest = {
            "schema": "malecns-go-stage4-dataset-v1",
            "artifacts": {
                "samples.jsonl": {"sha256": file_hash(records_path)},
                "features.npy": {"sha256": file_hash(root / "features.npy")},
                "labels.npy": {"sha256": file_hash(root / "labels.npy")},
            },
        }
        manifest["dataset_hash"] = object_hash(manifest)
        (root / "manifest.json").write_bytes(canonical_bytes(manifest) + b"\n")

    def test_synthetic_cache_hashes_are_checked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_synthetic_cache(root)
            features, labels, records, _ = load_cache(root)
            self.assertEqual(features.shape, (2, FEATURE_COUNT))
            self.assertEqual(labels.tolist(), [0, 1])
            self.assertEqual([item["split"] for item in records], ["train", "validation"])
            with (root / "labels.npy").open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
                load_cache(root)

    def test_checkpoint_loads_and_tampering_is_rejected(self) -> None:
        weights = np.zeros((82, FEATURE_COUNT), dtype=np.float32)
        bias = np.arange(82, dtype=np.float32)
        training_info = {
            "method": "synthetic-test",
            "dataset_id": "synthetic",
            "sample_count": 2,
            "completed_at": "not-a-wall-clock",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            digest = write_policy_checkpoint(
                root,
                weights=weights,
                bias=bias,
                model_id="synthetic-model",
                graph_manifest_sha256="a" * 64,
                output_pool_group_hash="b" * 64,
                training_status="trained",
                training_info=training_info,
            )
            loaded = load_policy_checkpoint(root)
            self.assertEqual(loaded.checkpoint_hash, digest)
            self.assertEqual(loaded.weights.shape, (82, FEATURE_COUNT))
            del loaded
            gc.collect()

            path = root / "weights.npy"
            payload = bytearray(path.read_bytes())
            payload[-1] ^= 1
            path.write_bytes(payload)
            with self.assertRaisesRegex(CheckpointValidationError, "SHA-256 mismatch"):
                load_policy_checkpoint(root)


if __name__ == "__main__":
    unittest.main()
