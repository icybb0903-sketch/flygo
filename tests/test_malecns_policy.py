"""Strict checkpoint and action-selection tests for the neural readout."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.go_engine import initial_state, step
from src.go_neural_encoding import PASS_ACTION_INDEX, encode_go_state
from src.malecns_dynamics import (
    GraphArrays,
    LIFParameters,
    go_encoding_to_stimulus,
    simulate_frame,
)
from src.malecns_policy import (
    CheckpointValidationError,
    PolicyInputError,
    initialise_linear_parameters,
    load_policy_checkpoint,
    select_action,
    write_policy_checkpoint,
)


def _json_hash(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def graph() -> GraphArrays:
    return GraphArrays.from_arrays(
        model_id="synthetic-policy-graph",
        manifest_sha256="a" * 64,
        node_ids=np.array([100, 101, 102], dtype=np.int64),
        soma_xyz=np.arange(9, dtype=np.float32).reshape(3, 3),
        indptr=np.array([0, 2, 2, 2], dtype=np.uint64),
        indices=np.array([1, 2], dtype=np.uint32),
        weights=np.array([30.0, 30.0], dtype=np.float32),
        nt_code=np.array([1, 1, 1], dtype=np.int8),
    )


def parameters() -> LIFParameters:
    return LIFParameters(
        dt_ms=1.0,
        duration_ms=15.0,
        synaptic_delay_ms=2.0,
        refractory_ms=2.0,
        mv_per_contact=0.275,
        stimulus_group_size=1,
        display_node_limit=2,
        display_edge_limit=2,
        max_synaptic_events=10_000,
        max_total_spikes=10_000,
        max_spikes_per_tick=1_000,
    )


def controlled_frame(rate: float, encoding: dict[str, object]) -> dict[str, object]:
    stimulus = go_encoding_to_stimulus(encoding)
    stimulus["rates_hz"] = [rate] * 32
    stimulus.pop("stimulus_hash")
    stimulus["stimulus_hash"] = _json_hash(stimulus)
    return simulate_frame(graph(), stimulus, parameters=parameters(), seed=4)


def training_info() -> dict[str, object]:
    return {
        "method": "synthetic-test-fixture",
        "dataset_id": "unit-test-only",
        "sample_count": 2,
        "completed_at": "2026-09-21T00:00:00Z",
    }


class MaleCNSPolicyTests(unittest.TestCase):
    def write_checkpoint(
        self,
        root: Path,
        frame: dict[str, object],
        weights: np.ndarray,
        bias: np.ndarray,
        *,
        status: str = "trained",
        info: dict[str, object] | None = None,
    ):
        resolved_info = (
            info
            if info is not None
            else training_info() if status == "trained" else {"reason": "fixture only"}
        )
        write_policy_checkpoint(
            root,
            weights=weights,
            bias=bias,
            model_id=str(frame["model_id"]),
            graph_manifest_sha256=str(frame["manifest_sha256"]),
            output_pool_group_hash=str(frame["output_pool"]["group_hash"]),
            training_status=status,
            training_info=resolved_info,
        )
        return load_policy_checkpoint(root)

    def test_checkpoint_round_trip_and_deterministic_legal_argmax(self) -> None:
        encoding = encode_go_state(initial_state())
        frame = controlled_frame(150, encoding)
        weights = np.zeros((82, 128), dtype=np.float32)
        bias = np.zeros(82, dtype=np.float32)
        bias[7] = 3.0
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = self.write_checkpoint(Path(temporary), frame, weights, bias)
            first = select_action(frame, encoding, checkpoint)
            second = select_action(frame, encoding, checkpoint)
        self.assertEqual(first, second)
        self.assertEqual(first["action_index"], 7)
        self.assertEqual(first["move"], [0, 7])
        self.assertFalse(first["fallback_used"])
        self.assertFalse(first["baseline_controller_called"])
        self.assertFalse(first["teacher_accessed_at_runtime"])
        json.dumps(first, allow_nan=False)

    def test_illegal_highest_logit_is_masked_and_pass_is_supported(self) -> None:
        state = step(initial_state(), (0, 0))
        encoding = encode_go_state(state)
        frame = controlled_frame(150, encoding)
        weights = np.zeros((82, 128), dtype=np.float32)
        bias = np.zeros(82, dtype=np.float32)
        bias[0] = 100.0  # occupied and illegal
        bias[1] = 10.0
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = self.write_checkpoint(Path(temporary), frame, weights, bias)
            decision = select_action(frame, encoding, checkpoint)
        self.assertEqual(decision["action_index"], 1)

        initial_encoding = encode_go_state(initial_state())
        initial_frame = controlled_frame(150, initial_encoding)
        bias[:] = 0.0
        bias[81] = 20.0
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = self.write_checkpoint(Path(temporary), initial_frame, weights, bias)
            decision = select_action(initial_frame, initial_encoding, checkpoint)
        self.assertEqual(decision["action_index"], 81)
        self.assertEqual(decision["move"], "pass")
        self.assertTrue(decision["pass_selected"])

    def test_rule_pass_control_is_explicit_and_preserves_point_readout(self) -> None:
        info = {
            **training_info(),
            "pass_control": "rule-after-opponent-pass",
            "controller_sources": {
                "point_actions": "malecns-linear-readout",
                "pass_after_opponent_pass": "go-rule-pass-gate",
                "pass_when_only_legal": "go-rule-pass-gate",
            },
            "fit_sample_count": 2,
            "decision_architecture": "hybrid-rule-after-opponent-pass-v1",
            "neural_action_space": "point-actions-0..80",
            "pass_selection_source": "deterministic-go-rule",
        }
        weights = np.zeros((82, 128), dtype=np.float32)
        bias = np.zeros(82, dtype=np.float32)
        bias[PASS_ACTION_INDEX] = 100.0
        bias[7] = 10.0

        initial_encoding = encode_go_state(initial_state())
        initial_frame = controlled_frame(150, initial_encoding)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = self.write_checkpoint(
                Path(temporary), initial_frame, weights, bias, info=info
            )
            point_decision = select_action(initial_frame, initial_encoding, checkpoint)
        self.assertEqual(point_decision["action_index"], 7)
        self.assertEqual(point_decision["controller_source"], "malecns-linear-readout")
        self.assertEqual(point_decision["controller_reason"], "highest-legal-point-logit")
        self.assertTrue(point_decision["neural_readout_used"])

        after_pass = step(initial_state(), None)
        pass_encoding = encode_go_state(after_pass)
        pass_frame = controlled_frame(150, pass_encoding)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = self.write_checkpoint(
                Path(temporary), pass_frame, weights, bias, info=info
            )
            pass_decision = select_action(pass_frame, pass_encoding, checkpoint)
        self.assertEqual(pass_decision["action_index"], PASS_ACTION_INDEX)
        self.assertEqual(pass_decision["controller_source"], "go-rule-pass-gate")
        self.assertEqual(
            pass_decision["controller_reason"], "opponent-last-move-was-pass"
        )
        self.assertFalse(pass_decision["neural_readout_used"])
        self.assertIsNone(pass_decision["selected_logit"])
        self.assertIsNone(pass_decision["logits"])

    def test_finished_game_fails_closed_instead_of_empty_argmax(self) -> None:
        state = step(step(initial_state(), None), None)
        encoding = encode_go_state(state)
        frame = controlled_frame(150, encoding)
        weights = np.zeros((82, 128), dtype=np.float32)
        bias = np.zeros(82, dtype=np.float32)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = self.write_checkpoint(Path(temporary), frame, weights, bias)
            with self.assertRaisesRegex(PolicyInputError, "finished game"):
                select_action(frame, encoding, checkpoint)

    def test_different_real_frames_change_features_and_logits(self) -> None:
        encoding = encode_go_state(initial_state())
        silent = controlled_frame(0, encoding)
        active = controlled_frame(150, encoding)
        weights = np.zeros((82, 128), dtype=np.float32)
        weights[0, :] = 1.0
        bias = np.zeros(82, dtype=np.float32)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = self.write_checkpoint(Path(temporary), active, weights, bias)
            silent_result = select_action(silent, encoding, checkpoint)
            active_result = select_action(active, encoding, checkpoint)
        self.assertNotEqual(silent["output_pool"]["features_hash"], active["output_pool"]["features_hash"])
        self.assertNotEqual(silent_result["logits"], active_result["logits"])

    def test_untrained_checkpoint_is_explicitly_rejected(self) -> None:
        encoding = encode_go_state(initial_state())
        frame = controlled_frame(150, encoding)
        weights, bias = initialise_linear_parameters(seed=9)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = self.write_checkpoint(
                Path(temporary), frame, weights, bias, status="untrained"
            )
            with self.assertRaisesRegex(CheckpointValidationError, "untrained"):
                select_action(frame, encoding, checkpoint)

    def test_missing_or_tampered_checkpoint_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(CheckpointValidationError):
                load_policy_checkpoint(temporary)

        encoding = encode_go_state(initial_state())
        frame = controlled_frame(150, encoding)
        weights, bias = initialise_linear_parameters(seed=1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_checkpoint(root, frame, weights, bias)
            with (root / "weights.npy").open("r+b") as stream:
                stream.seek(-1, 2)
                byte = stream.read(1)
                stream.seek(-1, 2)
                stream.write(bytes([byte[0] ^ 1]))
            with self.assertRaisesRegex(CheckpointValidationError, "SHA-256"):
                load_policy_checkpoint(root)

    def test_tampered_frame_encoding_or_model_binding_is_rejected(self) -> None:
        encoding = encode_go_state(initial_state())
        frame = controlled_frame(150, encoding)
        weights = np.zeros((82, 128), dtype=np.float32)
        bias = np.zeros(82, dtype=np.float32)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = self.write_checkpoint(Path(temporary), frame, weights, bias)
            bad_frame = copy.deepcopy(frame)
            bad_frame["output_pool"]["features"][0] = 0.123
            with self.assertRaisesRegex(PolicyInputError, "frame_id"):
                select_action(bad_frame, encoding, checkpoint)

            bad_encoding = copy.deepcopy(encoding)
            bad_encoding["legal_mask"][0] = 0
            with self.assertRaisesRegex(PolicyInputError, "encoding_hash"):
                select_action(frame, bad_encoding, checkpoint)

            wrong_model = copy.deepcopy(frame)
            wrong_model["model_id"] = "another-model"
            core = dict(wrong_model)
            core.pop("frame_id")
            core.pop("wall_time_ms")
            wrong_model["frame_id"] = _json_hash(core)
            with self.assertRaisesRegex(PolicyInputError, "model_id"):
                select_action(wrong_model, encoding, checkpoint)


if __name__ == "__main__":
    unittest.main()
