"""Synthetic-fixture tests for the Phase 3 MaleCNS reference dynamics."""

from __future__ import annotations

import json
import threading
import unittest

import numpy as np

from src.go_engine import initial_state, step
from src.go_neural_encoding import encode_go_state
from src.malecns_dynamics import (
    DynamicsCancelled,
    DynamicsError,
    EventLimitExceeded,
    GraphArrays,
    LIFParameters,
    SPATIAL_STIMULUS_ADAPTER_VERSION,
    go_encoding_to_spatial_stimulus,
    go_encoding_to_stimulus,
    select_downstream_output_pools,
    select_input_groups,
    simulate_frame,
)


def graph_with_sign(sign: int, *, fanout: int = 1) -> GraphArrays:
    node_count = fanout + 1
    return GraphArrays.from_arrays(
        model_id=f"synthetic-sign-{sign}-fanout-{fanout}",
        node_ids=np.arange(100, 100 + node_count, dtype=np.int64),
        soma_xyz=np.arange(node_count * 3, dtype=np.float32).reshape(node_count, 3),
        indptr=np.array([0, fanout] + [fanout] * fanout, dtype=np.uint64),
        indices=np.arange(1, node_count, dtype=np.uint32),
        weights=np.full(fanout, 30.0, dtype=np.float32),
        nt_code=np.array([sign] + [1] * fanout, dtype=np.int8),
    )


def stimulus(rate: float) -> dict[str, object]:
    encoded = encode_go_state(initial_state())
    result = go_encoding_to_stimulus(encoded)
    # Replace rates for controlled dynamics fixtures and re-hash the documented payload.
    result["rates_hz"] = [rate] * 32
    result.pop("stimulus_hash")
    import hashlib

    raw = json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    result["stimulus_hash"] = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return result


def fast_parameters(**overrides: object) -> LIFParameters:
    values: dict[str, object] = {
        "dt_ms": 1.0,
        "duration_ms": 15.0,
        "synaptic_delay_ms": 2.0,
        "refractory_ms": 2.0,
        "mv_per_contact": 0.275,
        "stimulus_group_size": 1,
        "display_node_limit": 32,
        "display_edge_limit": 64,
        "max_synaptic_events": 10_000,
        "max_total_spikes": 10_000,
        "max_spikes_per_tick": 1_000,
    }
    values.update(overrides)
    return LIFParameters(**values)


class MaleCNSDynamicsTests(unittest.TestCase):
    def test_415_to_32_adapter_is_deterministic_causal_and_json_safe(self) -> None:
        encoded = encode_go_state(initial_state())
        first = go_encoding_to_stimulus(encoded)
        second = go_encoding_to_stimulus(encoded)
        self.assertEqual(first, second)
        self.assertEqual(len(first["rates_hz"]), 32)
        self.assertTrue(all(0 <= value <= 150 for value in first["rates_hz"]))
        self.assertFalse(first["teacher_or_baseline_action_used"])
        json.dumps(first)

    def test_different_public_positions_change_stimulus(self) -> None:
        first = go_encoding_to_stimulus(encode_go_state(initial_state()))
        second = go_encoding_to_stimulus(encode_go_state(step(initial_state(), (4, 4))))
        self.assertNotEqual(first["encoding_hash"], second["encoding_hash"])
        self.assertNotEqual(first["rates_hz"], second["rates_hz"])
        self.assertNotEqual(first["stimulus_hash"], second["stimulus_hash"])

    def test_spatial_stimulus_is_versioned_and_does_not_change_default(self) -> None:
        opening = encode_go_state(initial_state())
        played = encode_go_state(step(initial_state(), (4, 4)))
        original = go_encoding_to_stimulus(opening)
        spatial_opening = go_encoding_to_spatial_stimulus(opening)
        spatial_played = go_encoding_to_spatial_stimulus(played)
        self.assertEqual(original, go_encoding_to_stimulus(opening))
        self.assertEqual(spatial_opening["version"], SPATIAL_STIMULUS_ADAPTER_VERSION)
        self.assertEqual(spatial_opening["rates_hz"], [30.0] * 32)
        self.assertNotEqual(spatial_played["rates_hz"], spatial_opening["rates_hz"])
        self.assertEqual(spatial_played["rates_hz"][26], 60.0)
        self.assertEqual(
            simulate_frame(graph_with_sign(1), spatial_played, parameters=fast_parameters())["stimulus_hash"],
            spatial_played["stimulus_hash"],
        )

    def test_tampered_encoding_is_rejected(self) -> None:
        encoded = encode_go_state(initial_state())
        encoded["vector"][0] = 0.25
        with self.assertRaisesRegex(DynamicsError, "encoding_hash"):
            go_encoding_to_stimulus(encoded)

    def test_input_groups_are_deterministic_and_expose_actual_body_ids(self) -> None:
        graph = graph_with_sign(1, fanout=3)
        first = select_input_groups(graph, group_size=2)
        self.assertEqual(first, select_input_groups(graph, group_size=2))
        self.assertEqual(len(first), 32)
        self.assertTrue(all(len(group) == 2 for group in first))
        self.assertTrue(all(index == 0 for group in first for index in group))

    def test_downstream_output_pools_are_deterministic_and_versioned(self) -> None:
        graph = graph_with_sign(1, fanout=3)
        first = select_downstream_output_pools(
            graph, input_group_size=1, output_group_size=1
        )
        self.assertEqual(len(first), 128)
        self.assertEqual(first, select_downstream_output_pools(
            graph, input_group_size=1, output_group_size=1
        ))
        experimental = simulate_frame(
            graph, stimulus(150), parameters=fast_parameters(),
            output_pool_selection="input-downstream-v1",
        )
        ordinary = simulate_frame(graph, stimulus(150), parameters=fast_parameters())
        self.assertNotEqual(
            experimental["output_pool"]["version"], ordinary["output_pool"]["version"]
        )

    def test_positive_sign_propagates_and_frame_metadata_is_consistent(self) -> None:
        graph = graph_with_sign(1)
        frame = simulate_frame(graph, stimulus(150.0), parameters=fast_parameters(), seed=7)
        by_body = {entry["body_id"]: entry for entry in frame["display_nodes"]}
        self.assertGreater(by_body[101]["spike_count"], 0)
        self.assertGreater(frame["synaptic_event_count"], 0)
        self.assertEqual(frame["display_edge_count"], 1)
        self.assertEqual(frame["display_edges"][0]["source_body_id"], 100)
        self.assertEqual(frame["display_edges"][0]["target_body_id"], 101)
        self.assertEqual(frame["display_edges"][0]["weight"], 30.0)
        self.assertEqual(frame["display_edges"][0]["sign"], 1)
        self.assertEqual(frame["activity_source"], "same_controller_frame")
        self.assertEqual(frame["model_id"], graph.model_id)
        self.assertFalse(frame["go_move_selected"])
        self.assertFalse(frame["baseline_controller_called"])
        self.assertEqual(
            frame["input_groups"]["body_ids_by_channel"], [[100]] * 32
        )
        self.assertTrue(all(event["body_id"] in (100, 101) for event in frame["recorded_spikes"]))
        json.dumps(frame)

    def test_negative_sign_suppresses_downstream_spiking(self) -> None:
        positive = simulate_frame(
            graph_with_sign(1), stimulus(150.0), parameters=fast_parameters(), seed=3
        )
        negative = simulate_frame(
            graph_with_sign(-1), stimulus(150.0), parameters=fast_parameters(), seed=3
        )
        pos = {entry["body_id"]: entry for entry in positive["display_nodes"]}
        neg = {entry["body_id"]: entry for entry in negative["display_nodes"]}
        self.assertGreater(pos[101]["spike_count"], 0)
        self.assertEqual(neg[101]["spike_count"], 0)
        self.assertLess(neg[101]["min_potential_mv"], -52.0)
        self.assertEqual(neg[101]["max_activity"], 0.0)
        self.assertEqual(negative["display_edges"][0]["sign"], -1)

    def test_display_edges_are_exact_csr_edges_with_displayed_endpoints(self) -> None:
        graph = graph_with_sign(-1, fanout=3)
        frame = simulate_frame(graph, stimulus(150), parameters=fast_parameters(), seed=4)
        displayed = {node["model_index"] for node in frame["display_nodes"]}
        self.assertGreater(frame["display_edge_count"], 0)
        for edge in frame["display_edges"]:
            source = edge["source_model_index"]
            target = edge["target_model_index"]
            csr_index = edge["csr_edge_index"]
            self.assertIn(source, displayed)
            self.assertIn(target, displayed)
            self.assertGreaterEqual(csr_index, int(graph.indptr[source]))
            self.assertLess(csr_index, int(graph.indptr[source + 1]))
            self.assertEqual(target, int(graph.indices[csr_index]))
            self.assertEqual(edge["weight"], float(graph.weights[csr_index]))
            self.assertEqual(edge["sign"], int(graph.nt_code[source]))

    def test_zero_input_is_silent(self) -> None:
        frame = simulate_frame(
            graph_with_sign(1), stimulus(0.0), parameters=fast_parameters(), seed=0
        )
        self.assertTrue(frame["silenced"])
        self.assertEqual(frame["total_spike_count"], 0)
        self.assertEqual(frame["synaptic_event_count"], 0)
        self.assertTrue(all(node["spike_count"] == 0 for node in frame["display_nodes"]))
        self.assertTrue(all(node["max_activity"] == 0 for node in frame["display_nodes"]))
        self.assertGreater(frame["display_edge_count"], 0)

    def test_repeat_is_deterministic_except_measured_wall_time(self) -> None:
        graph = graph_with_sign(1)
        first = simulate_frame(graph, stimulus(150), parameters=fast_parameters(), seed=99)
        second = simulate_frame(graph, stimulus(150), parameters=fast_parameters(), seed=99)
        self.assertEqual(first["frame_id"], second["frame_id"])
        first.pop("wall_time_ms")
        second.pop("wall_time_ms")
        self.assertEqual(first, second)

    def test_output_pool_is_full_frame_deterministic_bounded_and_json_safe(self) -> None:
        graph = graph_with_sign(1, fanout=3)
        small_display = simulate_frame(
            graph,
            stimulus(150),
            parameters=fast_parameters(display_node_limit=1, display_edge_limit=1),
            seed=13,
        )
        large_display = simulate_frame(
            graph,
            stimulus(150),
            parameters=fast_parameters(display_node_limit=4, display_edge_limit=8),
            seed=13,
        )
        output = small_display["output_pool"]
        self.assertEqual(output["feature_count"], 128)
        self.assertEqual(len(output["features"]), 128)
        self.assertEqual(len(output["body_ids_by_pool"]), 128)
        self.assertTrue(all(len(pool) == 32 for pool in output["body_ids_by_pool"]))
        self.assertTrue(all(0.0 <= value <= 1.0 for value in output["features"]))
        self.assertEqual(output["activity_source"], "same_simulate_frame_full_arrays")
        self.assertEqual(output["features"], large_display["output_pool"]["features"])
        self.assertEqual(output["group_hash"], large_display["output_pool"]["group_hash"])
        self.assertEqual(output["features_hash"], large_display["output_pool"]["features_hash"])
        self.assertNotEqual(small_display["frame_id"], large_display["frame_id"])
        json.dumps(output, allow_nan=False)

    def test_output_pool_changes_with_actual_neural_activity(self) -> None:
        graph = graph_with_sign(1)
        silent = simulate_frame(graph, stimulus(0), parameters=fast_parameters(), seed=2)
        active = simulate_frame(graph, stimulus(150), parameters=fast_parameters(), seed=2)
        self.assertNotEqual(
            silent["output_pool"]["features"], active["output_pool"]["features"]
        )
        self.assertNotEqual(silent["frame_id"], active["frame_id"])

    def test_display_edge_limit_is_deterministic_and_explicit(self) -> None:
        graph = graph_with_sign(1, fanout=8)
        params = fast_parameters(display_edge_limit=2)
        first = simulate_frame(graph, stimulus(150), parameters=params, seed=5)
        second = simulate_frame(graph, stimulus(150), parameters=params, seed=5)
        self.assertEqual(first["frame_id"], second["frame_id"])
        self.assertEqual(first["display_edge_count"], 2)
        self.assertEqual(first["display_edge_limit"], 2)
        self.assertTrue(first["display_edges_truncated"])

    def test_event_cap_fails_closed(self) -> None:
        graph = graph_with_sign(1, fanout=8)
        params = fast_parameters(max_synaptic_events=1)
        with self.assertRaisesRegex(EventLimitExceeded, "max_synaptic_events"):
            simulate_frame(graph, stimulus(150), parameters=params, seed=0)

    def test_cancellation_callable_and_event_are_honoured(self) -> None:
        graph = graph_with_sign(1)
        with self.assertRaises(DynamicsCancelled):
            simulate_frame(
                graph, stimulus(150), parameters=fast_parameters(), cancel=lambda: True
            )
        event = threading.Event()
        event.set()
        with self.assertRaises(DynamicsCancelled):
            simulate_frame(graph, stimulus(150), parameters=fast_parameters(), cancel=event)

    def test_invalid_graph_nt_code_is_rejected(self) -> None:
        with self.assertRaisesRegex(DynamicsError, "unsupported"):
            GraphArrays.from_arrays(
                model_id="bad",
                node_ids=np.array([1], dtype=np.int64),
                soma_xyz=np.zeros((1, 3), dtype=np.float32),
                indptr=np.array([0, 0], dtype=np.uint64),
                indices=np.array([], dtype=np.uint32),
                weights=np.array([], dtype=np.float32),
                nt_code=np.array([2], dtype=np.int8),
            )


if __name__ == "__main__":
    unittest.main()
