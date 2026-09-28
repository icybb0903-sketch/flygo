"""Offline acceptance and rejection tests for P2 toy dynamics."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.p1_graph import GraphDataError, load_verified_graph
from src.toy_dynamics import DynamicsError, ModelParameters, run_experiment_suite, simulate


PROJECT_ROOT = Path(__file__).resolve().parents[1]
P1_SUBGRAPH = PROJECT_ROOT / "data" / "p1" / "subgraph.json"
P1_PROVENANCE = PROJECT_ROOT / "data" / "p1" / "provenance.json"


def load_graph():
    return load_verified_graph(P1_SUBGRAPH, P1_PROVENANCE)


class P1IntegrityTests(unittest.TestCase):
    def _write_case(self, directory: Path, payload: dict, provenance: dict) -> tuple[Path, Path]:
        subgraph = directory / "subgraph.json"
        source = directory / "provenance.json"
        subgraph.write_text(json.dumps(payload), encoding="utf-8")
        source.write_text(json.dumps(provenance), encoding="utf-8")
        return subgraph, source

    def test_p1_hash_tampering_is_rejected(self) -> None:
        payload = json.loads(P1_SUBGRAPH.read_text(encoding="utf-8"))
        provenance = json.loads(P1_PROVENANCE.read_text(encoding="utf-8"))
        payload["data"][0][2] += 1
        with tempfile.TemporaryDirectory() as temp:
            paths = self._write_case(Path(temp), payload, provenance)
            with self.assertRaisesRegex(GraphDataError, "SHA-256"):
                load_verified_graph(*paths)

    def test_negative_weight_is_rejected(self) -> None:
        payload = json.loads(P1_SUBGRAPH.read_text(encoding="utf-8"))
        provenance = json.loads(P1_PROVENANCE.read_text(encoding="utf-8"))
        payload["data"][0][2] = -1
        with tempfile.TemporaryDirectory() as temp:
            paths = self._write_case(Path(temp), payload, provenance)
            with self.assertRaisesRegex(GraphDataError, "field weight"):
                load_verified_graph(*paths)

    def test_float_weight_is_rejected(self) -> None:
        payload = json.loads(P1_SUBGRAPH.read_text(encoding="utf-8"))
        provenance = json.loads(P1_PROVENANCE.read_text(encoding="utf-8"))
        payload["data"][0][2] = 288.0
        with tempfile.TemporaryDirectory() as temp:
            paths = self._write_case(Path(temp), payload, provenance)
            with self.assertRaisesRegex(GraphDataError, "field weight"):
                load_verified_graph(*paths)


class DynamicsAcceptanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = load_graph()
        self.sources = self.graph.nodes_of_type("DNge104")

    def test_suite_is_deterministic(self) -> None:
        self.assertEqual(run_experiment_suite(self.graph), run_experiment_suite(self.graph))

    def test_no_stimulus_has_no_activity(self) -> None:
        result = simulate(self.graph, (), edges_enabled=True)
        self.assertEqual(result["summary"]["total_spike_count"], 0)
        self.assertEqual(result["summary"]["downstream_peak_potential"], 0.0)

    def test_real_edges_produce_finite_downstream_response(self) -> None:
        result = simulate(self.graph, self.sources, edges_enabled=True)
        self.assertEqual(result["summary"]["source_spike_count"], 2)
        self.assertEqual(result["summary"]["downstream_spike_count"], 3)
        self.assertEqual(result["summary"]["downstream_peak_potential"], 1.0)

    def test_disconnected_edges_remove_downstream_response(self) -> None:
        result = simulate(self.graph, self.sources, edges_enabled=False)
        self.assertEqual(result["summary"]["source_spike_count"], 2)
        self.assertEqual(result["summary"]["downstream_spike_count"], 0)
        self.assertEqual(result["summary"]["downstream_peak_potential"], 0.0)

    def test_all_recorded_values_are_bounded(self) -> None:
        params = ModelParameters()
        suite = run_experiment_suite(self.graph, params)
        for experiment in suite["experiments"].values():
            for step in experiment["steps"]:
                for field in ("pre_reset_potential", "state_after_reset"):
                    for value in step[field].values():
                        self.assertGreaterEqual(value, params.state_min)
                        self.assertLessEqual(value, params.state_max)

    def test_unknown_stimulus_node_is_rejected(self) -> None:
        unknown = max(self.graph.nodes) + 1
        with self.assertRaisesRegex(DynamicsError, "unknown node"):
            simulate(self.graph, (unknown,), edges_enabled=True)

    def test_p1_hash_is_carried_into_result(self) -> None:
        suite = run_experiment_suite(self.graph)
        provenance = json.loads(P1_PROVENANCE.read_text(encoding="utf-8"))
        self.assertEqual(suite["p1_canonical_sha256"], provenance["canonical_sha256"])


if __name__ == "__main__":
    unittest.main()
