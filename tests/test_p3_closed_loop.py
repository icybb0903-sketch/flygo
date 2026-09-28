"""Offline tests for the P3 closed-loop behavior adapter."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.closed_loop import (
    CUE_TO_SOURCE,
    SCORE_SOURCE,
    ClosedLoopError,
    choose_from_scores,
    run_closed_loop_suite,
    run_trial,
)
from src.p1_graph import load_verified_graph


PROJECT_ROOT = Path(__file__).resolve().parents[1]
P1_SUBGRAPH = PROJECT_ROOT / "data" / "p1" / "subgraph.json"
P1_PROVENANCE = PROJECT_ROOT / "data" / "p1" / "provenance.json"


def load_graph():
    return load_verified_graph(P1_SUBGRAPH, P1_PROVENANCE)


class ClosedLoopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = load_graph()

    def test_suite_is_deterministic(self) -> None:
        self.assertEqual(
            run_closed_loop_suite(self.graph), run_closed_loop_suite(self.graph)
        )

    def test_left_cue_stimulates_only_left_source_and_chooses_left(self) -> None:
        record = run_trial(self.graph, "left", "real_edges")
        self.assertEqual(record["stimulated_body"], 12781)
        self.assertGreater(record["left_score"], 0.0)
        self.assertEqual(record["right_score"], 0.0)
        self.assertEqual(record["chosen_action"], "left")

    def test_right_cue_stimulates_only_right_source_and_chooses_right(self) -> None:
        record = run_trial(self.graph, "right", "real_edges")
        self.assertEqual(record["stimulated_body"], 556329)
        self.assertEqual(record["left_score"], 0.0)
        self.assertGreater(record["right_score"], 0.0)
        self.assertEqual(record["chosen_action"], "right")

    def test_disconnected_edges_have_zero_scores_and_abstain(self) -> None:
        for cue in CUE_TO_SOURCE:
            with self.subTest(cue=cue):
                record = run_trial(self.graph, cue, "disconnected_edges")
                self.assertEqual(record["left_score"], 0.0)
                self.assertEqual(record["right_score"], 0.0)
                self.assertEqual(record["chosen_action"], "abstain")
                self.assertTrue(record["abstained"])

    def test_unknown_cue_is_rejected(self) -> None:
        with self.assertRaisesRegex(ClosedLoopError, "unknown cue"):
            run_trial(self.graph, "up", "real_edges")

    def test_p1_hash_is_propagated_to_every_trial(self) -> None:
        provenance = json.loads(P1_PROVENANCE.read_text(encoding="utf-8"))
        expected = provenance["canonical_sha256"]
        suite = run_closed_loop_suite(self.graph)
        self.assertEqual(suite["p1_canonical_sha256"], expected)
        for records in suite["conditions"].values():
            for record in records:
                self.assertEqual(record["p1_canonical_sha256"], expected)

    def test_readout_source_and_components_are_explicit(self) -> None:
        record = run_trial(self.graph, "left", "real_edges")
        self.assertEqual(record["score_source"], SCORE_SOURCE)
        self.assertEqual(record["decision_source"], "downstream_score_argmax")
        self.assertEqual(
            len(record["branch_activity"]["left"]),
            len([edge for edge in self.graph.edges if edge.pre == 12781]),
        )

    def test_model_decision_has_no_direct_cue_input(self) -> None:
        # The public decision function cannot see a cue and follows the scores,
        # including a score pattern opposite to any surrounding trial cue.
        action, source = choose_from_scores(0.1, 0.9)
        self.assertEqual(action, "right")
        self.assertEqual(source, "downstream_score_argmax")
        self.assertNotIn("cue", choose_from_scores.__annotations__)

    def test_all_result_scores_and_branch_values_are_bounded(self) -> None:
        suite = run_closed_loop_suite(self.graph)
        for records in suite["conditions"].values():
            for record in records:
                for key in ("left_score", "right_score"):
                    self.assertGreaterEqual(record[key], 0.0)
                    self.assertLessEqual(record[key], 1.0)
                for components in record["branch_activity"].values():
                    for component in components:
                        self.assertGreaterEqual(component["peak_potential"], 0.0)
                        self.assertLessEqual(component["peak_potential"], 1.0)

    def test_all_four_conditions_and_fixed_trial_count_are_recorded(self) -> None:
        suite = run_closed_loop_suite(self.graph)
        self.assertEqual(
            set(suite["conditions"]),
            {
                "real_edges",
                "disconnected_edges",
                "fixed_random_baseline",
                "direct_mapping_upper_bound",
            },
        )
        self.assertEqual(len(suite["cue_sequence"]), 10)
        self.assertTrue(all(len(records) == 10 for records in suite["conditions"].values()))


if __name__ == "__main__":
    unittest.main()
