"""P4 local web-server security and integration tests (standard library only)."""

from __future__ import annotations

import http.client
import hashlib
import io
import json
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app.server import MAX_REQUEST_BYTES, create_server, serve
from src.agent_monitor import append_event, create_event
from src.go_neural_encoding import encode_go_state
from src.go_session import GoSession
from src.malecns_dynamics import GraphArrays, simulate_go_encoding
from src.malecns_policy import (
    CheckpointValidationError,
    PolicyError,
    initialise_linear_parameters,
    load_policy_checkpoint,
    write_policy_checkpoint,
)


def _synthetic_neural_graph() -> GraphArrays:
    return GraphArrays.from_arrays(
        model_id="synthetic-web-neural-v1",
        manifest_sha256="a" * 64,
        node_ids=np.asarray([101, 202, 303, 404], dtype=np.int64),
        soma_xyz=np.asarray(
            [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]],
            dtype=np.float32,
        ),
        indptr=np.asarray([0, 1, 2, 3, 4], dtype=np.uint64),
        indices=np.asarray([1, 2, 3, 0], dtype=np.uint32),
        weights=np.asarray([1, 1, 1, 1], dtype=np.float32),
        nt_code=np.asarray([1, 1, 1, 1], dtype=np.int8),
    )


class WebServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.events_dir = Path(cls.temp_dir.name) / "events"
        cls.server = create_server(0, events_dir=cls.events_dir)
        cls.host, cls.port = cls.server.server_address
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=3)
        if cls.thread.is_alive():
            raise RuntimeError("P4 test server did not shut down cleanly")
        cls.temp_dir.cleanup()

    def request(
        self,
        method: str,
        path: str,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, str], bytes]:
        connection = http.client.HTTPConnection(self.host, self.port, timeout=3)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    @staticmethod
    def json_body(body: bytes) -> dict:
        return json.loads(body.decode("utf-8"))

    def simulate(self, payload: object) -> tuple[int, dict, dict[str, str]]:
        encoded = json.dumps(payload).encode("utf-8")
        status, headers, body = self.request(
            "POST",
            "/api/simulate",
            encoded,
            {"Content-Type": "application/json", "Content-Length": str(len(encoded))},
        )
        return status, self.json_body(body), headers

    def go_action(self, payload: object) -> tuple[int, dict, dict[str, str]]:
        encoded = json.dumps(payload).encode("utf-8")
        status, headers, body = self.request(
            "POST",
            "/api/go/action",
            encoded,
            {"Content-Type": "application/json", "Content-Length": str(len(encoded))},
        )
        return status, self.json_body(body), headers

    def test_homepage_and_static_assets(self) -> None:
        status, headers, body = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", headers["Content-Type"])
        self.assertIn("真实连接数据 + 工程化玩具动力学".encode(), body)
        self.assertIn("default-src 'none'", headers["Content-Security-Policy"])
        for path, content_type in (("/styles.css", "text/css"), ("/app.js", "text/javascript")):
            with self.subTest(path=path):
                asset_status, asset_headers, asset_body = self.request("GET", path)
                self.assertEqual(asset_status, 200)
                self.assertIn(content_type, asset_headers["Content-Type"])
                self.assertGreater(len(asset_body), 100)

    def test_agent_monitor_page_and_static_assets(self) -> None:
        for path in ("/monitor", "/monitor.html"):
            with self.subTest(path=path):
                status, headers, body = self.request("GET", path)
                self.assertEqual(status, 200)
                self.assertIn("text/html", headers["Content-Type"])
                self.assertIn(b"Agent Monitor", body)
        for path, content_type in (
            ("/monitor.css", "text/css"),
            ("/monitor.js", "text/javascript"),
        ):
            with self.subTest(path=path):
                status, headers, body = self.request("GET", path)
                self.assertEqual(status, 200)
                self.assertIn(content_type, headers["Content-Type"])
                self.assertGreater(len(body), 100)

    def test_go_page_and_local_three_assets(self) -> None:
        for path in ("/go", "/go.html"):
            with self.subTest(path=path):
                status, headers, body = self.request("GET", path)
                self.assertEqual(status, 200)
                self.assertIn("text/html", headers["Content-Type"])
                self.assertIn(b"WebGL", body)
        for path, content_type in (
            ("/go.css", "text/css"),
            ("/go.js", "text/javascript"),
            ("/vendor/three.module.js", "text/javascript"),
            ("/vendor/three.core.js", "text/javascript"),
            ("/vendor/THREE_LICENSE.txt", "text/plain"),
        ):
            with self.subTest(path=path):
                status, headers, body = self.request("GET", path)
                self.assertEqual(status, 200)
                self.assertIn(content_type, headers["Content-Type"])
                self.assertGreater(len(body), 100)

    def test_go_state_and_real_baseline_round_trip(self) -> None:
        status, payload, _ = self.go_action({"action": "reset"})
        self.assertEqual(status, 200)
        self.assertTrue(payload["ok"])
        session = payload["session"]
        self.assertEqual(session["board_size"], 9)
        self.assertEqual(session["to_move"], "black")
        self.assertFalse(session["policy"]["male_cns_used"])
        self.assertEqual(session["controller"], "baseline")
        self.assertEqual(len(session["board_hash"]), 64)

        status, human, _ = self.go_action(
            {"action": "human_move", "row": 4, "col": 4}
        )
        self.assertEqual(status, 200)
        self.assertEqual(human["session"]["board"][4][4], 1)
        self.assertEqual(human["session"]["to_move"], "white")

        status, bot, _ = self.go_action({"action": "bot_move"})
        self.assertEqual(status, 200)
        self.assertEqual(bot["session"]["move_number"], 2)
        self.assertEqual(bot["session"]["to_move"], "black")
        self.assertEqual(bot["session"]["last_move"]["actor"], "bot")

        status, undone, _ = self.go_action({"action": "undo"})
        self.assertEqual(status, 200)
        self.assertEqual(undone["session"]["move_number"], 0)
        self.assertFalse(undone["session"]["can_undo"])

        get_status, _, get_body = self.request("GET", "/api/go/state")
        self.assertEqual(get_status, 200)
        self.assertEqual(self.json_body(get_body)["session"]["move_number"], 0)

    def test_go_action_validation_and_illegal_move_are_bounded(self) -> None:
        self.go_action({"action": "reset"})
        for bad_payload, expected_code in (
            ({"action": "human_move", "row": True, "col": 0}, "invalid_coordinate"),
            ({"action": "human_move", "row": 9, "col": 0}, "invalid_coordinate"),
            ({"action": "reset", "extra": 1}, "invalid_go_fields"),
            ({"action": "invented"}, "invalid_go_action"),
        ):
            with self.subTest(payload=bad_payload):
                status, payload, _ = self.go_action(bad_payload)
                self.assertEqual(status, 400)
                self.assertEqual(payload["error"]["code"], expected_code)

        self.go_action({"action": "human_move", "row": 0, "col": 0})
        self.go_action({"action": "bot_move"})
        status, occupied, _ = self.go_action(
            {"action": "human_move", "row": 0, "col": 0}
        )
        self.assertEqual(status, 409)
        self.assertEqual(occupied["error"]["code"], "illegal_move_occupied")

    def test_two_passes_automatically_finish_and_score_game(self) -> None:
        self.go_action({"action": "reset"})
        initial_status, _, initial_body = self.request("GET", "/api/go/state")
        self.assertEqual(initial_status, 200)
        initial = self.json_body(initial_body)["session"]
        self.assertFalse(initial["game_over"])
        self.assertFalse(initial["result"]["final"])
        self.assertIsNone(initial["winner"])
        self.assertIsNone(initial["result"]["margin"])
        self.assertFalse(initial["score"]["final"])

        status, human, _ = self.go_action({"action": "human_pass"})
        self.assertEqual(status, 200)
        self.assertFalse(human["session"]["game_over"])
        self.assertEqual(human["session"]["passes"]["consecutive"], 1)
        self.assertIsNone(human["session"]["winner"])

        status, bot, _ = self.go_action({"action": "bot_move"})
        self.assertEqual(status, 200)
        final = bot["session"]
        self.assertTrue(final["game_over"])
        self.assertTrue(final["score"]["final"])
        self.assertTrue(final["result"]["final"])
        self.assertEqual(final["end_reason"], "two_consecutive_passes")
        self.assertEqual(final["winner"], "white")
        self.assertEqual(final["score"]["white_total"], 7.5)
        self.assertEqual(final["result"]["margin"], 7.5)

        status, _, final_body = self.request("GET", "/api/go/state")
        self.assertEqual(status, 200)
        self.assertEqual(self.json_body(final_body)["session"]["result"], final["result"])

        status, undone, _ = self.go_action({"action": "undo"})
        self.assertEqual(status, 200)
        self.assertFalse(undone["session"]["game_over"])
        self.assertFalse(undone["session"]["result"]["final"])
        self.assertIsNone(undone["session"]["winner"])

    def test_human_resignation_reports_white_winner_and_can_be_undone(self) -> None:
        self.go_action({"action": "reset"})
        status, resigned, _ = self.go_action({"action": "human_resign"})
        self.assertEqual(status, 200)
        final = resigned["session"]
        self.assertTrue(final["game_over"])
        self.assertEqual(final["end_reason"], "resignation")
        self.assertEqual(final["winner"], "white")
        self.assertEqual(final["result"]["winner"], "white")
        self.assertFalse(final["score"]["final"])
        self.assertTrue(final["can_undo"])

        status, undone, _ = self.go_action({"action": "undo"})
        self.assertEqual(status, 200)
        self.assertFalse(undone["session"]["game_over"])
        self.assertIsNone(undone["session"]["end_reason"])

    def test_agents_api_uses_only_validated_event_log(self) -> None:
        status, _, body = self.request("GET", "/api/agents")
        empty = self.json_body(body)
        self.assertEqual(status, 200)
        self.assertTrue(empty["ok"])
        self.assertEqual(empty["agents"], [])

        self.assertIn("validated agent event files", empty["source_note"])
        self.assertIn("没有向本项目提供直接读取", empty["source_note"])

        append_event(
            self.events_dir,
            create_event(
                "agent_created",
                agent_id="/root/test-agent",
                agent_name="Real Test Agent",
                parent_id="/root",
                status="working",
                source="unit-test",
                timestamp_utc="2026-09-15T01:00:00Z",
            ),
        )
        append_event(
            self.events_dir,
            create_event(
                "test_finished",
                agent_id="/root/test-agent",
                status="completed",
                source="unit-test",
                timestamp_utc="2026-09-15T01:00:01Z",
                test={
                    "status": "passed",
                    "summary": "1 test passed",
                    "command": "python -m unittest tests.test_agent_monitor",
                },
            ),
        )
        status, _, body = self.request("GET", "/api/agents")
        payload = self.json_body(body)
        self.assertEqual(status, 200)
        self.assertEqual(len(payload["agents"]), 1)
        agent = payload["agents"][0]
        self.assertEqual(agent["name"], "Real Test Agent")
        self.assertEqual(agent["status"], "completed")
        self.assertEqual(agent["last_test"]["status"], "passed")
        self.assertEqual(len(agent["timeline"]), 2)

    def test_evaluation_status_serves_only_hash_validated_evidence(self) -> None:
        report_path = Path(self.temp_dir.name) / "evaluation.json"
        report = {
            "schema": "malecns-go-stage5-evaluation-v1",
            "scope": "test",
            "status": "passed",
            "checkpoint": {"hash": self.server.policy_checkpoint.checkpoint_hash},
            "evaluation": {"sample_count": 12},
            "metrics": {
                "teacher_agreement_rate": 0.5,
                "random_legal_expected_rate": 0.1,
                "poisson_binomial_p_value": 0.001,
            },
        }
        encoded = json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        report["report_sha256"] = hashlib.sha256(encoded).hexdigest()
        report_path.write_text(json.dumps(report), encoding="utf-8")
        with patch("app.server.STAGE5_EVALUATION_REPORT", report_path):
            status, _, body = self.request("GET", "/api/go/evaluation-status")
        payload = self.json_body(body)
        self.assertEqual(status, 200)
        self.assertTrue(payload["available"])
        self.assertTrue(payload["matches_loaded_checkpoint"])

        report["metrics"]["teacher_agreement_rate"] = 1.0
        report_path.write_text(json.dumps(report), encoding="utf-8")
        with patch("app.server.STAGE5_EVALUATION_REPORT", report_path):
            status, _, body = self.request("GET", "/api/go/evaluation-status")
        self.assertEqual(status, 500)
        self.assertEqual(self.json_body(body)["error"]["code"], "evaluation_invalid")

    def test_tournament_status_preserves_role_specific_result(self) -> None:
        report_path = Path(self.temp_dir.name) / "tournament.json"
        report = {
            "schema": "malecns-go-stage6-tournament-v1",
            "status": "failed",
            "checkpoint": {"hash": self.server.policy_checkpoint.checkpoint_hash},
            "configuration": {"game_count": 10},
            "metrics": {"wins": 6},
            "role_evaluations": {
                "black": {"status": "failed", "wins": 1, "game_count": 5},
                "white": {"status": "passed", "wins": 5, "game_count": 5},
            },
        }
        encoded = json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        report["report_sha256"] = hashlib.sha256(encoded).hexdigest()
        report_path.write_text(json.dumps(report), encoding="utf-8")
        with patch("app.server.STAGE6_TOURNAMENT_REPORT", report_path):
            status, _, body = self.request("GET", "/api/go/tournament-status")
        payload = self.json_body(body)
        self.assertEqual(status, 200)
        self.assertTrue(payload["matches_loaded_checkpoint"])
        self.assertEqual(payload["tournament"]["status"], "failed")
        self.assertEqual(
            payload["tournament"]["role_evaluations"]["white"]["status"],
            "passed",
        )

    def test_stage7_status_is_explicitly_development_only(self) -> None:
        report_path = Path(self.temp_dir.name) / "stage7-development.json"
        report = {
            "schema": "malecns-go-stage7-production-white-v1",
            "status": "failed",
            "checkpoint": {"hash": self.server.policy_checkpoint.checkpoint_hash},
            "metrics": {
                "wins": 14,
                "losses": 6,
                "draws": 0,
                "win_rate": 0.7,
                "natural_finish_count": 0,
            },
        }
        encoded = json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        report["report_sha256"] = hashlib.sha256(encoded).hexdigest()
        report_path.write_text(json.dumps(report), encoding="utf-8")
        with patch("app.server.STAGE7_DEVELOPMENT_REPORT", report_path):
            status, _, body = self.request("GET", "/api/go/stage7-status")
        payload = self.json_body(body)
        self.assertEqual(status, 200)
        self.assertTrue(payload["matches_loaded_checkpoint"])
        self.assertEqual(payload["classification"], "development_only_not_final")
        self.assertEqual(payload["development"]["metrics"]["wins"], 14)

    def test_status_has_verified_real_topology_and_disclaimer(self) -> None:
        status, _, body = self.request("GET", "/api/status")
        payload = self.json_body(body)
        self.assertEqual(status, 200)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["topology"]["node_count"], 12)
        self.assertEqual(payload["topology"]["edge_count"], 10)
        self.assertEqual(len(payload["topology"]["nodes"]), 12)
        self.assertEqual(len(payload["topology"]["edges"]), 10)
        self.assertEqual(payload["scenario_source_nodes"]["left"], [12781])
        self.assertEqual(payload["scenario_source_nodes"]["right"], [556329])
        self.assertIn("not measured neural activity", payload["model_disclaimer"])
        self.assertEqual(len(payload["p1_canonical_sha256"]), 64)

    def test_default_neural_status_uses_strict_real_model(self) -> None:
        status, _, body = self.request("GET", "/api/neural/status")
        payload = self.json_body(body)
        self.assertEqual(status, 200)
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["available"])
        self.assertEqual(
            payload["model_id"], "malecns-v1.0-w5-3acb6434e71160fc"
        )
        self.assertEqual(payload["controller_node_count"], 139662)
        self.assertEqual(payload["controller_edge_count"], 5536347)
        self.assertEqual(len(payload["manifest_sha256"]), 64)
        self.assertIn("not measured neural activity", payload["disclaimer"])
        self.assertEqual(
            payload["stage3_role"], "observational_frame_only_not_go_controller"
        )
        self.assertFalse(payload["go_move_selected"])

    def test_verified_malecns_soma_coordinates_are_served_for_brain_view(self) -> None:
        status, headers, body = self.request(
            "GET", "/api/neural/anatomy/soma_xyz.npy"
        )
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "application/octet-stream")
        self.assertEqual(
            headers["X-MaleCNS-Model-ID"],
            "malecns-v1.0-w5-3acb6434e71160fc",
        )
        self.assertEqual(headers["X-MaleCNS-Node-Count"], "139662")
        self.assertTrue(body.startswith(b"\x93NUMPY"))
        self.assertEqual(len(body), 1_676_072)

    def test_all_simulation_scenarios_and_disconnected_control(self) -> None:
        expected_nodes = {
            "none": [], "left": [12781], "right": [556329], "both": [12781, 556329]
        }
        for scenario, nodes in expected_nodes.items():
            with self.subTest(scenario=scenario):
                status, payload, _ = self.simulate(
                    {"scenario": scenario, "edges_enabled": True}
                )
                self.assertEqual(status, 200)
                self.assertEqual(payload["simulation"]["stimulus"]["nodes"], nodes)
                self.assertIn("engineering labels", payload["engineering_label_notice"])
                self.assertEqual(len(payload["p1_canonical_sha256"]), 64)
                self.assertIn("not measured neural activity", payload["model_disclaimer"])

        status, payload, _ = self.simulate({"scenario": "both", "edges_enabled": False})
        self.assertEqual(status, 200)
        self.assertFalse(payload["simulation"]["edges_enabled"])
        self.assertEqual(payload["simulation"]["summary"]["downstream_spike_count"], 0)
        self.assertEqual(payload["simulation"]["summary"]["downstream_peak_potential"], 0.0)

    def test_bad_json_is_rejected(self) -> None:
        status, _, body = self.request(
            "POST",
            "/api/simulate",
            b"{not-json",
            {"Content-Type": "application/json", "Content-Length": "9"},
        )
        payload = self.json_body(body)
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["code"], "invalid_json")

    def test_oversized_request_is_rejected_without_processing(self) -> None:
        body = b"x" * (MAX_REQUEST_BYTES + 1)
        status, headers, response_body = self.request(
            "POST",
            "/api/simulate",
            body,
            {"Content-Type": "application/json", "Content-Length": str(len(body))},
        )
        payload = self.json_body(response_body)
        self.assertEqual(status, 413)
        self.assertEqual(headers.get("Connection"), "close")
        self.assertEqual(payload["error"]["code"], "request_too_large")

    def test_content_type_and_exact_fields_are_enforced(self) -> None:
        body = b'{"scenario":"both","edges_enabled":true}'
        status, _, response_body = self.request(
            "POST",
            "/api/simulate",
            body,
            {"Content-Type": "text/plain", "Content-Length": str(len(body))},
        )
        self.assertEqual(status, 415)
        self.assertEqual(self.json_body(response_body)["error"]["code"], "invalid_content_type")

        for payload in (
            {"scenario": "both"},
            {"scenario": "both", "edges_enabled": True, "extra": 1},
        ):
            with self.subTest(payload=payload):
                status, response, _ = self.simulate(payload)
                self.assertEqual(status, 400)
                self.assertEqual(response["error"]["code"], "invalid_fields")

    def test_field_types_and_values_are_enforced(self) -> None:
        cases = (
            ({"scenario": "unknown", "edges_enabled": True}, "invalid_scenario"),
            ({"scenario": 1, "edges_enabled": True}, "invalid_scenario"),
            ({"scenario": "both", "edges_enabled": 1}, "invalid_edges_enabled"),
            ({"scenario": "both", "edges_enabled": "true"}, "invalid_edges_enabled"),
        )
        for payload, code in cases:
            with self.subTest(payload=payload):
                status, response, _ = self.simulate(payload)
                self.assertEqual(status, 400)
                self.assertEqual(response["error"]["code"], code)

    def test_path_traversal_and_unknown_resources_are_rejected(self) -> None:
        for path in ("/../src/p1_graph.py", "/%2e%2e/src/p1_graph.py", "/app/../../data/p1/subgraph.json"):
            with self.subTest(path=path):
                status, _, body = self.request("GET", path)
                self.assertEqual(status, 400)
                self.assertEqual(self.json_body(body)["error"]["code"], "invalid_path")
        status, _, _ = self.request("GET", "/secret.txt")
        self.assertEqual(status, 404)

    def test_wrong_methods_are_rejected(self) -> None:
        for method, path in (("POST", "/"), ("PUT", "/api/simulate"), ("DELETE", "/api/status"), ("HEAD", "/")):
            with self.subTest(method=method, path=path):
                status, headers, body = self.request(method, path)
                self.assertEqual(status, 405 if method != "POST" else 404)
                if method == "HEAD":
                    self.assertEqual(body, b"")
                    self.assertEqual(headers["Allow"], "GET, POST")
                else:
                    self.assertEqual(self.json_body(body)["ok"], False)

    def test_serve_ctrl_c_closes_without_self_shutdown_deadlock(self) -> None:
        class InterruptingServer:
            server_address = ("127.0.0.1", 8000)

            def __init__(self) -> None:
                self.closed = False

            def serve_forever(self) -> None:
                raise KeyboardInterrupt

            def server_close(self) -> None:
                self.closed = True

        fake_server = InterruptingServer()
        with patch("app.server.create_server", return_value=fake_server):
            with redirect_stdout(io.StringIO()):
                serve(8000)
        self.assertTrue(fake_server.closed)


class Phase3NeuralAPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.graph = _synthetic_neural_graph()
        cls.server = create_server(
            0,
            events_dir=Path(cls.temp_dir.name) / "events",
            neural_graph=cls.graph,
        )
        cls.host, cls.port = cls.server.server_address
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=3)
        if cls.thread.is_alive():
            raise RuntimeError("Phase 3 test server did not shut down cleanly")
        cls.temp_dir.cleanup()

    @staticmethod
    def _request_to(
        host: str,
        port: int,
        method: str,
        path: str,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, str], bytes]:
        connection = http.client.HTTPConnection(host, port, timeout=3)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def request(
        self,
        method: str,
        path: str,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, str], bytes]:
        return self._request_to(self.host, self.port, method, path, body, headers)

    def frame(self, payload: object) -> tuple[int, dict]:
        body = json.dumps(payload).encode("utf-8")
        status, _, response = self.request(
            "POST",
            "/api/neural/frame",
            body,
            {"Content-Type": "application/json", "Content-Length": str(len(body))},
        )
        return status, json.loads(response.decode("utf-8"))

    def test_injected_status_and_frame_provenance_match_current_go_state(self) -> None:
        status, _, body = self.request("GET", "/api/neural/status")
        status_payload = json.loads(body.decode("utf-8"))
        self.assertEqual(status, 200)
        self.assertTrue(status_payload["available"])
        self.assertEqual(status_payload["model_id"], self.graph.model_id)
        self.assertEqual(status_payload["manifest_sha256"], self.graph.manifest_sha256)
        self.assertEqual(status_payload["controller_node_count"], 4)
        self.assertEqual(status_payload["controller_edge_count"], 4)

        expected_encoding = encode_go_state(self.server.go_session.state)
        status, payload = self.frame({"seed": 7})
        self.assertEqual(status, 200)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["board_hash"], expected_encoding["board_hash"])
        self.assertEqual(payload["encoding_hash"], expected_encoding["encoding_hash"])
        self.assertEqual(payload["board_hash"], payload["frame"]["board_hash"])
        self.assertEqual(payload["encoding_hash"], payload["frame"]["encoding_hash"])
        self.assertEqual(payload["decision_id"], payload["frame"]["decision_id"])
        self.assertEqual(payload["frame"]["model_id"], self.graph.model_id)
        self.assertEqual(
            payload["frame"]["manifest_sha256"], self.graph.manifest_sha256
        )
        self.assertFalse(payload["go_move_selected"])
        self.assertFalse(payload["baseline_controller_called"])
        self.assertFalse(payload["frame"]["go_move_selected"])
        self.assertFalse(payload["frame"]["baseline_controller_called"])

    def test_neural_frame_request_rejects_extra_fields_and_bad_seed(self) -> None:
        cases = (
            ({"extra": 1}, "invalid_neural_fields"),
            ({"seed": 1, "extra": 2}, "invalid_neural_fields"),
            ({"seed": True}, "invalid_neural_seed"),
            ({"seed": -1}, "invalid_neural_seed"),
            ({"seed": 2**63}, "invalid_neural_seed"),
            ([], "invalid_neural_fields"),
        )
        for request_payload, expected_code in cases:
            with self.subTest(payload=request_payload):
                status, payload = self.frame(request_payload)
                self.assertEqual(status, 400)
                self.assertEqual(payload["error"]["code"], expected_code)

        duplicate = b'{"seed":1,"seed":2}'
        status, _, response = self.request(
            "POST",
            "/api/neural/frame",
            duplicate,
            {
                "Content-Type": "application/json",
                "Content-Length": str(len(duplicate)),
            },
        )
        self.assertEqual(status, 400)
        self.assertEqual(
            json.loads(response.decode("utf-8"))["error"]["code"], "invalid_json"
        )

    def test_unavailable_model_fails_closed_without_a_fake_frame(self) -> None:
        unavailable = create_server(
            0,
            events_dir=Path(self.temp_dir.name) / "unavailable-events",
            neural_graph=None,
        )
        host, port = unavailable.server_address
        thread = threading.Thread(target=unavailable.serve_forever, daemon=True)
        thread.start()
        try:
            status, _, body = self._request_to(host, port, "GET", "/api/neural/status")
            payload = json.loads(body.decode("utf-8"))
            self.assertEqual(status, 200)
            self.assertFalse(payload["available"])
            self.assertEqual(payload["state"], "error")
            self.assertIsNotNone(payload["load_error"])

            encoded = b"{}"
            status, _, body = self._request_to(
                host,
                port,
                "POST",
                "/api/neural/frame",
                encoded,
                {"Content-Type": "application/json", "Content-Length": "2"},
            )
            payload = json.loads(body.decode("utf-8"))
            self.assertEqual(status, 503)
            self.assertEqual(payload["error"]["code"], "neural_unavailable")
            self.assertNotIn("frame", payload)
        finally:
            unavailable.shutdown()
            unavailable.server_close()
            thread.join(timeout=3)

    def test_neural_busy_and_failure_are_fail_closed_and_release_slot(self) -> None:
        entered = threading.Event()
        release = threading.Event()
        first_result: list[tuple[int, dict]] = []

        def slow_simulation(graph, encoding, **kwargs):
            entered.set()
            if not release.wait(timeout=2):
                raise RuntimeError("test release timed out")
            return simulate_go_encoding(graph, encoding, **kwargs)

        def run_first() -> None:
            first_result.append(self.frame({"seed": 9}))

        with patch("app.server.simulate_go_encoding", side_effect=slow_simulation):
            first_thread = threading.Thread(target=run_first, daemon=True)
            first_thread.start()
            self.assertTrue(entered.wait(timeout=2))
            status, busy = self.frame({})
            self.assertEqual(status, 409)
            self.assertEqual(busy["error"]["code"], "neural_busy")
            self.assertNotIn("frame", busy)
            release.set()
            first_thread.join(timeout=3)
        self.assertFalse(first_thread.is_alive())
        self.assertEqual(first_result[0][0], 200)

        with patch("app.server.simulate_go_encoding", side_effect=RuntimeError("boom")):
            status, failed = self.frame({})
        self.assertEqual(status, 500)
        self.assertEqual(failed["error"]["code"], "neural_frame_failed")
        self.assertNotIn("frame", failed)
        self.assertNotIn("decision_id", failed)

        # The failure path must release the single-flight slot.
        status, recovered = self.frame({"seed": 10})
        self.assertEqual(status, 200)
        self.assertTrue(recovered["ok"])


class Stage4NeuralMoveAPITests(unittest.TestCase):
    """End-to-end tests for the trained neural-policy controller route."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp_dir.name)
        cls.graph = _synthetic_neural_graph()

        # Derive the pool identity from a real frame for this exact graph, then
        # write and strictly reload the same checkpoint format used in Stage 4.
        probe_session = GoSession()
        probe_session.human_move((4, 4))
        probe_encoding = encode_go_state(probe_session.state)
        probe_frame = simulate_go_encoding(cls.graph, probe_encoding, seed=0)
        output_pool_hash = probe_frame["output_pool"]["group_hash"]
        weights, bias = initialise_linear_parameters(seed=0, scale=0.0)
        bias[0] = np.float32(10.0)
        cls.checkpoint_dir = cls.root / "checkpoint"
        write_policy_checkpoint(
            cls.checkpoint_dir,
            weights=weights,
            bias=bias,
            model_id=cls.graph.model_id,
            graph_manifest_sha256=cls.graph.manifest_sha256,
            output_pool_group_hash=output_pool_hash,
            training_status="trained",
            training_info={
                "method": "unit-test-fixed-linear-readout",
                "dataset_id": "stage4-http-contract-fixture",
                "sample_count": 1,
                "completed_at": "2026-09-21T00:00:00Z",
            },
        )
        cls.checkpoint = load_policy_checkpoint(cls.checkpoint_dir)
        cls.server = create_server(
            0,
            events_dir=cls.root / "events",
            neural_graph=cls.graph,
            policy_checkpoint=cls.checkpoint,
            policy_checkpoint_dir=cls.checkpoint_dir,
        )
        cls.host, cls.port = cls.server.server_address
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=3)
        if cls.thread.is_alive():
            raise RuntimeError("Stage 4 test server did not shut down cleanly")
        cls.temp_dir.cleanup()

    def setUp(self) -> None:
        self.server.go_session.reset()

    def request(
        self,
        method: str,
        path: str,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict]:
        connection = http.client.HTTPConnection(self.host, self.port, timeout=3)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
            return response.status, payload
        finally:
            connection.close()

    def post_json(self, path: str, payload: object) -> tuple[int, dict]:
        body = json.dumps(payload).encode("utf-8")
        return self.request(
            "POST",
            path,
            body,
            {"Content-Type": "application/json", "Content-Length": str(len(body))},
        )

    def human_center(self) -> dict:
        status, payload = self.post_json(
            "/api/go/action", {"action": "human_move", "row": 4, "col": 4}
        )
        self.assertEqual(status, 200)
        return payload

    def test_policy_status_reports_exact_strict_checkpoint_and_no_fallback(self) -> None:
        status, payload = self.request("GET", "/api/go/neural-policy-status")
        self.assertEqual(status, 200)
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["available"])
        self.assertEqual(payload["state"], "available")
        self.assertTrue(payload["stage4_controls_go_moves"])
        self.assertEqual(payload["controller"], "malecns-neural-policy")
        self.assertEqual(payload["checkpoint_hash"], self.checkpoint.checkpoint_hash)
        self.assertEqual(payload["training_status"], "trained")
        self.assertEqual(payload["model_id"], self.graph.model_id)
        self.assertEqual(
            payload["graph_manifest_sha256"], self.graph.manifest_sha256
        )
        self.assertFalse(payload["teacher_accessed_at_runtime"])
        self.assertFalse(payload["baseline_controller_called"])
        self.assertFalse(payload["fallback_used"])

    def test_matching_checkpoint_selects_and_commits_the_same_neural_move(self) -> None:
        self.human_center()
        pre_move = self.server.go_session.snapshot()
        status, payload = self.post_json("/api/go/neural-move", {"seed": 17})
        self.assertEqual(status, 200)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["controller"], "malecns-neural-policy")
        self.assertEqual(payload["decision"]["action_index"], 0)
        self.assertEqual(payload["decision"]["move"], [0, 0])
        self.assertEqual(payload["decision"]["frame_id"], payload["frame"]["frame_id"])
        self.assertEqual(
            payload["decision"]["encoding_hash"],
            payload["pre_move_encoding"]["encoding_hash"],
        )
        self.assertEqual(payload["session"]["move_number"], pre_move["move_number"] + 1)
        self.assertEqual(payload["session"]["board"][0][0], 2)
        self.assertEqual(payload["session"]["last_move"]["actor"], "neural_policy")
        self.assertEqual(
            payload["session"]["last_move"]["provenance"]["decision_hash"],
            payload["decision"]["decision_hash"],
        )
        self.assertEqual(payload["session"]["controller"], "malecns-neural-policy")
        self.assertEqual(payload["session"]["checkpoint"], self.checkpoint.checkpoint_hash)
        self.assertFalse(payload["teacher_accessed_at_runtime"])
        self.assertFalse(payload["baseline_controller_called"])
        self.assertFalse(payload["fallback_used"])

    def test_wrong_turn_does_not_change_board(self) -> None:
        before = self.server.go_session.snapshot()
        status, payload = self.post_json("/api/go/neural-move", {})
        self.assertEqual(status, 409)
        self.assertEqual(payload["error"]["code"], "wrong_turn")
        self.assertEqual(self.server.go_session.snapshot(), before)

    def test_policy_failure_is_atomic_and_releases_single_flight_lock(self) -> None:
        self.human_center()
        before = self.server.go_session.snapshot()
        with patch("app.server.select_action", side_effect=PolicyError("test failure")):
            status, payload = self.post_json("/api/go/neural-move", {"seed": 3})
        self.assertEqual(status, 500)
        self.assertEqual(payload["error"]["code"], "neural_move_failed")
        self.assertEqual(self.server.go_session.snapshot(), before)

        status, recovered = self.post_json("/api/go/neural-move", {"seed": 3})
        self.assertEqual(status, 200)
        self.assertTrue(recovered["ok"])

    def test_busy_slot_is_fail_closed_and_does_not_change_board(self) -> None:
        self.human_center()
        before = self.server.go_session.snapshot()
        self.assertTrue(self.server.neural_frame_lock.acquire(blocking=False))
        try:
            status, payload = self.post_json("/api/go/neural-move", {"seed": 4})
        finally:
            self.server.neural_frame_lock.release()
        self.assertEqual(status, 409)
        self.assertEqual(payload["error"]["code"], "neural_busy")
        self.assertEqual(self.server.go_session.snapshot(), before)

    def test_unavailable_policy_and_mismatched_checkpoint_fail_closed(self) -> None:
        mismatch = replace(self.checkpoint, model_id="other-model")
        with self.assertRaisesRegex(CheckpointValidationError, "does not match"):
            create_server(
                0,
                events_dir=self.root / "mismatch-events",
                neural_graph=self.graph,
                policy_checkpoint=mismatch,
            )

        unavailable = create_server(
            0,
            events_dir=self.root / "unavailable-policy-events",
            neural_graph=self.graph,
            policy_checkpoint=None,
        )
        host, port = unavailable.server_address
        thread = threading.Thread(target=unavailable.serve_forever, daemon=True)
        thread.start()
        try:
            unavailable.go_session.human_move((4, 4))
            before = unavailable.go_session.snapshot()
            body = b"{}"
            connection = http.client.HTTPConnection(host, port, timeout=3)
            try:
                connection.request(
                    "POST",
                    "/api/go/neural-move",
                    body=body,
                    headers={
                        "Content-Type": "application/json",
                        "Content-Length": str(len(body)),
                    },
                )
                response = connection.getresponse()
                payload = json.loads(response.read().decode("utf-8"))
                self.assertEqual(response.status, 503)
            finally:
                connection.close()
            self.assertEqual(payload["error"]["code"], "neural_policy_unavailable")
            self.assertEqual(unavailable.go_session.snapshot(), before)
        finally:
            unavailable.shutdown()
            unavailable.server_close()
            thread.join(timeout=3)


if __name__ == "__main__":
    unittest.main()
