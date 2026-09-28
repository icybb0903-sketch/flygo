"""Go-only release routes and the included production model."""

import http.client
import json
import threading
import unittest
from pathlib import Path

from scripts.run_flygo import create_flygo_server


class FlygoReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = create_flygo_server(0)
        cls.host, cls.port = cls.server.server_address
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def request(self, method, path, payload=None):
        connection = http.client.HTTPConnection(self.host, self.port, timeout=15)
        body = None if payload is None else json.dumps(payload)
        headers = {} if body is None else {"Content-Type": "application/json"}
        try:
            connection.request(method, path, body, headers)
            response = connection.getresponse()
            return response.status, response.read()
        finally:
            connection.close()

    def test_homepage_is_go_not_other_activities(self):
        for path in ("/", "/index.html", "/go"):
            status, body = self.request("GET", path)
            self.assertEqual(status, 200)
            self.assertIn(b"roomCanvas", body)
            self.assertNotIn(b'href="/monitor"', body)
            self.assertNotIn(b'href="/lab"', body)

    def test_only_go_resources_are_served(self):
        for path in ("/monitor", "/monitor.js", "/api/agents", "/api/status", "/simulations/piano/", "/../README.md"):
            status, _ = self.request("GET", path)
            self.assertIn(status, (400, 404))
        status, _ = self.request("POST", "/api/simulate", {"scenario": "left", "edges_enabled": True})
        self.assertEqual(status, 404)

    def test_included_model_and_trained_checkpoint_load(self):
        status, body = self.request("GET", "/api/go/neural-policy-status")
        policy = json.loads(body)
        self.assertEqual(status, 200)
        self.assertTrue(policy["available"])
        self.assertEqual(policy["training_status"], "trained")
        self.assertFalse(policy["fallback_used"])

    def test_neural_move_uses_published_runtime_without_baseline(self):
        self.assertEqual(self.request("POST", "/api/go/action", {"action": "reset"})[0], 200)
        self.assertEqual(self.request("POST", "/api/go/action", {"action": "human_move", "row": 4, "col": 4})[0], 200)
        self.assertEqual(self.request("POST", "/api/go/action", {"action": "bot_move"})[0], 409)
        status, body = self.request("POST", "/api/go/neural-move", {"seed": 0})
        result = json.loads(body)
        self.assertEqual(status, 200, result)
        self.assertFalse(result["fallback_used"])
        self.assertFalse(result["baseline_controller_called"])
        self.assertFalse(result["teacher_accessed_at_runtime"])
        self.assertEqual(result["controller"], "malecns-neural-policy")

    def test_no_private_agent_events_shipped(self):
        root = Path(__file__).resolve().parents[1]
        self.assertFalse((root / "data" / "agent_monitor" / "events").exists())
