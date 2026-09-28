"""Tests for the append-only agent monitor backend."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.agent_monitor import (
    EVENT_FIELDS,
    LEGACY_SCHEMA_VERSION,
    SCHEMA_VERSION,
    UNAVAILABLE,
    V1_EVENT_FIELDS,
    EventValidationError,
    append_event,
    create_event,
    load_events,
    reduce_snapshot,
    validate_event,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = PROJECT_ROOT / "scripts" / "agent_event.py"


class AgentMonitorTests(unittest.TestCase):
    def event(self, event_type: str = "activity", **overrides: object) -> dict:
        values = {
            "agent_id": "agent-1",
            "source": "unit-test",
            "timestamp_utc": "2026-09-15T00:00:00Z",
            "detail": "working" if event_type == "activity" else None,
        }
        values.update(overrides)
        return create_event(event_type, **values)

    def test_schema_contains_every_field_and_explicit_nulls(self) -> None:
        event = self.event()
        self.assertEqual(set(event), EVENT_FIELDS)
        self.assertEqual(event["schema_version"], 2)
        self.assertIsNone(event["agent_name"])
        self.assertIsNone(event["test"])
        self.assertIsNone(event["model_name"])
        self.assertIsNone(event["runtime_presence"])
        self.assertIsNone(event["observed_at"])
        self.assertEqual(event["files"], [])

    def test_schema_rejects_missing_and_extra_fields(self) -> None:
        event = self.event()
        del event["source"]
        with self.assertRaises(EventValidationError):
            validate_event(event)

    def test_legacy_v1_event_remains_loadable_with_exact_old_schema(self) -> None:
        event = self.event()
        legacy = {key: event[key] for key in V1_EVENT_FIELDS}
        legacy["schema_version"] = LEGACY_SCHEMA_VERSION
        self.assertEqual(validate_event(legacy), legacy)
        legacy["model_name"] = "invented-extra"
        with self.assertRaisesRegex(EventValidationError, "schema-v1"):
            validate_event(legacy)

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary, "legacy.json")
            path.write_text(json.dumps({key: event[key] for key in V1_EVENT_FIELDS} | {"schema_version": 1}), encoding="utf-8")
            loaded = load_events(temporary)
            self.assertEqual(loaded[0]["schema_version"], 1)
            snapshot_agent = reduce_snapshot(loaded)["agents"][0]
            self.assertEqual(snapshot_agent["runtime_presence"], UNAVAILABLE)
            self.assertEqual(snapshot_agent["record_group"], UNAVAILABLE)

    def test_agent_created_records_explicit_unavailable_truth_metadata(self) -> None:
        event = self.event(
            "agent_created",
            agent_name="worker",
            detail=None,
        )
        self.assertEqual(event["model_name"], UNAVAILABLE)
        self.assertEqual(event["reasoning_effort"], UNAVAILABLE)
        self.assertEqual(event["runtime_presence"], UNAVAILABLE)
        self.assertEqual(event["observed_at"], event["timestamp_utc"])

    def test_runtime_presence_requires_timestamp_and_strict_values(self) -> None:
        with self.assertRaisesRegex(EventValidationError, "provided together"):
            self.event(runtime_presence="present")
        with self.assertRaisesRegex(EventValidationError, "runtime_presence"):
            self.event(runtime_presence="online", observed_at="2026-09-15T00:00:00Z")
        with self.assertRaisesRegex(EventValidationError, "reasoning_effort"):
            self.event(reasoning_effort="extreme")
        with self.assertRaisesRegex(EventValidationError, "runtime_observed"):
            self.event("runtime_observed", detail=None)
        event = self.event()
        event["surprise"] = True
        with self.assertRaises(EventValidationError):
            validate_event(event)

    def test_invalid_uuid_is_rejected(self) -> None:
        with self.assertRaisesRegex(EventValidationError, "UUID"):
            self.event(event_id="not-a-uuid")
        with self.assertRaisesRegex(EventValidationError, "canonical"):
            self.event(event_id="A591A6D4-8F04-4C17-A21E-92D79DD00C42")

    def test_invalid_timestamp_and_non_utc_offset_are_rejected(self) -> None:
        for value in ("2026-09-15 00:00:00Z", "2026-09-15T00:00:00+08:00", "bad"):
            with self.subTest(value=value), self.assertRaises(EventValidationError):
                self.event(timestamp_utc=value)

    def test_invalid_status_and_event_type_are_rejected(self) -> None:
        with self.assertRaises(EventValidationError):
            self.event(status="busy")
        with self.assertRaises(EventValidationError):
            self.event("invented")

    def test_wrong_types_and_excessive_lengths_are_rejected(self) -> None:
        event = self.event()
        event["event_type"] = []
        with self.assertRaises(EventValidationError):
            validate_event(event)
        event = self.event()
        event["status"] = []
        with self.assertRaises(EventValidationError):
            validate_event(event)
        with self.assertRaises(EventValidationError):
            self.event(source="x" * 129)

    def test_absolute_parent_and_traversal_file_paths_are_rejected(self) -> None:
        for value in (
            "../secret.txt",
            "src/../secret.txt",
            "/etc/passwd",
            "C:\\Windows\\win.ini",
            "\\\\server\\share\\file.txt",
        ):
            with self.subTest(value=value), self.assertRaises(EventValidationError):
                self.event("files_changed", files=[value], detail=None)

    def test_event_specific_required_combinations(self) -> None:
        invalid = (
            ("agent_created", {}),
            ("task_assigned", {}),
            ("status_changed", {}),
            ("activity", {"detail": None}),
            ("files_changed", {}),
            ("test_started", {}),
            (
                "test_finished",
                {"test": {"status": "running", "summary": None, "command": None}},
            ),
            ("agent_completed", {"status": "working"}),
            ("agent_failed", {"status": "failed"}),
        )
        for event_type, overrides in invalid:
            with self.subTest(event_type=event_type), self.assertRaises(EventValidationError):
                values = {"detail": None, **overrides}
                self.event(event_type, **values)

    def test_append_is_one_atomic_json_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            events_dir = Path(temporary) / "events"
            event = self.event()
            path = append_event(events_dir, event)
            self.assertTrue(path.is_file())
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), event)
            self.assertEqual(list(events_dir.glob("*.tmp")), [])
            self.assertRegex(path.name, r"^\d{8}T\d{12}Z_[0-9a-f-]{36}\.json$")

    def test_concurrent_atomic_writes_do_not_lose_events(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            events_dir = Path(temporary) / "events"
            events = [
                self.event(
                    event_id=str(uuid.uuid4()),
                    timestamp_utc=f"2026-09-15T00:00:{index:02d}Z",
                    detail=f"event {index}",
                )
                for index in range(30)
            ]
            with ThreadPoolExecutor(max_workers=8) as executor:
                paths = list(executor.map(lambda item: append_event(events_dir, item), events))
            self.assertEqual(len(set(paths)), 30)
            self.assertEqual(len(load_events(events_dir)), 30)
            self.assertEqual(list(events_dir.glob("*.tmp")), [])

    def test_load_events_sorts_by_timestamp_then_event_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first_id = "00000000-0000-4000-8000-000000000001"
            second_id = "00000000-0000-4000-8000-000000000002"
            events = (
                self.event(event_id=second_id, detail="second id"),
                self.event(
                    event_id=str(uuid.uuid4()),
                    timestamp_utc="2026-09-14T23:59:59Z",
                    detail="earlier",
                ),
                self.event(event_id=first_id, detail="first id"),
            )
            for event in events:
                append_event(temporary, event)
            loaded = load_events(temporary)
            self.assertEqual(
                [event["detail"] for event in loaded],
                ["earlier", "first id", "second id"],
            )

    def test_load_events_rejects_invalid_json_and_duplicate_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            Path(temporary, "bad.json").write_text("{bad", encoding="utf-8")
            with self.assertRaises(EventValidationError):
                load_events(temporary)
        with tempfile.TemporaryDirectory() as temporary:
            event = self.event()
            for name in ("one.json", "two.json"):
                Path(temporary, name).write_text(json.dumps(event), encoding="utf-8")
            with self.assertRaisesRegex(EventValidationError, "duplicate"):
                load_events(temporary)

    def test_reducer_tracks_fields_files_tests_error_and_times(self) -> None:
        agent_id = "worker-7"
        events = [
            self.event(
                "agent_created",
                agent_id=agent_id,
                agent_name="backend",
                parent_id="root",
                status="working",
                detail=None,
                timestamp_utc="2026-09-15T01:00:00Z",
            ),
            self.event(
                "task_assigned",
                agent_id=agent_id,
                task="Implement store",
                detail="building",
                timestamp_utc="2026-09-15T01:01:00Z",
            ),
            self.event(
                "files_changed",
                agent_id=agent_id,
                files=["src/agent_monitor.py", "tests/test_agent_monitor.py"],
                detail=None,
                timestamp_utc="2026-09-15T01:02:00Z",
            ),
            self.event(
                "test_started",
                agent_id=agent_id,
                status="testing",
                test={"status": "running", "summary": None, "command": "unittest"},
                detail="testing",
                timestamp_utc="2026-09-15T01:03:00Z",
            ),
            self.event(
                "test_finished",
                agent_id=agent_id,
                test={"status": "failed", "summary": "one failure", "command": "unittest"},
                error="assertion failed",
                detail=None,
                timestamp_utc="2026-09-15T01:04:00Z",
            ),
            self.event(
                "agent_failed",
                agent_id=agent_id,
                status="failed",
                error="assertion failed",
                detail=None,
                timestamp_utc="2026-09-15T01:05:00Z",
            ),
        ]
        agent = reduce_snapshot(reversed(events))["agents"][0]
        self.assertEqual(agent["name"], "backend")
        self.assertEqual(agent["parent_id"], "root")
        self.assertEqual(agent["task"], "Implement store")
        self.assertEqual(agent["status"], "failed")
        self.assertEqual(agent["current"], "testing")
        self.assertEqual(agent["files"], ["src/agent_monitor.py", "tests/test_agent_monitor.py"])
        self.assertEqual(agent["last_test"]["status"], "failed")
        self.assertEqual(agent["error"], "assertion failed")
        self.assertEqual(agent["started_at"], "2026-09-15T01:00:00Z")
        self.assertEqual(agent["completed_at"], "2026-09-15T01:05:00Z")
        self.assertEqual(len(agent["timeline"]), 6)

    def test_reducer_does_not_infer_unknown_status_or_times(self) -> None:
        event = self.event("activity", agent_id="orphan", detail="observed")
        agent = reduce_snapshot([event])["agents"][0]
        self.assertIsNone(agent["status"])
        self.assertIsNone(agent["started_at"])
        self.assertIsNone(agent["completed_at"])
        self.assertIsNone(agent["name"])
        self.assertEqual(agent["runtime_presence"], UNAVAILABLE)
        self.assertEqual(agent["record_group"], UNAVAILABLE)

    def test_new_active_lifecycle_clears_old_completion_time(self) -> None:
        events = [
            self.event(
                "agent_completed",
                status="completed",
                detail="old task complete",
                timestamp_utc="2026-09-15T00:00:00Z",
            ),
            self.event(
                "task_assigned",
                task="new task",
                status="working",
                detail="new task started",
                timestamp_utc="2026-09-15T00:00:01Z",
            ),
            self.event(
                "runtime_observed",
                status="working",
                detail="actually present",
                runtime_presence="present",
                observed_at="2026-09-15T00:00:02Z",
                timestamp_utc="2026-09-15T00:00:02Z",
            ),
        ]
        agent = reduce_snapshot(events)["agents"][0]
        self.assertEqual(agent["status"], "working")
        self.assertIsNone(agent["completed_at"])
        self.assertEqual(agent["record_group"], "current")
        self.assertEqual(agent["timeline"][0]["event_type"], "agent_completed")

    def test_reducer_separates_current_historical_failed_and_interrupted(self) -> None:
        events = [
            self.event(
                "runtime_observed",
                agent_id="current",
                detail=None,
                runtime_presence="present",
                observed_at="2026-09-15T00:00:00Z",
            ),
            self.event(
                "runtime_observed",
                agent_id="history",
                detail=None,
                runtime_presence="not_present",
                observed_at="2026-09-15T00:00:00Z",
            ),
            self.event(
                "agent_failed",
                agent_id="failed",
                detail=None,
                status="failed",
                error="boom",
            ),
            self.event(
                "agent_interrupted",
                agent_id="interrupted",
                detail=None,
                status="interrupted",
            ),
        ]
        groups = {
            agent["agent_id"]: agent["record_group"]
            for agent in reduce_snapshot(events)["agents"]
        }
        self.assertEqual(
            groups,
            {
                "current": "current",
                "history": "historical",
                "failed": "failed",
                "interrupted": "interrupted",
            },
        )

    def test_latest_presence_is_selected_by_observation_time(self) -> None:
        events = [
            self.event(
                "runtime_observed",
                detail=None,
                timestamp_utc="2026-09-15T00:00:01Z",
                runtime_presence="present",
                observed_at="2026-09-15T00:00:01Z",
            ),
            self.event(
                "runtime_observed",
                detail=None,
                timestamp_utc="2026-09-15T00:00:03Z",
                runtime_presence="not_present",
                observed_at="2026-09-15T00:00:00Z",
            ),
        ]
        agent = reduce_snapshot(events)["agents"][0]
        self.assertEqual(agent["runtime_presence"], "present")
        self.assertEqual(agent["observed_at"], "2026-09-15T00:00:01Z")
        self.assertEqual(agent["record_group"], "current")

    def test_reducer_clears_resolved_current_error_but_keeps_timeline(self) -> None:
        events = [
            self.event(
                "test_finished",
                test={"status": "failed", "summary": "bad", "command": "test"},
                error="assertion failed",
                detail=None,
            ),
            self.event(
                "test_finished",
                test={"status": "passed", "summary": "fixed", "command": "test"},
                detail=None,
                timestamp_utc="2026-09-15T00:00:01Z",
            ),
            self.event(
                "agent_completed",
                status="completed",
                detail=None,
                timestamp_utc="2026-09-15T00:00:02Z",
            ),
        ]
        agent = reduce_snapshot(events)["agents"][0]
        self.assertIsNone(agent["error"])
        self.assertEqual(agent["timeline"][0]["error"], "assertion failed")
        self.assertEqual(agent["record_group"], "historical")

    def test_reducer_aggregates_files_once_and_keeps_latest_test(self) -> None:
        events = [
            self.event("files_changed", files=["src/a.py", "src/b.py"], detail=None),
            self.event(
                "files_changed",
                files=["src/b.py", "src/c.py"],
                detail=None,
                timestamp_utc="2026-09-15T00:00:01Z",
            ),
            self.event(
                "test_started",
                test={"status": "running", "summary": None, "command": "python -m unittest"},
                detail=None,
                timestamp_utc="2026-09-15T00:00:02Z",
            ),
            self.event(
                "test_finished",
                test={"status": "passed", "summary": "3 passed", "command": "python -m unittest"},
                detail=None,
                timestamp_utc="2026-09-15T00:00:03Z",
            ),
        ]
        agent = reduce_snapshot(events)["agents"][0]
        self.assertEqual(agent["files"], ["src/a.py", "src/b.py", "src/c.py"])
        self.assertEqual(agent["last_test"]["status"], "passed")

    def test_cli_preserves_historical_timestamp_and_prints_only_id_and_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            event_id = "00000000-0000-4000-8000-000000000123"
            result = subprocess.run(
                [
                    sys.executable,
                    str(CLI_PATH),
                    "activity",
                    "--event-id",
                    event_id,
                    "--timestamp-utc",
                    "2024-03-02T01:02:03Z",
                    "--source",
                    "codex-local-log",
                    "--agent-id",
                    "historical-worker",
                    "--detail",
                    "replayed from log",
                    "--file",
                    "src/a.py",
                    "--file",
                    "tests/test_a.py",
                    "--events-dir",
                    temporary,
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(result.stdout.strip().splitlines()), 2)
            self.assertIn(f"event_id={event_id}", result.stdout)
            loaded = load_events(temporary)
            self.assertEqual(loaded[0]["timestamp_utc"], "2024-03-02T01:02:03Z")
            self.assertEqual(loaded[0]["files"], ["src/a.py", "tests/test_a.py"])

    def test_cli_supports_flag_event_type_and_test_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run(
                [
                    sys.executable,
                    str(CLI_PATH),
                    "--event-type",
                    "test_finished",
                    "--agent-id",
                    "worker",
                    "--test-status",
                    "passed",
                    "--test-summary",
                    "all green",
                    "--test-command",
                    "python -m unittest",
                    "--events-dir",
                    temporary,
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(load_events(temporary)[0]["test"]["status"], "passed")

    def test_cli_writes_real_runtime_observation_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run(
                [
                    sys.executable,
                    str(CLI_PATH),
                    "runtime_observed",
                    "--agent-id",
                    "/root/worker",
                    "--model-name",
                    "unavailable",
                    "--reasoning-effort",
                    "unavailable",
                    "--runtime-presence",
                    "present",
                    "--observed-at",
                    "2026-09-15T00:00:00Z",
                    "--events-dir",
                    temporary,
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            loaded = load_events(temporary)[0]
            self.assertEqual(loaded["schema_version"], SCHEMA_VERSION)
            self.assertEqual(loaded["runtime_presence"], "present")
            self.assertEqual(loaded["model_name"], UNAVAILABLE)
            self.assertEqual(loaded["reasoning_effort"], UNAVAILABLE)


if __name__ == "__main__":
    unittest.main()
