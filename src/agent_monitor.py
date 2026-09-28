"""Validated, append-only event storage for the local agent monitor.

The store intentionally uses one JSON document per event.  Writers create a
temporary file in the destination directory and atomically replace it with the
final, timestamp-sortable filename, so concurrent writers never share a file.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any


SCHEMA_VERSION = 2
LEGACY_SCHEMA_VERSION = 1
EVENT_TYPES = frozenset(
    {
        "agent_created",
        "task_assigned",
        "status_changed",
        "activity",
        "files_changed",
        "test_started",
        "test_finished",
        "agent_completed",
        "agent_failed",
        "agent_interrupted",
        "runtime_observed",
        "integration_started",
        "integration_finished",
    }
)
V1_EVENT_TYPES = EVENT_TYPES - {"agent_interrupted", "runtime_observed"}
AGENT_STATUSES = frozenset(
    {"waiting", "working", "testing", "completed", "failed", "interrupted"}
)
TEST_STATUSES = frozenset({"running", "passed", "failed"})
RUNTIME_PRESENCES = frozenset({"present", "not_present", "unavailable"})
REASONING_EFFORTS = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra", "unavailable"}
)
UNAVAILABLE = "unavailable"
V1_EVENT_FIELDS = frozenset(
    {
        "schema_version",
        "event_id",
        "timestamp_utc",
        "source",
        "event_type",
        "agent_id",
        "agent_name",
        "parent_id",
        "task",
        "status",
        "detail",
        "files",
        "test",
        "error",
    }
)
EVENT_FIELDS = V1_EVENT_FIELDS | frozenset(
    {"model_name", "reasoning_effort", "runtime_presence", "observed_at"}
)
TEST_FIELDS = frozenset({"status", "summary", "command"})

MAX_SOURCE_LENGTH = 128
MAX_ID_LENGTH = 128
MAX_NAME_LENGTH = 256
MAX_TASK_LENGTH = 4_096
MAX_DETAIL_LENGTH = 16_384
MAX_ERROR_LENGTH = 16_384
MAX_TEST_TEXT_LENGTH = 4_096
MAX_FILE_PATH_LENGTH = 1_024
MAX_FILES_PER_EVENT = 512
MAX_MODEL_NAME_LENGTH = 256

_UTC_TIMESTAMP_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})T(?P<time>\d{2}:\d{2}:\d{2})"
    r"(?P<fraction>\.\d{1,6})?Z$"
)


class EventValidationError(ValueError):
    """Raised when an event does not conform to the monitor schema."""


def utc_now() -> str:
    """Return a schema-compatible UTC timestamp."""
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _parse_timestamp(value: Any, field: str = "timestamp_utc") -> datetime:
    if not isinstance(value, str) or not _UTC_TIMESTAMP_RE.fullmatch(value):
        raise EventValidationError(
            f"{field} must be an ISO 8601 UTC timestamp ending in Z"
        )
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise EventValidationError(f"{field} is not a valid UTC timestamp") from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise EventValidationError(f"{field} must be UTC")
    return parsed


def _required_text(value: Any, field: str, maximum: int) -> str:
    if not isinstance(value, str):
        raise EventValidationError(f"{field} must be a string")
    if not value.strip():
        raise EventValidationError(f"{field} must not be empty")
    if len(value) > maximum:
        raise EventValidationError(f"{field} exceeds {maximum} characters")
    if "\x00" in value:
        raise EventValidationError(f"{field} must not contain NUL")
    return value


def _optional_text(value: Any, field: str, maximum: int) -> str | None:
    if value is None:
        return None
    return _required_text(value, field, maximum)


def _validate_uuid(value: Any) -> str:
    if not isinstance(value, str):
        raise EventValidationError("event_id must be a UUID string")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise EventValidationError("event_id must be a valid UUID") from exc
    canonical = str(parsed)
    if value != canonical:
        raise EventValidationError("event_id must use canonical lowercase UUID form")
    return value


def _validate_file_path(value: Any) -> str:
    path = _required_text(value, "files item", MAX_FILE_PATH_LENGTH)
    normalized = path.replace("\\", "/")
    parts = normalized.split("/")
    windows_path = PureWindowsPath(path)
    if (
        normalized.startswith("/")
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise EventValidationError(
            f"files item must be a normalized relative project path: {path!r}"
        )
    return "/".join(parts)


def _validate_test(value: Any) -> dict[str, str | None] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise EventValidationError("test must be an object or null")
    keys = set(value)
    if keys != TEST_FIELDS:
        missing = sorted(TEST_FIELDS - keys)
        extra = sorted(keys - TEST_FIELDS)
        raise EventValidationError(
            f"test fields must be exactly {sorted(TEST_FIELDS)!r}; "
            f"missing={missing!r}, extra={extra!r}"
        )
    status = value["status"]
    if not isinstance(status, str) or status not in TEST_STATUSES:
        raise EventValidationError(
            f"test.status must be one of {sorted(TEST_STATUSES)!r}"
        )
    return {
        "status": status,
        "summary": _optional_text(
            value["summary"], "test.summary", MAX_TEST_TEXT_LENGTH
        ),
        "command": _optional_text(
            value["command"], "test.command", MAX_TEST_TEXT_LENGTH
        ),
    }


def validate_event(event: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an event and return a detached, normalized dictionary.

    The top-level and nested test objects have exact schemas.  Optional scalar
    fields must be explicit ``null`` values; ``files`` is always a list.
    """
    if not isinstance(event, Mapping):
        raise EventValidationError("event must be an object")
    schema_version = event.get("schema_version")
    if type(schema_version) is not int or schema_version not in {
        LEGACY_SCHEMA_VERSION,
        SCHEMA_VERSION,
    }:
        raise EventValidationError(
            f"schema_version must be {LEGACY_SCHEMA_VERSION} or {SCHEMA_VERSION}"
        )
    expected_fields = (
        V1_EVENT_FIELDS if schema_version == LEGACY_SCHEMA_VERSION else EVENT_FIELDS
    )
    keys = set(event)
    if keys != expected_fields:
        missing = sorted(expected_fields - keys)
        extra = sorted(keys - expected_fields)
        raise EventValidationError(
            f"schema-v{schema_version} event fields must be exactly "
            f"{sorted(expected_fields)!r}; "
            f"missing={missing!r}, extra={extra!r}"
        )

    event_id = _validate_uuid(event["event_id"])
    timestamp_utc = event["timestamp_utc"]
    _parse_timestamp(timestamp_utc)
    source = _required_text(event["source"], "source", MAX_SOURCE_LENGTH)
    event_type = event["event_type"]
    allowed_event_types = (
        V1_EVENT_TYPES if schema_version == LEGACY_SCHEMA_VERSION else EVENT_TYPES
    )
    if not isinstance(event_type, str) or event_type not in allowed_event_types:
        raise EventValidationError(
            f"event_type must be one of {sorted(allowed_event_types)!r}"
        )
    agent_id = _required_text(event["agent_id"], "agent_id", MAX_ID_LENGTH)
    agent_name = _optional_text(event["agent_name"], "agent_name", MAX_NAME_LENGTH)
    parent_id = _optional_text(event["parent_id"], "parent_id", MAX_ID_LENGTH)
    if parent_id == agent_id:
        raise EventValidationError("parent_id must not equal agent_id")
    task = _optional_text(event["task"], "task", MAX_TASK_LENGTH)
    status = event["status"]
    allowed_statuses = (
        AGENT_STATUSES - {"interrupted"}
        if schema_version == LEGACY_SCHEMA_VERSION
        else AGENT_STATUSES
    )
    if status is not None and (
        not isinstance(status, str) or status not in allowed_statuses
    ):
        raise EventValidationError(
            f"status must be null or one of {sorted(allowed_statuses)!r}"
        )
    detail = _optional_text(event["detail"], "detail", MAX_DETAIL_LENGTH)
    error = _optional_text(event["error"], "error", MAX_ERROR_LENGTH)

    raw_files = event["files"]
    if not isinstance(raw_files, list):
        raise EventValidationError("files must be a list")
    if len(raw_files) > MAX_FILES_PER_EVENT:
        raise EventValidationError(
            f"files must contain at most {MAX_FILES_PER_EVENT} paths"
        )
    files = [_validate_file_path(item) for item in raw_files]
    if len(files) != len(set(files)):
        raise EventValidationError("files must not contain duplicate paths")

    test = _validate_test(event["test"])

    model_name = None
    reasoning_effort = None
    runtime_presence = None
    observed_at = None
    if schema_version == SCHEMA_VERSION:
        model_name = _optional_text(
            event["model_name"], "model_name", MAX_MODEL_NAME_LENGTH
        )
        reasoning_effort = event["reasoning_effort"]
        if reasoning_effort is not None and (
            not isinstance(reasoning_effort, str)
            or reasoning_effort not in REASONING_EFFORTS
        ):
            raise EventValidationError(
                "reasoning_effort must be null or one of "
                f"{sorted(REASONING_EFFORTS)!r}"
            )
        runtime_presence = event["runtime_presence"]
        if runtime_presence is not None and (
            not isinstance(runtime_presence, str)
            or runtime_presence not in RUNTIME_PRESENCES
        ):
            raise EventValidationError(
                "runtime_presence must be null or one of "
                f"{sorted(RUNTIME_PRESENCES)!r}"
            )
        observed_at = event["observed_at"]
        if observed_at is not None:
            _parse_timestamp(observed_at, "observed_at")
        if (runtime_presence is None) != (observed_at is None):
            raise EventValidationError(
                "runtime_presence and observed_at must be provided together"
            )

    if event_type == "agent_created" and agent_name is None:
        raise EventValidationError("agent_created requires agent_name")
    if event_type == "agent_created" and schema_version == SCHEMA_VERSION:
        if model_name is None or reasoning_effort is None:
            raise EventValidationError(
                "schema-v2 agent_created requires model_name and reasoning_effort; "
                "use 'unavailable' when Codex does not expose them"
            )
        if runtime_presence is None or observed_at is None:
            raise EventValidationError(
                "schema-v2 agent_created requires runtime_presence and observed_at"
            )
    if event_type == "task_assigned" and task is None:
        raise EventValidationError("task_assigned requires task")
    if event_type == "status_changed" and status is None:
        raise EventValidationError("status_changed requires status")
    if event_type == "activity" and detail is None:
        raise EventValidationError("activity requires detail")
    if event_type == "files_changed" and not files:
        raise EventValidationError("files_changed requires at least one file")
    if event_type == "test_started":
        if test is None or test["status"] != "running":
            raise EventValidationError("test_started requires test.status=running")
    if event_type == "test_finished":
        if test is None or test["status"] not in {"passed", "failed"}:
            raise EventValidationError(
                "test_finished requires test.status=passed or failed"
            )
    if event_type == "agent_completed" and status != "completed":
        raise EventValidationError("agent_completed requires status=completed")
    if event_type == "agent_failed":
        if status != "failed":
            raise EventValidationError("agent_failed requires status=failed")
        if error is None:
            raise EventValidationError("agent_failed requires error")
    if event_type == "agent_interrupted" and status != "interrupted":
        raise EventValidationError("agent_interrupted requires status=interrupted")
    if event_type == "runtime_observed":
        if runtime_presence is None or observed_at is None:
            raise EventValidationError(
                "runtime_observed requires runtime_presence and observed_at"
            )

    normalized = {
        "schema_version": schema_version,
        "event_id": event_id,
        "timestamp_utc": timestamp_utc,
        "source": source,
        "event_type": event_type,
        "agent_id": agent_id,
        "agent_name": agent_name,
        "parent_id": parent_id,
        "task": task,
        "status": status,
        "detail": detail,
        "files": files,
        "test": test,
        "error": error,
    }
    if schema_version == SCHEMA_VERSION:
        normalized.update(
            {
                "model_name": model_name,
                "reasoning_effort": reasoning_effort,
                "runtime_presence": runtime_presence,
                "observed_at": observed_at,
            }
        )
    return normalized


def create_event(
    event_type: str,
    *,
    agent_id: str,
    source: str = "agent_event_cli",
    event_id: str | None = None,
    timestamp_utc: str | None = None,
    agent_name: str | None = None,
    parent_id: str | None = None,
    task: str | None = None,
    status: str | None = None,
    detail: str | None = None,
    files: Iterable[str] = (),
    test: Mapping[str, Any] | None = None,
    error: str | None = None,
    model_name: str | None = None,
    reasoning_effort: str | None = None,
    runtime_presence: str | None = None,
    observed_at: str | None = None,
) -> dict[str, Any]:
    """Create and validate a complete schema-v2 event.

    Creation events must record all truth metadata.  When Codex does not expose
    a value, the explicit ``unavailable`` sentinel is used instead of guessing.
    Other event types only carry metadata when it was actually observed.
    """
    event_timestamp = timestamp_utc or utc_now()
    if event_type == "agent_created":
        model_name = model_name or UNAVAILABLE
        reasoning_effort = reasoning_effort or UNAVAILABLE
        runtime_presence = runtime_presence or UNAVAILABLE
        observed_at = observed_at or event_timestamp
    event = {
        "schema_version": SCHEMA_VERSION,
        "event_id": event_id or str(uuid.uuid4()),
        "timestamp_utc": event_timestamp,
        "source": source,
        "event_type": event_type,
        "agent_id": agent_id,
        "agent_name": agent_name,
        "parent_id": parent_id,
        "task": task,
        "status": status,
        "detail": detail,
        "files": list(files),
        "test": dict(test) if test is not None else None,
        "error": error,
        "model_name": model_name,
        "reasoning_effort": reasoning_effort,
        "runtime_presence": runtime_presence,
        "observed_at": observed_at,
    }
    return validate_event(event)


def _event_filename(event: Mapping[str, Any]) -> str:
    timestamp = _parse_timestamp(event["timestamp_utc"])
    sort_key = timestamp.strftime("%Y%m%dT%H%M%S%fZ")
    return f"{sort_key}_{event['event_id']}.json"


def append_event(events_dir: str | os.PathLike[str], event: Mapping[str, Any]) -> Path:
    """Atomically append one validated event and return its final path."""
    validated = validate_event(event)
    directory = Path(events_dir)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / _event_filename(validated)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{validated['event_id']}.", suffix=".tmp", dir=directory
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(validated, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        finally:
            raise
    return destination


def load_events(events_dir: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Load, validate, de-duplicate, and chronologically sort event files."""
    directory = Path(events_dir)
    if not directory.exists():
        return []
    if not directory.is_dir():
        raise EventValidationError(f"events path is not a directory: {directory}")

    events: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for path in directory.glob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
            event = validate_event(raw)
        except (OSError, UnicodeError, json.JSONDecodeError, EventValidationError) as exc:
            raise EventValidationError(f"invalid event file {path.name}: {exc}") from exc
        if event["event_id"] in seen_ids:
            raise EventValidationError(f"duplicate event_id: {event['event_id']}")
        seen_ids.add(event["event_id"])
        events.append(event)
    events.sort(key=lambda item: (_parse_timestamp(item["timestamp_utc"]), item["event_id"]))
    return events


def reduce_snapshot(events: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Reduce validated events into a deterministic per-agent snapshot.

    Agent status is copied only from a non-null event ``status`` field.  In
    particular, neither event type nor missing history is used to invent it.
    """
    validated_events = [validate_event(event) for event in events]
    event_ids = [event["event_id"] for event in validated_events]
    if len(event_ids) != len(set(event_ids)):
        raise EventValidationError("events contain duplicate event_id values")
    validated_events.sort(
        key=lambda item: (_parse_timestamp(item["timestamp_utc"]), item["event_id"])
    )

    agents: dict[str, dict[str, Any]] = {}
    file_sets: dict[str, set[str]] = {}
    for event in validated_events:
        agent_id = event["agent_id"]
        if agent_id not in agents:
            agents[agent_id] = {
                "agent_id": agent_id,
                "name": None,
                "parent_id": None,
                "task": None,
                "status": None,
                "current": None,
                "files": [],
                "last_test": None,
                "started_at": None,
                "completed_at": None,
                "error": None,
                "model_name": UNAVAILABLE,
                "reasoning_effort": UNAVAILABLE,
                "runtime_presence": UNAVAILABLE,
                "observed_at": None,
                "record_group": UNAVAILABLE,
                "timeline": [],
            }
            file_sets[agent_id] = set()
        agent = agents[agent_id]

        if event["agent_name"] is not None:
            agent["name"] = event["agent_name"]
        if event["parent_id"] is not None:
            agent["parent_id"] = event["parent_id"]
        if event["task"] is not None:
            agent["task"] = event["task"]
        if event["status"] is not None:
            agent["status"] = event["status"]
            if event["status"] in {"waiting", "working", "testing"}:
                # A new active lifecycle supersedes an earlier terminal time.
                # The earlier completion remains auditable in ``timeline``.
                agent["completed_at"] = None
        if event["detail"] is not None:
            agent["current"] = event["detail"]
        for changed_file in event["files"]:
            if changed_file not in file_sets[agent_id]:
                file_sets[agent_id].add(changed_file)
                agent["files"].append(changed_file)
        if event["test"] is not None:
            agent["last_test"] = dict(event["test"])
        if event["error"] is not None:
            agent["error"] = event["error"]
        if event.get("model_name") is not None:
            agent["model_name"] = event["model_name"]
        if event.get("reasoning_effort") is not None:
            agent["reasoning_effort"] = event["reasoning_effort"]
        if event.get("runtime_presence") is not None:
            observed_at = event["observed_at"]
            previous_observed_at = agent["observed_at"]
            if previous_observed_at is None or _parse_timestamp(
                observed_at, "observed_at"
            ) >= _parse_timestamp(previous_observed_at, "observed_at"):
                agent["runtime_presence"] = event["runtime_presence"]
                agent["observed_at"] = observed_at
        if (
            event["event_type"] == "test_finished"
            and event["test"] is not None
            and event["test"]["status"] == "passed"
            and event["error"] is None
        ):
            agent["error"] = None
        if event["event_type"] == "agent_created" and agent["started_at"] is None:
            agent["started_at"] = event["timestamp_utc"]
            agent["parent_id"] = event["parent_id"]
        if event["event_type"] in {
            "agent_completed",
            "agent_failed",
            "agent_interrupted",
        }:
            agent["completed_at"] = event["timestamp_utc"]
        if event["event_type"] == "agent_completed" or (
            event["event_type"] == "status_changed"
            and event["status"] == "completed"
        ):
            agent["error"] = None
        agent["timeline"].append(dict(event))

    for agent in agents.values():
        if agent["status"] == "failed":
            agent["record_group"] = "failed"
        elif agent["status"] == "interrupted":
            agent["record_group"] = "interrupted"
        elif agent["status"] == "completed":
            agent["record_group"] = "historical"
        elif agent["runtime_presence"] == "present":
            agent["record_group"] = "current"
        elif agent["runtime_presence"] == "not_present":
            agent["record_group"] = "historical"
        else:
            agent["record_group"] = UNAVAILABLE

    ordered_agents = sorted(
        agents.values(),
        key=lambda item: (
            item["started_at"] is None,
            _parse_timestamp(item["started_at"])
            if item["started_at"] is not None
            else datetime.max.replace(tzinfo=timezone.utc),
            item["agent_id"],
        ),
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "source_note": (
            "Codex 没有向本项目提供直接读取子 Agent 内部状态的 API；"
            "页面仅由项目内 validated agent event files 生成。runtime_presence "
            "只接受主 Agent 根据实际 collaboration 列表写入的观察值；缺失时显示 unavailable，"
            "不根据 working 等历史文字推断当前在线。"
        ),
        "agents": ordered_agents,
    }


__all__ = [
    "AGENT_STATUSES",
    "EVENT_FIELDS",
    "EVENT_TYPES",
    "LEGACY_SCHEMA_VERSION",
    "REASONING_EFFORTS",
    "RUNTIME_PRESENCES",
    "SCHEMA_VERSION",
    "TEST_STATUSES",
    "UNAVAILABLE",
    "V1_EVENT_FIELDS",
    "EventValidationError",
    "append_event",
    "create_event",
    "load_events",
    "reduce_snapshot",
    "utc_now",
    "validate_event",
]
