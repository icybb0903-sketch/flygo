"""Append one validated event to the local agent-monitor event store."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent_monitor import (
    AGENT_STATUSES,
    EVENT_TYPES,
    REASONING_EFFORTS,
    RUNTIME_PRESENCES,
    SCHEMA_VERSION,
    TEST_STATUSES,
    EventValidationError,
    append_event,
    create_event,
)


DEFAULT_EVENTS_DIR = PROJECT_ROOT / "data" / "agent_monitor" / "events"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Atomically append one schema-v2 agent monitor event. "
            "Use runtime_observed for collaboration presence snapshots."
        )
    )
    parser.add_argument("event_type", nargs="?", choices=sorted(EVENT_TYPES))
    parser.add_argument(
        "--event-type", dest="event_type_option", choices=sorted(EVENT_TYPES)
    )
    parser.add_argument("--schema-version", type=int, default=SCHEMA_VERSION)
    parser.add_argument("--event-id", help="canonical UUID; generated when omitted")
    parser.add_argument(
        "--timestamp-utc",
        help="ISO 8601 UTC (Z) timestamp; defaults to the current UTC time",
    )
    parser.add_argument("--source", default="agent_event_cli")
    parser.add_argument("--agent-id", required=True)
    parser.add_argument("--agent-name")
    parser.add_argument("--parent-id")
    parser.add_argument("--task")
    parser.add_argument("--status", choices=sorted(AGENT_STATUSES))
    parser.add_argument("--detail")
    parser.add_argument("--file", action="append", default=[], dest="files")
    parser.add_argument("--test-status", choices=sorted(TEST_STATUSES))
    parser.add_argument("--test-summary")
    parser.add_argument("--test-command")
    parser.add_argument("--error")
    parser.add_argument(
        "--model-name",
        help="exact model name reported by Codex, or 'unavailable'; never infer",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=sorted(REASONING_EFFORTS),
        help="reported reasoning effort, or unavailable",
    )
    parser.add_argument(
        "--runtime-presence",
        choices=sorted(RUNTIME_PRESENCES),
        help="presence from an actual collaboration-list observation",
    )
    parser.add_argument(
        "--observed-at",
        help="UTC time of the runtime presence observation; required with presence",
    )
    parser.add_argument("--events-dir", type=Path, default=DEFAULT_EVENTS_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.schema_version != SCHEMA_VERSION:
        parser.error(f"--schema-version must be {SCHEMA_VERSION}")
    if args.event_type and args.event_type_option:
        parser.error("provide event type either positionally or with --event-type, not both")
    event_type = args.event_type or args.event_type_option
    if event_type is None:
        parser.error("an event type is required")

    test_values = (args.test_status, args.test_summary, args.test_command)
    test = None
    if any(value is not None for value in test_values):
        if args.test_status is None:
            parser.error("--test-status is required when test fields are provided")
        test = {
            "status": args.test_status,
            "summary": args.test_summary,
            "command": args.test_command,
        }

    try:
        event = create_event(
            event_type,
            agent_id=args.agent_id,
            source=args.source,
            event_id=args.event_id,
            timestamp_utc=args.timestamp_utc,
            agent_name=args.agent_name,
            parent_id=args.parent_id,
            task=args.task,
            status=args.status,
            detail=args.detail,
            files=args.files,
            test=test,
            error=args.error,
            model_name=args.model_name,
            reasoning_effort=args.reasoning_effort,
            runtime_presence=args.runtime_presence,
            observed_at=args.observed_at,
        )
        path = append_event(args.events_dir, event)
    except (EventValidationError, OSError) as exc:
        print(f"error={exc}", file=sys.stderr)
        return 1

    print(f"event_id={event['event_id']}")
    print(f"path={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
