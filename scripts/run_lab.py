"""Start the local-only P4 MaleCNS animation lab."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _expected_python() -> Path:
    windows = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    posix = PROJECT_ROOT / ".venv" / "bin" / "python"
    return windows if windows.exists() or not posix.exists() else posix


def _runtime_error() -> str | None:
    expected = _expected_python().resolve()
    if not expected.is_file():
        return (
            "project virtual environment is missing; create .venv and install "
            "requirements-phase3.txt before starting the neural API"
        )
    if Path(sys.executable).resolve() != expected:
        return (
            "Phase 3 neural API must use the project virtual environment; run "
            f'"{expected}" scripts/run_lab.py'
        )
    try:
        import numpy  # noqa: F401 - explicit runtime dependency check
    except ImportError:
        return (
            "NumPy is missing from the project virtual environment; run "
            f'"{expected}" -m pip install -r requirements-phase3.txt'
        )
    return None


def _port(value: str) -> int:
    try:
        port = int(value, 10)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port must be an integer") from exc
    if not 1 <= port <= 65_535:
        raise argparse.ArgumentTypeError("port must be in [1, 65535]")
    return port


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Serve the P4 animation on 127.0.0.1 without external dependencies."
    )
    parser.add_argument("--port", type=_port, default=8000, help="loopback port (default: 8000)")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="optional trained policy checkpoint directory for a local preview",
    )
    args = parser.parse_args()
    runtime_error = _runtime_error()
    if runtime_error is not None:
        print(f"P4_FAIL dependency_error: {runtime_error}", file=sys.stderr)
        return 1
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    try:
        from src.p1_graph import GraphDataError
        from app.server import serve
    except (ImportError, OSError, ValueError) as exc:
        print(f"P4_FAIL dependency_error={type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    try:
        if args.checkpoint is None:
            serve(args.port)
        else:
            serve(args.port, policy_checkpoint_dir=args.checkpoint)
    except (GraphDataError, OSError, ValueError) as exc:
        print(f"P4_FAIL error={type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
