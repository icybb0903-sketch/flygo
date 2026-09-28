"""Launch the public, local-only flygo Go application."""

from __future__ import annotations

import argparse
from http import HTTPStatus
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

GET_ROUTES = frozenset({
    "/go", "/go.html", "/go.css", "/go.js",
    "/vendor/three.module.js", "/vendor/three.core.js", "/vendor/THREE_LICENSE.txt",
    "/api/neural/status", "/api/neural/anatomy/soma_xyz.npy",
    "/api/go/state", "/api/go/neural-policy-status", "/api/go/evaluation-status",
    "/api/go/tournament-status", "/api/go/stage7-status", "/api/go/stage8-status",
})
POST_ROUTES = frozenset({"/api/go/action", "/api/go/neural-move", "/api/neural/frame"})


def create_flygo_server(port: int = 8000, **options):
    """Reuse the verified Go backend but expose only Go routes."""
    from app.server import LabRequestHandler, create_server

    class FlygoRequestHandler(LabRequestHandler):
        def _validated_path(self):
            path = super()._validated_path()
            return "/go" if path in {"/", "/index.html"} else path

        def do_GET(self):
            path = self._validated_path()
            if path is None:
                return
            if path not in GET_ROUTES:
                self._send_error_json(
                    HTTPStatus.NOT_FOUND, "not_found",
                    "This flygo release exposes only Go resources.", close=True,
                )
                return
            super().do_GET()

        def do_POST(self):
            path = self._validated_path()
            if path is None:
                return
            if path not in POST_ROUTES:
                self._send_error_json(
                    HTTPStatus.NOT_FOUND, "not_found",
                    "This flygo release exposes only Go resources.", close=True,
                )
                return
            super().do_POST()

        def _handle_go_action(self, payload):
            if payload["action"] == "bot_move":
                self._send_error_json(
                    HTTPStatus.CONFLICT, "baseline_disabled",
                    "Use the verified neural-move endpoint; baseline moves are disabled.",
                )
                return
            super()._handle_go_action(payload)

    server = create_server(port, **options)
    server.RequestHandlerClass = FlygoRequestHandler
    return server


def main(argv=None) -> int:
    from scripts.run_lab import _port, _runtime_error

    parser = argparse.ArgumentParser(description="Run flygo locally (Go only).")
    parser.add_argument("--port", type=_port, default=8000)
    parser.add_argument("--checkpoint", type=Path, help="Optional trained checkpoint directory")
    args = parser.parse_args(argv)

    runtime_error = _runtime_error()
    if runtime_error:
        print(f"flygo: {runtime_error}", file=sys.stderr)
        return 1

    try:
        options = {} if args.checkpoint is None else {"policy_checkpoint_dir": args.checkpoint}
        server = create_flygo_server(args.port, **options)
    except (ImportError, OSError, ValueError) as error:
        print(f"flygo: startup failed: {error}", file=sys.stderr)
        return 1
    if not server.neural_policy_status()["available"]:
        status = server.neural_policy_status()
        server.server_close()
        print(f"flygo: verified neural policy unavailable: {status}", file=sys.stderr)
        return 1
    host, port = server.server_address
    print(f"flygo: http://{host}:{port}", flush=True)
    print("Local loopback only. Press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping flygo.", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
