"""Local-only HTTP server for the MaleCNS P4 animation lab.

The server deliberately exposes a very small surface: three static files, one
read-only status endpoint, and one bounded simulation endpoint.  All graph and
dynamics data come from the verified P1/P2 Python modules.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Final
from urllib.parse import unquote, urlsplit

from src.agent_monitor import EventValidationError, load_events, reduce_snapshot
from src.go_engine import IllegalMoveError
from src.go_neural_encoding import encode_go_state
from src.go_session import GoSession, SessionTurnError, StaleSessionError
from src.malecns_assets import load_runtime_assets
from src.malecns_dynamics import (
    DISCLAIMER as NEURAL_DISCLAIMER,
    DynamicsError as NeuralDynamicsError,
    GraphArrays,
    simulate_go_encoding,
)
from src.malecns_policy import (
    CheckpointValidationError,
    PolicyCheckpoint,
    PolicyError,
    load_policy_checkpoint,
    select_action,
)
from src.p1_graph import GraphDataError, VerifiedGraph, load_verified_graph
from src.toy_dynamics import MODEL_DISCLAIMER, MODEL_NAME, DynamicsError, simulate


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = PROJECT_ROOT / "app"
P1_SUBGRAPH = PROJECT_ROOT / "data" / "p1" / "subgraph.json"
P1_PROVENANCE = PROJECT_ROOT / "data" / "p1" / "provenance.json"
AGENT_EVENTS_DIR = PROJECT_ROOT / "data" / "agent_monitor" / "events"
DEFAULT_NEURAL_MODEL_DIR = (
    PROJECT_ROOT
    / "data"
    / "malecns"
    / "runtime"
    / "malecns-v1.0-w5-3acb6434e71160fc"
)
DEFAULT_POLICY_CHECKPOINT_DIR = PROJECT_ROOT / "data" / "checkpoints" / "stage4"
STAGE5_EVALUATION_REPORT = (
    PROJECT_ROOT / "data" / "evaluation" / "stage6" / "balanced_candidate_unseen.json"
)
STAGE6_TOURNAMENT_REPORT = (
    PROJECT_ROOT / "data" / "evaluation" / "stage6" / "balanced_candidate_report.json"
)
STAGE7_DEVELOPMENT_REPORT = (
    PROJECT_ROOT / "data" / "evaluation" / "stage7" / "white_20_games.json"
)
STAGE8_UNSEEN_REPORT = PROJECT_ROOT / "data" / "evaluation" / "stage8" / "hybrid_pilot_unseen.json"
STAGE8_BOTH_REPORT = PROJECT_ROOT / "data" / "evaluation" / "stage8" / "hybrid_pilot_both_smoke.json"
STAGE8_WHITE_REPORT = PROJECT_ROOT / "data" / "evaluation" / "stage8" / "hybrid_pilot_white_8.json"
STAGE8_ABLATION_REPORT = PROJECT_ROOT / "data" / "evaluation" / "stage8" / "hybrid_pilot_signal_ablation.json"
STAGE8_ZERO_GAMES_REPORT = PROJECT_ROOT / "data" / "evaluation" / "stage8" / "hybrid_pilot_zero_signal_white_8.json"
STAGE8_TRAINING_CACHE = PROJECT_ROOT / "data" / "training" / "stage8_role_symmetric_pilot"
MAX_REQUEST_BYTES = 1_024
MAX_NEURAL_SEED = 2**63 - 1
NEURAL_TIMEOUT_SECONDS = 10.0
NEURAL_STAGE_ROLE: Final = "observational_frame_only_not_go_controller"
_AUTO_LOAD_NEURAL: Final = object()
_AUTO_LOAD_POLICY: Final = object()

STATIC_ALLOWLIST = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/monitor": ("monitor.html", "text/html; charset=utf-8"),
    "/monitor.html": ("monitor.html", "text/html; charset=utf-8"),
    "/monitor.css": ("monitor.css", "text/css; charset=utf-8"),
    "/monitor.js": ("monitor.js", "text/javascript; charset=utf-8"),
    "/go": ("go.html", "text/html; charset=utf-8"),
    "/go.html": ("go.html", "text/html; charset=utf-8"),
    "/go.css": ("go.css", "text/css; charset=utf-8"),
    "/go.js": ("go.js", "text/javascript; charset=utf-8"),
    "/vendor/three.module.js": (
        "vendor/three.module.js",
        "text/javascript; charset=utf-8",
    ),
    "/vendor/three.core.js": (
        "vendor/three.core.js",
        "text/javascript; charset=utf-8",
    ),
    "/vendor/THREE_LICENSE.txt": (
        "vendor/THREE_LICENSE.txt",
        "text/plain; charset=utf-8",
    ),
}

SCENARIOS: dict[str, tuple[int, ...]] = {
    "none": (),
    "left": (12781,),
    "right": (556329,),
    "both": (12781, 556329),
}

SCENARIO_LABELS = {
    "none": "无刺激",
    "left": "刺激左源（工程标签：12781）",
    "right": "刺激右源（工程标签：556329）",
    "both": "刺激双源（工程标签：12781 + 556329）",
}

JSON_CONTENT_TYPE = re.compile(
    r"^application/json(?:\s*;\s*charset\s*=\s*utf-8)?$", re.IGNORECASE
)


def _load_hashed_report(path: Path, schema: str) -> dict[str, Any]:
    """Load and hash-check one persisted evaluation before exposing it."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != schema:
        raise ValueError(f"unsupported evaluation schema; expected {schema}")
    claimed = payload.get("report_sha256")
    unsigned = dict(payload)
    unsigned.pop("report_sha256", None)
    encoded = json.dumps(
        unsigned,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if not isinstance(claimed, str) or hashlib.sha256(encoded).hexdigest() != claimed:
        raise ValueError("evaluation report hash mismatch")
    return payload


def _node_payload(graph: VerifiedGraph) -> list[dict[str, Any]]:
    node_types = dict(graph.node_types)
    sources = set(graph.source_nodes)
    engineering_sides = {12781: "left", 556329: "right"}
    return [
        {
            "id": node,
            "type": node_types[node],
            "role": "source" if node in sources else "downstream",
            "engineering_side": engineering_sides.get(node),
        }
        for node in graph.nodes
    ]


def _topology_payload(graph: VerifiedGraph) -> dict[str, Any]:
    return {
        "dataset": "male-cns:v1.0",
        "node_count": len(graph.nodes),
        "edge_count": len(graph.edges),
        "nodes": _node_payload(graph),
        "edges": [
            {
                "pre": edge.pre,
                "post": edge.post,
                "weight": edge.weight,
                "pre_type": edge.pre_type,
                "post_type": edge.post_type,
            }
            for edge in graph.edges
        ],
    }


def _common_payload(graph: VerifiedGraph) -> dict[str, Any]:
    return {
        "model": MODEL_NAME,
        "model_disclaimer": MODEL_DISCLAIMER,
        "p1_canonical_sha256": graph.canonical_sha256,
    }


def _strict_load_neural_graph(model_dir: Path) -> GraphArrays:
    """Load the canonical runtime contract, then adapt it without copying."""
    assets = load_runtime_assets(model_dir, verify_hashes=True)
    graph = GraphArrays(
        model_id=assets.model_id,
        manifest_sha256=assets.manifest_sha256,
        node_ids=assets.node_ids,
        soma_xyz=assets.soma_xyz,
        indptr=assets.indptr,
        indices=assets.indices,
        weights=assets.weights,
        nt_code=assets.nt_code,
    )
    graph.validate()
    return graph


class LabServer(ThreadingHTTPServer):
    """Threaded loopback server holding one already-verified graph."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        graph: VerifiedGraph,
        events_dir: Path,
        *,
        neural_graph: GraphArrays | None,
        neural_model_dir: Path,
        neural_load_error: str | None,
        policy_checkpoint: PolicyCheckpoint | None,
        policy_checkpoint_dir: Path,
        policy_load_error: str | None,
    ) -> None:
        super().__init__(server_address, LabRequestHandler)
        self.graph = graph
        self.events_dir = events_dir
        self.go_session = GoSession()
        self.neural_graph = neural_graph
        self.neural_model_dir = neural_model_dir
        self.neural_load_error = neural_load_error
        self.policy_checkpoint = policy_checkpoint
        self.policy_checkpoint_dir = policy_checkpoint_dir
        self.policy_load_error = policy_load_error
        # A full-graph propagation is intentionally single-flight.  Request
        # threads never queue behind this lock: a concurrent request gets a
        # bounded, explicit busy response instead.
        self.neural_frame_lock = threading.Lock()

    def neural_status(self) -> dict[str, Any]:
        graph = self.neural_graph
        available = graph is not None
        model_id = graph.model_id if graph is not None else None
        manifest_sha256 = graph.manifest_sha256 if graph is not None else None
        node_count = graph.node_count if graph is not None else 0
        edge_count = graph.edge_count if graph is not None else 0
        return {
            "available": available,
            "state": "available" if available else "error",
            "model_id": model_id,
            "manifest_sha256": manifest_sha256,
            "controller_node_count": node_count,
            "controller_edge_count": edge_count,
            "model_directory_name": self.neural_model_dir.name,
            "model": {"id": model_id, "directory_name": self.neural_model_dir.name},
            "manifest": {"sha256": manifest_sha256, "strictly_verified": available},
            "controller": {"node_count": node_count, "edge_count": edge_count},
            "load_error": self.neural_load_error,
            "simulation_timeout_seconds": NEURAL_TIMEOUT_SECONDS,
            "max_concurrent_propagations": 1,
            "disclaimer": NEURAL_DISCLAIMER,
            "stage3_role": NEURAL_STAGE_ROLE,
            "stage3_controls_go_moves": False,
            "go_controller": "baseline",
            "go_move_selected": False,
            "baseline_controller_called": False,
        }

    def neural_policy_status(self) -> dict[str, Any]:
        checkpoint = self.policy_checkpoint
        graph = self.neural_graph
        available = checkpoint is not None and graph is not None
        return {
            "available": available,
            "state": "available" if available else "error",
            "controller": "malecns-neural-policy",
            "stage4_controls_go_moves": available,
            "checkpoint_hash": checkpoint.checkpoint_hash if checkpoint else None,
            "training_status": checkpoint.training_status if checkpoint else None,
            "training_info": dict(checkpoint.training_info) if checkpoint else None,
            "model_id": checkpoint.model_id if checkpoint else None,
            "graph_manifest_sha256": (
                checkpoint.graph_manifest_sha256 if checkpoint else None
            ),
            "output_pool_group_hash": (
                checkpoint.output_pool_group_hash if checkpoint else None
            ),
            "checkpoint_directory_name": self.policy_checkpoint_dir.name,
            "load_error": self.policy_load_error,
            "teacher_accessed_at_runtime": False,
            "baseline_controller_called": False,
            "fallback_used": False,
            "disclaimer": (
                "A weak engineered linear readout over a simulated MaleCNS frame; "
                "not biological cognition and not evidence of strong Go play."
            ),
        }


class LabRequestHandler(BaseHTTPRequestHandler):
    """Serve the fixed P4 application and its two API routes."""

    protocol_version = "HTTP/1.1"
    server: LabServer

    def log_message(self, format: str, *args: Any) -> None:
        # Avoid echoing attacker-controlled paths or request bodies to terminals.
        return

    def _security_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'none'; script-src 'self'; style-src 'self'; "
            "connect-src 'self'; img-src 'self' data:; font-src 'none'; "
            "object-src 'none'; base-uri 'none'; form-action 'none'; "
            "frame-ancestors 'none'",
        )

    def _send_bytes(
        self,
        status: HTTPStatus,
        body: bytes,
        content_type: str,
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self._security_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if extra_headers:
            for name, value in extra_headers.items():
                self.send_header(name, value)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            # A browser may deliberately cancel an obsolete neural-frame
            # request when the board changes.  The computation remains valid,
            # but there is no client left to receive it; do not turn that
            # ordinary cancellation into a noisy server traceback.
            self.close_connection = True

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        self._send_bytes(status, body, "application/json; charset=utf-8")

    def _send_error_json(
        self, status: HTTPStatus, code: str, message: str, *, close: bool = False
    ) -> None:
        payload = {
            "ok": False,
            **_common_payload(self.server.graph),
            "error": {"code": code, "message": message},
        }
        headers = {"Connection": "close"} if close else None
        if close:
            self.close_connection = True
        body = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        self._send_bytes(
            status, body, "application/json; charset=utf-8", extra_headers=headers
        )

    def _validated_path(self) -> str | None:
        split = urlsplit(self.path)
        if split.query or split.fragment:
            self._send_error_json(
                HTTPStatus.BAD_REQUEST, "invalid_path", "Query strings are not accepted."
            )
            return None
        try:
            decoded = unquote(split.path, errors="strict")
        except UnicodeError:
            self._send_error_json(
                HTTPStatus.BAD_REQUEST, "invalid_path", "Path encoding is invalid."
            )
            return None
        segments = decoded.replace("\\", "/").split("/")
        if "\x00" in decoded or any(segment in {".", ".."} for segment in segments):
            self._send_error_json(
                HTTPStatus.BAD_REQUEST, "invalid_path", "Path traversal is rejected."
            )
            return None
        return decoded

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        path = self._validated_path()
        if path is None:
            return
        if path == "/api/status":
            payload = {
                "ok": True,
                **_common_payload(self.server.graph),
                "scenario_labels": SCENARIO_LABELS,
                "scenario_source_nodes": {
                    key: list(nodes) for key, nodes in SCENARIOS.items()
                },
                "topology": _topology_payload(self.server.graph),
            }
            self._send_json(HTTPStatus.OK, payload)
            return
        if path == "/api/neural/status":
            self._send_json(
                HTTPStatus.OK,
                {"ok": True, **self.server.neural_status()},
            )
            return
        if path == "/api/neural/anatomy/soma_xyz.npy":
            if self.server.neural_graph is None:
                self._send_error_json(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "neural_unavailable",
                    "The verified MaleCNS anatomy is unavailable.",
                )
                return
            try:
                body = (self.server.neural_model_dir / "soma_xyz.npy").read_bytes()
            except OSError:
                self._send_error_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    "anatomy_unavailable",
                    "The verified MaleCNS soma coordinate asset could not be read.",
                )
                return
            self._send_bytes(
                HTTPStatus.OK,
                body,
                "application/octet-stream",
                extra_headers={
                    "X-MaleCNS-Model-ID": self.server.neural_graph.model_id,
                    "X-MaleCNS-Node-Count": str(self.server.neural_graph.node_count),
                },
            )
            return
        if path == "/api/go/neural-policy-status":
            self._send_json(
                HTTPStatus.OK,
                {"ok": True, **self.server.neural_policy_status()},
            )
            return
        if path == "/api/go/evaluation-status":
            if not STAGE5_EVALUATION_REPORT.is_file():
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "ok": True,
                        "available": False,
                        "state": "not_run",
                        "message": "No persisted Stage 5 evaluation report is available.",
                    },
                )
                return
            try:
                evaluation = _load_hashed_report(
                    STAGE5_EVALUATION_REPORT,
                    "malecns-go-stage5-evaluation-v1",
                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                self._send_error_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    "evaluation_invalid",
                    f"The persisted Stage 5 evaluation report failed validation: {exc}",
                )
                return
            loaded_checkpoint = self.server.policy_checkpoint
            evaluated_hash = evaluation.get("checkpoint", {}).get("hash")
            current_hash = loaded_checkpoint.checkpoint_hash if loaded_checkpoint else None
            self._send_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "available": True,
                    "state": "current" if evaluated_hash == current_hash else "stale",
                    "matches_loaded_checkpoint": evaluated_hash == current_hash,
                    "evaluation": evaluation,
                },
            )
            return
        if path == "/api/go/tournament-status":
            if not STAGE6_TOURNAMENT_REPORT.is_file():
                self._send_json(
                    HTTPStatus.OK,
                    {"ok": True, "available": False, "state": "not_run"},
                )
                return
            try:
                tournament = _load_hashed_report(
                    STAGE6_TOURNAMENT_REPORT,
                    "malecns-go-stage6-tournament-v1",
                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                self._send_error_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    "tournament_invalid",
                    f"The persisted Stage 6 tournament report failed validation: {exc}",
                )
                return
            loaded_checkpoint = self.server.policy_checkpoint
            evaluated_hash = tournament.get("checkpoint", {}).get("hash")
            current_hash = loaded_checkpoint.checkpoint_hash if loaded_checkpoint else None
            self._send_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "available": True,
                    "state": "current" if evaluated_hash == current_hash else "stale",
                    "matches_loaded_checkpoint": evaluated_hash == current_hash,
                    "tournament": tournament,
                },
            )
            return
        if path == "/api/go/stage7-status":
            if not STAGE7_DEVELOPMENT_REPORT.is_file():
                self._send_json(
                    HTTPStatus.OK,
                    {"ok": True, "available": False, "state": "not_run"},
                )
                return
            try:
                development = _load_hashed_report(
                    STAGE7_DEVELOPMENT_REPORT,
                    "malecns-go-stage7-production-white-v1",
                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                self._send_error_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    "stage7_development_invalid",
                    f"The persisted Stage 7 development report failed validation: {exc}",
                )
                return
            loaded_checkpoint = self.server.policy_checkpoint
            evaluated_hash = development.get("checkpoint", {}).get("hash")
            current_hash = loaded_checkpoint.checkpoint_hash if loaded_checkpoint else None
            self._send_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "available": True,
                    "state": "current" if evaluated_hash == current_hash else "stale",
                    "matches_loaded_checkpoint": evaluated_hash == current_hash,
                    "classification": "development_only_not_final",
                    "development": development,
                },
            )
            return
        if path == "/api/go/stage8-status":
            sources = (STAGE8_UNSEEN_REPORT, STAGE8_BOTH_REPORT, STAGE8_WHITE_REPORT)
            if not all(item.is_file() for item in sources):
                self._send_json(
                    HTTPStatus.OK, {"ok": True, "available": False, "state": "not_run"}
                )
                return
            try:
                from scripts.train_neural_go_readout import load_cache

                _, _, training_records, dataset = load_cache(STAGE8_TRAINING_CACHE)
                unseen = _load_hashed_report(
                    STAGE8_UNSEEN_REPORT, "malecns-go-stage5-evaluation-v1"
                )
                both = _load_hashed_report(
                    STAGE8_BOTH_REPORT, "malecns-go-stage6-games-v4"
                )
                white = _load_hashed_report(
                    STAGE8_WHITE_REPORT, "malecns-go-stage7-production-white-v2"
                )
                dataset_hash = dataset["dataset_hash"]
                if any(
                    report["checkpoint"]["training_dataset_hash"] != dataset_hash
                    for report in (unseen, both, white)
                ):
                    raise ValueError("Stage 8 reports do not match the training cache")
                training_states = {
                    item["hashes"]["state_sha256"] for item in training_records
                }
                unseen_records = unseen["records"]
                fresh = [
                    item for item in unseen_records
                    if item["state_sha256"] not in training_states
                ]
                if not fresh:
                    raise ValueError("Stage 8 unseen report has no independent states")
                for report in (both, white):
                    neural_moves = [
                        move for game in report["games"] for move in game["moves"]
                        if move["actor"] == "malecns-neural-policy"
                    ]
                    if any(
                        not isinstance(move.get("state_sha256_before"), str)
                        or move["state_sha256_before"] in training_states
                        for move in neural_moves
                    ):
                        raise ValueError("Stage 8 game report has missing or overlapping neural states")
                report_hashes = {
                    report["checkpoint"]["hash"] for report in (unseen, both, white)
                }
                if len(report_hashes) != 1:
                    raise ValueError("Stage 8 reports use different checkpoints")
                ablation = None
                if STAGE8_ABLATION_REPORT.is_file():
                    ablation = _load_hashed_report(
                        STAGE8_ABLATION_REPORT,
                        "malecns-go-neural-signal-ablation-v1",
                    )
                    if (
                        ablation.get("dataset_hash") != dataset_hash
                        or ablation.get("checkpoint_hash") not in report_hashes
                        or ablation.get("selection", {}).get("split") != "validation"
                        or ablation.get("selection", {}).get("supervision_kind")
                        != "natural-teacher-turn"
                    ):
                        raise ValueError("Stage 8 ablation does not match checkpoint or held-out cache")
                zero_games = None
                if STAGE8_ZERO_GAMES_REPORT.is_file():
                    zero_games = _load_hashed_report(
                        STAGE8_ZERO_GAMES_REPORT,
                        "malecns-go-stage8-zero-signal-games-v1",
                    )
                    if (
                        zero_games.get("checkpoint_hash") not in report_hashes
                        or zero_games.get("reference_report_sha256") != white["report_sha256"]
                        or zero_games.get("configuration", {}).get("same_opening_board_hashes") is not True
                        or zero_games.get("configuration", {}).get("neural_runtime_called_for_control") is not False
                        or zero_games.get("metrics", {}).get("game_count") != len(white["games"])
                    ):
                        raise ValueError("Stage 8 zero-signal games do not match White reference")
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                self._send_error_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    "stage8_development_invalid",
                    f"The persisted Stage 8 reports failed validation: {exc}",
                )
                return
            current_hash = (
                self.server.policy_checkpoint.checkpoint_hash
                if self.server.policy_checkpoint else None
            )
            matches = current_hash in report_hashes
            fresh_matches = sum(bool(item["teacher_match"]) for item in fresh)
            random_rate = sum(
                float(item["random_legal_match_probability"]) for item in fresh
            ) / len(fresh)
            self._send_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "available": True,
                    "state": "current" if matches else "stale",
                    "matches_loaded_checkpoint": matches,
                    "classification": "development_candidate_not_final",
                    "checkpoint_hash": next(iter(report_hashes)),
                    "unseen": {
                        "report_sha256": unseen["report_sha256"],
                        "sample_count": len(fresh),
                        "excluded_training_state_count": len(unseen_records) - len(fresh),
                        "teacher_matches": fresh_matches,
                        "teacher_agreement_rate": fresh_matches / len(fresh),
                        "random_legal_expected_rate": random_rate,
                    },
                    "both": {
                        "report_sha256": both["report_sha256"],
                        "metrics": both["metrics"],
                    },
                    "white": {
                        "report_sha256": white["report_sha256"],
                        "metrics": white["metrics"],
                    },
                    "signal_ablation": (
                        {
                            "available": True,
                            "report_sha256": ablation["report_sha256"],
                            "sample_count": ablation["selection"]["sample_count"],
                            "intact_agreement_rate": ablation["results"]["intact_agreement_rate"],
                            "zero_agreement_rate": ablation["results"]["zero_agreement_rate"],
                            "shuffled_mean_agreement_rate": ablation["results"]["shuffled_mean_agreement_rate"],
                            "empirical_tail_probability_shuffled_ge_intact": ablation["results"]["empirical_tail_probability_shuffled_ge_intact"],
                            "limitation": ablation["interpretation_limit"],
                        }
                        if ablation is not None else {"available": False}
                    ),
                    "zero_signal_games": (
                        {
                            "available": True,
                            "report_sha256": zero_games["report_sha256"],
                            "game_count": zero_games["metrics"]["game_count"],
                            "neural_white_wins": zero_games["metrics"]["neural_reference_white_wins"],
                            "zero_white_wins": zero_games["metrics"]["zero_control_white_wins"],
                            "neural_natural_finishes": zero_games["metrics"]["neural_reference_natural_finishes"],
                            "zero_natural_finishes": zero_games["metrics"]["zero_control_natural_finishes"],
                            "limitation": zero_games["limitation"],
                        }
                        if zero_games is not None else {"available": False}
                    ),
                },
            )
            return
        if path == "/api/agents":
            try:
                snapshot = reduce_snapshot(load_events(self.server.events_dir))
            except (EventValidationError, OSError):
                self._send_error_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    "agent_events_unavailable",
                    "The validated agent event log could not be read.",
                )
                return
            self._send_json(HTTPStatus.OK, {"ok": True, **snapshot})
            return
        if path == "/api/go/state":
            self._send_json(
                HTTPStatus.OK,
                {"ok": True, "session": self._go_api_snapshot()},
            )
            return
        static = STATIC_ALLOWLIST.get(path)
        if static is None:
            self._send_error_json(
                HTTPStatus.NOT_FOUND, "not_found", "Resource is not on the allowlist."
            )
            return
        filename, content_type = static
        try:
            body = (STATIC_ROOT / filename).read_bytes()
        except OSError:
            self._send_error_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "static_unavailable",
                "A required application asset is unavailable.",
            )
            return
        self._send_bytes(HTTPStatus.OK, body, content_type)

    def _read_simulation_request(self) -> dict[str, Any] | None:
        content_type = self.headers.get("Content-Type", "")
        if not JSON_CONTENT_TYPE.fullmatch(content_type.strip()):
            self._send_error_json(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                "invalid_content_type",
                "Content-Type must be application/json with optional UTF-8 charset.",
                close=True,
            )
            return None

        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            self._send_error_json(
                HTTPStatus.LENGTH_REQUIRED,
                "content_length_required",
                "Content-Length is required.",
                close=True,
            )
            return None
        try:
            length = int(raw_length, 10)
        except ValueError:
            length = -1
        if length < 1:
            self._send_error_json(
                HTTPStatus.BAD_REQUEST,
                "invalid_content_length",
                "Content-Length must describe a non-empty body.",
                close=True,
            )
            return None
        if length > MAX_REQUEST_BYTES:
            self._send_error_json(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                "request_too_large",
                f"Request body exceeds {MAX_REQUEST_BYTES} bytes.",
                close=True,
            )
            return None

        body = self.rfile.read(length)
        if len(body) != length:
            self._send_error_json(
                HTTPStatus.BAD_REQUEST,
                "incomplete_body",
                "Request body ended before Content-Length bytes were received.",
                close=True,
            )
            return None
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_error_json(
                HTTPStatus.BAD_REQUEST, "invalid_json", "Body must be valid UTF-8 JSON."
            )
            return None

        required_fields = {"scenario", "edges_enabled"}
        if type(payload) is not dict or set(payload) != required_fields:
            self._send_error_json(
                HTTPStatus.BAD_REQUEST,
                "invalid_fields",
                "JSON must contain exactly scenario and edges_enabled.",
            )
            return None
        if type(payload["scenario"]) is not str or payload["scenario"] not in SCENARIOS:
            self._send_error_json(
                HTTPStatus.BAD_REQUEST,
                "invalid_scenario",
                "scenario must be one of: none, left, right, both.",
            )
            return None
        if type(payload["edges_enabled"]) is not bool:
            self._send_error_json(
                HTTPStatus.BAD_REQUEST,
                "invalid_edges_enabled",
                "edges_enabled must be a JSON boolean.",
            )
            return None
        return payload

    def _read_go_action_request(self) -> dict[str, Any] | None:
        content_type = self.headers.get("Content-Type", "")
        if not JSON_CONTENT_TYPE.fullmatch(content_type.strip()):
            self._send_error_json(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                "invalid_content_type",
                "Content-Type must be application/json with optional UTF-8 charset.",
                close=True,
            )
            return None

        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            self._send_error_json(
                HTTPStatus.LENGTH_REQUIRED,
                "content_length_required",
                "Content-Length is required.",
                close=True,
            )
            return None
        try:
            length = int(raw_length, 10)
        except ValueError:
            length = -1
        if length < 1:
            self._send_error_json(
                HTTPStatus.BAD_REQUEST,
                "invalid_content_length",
                "Content-Length must describe a non-empty body.",
                close=True,
            )
            return None
        if length > MAX_REQUEST_BYTES:
            self._send_error_json(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                "request_too_large",
                f"Request body exceeds {MAX_REQUEST_BYTES} bytes.",
                close=True,
            )
            return None
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_error_json(
                HTTPStatus.BAD_REQUEST,
                "invalid_json",
                "Body must be valid UTF-8 JSON.",
            )
            return None
        if type(payload) is not dict or type(payload.get("action")) is not str:
            self._send_error_json(
                HTTPStatus.BAD_REQUEST,
                "invalid_go_action",
                "Go action must be a JSON object with a string action.",
            )
            return None

        action = payload["action"]
        if action in {"reset", "undo", "bot_move", "human_pass", "human_resign"}:
            if set(payload) != {"action"}:
                self._send_error_json(
                    HTTPStatus.BAD_REQUEST,
                    "invalid_go_fields",
                    f"{action} accepts only the action field.",
                )
                return None
            return payload
        if action == "human_move":
            if set(payload) != {"action", "row", "col"}:
                self._send_error_json(
                    HTTPStatus.BAD_REQUEST,
                    "invalid_go_fields",
                    "human_move requires exactly action, row, and col.",
                )
                return None
            row, col = payload["row"], payload["col"]
            if (
                type(row) is not int
                or type(col) is not int
                or not 0 <= row < 9
                or not 0 <= col < 9
            ):
                self._send_error_json(
                    HTTPStatus.BAD_REQUEST,
                    "invalid_coordinate",
                    "row and col must be integers from 0 through 8.",
                )
                return None
            return payload

        self._send_error_json(
            HTTPStatus.BAD_REQUEST,
            "invalid_go_action",
            "action must be human_move, human_pass, human_resign, bot_move, undo, or reset.",
        )
        return None

    def _read_neural_frame_request(self) -> dict[str, Any] | None:
        """Read the deliberately tiny neural-frame request contract."""
        content_type = self.headers.get("Content-Type", "")
        if not JSON_CONTENT_TYPE.fullmatch(content_type.strip()):
            self._send_error_json(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                "invalid_content_type",
                "Content-Type must be application/json with optional UTF-8 charset.",
                close=True,
            )
            return None

        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            self._send_error_json(
                HTTPStatus.LENGTH_REQUIRED,
                "content_length_required",
                "Content-Length is required.",
                close=True,
            )
            return None
        try:
            length = int(raw_length, 10)
        except ValueError:
            length = -1
        if length < 1:
            self._send_error_json(
                HTTPStatus.BAD_REQUEST,
                "invalid_content_length",
                "Content-Length must describe a non-empty body.",
                close=True,
            )
            return None
        if length > MAX_REQUEST_BYTES:
            self._send_error_json(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                "request_too_large",
                f"Request body exceeds {MAX_REQUEST_BYTES} bytes.",
                close=True,
            )
            return None
        raw = self.rfile.read(length)
        if len(raw) != length:
            self._send_error_json(
                HTTPStatus.BAD_REQUEST,
                "incomplete_body",
                "Request body ended before Content-Length bytes were received.",
                close=True,
            )
            return None
        def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            value: dict[str, Any] = {}
            for key, item in pairs:
                if key in value:
                    raise ValueError(f"duplicate JSON field: {key}")
                value[key] = item
            return value

        try:
            payload = json.loads(
                raw.decode("utf-8"), object_pairs_hook=unique_object
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            self._send_error_json(
                HTTPStatus.BAD_REQUEST,
                "invalid_json",
                "Body must be valid UTF-8 JSON.",
            )
            return None
        if type(payload) is not dict or not set(payload).issubset({"seed"}):
            self._send_error_json(
                HTTPStatus.BAD_REQUEST,
                "invalid_neural_fields",
                "JSON must be an object containing only the optional seed field.",
            )
            return None
        seed = payload.get("seed", 0)
        if type(seed) is not int or not 0 <= seed <= MAX_NEURAL_SEED:
            self._send_error_json(
                HTTPStatus.BAD_REQUEST,
                "invalid_neural_seed",
                f"seed must be an integer from 0 through {MAX_NEURAL_SEED}.",
            )
            return None
        return {"seed": seed}

    def _go_api_snapshot(self) -> dict[str, Any]:
        snapshot = self.server.go_session.snapshot()
        board_payload = json.dumps(
            snapshot["board"], ensure_ascii=True, separators=(",", ":")
        ).encode("ascii")
        move_number = snapshot["move_number"]
        result = snapshot.get("result") or {}
        winner_code = result.get("winner_code") if snapshot["game_over"] else None
        winner = {0: "draw", 1: "black", 2: "white"}.get(winner_code)
        last_decision = snapshot.get("strategy", {}).get("last_decision")
        last_controller = snapshot.get("strategy", {}).get("last_controller")
        controller = last_controller or "baseline"
        checkpoint = (
            last_decision.get("checkpoint_hash")
            if isinstance(last_decision, dict)
            else "not-applicable-stage2-baseline"
        )
        return {
            **snapshot,
            "to_move": snapshot["to_play"],
            "human_color": "black",
            "can_undo": snapshot["history_length"] > 0 or snapshot.get("end_reason") == "resignation",
            "decision_id": f"stage2-move-{move_number:04d}",
            "board_hash": hashlib.sha256(board_payload).hexdigest(),
            "controller": controller,
            "checkpoint": checkpoint,
            "winner": winner,
        }

    def _handle_go_action(self, payload: dict[str, Any]) -> None:
        action = payload["action"]
        try:
            if action == "reset":
                self.server.go_session.reset()
            elif action == "undo":
                self.server.go_session.undo_round()
            elif action == "human_move":
                self.server.go_session.human_move((payload["row"], payload["col"]))
            elif action == "human_pass":
                self.server.go_session.human_move(None)
            elif action == "human_resign":
                self.server.go_session.human_resign()
            else:
                self.server.go_session.bot_move()
        except IllegalMoveError as exc:
            self._send_error_json(
                HTTPStatus.CONFLICT,
                f"illegal_move_{exc.reason}",
                str(exc),
            )
            return
        except SessionTurnError as exc:
            self._send_error_json(
                HTTPStatus.CONFLICT,
                "wrong_turn",
                str(exc),
            )
            return
        self._send_json(
            HTTPStatus.OK,
            {"ok": True, "action": action, "session": self._go_api_snapshot()},
        )

    def _handle_neural_frame(self, payload: dict[str, Any]) -> None:
        graph = self.server.neural_graph
        if graph is None:
            self._send_error_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "neural_unavailable",
                "The verified MaleCNS runtime graph is unavailable; no frame was produced.",
            )
            return
        if not self.server.neural_frame_lock.acquire(blocking=False):
            self._send_error_json(
                HTTPStatus.CONFLICT,
                "neural_busy",
                "A full-graph propagation is already running; retry later.",
            )
            return
        try:
            # GoState is immutable.  This is a coherent snapshot even if a Go
            # action occurs after the property returns.
            encoding = encode_go_state(self.server.go_session.state)
            frame = simulate_go_encoding(
                graph,
                encoding,
                seed=payload["seed"],
                timeout_seconds=NEURAL_TIMEOUT_SECONDS,
            )
            board_hash = encoding["board_hash"]
            encoding_hash = encoding["encoding_hash"]
            if (
                frame.get("board_hash") != board_hash
                or frame.get("encoding_hash") != encoding_hash
                or frame.get("model_id") != graph.model_id
                or frame.get("manifest_sha256") != graph.manifest_sha256
            ):
                raise NeuralDynamicsError("neural frame provenance mismatch")
            decision_id = f"stage3-neural-frame-{frame['frame_id']}"
            frame["decision_id"] = decision_id
        except Exception:
            # Fail closed: never substitute the Stage 2 baseline or a toy
            # frame after any encoding/dynamics/provenance failure.
            self._send_error_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "neural_frame_failed",
                "The bounded neural propagation failed; no frame or move decision was produced.",
            )
            return
        finally:
            self.server.neural_frame_lock.release()

        self._send_json(
            HTTPStatus.OK,
            {
                "ok": True,
                "decision_id": decision_id,
                "board_hash": board_hash,
                "encoding_hash": encoding_hash,
                "seed": payload["seed"],
                "controller": "malecns-engineered-lif-observer",
                "stage3_role": NEURAL_STAGE_ROLE,
                "stage3_controls_go_moves": False,
                "go_controller": "baseline",
                "go_move_selected": False,
                "baseline_controller_called": False,
                "encoding": encoding,
                "frame": frame,
            },
        )

    def _handle_neural_move(self, payload: dict[str, Any]) -> None:
        graph = self.server.neural_graph
        checkpoint = self.server.policy_checkpoint
        if graph is None or checkpoint is None:
            self._send_error_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "neural_policy_unavailable",
                "The verified graph or trained policy checkpoint is unavailable; no move was made.",
            )
            return
        if not self.server.neural_frame_lock.acquire(blocking=False):
            self._send_error_json(
                HTTPStatus.CONFLICT,
                "neural_busy",
                "A full-graph propagation is already running; retry later.",
            )
            return

        try:
            expected_state = self.server.go_session.external_policy_state()
            encoding = encode_go_state(expected_state)
            frame = simulate_go_encoding(
                graph,
                encoding,
                seed=payload["seed"],
                timeout_seconds=NEURAL_TIMEOUT_SECONDS,
            )
            decision = select_action(frame, encoding, checkpoint)
            if (
                decision.get("baseline_controller_called") is not False
                or decision.get("fallback_used") is not False
                or decision.get("frame_id") != frame.get("frame_id")
            ):
                raise PolicyError("neural decision provenance mismatch")
            raw_move = decision["move"]
            move = None if raw_move == "pass" else tuple(raw_move)
            provenance = {
                "controller": "malecns-neural-policy",
                "policy_version": decision["policy_version"],
                "checkpoint_hash": decision["checkpoint_hash"],
                "decision_hash": decision["decision_hash"],
                "frame_id": decision["frame_id"],
                "board_hash": decision["board_hash"],
                "encoding_hash": decision["encoding_hash"],
                "action_index": decision["action_index"],
                "selected_logit": decision["selected_logit"],
                "output_features_hash": decision["output_features_hash"],
                "controller_source": decision["controller_source"],
                "controller_reason": decision["controller_reason"],
                "neural_readout_used": decision["neural_readout_used"],
                "pass_control": decision["pass_control"],
                "training_status": decision["training_status"],
                "teacher_accessed_at_runtime": False,
                "baseline_controller_called": False,
                "fallback_used": False,
            }
            self.server.go_session.commit_external_policy_move(
                expected_state, move, provenance
            )
            session = self._go_api_snapshot()
        except SessionTurnError as exc:
            self._send_error_json(HTTPStatus.CONFLICT, "wrong_turn", str(exc))
            return
        except StaleSessionError as exc:
            self._send_error_json(HTTPStatus.CONFLICT, "stale_neural_decision", str(exc))
            return
        except IllegalMoveError as exc:
            self._send_error_json(
                HTTPStatus.CONFLICT, f"illegal_neural_move_{exc.reason}", str(exc)
            )
            return
        except (NeuralDynamicsError, PolicyError, CheckpointValidationError, ValueError, TypeError):
            self._send_error_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "neural_move_failed",
                "The neural propagation or trained readout failed; the board was not changed and no fallback was used.",
            )
            return
        finally:
            self.server.neural_frame_lock.release()

        self._send_json(
            HTTPStatus.OK,
            {
                "ok": True,
                "controller": "malecns-neural-policy",
                "seed": payload["seed"],
                "pre_move_encoding": encoding,
                "frame": frame,
                "decision": decision,
                "session": session,
                "teacher_accessed_at_runtime": False,
                "baseline_controller_called": False,
                "fallback_used": False,
            },
        )

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        path = self._validated_path()
        if path is None:
            return
        if path == "/api/go/action":
            payload = self._read_go_action_request()
            if payload is not None:
                self._handle_go_action(payload)
            return
        if path == "/api/neural/frame":
            payload = self._read_neural_frame_request()
            if payload is not None:
                self._handle_neural_frame(payload)
            return
        if path == "/api/go/neural-move":
            payload = self._read_neural_frame_request()
            if payload is not None:
                self._handle_neural_move(payload)
            return
        if path != "/api/simulate":
            self._send_error_json(
                HTTPStatus.NOT_FOUND,
                "not_found",
                "Resource is not on the allowlist.",
                close=True,
            )
            return
        payload = self._read_simulation_request()
        if payload is None:
            return
        scenario = payload["scenario"]
        edges_enabled = payload["edges_enabled"]
        try:
            result = simulate(
                self.server.graph,
                SCENARIOS[scenario],
                edges_enabled=edges_enabled,
            )
        except DynamicsError:
            self._send_error_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "simulation_failed",
                "The bounded toy simulation could not be completed.",
            )
            return
        response = {
            "ok": True,
            **_common_payload(self.server.graph),
            "scenario": scenario,
            "scenario_label": SCENARIO_LABELS[scenario],
            "engineering_label_notice": (
                "left/right are interface engineering labels for body IDs 12781/556329; "
                "they are not biological laterality claims."
            ),
            "simulation": result,
        }
        self._send_json(HTTPStatus.OK, response)

    def _method_not_allowed(self) -> None:
        self._send_error_json(
            HTTPStatus.METHOD_NOT_ALLOWED,
            "method_not_allowed",
            "Only GET and POST on their documented routes are allowed.",
        )

    def do_HEAD(self) -> None:  # noqa: N802 - stdlib handler API
        # HEAD never carries a response body, including for an error response.
        self.send_response(HTTPStatus.METHOD_NOT_ALLOWED)
        self._security_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", "0")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Allow", "GET, POST")
        self.end_headers()

    do_PUT = _method_not_allowed
    do_PATCH = _method_not_allowed
    do_DELETE = _method_not_allowed
    do_OPTIONS = _method_not_allowed


def create_server(
    port: int = 8000,
    events_dir: Path | None = None,
    *,
    neural_graph: GraphArrays | None | object = _AUTO_LOAD_NEURAL,
    neural_model_dir: Path | str = DEFAULT_NEURAL_MODEL_DIR,
    policy_checkpoint: PolicyCheckpoint | None | object = _AUTO_LOAD_POLICY,
    policy_checkpoint_dir: Path | str = DEFAULT_POLICY_CHECKPOINT_DIR,
) -> LabServer:
    """Create a server bound only to IPv4 loopback; port 0 selects a test port."""
    if type(port) is not int or not 0 <= port <= 65_535:
        raise ValueError("port must be an integer in [0, 65535]")
    graph = load_verified_graph(P1_SUBGRAPH, P1_PROVENANCE)
    model_dir = Path(neural_model_dir).resolve()
    neural_error: str | None = None
    if neural_graph is _AUTO_LOAD_NEURAL:
        try:
            loaded_neural_graph: GraphArrays | None = _strict_load_neural_graph(
                model_dir
            )
        except Exception as exc:
            loaded_neural_graph = None
            neural_error = f"{type(exc).__name__}: {exc}"
    elif neural_graph is None:
        loaded_neural_graph = None
        neural_error = "Neural runtime explicitly disabled or unavailable."
    elif isinstance(neural_graph, GraphArrays):
        neural_graph.validate()
        loaded_neural_graph = neural_graph
    else:
        raise TypeError("neural_graph must be GraphArrays, None, or omitted")

    checkpoint_dir = Path(policy_checkpoint_dir).resolve()
    policy_error: str | None = None
    if policy_checkpoint is _AUTO_LOAD_POLICY:
        try:
            loaded_policy: PolicyCheckpoint | None = load_policy_checkpoint(
                checkpoint_dir
            )
            if loaded_policy.training_status != "trained":
                raise CheckpointValidationError("policy checkpoint is not trained")
            if loaded_neural_graph is None:
                raise CheckpointValidationError("verified neural graph is unavailable")
            if (
                loaded_policy.model_id != loaded_neural_graph.model_id
                or loaded_policy.graph_manifest_sha256
                != loaded_neural_graph.manifest_sha256
            ):
                raise CheckpointValidationError(
                    "policy checkpoint does not match the loaded neural graph"
                )
        except Exception as exc:
            loaded_policy = None
            policy_error = f"{type(exc).__name__}: {exc}"
    elif policy_checkpoint is None:
        loaded_policy = None
        policy_error = "Neural policy explicitly disabled or unavailable."
    elif isinstance(policy_checkpoint, PolicyCheckpoint):
        if policy_checkpoint.training_status != "trained":
            raise CheckpointValidationError("injected policy checkpoint is not trained")
        if loaded_neural_graph is None or (
            policy_checkpoint.model_id != loaded_neural_graph.model_id
            or policy_checkpoint.graph_manifest_sha256
            != loaded_neural_graph.manifest_sha256
        ):
            raise CheckpointValidationError(
                "injected policy checkpoint does not match the neural graph"
            )
        loaded_policy = policy_checkpoint
    else:
        raise TypeError("policy_checkpoint must be PolicyCheckpoint, None, or omitted")
    return LabServer(
        ("127.0.0.1", port),
        graph,
        AGENT_EVENTS_DIR if events_dir is None else Path(events_dir),
        neural_graph=loaded_neural_graph,
        neural_model_dir=model_dir,
        neural_load_error=neural_error,
        policy_checkpoint=loaded_policy,
        policy_checkpoint_dir=checkpoint_dir,
        policy_load_error=policy_error,
    )


def serve(
    port: int = 8000,
    *,
    policy_checkpoint_dir: Path | str = DEFAULT_POLICY_CHECKPOINT_DIR,
) -> None:
    """Run the lab until interrupted by the local operator."""
    server = create_server(port, policy_checkpoint_dir=policy_checkpoint_dir)
    host, actual_port = server.server_address
    print(f"P4 lab: http://{host}:{actual_port}")
    print("Local loopback only. Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping P4 lab.")
    finally:
        # ``shutdown`` must be called from a different thread than
        # ``serve_forever``.  A KeyboardInterrupt already exits that loop, so
        # calling shutdown here would deadlock the command on Ctrl+C.
        server.server_close()


__all__ = [
    "MAX_REQUEST_BYTES",
    "MAX_NEURAL_SEED",
    "NEURAL_TIMEOUT_SECONDS",
    "DEFAULT_NEURAL_MODEL_DIR",
    "DEFAULT_POLICY_CHECKPOINT_DIR",
    "AGENT_EVENTS_DIR",
    "SCENARIOS",
    "LabRequestHandler",
    "LabServer",
    "create_server",
    "serve",
]
