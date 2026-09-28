"""Stage 6 fixed-horizon games: MaleCNS policy versus legal random play.

Every neural move uses the production board encoder, full verified MaleCNS LIF
runtime, strict trained checkpoint and legal mask.  The opponent samples only
legal point moves and passes only when no point move exists.  Games swap the
neural side for every seed.  A game that reaches the configured horizon is
explicitly marked as adjudicated rather than misreported as a natural finish.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Callable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_neural_go_readout import (
    DEFAULT_CHECKPOINT,
    DEFAULT_GRAPH,
    canonical_bytes,
    object_hash,
)
from src.go_engine import (
    BLACK,
    EMPTY,
    PASS,
    WHITE,
    GoState,
    initial_state,
    legal_moves,
    score_area,
    step,
)
from src.go_neural_encoding import action_to_move, board_hash, encode_go_state
from src.malecns_dynamics import LIFParameters, load_graph_assets, simulate_go_encoding
from src.malecns_policy import load_policy_checkpoint, select_action


REPORT_SCHEMA = "malecns-go-stage6-games-v4"
LEGACY_REPORT_SCHEMAS = frozenset(
    {
        "malecns-go-stage6-games-v1",
        "malecns-go-stage6-games-v2",
        "malecns-go-stage6-games-v3",
    }
)
SUPPORTED_REPORT_SCHEMAS = LEGACY_REPORT_SCHEMAS | {REPORT_SCHEMA}
OPPONENT_RANDOM_STREAM_VERSION = "seed-only-v2"
OPENING_RANDOM_STREAM_VERSION = "seed-and-neural-colour-v1"
STATE_HASH_VERSION = "go-state-to-dict-sha256-v1"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "evaluation" / "stage6" / "games.json"
NeuralSelector = Callable[[GoState, int], tuple[object, Mapping[str, Any]]]


def color_name(color: int) -> str:
    if color == BLACK:
        return "black"
    if color == WHITE:
        return "white"
    if color == EMPTY:
        return "draw"
    raise ValueError("unknown Go color")


def select_legal_random_move(state: GoState, rng: random.Random) -> object:
    """Sample a legal point; pass only when the board has no legal point."""
    points = legal_moves(state, include_pass=False)
    return points[rng.randrange(len(points))] if points else PASS


def opponent_random_stream_id(seed: int) -> str:
    """Return the colour-independent identity of a paired opponent RNG stream."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return object_hash(["stage6-legal-random", OPPONENT_RANDOM_STREAM_VERSION, seed])


def opening_random_stream_id(seed: int, neural_color: int) -> str:
    """Return a reproducible, role-specific opening stream identity."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if neural_color not in (BLACK, WHITE):
        raise ValueError("neural_color must be BLACK or WHITE")
    return object_hash(
        ["stage6-random-opening", OPENING_RANDOM_STREAM_VERSION, seed, neural_color]
    )


def schedule(base_seed: int, seed_count: int) -> list[tuple[int, int]]:
    """Return paired seeds with the neural policy playing both colours."""
    if type(base_seed) is not int or base_seed < 0:
        raise ValueError("base_seed must be a nonnegative integer")
    if type(seed_count) is not int or seed_count <= 0:
        raise ValueError("seed_count must be positive")
    return [
        (base_seed + offset, neural_color)
        for offset in range(seed_count)
        for neural_color in (BLACK, WHITE)
    ]


def play_game(
    *,
    seed: int,
    neural_color: int,
    max_moves: int,
    neural_selector: NeuralSelector,
    opening_plies: int = 0,
) -> dict[str, Any]:
    """Play one auditable game and return compact per-move evidence."""
    if neural_color not in (BLACK, WHITE):
        raise ValueError("neural_color must be BLACK or WHITE")
    if type(max_moves) is not int or max_moves <= 1:
        raise ValueError("max_moves must be greater than one")
    if type(opening_plies) is not int or opening_plies < 0:
        raise ValueError("opening_plies must be a nonnegative integer")
    random_stream_id = opponent_random_stream_id(seed)
    rng = random.Random(random_stream_id)
    opening_stream_id = opening_random_stream_id(seed, neural_color)
    opening_rng = random.Random(opening_stream_id)
    state = initial_state()
    opening_moves: list[dict[str, Any]] = []
    for opening_ply in range(1, opening_plies + 1):
        before = state
        move = select_legal_random_move(before, opening_rng)
        state = step(before, move)
        opening_record = {
            "opening_ply": opening_ply,
            "board_ply": state.move_number,
            "actor": "random-opening",
            "color": color_name(before.to_play),
            "action": 81 if move is PASS else int(move[0]) * 9 + int(move[1]),
            "move": "pass" if move is PASS else [int(move[0]), int(move[1])],
            "board_hash_before": board_hash(before),
            "board_hash_after": board_hash(state),
            "state_sha256_before": object_hash(before.to_dict()),
            "state_sha256_after": object_hash(state.to_dict()),
            "provenance": {
                "policy": "uniform-legal-point-random-opening-v1",
                "random_stream_version": OPENING_RANDOM_STREAM_VERSION,
                "random_stream_id": opening_stream_id,
                "seed_scope": "game-seed-and-neural-colour",
                "neural_policy_called": False,
                "attributed_to_evaluated_policy": False,
            },
        }
        opening_record["record_sha256"] = object_hash(opening_record)
        opening_moves.append(opening_record)
        if state.game_over:
            break

    moves: list[dict[str, Any]] = []
    while not state.game_over and len(moves) < max_moves:
        before = state
        actor_color = before.to_play
        if actor_color == neural_color:
            move, provenance = neural_selector(before, seed)
            controller_source = provenance.get("controller_source")
            if controller_source == "go-rule-pass-gate":
                if move is not PASS or provenance.get("neural_readout_used") is not False:
                    raise ValueError("rule pass provenance contradicts selected move")
                actor = "go-rule-pass-gate"
            elif controller_source in (None, "malecns-linear-readout"):
                actor = "malecns-neural-policy"
            else:
                raise ValueError("unknown evaluated controller source")
        else:
            move = select_legal_random_move(before, rng)
            provenance = {
                "policy": "uniform-legal-point-random-v1",
                "random_stream_version": OPPONENT_RANDOM_STREAM_VERSION,
                "random_stream_id": random_stream_id,
                "seed_binding": random_stream_id,
                "seed_scope": "game-seed-only",
                "pass_only_if_no_legal_point": True,
            }
            actor = "legal-random"
        state = step(before, move)
        record = {
            "ply": state.move_number,
            "evaluation_ply": len(moves) + 1,
            "actor": actor,
            "color": color_name(actor_color),
            "action": 81 if move is PASS else int(move[0]) * 9 + int(move[1]),
            "move": "pass" if move is PASS else [int(move[0]), int(move[1])],
            "board_hash_before": board_hash(before),
            "board_hash_after": board_hash(state),
            "state_sha256_before": object_hash(before.to_dict()),
            "state_sha256_after": object_hash(state.to_dict()),
            "provenance": dict(provenance),
        }
        record["record_sha256"] = object_hash(record)
        moves.append(record)

    score = score_area(state)
    neural_won = score.winner == neural_color
    neural_total = score.black_total if neural_color == BLACK else score.white_total
    opponent_total = score.white_total if neural_color == BLACK else score.black_total
    game: dict[str, Any] = {
        "game_id": object_hash(
            {
                "seed": seed,
                "neural_color": neural_color,
                "max_moves": max_moves,
                "opening_move_hashes": [item["record_sha256"] for item in opening_moves],
                "move_hashes": [item["record_sha256"] for item in moves],
            }
        ),
        "seed": seed,
        "neural_color": color_name(neural_color),
        "random_color": color_name(WHITE if neural_color == BLACK else BLACK),
        "opponent_random_stream": {
            "version": OPPONENT_RANDOM_STREAM_VERSION,
            "id": random_stream_id,
            "seed_scope": "game-seed-only",
        },
        "opening": {
            "requested_plies": opening_plies,
            "completed_plies": len(opening_moves),
            "random_stream": {
                "version": OPENING_RANDOM_STREAM_VERSION,
                "id": opening_stream_id,
                "seed_scope": "game-seed-and-neural-colour",
            },
            "neural_policy_called": False,
            "attributed_to_evaluated_policy": False,
            "moves": opening_moves,
        },
        "move_count": len(moves),
        "evaluation_move_count": len(moves),
        "total_move_count": state.move_number,
        "termination": "two_passes" if state.game_over else "fixed_horizon_adjudication",
        "max_moves": max_moves,
        "winner": color_name(score.winner),
        "neural_won": neural_won,
        "draw": score.winner == EMPTY,
        "neural_score": neural_total,
        "opponent_score": opponent_total,
        "neural_score_difference": neural_total - opponent_total,
        "score": score.to_dict(),
        "final_board_hash": board_hash(state),
        "moves": moves,
    }
    game["game_sha256"] = object_hash(game)
    return game


def evaluate_games(
    *,
    graph_directory: Path,
    checkpoint_directory: Path,
    output_path: Path,
    base_seed: int,
    seed_count: int,
    max_moves: int,
    duration_ms: float,
    lif_seed: int,
    opening_plies: int = 0,
) -> dict[str, Any]:
    graph = load_graph_assets(graph_directory)
    checkpoint = load_policy_checkpoint(checkpoint_directory)
    if (
        checkpoint.model_id != graph.model_id
        or checkpoint.graph_manifest_sha256 != graph.manifest_sha256
    ):
        raise ValueError("checkpoint does not match the evaluation graph")
    parameters = LIFParameters(duration_ms=duration_ms)

    if type(lif_seed) is not int or lif_seed < 0:
        raise ValueError("lif_seed must be a nonnegative integer")

    def production_selector(state: GoState, _game_seed: int) -> tuple[object, Mapping[str, Any]]:
        encoding = encode_go_state(state)
        frame = simulate_go_encoding(
            graph,
            encoding,
            parameters=parameters,
            seed=lif_seed,
        )
        decision = select_action(frame, encoding, checkpoint)
        return action_to_move(decision["action_index"]), {
            "checkpoint_hash": decision["checkpoint_hash"],
            "encoding_hash": decision["encoding_hash"],
            "frame_id": decision["frame_id"],
            "decision_hash": decision["decision_hash"],
            "action_index": decision["action_index"],
            "output_features_hash": decision["output_features_hash"],
            "controller_source": decision["controller_source"],
            "controller_reason": decision["controller_reason"],
            "neural_readout_used": decision["neural_readout_used"],
            "pass_control": decision["pass_control"],
            "teacher_accessed_at_runtime": decision["teacher_accessed_at_runtime"],
            "baseline_controller_called": decision["baseline_controller_called"],
            "fallback_used": decision["fallback_used"],
        }

    games: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, (seed, neural_color) in enumerate(schedule(base_seed, seed_count), start=1):
        game = play_game(
            seed=seed,
            neural_color=neural_color,
            max_moves=max_moves,
            neural_selector=production_selector,
            opening_plies=opening_plies,
        )
        games.append(game)
        print(
            f"game {index}/{seed_count * 2}: seed={seed} neural={color_name(neural_color)} "
            f"winner={game['winner']} diff={game['neural_score_difference']:+.1f} "
            f"termination={game['termination']}",
            flush=True,
        )

    win_count = sum(game["neural_won"] for game in games)
    draw_count = sum(game["draw"] for game in games)
    losses = len(games) - win_count - draw_count
    differences = [float(game["neural_score_difference"]) for game in games]
    natural_finishes = sum(game["termination"] == "two_passes" for game in games)
    neural_decisions = sum(
        move["actor"] == "malecns-neural-policy"
        for game in games for move in game["moves"]
    )
    rule_pass_decisions = sum(
        move["actor"] == "go-rule-pass-gate"
        for game in games for move in game["moves"]
    )
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "scope": "paired fixed-horizon games versus legal random; pilot strength evidence",
        "status": "completed",
        "checkpoint": {
            "hash": checkpoint.checkpoint_hash,
            "training_seed": checkpoint.training_info.get("seed"),
            "training_dataset_hash": checkpoint.training_info.get("dataset_hash"),
        },
        "graph": {
            "model_id": graph.model_id,
            "manifest_sha256": graph.manifest_sha256,
            "node_count": graph.node_count,
            "edge_count": graph.edge_count,
        },
        "configuration": {
            "base_seed": base_seed,
            "seed_count": seed_count,
            "game_count": len(games),
            "max_moves": max_moves,
            "komi": 7.5,
            "opponent": "uniform-legal-point-random-v1",
            "opponent_random_stream_version": OPPONENT_RANDOM_STREAM_VERSION,
            "opponent_random_stream_seed_scope": "game-seed-only",
            "opening_plies": opening_plies,
            "opening_random_stream_version": OPENING_RANDOM_STREAM_VERSION,
            "opening_random_stream_seed_scope": "game-seed-and-neural-colour",
            "opening_moves_attributed_to_evaluated_policy": False,
            "move_state_hash": {
                "version": STATE_HASH_VERSION,
                "algorithm": "sha256-canonical-json",
                "payload": "GoState.to_dict()",
            },
            "colour_swap_per_seed": True,
            "lif_seed": lif_seed,
            "horizon_scoring": "Chinese area score without dead-stone adjudication",
            "lif_parameters": asdict(parameters),
        },
        "metrics": {
            "wins": win_count,
            "losses": losses,
            "draws": draw_count,
            "win_rate": win_count / len(games),
            "mean_neural_score_difference": sum(differences) / len(differences),
            "minimum_neural_score_difference": min(differences),
            "maximum_neural_score_difference": max(differences),
            "natural_finish_count": natural_finishes,
            "horizon_adjudication_count": len(games) - natural_finishes,
            "neural_readout_decision_count": neural_decisions,
            "go_rule_pass_decision_count": rule_pass_decisions,
        },
        "limitations": [
            "This pilot has too few games for a strong general playing-strength claim.",
            "Horizon-adjudicated games are not natural two-pass finishes.",
            "The random opponent is a weak baseline and does not search.",
            "The readout was trained by capture-first imitation, not whole-game reinforcement learning.",
        ],
        "runtime_guarantees": {
            "same_select_action_path_as_web_app": True,
            "teacher_accessed_at_runtime": False,
            "baseline_controller_called": False,
            "fallback_used": False,
        },
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "games": games,
    }
    report["report_sha256"] = object_hash(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(canonical_bytes(report) + b"\n")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-seed", type=int, default=20260923)
    parser.add_argument("--seed-count", type=int, default=2)
    parser.add_argument("--max-moves", type=int, default=100)
    parser.add_argument(
        "--opening-plies",
        type=int,
        default=0,
        help="legal random setup plies before evaluated play; default 0 preserves prior behaviour",
    )
    parser.add_argument("--duration-ms", type=float, default=12.0)
    parser.add_argument(
        "--lif-seed",
        type=int,
        default=0,
        help="fixed neural phase seed; web production requests use 0 by default",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = evaluate_games(
        graph_directory=args.graph,
        checkpoint_directory=args.checkpoint,
        output_path=args.output,
        base_seed=args.base_seed,
        seed_count=args.seed_count,
        max_moves=args.max_moves,
        duration_ms=args.duration_ms,
        lif_seed=args.lif_seed,
        opening_plies=args.opening_plies,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(args.output.resolve()),
                "report_sha256": report["report_sha256"],
                "elapsed_seconds": report["elapsed_seconds"],
                "metrics": report["metrics"],
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
