"""Cache real MaleCNS features for a failed candidate's own white-turn states.

Each state is replayed from a verified game report.  The full graph is rerun
with the report's original LIF parameters and seed; frame and output hashes
must match the original move record.  Capture-first labels are computed only
after the neural frame is verified, and are offline supervision, not runtime
decisions.  This is a DAgger-style correction dataset, not fly plasticity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_neural_go_readout import canonical_bytes, file_hash, load_cache, object_hash
from src.go_engine import BLACK, PASS, initial_state, legal_moves, step
from src.go_neural_encoding import PASS_ACTION_INDEX, action_to_move, encode_go_state, move_to_action
from src.go_session import select_capture_first_move
from src.malecns_dynamics import LIFParameters, load_graph_assets, simulate_go_encoding


def _max_black_capture(state) -> int:
    if state.game_over or state.to_play != BLACK:
        return 0
    return max(
        (step(state, move).black_captures - state.black_captures
         for move in legal_moves(state, include_pass=False)),
        default=0,
    )


def build(
    *,
    base_cache: Path,
    game_report: Path,
    graph_directory: Path,
    output_directory: Path,
) -> dict[str, object]:
    if output_directory.exists():
        raise ValueError("on-policy cache output directory already exists")
    _, _, base_records, base_manifest = load_cache(base_cache)
    report = json.loads(game_report.read_text(encoding="utf-8"))
    report_hash = report.pop("report_sha256", None)
    if report.get("schema") != "malecns-go-stage7-production-white-v2" or object_hash(report) != report_hash:
        raise ValueError("game report hash or schema mismatch")
    if report["checkpoint"]["training_dataset_hash"] != base_manifest["dataset_hash"]:
        raise ValueError("game report does not descend from the base cache")
    graph = load_graph_assets(graph_directory)
    if (
        graph.model_id != report["graph"]["model_id"]
        or graph.manifest_sha256 != report["graph"]["manifest_sha256"]
    ):
        raise ValueError("game graph does not match graph assets")
    parameters = LIFParameters(**report["configuration"]["lif_parameters"])
    lif_seed = int(report["configuration"]["lif_seed"])
    base_validation_states = {
        row["hashes"]["state_sha256"]
        for row in base_records if row["split"] == "validation"
    }
    base_validation_features = {
        row["hashes"]["features_sha256"]
        for row in base_records if row["split"] == "validation"
    }
    feature_rows: list[list[float]] = []
    labels: list[int] = []
    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for game in report["games"]:
        state = initial_state()
        for move_record in game["opening"]["moves"] + game["moves"]:
            source_state_hash = object_hash(state.to_dict())
            if source_state_hash != move_record["state_sha256_before"]:
                raise ValueError("replayed state differs from source move record")
            if move_record["actor"] == "malecns-neural-policy":
                encoding = encode_go_state(state)
                if source_state_hash in base_validation_states:
                    raise ValueError("on-policy state overlaps base validation state")
                frame = simulate_go_encoding(
                    graph, encoding, parameters=parameters, seed=lif_seed
                )
                provenance = move_record["provenance"]
                if (
                    frame["frame_id"] != provenance["frame_id"]
                    or encoding["encoding_hash"] != provenance["encoding_hash"]
                    or frame["output_pool"]["features_hash"]
                    != provenance["output_features_hash"]
                ):
                    raise ValueError("replayed MaleCNS frame differs from source game")
                if frame["output_pool"]["features_hash"] in base_validation_features:
                    raise ValueError("on-policy neural feature overlaps base validation feature")
                teacher_move = select_capture_first_move(state)
                teacher_action = move_to_action(teacher_move)
                if teacher_action == PASS_ACTION_INDEX:
                    raise ValueError("point-only training encountered a teacher PASS label")
                actual_move = action_to_move(int(move_record["action"]))
                model_risk = _max_black_capture(step(state, actual_move))
                teacher_risk = _max_black_capture(step(state, teacher_move))
                feature_rows.append(list(frame["output_pool"]["features"]))
                labels.append(teacher_action)
                rows.append({
                    "feature_row": len(rows),
                    "source_report_sha256": report_hash,
                    "source_checkpoint_hash": report["checkpoint"]["hash"],
                    "game_seed": game["seed"],
                    "board_ply": move_record["ply"],
                    "state_sha256": source_state_hash,
                    "encoding_hash": encoding["encoding_hash"],
                    "frame_id": frame["frame_id"],
                    "features_hash": frame["output_pool"]["features_hash"],
                    "legal_mask": encoding["legal_mask"],
                    "model_action": int(move_record["action"]),
                    "teacher_action": teacher_action,
                    "model_max_next_black_capture": model_risk,
                    "teacher_max_next_black_capture": teacher_risk,
                    "teacher_safer_by_at_least_3": model_risk - teacher_risk >= 3,
                    "supervision_source": "offline_capture_first_on_real_model_visited_state",
                    "male_cns_graph_rerun_and_frame_hash_verified": True,
                })
            state = step(state, action_to_move(int(move_record["action"])))
            if object_hash(state.to_dict()) != move_record["state_sha256_after"]:
                raise ValueError("replayed next state differs from source move record")
    if not rows:
        raise ValueError("source report has no neural point moves")
    features = np.asarray(feature_rows, dtype=np.float32)
    label_array = np.asarray(labels, dtype=np.int64)
    if features.shape != (len(rows), 128):
        raise ValueError("MaleCNS output shape mismatch")
    output_directory.mkdir(parents=True)
    features_path = output_directory / "features.npy"
    labels_path = output_directory / "labels.npy"
    records_path = output_directory / "records.jsonl"
    np.save(features_path, features, allow_pickle=False)
    np.save(labels_path, label_array, allow_pickle=False)
    records_path.write_bytes(b"".join(canonical_bytes(row) + b"\n" for row in rows))
    manifest = {
        "schema": "stage9-on-policy-male-cns-correction-cache-v1",
        "base_cache_dataset_hash": base_manifest["dataset_hash"],
        "source_game_report_sha256": report_hash,
        "source_checkpoint_hash": report["checkpoint"]["hash"],
        "graph_model_id": graph.model_id,
        "graph_manifest_sha256": graph.manifest_sha256,
        "lif_parameters": report["configuration"]["lif_parameters"],
        "lif_seed": lif_seed,
        "feature_count": 128,
        "sample_count": len(rows),
        "game_seed_count": len({row["game_seed"] for row in rows}),
        "teacher_disagreement_count": sum(row["teacher_action"] != row["model_action"] for row in rows),
        "teacher_safer_by_at_least_3_count": sum(row["teacher_safer_by_at_least_3"] for row in rows),
        "base_validation_state_overlap_count": 0,
        "base_validation_feature_overlap_count": 0,
        "artifacts": {
            "features.npy": file_hash(features_path),
            "labels.npy": file_hash(labels_path),
            "records.jsonl": file_hash(records_path),
        },
    }
    manifest["dataset_hash"] = object_hash(manifest)
    (output_directory / "manifest.json").write_bytes(canonical_bytes(manifest) + b"\n")
    return {
        "output_directory": str(output_directory.resolve()),
        "dataset_hash": manifest["dataset_hash"],
        "sample_count": len(rows),
        "teacher_disagreement_count": manifest["teacher_disagreement_count"],
        "teacher_safer_by_at_least_3_count": manifest["teacher_safer_by_at_least_3_count"],
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-cache", type=Path, required=True)
    parser.add_argument("--game-report", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(
        base_cache=args.base_cache,
        game_report=args.game_report,
        graph_directory=args.graph,
        output_directory=args.output,
    ), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
