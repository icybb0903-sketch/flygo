"""Reproducible offline Stage 4 data generation, training, and evaluation.

The capture-first controller is used only to create supervised labels.  The
exported runtime checkpoint contains a 128 -> 82 linear readout and provenance;
it contains neither teacher code nor a fallback policy.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.go_engine import BLACK, WHITE, GoState, legal_moves, step
from src.go_neural_encoding import (
    ACTION_COUNT,
    PASS_ACTION_INDEX,
    encode_go_state,
    move_to_action,
)
from src.go_session import BASELINE_POLICY_ID, select_capture_first_move
from src.malecns_dynamics import LIFParameters, load_graph_assets, simulate_go_encoding


DATASET_SCHEMA = "malecns-go-stage4-dataset-v1"
SAMPLE_SCHEMA = "malecns-go-stage4-sample-v1"
TRAINING_SCHEMA = "malecns-go-stage4-training-v1"
FEATURE_COUNT = 128
LIF_SEED_MODES = ("fixed", "per-position")
RETAINED_COLOUR_MODES = ("white-only", "both")
TRAJECTORY_MODES = ("legacy", "role-symmetric")
PASS_CONTROL_MODES = ("joint-neural", "rule-after-opponent-pass")
DEFAULT_GRAPH = (
    PROJECT_ROOT
    / "data"
    / "malecns"
    / "runtime"
    / "malecns-v1.0-w5-3acb6434e71160fc"
)
DEFAULT_CACHE = PROJECT_ROOT / "data" / "training" / "stage4"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "data" / "checkpoints" / "stage4"


def canonical_bytes(value: object) -> bytes:
    """Return the byte representation used for every semantic JSON hash."""
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def object_hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value) + b"\n")


@dataclass(frozen=True)
class Position:
    game_id: str
    ply: int
    state: GoState
    teacher_action: int
    teacher_captures: int
    teacher_colour: int | None = None
    trajectory_kind: str = "legacy"
    split_group_id: str | None = None
    random_stream_id: str | None = None
    label_teacher: str = BASELINE_POLICY_ID
    supervision_kind: str = "natural-teacher-turn"
    source_state_sha256: str | None = None
    source_ply: int | None = None


def derive_lif_seed(position: Position, *, base_seed: int, mode: str) -> int:
    """Choose a reproducible LIF phase seed without inspecting the label."""
    if mode == "fixed":
        return base_seed
    if mode == "per-position":
        digest = object_hash(["lif-phase-v1", base_seed, position.game_id, position.ply])
        return int(digest[:16], 16) & (2**63 - 1)
    raise ValueError(f"lif seed mode must be one of {LIF_SEED_MODES}")


def split_game_ids(
    game_ids: Iterable[str], *, validation_fraction: float, seed: int
) -> dict[str, str]:
    """Assign complete games to one split using a stable hash ordering."""
    unique = sorted(set(game_ids))
    if len(unique) < 2:
        raise ValueError("at least two games are required for a train/validation split")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    ordered = sorted(unique, key=lambda item: (object_hash([seed, item]), item))
    validation_count = min(len(unique) - 1, max(1, round(len(unique) * validation_fraction)))
    validation = set(ordered[:validation_count])
    return {item: ("validation" if item in validation else "train") for item in unique}


def _capture_count(before: GoState, after: GoState) -> int:
    if before.to_play == BLACK:
        return after.black_captures - before.black_captures
    return after.white_captures - before.white_captures


def _seeded_random_move(state: GoState, rng: random.Random) -> tuple[int, int] | None:
    """Select the legacy exploratory opponent move from an explicit RNG stream."""
    point_moves = list(legal_moves(state, include_pass=False))
    if not point_moves:
        return None
    captures: list[tuple[int, tuple[int, int]]] = []
    for candidate in point_moves:
        captured = _capture_count(state, step(state, candidate))
        if captured:
            captures.append((captured, candidate))
    if captures and rng.random() < 0.35:
        best = max(count for count, _ in captures)
        choices = [move for count, move in captures if count == best]
        return choices[rng.randrange(len(choices))]
    return point_moves[rng.randrange(len(point_moves))]


def deduplicate_positions_by_state(
    positions: Iterable[Position],
) -> tuple[list[Position], int]:
    """Keep the first occurrence of each complete serialized Go state."""
    unique: list[Position] = []
    seen: set[str] = set()
    duplicate_count = 0
    for position in positions:
        state_hash = object_hash(position.state.to_dict())
        if state_hash in seen:
            duplicate_count += 1
            continue
        seen.add(state_hash)
        unique.append(position)
    return unique, duplicate_count


def feature_equivalence_components(
    positions: Sequence[Position], feature_hashes: Sequence[str]
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Connect complete paired groups that share any model feature vector."""
    if len(positions) != len(feature_hashes):
        raise ValueError("positions and feature_hashes must have equal lengths")
    group_ids = sorted(
        {position.split_group_id or position.game_id for position in positions}
    )
    parent = {group_id: group_id for group_id in group_ids}

    def find(group_id: str) -> str:
        while parent[group_id] != group_id:
            parent[group_id] = parent[parent[group_id]]
            group_id = parent[group_id]
        return group_id

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        first, second = sorted((left_root, right_root))
        parent[second] = first

    groups_by_feature: dict[str, set[str]] = {}
    for position, feature_hash in zip(positions, feature_hashes, strict=True):
        if not isinstance(feature_hash, str) or len(feature_hash) != 64:
            raise ValueError("feature hashes must be SHA-256 strings")
        group_id = position.split_group_id or position.game_id
        groups_by_feature.setdefault(feature_hash, set()).add(group_id)
    for groups in groups_by_feature.values():
        ordered_groups = sorted(groups)
        for group_id in ordered_groups[1:]:
            union(ordered_groups[0], group_id)

    components_by_root: dict[str, list[str]] = {}
    for group_id in group_ids:
        components_by_root.setdefault(find(group_id), []).append(group_id)
    component_id_by_group: dict[str, str] = {}
    component_groups: dict[str, list[str]] = {}
    for groups in components_by_root.values():
        ordered_groups = sorted(groups)
        component_id = object_hash(
            ["feature-equivalence-split-component-v1", ordered_groups]
        )
        component_groups[component_id] = ordered_groups
        for group_id in ordered_groups:
            component_id_by_group[group_id] = component_id
    return component_id_by_group, component_groups


def generate_teacher_positions(
    *,
    game_count: int,
    max_moves: int,
    seed: int,
    retained_colours: str = "white-only",
    trajectory_mode: str = "legacy",
    pass_response_samples_per_trajectory: int = 0,
) -> list[Position]:
    """Generate legal games and capture-first labels for selected colours.

    ``legacy`` preserves the historical random-Black/teacher-White trajectory.
    ``role-symmetric`` creates paired games in which each colour is teacher in
    one trajectory and only that controlled role contributes labels.  The
    teacher is never passed to the neural simulation or checkpoint arrays.
    """
    from src.go_engine import initial_state

    if game_count < 2:
        raise ValueError("game_count must be at least two")
    if max_moves < 4:
        raise ValueError("max_moves must be at least four")
    if retained_colours not in RETAINED_COLOUR_MODES:
        raise ValueError(f"retained_colours must be one of {RETAINED_COLOUR_MODES}")
    if trajectory_mode not in TRAJECTORY_MODES:
        raise ValueError(f"trajectory_mode must be one of {TRAJECTORY_MODES}")
    if (
        type(pass_response_samples_per_trajectory) is not int
        or pass_response_samples_per_trajectory < 0
    ):
        raise ValueError("pass_response_samples_per_trajectory must be a nonnegative integer")
    if trajectory_mode == "legacy" and pass_response_samples_per_trajectory:
        raise ValueError("pass-response augmentation is available only in role-symmetric mode")
    positions: list[Position] = []
    for game_index in range(game_count):
        split_group_id = f"game-{game_index:04d}-seed-{seed}"
        if trajectory_mode == "legacy":
            trajectory_colours = (WHITE,)
        else:
            trajectory_colours = (BLACK, WHITE)
        for teacher_colour in trajectory_colours:
            role_name = "black" if teacher_colour == BLACK else "white"
            game_id = (
                split_group_id
                if trajectory_mode == "legacy"
                else f"{split_group_id}-teacher-{role_name}"
            )
            # Paired trajectories deliberately start from the same random stream.
            # Its seed contains no neural/teacher colour, preventing colour-bound
            # opponent randomness from becoming a training signal.
            random_stream_id = (
                object_hash([seed, game_id])
                if trajectory_mode == "legacy"
                else object_hash(["seeded-random-opponent-v1", seed, game_index])
            )
            rng = random.Random(random_stream_id)
            state = initial_state()
            pass_response_candidates: list[Position] = []
            late_ply = max_moves * 3 // 4
            while not state.game_over and state.move_number < max_moves:
                teacher_move = select_capture_first_move(state)
                teacher_after = step(state, teacher_move)
                teacher_controls_turn = state.to_play == teacher_colour
                if (
                    trajectory_mode == "role-symmetric"
                    and pass_response_samples_per_trajectory
                    and not teacher_controls_turn
                    and state.consecutive_passes == 0
                    and state.move_number >= late_ply
                ):
                    # This is an explicit offline branch, not a claim that the
                    # opponent passed in the natural generated trajectory.
                    passed_state = step(state, None)
                    if not passed_state.game_over and passed_state.to_play == teacher_colour:
                        pass_response_candidates.append(
                            Position(
                                game_id=game_id,
                                ply=passed_state.move_number,
                                state=passed_state,
                                teacher_action=PASS_ACTION_INDEX,
                                teacher_captures=0,
                                teacher_colour=teacher_colour,
                                trajectory_kind=f"teacher-{role_name}-vs-seeded-random",
                                split_group_id=split_group_id,
                                random_stream_id=random_stream_id,
                                label_teacher="pass-response-augmentation-v1",
                                supervision_kind="constructed-opponent-pass-response",
                                source_state_sha256=object_hash(state.to_dict()),
                                source_ply=state.move_number,
                            )
                        )
                retain = (
                    teacher_controls_turn
                    if trajectory_mode == "role-symmetric"
                    else retained_colours == "both" or state.to_play == WHITE
                )
                if retain:
                    positions.append(
                        Position(
                            game_id=game_id,
                            ply=state.move_number,
                            state=state,
                            teacher_action=move_to_action(teacher_move),
                            teacher_captures=_capture_count(state, teacher_after),
                            teacher_colour=(
                                teacher_colour
                                if trajectory_mode == "role-symmetric"
                                else state.to_play
                            ),
                            trajectory_kind=(
                                f"teacher-{role_name}-vs-seeded-random"
                                if trajectory_mode == "role-symmetric"
                                else "legacy-teacher-white-vs-seeded-random-black"
                            ),
                            split_group_id=split_group_id,
                            random_stream_id=random_stream_id,
                        )
                    )
                if teacher_controls_turn:
                    after = teacher_after
                else:
                    after = step(state, _seeded_random_move(state, rng))
                state = after
            if pass_response_samples_per_trajectory:
                positions.extend(
                    pass_response_candidates[-pass_response_samples_per_trajectory:]
                )
    if not positions:
        raise RuntimeError("game generation produced no White training positions")
    return positions


def _sample_record(
    position: Position,
    *,
    split: str,
    encoding: Mapping[str, Any],
    frame: Mapping[str, Any],
    cached_features: np.ndarray,
    feature_row: int,
    lif_seed: int,
) -> dict[str, Any]:
    output_pool = frame["output_pool"]
    state_payload = position.state.to_dict()
    label_payload = {
        "teacher": position.label_teacher,
        "action": position.teacher_action,
    }
    hashes = {
        "state_sha256": object_hash(state_payload),
        "encoding_sha256": str(encoding["encoding_hash"]),
        "frame_sha256": str(frame["frame_id"]),
        "label_sha256": object_hash(label_payload),
        "features_sha256": str(output_pool["features_hash"]),
        "cached_feature_row_sha256": object_hash(
            {
                "dtype": "float32",
                "values": [float(value) for value in cached_features],
            }
        ),
    }
    sample_id = object_hash(
        {
            "game_id": position.game_id,
            "ply": position.ply,
            "hashes": hashes,
        }
    )
    record: dict[str, Any] = {
        "schema": SAMPLE_SCHEMA,
        "sample_id": sample_id,
        "game_id": position.game_id,
        "split": split,
        "ply": position.ply,
        "feature_row": feature_row,
        "state": state_payload,
        "board_hash": encoding["board_hash"],
        "legal_mask": encoding["legal_mask"],
        "label_action": position.teacher_action,
        "label_teacher": position.label_teacher,
        "teacher_captures": position.teacher_captures,
        "teacher_colour": position.teacher_colour,
        "trajectory_kind": position.trajectory_kind,
        "split_group_id": position.split_group_id or position.game_id,
        "random_stream_id": position.random_stream_id,
        "supervision_provenance": {
            "kind": position.supervision_kind,
            "is_natural_trajectory_state": position.supervision_kind == "natural-teacher-turn",
            "method": (
                "opponent-pass-applied-to-late-legal-state"
                if position.supervision_kind == "constructed-opponent-pass-response"
                else "trajectory-state"
            ),
            "source_state_sha256": position.source_state_sha256,
            "source_ply": position.source_ply,
            "source_game_id": position.game_id if position.source_state_sha256 else None,
        },
        "lif_seed": lif_seed,
        "model_id": frame["model_id"],
        "graph_manifest_sha256": frame["manifest_sha256"],
        "parameters_hash": frame["parameters_hash"],
        "output_pool_version": output_pool["version"],
        "output_pool_group_hash": output_pool["group_hash"],
        "hashes": hashes,
    }
    record["record_sha256"] = object_hash(record)
    return record


def generate_cache(
    *,
    graph_directory: Path,
    cache_directory: Path,
    game_count: int,
    max_moves: int,
    seed: int,
    validation_fraction: float,
    duration_ms: float,
    progress_every: int = 0,
    lif_seed_mode: str = "fixed",
    lif_seed_base: int | None = None,
    retained_colours: str = "white-only",
    trajectory_mode: str = "legacy",
    pass_response_samples_per_trajectory: int = 0,
) -> dict[str, Any]:
    """Run real project LIF frames and write aligned JSONL/NPY evidence."""
    graph = load_graph_assets(graph_directory)
    positions = generate_teacher_positions(
        game_count=game_count,
        max_moves=max_moves,
        seed=seed,
        retained_colours=retained_colours,
        trajectory_mode=trajectory_mode,
        pass_response_samples_per_trajectory=pass_response_samples_per_trajectory,
    )
    generated_position_count = len(positions)
    duplicate_state_count = 0
    if trajectory_mode == "role-symmetric":
        positions, duplicate_state_count = deduplicate_positions_by_state(positions)
    state_unique_position_count = len(positions)

    # Encoding is the actual model input.  Full GoState equality is stricter
    # (it includes replay history), so remove duplicate encodings before the
    # expensive full-graph LIF simulation.  Representatives are selected by a
    # stable semantic key and retain their complete paired split group.
    encoded_candidates: list[tuple[int, Position, Mapping[str, Any]]] = []
    for index, position in enumerate(positions):
        encoding = encode_go_state(position.state)
        encoding_hash = encoding.get("encoding_hash")
        if not isinstance(encoding_hash, str) or len(encoding_hash) != 64:
            raise RuntimeError("Go encoding has no valid encoding_hash")
        encoded_candidates.append((index, position, encoding))
    encoding_duplicate_count = 0
    encoding_label_conflict_count = 0
    if trajectory_mode == "role-symmetric":
        candidates_by_encoding: dict[
            str, list[tuple[int, Position, Mapping[str, Any]]]
        ] = {}
        for item in encoded_candidates:
            candidates_by_encoding.setdefault(str(item[2]["encoding_hash"]), []).append(item)
        retained_candidates: list[tuple[int, Position, Mapping[str, Any]]] = []
        for candidates in candidates_by_encoding.values():
            labels = {item[1].teacher_action for item in candidates}
            if len(labels) > 1:
                encoding_label_conflict_count += 1
                continue
            representative = min(
                candidates,
                key=lambda item: (
                    item[1].split_group_id or item[1].game_id,
                    item[1].game_id,
                    item[1].ply,
                    item[1].supervision_kind,
                    item[0],
                ),
            )
            retained_candidates.append(representative)
            encoding_duplicate_count += len(candidates) - 1
        if encoding_label_conflict_count:
            raise RuntimeError(
                "role-symmetric data contains identical encodings with conflicting labels"
            )
        encoded_candidates = sorted(retained_candidates, key=lambda item: item[0])
    positions = [item[1] for item in encoded_candidates]
    encodings = [item[2] for item in encoded_candidates]
    provisional_split_by_game = split_game_ids(
        (item.split_group_id or item.game_id for item in positions),
        validation_fraction=validation_fraction,
        seed=seed,
    )
    parameters = LIFParameters(duration_ms=duration_ms)
    features = np.empty((len(positions), FEATURE_COUNT), dtype=np.float32)
    labels = np.empty((len(positions),), dtype=np.int64)
    frames: list[Mapping[str, Any]] = []
    resolved_lif_seed_base = seed if lif_seed_base is None else lif_seed_base
    if type(resolved_lif_seed_base) is not int or resolved_lif_seed_base < 0:
        raise ValueError("lif_seed_base must be a nonnegative integer")
    for row, (position, encoding) in enumerate(zip(positions, encodings, strict=True)):
        lif_seed = derive_lif_seed(
            position,
            base_seed=resolved_lif_seed_base,
            mode=lif_seed_mode,
        )
        frame = simulate_go_encoding(
            graph, encoding, parameters=parameters, seed=lif_seed
        )
        output_pool = frame.get("output_pool")
        if not isinstance(output_pool, Mapping):
            raise RuntimeError("LIF frame has no output_pool")
        raw_features = output_pool.get("features")
        vector = np.asarray(raw_features, dtype=np.float32)
        if vector.shape != (FEATURE_COUNT,) or not np.all(np.isfinite(vector)):
            raise RuntimeError("LIF output features must be 128 finite values")
        expected_feature_hash = object_hash(
            {
                "version": output_pool.get("version"),
                "group_hash": output_pool.get("group_hash"),
                "features": raw_features,
            }
        )
        if output_pool.get("features_hash") != expected_feature_hash:
            raise RuntimeError("LIF output features_hash does not match feature values")
        features[row] = vector
        labels[row] = position.teacher_action
        frames.append(frame)
        completed = row + 1
        if progress_every > 0 and (completed % progress_every == 0 or completed == len(positions)):
            print(
                f"generated {completed}/{len(positions)} full-graph training frames",
                flush=True,
            )

    # A short deterministic LIF window can quantize different encodings to the
    # same 128-value feature vector.  Treat paired trajectory groups connected
    # by any equal feature hash as one indivisible split component.  This split
    # is computed from inputs only; teacher labels never influence it.
    group_ids = sorted(
        {position.split_group_id or position.game_id for position in positions}
    )
    feature_hashes = [str(frame["output_pool"]["features_hash"]) for frame in frames]
    if trajectory_mode == "role-symmetric":
        component_id_by_group, component_groups = feature_equivalence_components(
            positions, feature_hashes
        )
        if len(component_groups) < 2:
            raise RuntimeError(
                "feature equivalence connects every paired group; use a higher-information "
                "LIF configuration before claiming an independent validation split"
            )
        split_by_component = split_game_ids(
            component_groups,
            validation_fraction=validation_fraction,
            seed=seed,
        )
        split_by_game = {
            group_id: split_by_component[component_id_by_group[group_id]]
            for group_id in group_ids
        }
    else:
        component_id_by_group = {
            group_id: object_hash(
                ["legacy-split-component-v1", group_id]
            )
            for group_id in group_ids
        }
        component_groups = {
            component_id_by_group[group_id]: [group_id] for group_id in group_ids
        }
        split_by_component = {}
        split_by_game = provisional_split_by_game

    records: list[dict[str, Any]] = []
    for row, (position, encoding, frame) in enumerate(
        zip(positions, encodings, frames, strict=True)
    ):
        records.append(
            _sample_record(
                position,
                split=split_by_game[position.split_group_id or position.game_id],
                encoding=encoding,
                frame=frame,
                cached_features=features[row],
                feature_row=row,
                lif_seed=derive_lif_seed(
                    position,
                    base_seed=resolved_lif_seed_base,
                    mode=lif_seed_mode,
                ),
            )
        )

    split_groups = {
        name: sorted(game for game, split in split_by_game.items() if split == name)
        for name in ("train", "validation")
    }
    split_games = {
        name: sorted(
            {
                item.game_id
                for item in positions
                if split_by_game[item.split_group_id or item.game_id] == name
            }
        )
        for name in ("train", "validation")
    }
    provisional_feature_records: dict[str, dict[str, list[dict[str, Any]]]] = {
        "train": {},
        "validation": {},
    }
    for position, encoding, frame in zip(positions, encodings, frames, strict=True):
        group_id = position.split_group_id or position.game_id
        provisional_split = provisional_split_by_game[group_id]
        feature_hash = str(frame["output_pool"]["features_hash"])
        provisional_feature_records[provisional_split].setdefault(
            feature_hash, []
        ).append(
            {
                "game_id": position.game_id,
                "split_group_id": group_id,
                "ply": position.ply,
                "label_action": position.teacher_action,
                "supervision_kind": position.supervision_kind,
                "encoding_sha256": str(encoding["encoding_hash"]),
            }
        )
    provisional_feature_overlap = sorted(
        set(provisional_feature_records["train"])
        & set(provisional_feature_records["validation"])
    )
    provisional_feature_overlap_examples = [
        {
            "features_sha256": feature_hash,
            "train": provisional_feature_records["train"][feature_hash][:3],
            "validation": provisional_feature_records["validation"][feature_hash][:3],
        }
        for feature_hash in provisional_feature_overlap[:20]
    ]
    state_hashes_by_split = {
        name: {
            item["hashes"]["state_sha256"]
            for item in records
            if item["split"] == name
        }
        for name in ("train", "validation")
    }
    state_hash_overlap = sorted(
        state_hashes_by_split["train"] & state_hashes_by_split["validation"]
    )
    if trajectory_mode == "role-symmetric" and state_hash_overlap:
        raise RuntimeError("train/validation state hash overlap detected")
    encoding_hashes_by_split = {
        name: {
            item["hashes"]["encoding_sha256"]
            for item in records
            if item["split"] == name
        }
        for name in ("train", "validation")
    }
    encoding_hash_overlap = sorted(
        encoding_hashes_by_split["train"] & encoding_hashes_by_split["validation"]
    )
    if trajectory_mode == "role-symmetric" and encoding_hash_overlap:
        raise RuntimeError("train/validation encoding hash overlap detected")
    feature_hashes_by_split = {
        name: {
            item["hashes"]["features_sha256"]
            for item in records
            if item["split"] == name
        }
        for name in ("train", "validation")
    }
    feature_hash_overlap = sorted(
        feature_hashes_by_split["train"] & feature_hashes_by_split["validation"]
    )
    if trajectory_mode == "role-symmetric" and feature_hash_overlap:
        raise RuntimeError("train/validation feature hash overlap detected")
    pass_response_records = [
        item
        for item in records
        if item["supervision_provenance"]["kind"]
        == "constructed-opponent-pass-response"
    ]
    if any(
        item["label_action"] != PASS_ACTION_INDEX
        or item["state"]["consecutive_passes"] != 1
        or item["state"]["to_play"] != item["teacher_colour"]
        for item in pass_response_records
    ):
        raise RuntimeError("invalid constructed pass-response supervision record")

    # Do not overwrite an existing valid cache until every leakage and
    # provenance check above has passed.
    cache_directory.mkdir(parents=True, exist_ok=True)
    records_path = cache_directory / "samples.jsonl"
    records_path.write_bytes(b"".join(canonical_bytes(item) + b"\n" for item in records))
    features_path = cache_directory / "features.npy"
    labels_path = cache_directory / "labels.npy"
    with features_path.open("wb") as stream:
        np.save(stream, features, allow_pickle=False)
    with labels_path.open("wb") as stream:
        np.save(stream, labels, allow_pickle=False)

    manifest: dict[str, Any] = {
        "schema": DATASET_SCHEMA,
        "seed": seed,
        "teacher": {
            "policy_id": BASELINE_POLICY_ID,
            "scope": "offline_supervised_labels_only",
            "included_in_runtime_checkpoint": False,
        },
        "generator": {
            "game_count": game_count,
            "max_moves": max_moves,
            "validation_fraction": validation_fraction,
            "black_policy": (
                "role_dependent_see_trajectories"
                if trajectory_mode == "role-symmetric"
                else "seeded_legal_exploration_with_capture_mix-v1"
            ),
            "white_policy": (
                "role_dependent_see_trajectories"
                if trajectory_mode == "role-symmetric"
                else BASELINE_POLICY_ID
            ),
            "retained_colours": (
                "controlled-teacher-only"
                if trajectory_mode == "role-symmetric"
                else retained_colours
            ),
            "legacy_retained_colours_argument": retained_colours,
            "trajectory_mode": trajectory_mode,
            "pass_response_samples_per_trajectory": pass_response_samples_per_trajectory,
            "trajectories": (
                [
                    "teacher-black-vs-seeded-random-white",
                    "teacher-white-vs-seeded-random-black",
                ]
                if trajectory_mode == "role-symmetric"
                else ["seeded-random-black-vs-teacher-white"]
            ),
            "random_stream_policy": (
                "shared_per_game_pair_independent_of_teacher_colour-v1"
                if trajectory_mode == "role-symmetric"
                else "legacy_game_id_seeded-v1"
            ),
            "lif_parameters": asdict(parameters),
            "lif_seed_mode": lif_seed_mode,
            "lif_seed_base": resolved_lif_seed_base,
        },
        "graph": {
            "model_id": graph.model_id,
            "manifest_sha256": graph.manifest_sha256,
        },
        "output_pool": {
            "version": records[0]["output_pool_version"],
            "group_hash": records[0]["output_pool_group_hash"],
            "feature_count": FEATURE_COUNT,
        },
        "sample_count": len(records),
        "split_games": split_games,
        "split_groups": split_groups,
        "split_sample_counts": {
            name: sum(item["split"] == name for item in records)
            for name in ("train", "validation")
        },
        "teacher_action_coverage": sorted(set(int(value) for value in labels)),
        "teacher_action_coverage_count": len(set(int(value) for value in labels)),
        "teacher_label_legal_rate": float(
            np.mean(
                [
                    bool(item["legal_mask"][int(label)])
                    for item, label in zip(records, labels, strict=True)
                ]
            )
        ),
        "capture_sample_count": sum(item["teacher_captures"] > 0 for item in records),
        "late_position_count": sum(item["ply"] >= max_moves * 3 // 4 for item in records),
        "trajectory_sample_counts": {
            kind: sum(item.get("trajectory_kind") == kind for item in records)
            for kind in sorted({item.get("trajectory_kind", "legacy") for item in records})
        },
        "pass_response_augmentation": {
            "enabled": pass_response_samples_per_trajectory > 0,
            "schema": "pass-response-augmentation-v1",
            "method": "opponent-pass-applied-to-late-legal-state",
            "late_state_threshold": "ply >= floor(max_moves * 3 / 4)",
            "natural_trajectory_claim": False,
            "constructed_states_are_terminal": False,
            "requested_samples_per_trajectory": pass_response_samples_per_trajectory,
            "sample_count": len(pass_response_records),
            "split_sample_counts": {
                name: sum(
                    item["split"] == name for item in pass_response_records
                )
                for name in ("train", "validation")
            },
            "trajectory_sample_counts": {
                game_id: sum(item["game_id"] == game_id for item in pass_response_records)
                for game_id in sorted({item["game_id"] for item in pass_response_records})
            },
        },
        "state_hash_integrity": {
            "deduplication": (
                "first_complete_state_occurrence"
                if trajectory_mode == "role-symmetric"
                else "not_applied_legacy_compatibility"
            ),
            "generated_position_count": generated_position_count,
            "duplicate_state_count_removed": duplicate_state_count,
            "retained_unique_state_count": len(records),
            "train_unique_state_count": len(state_hashes_by_split["train"]),
            "validation_unique_state_count": len(state_hashes_by_split["validation"]),
            "train_validation_overlap_count": len(state_hash_overlap),
            "train_validation_overlap_sha256": object_hash(state_hash_overlap),
        },
        "model_input_integrity": {
            "split_unit": "paired_trajectory_group",
            "split_strategy": (
                "feature_equivalence_components_then_deterministic_split-v1"
                if trajectory_mode == "role-symmetric"
                else "legacy_deterministic_game_split-v1"
            ),
            "split_uses_teacher_labels": False,
            "feature_equivalence_component_count": len(component_groups),
            "multi_group_feature_component_count": sum(
                len(groups) > 1 for groups in component_groups.values()
            ),
            "max_groups_per_feature_component": max(
                (len(groups) for groups in component_groups.values()), default=0
            ),
            "feature_component_membership_sha256": object_hash(component_groups),
            "split_component_ids": {
                name: sorted(
                    {
                        component_id_by_group[group_id]
                        for group_id, split in split_by_game.items()
                        if split == name
                    }
                )
                for name in ("train", "validation")
            },
            "provisional_paired_group_split_feature_overlap_count": len(
                provisional_feature_overlap
            ),
            "provisional_paired_group_split_feature_overlap_sha256": object_hash(
                provisional_feature_overlap
            ),
            "provisional_paired_group_split_feature_overlap_examples": (
                provisional_feature_overlap_examples
            ),
            "encoding_deduplication": (
                "deterministic_representative_before_lif"
                if trajectory_mode == "role-symmetric"
                else "not_applied_legacy_compatibility"
            ),
            "state_unique_position_count_before_encoding_dedup": state_unique_position_count,
            "encoding_duplicate_count_removed": encoding_duplicate_count,
            "encoding_label_conflict_count": encoding_label_conflict_count,
            "encoding_unique_count": len(
                {item["hashes"]["encoding_sha256"] for item in records}
            ),
            "train_encoding_unique_count": len(encoding_hashes_by_split["train"]),
            "validation_encoding_unique_count": len(
                encoding_hashes_by_split["validation"]
            ),
            "train_validation_encoding_overlap_count": len(encoding_hash_overlap),
            "train_validation_encoding_overlap_sha256": object_hash(
                encoding_hash_overlap
            ),
            "feature_unique_count": len(
                {item["hashes"]["features_sha256"] for item in records}
            ),
            "train_feature_unique_count": len(feature_hashes_by_split["train"]),
            "validation_feature_unique_count": len(
                feature_hashes_by_split["validation"]
            ),
            "train_validation_feature_overlap_count": len(feature_hash_overlap),
            "train_validation_feature_overlap_sha256": object_hash(
                feature_hash_overlap
            ),
        },
        "artifacts": {
            "samples.jsonl": {"sha256": file_hash(records_path)},
            "features.npy": {"sha256": file_hash(features_path), "shape": list(features.shape)},
            "labels.npy": {"sha256": file_hash(labels_path), "shape": list(labels.shape)},
        },
    }
    manifest["dataset_hash"] = object_hash(manifest)
    _write_json(cache_directory / "manifest.json", manifest)
    return manifest


def load_cache(cache_directory: Path) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    manifest_path = cache_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_dataset_hash = manifest.pop("dataset_hash")
    if object_hash(manifest) != expected_dataset_hash:
        raise ValueError("dataset manifest hash mismatch")
    manifest["dataset_hash"] = expected_dataset_hash
    for name, metadata in manifest["artifacts"].items():
        if file_hash(cache_directory / name) != metadata["sha256"]:
            raise ValueError(f"cached artifact hash mismatch: {name}")
    features = np.load(cache_directory / "features.npy", allow_pickle=False)
    labels = np.load(cache_directory / "labels.npy", allow_pickle=False)
    records = [
        json.loads(line)
        for line in (cache_directory / "samples.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    if features.shape != (len(records), FEATURE_COUNT) or labels.shape != (len(records),):
        raise ValueError("cached array shapes do not match JSONL records")
    for row, record in enumerate(records):
        if record["feature_row"] != row:
            raise ValueError("feature_row is not aligned with the NPY arrays")
        core = dict(record)
        claimed = core.pop("record_sha256")
        if object_hash(core) != claimed:
            raise ValueError(f"sample record hash mismatch at row {row}")
        if object_hash(record["state"]) != record["hashes"]["state_sha256"]:
            raise ValueError(f"state hash mismatch at row {row}")
        cached_feature_hash = object_hash(
            {
                "dtype": "float32",
                "values": [float(value) for value in features[row]],
            }
        )
        if cached_feature_hash != record["hashes"]["cached_feature_row_sha256"]:
            raise ValueError(f"cached feature row hash mismatch at row {row}")
        if object_hash({"teacher": record["label_teacher"], "action": int(labels[row])}) != record["hashes"]["label_sha256"]:
            raise ValueError(f"label hash mismatch at row {row}")
    return features, labels, records, manifest


def fit_ridge_readout(
    features: np.ndarray, labels: np.ndarray, *, ridge: float
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a deterministic multi-output ridge least-squares classifier."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if x.ndim != 2 or x.shape[1] != FEATURE_COUNT or y.shape != (x.shape[0],):
        raise ValueError("expected features [N,128] and labels [N]")
    if x.shape[0] == 0 or np.any(y < 0) or np.any(y >= ACTION_COUNT):
        raise ValueError("training samples or labels are invalid")
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge must be finite and positive")
    design = np.column_stack((x, np.ones(x.shape[0], dtype=np.float64)))
    targets = np.eye(ACTION_COUNT, dtype=np.float64)[y]
    penalty = np.eye(FEATURE_COUNT + 1, dtype=np.float64) * ridge
    penalty[-1, -1] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ targets)
    return coefficients[:-1].T.astype(np.float32), coefficients[-1].astype(np.float32)


def evaluate_readout(
    features: np.ndarray,
    labels: np.ndarray,
    legal_masks: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
) -> dict[str, Any]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    masks = np.asarray(legal_masks, dtype=bool)
    scores = x @ np.asarray(weights, dtype=np.float64).T + np.asarray(bias, dtype=np.float64)
    if masks.shape != scores.shape or y.shape != (scores.shape[0],):
        raise ValueError("evaluation arrays have incompatible shapes")
    raw_top1 = np.argmax(scores, axis=1)
    # Stable mergesort plus explicit action order makes ties deterministic.
    top5 = np.argsort(-scores, axis=1, kind="stable")[:, :5]
    masked_scores = np.where(masks, scores, -np.inf)
    masked_top1 = np.argmax(masked_scores, axis=1)
    count = int(y.size)
    return {
        "sample_count": count,
        "top1_accuracy": float(np.mean(raw_top1 == y)),
        "top5_accuracy": float(np.mean(np.any(top5 == y[:, None], axis=1))),
        "legal_action_rate": float(np.mean(masks[np.arange(count), raw_top1])),
        "teacher_agreement_rate": float(np.mean(masked_top1 == y)),
    }


def _record_rule_pass_gate(record: Mapping[str, Any]) -> bool:
    """Mirror the public-state pass rule used by the hybrid runtime policy."""
    legal_mask = record.get("legal_mask")
    if not isinstance(legal_mask, list) or len(legal_mask) != ACTION_COUNT:
        raise ValueError("sample legal_mask must contain 82 values")
    state = record.get("state")
    if not isinstance(state, Mapping):
        raise ValueError("sample state is missing")
    last_move_was_pass = state.get("consecutive_passes") == 1
    only_pass_legal = bool(legal_mask[PASS_ACTION_INDEX]) and sum(
        bool(value) for value in legal_mask
    ) == 1
    return last_move_was_pass or only_pass_legal


def evaluate_controlled_readout(
    features: np.ndarray,
    labels: np.ndarray,
    records: Sequence[Mapping[str, Any]],
    weights: np.ndarray,
    bias: np.ndarray,
    *,
    pass_control: str,
) -> dict[str, Any]:
    """Evaluate the action actually selected by the declared controller."""
    if pass_control not in PASS_CONTROL_MODES:
        raise ValueError(f"pass_control must be one of {PASS_CONTROL_MODES}")
    legal_masks = np.asarray([item["legal_mask"] for item in records], dtype=bool)
    if pass_control == "joint-neural":
        return evaluate_readout(features, labels, legal_masks, weights, bias)

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    scores = x @ np.asarray(weights, dtype=np.float64).T + np.asarray(
        bias, dtype=np.float64
    )
    if legal_masks.shape != scores.shape or y.shape != (scores.shape[0],):
        raise ValueError("evaluation arrays have incompatible shapes")
    gate = np.asarray([_record_rule_pass_gate(item) for item in records], dtype=bool)
    point_masks = legal_masks.copy()
    point_masks[:, PASS_ACTION_INDEX] = False
    if np.any(~gate & ~np.any(point_masks, axis=1)):
        raise ValueError("rule pass gate left a sample with no legal point action")
    selected = np.full(y.shape, PASS_ACTION_INDEX, dtype=np.int64)
    selected[~gate] = np.argmax(
        np.where(point_masks[~gate], scores[~gate], -np.inf), axis=1
    )
    point_rankings = np.argsort(
        -np.where(point_masks, scores, -np.inf), axis=1, kind="stable"
    )[:, :5]
    top5_match = np.where(
        gate,
        y == PASS_ACTION_INDEX,
        np.any(point_rankings == y[:, None], axis=1),
    )
    count = int(y.size)
    agreement = float(np.mean(selected == y))
    return {
        "sample_count": count,
        "top1_accuracy": agreement,
        "top5_accuracy": float(np.mean(top5_match)),
        "legal_action_rate": float(
            np.mean(legal_masks[np.arange(count), selected])
        ),
        "teacher_agreement_rate": agreement,
    }


def train_from_cache(
    *,
    cache_directory: Path,
    checkpoint_directory: Path,
    ridge: float,
    pass_control: str = "joint-neural",
) -> dict[str, Any]:
    if pass_control not in PASS_CONTROL_MODES:
        raise ValueError(f"pass_control must be one of {PASS_CONTROL_MODES}")
    features, labels, records, dataset = load_cache(cache_directory)
    train_rows = np.asarray([item["split"] == "train" for item in records], dtype=bool)
    validation_rows = ~train_rows
    if not np.any(train_rows) or not np.any(validation_rows):
        raise ValueError("cache must contain both train and validation samples")
    legal_masks = np.asarray([item["legal_mask"] for item in records], dtype=bool)
    natural_rows = np.asarray(
        [
            item.get("supervision_provenance", {}).get(
                "kind", "natural-teacher-turn"
            )
            == "natural-teacher-turn"
            for item in records
        ],
        dtype=bool,
    )
    pass_response_rows = np.asarray(
        [
            item.get("supervision_provenance", {}).get("kind")
            == "constructed-opponent-pass-response"
            for item in records
        ],
        dtype=bool,
    )
    if pass_control == "rule-after-opponent-pass":
        fit_rows = train_rows & natural_rows & (labels != PASS_ACTION_INDEX)
    else:
        fit_rows = train_rows
    if not np.any(fit_rows):
        raise ValueError("pass-control selection left no samples for ridge fitting")
    weights, bias = fit_ridge_readout(
        features[fit_rows], labels[fit_rows], ridge=ridge
    )

    def evaluate_rows(rows: np.ndarray) -> dict[str, Any]:
        if not np.any(rows):
            return {
                "sample_count": 0,
                "top1_accuracy": None,
                "top5_accuracy": None,
                "legal_action_rate": None,
                "teacher_agreement_rate": None,
            }
        selected_records = [
            record for record, selected in zip(records, rows, strict=True) if selected
        ]
        return evaluate_controlled_readout(
            features[rows],
            labels[rows],
            selected_records,
            weights,
            bias,
            pass_control=pass_control,
        )

    def control_metrics(rows: np.ndarray) -> dict[str, Any]:
        selected_records = [
            record for record, selected in zip(records, rows, strict=True) if selected
        ]
        selected_labels = labels[rows]
        gates = np.asarray(
            [_record_rule_pass_gate(record) for record in selected_records], dtype=bool
        )
        pass_targets = selected_labels == PASS_ACTION_INDEX
        point_targets = ~pass_targets
        natural_point_rows = rows & natural_rows & (labels != PASS_ACTION_INDEX)
        natural_point_result = evaluate_rows(natural_point_rows)
        return {
            "natural_point": {
                "sample_count": natural_point_result["sample_count"],
                "agreement_rate": natural_point_result["teacher_agreement_rate"],
            },
            "rule_pass_gate": {
                "target_pass_sample_count": int(np.sum(pass_targets)),
                "hit_count": int(np.sum(gates & pass_targets)),
                "hit_rate": (
                    float(np.mean(gates[pass_targets]))
                    if np.any(pass_targets)
                    else None
                ),
                "non_pass_sample_count": int(np.sum(point_targets)),
                "false_pass_count": int(np.sum(gates & point_targets)),
                "false_pass_rate": (
                    float(np.mean(gates[point_targets]))
                    if np.any(point_targets)
                    else None
                ),
                "triggered_count": int(np.sum(gates)),
                "learned_by_neural_readout": False,
            },
        }

    metrics = {
        "train": evaluate_rows(train_rows),
        "validation": evaluate_rows(validation_rows),
        "stratified": {
            "train": {
                "natural": evaluate_rows(train_rows & natural_rows),
                "pass_response": evaluate_rows(train_rows & pass_response_rows),
            },
            "validation": {
                "natural": evaluate_rows(validation_rows & natural_rows),
                "pass_response": evaluate_rows(validation_rows & pass_response_rows),
            },
        },
        "model_selection_metric": (
            "stratified.validation.natural.teacher_agreement_rate"
        ),
        "overall_metrics_include_constructed_pass_response": bool(
            np.any(pass_response_rows)
        ),
        "pass_control": {
            "mode": pass_control,
            "fit_sample_count": int(np.sum(fit_rows)),
            "train": control_metrics(train_rows),
            "validation": control_metrics(validation_rows),
        },
    }
    method = (
        "closed_form_ridge_natural_point_only_plus_rule_pass_gate_v1"
        if pass_control == "rule-after-opponent-pass"
        else "closed_form_ridge_one_hot_v1"
    )
    controller_sources = (
        {
            "point_actions": "malecns-linear-readout",
            "pass_after_opponent_pass": "go-rule-pass-gate",
            "pass_when_only_legal": "go-rule-pass-gate",
        }
        if pass_control == "rule-after-opponent-pass"
        else {"all_actions": "malecns-linear-readout"}
    )
    decision_architecture = (
        "hybrid-rule-after-opponent-pass-v1"
        if pass_control == "rule-after-opponent-pass"
        else "joint-neural-82-action-v1"
    )
    training_info = {
        "schema": TRAINING_SCHEMA,
        "method": method,
        "algorithm": method,
        "pass_control": pass_control,
        "controller_sources": controller_sources,
        "decision_architecture": decision_architecture,
        "neural_action_space": (
            "point-actions-0..80"
            if pass_control == "rule-after-opponent-pass"
            else "point-actions-0..80-plus-pass-81"
        ),
        "pass_selection_source": (
            "deterministic-go-rule"
            if pass_control == "rule-after-opponent-pass"
            else "malecns-linear-readout"
        ),
        "fit_sample_count": int(np.sum(fit_rows)),
        "ridge": ridge,
        "seed": dataset["seed"],
        "dataset_id": dataset["dataset_hash"],
        "dataset_hash": dataset["dataset_hash"],
        "completed_at": "omitted_for_byte_reproducibility",
        "teacher": dataset["teacher"],
        "sample_count": dataset["sample_count"],
        "split_sample_counts": dataset["split_sample_counts"],
        "split_games": dataset["split_games"],
        "teacher_action_coverage": dataset["teacher_action_coverage"],
        "teacher_action_coverage_count": dataset["teacher_action_coverage_count"],
        "teacher_label_legal_rate": dataset["teacher_label_legal_rate"],
        "capture_sample_count": dataset["capture_sample_count"],
        "late_position_count": dataset["late_position_count"],
        "generator": dataset["generator"],
        "trajectory_sample_counts": dataset["trajectory_sample_counts"],
        "pass_response_augmentation": dataset["pass_response_augmentation"],
        "state_hash_integrity": dataset["state_hash_integrity"],
        "model_input_integrity": dataset["model_input_integrity"],
        "metrics": metrics,
        "runtime_dependency": "linear readout only; no teacher/fallback call",
    }
    from src.malecns_policy import write_policy_checkpoint

    checkpoint_hash = write_policy_checkpoint(
        checkpoint_directory,
        weights=weights,
        bias=bias,
        model_id=dataset["graph"]["model_id"],
        graph_manifest_sha256=dataset["graph"]["manifest_sha256"],
        output_pool_group_hash=dataset["output_pool"]["group_hash"],
        training_status="trained",
        training_info=training_info,
    )
    checkpoint_manifest = json.loads(
        (checkpoint_directory / "manifest.json").read_text(encoding="utf-8")
    )
    result = {
        "dataset_manifest": dataset,
        "checkpoint_manifest": checkpoint_manifest,
        "checkpoint_hash": checkpoint_hash,
        "metrics": metrics,
    }
    _write_json(cache_directory / "training_result.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--max-moves", type=int, default=72)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--duration-ms", type=float, default=12.0)
    parser.add_argument("--ridge", type=float, default=0.01)
    parser.add_argument(
        "--pass-control",
        choices=PASS_CONTROL_MODES,
        default="joint-neural",
        help="joint neural PASS selection, or an explicit public-state rule PASS gate",
    )
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--lif-seed-mode", choices=LIF_SEED_MODES, default="fixed")
    parser.add_argument("--lif-seed-base", type=int)
    parser.add_argument(
        "--retained-colours",
        choices=RETAINED_COLOUR_MODES,
        default="white-only",
    )
    parser.add_argument(
        "--trajectory-mode",
        choices=TRAJECTORY_MODES,
        default="legacy",
        help="legacy keeps historical generation; role-symmetric creates paired teacher roles",
    )
    parser.add_argument(
        "--pass-response-samples-per-trajectory",
        type=int,
        default=0,
        help="role-symmetric only: late offline opponent-pass response branches per trajectory",
    )
    parser.add_argument("--reuse-cache", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    started = time.perf_counter()
    if not args.reuse_cache:
        generate_cache(
            graph_directory=args.graph,
            cache_directory=args.cache,
            game_count=args.games,
            max_moves=args.max_moves,
            seed=args.seed,
            validation_fraction=args.validation_fraction,
            duration_ms=args.duration_ms,
            progress_every=args.progress_every,
            lif_seed_mode=args.lif_seed_mode,
            lif_seed_base=args.lif_seed_base,
            retained_colours=args.retained_colours,
            trajectory_mode=args.trajectory_mode,
            pass_response_samples_per_trajectory=args.pass_response_samples_per_trajectory,
        )
    result = train_from_cache(
        cache_directory=args.cache,
        checkpoint_directory=args.checkpoint,
        ridge=args.ridge,
        pass_control=args.pass_control,
    )
    summary = {
        "status": "success",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "cache": str(args.cache.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset_hash": result["dataset_manifest"]["dataset_hash"],
        "metrics": result["metrics"],
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
