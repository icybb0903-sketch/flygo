"""Read-only diagnostic of where the Go-to-MaleCNS adapter loses signal.

The direct encoding and stimulus regressions are probes, not deployable
MaleCNS policies. They use the frozen cache split and never write a checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_neural_go_readout import load_cache
from src.go_engine import GoState
from src.go_neural_encoding import PASS_ACTION_INDEX, encode_go_state
from src.malecns_dynamics import DYNAMICS_VERSION, go_encoding_to_stimulus


def _fixed_seed_input_schedule(rates: np.ndarray, manifest: dict[str, object]) -> np.ndarray:
    """Reproduce only the actual 32-channel forced-spike schedule, not graph activity."""
    generator = manifest["generator"]
    if generator["lif_seed_mode"] != "fixed":
        raise ValueError("input schedule diagnostic requires fixed LIF seed mode")
    params = generator["lif_parameters"]
    dt_ms = float(params["dt_ms"])
    tick_count = int(np.ceil(float(params["duration_ms"]) / dt_ms))
    seed = int(generator["lif_seed_base"])
    phases = np.asarray([
        int.from_bytes(
            hashlib.sha256(f"{seed}|{channel}|{DYNAMICS_VERSION}".encode("ascii")).digest()[:8],
            "little",
        ) / 2**64
        for channel in range(rates.shape[1])
    ])
    cumulative = phases[None, None, :] + (
        np.arange(1, tick_count + 1)[None, :, None]
        * rates[:, None, :] * dt_ms / 1000.0
    )
    previous = np.concatenate((
        np.broadcast_to(phases, (rates.shape[0], 1, rates.shape[1])),
        cumulative[:, :-1, :],
    ), axis=1)
    return np.floor(cumulative) > np.floor(previous)


def _fit_and_score(
    values: np.ndarray,
    labels: np.ndarray,
    masks: np.ndarray,
    fit_rows: np.ndarray,
    validation_rows: np.ndarray,
    ridge: float,
) -> dict[str, float | int]:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] != labels.shape[0]:
        raise ValueError("feature rows do not match labels")
    design = np.column_stack((x[fit_rows], np.ones(int(fit_rows.sum()))))
    targets = np.eye(PASS_ACTION_INDEX, dtype=np.float64)[labels[fit_rows]]
    penalty = np.eye(design.shape[1], dtype=np.float64) * ridge
    penalty[-1, -1] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + penalty, design.T @ targets
    )
    validation_design = np.column_stack(
        (x[validation_rows], np.ones(int(validation_rows.sum())))
    )
    scores = validation_design @ coefficients
    legal_scores = np.where(masks[validation_rows, :PASS_ACTION_INDEX], scores, -np.inf)
    selected = np.argmax(legal_scores, axis=1)
    return {
        "feature_count": int(x.shape[1]),
        "fit_sample_count": int(fit_rows.sum()),
        "validation_sample_count": int(validation_rows.sum()),
        "validation_teacher_matches": int(np.sum(selected == labels[validation_rows])),
        "validation_teacher_agreement_rate": float(
            np.mean(selected == labels[validation_rows])
        ),
    }


def probe(cache: Path, ridge: float) -> dict[str, object]:
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge must be finite and positive")
    neural, labels, records, manifest = load_cache(cache)
    encodings = [encode_go_state(GoState.from_dict(item["state"])) for item in records]
    raw = np.asarray([item["vector"] for item in encodings], dtype=np.float64)
    stimuli = [go_encoding_to_stimulus(item) for item in encodings]
    rates = np.asarray([item["rates_hz"] for item in stimuli], dtype=np.float64)
    input_schedule = _fixed_seed_input_schedule(rates, manifest)
    masks = np.asarray([item["legal_mask"] for item in records], dtype=bool)
    natural_point = np.asarray(
        [
            item.get("supervision_provenance", {}).get("kind") == "natural-teacher-turn"
            and label != PASS_ACTION_INDEX
            for item, label in zip(records, labels, strict=True)
        ],
        dtype=bool,
    )
    train = np.asarray([item["split"] == "train" for item in records]) & natural_point
    validation = np.asarray([item["split"] == "validation" for item in records]) & natural_point
    if not np.any(train) or not np.any(validation):
        raise ValueError("cache has no natural point examples in one split")
    singular_values = np.linalg.svd(neural - neural.mean(axis=0), compute_uv=False)
    explained = np.cumsum(singular_values**2) / np.sum(singular_values**2)
    # These alternate 32-dimensional projections are diagnostics only.  None
    # of them is fed into the MaleCNS graph or exposed as a deployed policy.
    planes = raw[:, :405].reshape(-1, 5, 9, 9)
    spatial = np.empty((raw.shape[0], 32), dtype=np.float64)
    for tile_row in range(4):
        row_start, row_end = (0, 2, 4, 6)[tile_row], (2, 4, 6, 9)[tile_row]
        for tile_col in range(4):
            col_start, col_end = (0, 2, 4, 6)[tile_col], (2, 4, 6, 9)[tile_col]
            tile = planes[:, :2, row_start:row_end, col_start:col_end]
            spatial[:, tile_row * 4 + tile_col] = tile[:, 0].mean(axis=(1, 2))
            spatial[:, 16 + tile_row * 4 + tile_col] = tile[:, 1].mean(axis=(1, 2))

    # Fit PCA on training states only.  The validation labels never enter it.
    center = raw[train].mean(axis=0)
    centered_train = raw[train] - center
    covariance = centered_train.T @ centered_train
    _, eigenvectors = np.linalg.eigh(covariance)
    pca_32 = (raw - center) @ eigenvectors[:, -32:]
    # A label-blind, fixed signed projection tests whether the current positive
    # bucket averages discard useful spatial contrasts.  This is an offline
    # proxy only: these numbers have not been run through MaleCNS.
    feature_upper = np.asarray(
        [1.0] * 405 + [1.0, 1.0, 81.0, 81.0, 2.0, 1.0, 1.0, 1.0, 162.0, 20.0],
        dtype=np.float64,
    )
    normalised = np.clip(raw / feature_upper, 0.0, 1.0)
    random_signs = np.random.default_rng(20260924).choice(
        (-1.0, 1.0), size=(raw.shape[1], 32)
    )
    signed_projection = normalised @ random_signs / np.sqrt(raw.shape[1])
    # Match rough input scales for a diagnostic hybrid readout.  The factor is
    # fixed before seeing either split's labels; no deployment path uses it.
    hybrid = np.column_stack((raw, 50.0 * neural))
    return {
        "schema": "stage8-adapter-information-probe-v1",
        "scope": "read-only adapter diagnostic; direct encoding/stimulus are not deployed policies",
        "dataset_hash": manifest["dataset_hash"],
        "ridge": ridge,
        "split": "unchanged cached train/validation groups",
        "probes": {
            "direct_go_encoding_415": _fit_and_score(
                raw, labels, masks, train, validation, ridge
            ),
            "stimulus_rates_32": _fit_and_score(
                rates, labels, masks, train, validation, ridge
            ),
            "spatial_tiles_32_diagnostic": _fit_and_score(
                spatial, labels, masks, train, validation, ridge
            ),
            "equal_hash_spatial_blend_32_diagnostic": _fit_and_score(
                0.5 * (rates / 150.0) + 0.5 * spatial,
                labels, masks, train, validation, ridge,
            ),
            "train_only_pca_32_diagnostic": _fit_and_score(
                pca_32, labels, masks, train, validation, ridge
            ),
            "fixed_signed_projection_32_diagnostic": _fit_and_score(
                signed_projection, labels, masks, train, validation, ridge
            ),
            "malecns_output_128": _fit_and_score(
                neural, labels, masks, train, validation, ridge
            ),
            "direct_plus_malecns_diagnostic": _fit_and_score(
                hybrid, labels, masks, train, validation, ridge
            ),
        },
        "distinct_stimulus_vectors": len({tuple(row) for row in rates}),
        "distinct_fixed_seed_input_spike_schedules": len({
            row.tobytes() for row in input_schedule
        }),
        "mean_forced_input_spikes_per_frame": float(np.mean(input_schedule.sum(axis=(1, 2)))),
        "distinct_signed_projection_vectors": len({tuple(row) for row in signed_projection}),
        "distinct_neural_feature_vectors": len({tuple(row) for row in neural}),
        "neural_output_information": {
            "nonconstant_dimensions": int(np.count_nonzero(np.std(neural, axis=0) > 1e-8)),
            "mean_nonzero_dimensions_per_frame": float(np.mean(np.count_nonzero(neural > 0, axis=1))),
            "median_nonzero_dimensions_per_frame": float(np.median(np.count_nonzero(neural > 0, axis=1))),
            "components_for_95_percent_variance": int(np.searchsorted(explained, 0.95) + 1),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--ridge", type=float, default=0.1)
    args = parser.parse_args()
    print(json.dumps(probe(args.cache, args.ridge), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
