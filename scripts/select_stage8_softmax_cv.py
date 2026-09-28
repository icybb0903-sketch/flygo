"""Select softmax regularization using only Stage8 training trajectory groups.

The old held-out validation split is not consulted.  Groups linked by an
identical cached MaleCNS feature vector remain in the same inner fold.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_neural_go_readout import canonical_bytes, load_cache, object_hash
from scripts.train_stage8_legal_softmax import fit_legal_softmax
from src.go_neural_encoding import PASS_ACTION_INDEX


L2_CANDIDATES = (0.005, 0.02, 0.08, 0.2, 0.5)
FOLD_SEED = 20260924


def select_l2(cache: Path, *, steps: int = 400) -> dict[str, object]:
    features, labels, records, manifest = load_cache(cache)
    selected = np.asarray([
        row["split"] == "train"
        and row.get("supervision_provenance", {}).get("kind") == "natural-teacher-turn"
        and label != PASS_ACTION_INDEX
        for row, label in zip(records, labels, strict=True)
    ], dtype=bool)
    groups = sorted({row["split_group_id"] for row, use in zip(records, selected, strict=True) if use})
    parent = {group: group for group in groups}

    def root(group: str) -> str:
        while parent[group] != group:
            parent[group] = parent[parent[group]]
            group = parent[group]
        return group

    by_feature: dict[str, set[str]] = defaultdict(set)
    for row, use in zip(records, selected, strict=True):
        if use:
            by_feature[row["hashes"]["features_sha256"]].add(row["split_group_id"])
    for linked in by_feature.values():
        ordered = sorted(linked)
        for group in ordered[1:]:
            parent[root(group)] = root(ordered[0])

    components: dict[str, list[str]] = defaultdict(list)
    for group in groups:
        components[root(group)].append(group)
    if len(components) < 3:
        raise ValueError("need at least three feature-disjoint train components")
    component_keys = sorted(components)
    np.random.default_rng(FOLD_SEED).shuffle(component_keys)
    fold_by_group = {
        group: index % 3
        for index, key in enumerate(component_keys)
        for group in components[key]
    }
    folds = np.asarray([
        fold_by_group.get(row["split_group_id"], -1) if use else -1
        for row, use in zip(records, selected, strict=True)
    ])
    masks = np.asarray([row["legal_mask"] for row in records], dtype=bool)
    results = []
    for l2 in L2_CANDIDATES:
        fold_results = []
        for fold in range(3):
            fit = selected & (folds != fold)
            holdout = selected & (folds == fold)
            weights, bias, _ = fit_legal_softmax(
                features[fit], labels[fit], masks[fit], steps=steps, l2=l2
            )
            scores = features[holdout] @ weights[:PASS_ACTION_INDEX].T + bias[:PASS_ACTION_INDEX]
            prediction = np.argmax(
                np.where(masks[holdout, :PASS_ACTION_INDEX], scores, -np.inf),
                axis=1,
            )
            fold_results.append({
                "fold": fold,
                "sample_count": int(holdout.sum()),
                "matches": int(np.sum(prediction == labels[holdout])),
            })
        matches = sum(row["matches"] for row in fold_results)
        count = sum(row["sample_count"] for row in fold_results)
        results.append({
            "l2": l2,
            "matches": matches,
            "sample_count": count,
            "agreement_rate": matches / count,
            "folds": fold_results,
        })
    chosen = max(results, key=lambda row: (row["matches"], row["l2"]))
    return {
        "schema": "stage8-softmax-train-only-group-cv-v2",
        "dataset_hash": manifest["dataset_hash"],
        "scope": "train-split-only selection; old validation labels not used",
        "fold_seed": FOLD_SEED,
        "fold_count": 3,
        "train_group_count": len(groups),
        "feature_disjoint_component_count": len(components),
        "fit_sample_count": int(selected.sum()),
        "steps": steps,
        "candidates": results,
        "selected_l2": chosen["l2"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = select_l2(args.cache, steps=args.steps)
    report["report_sha256"] = object_hash(report)
    if args.output is not None:
        if args.output.exists():
            raise ValueError("CV output path already exists")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(canonical_bytes(report) + b"\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
