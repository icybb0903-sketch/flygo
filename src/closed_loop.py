"""Deterministic two-action adapter over the verified P2 toy dynamics.

``left`` and ``right`` are engineering labels in this module.  They identify
two source bodies and their retained downstream branches; no biological
left/right meaning is claimed.
"""

from __future__ import annotations

import random
from typing import Any, Iterable

from src.p1_graph import VerifiedGraph
from src.toy_dynamics import MODEL_NAME, ModelParameters, simulate


LEFT_SOURCE_BODY = 12781
RIGHT_SOURCE_BODY = 556329
CUE_TO_SOURCE = {"left": LEFT_SOURCE_BODY, "right": RIGHT_SOURCE_BODY}
DEFAULT_CUE_SEQUENCE = (
    "left",
    "right",
    "right",
    "left",
    "right",
    "left",
    "left",
    "right",
    "left",
    "right",
)
DEFAULT_RANDOM_SEED = 10412781
CONDITIONS = (
    "real_edges",
    "disconnected_edges",
    "fixed_random_baseline",
    "direct_mapping_upper_bound",
)
SCORE_SOURCE = "mean_downstream_branch_peak_potential"
INTERPRETATION = (
    "left/right are engineered action labels tied to source bodies 12781/556329; "
    "they are not claims about biological laterality or behavior"
)


class ClosedLoopError(RuntimeError):
    """Raised when the closed-loop input or readout contract is invalid."""


def _rounded(value: float) -> float:
    return round(value, 9)


def _branches_by_action(graph: VerifiedGraph) -> dict[str, tuple[int, ...]]:
    graph_sources = set(graph.source_nodes)
    required_sources = set(CUE_TO_SOURCE.values())
    if graph_sources != required_sources:
        raise ClosedLoopError(
            f"expected exactly source bodies {sorted(required_sources)!r}, "
            f"found {sorted(graph_sources)!r}"
        )

    result: dict[str, tuple[int, ...]] = {}
    for action, source in CUE_TO_SOURCE.items():
        branches = tuple(sorted(edge.post for edge in graph.edges if edge.pre == source))
        if not branches:
            raise ClosedLoopError(f"source body {source} has no retained downstream branch")
        result[action] = branches
    return result


def read_downstream_scores(
    graph: VerifiedGraph, simulation: dict[str, Any]
) -> dict[str, Any]:
    """Read action scores only from downstream activity in a P2 simulation.

    For each action/source, each retained outgoing branch contributes its peak
    pre-reset potential.  The score is the mean across that source's branches,
    preserving the P2 potential bound of [0, 1].
    """
    branches_by_action = _branches_by_action(graph)
    steps = simulation.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ClosedLoopError("simulation must contain at least one recorded step")

    scores: dict[str, float] = {}
    components: dict[str, list[dict[str, Any]]] = {}
    for action, branches in branches_by_action.items():
        action_components: list[dict[str, Any]] = []
        for body in branches:
            key = str(body)
            try:
                peak = max(float(step["pre_reset_potential"][key]) for step in steps)
            except (KeyError, TypeError, ValueError) as exc:
                raise ClosedLoopError(
                    f"simulation is missing numeric downstream activity for body {body}"
                ) from exc
            if not 0.0 <= peak <= 1.0:
                raise ClosedLoopError(
                    f"downstream activity for body {body} is outside [0, 1]: {peak}"
                )
            action_components.append(
                {"downstream_body": body, "peak_potential": _rounded(peak)}
            )
        score = sum(item["peak_potential"] for item in action_components) / len(
            action_components
        )
        scores[action] = _rounded(score)
        components[action] = action_components

    return {
        "left_score": scores["left"],
        "right_score": scores["right"],
        "score_source": SCORE_SOURCE,
        "branch_activity": components,
    }


def choose_from_scores(left_score: float, right_score: float) -> tuple[str, str]:
    """Choose from scores alone; cue is deliberately absent from this API."""
    for name, value in (("left_score", left_score), ("right_score", right_score)):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ClosedLoopError(f"{name} must be numeric")
        if not 0.0 <= value <= 1.0:
            raise ClosedLoopError(f"{name} must be inside [0, 1]")
    if left_score == 0.0 and right_score == 0.0:
        return "abstain", "downstream_zero_activity_abstain"
    if left_score == right_score:
        return "abstain", "downstream_equal_score_abstain"
    if left_score > right_score:
        return "left", "downstream_score_argmax"
    return "right", "downstream_score_argmax"


def run_trial(
    graph: VerifiedGraph,
    cue: str,
    condition: str,
    *,
    trial_index: int = 0,
    random_generator: random.Random | None = None,
    parameters: ModelParameters | None = None,
) -> dict[str, Any]:
    """Run one cue trial, using :func:`src.toy_dynamics.simulate`."""
    if cue not in CUE_TO_SOURCE:
        raise ClosedLoopError(f"unknown cue: {cue!r}")
    if condition not in CONDITIONS:
        raise ClosedLoopError(f"unknown condition: {condition!r}")
    if type(trial_index) is not int or trial_index < 0:
        raise ClosedLoopError("trial_index must be a nonnegative integer")

    stimulated_body = CUE_TO_SOURCE[cue]
    edges_enabled = condition != "disconnected_edges"
    simulation = simulate(
        graph,
        (stimulated_body,),
        edges_enabled=edges_enabled,
        parameters=parameters,
    )
    readout = read_downstream_scores(graph, simulation)

    if condition in ("real_edges", "disconnected_edges"):
        chosen_action, decision_source = choose_from_scores(
            readout["left_score"], readout["right_score"]
        )
    elif condition == "fixed_random_baseline":
        if random_generator is None:
            raise ClosedLoopError("fixed random baseline requires a seeded generator")
        chosen_action = "right" if random_generator.getrandbits(1) else "left"
        decision_source = "fixed_seed_random_baseline"
    else:
        chosen_action = cue
        decision_source = "direct_cue_mapping_upper_bound"

    return {
        "trial_index": trial_index,
        "condition": condition,
        "cue": cue,
        "stimulated_body": stimulated_body,
        "left_score": readout["left_score"],
        "right_score": readout["right_score"],
        "chosen_action": chosen_action,
        "abstained": chosen_action == "abstain",
        "correct": chosen_action == cue,
        "p1_canonical_sha256": graph.canonical_sha256,
        "score_source": readout["score_source"],
        "decision_source": decision_source,
        "branch_activity": readout["branch_activity"],
    }


def _condition_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    trial_count = len(records)
    correct_count = sum(record["correct"] for record in records)
    abstain_count = sum(record["abstained"] for record in records)
    decided_count = trial_count - abstain_count
    return {
        "trial_count": trial_count,
        "correct_count": correct_count,
        "abstain_count": abstain_count,
        "accuracy": _rounded(correct_count / trial_count),
        "decision_coverage": _rounded(decided_count / trial_count),
        "accuracy_when_decided": (
            _rounded(correct_count / decided_count) if decided_count else None
        ),
    }


def run_closed_loop_suite(
    graph: VerifiedGraph,
    cue_sequence: Iterable[str] = DEFAULT_CUE_SEQUENCE,
    *,
    random_seed: int = DEFAULT_RANDOM_SEED,
    parameters: ModelParameters | None = None,
) -> dict[str, Any]:
    """Run real, disconnected, random, and direct-mapping conditions."""
    cues = tuple(cue_sequence)
    if len(cues) < 8:
        raise ClosedLoopError("cue sequence must contain at least eight trials")
    unknown = [cue for cue in cues if cue not in CUE_TO_SOURCE]
    if unknown:
        raise ClosedLoopError(f"cue sequence contains unknown cue(s): {unknown!r}")
    if type(random_seed) is not int:
        raise ClosedLoopError("random_seed must be an integer")
    branches = _branches_by_action(graph)

    trials: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for condition in CONDITIONS:
        generator = random.Random(random_seed) if condition == "fixed_random_baseline" else None
        records = [
            run_trial(
                graph,
                cue,
                condition,
                trial_index=index,
                random_generator=generator,
                parameters=parameters,
            )
            for index, cue in enumerate(cues)
        ]
        trials[condition] = records
        summaries[condition] = _condition_summary(records)

    real = trials["real_edges"]
    disconnected = trials["disconnected_edges"]
    checks = {
        "fixed_sequence_has_at_least_eight_trials": len(cues) >= 8,
        "cue_stimulates_exactly_mapped_source": all(
            record["stimulated_body"] == CUE_TO_SOURCE[record["cue"]]
            for condition_records in trials.values()
            for record in condition_records
        ),
        "real_edges_nonzero_readout_every_trial": all(
            record[f"{record['cue']}_score"] > 0.0 for record in real
        ),
        "real_edges_correct_choice_every_trial": all(record["correct"] for record in real),
        "disconnected_scores_zero_every_trial": all(
            record["left_score"] == 0.0 and record["right_score"] == 0.0
            for record in disconnected
        ),
        "disconnected_abstains_every_trial": all(
            record["chosen_action"] == "abstain" for record in disconnected
        ),
        "p1_hash_propagated_every_trial": all(
            record["p1_canonical_sha256"] == graph.canonical_sha256
            for condition_records in trials.values()
            for record in condition_records
        ),
        "all_scores_bounded": all(
            0.0 <= record[side] <= 1.0
            for condition_records in trials.values()
            for record in condition_records
            for side in ("left_score", "right_score")
        ),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "p1_canonical_sha256": graph.canonical_sha256,
        "p2_model": MODEL_NAME,
        "interpretation": INTERPRETATION,
        "cue_to_source_body": CUE_TO_SOURCE.copy(),
        "downstream_branches": {
            action: list(action_branches) for action, action_branches in branches.items()
        },
        "cue_sequence": list(cues),
        "random_seed": random_seed,
        "score_contract": {
            "source": SCORE_SOURCE,
            "formula": (
                "for each action, mean over that source's retained downstream branches "
                "of the maximum P2 pre-reset potential across simulated steps"
            ),
            "bounds": [0.0, 1.0],
        },
        "conditions": trials,
        "summaries": summaries,
        "checks": checks,
    }
