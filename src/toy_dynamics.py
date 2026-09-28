"""Deterministic engineered toy dynamics over a verified MaleCNS topology.

This module is not a biological model. It uses real P1 connection weights inside
an explicitly artificial, bounded leaky integrate-and-fire-style computation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from src.p1_graph import VerifiedGraph


MODEL_NAME = "engineered_toy_lif_v1"
MODEL_DISCLAIMER = (
    "Engineered toy dynamics over real MaleCNS connection weights. "
    "This is not measured neural activity, a validated biological simulation, "
    "evidence of cognition, or evidence that a fruit fly can perform a task."
)


class DynamicsError(RuntimeError):
    """Raised when a dynamics input or invariant is invalid."""


@dataclass(frozen=True)
class ModelParameters:
    steps: int = 4
    dt_ms: float = 1.0
    leak_fraction_per_step: float = 0.5
    threshold: float = 0.5
    reset_potential: float = 0.0
    external_pulse: float = 1.0
    synaptic_gain: float = 1.0
    weight_scale: float = 288.0
    state_min: float = 0.0
    state_max: float = 1.0

    def validate(self) -> None:
        if type(self.steps) is not int or self.steps < 2:
            raise DynamicsError("steps must be an integer of at least 2")
        if self.dt_ms <= 0:
            raise DynamicsError("dt_ms must be positive")
        if not 0 <= self.leak_fraction_per_step <= 1:
            raise DynamicsError("leak_fraction_per_step must be in [0, 1]")
        if not self.state_min <= self.reset_potential <= self.state_max:
            raise DynamicsError("reset_potential must be inside state bounds")
        if not self.state_min < self.threshold <= self.state_max:
            raise DynamicsError("threshold must be inside state bounds")
        if self.external_pulse < 0 or self.synaptic_gain < 0 or self.weight_scale <= 0:
            raise DynamicsError("pulse/gain must be nonnegative and weight_scale positive")


def _rounded(value: float) -> float:
    return round(value, 9)


def _bounded(value: float, parameters: ModelParameters) -> float:
    return _rounded(min(parameters.state_max, max(parameters.state_min, value)))


def simulate(
    graph: VerifiedGraph,
    stimulus_nodes: Iterable[int],
    *,
    edges_enabled: bool,
    parameters: ModelParameters | None = None,
) -> dict[str, Any]:
    """Apply a one-step external pulse, then bounded leak/propagation updates."""
    params = parameters or ModelParameters()
    params.validate()
    requested = tuple(sorted(set(stimulus_nodes)))
    known_nodes = set(graph.nodes)
    unknown = [node for node in requested if type(node) is not int or node not in known_nodes]
    if unknown:
        raise DynamicsError(f"stimulus contains unknown node(s): {unknown!r}")

    states = {node: 0.0 for node in graph.nodes}
    previous_spikes: set[int] = set()
    records: list[dict[str, Any]] = []
    for step in range(params.steps):
        incoming = {node: 0.0 for node in graph.nodes}
        if edges_enabled:
            for edge in graph.edges:
                if edge.pre in previous_spikes:
                    incoming[edge.post] += edge.weight / params.weight_scale

        external = {
            node: params.external_pulse if step == 0 and node in requested else 0.0
            for node in graph.nodes
        }
        pre_reset: dict[int, float] = {}
        for node in graph.nodes:
            retained = states[node] * (1.0 - params.leak_fraction_per_step)
            candidate = retained + params.synaptic_gain * incoming[node] + external[node]
            pre_reset[node] = _bounded(candidate, params)

        spikes = tuple(node for node in graph.nodes if pre_reset[node] >= params.threshold)
        spike_set = set(spikes)
        states = {
            node: params.reset_potential if node in spike_set else pre_reset[node]
            for node in graph.nodes
        }
        if any(not params.state_min <= value <= params.state_max for value in (*pre_reset.values(), *states.values())):
            raise DynamicsError("state escaped the documented numeric bounds")

        records.append(
            {
                "step": step,
                "time_ms": _rounded(step * params.dt_ms),
                "external_drive": {str(node): external[node] for node in graph.nodes if external[node]},
                "spikes": list(spikes),
                "pre_reset_potential": {str(node): pre_reset[node] for node in graph.nodes},
                "state_after_reset": {str(node): states[node] for node in graph.nodes},
            }
        )
        previous_spikes = spike_set

    downstream = set(graph.downstream_nodes)
    source = set(graph.source_nodes)
    downstream_spikes = sum(
        1 for record in records for node in record["spikes"] if node in downstream
    )
    source_spikes = sum(1 for record in records for node in record["spikes"] if node in source)
    downstream_values = [
        record["pre_reset_potential"][str(node)]
        for record in records
        for node in graph.downstream_nodes
    ]
    all_values = [
        value
        for record in records
        for field in ("pre_reset_potential", "state_after_reset")
        for value in record[field].values()
    ]
    return {
        "edges_enabled": edges_enabled,
        "stimulus": {
            "nodes": list(requested),
            "pulse_step": 0,
            "amplitude": params.external_pulse if requested else 0.0,
        },
        "steps": records,
        "summary": {
            "downstream_peak_potential": max(downstream_values, default=0.0),
            "downstream_spike_count": downstream_spikes,
            "source_spike_count": source_spikes,
            "total_spike_count": source_spikes + downstream_spikes,
            "observed_state_min": min(all_values, default=0.0),
            "observed_state_max": max(all_values, default=0.0),
            "state_within_bounds": all(
                params.state_min <= value <= params.state_max for value in all_values
            ),
        },
    }


def observed_topology_depth(graph: VerifiedGraph) -> tuple[bool, int | None]:
    """Return (has_cycle, longest directed path in edges) for the retained graph."""
    adjacency = {node: [] for node in graph.nodes}
    for edge in graph.edges:
        adjacency[edge.pre].append(edge.post)
    visiting: set[int] = set()
    complete: dict[int, int] = {}

    def visit(node: int) -> int:
        if node in visiting:
            raise DynamicsError("cycle")
        if node in complete:
            return complete[node]
        visiting.add(node)
        depth = max((1 + visit(child) for child in adjacency[node]), default=0)
        visiting.remove(node)
        complete[node] = depth
        return depth

    try:
        depth = max(visit(node) for node in graph.nodes)
    except DynamicsError:
        return True, None
    return False, depth


def run_experiment_suite(
    graph: VerifiedGraph, parameters: ModelParameters | None = None
) -> dict[str, Any]:
    params = parameters or ModelParameters()
    source_nodes = graph.nodes_of_type("DNge104")
    if not source_nodes:
        raise DynamicsError("verified graph has no DNge104 source node")
    has_cycle, path_depth = observed_topology_depth(graph)

    experiments = {
        "A_no_stimulus": simulate(graph, (), edges_enabled=True, parameters=params),
        "B_stimulated_real_edges": simulate(
            graph, source_nodes, edges_enabled=True, parameters=params
        ),
        "C_stimulated_edges_disconnected": simulate(
            graph, source_nodes, edges_enabled=False, parameters=params
        ),
    }
    checks = {
        "A_downstream_quiet": experiments["A_no_stimulus"]["summary"]["downstream_spike_count"] == 0
        and experiments["A_no_stimulus"]["summary"]["downstream_peak_potential"] == 0.0,
        "B_finite_downstream_response": experiments["B_stimulated_real_edges"]["summary"]["downstream_spike_count"] > 0
        and 0.0 < experiments["B_stimulated_real_edges"]["summary"]["downstream_peak_potential"] <= params.state_max,
        "C_downstream_quiet_when_disconnected": experiments["C_stimulated_edges_disconnected"]["summary"]["downstream_spike_count"] == 0
        and experiments["C_stimulated_edges_disconnected"]["summary"]["downstream_peak_potential"] == 0.0,
        "all_states_bounded": all(
            experiment["summary"]["state_within_bounds"]
            for experiment in experiments.values()
        ),
    }
    if not all(checks.values()):
        raise DynamicsError(f"experiment acceptance check failed: {checks!r}")
    return {
        "model": MODEL_NAME,
        "model_disclaimer": MODEL_DISCLAIMER,
        "p1_canonical_sha256": graph.canonical_sha256,
        "parameters": asdict(params),
        "topology": {
            "node_count": len(graph.nodes),
            "edge_count": len(graph.edges),
            "source_nodes": list(graph.source_nodes),
            "downstream_nodes": list(graph.downstream_nodes),
            "directed_cycle_detected": has_cycle,
            "observed_max_path_edges": path_depth,
            "interpretation": "The retained P1 graph is one edge deep; no recurrence is claimed.",
        },
        "experiments": experiments,
        "checks": checks,
    }
