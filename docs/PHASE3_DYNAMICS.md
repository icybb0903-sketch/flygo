# Phase 3 engineering neural-dynamics reference

`src/malecns_dynamics.py` is a deterministic Python reference runtime for the
Phase 3 MaleCNS sparse graph. It is deliberately separate from Go move choice.

## What this is — and is not

The graph builder supplies measured MaleCNS **structural connectivity**. The
membrane equation, transmitter sign mapping, stimulus selection and every
constant in this runtime are engineering choices. The resulting frame is a
real output of that exact computation, but it is **not recorded neural
activity, not biologically calibrated, and not evidence that a fly understands
Go**. No teacher answer, search score or capture-first baseline action enters
the computation. The runtime never returns a Go action.

## Asset contract

`load_graph_assets(path)` requires `manifest.json` plus six hash-checked,
memory-mapped arrays. The outgoing CSR direction is source/pre neuron to
destination/post neuron.

| file key | dtype | shape |
|---|---|---|
| `node_ids` | `int64` | `[N]` |
| `soma_xyz` | `float32` | `[N,3]` |
| `indptr` | `uint64` | `[N+1]` |
| `indices` | `uint32` | `[E]` |
| `weights` | `float32` | `[E]` |
| `nt_code` | `int8` | `[N]` |

The manifest may name its mapping `artifacts`, `assets` or `files`; the graph
builder uses `artifacts`. Each record's filename, byte count (if present),
dtype, shape and SHA-256 are checked before simulation. All structural and
numeric inconsistencies fail closed.

`nt_code` is source-neuron based: `1` adds fast current, `-1` subtracts it,
`0` is the explicit zero-fast-current assumption for modulatory transmitters,
and `-128` is unknown and also contributes zero fast current. These signs are
not universal biological truths.

## 415 values to 32 stimulus rates

`go415-to-32-hash-bucket-v1` validates the original encoding hash, normalises
the five board planes and ten public scalar fields, and sends every one of 415
features to two fixed buckets:

- primary: `(17 * feature_index + 3) mod 32`, weight `1.0`;
- secondary: `(29 * feature_index + 11) mod 32`, weight `0.5`.

The weighted bucket means are mapped to `0..150 Hz`. The complete rule and its
version are returned with `stimulus_hash`. This is a lossy engineered adapter,
not a model of fly vision.

## Input groups and sparse LIF

For each graph, `malecns-input-groups-sha256-v1` selects neurons with at least
one outgoing edge and known transmitter classification, ranks them by SHA-256
of version/model/body ID, and partitions the rank into 32 groups (eight neurons
per group by default). Every group's actual body IDs and the reuse flag for tiny
test graphs are returned in the frame.

The default `malecns-engineered-sparse-lif-v2` constants are versioned in every
frame: 0.2 ms ticks, 40 ms window, -52 mV rest/reset, -45 mV threshold, 20 ms
membrane time constant, 1.8 ms synaptic delay, 2.2 ms refractory period and
0.05 mV per contact. The amplitude is an engineering stability value for this
weighted graph, not a measured synaptic amplitude. The earlier 0.275 value
tripped the five-million-event guard on the full graph and correctly produced
no partial frame. None of these constants is calibrated against MaleCNS
physiology.

Simulation is bounded by per-tick spike, total spike, synaptic-event, recorded
spike and tick limits. Cancellation and wall-time deadlines are checked during
the run. A limit, cancellation, timeout or invalid asset raises an error and
returns no partial frame; integration must withhold any downstream decision.

## Frame provenance

Successful output includes `frame_id`, `model_id`, manifest hash, board and
encoding hashes, stimulus hash, parameter hash, seed, real spike/event counts,
actual spike body IDs and a bounded display subset with real soma coordinates.
Every display node contains its true spike count, peak normalised activity and
minimum/maximum membrane potential from that run. An inhibitory or silent node
may therefore appear as structural context while retaining zero activity.

The subset first reserves up to half its capacity for actual stimulus sources,
then adds activity-ranked nodes and genuine one-hop CSR destinations.
`display_edges` is
read directly from that same controller CSR and contains the CSR edge index,
both model/body-ID endpoints, weight and source transmitter sign. Both endpoints
must be present in `display_nodes`. Node and edge limits, counts and truncation
flags are explicit (`512` nodes and `2,048` edges by default). There is no
decorative diffusion or generated edge. `activity_source` remains
`same_controller_frame`. Wall-clock duration is measured but excluded from the
deterministic `frame_id`.

## Current validation boundary

`tests/test_malecns_dynamics.py` validates synthetic CSR fixtures: positive and
negative signs, silence, deterministic replay/frame IDs, changed inputs, tamper
rejection, exact metadata/body IDs, CSR-accurate display edges, display limits,
event protection and cancellation. The negative test checks that the actual
downstream membrane potential falls below rest while spike/activity stay zero;
it does not turn inhibition into positive activity for display.

## Full graph benchmark (2026-09-21)

`data/malecns/runtime/malecns-v1.0-w5-3acb6434e71160fc/benchmark-initial-9x9-v2.json`
records an actual run using the verified graph assets and the public encoding
of an initial 9x9 board. It used Python 3.12.14 and NumPy 2.3.5:

- controller: 139,662 nodes and 5,536,347 CSR edges;
- simulated window: 40 ms / 200 ticks;
- result: 8,529 spikes and 1,218,214 traversed synaptic events;
- displayed: 512 real nodes (486 active) and 2,048 real edges, with truncation
  flags truthfully set because 36,180 activity candidates and more internal
  edges existed;
- first final-contract call: 678.276 ms wall clock (frame simulation reported
  421.254 ms; the call also serialised the attached stimulus); repeat call:
  661.218 ms;
- repeat `frame_id` matched exactly;
- `go_move_selected=false` and `baseline_controller_called=false`.

The benchmark is a reproducible software execution record, not biological
validation and not evidence that MaleCNS can play Go.
