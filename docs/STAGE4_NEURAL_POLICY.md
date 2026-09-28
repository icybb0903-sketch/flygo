# Stage 4: MaleCNS 128→82 neural readout contract

## What changed

`src/malecns_dynamics.py` now emits `malecns-engineered-lif-frame-v3`.
Every successful simulation contains a versioned `output_pool` with 128
bounded features. These values are computed inside the same `simulate_frame`
call from its complete `spike_counts` and `max_activity` arrays **before** the
browser display subset is selected. They are not reconstructed from
`display_nodes`.

The selection rule `malecns-output-pools-sha256-v1` ranks controller neurons
by SHA-256 of `(version, model_id, body_id)`, then partitions 4,096 neurons into
128 pools of 32. Tiny test graphs cycle their deterministic ordering and set
`reused_for_small_graph=true`. The frame records all selected real body IDs,
the exact rule, a `group_hash`, all 128 features and a `features_hash`.

Each feature is:

```text
0.5 * mean(clip(spike_count / tick_count, 0, 1))
+ 0.5 * mean(clip(max_activity, 0, 1))
```

It is therefore finite and bounded to `[0,1]`. The entire `output_pool` is part
of the deterministic frame core, so it contributes to `frame_id`.

## Checkpoint contract

`src/malecns_policy.py` defines the strict
`malecns-go-linear-policy-checkpoint-v1` directory:

```text
checkpoint/
  manifest.json
  weights.npy  float32 [82, 128]
  bias.npy     float32 [82]
```

The manifest binds the parameters to all of the following:

- policy version `malecns-go-linear-128x82-v1`;
- exact MaleCNS `model_id` and graph manifest SHA-256;
- output-pool version, 128-feature count and `group_hash`;
- 9×9 row-major 81-point plus pass action mapping;
- explicit `trained` or `untrained` status and training provenance;
- artifact byte counts, dtypes, shapes and SHA-256 hashes;
- explicit `teacher_access=false`, `baseline_access=false` and
  `fallback_controller=false` runtime guarantees.

Loading checks every field and file. Missing, malformed, non-finite, altered
or wrong-shape artifacts fail closed. An `untrained` checkpoint can be created
and inspected, but `select_action` refuses to use it.

Public APIs:

```python
initialise_linear_parameters(seed=0, scale=0.01)
build_checkpoint_manifest(...)
write_policy_checkpoint(directory, ...)
load_policy_checkpoint(directory)
select_action(frame, encoding, checkpoint)
```

`write_policy_checkpoint` does not train anything. A caller claiming
`training_status="trained"` must supply real `method`, `dataset_id`, positive
`sample_count` and `completed_at` provenance. The training workflow is
responsible for that claim.

## Runtime decision boundary

`select_action` independently verifies:

1. encoding schema/hash, board hash and 82-value binary legal mask;
2. frame schema and recomputed deterministic `frame_id`;
3. frame/encoding board and encoding hashes;
4. model ID and graph manifest hash against the checkpoint;
5. output-pool version, group hash, feature count/range and feature hash;
6. checkpoint training status and finite matrix output.

It then computes `weights @ features + bias`, masks illegal actions and uses a
deterministic first-index argmax over legal actions. Index 81 maps to pass.
The result carries the frame, encoding, checkpoint and feature hashes and sets
`teacher_accessed_at_runtime=false`, `baseline_controller_called=false` and
`fallback_used=false`. There is no branch to the existing capture-first
baseline.

## Real-graph acceptance run

On 2026-09-21 the verified runtime graph at
`data/malecns/runtime/malecns-v1.0-w5-3acb6434e71160fc` was loaded and the
initial 9×9 encoding was simulated twice with seed 0:

- controller graph: 139,662 nodes / 5,536,347 edges;
- 8,529 spikes / 1,218,214 traversed synaptic events per run;
- exactly 128 output features, range `0.00414225..0.10190292`;
- group hash:
  `ea8642327476f244d6f6ec46e52400d566709da80ee4d04e6c203d162de6552c`;
- feature hash:
  `1b332aecbc575ed0d6a33c15eb93984605a1dc5c8ebfe9eab41781e004d8ced1`;
- features, feature hash and frame ID matched across both runs;
- measured simulation wall times were 727.028 ms and 747.600 ms;
- both frames retained `go_move_selected=false` and
  `baseline_controller_called=false`.

This validates deterministic software coupling to the real preprocessed
MaleCNS structural graph. It does not validate the engineering dynamics as
biology, prove cognition, or make an untrained readout capable of Go.

## Tests

The focused dynamics/policy suite covers same-frame source semantics,
display-subset independence, bounds, JSON safety, determinism, changed neural
frames, exact legal masking, illegal-highest-logit rejection, pass, tampered or
missing checkpoints, frame/encoding/model mismatch and explicit untrained
rejection. At completion it passed 21/21. The complete repository suite passed
146/146.
