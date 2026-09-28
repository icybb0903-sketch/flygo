# Phase 3: Go neural encoding contract

`src/go_neural_encoding.py` is a deterministic engineering adapter between
the existing 9×9 Go rules engine and later neural-runtime work.

It is **not** a model of biological fly vision or sensation. Its presence does
not mean that MaleCNS has been loaded, simulated, trained, or used for a Go
decision. Every returned record therefore contains `male_cns_used: false`.

## Action contract

- Actions `0..80`: board points in row-major order, `row * 9 + column`.
- Action `81`: pass.
- `move_to_action` and `action_to_move` are exact inverses for all 82 actions.
- `legal_action_mask` returns 82 integer values (`0` or `1`) in the same order.
- A completed game has an all-zero mask because no further action is legal.

## Input vector contract

`encode_go_state` returns a JSON-safe object. Its numeric `vector` has a fixed
length of 415 and can be passed directly to `numpy.asarray` when NumPy is
available. Values are ordered plane-major:

1. `own_stone` — 81 points relative to the player to move.
2. `opponent_stone` — 81 points relative to the player to move.
3. `empty` — 81 points.
4. `legal_point` — the first 81 values of the legal action mask.
5. `last_move` — 81 points; all zero if the last action was pass or no move.
6. Ten scalar values in this exact order:
   `to_play_black`, `to_play_white`, `own_captures`, `opponent_captures`,
   `consecutive_passes`, `last_move_was_pass`, `pass_legal`, `game_over`,
   `move_number`, `komi`.

`last_move_action` is `null` for an initial state, `0..80` for a point, and
`81` for pass. The value is derived from the final two real history boards.
An inconsistent synthetic history is rejected rather than guessed.

## Stable identifiers

- `board_hash` is SHA-256 over the canonical board geometry only. Positions
  with the same stones have the same board hash even if the player changes.
- `encoding_hash` is SHA-256 over the complete encoding record (before that
  hash is inserted). It changes with side to play, legality, history-derived
  last move, captures, pass state, move number, or komi.

Both hashes use sorted, compact UTF-8 JSON, so they are stable across process
runs and independent of Python's randomized object hashing.
