"""Checks for the read-only Stage8 adapter diagnostic."""

from __future__ import annotations

import hashlib
import unittest

import numpy as np

from scripts.probe_stage8_adapter import _fixed_seed_input_schedule
from src.malecns_dynamics import DYNAMICS_VERSION


class FixedSeedInputScheduleTests(unittest.TestCase):
    def test_matches_simulator_accumulator_rule(self) -> None:
        rates = np.random.default_rng(88).uniform(0.0, 150.0, (3, 32))
        manifest = {
            "generator": {
                "lif_seed_mode": "fixed",
                "lif_seed_base": 7,
                "lif_parameters": {"dt_ms": 0.2, "duration_ms": 12.0},
            }
        }
        actual = _fixed_seed_input_schedule(rates, manifest)
        self.assertEqual(actual.shape, (3, 60, 32))
        for row in range(3):
            phases = np.asarray([
                int.from_bytes(
                    hashlib.sha256(
                        f"7|{channel}|{DYNAMICS_VERSION}".encode("ascii")
                    ).digest()[:8],
                    "little",
                ) / 2**64
                for channel in range(32)
            ])
            for tick in range(60):
                phases += rates[row] * 0.2 / 1000.0
                fired = phases >= 1.0
                phases[fired] -= np.floor(phases[fired])
                np.testing.assert_array_equal(actual[row, tick], fired)

    def test_rejects_per_position_seed(self) -> None:
        with self.assertRaisesRegex(ValueError, "fixed LIF seed"):
            _fixed_seed_input_schedule(
                np.zeros((1, 32)),
                {"generator": {"lif_seed_mode": "per-position"}},
            )


if __name__ == "__main__":
    unittest.main()
