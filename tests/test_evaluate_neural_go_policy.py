"""Tests for the independent Stage 5 policy evaluation statistics."""

from __future__ import annotations

import math
import unittest

from scripts.evaluate_neural_go_policy import poisson_binomial_tail, wilson_interval


class Stage5EvaluationTests(unittest.TestCase):
    def test_poisson_binomial_tail_matches_known_fair_coin_values(self) -> None:
        probabilities = [0.5, 0.5, 0.5]
        self.assertAlmostEqual(poisson_binomial_tail(probabilities, 0), 1.0)
        self.assertAlmostEqual(poisson_binomial_tail(probabilities, 2), 0.5)
        self.assertAlmostEqual(poisson_binomial_tail(probabilities, 3), 0.125)
        self.assertEqual(poisson_binomial_tail(probabilities, 4), 0.0)

    def test_poisson_binomial_handles_unequal_probabilities(self) -> None:
        # P(X >= 1) = 1 - P(X = 0) = 1 - 0.8 * 0.7.
        self.assertAlmostEqual(poisson_binomial_tail([0.2, 0.3], 1), 0.44)

    def test_wilson_interval_contains_observed_proportion(self) -> None:
        low, high = wilson_interval(25, 72)
        self.assertLess(low, 25 / 72)
        self.assertGreater(high, 25 / 72)
        self.assertTrue(math.isfinite(low))
        self.assertTrue(math.isfinite(high))

    def test_statistic_inputs_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            poisson_binomial_tail([1.1], 1)
        with self.assertRaises(ValueError):
            poisson_binomial_tail([0.5], -1)
        with self.assertRaises(ValueError):
            wilson_interval(1, 0)


if __name__ == "__main__":
    unittest.main()
