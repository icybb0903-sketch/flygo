"""Small deterministic checks for the isolated Stage8 softmax trainer."""

from __future__ import annotations

import unittest

import numpy as np

from scripts.train_stage8_legal_softmax import fit_legal_softmax


class LegalSoftmaxTests(unittest.TestCase):
    def test_learns_from_features_and_exports_runtime_shape(self) -> None:
        features = np.zeros((40, 128), dtype=np.float64)
        features[:20, 0] = 0.01
        features[20:, 1] = 0.01
        labels = np.asarray([0] * 20 + [1] * 20)
        masks = np.zeros((40, 82), dtype=bool)
        masks[:, :2] = True
        masks[:, 81] = True
        first_w, first_b, info = fit_legal_softmax(
            features, labels, masks, steps=60
        )
        second_w, second_b, _ = fit_legal_softmax(
            features, labels, masks, steps=60
        )
        self.assertEqual(first_w.shape, (82, 128))
        self.assertEqual(first_b.shape, (82,))
        self.assertEqual(first_w.dtype, np.dtype("float32"))
        self.assertTrue(np.array_equal(first_w, second_w))
        self.assertTrue(np.array_equal(first_b, second_b))
        scores = features @ first_w.T + first_b
        np.testing.assert_array_equal(np.argmax(scores[:, :2], axis=1), labels)
        self.assertTrue(np.all(first_w[81] == 0))
        self.assertTrue(np.isfinite(info["train_final_objective"]))

    def test_rejects_illegal_teacher_label(self) -> None:
        features = np.zeros((1, 128))
        mask = np.zeros((1, 82), dtype=bool)
        mask[0, 1] = True
        with self.assertRaisesRegex(ValueError, "legal point move"):
            fit_legal_softmax(features, np.asarray([0]), mask, steps=1)

    def test_sample_weights_are_validated(self) -> None:
        features = np.zeros((2, 128))
        labels = np.asarray([0, 1])
        masks = np.zeros((2, 82), dtype=bool)
        masks[:, :2] = True
        with self.assertRaisesRegex(ValueError, "sample_weights"):
            fit_legal_softmax(
                features, labels, masks, steps=1,
                sample_weights=np.asarray([1.0, -1.0]),
            )

    def test_larger_sample_weight_changes_ambiguous_label(self) -> None:
        features = np.zeros((2, 128))
        labels = np.asarray([0, 1])
        masks = np.zeros((2, 82), dtype=bool)
        masks[:, :2] = True
        weights, bias, _ = fit_legal_softmax(
            features, labels, masks, steps=40,
            sample_weights=np.asarray([1.0, 4.0]),
        )
        self.assertEqual(int(np.argmax(features[0] @ weights[:2].T + bias[:2])), 1)


if __name__ == "__main__":
    unittest.main()
