import unittest
from pathlib import Path
import numpy as np

from src.datasets import DATA_ROOT, fuzzify_crisp_target
from src.experiments import contaminate


class TestDataProcessing(unittest.TestCase):
    def test_data_root_is_repository_relative(self):
        repository = Path(__file__).resolve().parents[1]
        self.assertEqual(DATA_ROOT, repository / "data")

    def test_deterministic_fuzzification(self):
        observed = fuzzify_crisp_target(np.array([-2.0, 0.0, 10.0]))
        expected = np.array([[-2.0, 0.16, 0.24], [0.0, 0.08, 0.12], [10.0, 0.8, 1.2]])
        self.assertTrue(np.allclose(observed, expected))

    def test_contamination_is_deterministic_and_spreads_valid(self):
        features = np.arange(20.0).reshape(10, 2)
        targets = np.column_stack((np.arange(10.0), np.ones(10), np.ones(10) * 2))
        first = contaminate(features, targets, "fuzzy_spread", 0.2, 20260901)
        second = contaminate(features, targets, "fuzzy_spread", 0.2, 20260901)
        self.assertTrue(np.array_equal(first[0], second[0]))
        self.assertTrue(np.array_equal(first[1], second[1]))
        self.assertEqual(first[2], second[2])
        self.assertTrue(np.all(first[1][:, 1:] >= 0))


if __name__ == "__main__":
    unittest.main()
