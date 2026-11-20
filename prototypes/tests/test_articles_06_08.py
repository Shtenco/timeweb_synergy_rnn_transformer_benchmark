import unittest

from prototypes.scenarios.article06 import estimate_work, run as run_06
from prototypes.scenarios.article07 import run as run_07
from prototypes.scenarios.article08 import run as run_08


class MiddleScenarioTests(unittest.TestCase):
    def test_06_attention_estimate_scales_quadratically_in_sequence_length(self):
        short = estimate_work(sequence_length=16, hidden_size=32)
        long = estimate_work(sequence_length=32, hidden_size=32)
        self.assertEqual(long["transformer_attention_pairs"], 4 * short["transformer_attention_pairs"])
        self.assertEqual(long["rnn_sequential_steps"], 2 * short["rnn_sequential_steps"])

    def test_06_training_respects_the_same_estimated_compute_budget(self):
        result = run_06(seed=9, budget_macs=4_000_000)
        self.assertEqual(result["evidence_type"], "synthetic")
        for model in result["training"]:
            self.assertLessEqual(model["estimated_macs_used"], result["compute_budget_macs"])
            self.assertGreaterEqual(model["accuracy"], 0.0)
            self.assertLessEqual(model["accuracy"], 1.0)

    def test_07_python_implementation_returns_exact_integer_result(self):
        result = run_07(seed=9, n=100)
        self.assertEqual(result["python_result"], sum(i * i for i in range(100)))
        self.assertEqual(result["python_correctness"], "PASS")
        self.assertIn(result["native_correctness"], {"PASS", "NOT_RUN"})
        self.assertTrue(result["cpp_source_available"])

    def test_08_group_split_has_no_entity_overlap(self):
        result = run_08(seed=9, groups=30, rows_per_group=4)
        self.assertGreater(result["random_row_split"]["group_overlap_count"], 0)
        self.assertEqual(result["group_split"]["group_overlap_count"], 0)
        self.assertGreater(result["random_row_split"]["accuracy"], result["group_split"]["accuracy"])


if __name__ == "__main__":
    unittest.main()
