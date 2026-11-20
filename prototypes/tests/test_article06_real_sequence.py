import unittest

import torch

from prototypes.real_experiments.article06_har_sequences import GRUClassifier, TransformerClassifier, estimate_macs


class Article06RealSequenceTests(unittest.TestCase):
    def test_both_models_return_six_activity_logits(self):
        inputs = torch.zeros((2, 128, 9), dtype=torch.float32)
        self.assertEqual(GRUClassifier()(inputs).shape, (2, 6))
        self.assertEqual(TransformerClassifier()(inputs).shape, (2, 6))

    def test_attention_mac_estimate_grows_quadratically_with_window_length(self):
        short = estimate_macs(sequence_length=32, channels=9, hidden=32)
        long = estimate_macs(sequence_length=64, channels=9, hidden=32)
        self.assertEqual(long["transformer_macs"], 4 * short["transformer_macs"])
        self.assertEqual(long["gru_macs"], 2 * short["gru_macs"])


if __name__ == "__main__":
    unittest.main()
