"""Check CSV preprocessing, model training, and multiclass metrics."""
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from homework_datasets import CSVDataset, fit_preprocessor, read_csv
from homework_model_modification import classification_metrics, fit


class HomeworkTests(unittest.TestCase):
    def test_csv_normalization_and_unknown_category(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.csv"
            path.write_text("amount,group,flag,target\n1,a,0,no\n3,b,1,yes\n5,c,1,yes\n",
                            encoding="utf-8")
            x, y = read_csv(path, "target")
        self.assertEqual(y.tolist(), ["no", "yes", "yes"])
        processor = fit_preprocessor(x[:2])
        dataset = CSVDataset(x[2:], [1], processor, classification=True)
        features, label = dataset[0]
        torch.testing.assert_close(features, torch.tensor([3., 1., 0., 0.]))
        self.assertEqual(label.item(), 1)

    def test_early_stopping_on_constant_validation_loss(self):
        x = torch.arange(4, dtype=torch.float32).reshape(-1, 1)
        y = 2 * x + 1
        _, history = fit((x, y), (x, y), epochs=20, lr=0, patience=3, l1=.01, l2=.01)
        self.assertEqual(len(history), 4)

    def test_linear_training_learns_relationship(self):
        x = torch.linspace(-1, 1, 40).reshape(-1, 1)
        y = 2 * x + 1
        model, _ = fit((x[::2], y[::2]), (x[1::2], y[1::2]), epochs=120, lr=.05)
        torch.testing.assert_close(model(x), y, rtol=0, atol=.02)

    def test_multiclass_metrics_use_probabilities(self):
        y = np.array([0, 1, 2])
        probabilities = np.array([[.8, .1, .1], [.1, .8, .1], [.1, .1, .8]])
        self.assertEqual(classification_metrics(y, probabilities),
                         {"precision": 1., "recall": 1., "f1": 1., "roc_auc": 1.})


if __name__ == "__main__":
    unittest.main()
