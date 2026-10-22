import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from src.experiment import choose_architecture, evaluate_experiment, load_data, make_model, make_splits, train_experiment


class ExperimentTest(unittest.TestCase):
    # Une relance ne doit ni ecraser les resultats ni reevaluer le test par inadvertance.
    def test_existing_experiment_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(output=Path(directory), epochs=1, batch_size=8,
                                   learning_rate=0.001, seeds=[17])
            with self.assertRaises(FileExistsError):
                train_experiment(args)

    def test_test_evaluation_is_guarded(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "protocol.json").write_text("{}")
            (output / "test-metrics.csv").touch()
            with self.assertRaises(FileExistsError):
                evaluate_experiment(SimpleNamespace(output=output))

    # Les trois ensembles doivent etre reproductibles, disjoints et couvrir toutes les lignes.
    def test_split_is_disjoint_and_reproducible(self):
        labels = np.tile(np.arange(10), 30)
        first, second = make_splits(labels, 42), make_splits(labels, 42)
        all_indices = np.concatenate(list(first.values()))
        self.assertEqual(len(np.unique(all_indices)), len(labels))
        self.assertEqual(sorted(all_indices), list(range(len(labels))))
        self.assertEqual([len(first[k]) for k in ("train", "validation", "test")], [180, 60, 60])
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])
            self.assertEqual(set(labels[first[name]]), set(range(10)))

    def test_models_return_ten_logits(self):
        for name in ("linear", "mlp", "cnn"):
            self.assertEqual(tuple(make_model(name)(torch.zeros(3, 784)).shape), (3, 10))

    # On moyenne les graines : une seule execution chanceuse ne doit pas decider du modele.
    def test_selection_uses_mean_validation_accuracy(self):
        table = pd.DataFrame({"model": ["linear", "linear", "cnn", "cnn"],
                              "accuracy": [0.99, 0.7, 0.9, 0.9], "loss": [0.1, 0.6, 0.2, 0.2]})
        self.assertEqual(choose_architecture(table), "cnn")

    def test_loader_rejects_invalid_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.csv"
            frame = pd.DataFrame(np.zeros((2, 784)), columns=[f"pixel{i}" for i in range(784)])
            frame.insert(0, "label", [0, 1])
            frame.to_csv(path, index=False)
            self.assertEqual(len(load_data(path)), 2)
            frame.loc[0, "pixel0"] = 256
            frame.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_data(path)


if __name__ == "__main__":
    unittest.main()
