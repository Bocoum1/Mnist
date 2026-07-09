import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch n'est pas installé")
class MnistCnnTest(unittest.TestCase):
    def test_forward_shape(self):
        import torch

        from src.mnist_cnn import SimpleCNN

        model = SimpleCNN()
        batch = torch.zeros((4, 784), dtype=torch.float32)
        output = model(batch)

        self.assertEqual(tuple(output.shape), (4, 10))

    def test_split_dataset_rejects_invalid_ratio(self):
        import torch
        from torch.utils.data import TensorDataset

        from src.mnist_cnn import split_dataset

        dataset = TensorDataset(torch.zeros((10, 784)), torch.zeros(10, dtype=torch.long))

        with self.assertRaises(ValueError):
            split_dataset(dataset, validation_ratio=1.0, seed=42)


if __name__ == "__main__":
    unittest.main()
