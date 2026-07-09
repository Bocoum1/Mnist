from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split


class SimpleCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 1, 28, 28)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_train_dataset(csv_path: Path, max_rows: int | None = None) -> TensorDataset:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} introuvable. Place train.csv dans data/ ou passe --train-csv."
        )

    df = pd.read_csv(csv_path, nrows=max_rows)
    if "label" not in df.columns:
        raise ValueError("train.csv doit contenir une colonne `label`.")

    labels = torch.tensor(df["label"].to_numpy(), dtype=torch.long)
    pixels = df.drop(columns=["label"]).to_numpy(dtype=np.float32) / 255.0
    images = torch.from_numpy(pixels)
    return TensorDataset(images, labels)


def load_test_tensor(csv_path: Path, max_rows: int | None = None) -> torch.Tensor:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} introuvable. Place test.csv dans data/ ou passe --test-csv."
        )

    df = pd.read_csv(csv_path, nrows=max_rows)
    pixels = df.to_numpy(dtype=np.float32) / 255.0
    return torch.from_numpy(pixels)


def split_dataset(
    dataset: TensorDataset,
    validation_ratio: float,
    seed: int,
) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    if not 0 < validation_ratio < 1:
        raise ValueError("--validation-ratio doit être compris entre 0 et 1.")

    validation_size = max(1, int(len(dataset) * validation_ratio))
    train_size = len(dataset) - validation_size
    if train_size <= 0:
        raise ValueError("Le dataset est trop petit pour créer un split train/validation.")

    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, validation_size], generator=generator)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            predictions = torch.argmax(model(images), dim=1)
            correct += (predictions == labels).sum().item()

    return correct / len(loader.dataset)


def predict(model: nn.Module, images: torch.Tensor, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    loader = DataLoader(TensorDataset(images), batch_size=batch_size, shuffle=False)
    predictions: list[torch.Tensor] = []

    with torch.no_grad():
        for (batch,) in loader:
            logits = model(batch.to(device))
            predictions.append(torch.argmax(logits, dim=1).cpu())

    return torch.cat(predictions).numpy()


def write_submission(labels: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame(
        {
            "ImageId": np.arange(1, len(labels) + 1),
            "Label": labels,
        }
    )
    submission.to_csv(output_path, index=False)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Entraîner un CNN PyTorch sur MNIST CSV.")
    parser.add_argument("--train-csv", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--test-csv", type=Path, default=Path("data/test.csv"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--model-out", type=Path, default=Path("artifacts/mnist_cnn.pt"))
    parser.add_argument("--submission-out", type=Path, default=None)
    parser.add_argument("--cpu", action="store_true", help="Forcer l'entraînement sur CPU.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs doit être > 0.")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    dataset = load_train_dataset(args.train_csv, max_rows=args.max_rows)
    train_set, validation_set = split_dataset(dataset, args.validation_ratio, args.seed)

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False)

    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    print(f"device={device}")
    print(f"train_size={len(train_set)} validation_size={len(validation_set)}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        accuracy = evaluate(model, validation_loader, device)
        print(f"epoch={epoch} loss={train_loss:.4f} validation_accuracy={accuracy:.4f}")

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.model_out)
    print(f"model_saved={args.model_out}")

    if args.submission_out is not None:
        test_images = load_test_tensor(args.test_csv, max_rows=args.max_rows)
        labels = predict(model, test_images, args.batch_size, device)
        write_submission(labels, args.submission_out)
        print(f"submission_saved={args.submission_out}")


if __name__ == "__main__":
    main()
