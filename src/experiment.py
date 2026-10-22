"""Comparer trois architectures sur un protocole commun, puis evaluer le test separement."""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from src.mnist_cnn import SimpleCNN, set_seed, train_one_epoch


#============= 1) Donnees et separation des echantillons ====================

def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def code_digest():
    return {name: digest(Path(__file__).with_name(name))
            for name in ("experiment.py", "mnist_cnn.py")}


def load_data(path):
    frame = pd.read_csv(path)
    expected = ["label"] + [f"pixel{i}" for i in range(784)]
    if list(frame.columns) != expected:
        raise ValueError("Colonnes attendues : label, puis pixel0 a pixel783.")
    labels = frame["label"].to_numpy()
    pixels = frame.iloc[:, 1:].to_numpy(dtype=np.float32)
    if not np.isin(labels, np.arange(10)).all():
        raise ValueError("Les labels doivent etre des entiers de 0 a 9.")
    if not np.isfinite(pixels).all() or pixels.min() < 0 or pixels.max() > 255:
        raise ValueError("Les pixels doivent etre finis et compris entre 0 et 255.")
    # Division fixe par 255 : aucune moyenne ni autre statistique apprise sur le test.
    return TensorDataset(torch.from_numpy(pixels / 255), torch.tensor(labels, dtype=torch.long))


def make_splits(labels, seed):
    # Le fichier Kaggle test.csv n'a pas de labels : le test interne vient de train.csv.
    # Stratification pour conserver approximativement la proportion des dix chiffres.
    development, test = train_test_split(np.arange(len(labels)), test_size=0.2,
                                         stratify=labels, random_state=seed)
    train, validation = train_test_split(development, test_size=0.25,
                                         stratify=labels[development], random_state=seed)
    return {"train": train, "validation": validation, "test": test}


#============= 2) Modeles et mesure des performances ====================

def make_model(name):
    # Linear + CrossEntropyLoss correspond a un classifieur lineaire softmax.
    # Le MLP garde 128 neurones caches ; le CNN reprend l'architecture du depot.
    if name == "linear":
        return nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
    if name == "mlp":
        return nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
    if name == "cnn":
        return SimpleCNN()
    raise ValueError(f"Architecture inconnue : {name}")


def assess(model, loader):
    model.eval()
    total_loss, truth, predictions = 0.0, [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images)
            total_loss += nn.functional.cross_entropy(logits, labels, reduction="sum").item()
            truth.extend(labels.tolist())
            predictions.extend(logits.argmax(dim=1).tolist())
    metrics = {"loss": total_loss / len(truth), "accuracy": accuracy_score(truth, predictions),
               "macro_f1": f1_score(truth, predictions, labels=list(range(10)),
                                    average="macro", zero_division=0)}
    return metrics, np.asarray(predictions)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def choose_architecture(validation):
    # Choix fait uniquement sur la validation : accuracy moyenne, puis perte en cas d'egalite.
    means = validation.groupby("model")[["accuracy", "loss"]].mean()
    return means.sort_values(["accuracy", "loss"], ascending=[False, True]).index[0]


def plot_learning(history, output):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = {"linear": "#176076", "mlp": "#df5b2d", "cnn": "#47753d"}
    for name, group in history.groupby("model", sort=False):
        means = group.groupby("epoch").mean(numeric_only=True)
        axes[0].plot(means.index, means.validation_loss, color=colors[name], label=name)
        axes[0].plot(means.index, means.train_loss, color=colors[name], linestyle="--", alpha=0.6)
        axes[1].plot(means.index, means.validation_accuracy, color=colors[name], label=name)
    axes[0].set(title="Perte : validation / entrainement (pointilles)", ylabel="Entropie croisee")
    axes[1].set(title="Accuracy de validation", ylabel="Proportion correcte")
    for ax in axes:
        ax.set_xlabel("Epoque")
        ax.legend()
        ax.grid(alpha=0.2)
    fig.suptitle("Moyennes sur les graines ; meme separation pour tous les modeles")
    fig.tight_layout()
    fig.savefig(output / "learning-curves.png", dpi=150)
    plt.close(fig)


#============= 3) Entrainement et selection, sans evaluation du test ====================

def train_experiment(args):
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise ValueError("Epoques, batch-size et learning-rate doivent etre positifs.")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Les graines doivent etre distinctes.")
    if args.output.exists():
        raise FileExistsError("Choisir un nouveau dossier de resultats pour ne pas melanger les experiences.")
    dataset = load_data(args.train_csv)
    splits = make_splits(dataset.tensors[1].numpy(), args.split_seed)
    args.output.mkdir(parents=True)
    np.savez(args.output / "splits.npz", **splits)
    manifest = {
        "dataset_sha256": digest(args.train_csv), "code_sha256": code_digest(),
        "split_seed": args.split_seed, "sizes": {k: len(v) for k, v in splits.items()},
        "seeds": args.seeds, "models": ["linear", "mlp", "cnn"], "epochs": args.epochs,
        "batch_size": args.batch_size, "learning_rate": args.learning_rate, "threads": args.threads,
        "device": "cpu", "optimizer": "Adam", "checkpoint_rule": "minimum validation loss",
        "selection_rule": "mean validation accuracy, then mean validation loss",
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
    }
    save_json(args.output / "protocol.json", manifest)
    validation_loader = DataLoader(Subset(dataset, splits["validation"]), batch_size=args.batch_size)
    history, validation_rows = [], []
    for name in manifest["models"]:
        for seed in args.seeds:
            set_seed(seed)
            generator = torch.Generator().manual_seed(seed)
            train_loader = DataLoader(Subset(dataset, splits["train"]), batch_size=args.batch_size,
                                      shuffle=True, generator=generator)
            model = make_model(name)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
            best_loss, best_epoch, best_state = float("inf"), 0, None
            start = time.perf_counter()
            for epoch in range(1, args.epochs + 1):
                loss = train_one_epoch(model, train_loader, optimizer, nn.CrossEntropyLoss(), torch.device("cpu"))
                scores, _ = assess(model, validation_loader)
                history.append({"model": name, "seed": seed, "epoch": epoch, "train_loss": loss,
                                "validation_loss": scores["loss"], "validation_accuracy": scores["accuracy"]})
                # La meilleure epoque est choisie sur la validation, jamais sur le test.
                if scores["loss"] < best_loss:
                    best_loss, best_epoch = scores["loss"], epoch
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                print(f"{name} seed={seed} epoch={epoch} val_accuracy={scores['accuracy']:.4f}", flush=True)
            elapsed = time.perf_counter() - start
            model.load_state_dict(best_state)
            scores, _ = assess(model, validation_loader)
            torch.save(best_state, args.output / f"{name}-{seed}.pt")
            validation_rows.append({"model": name, "seed": seed, "best_epoch": best_epoch,
                                    "parameters": sum(p.numel() for p in model.parameters()),
                                    "training_seconds": elapsed, **scores})
    validation = pd.DataFrame(validation_rows)
    validation.to_csv(args.output / "validation.csv", index=False)
    history = pd.DataFrame(history)
    history.to_csv(args.output / "history.csv", index=False)
    plot_learning(history, args.output)
    manifest["selected_model"] = choose_architecture(validation)
    save_json(args.output / "protocol.json", manifest)
    print(f"Architecture retenue sur validation : {manifest['selected_model']}. Test non evalue.")


#============= 4) Evaluation finale et analyse des erreurs ====================

def evaluate_experiment(args):
    manifest = json.loads((args.output / "protocol.json").read_text())
    if (args.output / "test-metrics.csv").exists():
        raise FileExistsError("Test deja evalue pour cette experience ; ses resultats sont conserves.")
    if manifest["dataset_sha256"] != digest(args.train_csv) or manifest["code_sha256"] != code_digest():
        raise ValueError("Les donnees ou le code ont change depuis l'entrainement.")
    selected = manifest["selected_model"]  # Enregistre AVANT de consulter le test.
    dataset = load_data(args.train_csv)
    test_indices = np.load(args.output / "splits.npz")["test"]
    loader = DataLoader(Subset(dataset, test_indices), batch_size=manifest["batch_size"])
    truth = dataset.tensors[1][test_indices].numpy()
    rows = []
    for name in manifest["models"]:
        for seed in manifest["seeds"]:
            model = make_model(name)
            model.load_state_dict(torch.load(args.output / f"{name}-{seed}.pt", map_location="cpu", weights_only=True))
            scores, predictions = assess(model, loader)
            rows.append({"model": name, "seed": seed, **scores})
            # Exemples fixes : premiere graine prevue, pas celle qui a le meilleur score test.
            if name == selected and seed == manifest["seeds"][0]:
                plot_errors(dataset, test_indices, truth, predictions, args.output)
    results = pd.DataFrame(rows)
    results.to_csv(args.output / "test-metrics.csv", index=False)
    # L'ecart-type mesure ici la variabilite de l'entrainement, pas un intervalle de confiance.
    summary = results.groupby("model")[["accuracy", "macro_f1", "loss"]].agg(["mean", "std"])
    summary.columns = ["_".join(col) for col in summary.columns]
    summary.to_csv(args.output / "test-summary.csv")
    print(summary.to_string())


def plot_errors(dataset, indices, truth, predictions, output):
    matrix = confusion_matrix(truth, predictions, labels=list(range(10)))
    pd.DataFrame(matrix).to_csv(output / "confusion.csv", index_label="true_digit")
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set(xticks=range(10), yticks=range(10), xlabel="Prediction", ylabel="Chiffre reel",
           title="Matrice de confusion : modele retenu, premiere graine")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output / "confusion.png", dpi=150)
    plt.close(fig)
    errors = np.flatnonzero(truth != predictions)[:12]
    fig, axes = plt.subplots(2, 6, figsize=(12, 4))
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, errors):
        ax.imshow(dataset.tensors[0][indices[i]].reshape(28, 28), cmap="gray")
        ax.set_title(f"Reel {truth[i]} / predit {predictions[i]}")
    fig.suptitle("Premieres erreurs dans l'ordre du test, sans selection visuelle")
    fig.tight_layout()
    fig.savefig(output / "errors.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["train", "evaluate"])
    parser.add_argument("--train-csv", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/study"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--split-seed", type=int, default=20261015)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 42, 73])
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads doit etre positif.")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    (train_experiment if args.stage == "train" else evaluate_experiment)(args)


if __name__ == "__main__":
    main()
