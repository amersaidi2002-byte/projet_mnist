"""Reproducible training with validation-based selection and one final test evaluation."""

import argparse
import hashlib
import json
import os
import platform
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from training.models import load_model, make_model
from training.preprocess import preprocess_digit


class CustomDigits(Dataset):
    def __init__(self, root: Path, records: list[dict]):
        self.root, self.records = root, records
        for record in records:
            path = (root / record["path"]).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError("Image path escapes the dataset directory")
            if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError(f"Dataset checksum mismatch: {path}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = cv2.imread(str(self.root / record["path"]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Cannot read image: {record['path']}")
        return prepare_tensor(image), record["label"]


def prepare_tensor(image):
    prepared = preprocess_digit(np.asarray(image, dtype=np.uint8), light_foreground=True)
    if prepared is None:
        raise ValueError("No eligible digit in training image; inspect the dataset")
    return torch.from_numpy(prepared)


def datasets(kind: str, root: Path, seed: int):
    if kind == "custom":
        manifest = json.loads((root / "splits.json").read_text())
        splits = manifest["splits"]
        paths = [r["path"] for split in splits.values() for r in split]
        if len(paths) != len(set(paths)):
            raise ValueError("Dataset splits overlap")
        return tuple(
            CustomDigits(root, splits[key]) for key in ("train", "validation", "test")
        ), manifest
    from torchvision.datasets import MNIST

    training = MNIST(root=root, train=True, download=True, transform=prepare_tensor)
    testing = MNIST(root=root, train=False, download=True, transform=prepare_tensor)
    indices = torch.randperm(len(training), generator=torch.Generator().manual_seed(seed)).tolist()
    validation_count = len(indices) // 10
    validation_indices, train_indices = indices[:validation_count], indices[validation_count:]
    manifest = {
        "seed": seed,
        "dataset": "MNIST",
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "test": "official test partition",
    }
    return (
        Subset(training, train_indices),
        Subset(training, validation_indices),
        testing,
    ), manifest


def evaluate(model, loader, device):
    model.eval()
    confusion = torch.zeros((10, 10), dtype=torch.int64)
    loss_sum = 0.0
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss_sum += torch.nn.functional.cross_entropy(logits, labels, reduction="sum").item()
            predicted = logits.argmax(1)
            confusion += torch.bincount((10 * labels + predicted).cpu(), minlength=100).reshape(
                10, 10
            )
    total = int(confusion.sum())
    if not total:
        raise ValueError("Cannot evaluate an empty split")
    return {
        "samples": total,
        "loss": loss_sum / total,
        "accuracy": float(confusion.diag().sum()) / total,
        "confusion_matrix": confusion.tolist(),
        "recall_per_class": [
            float(confusion[i, i]) / int(confusion[i].sum()) if confusion[i].sum() else None
            for i in range(10)
        ],
    }


def train(args):
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0 or args.threads < 1:
        raise ValueError("Epochs, batch size, learning rate and threads must be positive")
    if args.output.exists():
        raise ValueError(f"Output exists: {args.output}; choose a new run directory")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    model = load_model(args.model, args.checkpoint) if args.checkpoint else make_model(args.model)
    model.to(device)
    (train_set, validation_set, test_set), manifest = datasets(args.dataset, args.data, args.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    validation_loader = DataLoader(validation_set, batch_size=args.batch_size)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)
    args.output.mkdir(parents=True)
    config = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    config["versions"] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
    }
    if args.checkpoint:
        config["initial_checkpoint_sha256"] = hashlib.sha256(
            args.checkpoint.read_bytes()
        ).hexdigest()
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.output / "splits.json").write_text(json.dumps(manifest, indent=2) + "\n")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history, best_loss = [], float("inf")
    checkpoint = args.output / "best.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()  # Evaluation switches mode; explicitly restore it each epoch.
        train_loss, samples = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(images), labels)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(labels)
            samples += len(labels)
        validation = evaluate(model, validation_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss / samples,
                "validation_loss": validation["loss"],
                "validation_accuracy": validation["accuracy"],
            }
        )
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            torch.save(
                {
                    "model": args.model,
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "preprocessing": "threshold125-area-v1",
                },
                checkpoint,
            )
        print(json.dumps(history[-1]), flush=True)
    selected = load_model(args.model, checkpoint).to(device)
    metrics = evaluate(selected, test_loader, device)
    metrics["selected_epoch"] = torch.load(checkpoint, map_location="cpu", weights_only=True)[
        "epoch"
    ]
    metrics["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (args.output / "test_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({"test_accuracy": metrics["accuracy"], "checkpoint": str(checkpoint)}))


def main(default_model=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("mlp", "cnn"), default=default_model or "cnn")
    parser.add_argument("--dataset", choices=("mnist", "custom"), default="mnist")
    parser.add_argument("--data", type=Path, default=Path("data/mnist"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
