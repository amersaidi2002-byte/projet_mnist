"""Model definitions shared by training, evaluation and C++ parity checks."""

from pathlib import Path

import torch
from torch import nn


class MLP(nn.Module):
    """784 -> 128 -> 64 -> 10, preserving the original checkpoint keys."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x.flatten(1)))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class CNN(nn.Module):
    """Two single-channel 5x5 convolutions and a linear classification head."""

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 1, 5, padding=2), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(1, 1, 5, padding=2), nn.ReLU())
        self.layer3 = nn.Linear(784, 10)

    def forward(self, x):
        return self.layer3(self.layer2(self.layer1(x)).flatten(1))


def make_model(name: str) -> nn.Module:
    if name not in {"mlp", "cnn"}:
        raise ValueError(f"Unknown model: {name}")
    return MLP() if name == "mlp" else CNN()


def load_model(name: str, checkpoint: str | Path) -> nn.Module:
    """Load original state dictionaries or checkpoints created by training.train."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload.get("state_dict", payload)
    if "model" in payload and payload["model"] != name:
        raise ValueError(f"Checkpoint is for {payload['model']}, not {name}")
    model = make_model(name)
    model.load_state_dict(state, strict=True)
    for parameter in model.parameters():
        if not torch.isfinite(parameter).all():
            raise ValueError("Checkpoint contains non-finite weights")
    return model.eval()
