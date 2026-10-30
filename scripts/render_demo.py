"""Render fixed original examples with predictions from both committed checkpoints."""

import argparse
from pathlib import Path

import cv2
import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from training.models import load_model
from training.preprocess import preprocess_digit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/custom"))
    parser.add_argument("--output", type=Path, default=Path("docs/assets/digit-demo.png"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    models = {
        name: load_model(name, Path("training") / file)
        for name, file in [("cnn", "CNN_weights.pth"), ("mlp", "mlp_weights.pth")]
    }
    figure, axes = plt.subplots(2, 5, figsize=(12, 6.7), facecolor="#f5f7fb")
    for label, axis in enumerate(axes.flat):
        path = args.data / str(label) / "digit_0.png"
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Cannot read demo example: {path}")
        prepared = preprocess_digit(image, light_foreground=True)
        predictions = {}
        with torch.inference_mode():
            for name, model in models.items():
                predictions[name] = int(model(torch.from_numpy(prepared).unsqueeze(0)).argmax())
        axis.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        axis.set_title(
            f"Digit {label}  |  CNN {predictions['cnn']} · MLP {predictions['mlp']}",
            fontsize=10,
            color="#152334",
        )
        axis.axis("off")
    figure.suptitle(
        "Embedded Digit Recognition",
        x=0.06,
        ha="left",
        fontsize=21,
        fontweight="bold",
        color="#152334",
    )
    figure.text(
        0.06,
        0.91,
        "PyTorch checkpoints → C++ inference · fixed original examples",
        fontsize=11,
        color="#41566e",
    )
    figure.text(
        0.06,
        0.025,
        "Examples were used during model development; this gallery is not a held-out benchmark.",
        fontsize=10,
        color="#41566e",
    )
    figure.subplots_adjust(left=0.06, right=0.98, bottom=0.12, top=0.80, wspace=0.25, hspace=0.45)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=150, facecolor=figure.get_facecolor())
    plt.close(figure)


if __name__ == "__main__":
    main()
