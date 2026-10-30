"""Classify a local image with a supplied checkpoint; no retraining is required."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import cv2
import torch

from training.models import load_model
from training.preprocess import preprocess_digit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("mlp", "cnn"), default="cnn")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--light-foreground", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    checkpoint = args.checkpoint or Path(__file__).parent / (
        "CNN_weights.pth" if args.model == "cnn" else "mlp_weights.pth"
    )
    model = load_model(args.model, checkpoint)
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        parser.error(f"Cannot read image: {args.image}")
    prepared = preprocess_digit(image, light_foreground=args.light_foreground)
    result = {"detected": prepared is not None}
    if prepared is not None:
        with torch.inference_mode():
            started = perf_counter()
            logits = model(torch.from_numpy(prepared).unsqueeze(0))[0]
            milliseconds = 1000 * (perf_counter() - started)
        result.update(
            prediction=int(logits.argmax()), logits=logits.tolist(), inference_ms=milliseconds
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        overlay = cv2.resize(image, (max(280, image.shape[1]), max(280, image.shape[0])))
        caption = f"Prediction: {result['prediction']}" if result["detected"] else "No digit found"
        cv2.putText(overlay, caption, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 180, 0), 2)
        if not cv2.imwrite(str(args.output), overlay):
            parser.error(f"Cannot write output: {args.output}")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
