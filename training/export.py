"""Export checked PyTorch parameters to the formats consumed by the C++ models."""

import argparse
import hashlib
import json
from pathlib import Path

from training.models import load_model


def export(model_name: str, checkpoint: Path, output: Path):
    if output.exists():
        raise ValueError(f"Output exists: {output}; choose a new directory")
    model = load_model(model_name, checkpoint)
    output.mkdir(parents=True)
    state = model.state_dict()
    for name, tensor in state.items():
        tensor.detach().cpu().numpy().astype("<f4").tofile(output / f"{name}.bin")
    manifest = {
        "model": model_name,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "files": {
            p.name: {
                "bytes": p.stat().st_size,
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            }
            for p in sorted(output.iterdir())
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("mlp", "cnn"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.model, args.checkpoint, args.output), indent=2))


if __name__ == "__main__":
    main()
