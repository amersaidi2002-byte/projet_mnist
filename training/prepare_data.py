"""Extract the original custom digits and record stratified split membership."""

import argparse
import hashlib
import json
from pathlib import Path
from random import Random
from zipfile import ZipFile


def prepare(archive: Path, destination: Path, seed: int = 42) -> dict:
    if destination.exists():
        raise ValueError(f"Output already exists: {destination}; choose a new directory")
    records = []
    with ZipFile(archive) as source:
        for name in sorted(source.namelist()):
            parts = Path(name).parts
            if not name.endswith(".png"):
                continue
            if (
                len(parts) not in (3, 4)
                or parts[0] != "digits"
                or parts[1] not in {f"digits{i}" for i in range(10)}
                or (len(parts) == 4 and parts[2] != "digits")
                or ".." in parts
            ):
                raise ValueError(f"Unexpected archive path: {name}")
            records.append((name, int(parts[1][-1]), source.read(name)))
    if {label for _, label, _ in records} != set(range(10)):
        raise ValueError("Archive must contain all ten digit classes")
    rng = Random(seed)
    manifest = {
        "seed": seed,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "splits": {"train": [], "validation": [], "test": []},
    }
    destination.mkdir(parents=True)
    for label in range(10):
        group = [r for r in records if r[1] == label]
        if len(group) < 5:
            raise ValueError("Each class needs at least five images")
        rng.shuffle(group)
        train_end, val_end = int(0.6 * len(group)), int(0.8 * len(group))
        for i, (name, _, content) in enumerate(group):
            relative = f"{label}/{Path(name).name}"
            target = destination / relative
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(content)
            split = "train" if i < train_end else "validation" if i < val_end else "test"
            manifest["splits"][split].append(
                {"path": relative, "label": label, "sha256": hashlib.sha256(content).hexdigest()}
            )
    (destination / "splits.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("digits_IA_embarque.zip"))
    parser.add_argument("--output", type=Path, default=Path("data/custom"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = prepare(args.archive, args.output, args.seed)
    print(json.dumps({key: len(value) for key, value in manifest["splits"].items()}))


if __name__ == "__main__":
    main()
