"""Compatibility entry point: python -m training.train_cnn --output runs/cnn."""

from training.train import main

if __name__ == "__main__":
    main(default_model="cnn")
