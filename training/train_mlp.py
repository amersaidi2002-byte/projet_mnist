"""Compatibility entry point: python -m training.train_mlp --output runs/mlp."""

from training.train import main

if __name__ == "__main__":
    main(default_model="mlp")
