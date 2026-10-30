# Implementation and validation

## Corrections

| Original issue | Current behavior | Verification |
|---|---|---|
| CNN read a `CV_32F` matrix through `at<double>` | Both paths use float32 pixels | Python/C++ input and logit comparisons |
| No contour led to indexing `contours[-1]` | No eligible contour returns a normal no-detection result | Black/white blank-frame tests |
| CNN weights used unchecked reads and leaked allocations | Exact length and finite-value checks, instance-owned vectors | Missing/truncated/non-finite parameter tests |
| Included C++ parameters differed from PyTorch checkpoints | Both parameter sets are exported from the supplied `.pth` files | Logit comparisons and manifests with SHA-256 |
| Two copied video loops diverged | Shared application, preprocessing and argument handling | Image, video and input-error tests |
| Training/evaluation logic was duplicated and ran at import | Shared modules and guarded CLI entry points | Training smoke runs and Python checks |
| MLP script did not save its model | Both architectures save a selected checkpoint and metrics | Offline two-epoch training checks |
| Training mode was not restored after evaluation | `model.train()` is set at each epoch | Explicit training loop; validation/test use inference mode |
| Custom archive had an extra nested class-6 folder | Preparation normalizes paths and records stratified splits | All 199 images are processed and checksummed |
| MLP Dockerfile copied CNN source names | One root Dockerfile with `MODEL=cnn` or `MODEL=mlp` | CI builds both variants and checks `--help` |

## Numerical protocol

The reference executables accept 784 float values and print the ten output
logits. The tests compare the C++ implementation against a freshly loaded PyTorch
checkpoint on zero input, a fixed seeded random input and a diagonal pattern.
The tolerance is `atol=5e-5, rtol=1e-4`, accounting for float32 accumulation order.
Both predicted class and logits are checked. These are implementation tests,
not classification-accuracy measurements.

Image tests exercise both models on an original custom example, dump the C++
preprocessed pixels and compare them with Python. OpenCV versions may differ by
one uint8 gray level in resize rounding, so input tolerance is `1/255 + 1e-7`.
The forward-pass tests use exactly identical input values and the tighter logit
tolerance. Empty images and invalid parameters are tested separately.

## Data and checkpoint provenance

The original archive has 199 PNG files: 20 per class except digit 5, which has
19. Class 6 has an additional `digits/` directory. `training.prepare_data`
normalizes this layout and writes per-file SHA-256 values with seeded split
membership: 119 train, 40 validation and 40 test images.

`training/CNN_weights.pth` and `training/mlp_weights.pth` are the original supplied
checkpoints. Their training partitions cannot be reconstructed reliably from
the original unseeded notebooks. The current parameter files under `inference_c/`
are exported from these checkpoints, and their manifests name the exact source
checkpoint digest. The old generated MLP header and mismatched CNN bias are
recoverable from Git history.

A new MNIST training run reserves 10% of the official training set for validation
and evaluates the official test set only after selecting the best validation-loss
checkpoint. A custom run uses `splits.json` and checks file digests before fitting.
The same preprocessing is used for training and image inference. Original
checkpoints were trained with the historical preprocessing; do not attach the
old notebook scores to this updated pipeline.

## What a smoke run proves

Two epochs on the small custom dataset exercise loading, deterministic splitting,
training, validation selection, checkpoint reload and metric export. They are
workflow checks, not tuned models or comparative benchmarks. Training outputs
remain in ignored `runs/` directories; the original included checkpoints are
not overwritten by smoke checks.

## Remaining deployment checks

Image and video processing can be tested on a desktop CPU. Live Raspberry Pi
camera capture, H.264 TCP transport, end-to-end latency, sustained frame rate and
thermal behavior need measurements on the actual target. Report inference-only
latency separately from preprocessing, decode/encode and transport. A zero-digit
frame is handled safely, but a non-digit contour can still be classified: there
is no trained out-of-distribution rejection stage.

## Refactoring check run — 30 October 2026

- GCC 13.3, C++17: core and both OpenCV applications compiled with warnings
  treated as errors for the application builds.
- CMake core test: passed. Address/undefined-behavior sanitizer checks passed
  with leak detection disabled because this container cannot inspect thread state.
  LeakSanitizer itself could not run in this environment.
- Python regression suite: **16 passed, no skips** with both C++ applications built.
- Python 3.12, PyTorch 2.15.0+cpu, Python OpenCV 4.12.0; C++ OpenCV 4.6.0.
- Both models completed two-epoch offline training smoke runs on the new custom
  split, saved a best checkpoint and wrote final metrics.
- Ruff lint and formatting checks passed for all maintained Python code.

These checks ran on a Linux x86_64 CPU, not a Raspberry Pi. They make no new
classification or performance claim. GitHub Actions additionally contains Docker
build checks, which need a Docker-capable runner.
