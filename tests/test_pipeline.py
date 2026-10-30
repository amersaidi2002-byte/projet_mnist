"""Regression tests for unsafe image access, dataset splits and C++/PyTorch parity."""

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from training.export import export
from training.models import load_model, make_model
from training.prepare_data import prepare
from training.preprocess import preprocess_digit
from training.train import CustomDigits

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def custom(tmp_path_factory):
    path = tmp_path_factory.mktemp("digits") / "data"
    manifest = prepare(ROOT / "digits_IA_embarque.zip", path)
    return path, manifest


@pytest.mark.parametrize("background", [0, 255])
def test_empty_frame_is_not_classified(background):
    assert preprocess_digit(np.full((64, 64), background, np.uint8)) is None


def test_invalid_image_type_is_rejected():
    with pytest.raises(ValueError):
        preprocess_digit(np.zeros((28, 28), dtype=np.float32))


def test_polarity_and_centering_are_consistent():
    light = np.zeros((80, 100), np.uint8)
    cv2.rectangle(light, (40, 10), (55, 65), 255, -1)
    a = preprocess_digit(light, light_foreground=True)
    b = preprocess_digit(255 - light)
    assert a.shape == (1, 28, 28) and a.dtype == np.float32
    np.testing.assert_array_equal(a, b)
    assert 0 <= a.min() <= a.max() <= 1


def test_stratified_splits_and_checksums(custom):
    root, manifest = custom
    sets = [
        {r["path"] for r in manifest["splits"][name]} for name in ("train", "validation", "test")
    ]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert [len(s) for s in sets] == [119, 40, 40]
    for records in manifest["splits"].values():
        assert {r["label"] for r in records} == set(range(10))
        dataset = CustomDigits(root, records)
        for image, label in dataset:
            assert image.shape == (1, 28, 28) and 0 <= label < 10
    sample = root / manifest["splits"]["train"][0]["path"]
    original = sample.read_bytes()
    try:
        sample.write_bytes(b"corrupted")
        with pytest.raises(ValueError, match="checksum"):
            CustomDigits(root, manifest["splits"]["train"])
    finally:
        sample.write_bytes(original)


@pytest.mark.parametrize(
    "name,checkpoint", [("cnn", "CNN_weights.pth"), ("mlp", "mlp_weights.pth")]
)
def test_original_checkpoints_and_export(name, checkpoint, tmp_path):
    model = load_model(name, ROOT / "training" / checkpoint)
    assert model(torch.zeros(2, 1, 28, 28)).shape == (2, 10)
    result = export(name, ROOT / "training" / checkpoint, tmp_path / name)
    assert result["files"] and (tmp_path / name / "manifest.json").exists()
    with pytest.raises(ValueError, match="exists"):
        export(name, ROOT / "training" / checkpoint, tmp_path / name)


@pytest.mark.parametrize(
    "name,checkpoint", [("cnn", "CNN_weights.pth"), ("mlp", "mlp_weights.pth")]
)
def test_cpp_logits_match_pytorch(name, checkpoint):
    executable = ROOT / "build" / f"{name}_reference"
    if not executable.exists():
        pytest.skip("Build C++ targets first to run numerical parity tests")
    model = load_model(name, ROOT / "training" / checkpoint)
    torch.set_num_threads(1)
    rng = np.random.default_rng(1234)
    for sample in [
        np.zeros(784, np.float32),
        rng.random(784, dtype=np.float32),
        np.eye(28, dtype=np.float32).reshape(-1),
    ]:
        command = [str(executable), str(ROOT / "inference_c" / name.upper())]
        result = subprocess.run(
            command, input=" ".join(map(str, sample)), capture_output=True, text=True, check=True
        )
        actual = np.fromstring(result.stdout, sep="\n")
        with torch.inference_mode():
            expected = model(torch.from_numpy(sample).reshape(1, 1, 28, 28)).numpy()[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=5e-5)
        assert actual.argmax() == expected.argmax()


@pytest.mark.parametrize("name", ["mlp", "cnn"])
def test_cpp_image_preprocessing_and_blank_frame(name, custom, tmp_path):
    executable = ROOT / "build" / f"digit_{name}_app"
    if not executable.exists():
        pytest.skip("OpenCV C++ applications are not built")
    root, manifest = custom
    record = manifest["splits"]["test"][0]
    image = root / record["path"]
    dump = tmp_path / "input.bin"
    command = [
        str(executable),
        "--image",
        str(image),
        "--light-foreground",
        "--dump-input",
        str(dump),
        "--weights",
        str(ROOT / "inference_c" / name.upper()),
    ]
    result = json.loads(subprocess.check_output(command, text=True))
    expected = preprocess_digit(cv2.imread(str(image)), light_foreground=True).reshape(-1)
    # OpenCV versions can differ slightly in fixed-point resize rounding.
    np.testing.assert_allclose(np.fromfile(dump, dtype=np.float32), expected, atol=1 / 255 + 1e-7)
    assert result["detected"] and 0 <= result["prediction"] < 10
    blank = tmp_path / "blank.png"
    cv2.imwrite(str(blank), np.full((64, 64), 255, np.uint8))
    command = [
        str(executable),
        "--image",
        str(blank),
        "--weights",
        str(ROOT / "inference_c" / name.upper()),
    ]
    assert json.loads(subprocess.check_output(command, text=True)) == {"detected": False}


def test_model_returns_logits():
    for name in ("mlp", "cnn"):
        model = make_model(name)
        loss = torch.nn.functional.cross_entropy(
            model(torch.zeros(2, 1, 28, 28)), torch.tensor([0, 1])
        )
        loss.backward()
        assert all(p.grad is not None for p in model.parameters())


@pytest.mark.parametrize("bad_values", [b"short", np.full(25, np.nan, dtype="<f4").tobytes()])
def test_cpp_rejects_invalid_weight_files(bad_values, tmp_path):
    import shutil

    executable = ROOT / "build" / "cnn_reference"
    if not executable.exists():
        pytest.skip("Build the C++ targets first")
    weights = tmp_path / "weights"
    shutil.copytree(ROOT / "inference_c/CNN", weights)
    (weights / "layer1.0.weight.bin").write_bytes(bad_values)
    result = subprocess.run([str(executable), str(weights)], text=True, capture_output=True)
    assert result.returncode == 1
    assert "weight" in result.stderr.lower()


@pytest.mark.parametrize("name", ["cnn", "mlp"])
def test_cpp_video_handles_no_digit_and_invalid_arguments(name, tmp_path):
    executable = ROOT / "build" / f"digit_{name}_app"
    if not executable.exists():
        pytest.skip("Build the OpenCV applications first")
    video = tmp_path / "blank.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 64))
    assert writer.isOpened()
    for _ in range(5):
        writer.write(np.full((64, 64, 3), 255, np.uint8))
    writer.release()
    command = [
        str(executable),
        "--video",
        str(video),
        "--output",
        str(tmp_path / "output.avi"),
        "--weights",
        str(ROOT / "inference_c" / name.upper()),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    assert "Frames: 5, detected: 0" in result.stdout
    assert (tmp_path / "output.avi").stat().st_size > 0
    invalid = subprocess.run([str(executable), "--camera", "-1"], capture_output=True, text=True)
    assert invalid.returncode == 1 and "camera" in invalid.stderr
    missing = subprocess.run([str(executable), "--width"], capture_output=True, text=True)
    assert missing.returncode == 1 and "Missing value" in missing.stderr
