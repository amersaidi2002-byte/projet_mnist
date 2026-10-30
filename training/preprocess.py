"""The same threshold/crop/resize contract as inference_c/common/preprocess.hpp."""

import cv2
import numpy as np


def preprocess_digit(image: np.ndarray, *, light_foreground: bool = False):
    """Return float32 [1, 28, 28] in [0, 1], or None when no digit is found.

    Color inputs are BGR (as returned by cv2.imread). Camera images normally
    have a dark digit on a light background; MNIST and the bundled digits use
    light foreground. A single largest contour is used; this is not a detector
    for multiple digits or an out-of-distribution rejection model.
    """
    if image.dtype != np.uint8 or image.size == 0:
        raise ValueError("Expected a nonempty uint8 image")
    if image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 2:
        gray = image
    else:
        raise ValueError("Expected a grayscale or three-channel BGR image")
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    mode = cv2.THRESH_BINARY if light_foreground else cv2.THRESH_BINARY_INV
    _, binary = cv2.threshold(gray, 125, 255, mode)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    eligible = [c for c in contours if 4 <= cv2.contourArea(c) <= 0.9 * binary.size]
    if not eligible:
        return None
    x, y, width, height = cv2.boundingRect(max(eligible, key=cv2.contourArea))
    side = max(width, height)
    square = np.zeros((side, side), dtype=np.uint8)
    dx, dy = (side - width) // 2, (side - height) // 2
    square[dy : dy + height, dx : dx + width] = binary[y : y + height, x : x + width]
    resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(resized[np.newaxis], dtype=np.float32) / 255.0
