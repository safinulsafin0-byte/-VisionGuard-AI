"""Tests for image validation and in-memory decoding."""

import pytest
import cv2
import numpy as np

from visionguard.config import SafetyLimits
from visionguard.validation import (
    ImageValidationError,
    detect_magic_format,
    validate_and_decode_image,
)


def create_dummy_png_bytes(width: int = 100, height: int = 100) -> bytes:
    img = np.full((height, width, 3), 128, dtype=np.uint8)
    success, buffer = cv2.imencode(".png", img)
    assert success
    return buffer.tobytes()


def create_dummy_jpeg_bytes(width: int = 100, height: int = 100) -> bytes:
    img = np.full((height, width, 3), 128, dtype=np.uint8)
    success, buffer = cv2.imencode(".jpg", img)
    assert success
    return buffer.tobytes()


def test_empty_bytes_raises_error():
    with pytest.raises(ImageValidationError, match="empty file"):
        validate_and_decode_image(b"")


def test_corrupted_or_non_image_bytes():
    with pytest.raises(ImageValidationError, match="Unsupported or unrecognized file header"):
        validate_and_decode_image(b"not an image file just random text")


def test_magic_detection():
    png_bytes = create_dummy_png_bytes(10, 10)
    jpeg_bytes = create_dummy_jpeg_bytes(10, 10)
    assert detect_magic_format(png_bytes) == "png"
    assert detect_magic_format(jpeg_bytes) == "jpeg"
    assert detect_magic_format(b"123456789012") is None


def test_valid_image_decoding():
    png_bytes = create_dummy_png_bytes(120, 80)
    res = validate_and_decode_image(png_bytes, filename="sample.png")
    assert res.format_detected == "png"
    assert res.width == 120
    assert res.height == 80
    assert res.channels == 3
    assert res.image_bgr.shape == (80, 120, 3)


def test_excessive_file_size_raises_error(monkeypatch):
    small_bytes = create_dummy_png_bytes(10, 10)
    # Mock max bytes to be smaller than the test image
    monkeypatch.setattr(SafetyLimits, "MAX_FILE_BYTES", 10)
    with pytest.raises(ImageValidationError, match="exceeds maximum allowed limit"):
        validate_and_decode_image(small_bytes)


def test_excessive_dimensions_raises_error(monkeypatch):
    img_bytes = create_dummy_png_bytes(50, 50)
    monkeypatch.setattr(SafetyLimits, "MAX_WIDTH", 40)
    with pytest.raises(ImageValidationError, match="exceed maximum allowed dimension"):
        validate_and_decode_image(img_bytes)

