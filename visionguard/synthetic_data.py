"""Deterministic synthetic image generators for evaluation and testing.

Generates reproducible test patterns with known mathematical properties
to systematically verify quality heuristics and candidate extraction.
"""

from typing import Dict, Tuple
import cv2
import numpy as np


def generate_sharp_checkerboard(
    width: int = 512,
    height: int = 512,
    square_size: int = 32,
) -> np.ndarray:
    """Generate high-contrast sharp checkerboard with high Laplacian variance."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(0, height, square_size):
        for x in range(0, width, square_size):
            if ((x // square_size) + (y // square_size)) % 2 == 0:
                img[y : y + square_size, x : x + square_size] = [240, 240, 240]
            else:
                img[y : y + square_size, x : x + square_size] = [20, 20, 20]
    return img


def generate_blurred_checkerboard(
    width: int = 512,
    height: int = 512,
    blur_kernel: int = 31,
) -> np.ndarray:
    """Generate heavily blurred checkerboard with low Laplacian variance."""
    sharp = generate_sharp_checkerboard(width, height)
    return cv2.GaussianBlur(sharp, (blur_kernel, blur_kernel), 0)


def generate_underexposed_pattern(
    width: int = 512,
    height: int = 512,
    mean_target: int = 20,
    seed: int = 42,
) -> np.ndarray:
    """Generate low-light / underexposed synthetic surface."""
    rng = np.random.default_rng(seed)
    noise = rng.integers(-5, 6, size=(height, width, 3), dtype=np.int16)
    base = np.full((height, width, 3), mean_target, dtype=np.int16) + noise
    return np.clip(base, 0, 255).astype(np.uint8)


def generate_overexposed_pattern(
    width: int = 512,
    height: int = 512,
    mean_target: int = 235,
    seed: int = 42,
) -> np.ndarray:
    """Generate saturated / overexposed synthetic surface."""
    rng = np.random.default_rng(seed)
    noise = rng.integers(-5, 6, size=(height, width, 3), dtype=np.int16)
    base = np.full((height, width, 3), mean_target, dtype=np.int16) + noise
    return np.clip(base, 0, 255).astype(np.uint8)


def generate_low_contrast_flat(
    width: int = 512,
    height: int = 512,
    base_gray: int = 128,
    seed: int = 42,
) -> np.ndarray:
    """Generate uniform low-contrast field."""
    rng = np.random.default_rng(seed)
    noise = rng.integers(-3, 4, size=(height, width, 3), dtype=np.int16)
    base = np.full((height, width, 3), base_gray, dtype=np.int16) + noise
    return np.clip(base, 0, 255).astype(np.uint8)


def generate_surface_with_artifact(
    width: int = 512,
    height: int = 512,
    seed: int = 42,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Generate uniform brushed surface with an intentional scratch/blemish artifact.
    
    Returns:
        Tuple of (image_bgr, (artifact_x, artifact_y, artifact_w, artifact_h))
    """
    rng = np.random.default_rng(seed)
    # Background: brushed metal tone ~160 with subtle gradient
    x_coords = np.linspace(150, 170, width, dtype=np.float32)
    bg_gray = np.tile(x_coords, (height, 1))
    noise = rng.normal(0, 2, (height, width)).astype(np.float32)
    bg = np.clip(bg_gray + noise, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)

    # Insert high-contrast dark scratch/artifact
    art_x, art_y = 200, 220
    art_w, art_h = 50, 40
    cv2.ellipse(
        img,
        (art_x + art_w // 2, art_y + art_h // 2),
        (art_w // 2, art_h // 3),
        angle=30,
        startAngle=0,
        endAngle=360,
        color=(25, 25, 25),
        thickness=-1,
    )
    # Add a thin cross scratch
    cv2.line(
        img,
        (art_x + 5, art_y + art_h - 5),
        (art_x + art_w - 5, art_y + 5),
        color=(10, 10, 10),
        thickness=2,
    )

    return img, (art_x, art_y, art_w, art_h)


def get_all_synthetic_test_cases() -> Dict[str, np.ndarray]:
    """Retrieve full suite of labeled synthetic test patterns."""
    artifact_img, _ = generate_surface_with_artifact()
    return {
        "sharp_pattern": generate_sharp_checkerboard(),
        "blurred_pattern": generate_blurred_checkerboard(),
        "underexposed_pattern": generate_underexposed_pattern(),
        "overexposed_pattern": generate_overexposed_pattern(),
        "low_contrast_pattern": generate_low_contrast_flat(),
        "surface_with_artifact": artifact_img,
    }

