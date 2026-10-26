"""Tests for synthetic data generation reproducibility."""

import numpy as np
from visionguard.synthetic_data import (
    generate_sharp_checkerboard,
    generate_blurred_checkerboard,
    generate_underexposed_pattern,
    generate_overexposed_pattern,
    generate_low_contrast_flat,
    generate_surface_with_artifact,
    get_all_synthetic_test_cases,
)


def test_synthetic_reproducibility():
    img1 = generate_underexposed_pattern(128, 128, seed=123)
    img2 = generate_underexposed_pattern(128, 128, seed=123)
    np.testing.assert_array_equal(img1, img2)

    flat1 = generate_low_contrast_flat(128, 128, seed=999)
    flat2 = generate_low_contrast_flat(128, 128, seed=999)
    np.testing.assert_array_equal(flat1, flat2)


def test_synthetic_suite_shapes_and_types():
    suite = get_all_synthetic_test_cases()
    assert len(suite) == 6
    for name, img in suite.items():
        assert isinstance(img, np.ndarray), f"{name} is not numpy array"
        assert img.dtype == np.uint8, f"{name} dtype is {img.dtype}"
        assert len(img.shape) == 3, f"{name} shape is {img.shape}"
        assert img.shape[2] == 3, f"{name} channels count is not 3"

