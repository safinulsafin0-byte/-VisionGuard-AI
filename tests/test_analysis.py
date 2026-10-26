"""Tests for OpenCV analysis pipeline, metrics, rules, and candidate extraction."""

import cv2
import numpy as np
import pytest

from visionguard.analysis import (
    analyze_image,
    compute_quality_metrics,
    evaluate_inspection_rules,
    extract_candidate_regions,
)
from visionguard.config import InspectionThresholds
from visionguard.synthetic_data import (
    generate_blurred_checkerboard,
    generate_low_contrast_flat,
    generate_overexposed_pattern,
    generate_sharp_checkerboard,
    generate_surface_with_artifact,
    generate_underexposed_pattern,
)


def test_sharp_vs_blurred_laplacian_variance():
    sharp = generate_sharp_checkerboard(256, 256)
    blurred = generate_blurred_checkerboard(256, 256)

    res_sharp = analyze_image(sharp)
    res_blurred = analyze_image(blurred)

    # Sharp checkerboard should have high Laplacian variance (> 1000)
    assert res_sharp.metrics.laplacian_variance > 1000.0
    # Blurred checkerboard should have very low Laplacian variance (< 50)
    assert res_blurred.metrics.laplacian_variance < 50.0

    # Verify that blur warning is triggered for blurred pattern and NOT for sharp pattern
    sharp_warnings = [n for n in res_sharp.notices if n.category == "Sharpness" and n.severity == "warning"]
    blurred_warnings = [n for n in res_blurred.notices if n.category == "Sharpness" and n.severity == "warning"]

    assert len(sharp_warnings) == 0
    assert len(blurred_warnings) == 1
    assert "focus blur" in blurred_warnings[0].reason


def test_underexposure_and_overexposure_rules():
    under = generate_underexposed_pattern(256, 256, mean_target=20)
    over = generate_overexposed_pattern(256, 256, mean_target=240)

    res_under = analyze_image(under)
    res_over = analyze_image(over)

    assert res_under.metrics.mean_brightness < 30.0
    under_notices = [n for n in res_under.notices if n.category == "Illumination" and n.severity == "warning"]
    assert len(under_notices) == 1
    assert "underexposure" in under_notices[0].reason

    assert res_over.metrics.mean_brightness > 230.0
    over_notices = [n for n in res_over.notices if n.category == "Illumination" and n.severity == "warning"]
    assert len(over_notices) == 1
    assert "saturation" in over_notices[0].reason


def test_low_contrast_rule():
    flat = generate_low_contrast_flat(256, 256, base_gray=128)
    res_flat = analyze_image(flat)

    assert res_flat.metrics.rms_contrast < 10.0
    contrast_notices = [n for n in res_flat.notices if n.category == "Contrast" and n.severity == "warning"]
    assert len(contrast_notices) == 1


def test_candidate_region_extraction_on_synthetic_artifact():
    artifact_img, (art_x, art_y, art_w, art_h) = generate_surface_with_artifact(512, 512)
    res = analyze_image(artifact_img)

    # We expect at least one candidate region identified on the surface
    assert len(res.candidate_regions) >= 1

    # Check that at least one candidate bounding box overlaps the known artifact coordinates
    found_match = False
    for cand in res.candidate_regions:
        cx, cy, cw, ch = cand.bounding_box
        # Check intersection
        x_left = max(art_x, cx)
        y_top = max(art_y, cy)
        x_right = min(art_x + art_w, cx + cw)
        y_bottom = min(art_y + art_h, cy + ch)

        if x_right > x_left and y_bottom > y_top:
            found_match = True
            assert cand.review_label == "Candidate region for review"
            assert cand.internal_edge_density > 0.0
            break

    assert found_match, "Did not detect candidate region overlapping known artifact."

