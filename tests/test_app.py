"""Smoke and component tests for app.py."""

import py_compile
from unittest.mock import MagicMock, patch
import pytest

from visionguard.config import InspectionThresholds, VideoSafetyLimits
from visionguard.synthetic_data import generate_sharp_checkerboard


def test_app_py_syntax_and_compilation():
    """Verify that app.py compiles cleanly to bytecode without syntax errors."""
    res = py_compile.compile("app.py", doraise=True)
    assert res is not None


def test_sidebar_thresholds_instantiation():
    """Verify InspectionThresholds accepts custom parameters used in sidebar."""
    thresh = InspectionThresholds(
        BLUR_LAPLACIAN_VAR_WARN=120.0,
        UNDEREXPOSURE_MEAN_WARN=50.0,
        OVEREXPOSURE_MEAN_WARN=210.0,
        LOW_CONTRAST_STD_WARN=20.0,
        CANNY_THRESHOLD_1=40.0,
        CANNY_THRESHOLD_2=140.0,
    )
    assert thresh.BLUR_LAPLACIAN_VAR_WARN == 120.0
    assert thresh.CANNY_THRESHOLD_1 == 40.0


def test_video_limits_configuration():
    """Verify VideoSafetyLimits boundaries are well-configured."""
    assert VideoSafetyLimits.MAX_VIDEO_FILE_BYTES == 50 * 1024 * 1024
    assert ".mp4" in VideoSafetyLimits.ALLOWED_VIDEO_EXTENSIONS
    assert ".avi" in VideoSafetyLimits.ALLOWED_VIDEO_EXTENSIONS

