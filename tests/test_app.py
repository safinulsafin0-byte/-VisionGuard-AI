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


def test_ui_styling_text_clarity_and_contrast_rules():
    """Verify that all UI text styles enforce high contrast and zero text shadows or glows."""
    with open("app.py", "r", encoding="utf-8") as f:
        app_code = f.read()

    # Verify global shadow removal
    assert "text-shadow: none !important;" in app_code
    assert "filter: none !important;" in app_code

    # Verify no faint #9ca3af in metric hints or labels
    assert "color: #9ca3af;" not in app_code

    # Verify high-contrast text color defaults
    assert "color: #0f172a !important;" in app_code
    assert "color: #1e293b !important;" in app_code

    # Inspect sync_player index.html
    with open("visionguard/components/sync_player/index.html", "r", encoding="utf-8") as f:
        html_code = f.read()

    assert "text-shadow: none !important;" in html_code
    assert "color:#64748b;" not in html_code
    assert "font-weight: 700;" in html_code or "font-weight: 800;" in html_code


def test_streamlit_cloud_dependencies():
    """Verify headless OpenCV and Linux OS dependencies for Streamlit Cloud deployment."""
    import os

    with open("requirements.txt", "r", encoding="utf-8") as f:
        reqs = f.read()

    assert "opencv-python-headless" in reqs, "opencv-python-headless must be specified for cloud deployment"
    assert "opencv-python==" not in reqs, "Conflicting non-headless opencv-python must not be in requirements.txt"

    assert os.path.exists("packages.txt"), "packages.txt must be present for Streamlit Cloud system libraries"
    with open("packages.txt", "r", encoding="utf-8") as f:
        pkgs = f.read()

    assert "libgl1" in pkgs, "packages.txt must include libgl1"
    assert "libglib2.0-0" in pkgs, "packages.txt must include libglib2.0-0"



