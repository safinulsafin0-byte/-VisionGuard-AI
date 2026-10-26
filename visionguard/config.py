"""Configuration parameters, thresholds, and operational limits for VisionGuard AI."""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class SafetyLimits:
    """Hard boundaries for image upload and memory safety."""
    MAX_FILE_BYTES: int = 15 * 1024 * 1024  # 15 MB
    MAX_WIDTH: int = 4096                   # 4096 px
    MAX_HEIGHT: int = 4096                  # 4096 px
    MAX_PIXELS: int = 16_000_000            # 16 Megapixels
    ALLOWED_EXTENSIONS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp")
    ALLOWED_MIME_TYPES: Tuple[str, ...] = ("image/png", "image/jpeg", "image/webp")


@dataclass(frozen=True)
class VideoSafetyLimits:
    """Hard boundaries for video upload and memory safety."""
    MAX_VIDEO_FILE_BYTES: int = 50 * 1024 * 1024  # 50 MB
    MAX_FRAMES_TO_SAMPLE: int = 20                 # Upper bound on sampled frames
    DEFAULT_SAMPLE_COUNT: int = 5                  # Default number of frames sampled
    ALLOWED_VIDEO_EXTENSIONS: Tuple[str, ...] = (".mp4", ".avi", ".mov")


@dataclass(frozen=True)
class InspectionThresholds:
    """Configurable baseline thresholds for visual quality heuristics.
    
    All indicators are measured heuristics designed to assist human inspectors,
    not universal truths or autonomous pass/fail judges.
    """
    # Focus / Sharpness indicator: Laplacian variance
    # Note: Flat/textureless surfaces naturally yield low variance without being out of focus.
    BLUR_LAPLACIAN_VAR_WARN: float = 100.0

    # Brightness indicators (0-255 scale)
    UNDEREXPOSURE_MEAN_WARN: float = 45.0
    OVEREXPOSURE_MEAN_WARN: float = 215.0

    # Contrast indicators (RMS contrast = standard deviation of grayscale intensities)
    LOW_CONTRAST_STD_WARN: float = 25.0

    # Canny Edge Detection Parameters
    CANNY_THRESHOLD_1: float = 50.0
    CANNY_THRESHOLD_2: float = 150.0

    # Candidate Region Extraction
    MIN_CANDIDATE_AREA_PX: int = 120
    MAX_CANDIDATE_AREA_FRAC: float = 0.40  # Regions covering >40% of total area are treated as global features
    MAX_CANDIDATE_REGIONS: int = 25        # Cap candidates to prevent flooding the human reviewer
    MORPH_KERNEL_SIZE: int = 5             # Kernel size for edge clustering dilation


# Standard disclaimer appended to all visual inspection reports
HUMAN_REVIEW_DISCLAIMER: str = (
    "DISCLAIMER: VisionGuard AI is an engineering assistance tool for human reviewers. "
    "Measured indicators (Laplacian variance, intensity distributions, edge densities) are mathematical "
    "heuristics and do NOT constitute autonomous proof of defect, defect absence, or equipment safety. "
    "All highlighted regions are candidate areas suggested for human visual inspection."
)
