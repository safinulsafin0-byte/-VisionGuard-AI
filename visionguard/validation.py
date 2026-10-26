"""Safe image validation and in-memory decoding for VisionGuard AI."""

from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np

from visionguard.config import SafetyLimits


class ImageValidationError(ValueError):
    """Raised when an uploaded file violates safety limits or cannot be safely decoded."""
    pass


@dataclass(frozen=True)
class ValidationResult:
    """Encapsulates a successfully validated and decoded image."""
    image_bgr: np.ndarray
    format_detected: str
    height: int
    width: int
    channels: int
    byte_size: int

    @property
    def total_pixels(self) -> int:
        return self.height * self.width


def detect_magic_format(data: bytes) -> Optional[str]:
    """Inspect leading bytes to detect image container without trusting filename extension."""
    if len(data) < 12:
        return None
    # PNG: 89 50 4E 47 0D 0A 1A 0A
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    # JPEG: FF D8 FF
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    # WebP: RIFF .... WEBP
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "webp"
    return None


def validate_and_decode_image(data: bytes, filename: Optional[str] = None) -> ValidationResult:
    """Validate byte length, magic header, decode with OpenCV, and enforce dimension bounds.
    
    Args:
        data: Raw image bytes from memory.
        filename: Optional filename hint for diagnostic messaging.
        
    Returns:
        ValidationResult with decoded BGR array and verified metadata.
        
    Raises:
        ImageValidationError: If validation fails or the image is corrupt/unsupported.
    """
    byte_size = len(data)
    if byte_size == 0:
        raise ImageValidationError("Received an empty file (0 bytes).")

    if byte_size > SafetyLimits.MAX_FILE_BYTES:
        max_mb = SafetyLimits.MAX_FILE_BYTES / (1024 * 1024)
        curr_mb = byte_size / (1024 * 1024)
        raise ImageValidationError(
            f"File size ({curr_mb:.2f} MB) exceeds maximum allowed limit of {max_mb:.1f} MB."
        )

    detected_fmt = detect_magic_format(data)
    if not detected_fmt:
        name_hint = f" for '{filename}'" if filename else ""
        raise ImageValidationError(
            f"Unsupported or unrecognized file header{name_hint}. "
            f"Only valid PNG, JPEG, and WebP files are supported."
        )

    # Convert bytes buffer to numpy array and decode in-memory via OpenCV
    try:
        np_buf = np.frombuffer(data, dtype=np.uint8)
        img_bgr = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
    except Exception as exc:
        raise ImageValidationError(f"OpenCV decode error: {str(exc)}") from exc

    if img_bgr is None or img_bgr.size == 0:
        raise ImageValidationError(
            "Failed to decode image data into valid pixel buffers. "
            "The file may be corrupted, truncated, or malformed."
        )

    height, width = img_bgr.shape[:2]
    channels = img_bgr.shape[2] if len(img_bgr.shape) == 3 else 1

    if height <= 0 or width <= 0:
        raise ImageValidationError(f"Invalid image geometry: {width}x{height}.")

    if width > SafetyLimits.MAX_WIDTH or height > SafetyLimits.MAX_HEIGHT:
        raise ImageValidationError(
            f"Image dimensions ({width}x{height}) exceed maximum allowed dimension "
            f"({SafetyLimits.MAX_WIDTH}x{SafetyLimits.MAX_HEIGHT})."
        )

    total_pixels = width * height
    if total_pixels > SafetyLimits.MAX_PIXELS:
        raise ImageValidationError(
            f"Image total pixel count ({total_pixels:,}) exceeds limit of {SafetyLimits.MAX_PIXELS:,}."
        )

    return ValidationResult(
        image_bgr=img_bgr,
        format_detected=detected_fmt,
        height=height,
        width=width,
        channels=channels,
        byte_size=byte_size,
    )

