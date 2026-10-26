"""Visualization, annotation, and cropping utilities for VisionGuard AI."""

from typing import List, Optional, Tuple
import cv2
import numpy as np

from visionguard.analysis import CandidateRegion


def annotate_candidate_regions(
    image_rgb: np.ndarray,
    candidate_regions: List[CandidateRegion],
    box_color_rgb: Tuple[int, int, int] = (255, 140, 0),  # Amber / Warning Orange
    text_color_rgb: Tuple[int, int, int] = (255, 255, 255),
    line_thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes and region identifiers on a copy of the RGB image.
    
    Args:
        image_rgb: Base 3-channel RGB image.
        candidate_regions: List of extracted candidate regions.
        box_color_rgb: Tuple for box outline in RGB.
        text_color_rgb: Tuple for text label in RGB.
        line_thickness: Pixel thickness of outlines.
        
    Returns:
        New RGB numpy array with rendered annotations.
    """
    annotated = image_rgb.copy()
    h, w = annotated.shape[:2]

    # Scale font and box thickness dynamically with image resolution
    scale_factor = max(0.5, min(1.2, min(w, h) / 800.0))
    thickness = max(1, int(round(line_thickness * scale_factor)))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45 * scale_factor

    for cand in candidate_regions:
        x, y, bw, bh = cand.bounding_box
        x2 = min(w - 1, x + bw)
        y2 = min(h - 1, y + bh)

        # Draw candidate bounding box
        cv2.rectangle(annotated, (x, y), (x2, y2), box_color_rgb, thickness)

        # Draw banner for ID and label
        label = f"#{cand.region_id} Candidate"
        (lbl_w, lbl_h), baseline = cv2.getTextSize(label, font, font_scale, 1)

        # Ensure label banner fits inside image boundaries
        banner_y1 = max(0, y - lbl_h - baseline - 4)
        banner_y2 = y if banner_y1 < y else y + lbl_h + baseline + 4
        banner_x2 = min(w - 1, x + lbl_w + 6)

        cv2.rectangle(annotated, (x, banner_y1), (banner_x2, banner_y2), box_color_rgb, -1)
        text_y = max(lbl_h + 2, y - baseline - 2) if banner_y1 < y else y + lbl_h + 2
        cv2.putText(
            annotated,
            label,
            (x + 3, text_y),
            font,
            font_scale,
            text_color_rgb,
            1,
            cv2.LINE_AA,
        )

    return annotated


def extract_candidate_crop(
    image_rgb: np.ndarray,
    candidate: CandidateRegion,
    padding_fraction: float = 0.15,
) -> np.ndarray:
    """Extract a cropped view of a candidate region with context padding.
    
    Args:
        image_rgb: Base 3-channel RGB image.
        candidate: Selected candidate region.
        padding_fraction: Context margin around the bounding box.
        
    Returns:
        Cropped RGB numpy array.
    """
    h, w = image_rgb.shape[:2]
    x, y, bw, bh = candidate.bounding_box

    pad_x = int(bw * padding_fraction)
    pad_y = int(bh * padding_fraction)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + bw + pad_x)
    y2 = min(h, y + bh + pad_y)

    return image_rgb[y1:y2, x1:x2].copy()


def encode_image_bytes(image: np.ndarray, is_rgb: bool = True, fmt: str = ".png") -> bytes:
    """Encode an in-memory image array into bytes for download or report export."""
    img_to_encode = image
    if is_rgb and len(image.shape) == 3 and image.shape[2] == 3:
        img_to_encode = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    success, buffer = cv2.imencode(fmt, img_to_encode)
    if not success:
        raise RuntimeError(f"Failed to encode image array to {fmt} format.")
    return buffer.tobytes()

