"""Core OpenCV 5 image analysis pipeline for VisionGuard AI."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np

from visionguard.config import InspectionThresholds


@dataclass(frozen=True)
class QualityMetrics:
    """Quantitative image statistics computed directly from pixel data."""
    # Sharpness / Focus
    laplacian_variance: float

    # Brightness (0-255 scale)
    mean_brightness: float
    median_brightness: float
    std_brightness: float
    min_brightness: int
    max_brightness: int
    channel_means_bgr: Tuple[float, float, float]

    # Contrast
    rms_contrast: float
    dynamic_range_p95_p5: float

    # Edges & Gradients
    canny_edge_density: float
    mean_sobel_gradient: float


@dataclass(frozen=True)
class CandidateRegion:
    """A bounded area highlighted for human visual inspection.
    
    CRITICAL: This is an algorithmic suggestion, NOT a verified defect.
    """
    region_id: int
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    contour_area: float
    aspect_ratio: float
    internal_edge_density: float
    review_label: str = "Candidate region for review"

    @property
    def bounding_box(self) -> Tuple[int, int, int, int]:
        return (self.bbox_x, self.bbox_y, self.bbox_w, self.bbox_h)


@dataclass(frozen=True)
class InspectionNotice:
    """Transparent rule evaluation note indicating why a warning was surfaced."""
    category: str
    severity: str  # 'warning' | 'info' | 'nominal'
    metric_name: str
    measured_value: float
    threshold_value: float
    condition: str
    reason: str
    suggested_human_action: str


@dataclass
class InspectionResult:
    """Complete inspection output combining raw measurements, notes, and visual maps."""
    image_bgr: np.ndarray
    image_rgb: np.ndarray
    gray_image: np.ndarray
    edge_map: np.ndarray
    metrics: QualityMetrics
    candidate_regions: List[CandidateRegion]
    notices: List[InspectionNotice]
    opencv_version: str = field(default_factory=lambda: cv2.__version__)


def compute_quality_metrics(
    gray: np.ndarray,
    img_bgr: np.ndarray,
    canny_t1: float = InspectionThresholds.CANNY_THRESHOLD_1,
    canny_t2: float = InspectionThresholds.CANNY_THRESHOLD_2,
) -> Tuple[QualityMetrics, np.ndarray]:
    """Compute mathematical indicators of image quality using OpenCV 5.
    
    Args:
        gray: Single-channel uint8 grayscale image.
        img_bgr: 3-channel BGR image.
        canny_t1: Lower hysteresis threshold for Canny.
        canny_t2: Upper hysteresis threshold for Canny.
        
    Returns:
        Tuple of (QualityMetrics, canny_edge_map).
    """
    # 1. Laplacian variance (Sharpness heuristic)
    laplacian_64f = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = float(laplacian_64f.var())

    # 2. Brightness statistics
    mean_bright = float(np.mean(gray))
    median_bright = float(np.median(gray))
    std_bright = float(np.std(gray))
    min_bright = int(np.min(gray))
    max_bright = int(np.max(gray))

    # Channel means in BGR order
    b_mean = float(np.mean(img_bgr[:, :, 0]))
    g_mean = float(np.mean(img_bgr[:, :, 1]))
    r_mean = float(np.mean(img_bgr[:, :, 2]))

    # 3. Contrast statistics
    # RMS contrast is standard deviation of grayscale intensities
    rms_contrast = std_bright
    p5, p95 = np.percentile(gray, [5.0, 95.0])
    dyn_range = float(p95 - p5)

    # 4. Edge and Gradient statistics
    edges = cv2.Canny(gray, int(canny_t1), int(canny_t2))
    edge_count = int(np.count_nonzero(edges))
    total_pixels = gray.shape[0] * gray.shape[1]
    edge_density = float(edge_count / total_pixels) if total_pixels > 0 else 0.0

    # Sobel gradient magnitude
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    mean_grad = float(np.mean(grad_mag))

    metrics = QualityMetrics(
        laplacian_variance=lap_var,
        mean_brightness=mean_bright,
        median_brightness=median_bright,
        std_brightness=std_bright,
        min_brightness=min_bright,
        max_brightness=max_bright,
        channel_means_bgr=(b_mean, g_mean, r_mean),
        rms_contrast=rms_contrast,
        dynamic_range_p95_p5=dyn_range,
        canny_edge_density=edge_density,
        mean_sobel_gradient=mean_grad,
    )
    return metrics, edges


def extract_candidate_regions(
    edge_map: np.ndarray,
    thresholds: InspectionThresholds = InspectionThresholds(),
) -> List[CandidateRegion]:
    """Cluster edge discontinuities into bounded candidate regions for human review.
    
    Uses morphological dilation to group nearby edge fragments, then extracts
    external contours and filters by area bounds.
    """
    h, w = edge_map.shape[:2]
    total_area = h * w
    max_area_px = total_area * thresholds.MAX_CANDIDATE_AREA_FRAC

    # Morphological closing / dilation to connect adjacent edge fragments
    k = thresholds.MORPH_KERNEL_SIZE
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    dilated = cv2.dilate(edge_map, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: List[CandidateRegion] = []
    # Sort contours by area descending so largest salient regions are indexed first
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    region_idx = 1
    for cnt in sorted_contours:
        if region_idx > thresholds.MAX_CANDIDATE_REGIONS:
            break

        area = float(cv2.contourArea(cnt))
        if area < thresholds.MIN_CANDIDATE_AREA_PX or area > max_area_px:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw <= 0 or bh <= 0:
            continue

        aspect_ratio = float(bw / bh)

        # Calculate edge density inside this specific bounding box
        crop_edges = edge_map[y : y + bh, x : x + bw]
        box_area = bw * bh
        int_edge_density = float(np.count_nonzero(crop_edges) / box_area) if box_area > 0 else 0.0

        candidates.append(
            CandidateRegion(
                region_id=region_idx,
                bbox_x=int(x),
                bbox_y=int(y),
                bbox_w=int(bw),
                bbox_h=int(bh),
                contour_area=area,
                aspect_ratio=round(aspect_ratio, 3),
                internal_edge_density=round(int_edge_density, 4),
            )
        )
        region_idx += 1

    return candidates


def evaluate_inspection_rules(
    metrics: QualityMetrics,
    thresholds: InspectionThresholds = InspectionThresholds(),
) -> List[InspectionNotice]:
    """Evaluate deterministic inspection rules with complete explanation of thresholds."""
    notices: List[InspectionNotice] = []

    # 1. Blur / Sharpness check
    if metrics.laplacian_variance < thresholds.BLUR_LAPLACIAN_VAR_WARN:
        notices.append(
            InspectionNotice(
                category="Sharpness",
                severity="warning",
                metric_name="Laplacian Variance",
                measured_value=round(metrics.laplacian_variance, 2),
                threshold_value=thresholds.BLUR_LAPLACIAN_VAR_WARN,
                condition="measured < threshold",
                reason=(
                    f"Laplacian variance ({metrics.laplacian_variance:.1f}) is below "
                    f"threshold ({thresholds.BLUR_LAPLACIAN_VAR_WARN:.1f}). Image may suffer from "
                    f"focus blur or motion blur, or is an inherently smooth/textureless surface."
                ),
                suggested_human_action=(
                    "Inspect image sharpness manually. Verify camera focus and recapture if high-frequency details are expected."
                ),
            )
        )
    else:
        notices.append(
            InspectionNotice(
                category="Sharpness",
                severity="nominal",
                metric_name="Laplacian Variance",
                measured_value=round(metrics.laplacian_variance, 2),
                threshold_value=thresholds.BLUR_LAPLACIAN_VAR_WARN,
                condition="measured >= threshold",
                reason=f"Sharpness indicator ({metrics.laplacian_variance:.1f}) meets minimum focus threshold.",
                suggested_human_action="No focus adjustment suggested by this metric.",
            )
        )

    # 2. Exposure check (Underexposure / Overexposure)
    if metrics.mean_brightness < thresholds.UNDEREXPOSURE_MEAN_WARN:
        notices.append(
            InspectionNotice(
                category="Illumination",
                severity="warning",
                metric_name="Mean Brightness",
                measured_value=round(metrics.mean_brightness, 2),
                threshold_value=thresholds.UNDEREXPOSURE_MEAN_WARN,
                condition="measured < threshold",
                reason=(
                    f"Mean intensity ({metrics.mean_brightness:.1f}) is below threshold "
                    f"({thresholds.UNDEREXPOSURE_MEAN_WARN:.1f}), indicating underexposure or low scene lighting."
                ),
                suggested_human_action="Check illumination levels or increase sensor exposure.",
            )
        )
    elif metrics.mean_brightness > thresholds.OVEREXPOSURE_MEAN_WARN:
        notices.append(
            InspectionNotice(
                category="Illumination",
                severity="warning",
                metric_name="Mean Brightness",
                measured_value=round(metrics.mean_brightness, 2),
                threshold_value=thresholds.OVEREXPOSURE_MEAN_WARN,
                condition="measured > threshold",
                reason=(
                    f"Mean intensity ({metrics.mean_brightness:.1f}) is above threshold "
                    f"({thresholds.OVEREXPOSURE_MEAN_WARN:.1f}), indicating potential sensor saturation or glare."
                ),
                suggested_human_action="Inspect for clipped highlights and reduce lighting glare.",
            )
        )
    else:
        notices.append(
            InspectionNotice(
                category="Illumination",
                severity="nominal",
                metric_name="Mean Brightness",
                measured_value=round(metrics.mean_brightness, 2),
                threshold_value=thresholds.UNDEREXPOSURE_MEAN_WARN,
                condition="within nominal range [45.0, 215.0]",
                reason=f"Mean intensity ({metrics.mean_brightness:.1f}) is within normal exposure range.",
                suggested_human_action="Illumination is within expected bounds.",
            )
        )

    # 3. Contrast check
    if metrics.rms_contrast < thresholds.LOW_CONTRAST_STD_WARN:
        notices.append(
            InspectionNotice(
                category="Contrast",
                severity="warning",
                metric_name="RMS Contrast",
                measured_value=round(metrics.rms_contrast, 2),
                threshold_value=thresholds.LOW_CONTRAST_STD_WARN,
                condition="measured < threshold",
                reason=(
                    f"RMS contrast ({metrics.rms_contrast:.1f}) is low. Feature differentiation may be impaired."
                ),
                suggested_human_action="Verify dynamic range and adjust lighting angles or surface illumination.",
            )
        )
    else:
        notices.append(
            InspectionNotice(
                category="Contrast",
                severity="nominal",
                metric_name="RMS Contrast",
                measured_value=round(metrics.rms_contrast, 2),
                threshold_value=thresholds.LOW_CONTRAST_STD_WARN,
                condition="measured >= threshold",
                reason=f"RMS contrast ({metrics.rms_contrast:.1f}) is adequate.",
                suggested_human_action="Contrast is sufficient for visual feature inspection.",
            )
        )

    return notices


def analyze_image(
    img_bgr: np.ndarray,
    thresholds: InspectionThresholds = InspectionThresholds(),
) -> InspectionResult:
    """Execute complete OpenCV 5 analysis pipeline on a validated BGR image.
    
    Args:
        img_bgr: Decoded OpenCV 3-channel BGR numpy array.
        thresholds: Configurable thresholds for quality checks.
        
    Returns:
        InspectionResult containing metrics, candidate regions, notices, and maps.
    """
    # 1. Colorspace conversions
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 2. Quality metrics & Edge map
    metrics, edge_map = compute_quality_metrics(
        gray=gray,
        img_bgr=img_bgr,
        canny_t1=thresholds.CANNY_THRESHOLD_1,
        canny_t2=thresholds.CANNY_THRESHOLD_2,
    )

    # 3. Candidate region extraction
    candidates = extract_candidate_regions(edge_map=edge_map, thresholds=thresholds)

    # 4. Transparent rule notices
    notices = evaluate_inspection_rules(metrics=metrics, thresholds=thresholds)

    return InspectionResult(
        image_bgr=img_bgr,
        image_rgb=img_rgb,
        gray_image=gray,
        edge_map=edge_map,
        metrics=metrics,
        candidate_regions=candidates,
        notices=notices,
        opencv_version=str(cv2.__version__),
    )

