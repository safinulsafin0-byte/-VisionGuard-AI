"""Video frame extraction, validation, and sequence sampling for VisionGuard AI."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from visionguard.config import HUMAN_REVIEW_DISCLAIMER, SafetyLimits, VideoSafetyLimits


class VideoProcessingError(ValueError):
    """Raised when a video file violates limits, has unsupported codecs, or fails decoding."""
    pass


@dataclass(frozen=True)
class SampledVideoFrame:
    """A single frame extracted from an inspected video sequence."""
    frame_index: int
    timestamp_seconds: float
    image_bgr: np.ndarray
    width: int
    height: int


@dataclass
class VideoFrameInspectionResult:
    """Individual inspection results for a single sampled video frame."""
    frame_index: int
    timestamp_seconds: float
    width: int
    height: int
    metrics: Dict[str, Any]
    candidate_regions_count: int
    warnings_count: int
    verdict: str
    confidence: str


@dataclass
class VideoInspectionSummary:
    """Aggregated inspection summary across sampled keyframes of a video file.
    
    IMPORTANT: This represents independent frame-level observations.
    Temporal object tracking or optical flow is not performed.
    """
    video_filename: str
    total_frames_sampled: int
    frames_with_anomalies: int
    frames_with_quality_defects: int
    nominal_frames_count: int
    overall_video_verdict: str
    summary_text: str
    frame_details: List[VideoFrameInspectionResult] = field(default_factory=list)
    limitations_note: str = (
        "Video inspection summarizes independent sampled keyframes. "
        "Findings indicate frame-level image quality and candidate regions; "
        "no inter-frame motion or temporal tracking is asserted."
    )


def detect_video_container_format(data: bytes, filename_hint: Optional[str] = None) -> Optional[str]:
    """Inspect leading bytes to detect common video container signatures."""
    if len(data) < 16:
        return None

    # MP4 / QuickTime: contains 'ftyp' in first 32 bytes
    if b"ftyp" in data[:32]:
        return ".mp4"

    # AVI: starts with 'RIFF' and bytes 8..12 are 'AVI '
    if data.startswith(b"RIFF") and len(data) >= 12 and data[8:12] == b"AVI ":
        return ".avi"

    # QuickTime MOV: starts with or contains 'moov' or 'wide'
    if b"moov" in data[:128] or b"wide" in data[:32]:
        return ".mov"

    # Fallback to trusted filename extension if recognizable
    if filename_hint:
        ext = os.path.splitext(filename_hint)[1].lower()
        if ext in VideoSafetyLimits.ALLOWED_VIDEO_EXTENSIONS:
            return ext

    return None


def validate_video_bytes(data: bytes, filename: Optional[str] = None) -> str:
    """Verify size limits and container signature for uploaded video bytes.
    
    Returns:
        Verified file extension suffix (e.g. '.mp4', '.avi').
        
    Raises:
        VideoProcessingError: If the payload is empty, oversized, or unreadable.
    """
    byte_size = len(data)
    if byte_size == 0:
        raise VideoProcessingError("Received an empty video file (0 bytes).")

    if byte_size > VideoSafetyLimits.MAX_VIDEO_FILE_BYTES:
        max_mb = VideoSafetyLimits.MAX_VIDEO_FILE_BYTES / (1024 * 1024)
        curr_mb = byte_size / (1024 * 1024)
        raise VideoProcessingError(
            f"Video file size ({curr_mb:.2f} MB) exceeds maximum allowed limit of {max_mb:.0f} MB."
        )

    detected_ext = detect_video_container_format(data, filename_hint=filename)
    if not detected_ext:
        name_hint = f" for '{filename}'" if filename else ""
        raise VideoProcessingError(
            f"Unsupported or unrecognized video format{name_hint}. "
            f"Supported containers: MP4, AVI, and MOV."
        )

    return detected_ext


def sample_video_frames_from_file(
    file_path: str,
    max_frames_to_sample: int = VideoSafetyLimits.DEFAULT_SAMPLE_COUNT,
) -> List[SampledVideoFrame]:
    """Sample representative keyframes from a video file using OpenCV VideoCapture.
    
    Guarantees resource release and handles variable framerate/metadata safely.
    
    Args:
        file_path: Local filesystem path to video file.
        max_frames_to_sample: Maximum number of frames to return.
        
    Returns:
        List of SampledVideoFrame objects.
        
    Raises:
        VideoProcessingError: If the file cannot be opened, has zero readable frames,
                             or has invalid dimensions.
    """
    if not os.path.isfile(file_path):
        raise VideoProcessingError(f"Video file does not exist: {file_path}")

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise VideoProcessingError(
            "OpenCV VideoCapture failed to open the video container. "
            "The codec may be unsupported or the container may be truncated."
        )

    sampled_frames: List[SampledVideoFrame] = []

    try:
        raw_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = 25.0 if (raw_fps <= 0.0 or np.isnan(raw_fps) or np.isinf(raw_fps)) else float(raw_fps)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        max_sample = max(1, min(max_frames_to_sample, VideoSafetyLimits.MAX_FRAMES_TO_SAMPLE))

        # Calculate stride for sampling
        if total_frames > 0:
            step = max(1, total_frames // max_sample)
            target_indices = set(range(0, total_frames, step))
        else:
            # Fallback when container has no duration header: pick every 25th frame
            step = 25
            target_indices = None

        frame_idx = 0
        while len(sampled_frames) < max_sample:
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                break

            should_sample = (frame_idx in target_indices) if target_indices is not None else (frame_idx % step == 0)

            if should_sample:
                h, w = frame_bgr.shape[:2]
                if h <= 0 or w <= 0:
                    raise VideoProcessingError(f"Decoded invalid frame dimensions ({w}x{h}).")

                if w > SafetyLimits.MAX_WIDTH or h > SafetyLimits.MAX_HEIGHT:
                    raise VideoProcessingError(
                        f"Video frame dimensions ({w}x{h}) exceed safety limit "
                        f"({SafetyLimits.MAX_WIDTH}x{SafetyLimits.MAX_HEIGHT})."
                    )

                timestamp_sec = float(frame_idx / fps)
                sampled_frames.append(
                    SampledVideoFrame(
                        frame_index=frame_idx,
                        timestamp_seconds=round(timestamp_sec, 2),
                        image_bgr=frame_bgr,
                        width=w,
                        height=h,
                    )
                )

            frame_idx += 1

    except Exception as exc:
        if isinstance(exc, VideoProcessingError):
            raise
        raise VideoProcessingError(f"Unexpected error while decoding video frames: {str(exc)}") from exc
    finally:
        cap.release()

    if not sampled_frames:
        raise VideoProcessingError(
            "Video stream contained no readable image frames. "
            "The file may be corrupted, truncated, or encoded with an unsupported codec."
        )

    return sampled_frames


def sample_video_frames_from_bytes(
    video_bytes: bytes,
    filename: Optional[str] = None,
    max_frames_to_sample: int = VideoSafetyLimits.DEFAULT_SAMPLE_COUNT,
) -> List[SampledVideoFrame]:
    """Write video bytes to a secure temporary file and sample keyframes safely.
    
    Ensures safe temporary file cleanup and file handle closure on Windows.
    """
    ext = validate_video_bytes(video_bytes, filename=filename)

    temp_path: Optional[str] = None
    try:
        # Create temporary file, write bytes, and explicitly close handle before OpenCV opens it
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tf:
            tf.write(video_bytes)
            temp_path = tf.name

        return sample_video_frames_from_file(
            file_path=temp_path,
            max_frames_to_sample=max_frames_to_sample,
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def get_video_mime_type(filename_or_ext: str) -> str:
    """Determine appropriate MIME type for native HTML5 video playback."""
    ext = os.path.splitext(filename_or_ext)[1].lower()
    if ext == ".mp4":
        return "video/mp4"
    elif ext == ".avi":
        return "video/x-msvideo"
    elif ext in (".mov", ".qt"):
        return "video/quicktime"
    elif ext == ".webm":
        return "video/webm"
    return "video/mp4"


def create_h264_video_writer(
    output_path: str,
    fps: float,
    width: int,
    height: int,
) -> Tuple[Optional[cv2.VideoWriter], str]:
    """Create a VideoWriter targeting web-playable H.264 MP4 with optimal backend negotiation.
    
    Returns:
        (writer, selected_fourcc_str) or (None, "") if no suitable encoder opens.
    """
    import sys

    # Candidate backend and fourcc pairings ordered by web browser compatibility
    candidates = []

    # On Windows, Microsoft Media Foundation (MSMF) natively encodes hardware-accelerated H.264
    if sys.platform == "win32" and hasattr(cv2, "CAP_MSMF"):
        candidates.append((cv2.CAP_MSMF, "avc1"))
        candidates.append((cv2.CAP_MSMF, "H264"))

    # Universal avc1 / H264 via default backend (FFMPEG or system GStreamer)
    candidates.append((None, "avc1"))
    candidates.append((None, "H264"))
    candidates.append((None, "X264"))

    # On Windows MSMF, mp4v also writes using the system's H.264/MP4 MFT
    if sys.platform == "win32" and hasattr(cv2, "CAP_MSMF"):
        candidates.append((cv2.CAP_MSMF, "mp4v"))

    # Final fallback to standard mp4v
    candidates.append((None, "mp4v"))

    for backend, fourcc_str in candidates:
        try:
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            if backend is not None:
                writer = cv2.VideoWriter(output_path, backend, fourcc, fps, (width, height))
            else:
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if writer is not None and writer.isOpened():
                return writer, fourcc_str
        except Exception:
            continue

    return None, ""


def generate_annotated_video_bytes(
    video_bytes: bytes,
    filename: Optional[str] = None,
    thresholds: Any = None,
    max_frames_to_process: int = 150,
) -> Optional[bytes]:
    """Render OpenCV candidate region annotations across video frames and encode to playable H.264 MP4.
    
    Args:
        video_bytes: Raw bytes of the uploaded video.
        filename: Original video filename hint.
        thresholds: InspectionThresholds for candidate extraction.
        max_frames_to_process: Cap on total frames to annotate for responsive performance.
        
    Returns:
        Bytes of the annotated H.264 MP4 video, or None if encoding fails or codecs are unavailable.
    """
    from visionguard.analysis import analyze_image
    from visionguard.config import InspectionThresholds
    from visionguard.visualization import annotate_candidate_regions

    if thresholds is None:
        thresholds = InspectionThresholds()

    in_temp_path: Optional[str] = None
    out_temp_path: Optional[str] = None

    try:
        ext = validate_video_bytes(video_bytes, filename=filename)

        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tf_in:
            tf_in.write(video_bytes)
            in_temp_path = tf_in.name

        cap = cv2.VideoCapture(in_temp_path)
        if not cap.isOpened():
            return None

        raw_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = 25.0 if (raw_fps <= 0.0 or np.isnan(raw_fps) or np.isinf(raw_fps)) else float(raw_fps)
        raw_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        raw_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if raw_width <= 0 or raw_height <= 0:
            cap.release()
            return None

        # Ensure even dimensions for H.264 macroblock alignment
        width = raw_width if raw_width % 2 == 0 else raw_width - 1
        height = raw_height if raw_height % 2 == 0 else raw_height - 1

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf_out:
            out_temp_path = tf_out.name

        writer, used_codec = create_h264_video_writer(out_temp_path, fps, width, height)
        if writer is None or not writer.isOpened():
            cap.release()
            return None

        frames_written = 0
        while frames_written < max_frames_to_process:
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                break

            # Resize if dimensions needed macroblock alignment
            if frame_bgr.shape[1] != width or frame_bgr.shape[0] != height:
                frame_bgr = cv2.resize(frame_bgr, (width, height))

            # Analyze frame with OpenCV pipeline
            res = analyze_image(frame_bgr, thresholds=thresholds)
            annotated_rgb = annotate_candidate_regions(res.image_rgb, res.candidate_regions)
            annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

            # Add timestamp badge and candidate count overlay
            time_sec = frames_written / fps
            cv2.putText(
                annotated_bgr,
                f"t={time_sec:.2f}s | {len(res.candidate_regions)} candidate(s) flagged",
                (12, max(24, int(height * 0.06))),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            writer.write(annotated_bgr)
            frames_written += 1

        cap.release()
        writer.release()

        if frames_written == 0 or not os.path.exists(out_temp_path) or os.path.getsize(out_temp_path) == 0:
            return None

        with open(out_temp_path, "rb") as f_out:
            annotated_bytes = f_out.read()

        return annotated_bytes

    except Exception:
        # Graceful fallback: return None on any codec/write error without failing the app
        return None
    finally:
        for p in (in_temp_path, out_temp_path):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

