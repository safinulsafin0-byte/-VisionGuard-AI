"""Synchronized live video inspection engine for VisionGuard AI.

Connects playback timestamps to OpenCV 5 analysis and VisionAgent multi-step reasoning,
detecting keyframes based on regular sampling cadence and significant visual changes.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import os
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from visionguard.agent import AgentInspectionSession, VisionAgent
from visionguard.analysis import CandidateRegion, InspectionResult, analyze_image
from visionguard.config import (
    HUMAN_REVIEW_DISCLAIMER,
    InspectionThresholds,
    SafetyLimits,
    VideoSafetyLimits,
)
from visionguard.video import VideoProcessingError, validate_video_bytes
from visionguard.visualization import annotate_candidate_regions


@dataclass
class LiveKeyframeResult:
    """An individual analyzed keyframe detected during video playback."""
    frame_index: int
    timestamp_seconds: float
    selection_reason: str  # 'INITIAL_FRAME' | 'REGULAR_INTERVAL' | 'SIGNIFICANT_VISUAL_CHANGE' | 'SEEK_TRIGGER'
    visual_delta: float
    image_bgr: np.ndarray
    annotated_rgb: np.ndarray
    inspection_res: InspectionResult
    agent_session: AgentInspectionSession
    processing_latency_ms: float

    @property
    def resolution(self) -> Tuple[int, int]:
        h, w = self.image_bgr.shape[:2]
        return (w, h)


class SynchronizedVideoInspector:
    """Stateful inspector evaluating video keyframes synchronized to playback timestamps."""

    def __init__(
        self,
        video_source: Any,  # file path (str) or raw bytes (bytes)
        filename: str = "video_input.mp4",
        sample_interval_sec: float = 1.0,
        visual_change_threshold: float = 12.0,
        thresholds: InspectionThresholds = InspectionThresholds(),
        agent: Optional[VisionAgent] = None,
    ):
        self.filename = filename
        self.sample_interval_sec = max(0.2, float(sample_interval_sec))
        self.visual_change_threshold = max(2.0, float(visual_change_threshold))
        self.thresholds = thresholds
        self.agent = agent or VisionAgent(thresholds=thresholds)

        self._temp_path: Optional[str] = None
        if isinstance(video_source, bytes):
            ext = validate_video_bytes(video_source, filename=filename)
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tf:
                tf.write(video_source)
                self._temp_path = tf.name
            self.file_path = self._temp_path
        elif isinstance(video_source, str):
            if not os.path.isfile(video_source):
                raise VideoProcessingError(f"Video file does not exist: {video_source}")
            self.file_path = video_source
        else:
            raise TypeError("video_source must be a file path string or video bytes.")

        self.cap = cv2.VideoCapture(self.file_path)
        if not self.cap.isOpened():
            self.close()
            raise VideoProcessingError("OpenCV failed to open video stream.")

        raw_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = 25.0 if (raw_fps <= 0.0 or np.isnan(raw_fps) or np.isinf(raw_fps)) else float(raw_fps)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration_seconds = (
            float(self.total_frames / self.fps) if (self.total_frames > 0 and self.fps > 0) else 0.0
        )

        self.keyframes: List[LiveKeyframeResult] = []
        self.last_inspected_timestamp: float = -999.0
        self.last_gray_frame: Optional[np.ndarray] = None
        self.skipped_frames_count: int = 0
        self.latencies_ms: List[float] = []

    def reset(self):
        """Reset inspection progress while keeping video source open."""
        self.keyframes = []
        self.last_inspected_timestamp = -999.0
        self.last_gray_frame = None
        self.skipped_frames_count = 0
        self.latencies_ms = []
        if self.cap is not None and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_MSEC, 0.0)

    def update_thresholds(self, thresholds: InspectionThresholds):
        """Update inspection rules dynamically."""
        self.thresholds = thresholds
        if self.agent is not None:
            self.agent.thresholds = thresholds

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Release VideoCapture resources and clean up temporary files."""
        if hasattr(self, "cap") and self.cap is not None:
            self.cap.release()
            self.cap = None

        if self._temp_path and os.path.exists(self._temp_path):
            try:
                os.remove(self._temp_path)
            except OSError:
                pass
            self._temp_path = None

    def extract_frame_at_timestamp(self, timestamp_sec: float) -> Tuple[int, np.ndarray]:
        """Extract exact frame corresponding to playback timestamp."""
        if self.cap is None or not self.cap.isOpened():
            raise VideoProcessingError("VideoCapture handle is closed.")

        target_msec = max(0.0, timestamp_sec * 1000.0)
        self.cap.set(cv2.CAP_PROP_POS_MSEC, target_msec)

        ret, frame_bgr = self.cap.read()
        if not ret or frame_bgr is None:
            # Fallback to frame index calculation
            target_idx = max(0, int(round(timestamp_sec * self.fps)))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
            ret, frame_bgr = self.cap.read()

        if not ret or frame_bgr is None:
            raise VideoProcessingError(f"Failed to read frame at timestamp {timestamp_sec:.2f}s.")

        actual_pos_msec = self.cap.get(cv2.CAP_PROP_POS_MSEC)
        actual_frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        return max(0, actual_frame_idx), frame_bgr

    def evaluate_live_timestamp(
        self,
        playback_time_sec: float,
        force_inspect: bool = False,
    ) -> Optional[LiveKeyframeResult]:
        """Evaluate current playback timestamp and run analysis if keyframe criteria are satisfied.
        
        Selection Criteria:
        1. Initial frame (first timestamp received).
        2. Visual change: Delta intensity shift > visual_change_threshold.
        3. Regular cadence: Elapsed time since last keyframe >= sample_interval_sec.
        4. Forced trigger (e.g. user seeking).
        """
        t0 = time.perf_counter()

        # Check if already processed within a tight 0.15s window (avoid redundant duplicates)
        if not force_inspect and abs(playback_time_sec - self.last_inspected_timestamp) < 0.15:
            self.skipped_frames_count += 1
            return None

        try:
            frame_idx, frame_bgr = self.extract_frame_at_timestamp(playback_time_sec)
        except VideoProcessingError:
            return None

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # Compute visual delta against previous keyframe
        visual_delta = 0.0
        if self.last_gray_frame is not None:
            diff = cv2.absdiff(gray, self.last_gray_frame)
            visual_delta = float(np.mean(diff))

        # Determine selection reason
        time_elapsed = playback_time_sec - self.last_inspected_timestamp
        is_first_frame = len(self.keyframes) == 0
        is_visual_change = (self.last_gray_frame is not None) and (visual_delta >= self.visual_change_threshold)
        is_cadence_due = (time_elapsed >= self.sample_interval_sec)

        if is_first_frame:
            reason = "INITIAL_FRAME"
        elif is_visual_change:
            reason = "SIGNIFICANT_VISUAL_CHANGE"
        elif is_cadence_due or force_inspect:
            reason = "REGULAR_INTERVAL" if not force_inspect else "SEEK_TRIGGER"
        else:
            self.skipped_frames_count += 1
            return None

        # Execute OpenCV pipeline & VisionAgent multi-step reasoning
        agent_session = self.agent.inspect(frame_bgr)
        inspection_res = agent_session.initial_result

        annotated_rgb = annotate_candidate_regions(
            image_rgb=inspection_res.image_rgb,
            candidate_regions=inspection_res.candidate_regions,
        )

        latency_ms = (time.perf_counter() - t0) * 1000.0
        self.latencies_ms.append(latency_ms)

        result = LiveKeyframeResult(
            frame_index=frame_idx,
            timestamp_seconds=round(playback_time_sec, 2),
            selection_reason=reason,
            visual_delta=round(visual_delta, 2),
            image_bgr=frame_bgr,
            annotated_rgb=annotated_rgb,
            inspection_res=inspection_res,
            agent_session=agent_session,
            processing_latency_ms=round(latency_ms, 2),
        )

        self.keyframes.append(result)
        self.last_inspected_timestamp = playback_time_sec
        self.last_gray_frame = gray

        return result

    def run_synchronized_inspection_stream(
        self,
        on_keyframe_callback: Optional[Callable[[LiveKeyframeResult, float], None]] = None,
        max_duration_sec: Optional[float] = None,
        realtime_pacing: bool = False,
    ) -> List[LiveKeyframeResult]:
        """Process video frames sequentially, emitting keyframe events live."""
        duration = min(self.duration_seconds, max_duration_sec) if max_duration_sec else self.duration_seconds
        step_sec = self.sample_interval_sec / 2.0  # Probe at double frequency to catch visual transitions

        curr_t = 0.0
        while curr_t <= duration:
            t_start = time.perf_counter()
            kf = self.evaluate_live_timestamp(curr_t)

            if kf is not None and on_keyframe_callback is not None:
                on_keyframe_callback(kf, curr_t)

            if realtime_pacing:
                elapsed = time.perf_counter() - t_start
                sleep_time = max(0.0, step_sec - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            curr_t += step_sec

        return self.keyframes

    def compile_consolidated_report(self) -> Dict[str, Any]:
        """Compile a comprehensive final inspection report adhering to Schema v1.3.0."""
        now_utc = datetime.now(timezone.utc).isoformat()

        total_sampled = len(self.keyframes)
        anomalies_count = sum(
            1 for kf in self.keyframes if kf.agent_session.final_verdict.verdict_type == "ANOMALOUS_CANDIDATE_FLAGGED"
        )
        quality_defects_count = sum(
            1 for kf in self.keyframes if kf.agent_session.final_verdict.verdict_type == "IMAGE_QUALITY_DEFECT"
        )
        nominal_count = sum(
            1 for kf in self.keyframes if kf.agent_session.final_verdict.verdict_type == "NOMINAL_CLEAR"
        )

        coverage_pct = (
            round((self.last_inspected_timestamp / self.duration_seconds) * 100.0, 1)
            if self.duration_seconds > 0
            else 100.0
        )

        if anomalies_count > 0:
            final_verdict = "VIDEO_ANOMALY_CANDIDATE_DETECTED"
            summary_text = (
                f"Detected candidate anomalies in {anomalies_count} of {total_sampled} analyzed keyframes. "
                "Immediate human review recommended for flagged coordinates."
            )
        elif quality_defects_count > 0:
            final_verdict = "VIDEO_QUALITY_DEFECT_DETECTED"
            summary_text = (
                f"Identified capture quality defects (blur/exposure) in {quality_defects_count} of {total_sampled} "
                "analyzed keyframes. Camera focus and illumination stability require adjustment."
            )
        else:
            final_verdict = "VIDEO_NOMINAL_CLEAR"
            summary_text = f"All {total_sampled} analyzed keyframes satisfied baseline image quality and feature thresholds."

        keyframes_data = [
            {
                "frame_index": kf.frame_index,
                "timestamp_seconds": kf.timestamp_seconds,
                "selection_reason": kf.selection_reason,
                "visual_delta": kf.visual_delta,
                "metrics": {
                    "laplacian_variance": round(kf.inspection_res.metrics.laplacian_variance, 2),
                    "mean_brightness": round(kf.inspection_res.metrics.mean_brightness, 2),
                    "rms_contrast": round(kf.inspection_res.metrics.rms_contrast, 2),
                    "canny_edge_density": round(kf.inspection_res.metrics.canny_edge_density, 5),
                },
                "candidate_regions_count": len(kf.inspection_res.candidate_regions),
                "verdict": kf.agent_session.final_verdict.verdict_type,
                "confidence": kf.agent_session.final_verdict.confidence,
                "processing_latency_ms": kf.processing_latency_ms,
            }
            for kf in self.keyframes
        ]

        mean_latency = float(np.mean(self.latencies_ms)) if self.latencies_ms else 0.0

        report = {
            "schema_version": "1.3.0",
            "system": {
                "application": "VisionGuard AI — Synchronized Live Video Inspector",
                "opencv_runtime_version": str(cv2.__version__),
                "inspection_timestamp_utc": now_utc,
            },
            "asset": {
                "filename": self.filename,
                "modality": "synchronized_video_playback",
                "duration_seconds": round(self.duration_seconds, 2),
                "fps": round(self.fps, 2),
                "total_frames_in_container": self.total_frames,
                "resolution": f"{self.width}x{self.height}",
            },
            "inspection_configuration": {
                "sample_interval_sec": self.sample_interval_sec,
                "visual_change_threshold": self.visual_change_threshold,
                "sync_mode": "playback_timeline_synchronized",
            },
            "coverage_statistics": {
                "total_keyframes_detected": total_sampled,
                "skipped_identical_frames": self.skipped_frames_count,
                "last_analyzed_timestamp_sec": round(self.last_inspected_timestamp, 2),
                "timeline_coverage_percent": coverage_pct,
                "mean_keyframe_latency_ms": round(mean_latency, 2),
            },
            "executive_summary": {
                "overall_verdict": final_verdict,
                "summary_text": summary_text,
                "keyframes_with_anomalies": anomalies_count,
                "keyframes_with_quality_defects": quality_defects_count,
                "nominal_keyframes_count": nominal_count,
            },
            "detected_keyframes": keyframes_data,
            "disclaimer_and_limitations": (
                "Synchronized inspection summarizes analyzed keyframes selected via temporal cadence "
                "and visual change triggers. Inter-frame motion tracking or temporal optical flow is not claimed. "
                + HUMAN_REVIEW_DISCLAIMER
            ),
        }
        return report
