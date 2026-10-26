"""VisionGuard AI — Genuine Agentic Vision Engine.

Implements a bounded, auditable, multi-step computer vision decision workflow.
Executes targeted OpenCV operations based on intermediate visual evidence.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from visionguard.analysis import (
    CandidateRegion,
    InspectionNotice,
    InspectionResult,
    QualityMetrics,
    analyze_image,
)
from visionguard.config import InspectionThresholds
from visionguard.visualization import extract_candidate_crop


@dataclass(frozen=True)
class AgentAction:
    """An individual recorded step in the agent's decision trace."""
    step_number: int
    tool_name: str
    reason: str
    input_params: Dict[str, Any]
    observation: Dict[str, Any]
    timestamp_iso: str


@dataclass(frozen=True)
class AgentVerdict:
    """The final auditable conclusion reached by the agent."""
    verdict_type: str  # 'NOMINAL_CLEAR' | 'ANOMALOUS_CANDIDATE_FLAGGED' | 'IMAGE_QUALITY_DEFECT' | 'REQUIRES_HUMAN_INSPECTION'
    confidence: str    # 'HIGH' | 'MODERATE' | 'LOW_REQUIRES_REVIEW'
    summary: str
    recommended_actions: List[str]


@dataclass
class AgentInspectionSession:
    """Encapsulates the complete agentic inspection session."""
    initial_result: InspectionResult
    final_verdict: AgentVerdict
    action_trace: List[AgentAction]
    total_steps: int
    elapsed_time_ms: float
    target_crops_inspected: List[Dict[str, Any]] = field(default_factory=list)


class VisionAgent:
    """Auditable, evidence-driven vision agent for industrial inspection assistance."""

    def __init__(
        self,
        thresholds: InspectionThresholds = InspectionThresholds(),
        max_steps: int = 5,
        timeout_seconds: float = 5.0,
    ):
        self.thresholds = thresholds
        self.max_steps = max_steps
        self.timeout_seconds = timeout_seconds
        self.allowed_tools = {
            "survey_global_image",
            "compute_local_sharpness_grid",
            "analyze_regional_exposure",
            "reinspect_candidate_crop",
            "conclude_inspection",
        }

    def _log_action(
        self,
        step: int,
        tool: str,
        reason: str,
        params: Dict[str, Any],
        obs: Dict[str, Any],
    ) -> AgentAction:
        if tool not in self.allowed_tools:
            raise ValueError(f"Unauthorized tool invocation: {tool}")
        return AgentAction(
            step_number=step,
            tool_name=tool,
            reason=reason,
            input_params=params,
            observation=obs,
            timestamp_iso=datetime.now(timezone.utc).isoformat(),
        )

    def _tool_sharpness_grid(self, gray: np.ndarray, grid_size: int = 4) -> Dict[str, Any]:
        """Compute Laplacian variance across an NxN spatial grid to isolate local vs global blur."""
        h, w = gray.shape[:2]
        step_y = h // grid_size
        step_x = w // grid_size

        variances = []
        for r in range(grid_size):
            for c in range(grid_size):
                patch = gray[r * step_y : (r + 1) * step_y, c * step_x : (c + 1) * step_x]
                if patch.size > 0:
                    var_val = float(cv2.Laplacian(patch, cv2.CV_64F).var())
                    variances.append(round(var_val, 2))

        min_var = float(np.min(variances)) if variances else 0.0
        max_var = float(np.max(variances)) if variances else 0.0
        mean_var = float(np.mean(variances)) if variances else 0.0
        std_var = float(np.std(variances)) if variances else 0.0

        is_uniformly_blurred = (max_var < self.thresholds.BLUR_LAPLACIAN_VAR_WARN)
        has_sharp_patch = (max_var >= self.thresholds.BLUR_LAPLACIAN_VAR_WARN)

        return {
            "grid_dimensions": f"{grid_size}x{grid_size}",
            "patch_variances": variances,
            "min_variance": min_var,
            "max_variance": max_var,
            "mean_variance": round(mean_var, 2),
            "std_variance": round(std_var, 2),
            "is_uniformly_blurred": is_uniformly_blurred,
            "has_localized_sharpness": has_sharp_patch,
            "interpretation": (
                "Uniform blur across all spatial quadrants"
                if is_uniformly_blurred
                else "Non-uniform sharpness; image contains sharp areas despite low global average"
            ),
        }

    def _tool_regional_exposure(self, gray: np.ndarray) -> Dict[str, Any]:
        """Examine 4 spatial quadrants to diagnose illumination asymmetry, glare, or vignetting."""
        h, w = gray.shape[:2]
        mid_y, mid_x = h // 2, w // 2

        quadrants = {
            "top_left": float(np.mean(gray[:mid_y, :mid_x])),
            "top_right": float(np.mean(gray[:mid_y, mid_x:])),
            "bottom_left": float(np.mean(gray[mid_y:, :mid_x])),
            "bottom_right": float(np.mean(gray[mid_y:, mid_x:])),
        }
        q_vals = list(quadrants.values())
        max_delta = float(np.max(q_vals) - np.min(q_vals))

        return {
            "quadrant_means": {k: round(v, 2) for k, v in quadrants.items()},
            "max_quadrant_delta": round(max_delta, 2),
            "asymmetric_lighting_detected": max_delta > 35.0,
            "interpretation": (
                "Substantial asymmetric illumination or directional shadowing detected"
                if max_delta > 35.0
                else "Spatially uniform scene illumination"
            ),
        }

    def _tool_reinspect_crop(
        self,
        image_bgr: np.ndarray,
        candidate: CandidateRegion,
        global_rms: float,
    ) -> Dict[str, Any]:
        """Perform targeted micro-inspection on candidate region crop."""
        x, y, bw, bh = candidate.bounding_box
        h, w = image_bgr.shape[:2]

        crop_bgr = image_bgr[y : min(h, y + bh), x : min(w, x + bw)]
        crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

        crop_rms = float(np.std(crop_gray))
        crop_edges = cv2.Canny(
            crop_gray,
            int(self.thresholds.CANNY_THRESHOLD_1),
            int(self.thresholds.CANNY_THRESHOLD_2),
        )
        edge_density = float(np.count_nonzero(crop_edges) / crop_gray.size) if crop_gray.size > 0 else 0.0
        contrast_ratio = float(crop_rms / global_rms) if global_rms > 0 else 1.0

        is_salient_anomaly = (contrast_ratio > 1.2 and edge_density > 0.005) or (candidate.internal_edge_density > 0.01)

        return {
            "region_id": candidate.region_id,
            "crop_dimensions": f"{bw}x{bh}",
            "local_rms_contrast": round(crop_rms, 2),
            "contrast_ratio_vs_global": round(contrast_ratio, 2),
            "local_edge_density": round(edge_density, 4),
            "is_salient_candidate": is_salient_anomaly,
            "assessment": (
                "Confirmed high-contrast local discontinuity; priority candidate for human review"
                if is_salient_anomaly
                else "Low-salience texture variation; human review recommended to confirm"
            ),
        }

    def inspect(self, image_bgr: np.ndarray) -> AgentInspectionSession:
        """Execute bounded, evidence-driven inspection workflow."""
        start_time = time.perf_counter()
        action_trace: List[AgentAction] = []
        step = 1

        # Step 1: Global Survey
        initial_result = analyze_image(image_bgr, thresholds=self.thresholds)
        action_trace.append(
            self._log_action(
                step=step,
                tool="survey_global_image",
                reason="Execute baseline OpenCV 5 global quality survey and candidate contour extraction.",
                params={"width": image_bgr.shape[1], "height": image_bgr.shape[0]},
                obs={
                    "laplacian_variance": round(initial_result.metrics.laplacian_variance, 2),
                    "mean_brightness": round(initial_result.metrics.mean_brightness, 2),
                    "rms_contrast": round(initial_result.metrics.rms_contrast, 2),
                    "candidate_regions_found": len(initial_result.candidate_regions),
                    "warnings_count": sum(1 for n in initial_result.notices if n.severity == "warning"),
                },
            )
        )
        step += 1

        gray = initial_result.gray_image
        crops_inspected: List[Dict[str, Any]] = []

        # Step 2: Evidence-driven follow-up: Sharpness ambiguity check
        is_blur_warning = any(n.category == "Sharpness" and n.severity == "warning" for n in initial_result.notices)
        if is_blur_warning and step <= self.max_steps:
            sharpness_obs = self._tool_sharpness_grid(gray, grid_size=4)
            action_trace.append(
                self._log_action(
                    step=step,
                    tool="compute_local_sharpness_grid",
                    reason="Global Laplacian variance was below threshold. Check 4x4 spatial grid to determine if blur is uniform or localized.",
                    params={"grid_size": 4},
                    obs=sharpness_obs,
                )
            )
            step += 1

        # Step 3: Evidence-driven follow-up: Illumination asymmetry check
        is_illum_warning = any(n.category == "Illumination" and n.severity == "warning" for n in initial_result.notices)
        if is_illum_warning and step <= self.max_steps:
            exposure_obs = self._tool_regional_exposure(gray)
            action_trace.append(
                self._log_action(
                    step=step,
                    tool="analyze_regional_exposure",
                    reason="Illumination outside nominal bounds. Inspect 4 quadrants to detect asymmetric lighting, glare, or vignetting.",
                    params={"quadrants": 4},
                    obs=exposure_obs,
                )
            )
            step += 1

        # Step 4: Evidence-driven follow-up: Targeted crop reinspection
        if initial_result.candidate_regions and step <= self.max_steps:
            # Reinspect primary candidate region (largest area)
            primary_candidate = initial_result.candidate_regions[0]
            crop_obs = self._tool_reinspect_crop(
                image_bgr=image_bgr,
                candidate=primary_candidate,
                global_rms=initial_result.metrics.rms_contrast,
            )
            crops_inspected.append(crop_obs)
            action_trace.append(
                self._log_action(
                    step=step,
                    tool="reinspect_candidate_crop",
                    reason=f"Candidate region #{primary_candidate.region_id} detected. Reinspect crop at high resolution to evaluate local edge density and contrast delta.",
                    params={"candidate_id": primary_candidate.region_id, "bounding_box": primary_candidate.bounding_box},
                    obs=crop_obs,
                )
            )
            step += 1

        # Step 5: Synthesize and Conclude
        has_warnings = any(n.severity == "warning" for n in initial_result.notices)
        has_candidates = len(initial_result.candidate_regions) > 0
        has_salient_crops = any(c.get("is_salient_candidate") for c in crops_inspected)

        if has_candidates and has_salient_crops:
            v_type = "ANOMALOUS_CANDIDATE_FLAGGED"
            conf = "HIGH"
            summary = (
                f"Identified {len(initial_result.candidate_regions)} candidate region(s) with confirmed local contrast discontinuity. "
                "Targeted crop reinspection supports prioritizing these coordinates for human review."
            )
            recs = [
                "Human inspector to verify candidate crop #{primary_candidate.region_id} against reference tolerance.",
                "Review annotated coordinate report.",
            ]
        elif has_warnings:
            v_type = "IMAGE_QUALITY_DEFECT"
            conf = "HIGH"
            summary = "Image capture quality does not satisfy baseline thresholds (sharpness, exposure, or contrast)."
            recs = [
                "Verify camera focus, lighting stability, or sensor settings.",
                "Recapture image under controlled conditions before final acceptance.",
            ]
        elif has_candidates and not has_salient_crops:
            v_type = "REQUIRES_HUMAN_INSPECTION"
            conf = "MODERATE"
            summary = "Subtle candidate regions detected but local contrast is low. Automated verdict is ambiguous."
            recs = [
                "Manual human review recommended to distinguish surface texture from legitimate defect.",
            ]
        else:
            v_type = "NOMINAL_CLEAR"
            conf = "HIGH"
            summary = "All measured quality heuristics satisfy nominal thresholds. No prominent candidate regions detected."
            recs = ["Capture is suitable for downstream automated pipeline or standard archival."]

        action_trace.append(
            self._log_action(
                step=step,
                tool="conclude_inspection",
                reason="Synthesize global measurements and targeted tool observations into final audit verdict.",
                params={"total_prior_steps": step - 1},
                obs={"verdict": v_type, "confidence": conf, "summary": summary},
            )
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        verdict = AgentVerdict(
            verdict_type=v_type,
            confidence=conf,
            summary=summary,
            recommended_actions=recs,
        )

        return AgentInspectionSession(
            initial_result=initial_result,
            final_verdict=verdict,
            action_trace=action_trace,
            total_steps=len(action_trace),
            elapsed_time_ms=round(elapsed_ms, 2),
            target_crops_inspected=crops_inspected,
        )

    def inspect_video_sequence(
        self,
        sampled_frames: List[Any],
        video_filename: str = "video_input",
    ) -> Tuple[Any, List[AgentInspectionSession]]:
        """Execute VisionAgent inspection across sampled keyframes and generate an aggregated video summary.
        
        Args:
            sampled_frames: List of SampledVideoFrame objects.
            video_filename: Name of the video container.
            
        Returns:
            Tuple of (VideoInspectionSummary, List[AgentInspectionSession]).
        """
        from visionguard.video import VideoFrameInspectionResult, VideoInspectionSummary

        sessions: List[AgentInspectionSession] = []
        frame_details: List[VideoFrameInspectionResult] = []

        anomalies_count = 0
        quality_defects_count = 0
        nominal_count = 0

        for frame in sampled_frames:
            session = self.inspect(frame.image_bgr)
            sessions.append(session)

            v_type = session.final_verdict.verdict_type
            if v_type == "ANOMALOUS_CANDIDATE_FLAGGED":
                anomalies_count += 1
            elif v_type == "IMAGE_QUALITY_DEFECT":
                quality_defects_count += 1
            else:
                nominal_count += 1

            m = session.initial_result.metrics
            frame_details.append(
                VideoFrameInspectionResult(
                    frame_index=frame.frame_index,
                    timestamp_seconds=frame.timestamp_seconds,
                    width=frame.width,
                    height=frame.height,
                    metrics={
                        "laplacian_variance": round(m.laplacian_variance, 2),
                        "mean_brightness": round(m.mean_brightness, 2),
                        "rms_contrast": round(m.rms_contrast, 2),
                        "canny_edge_density": round(m.canny_edge_density, 5),
                    },
                    candidate_regions_count=len(session.initial_result.candidate_regions),
                    warnings_count=sum(1 for n in session.initial_result.notices if n.severity == "warning"),
                    verdict=v_type,
                    confidence=session.final_verdict.confidence,
                )
            )

        total_sampled = len(sampled_frames)
        if anomalies_count > 0:
            overall_verdict = "VIDEO_ANOMALY_CANDIDATE_DETECTED"
            summary_text = (
                f"Candidate anomalies detected in {anomalies_count} of {total_sampled} sampled keyframes. "
                "Human inspector review of flagged frame timestamps is recommended."
            )
        elif quality_defects_count > 0:
            overall_verdict = "VIDEO_QUALITY_DEFECT_DETECTED"
            summary_text = (
                f"Image capture quality defects (blur/exposure) detected in {quality_defects_count} of {total_sampled} "
                "sampled keyframes. Camera focus or lighting stability requires verification."
            )
        else:
            overall_verdict = "VIDEO_NOMINAL_CLEAR"
            summary_text = f"All {total_sampled} sampled keyframes satisfied baseline image quality and feature thresholds."

        summary = VideoInspectionSummary(
            video_filename=video_filename,
            total_frames_sampled=total_sampled,
            frames_with_anomalies=anomalies_count,
            frames_with_quality_defects=quality_defects_count,
            nominal_frames_count=nominal_count,
            overall_video_verdict=overall_verdict,
            summary_text=summary_text,
            frame_details=frame_details,
        )

        return summary, sessions


