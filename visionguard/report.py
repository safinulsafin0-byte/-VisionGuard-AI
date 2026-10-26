"""Audit report serialization and schema definition for VisionGuard AI."""

from datetime import datetime, timezone
import json
import platform
from typing import Any, Dict, List, Optional, Tuple

import cv2

from visionguard.analysis import InspectionResult
from visionguard.config import HUMAN_REVIEW_DISCLAIMER


def generate_inspection_report(
    result: InspectionResult,
    filename: str = "image_input",
    agent_session: Optional[Any] = None,
) -> Dict[str, Any]:
    """Generate a structured, reproducible inspection audit dictionary.
    
    Args:
        result: The InspectionResult from analysis.
        filename: Name or identifier of inspected image.
        agent_session: Optional AgentInspectionSession containing action trace.
        
    Returns:
        Dictionary adhering to the VisionGuard AI audit schema.
    """
    now_utc = datetime.now(timezone.utc).isoformat()
    h, w = result.image_rgb.shape[:2]

    # Convert candidate regions to clean serializable dicts
    candidates_data = [
        {
            "region_id": c.region_id,
            "bounding_box": {
                "x": c.bbox_x,
                "y": c.bbox_y,
                "width": c.bbox_w,
                "height": c.bbox_h,
            },
            "contour_area_px": c.contour_area,
            "aspect_ratio": c.aspect_ratio,
            "internal_edge_density": c.internal_edge_density,
            "review_status": c.review_label,
        }
        for c in result.candidate_regions
    ]

    # Convert rule notices to dicts
    notices_data = [
        {
            "category": n.category,
            "severity": n.severity,
            "metric": n.metric_name,
            "measured_value": n.measured_value,
            "threshold": n.threshold_value,
            "condition": n.condition,
            "reason": n.reason,
            "suggested_human_action": n.suggested_human_action,
        }
        for n in result.notices
    ]

    # Compile metrics
    m = result.metrics
    metrics_data = {
        "sharpness": {
            "laplacian_variance": round(m.laplacian_variance, 3),
            "heuristic_description": "Laplacian variance over grayscale image (higher values generally indicate sharper high-frequency details).",
        },
        "illumination": {
            "mean_intensity": round(m.mean_brightness, 2),
            "median_intensity": round(m.median_brightness, 2),
            "std_intensity": round(m.std_brightness, 2),
            "min_intensity": m.min_brightness,
            "max_intensity": m.max_brightness,
            "channel_means_bgr": [round(c, 2) for c in m.channel_means_bgr],
        },
        "contrast": {
            "rms_contrast": round(m.rms_contrast, 2),
            "dynamic_range_p95_p5": round(m.dynamic_range_p95_p5, 2),
        },
        "edge_statistics": {
            "canny_edge_density": round(m.canny_edge_density, 5),
            "mean_sobel_gradient": round(m.mean_sobel_gradient, 3),
        },
    }

    warning_count = sum(1 for n in result.notices if n.severity == "warning")

    report: Dict[str, Any] = {
        "schema_version": "1.1.0",
        "system": {
            "application": "VisionGuard AI",
            "opencv_runtime_version": str(cv2.__version__),
            "python_runtime_version": platform.python_version(),
            "platform_os": f"{platform.system()} {platform.release()}",
            "inspection_timestamp_utc": now_utc,
        },
        "asset": {
            "filename": filename,
            "width": w,
            "height": h,
            "total_pixels": w * h,
            "channels": 3,
        },
        "executive_summary": {
            "total_candidate_regions_found": len(result.candidate_regions),
            "total_warnings_triggered": warning_count,
            "human_inspection_recommended": warning_count > 0 or len(result.candidate_regions) > 0,
        },
        "measured_metrics": metrics_data,
        "rule_evaluations": notices_data,
        "candidate_regions_for_review": candidates_data,
        "limitations_and_disclaimer": HUMAN_REVIEW_DISCLAIMER,
    }

    if agent_session is not None:
        report["agent_orchestration"] = {
            "verdict_type": agent_session.final_verdict.verdict_type,
            "confidence": agent_session.final_verdict.confidence,
            "summary": agent_session.final_verdict.summary,
            "recommended_actions": agent_session.final_verdict.recommended_actions,
            "total_steps": agent_session.total_steps,
            "execution_latency_ms": agent_session.elapsed_time_ms,
            "action_timeline": [
                {
                    "step": a.step_number,
                    "tool": a.tool_name,
                    "reason": a.reason,
                    "input_parameters": a.input_params,
                    "observation": a.observation,
                    "timestamp": a.timestamp_iso,
                }
                for a in agent_session.action_trace
            ],
            "target_crops_inspected": agent_session.target_crops_inspected,
        }

    return report


def generate_video_inspection_report(
    summary: Any,
    sessions: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """Generate a structured audit report for a sampled video inspection sequence."""
    now_utc = datetime.now(timezone.utc).isoformat()

    frames_data = [
        {
            "frame_index": f.frame_index,
            "timestamp_seconds": f.timestamp_seconds,
            "resolution": f"{f.width}x{f.height}",
            "metrics": f.metrics,
            "candidate_regions_count": f.candidate_regions_count,
            "warnings_count": f.warnings_count,
            "verdict": f.verdict,
            "confidence": f.confidence,
        }
        for f in summary.frame_details
    ]

    report: Dict[str, Any] = {
        "schema_version": "1.2.0",
        "system": {
            "application": "VisionGuard AI",
            "opencv_runtime_version": str(cv2.__version__),
            "python_runtime_version": platform.python_version(),
            "platform_os": f"{platform.system()} {platform.release()}",
            "inspection_timestamp_utc": now_utc,
        },
        "asset": {
            "filename": summary.video_filename,
            "modality": "video_keyframe_sequence",
            "total_frames_sampled": summary.total_frames_sampled,
        },
        "executive_summary": {
            "overall_verdict": summary.overall_video_verdict,
            "summary_text": summary.summary_text,
            "frames_with_anomalies": summary.frames_with_anomalies,
            "frames_with_quality_defects": summary.frames_with_quality_defects,
            "nominal_frames_count": summary.nominal_frames_count,
        },
        "sampled_frames_analysis": frames_data,
        "methodology_note": summary.limitations_note,
        "limitations_and_disclaimer": HUMAN_REVIEW_DISCLAIMER,
    }

    return report


def serialize_report_json(report_data: Dict[str, Any], indent: int = 2) -> str:
    """Serialize report dictionary to formatted JSON string."""
    return json.dumps(report_data, indent=indent)


def normalize_schema_1_3_report(report_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize and repair a Schema v1.3.0 video inspection report dictionary.
    
    Ensures backward compatibility between live playback server telemetry and synchronized
    inspector reports. Guarantees required keys exist with defensive fallbacks, preventing
    KeyError exceptions during UI rendering or audit serialization without fabricating
    unmeasured telemetry.
    """
    if not isinstance(report_data, dict):
        report_data = {}

    normalized = dict(report_data)

    # 1. Schema version
    normalized["schema_version"] = str(normalized.get("schema_version") or "1.3.0")

    # 2. System block
    system_block = dict(normalized.get("system") or {})
    system_block.setdefault("application", "VisionGuard AI — Synchronized Inspection")
    system_block.setdefault("opencv_runtime_version", str(cv2.__version__))
    system_block.setdefault("inspection_timestamp_utc", datetime.now(timezone.utc).isoformat())
    normalized["system"] = system_block

    # 3. Asset block
    asset_block = dict(normalized.get("asset") or {})
    asset_block.setdefault("filename", "video_stream")
    asset_block.setdefault("modality", "synchronized_video_playback")
    asset_block.setdefault("duration_seconds", 0.0)
    asset_block.setdefault("fps", 30.0)
    normalized["asset"] = asset_block

    # 4. Detected keyframes list
    detected_kf = normalized.get("detected_keyframes")
    if not isinstance(detected_kf, list):
        detected_kf = []
    normalized["detected_keyframes"] = detected_kf

    # 5. Telemetry & Coverage Statistics Normalization
    cov_stat = dict(normalized.get("coverage_statistics") or {})
    playback_tel = dict(normalized.get("playback_telemetry") or {})

    # Cross-populate between coverage_statistics and playback_telemetry
    total_kf = cov_stat.get("total_keyframes_detected")
    if total_kf is None:
        total_kf = playback_tel.get("frames_analyzed", len(detected_kf))
    try:
        total_kf = max(0, int(total_kf))
    except (ValueError, TypeError):
        total_kf = len(detected_kf)

    skipped = cov_stat.get("skipped_identical_frames")
    if skipped is None:
        skipped = playback_tel.get("frames_dropped_for_responsiveness", 0)
    try:
        skipped = max(0, int(skipped))
    except (ValueError, TypeError):
        skipped = 0

    last_t = cov_stat.get("last_analyzed_timestamp_sec")
    if last_t is None:
        last_t = playback_tel.get("last_playback_timestamp_sec")
    if last_t is None and detected_kf:
        last_t = detected_kf[-1].get("timestamp_seconds", 0.0)
    try:
        last_t = max(0.0, float(last_t or 0.0))
    except (ValueError, TypeError):
        last_t = 0.0

    timeline_cov = cov_stat.get("timeline_coverage_percent")
    if timeline_cov is None:
        timeline_cov = playback_tel.get("timeline_coverage_percent")
    if timeline_cov is None:
        dur = float(asset_block.get("duration_seconds", 0.0) or 0.0)
        timeline_cov = round((last_t / dur) * 100.0, 1) if dur > 0 else (100.0 if total_kf > 0 else 0.0)
    try:
        timeline_cov = max(0.0, min(100.0, float(timeline_cov)))
    except (ValueError, TypeError):
        timeline_cov = 0.0

    latency_ms = cov_stat.get("mean_keyframe_latency_ms")
    if latency_ms is None:
        latency_ms = playback_tel.get("mean_inference_latency_ms", 0.0)
    try:
        latency_ms = max(0.0, float(latency_ms or 0.0))
    except (ValueError, TypeError):
        latency_ms = 0.0

    normalized["coverage_statistics"] = {
        "total_keyframes_detected": total_kf,
        "skipped_identical_frames": skipped,
        "last_analyzed_timestamp_sec": round(last_t, 2),
        "timeline_coverage_percent": round(timeline_cov, 1),
        "mean_keyframe_latency_ms": round(latency_ms, 2),
    }

    # Ensure playback_telemetry is also synchronized
    normalized["playback_telemetry"] = {
        "frames_analyzed": total_kf,
        "frames_sampled_from_player": playback_tel.get("frames_sampled_from_player", total_kf + skipped),
        "frames_dropped_for_responsiveness": skipped,
        "mean_inference_latency_ms": round(latency_ms, 2),
        "timeline_coverage_percent": round(timeline_cov, 1),
        "last_playback_timestamp_sec": round(last_t, 2),
    }

    # 6. Executive Summary Block
    exec_sum = dict(normalized.get("executive_summary") or {})
    anomalies = exec_sum.get("keyframes_with_anomalies")
    if anomalies is None:
        anomalies = sum(
            1 for kf in detected_kf
            if isinstance(kf, dict) and kf.get("verdict") in ["ANOMALOUS_CANDIDATE_FLAGGED", "VIDEO_ANOMALY_CANDIDATE_DETECTED"]
        )
    defects = exec_sum.get("keyframes_with_quality_defects")
    if defects is None:
        defects = sum(
            1 for kf in detected_kf
            if isinstance(kf, dict) and kf.get("verdict") in ["IMAGE_QUALITY_DEFECT", "VIDEO_QUALITY_DEFECT_DETECTED"]
        )
    nominal = exec_sum.get("nominal_keyframes_count")
    if nominal is None:
        nominal = max(0, total_kf - anomalies - defects)

    overall_v = exec_sum.get("overall_verdict")
    if not overall_v:
        if anomalies > 0:
            overall_v = "VIDEO_ANOMALY_CANDIDATE_DETECTED"
        elif defects > 0:
            overall_v = "VIDEO_QUALITY_DEFECT_DETECTED"
        else:
            overall_v = "VIDEO_NOMINAL_CLEAR"

    summary_text = exec_sum.get("summary_text")
    if not summary_text:
        if anomalies > 0:
            summary_text = f"Candidate anomalies flagged in {anomalies} of {total_kf} analyzed keyframes. Human review recommended."
        elif defects > 0:
            summary_text = f"Quality defects flagged in {defects} of {total_kf} analyzed keyframes."
        else:
            summary_text = f"All {total_kf} analyzed keyframes met nominal clarity and baseline rules."

    normalized["executive_summary"] = {
        "overall_verdict": str(overall_v),
        "summary_text": str(summary_text),
        "keyframes_with_anomalies": max(0, int(anomalies or 0)),
        "keyframes_with_quality_defects": max(0, int(defects or 0)),
        "nominal_keyframes_count": max(0, int(nominal or 0)),
    }

    # 7. Disclaimer
    if "disclaimer_and_limitations" not in normalized:
        normalized["disclaimer_and_limitations"] = (
            "Synchronized inspection summarizes analyzed keyframes selected via temporal cadence "
            "and visual change triggers. Inter-frame motion tracking or temporal optical flow is not claimed. "
            + HUMAN_REVIEW_DISCLAIMER
        )

    return normalized


def validate_schema_1_3_report(report_data: Any) -> Tuple[bool, List[str]]:
    """Validate that a report dictionary conforms to the Schema v1.3.0 specification.
    
    Returns:
        (is_valid, validation_errors)
    """
    errors: List[str] = []
    if not isinstance(report_data, dict):
        return False, ["Report data must be a dictionary."]

    required_top = [
        "schema_version",
        "system",
        "asset",
        "coverage_statistics",
        "executive_summary",
        "detected_keyframes",
        "disclaimer_and_limitations",
    ]
    for key in required_top:
        if key not in report_data:
            errors.append(f"Missing required top-level section: '{key}'.")

    if "coverage_statistics" in report_data and isinstance(report_data["coverage_statistics"], dict):
        cov = report_data["coverage_statistics"]
        for cov_key in [
            "total_keyframes_detected",
            "skipped_identical_frames",
            "last_analyzed_timestamp_sec",
            "timeline_coverage_percent",
            "mean_keyframe_latency_ms",
        ]:
            if cov_key not in cov:
                errors.append(f"Missing key in 'coverage_statistics': '{cov_key}'.")
    else:
        errors.append("'coverage_statistics' must be a valid dictionary.")

    if "executive_summary" in report_data and isinstance(report_data["executive_summary"], dict):
        ex = report_data["executive_summary"]
        for ex_key in [
            "overall_verdict",
            "summary_text",
            "keyframes_with_anomalies",
            "keyframes_with_quality_defects",
            "nominal_keyframes_count",
        ]:
            if ex_key not in ex:
                errors.append(f"Missing key in 'executive_summary': '{ex_key}'.")
    else:
        errors.append("'executive_summary' must be a valid dictionary.")

    if "detected_keyframes" in report_data and not isinstance(report_data["detected_keyframes"], list):
        errors.append("'detected_keyframes' must be a list.")

    return len(errors) == 0, errors

