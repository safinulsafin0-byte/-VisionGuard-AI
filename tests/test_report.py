"""Tests for report generation and JSON serialization."""

import json
from visionguard.agent import VisionAgent
from visionguard.analysis import analyze_image
from visionguard.report import (
    generate_inspection_report,
    normalize_schema_1_3_report,
    serialize_report_json,
    validate_schema_1_3_report,
)
from visionguard.server import compile_live_video_report, reset_live_session
from visionguard.synthetic_data import (
    generate_sharp_checkerboard,
    generate_surface_with_artifact,
)
from visionguard.video_stream import SynchronizedVideoInspector


def test_report_generation_structure():
    img = generate_sharp_checkerboard(128, 128)
    res = analyze_image(img)
    report = generate_inspection_report(res, filename="test_sample.png")

    assert report["schema_version"] == "1.1.0"
    assert "system" in report
    assert "asset" in report
    assert "executive_summary" in report
    assert "measured_metrics" in report
    assert "rule_evaluations" in report
    assert "candidate_regions_for_review" in report
    assert "limitations_and_disclaimer" in report

    assert report["asset"]["width"] == 128
    assert report["asset"]["height"] == 128
    assert report["asset"]["filename"] == "test_sample.png"

    json_str = serialize_report_json(report)
    parsed = json.loads(json_str)
    assert parsed["schema_version"] == "1.1.0"


def test_report_generation_with_agent_session():
    img, _ = generate_surface_with_artifact(256, 256)
    agent = VisionAgent()
    session = agent.inspect(img)

    report = generate_inspection_report(
        session.initial_result,
        filename="artifact.png",
        agent_session=session,
    )

    assert "agent_orchestration" in report
    assert report["agent_orchestration"]["verdict_type"] == "ANOMALOUS_CANDIDATE_FLAGGED"
    assert len(report["agent_orchestration"]["action_timeline"]) >= 2
    assert report["agent_orchestration"]["execution_latency_ms"] > 0

    json_str = serialize_report_json(report)
    parsed = json.loads(json_str)
    assert "agent_orchestration" in parsed


def test_normalize_schema_1_3_missing_coverage_statistics():
    """Regression test: Report lacking coverage_statistics must be normalized from playback_telemetry."""
    raw_report = {
        "schema_version": "1.3.0",
        "asset": {"filename": "test_video.mp4", "duration_seconds": 10.0, "fps": 30.0},
        "playback_telemetry": {
            "frames_analyzed": 5,
            "frames_sampled_from_player": 12,
            "frames_dropped_for_responsiveness": 2,
            "mean_inference_latency_ms": 14.2,
            "timeline_coverage_percent": 85.5,
            "last_playback_timestamp_sec": 8.5,
        },
        "executive_summary": {
            "overall_verdict": "VIDEO_NOMINAL_CLEAR",
            "summary_text": "All frames nominal.",
            "keyframes_with_anomalies": 0,
            "keyframes_with_quality_defects": 0,
            "nominal_keyframes_count": 5,
        },
        "detected_keyframes": [],
    }

    assert "coverage_statistics" not in raw_report

    normalized = normalize_schema_1_3_report(raw_report)
    assert "coverage_statistics" in normalized
    cov = normalized["coverage_statistics"]
    assert cov["total_keyframes_detected"] == 5
    assert cov["skipped_identical_frames"] == 2
    assert cov["timeline_coverage_percent"] == 85.5
    assert cov["last_analyzed_timestamp_sec"] == 8.5
    assert cov["mean_keyframe_latency_ms"] == 14.2

    # Verify validate_schema_1_3_report accepts it
    is_valid, errors = validate_schema_1_3_report(normalized)
    assert is_valid, f"Validation failed with errors: {errors}"


def test_normalize_schema_1_3_empty_input():
    """Test defensive normalization on empty dictionary or None input."""
    normalized = normalize_schema_1_3_report({})
    assert normalized["schema_version"] == "1.3.0"
    assert "system" in normalized
    assert "asset" in normalized
    assert "coverage_statistics" in normalized
    assert "executive_summary" in normalized
    assert "detected_keyframes" in normalized
    assert "disclaimer_and_limitations" in normalized

    cov = normalized["coverage_statistics"]
    assert cov["total_keyframes_detected"] == 0
    assert cov["timeline_coverage_percent"] == 0.0

    ex = normalized["executive_summary"]
    assert ex["overall_verdict"] == "VIDEO_NOMINAL_CLEAR"
    assert ex["keyframes_with_anomalies"] == 0

    is_valid, errors = validate_schema_1_3_report(normalized)
    assert is_valid, f"Validation failed: {errors}"


def test_normalize_schema_1_3_derives_from_detected_keyframes():
    """Verify that when telemetry blocks are absent, metrics are inferred from detected_keyframes."""
    raw_report = {
        "schema_version": "1.3.0",
        "asset": {"filename": "run.mp4", "duration_seconds": 10.0},
        "detected_keyframes": [
            {
                "frame_index": 0,
                "timestamp_seconds": 0.0,
                "verdict": "NOMINAL_CLEAR",
            },
            {
                "frame_index": 30,
                "timestamp_seconds": 5.0,
                "verdict": "ANOMALOUS_CANDIDATE_FLAGGED",
            },
            {
                "frame_index": 60,
                "timestamp_seconds": 10.0,
                "verdict": "IMAGE_QUALITY_DEFECT",
            },
        ],
    }

    normalized = normalize_schema_1_3_report(raw_report)
    cov = normalized["coverage_statistics"]
    assert cov["total_keyframes_detected"] == 3
    assert cov["last_analyzed_timestamp_sec"] == 10.0
    assert cov["timeline_coverage_percent"] == 100.0

    ex = normalized["executive_summary"]
    assert ex["keyframes_with_anomalies"] == 1
    assert ex["keyframes_with_quality_defects"] == 1
    assert ex["nominal_keyframes_count"] == 1
    assert ex["overall_verdict"] == "VIDEO_ANOMALY_CANDIDATE_DETECTED"

    is_valid, errors = validate_schema_1_3_report(normalized)
    assert is_valid, f"Validation failed: {errors}"


def test_validate_schema_1_3_report_invalid_inputs():
    """Verify that validate_schema_1_3_report identifies corrupt or missing structures."""
    is_valid, errors = validate_schema_1_3_report("not a dict")
    assert not is_valid
    assert "must be a dictionary" in errors[0]

    is_valid, errors = validate_schema_1_3_report({"schema_version": "1.3.0"})
    assert not is_valid
    assert any("coverage_statistics" in err for err in errors)
    assert any("executive_summary" in err for err in errors)


def test_compile_live_video_report_conforms_to_schema_1_3_0():
    """Verify that compile_live_video_report produces a valid Schema v1.3.0 report."""
    reset_live_session()
    report = compile_live_video_report(
        filename="live_run.mp4",
        video_duration_sec=12.0,
        fps=30.0,
        total_frames_sampled=25,
        dropped_frames_count=2,
    )
    is_valid, errors = validate_schema_1_3_report(report)
    assert is_valid, f"compile_live_video_report produced invalid report: {errors}"
    assert report["coverage_statistics"]["total_keyframes_detected"] == 0
    assert report["coverage_statistics"]["skipped_identical_frames"] == 2
    assert report["coverage_statistics"]["timeline_coverage_percent"] == 0.0


def test_app_ui_rendering_dict_access_safety():
    """Verify the exact dictionary access pattern in app.py never raises KeyError."""
    # Test on raw un-normalized report with missing coverage_statistics
    buggy_raw_report = {
        "schema_version": "1.3.0",
        "playback_telemetry": {
            "frames_analyzed": 3,
            "frames_dropped_for_responsiveness": 1,
            "timeline_coverage_percent": 90.0,
            "mean_inference_latency_ms": 11.5,
        },
        "executive_summary": {
            "overall_verdict": "VIDEO_NOMINAL_CLEAR",
            "summary_text": "Nominal clear.",
            "keyframes_with_anomalies": 0,
            "keyframes_with_quality_defects": 0,
        },
    }

    # Normalize as app.py does
    report_dict = normalize_schema_1_3_report(buggy_raw_report)

    # Replicate app.py access
    exec_sum = report_dict.get("executive_summary", {})
    cov_stat = report_dict.get("coverage_statistics", {})

    overall_verdict = exec_sum.get("overall_verdict", "VIDEO_NOMINAL_CLEAR")
    summary_text = exec_sum.get("summary_text", "Inspection completed.")
    timeline_cov = cov_stat.get("timeline_coverage_percent", 0.0)
    total_kf = cov_stat.get("total_keyframes_detected", 0)
    skipped_frames = cov_stat.get("skipped_identical_frames", 0)
    anomalies_count = exec_sum.get("keyframes_with_anomalies", 0)
    defects_count = exec_sum.get("keyframes_with_quality_defects", 0)
    mean_latency = cov_stat.get("mean_keyframe_latency_ms", 0.0)

    assert overall_verdict == "VIDEO_NOMINAL_CLEAR"
    assert summary_text == "Nominal clear."
    assert timeline_cov == 90.0
    assert total_kf == 3
    assert skipped_frames == 1
    assert anomalies_count == 0
    assert defects_count == 0
    assert mean_latency == 11.5

