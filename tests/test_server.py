"""Unit tests for the real-time live video inspection server and endpoints."""

import base64
import json
import numpy as np
import pytest

from visionguard.config import InspectionThresholds
from visionguard.server import (
    compile_live_video_report,
    get_shared_agent,
    process_live_frame_payload,
    reset_live_session,
)
from visionguard.synthetic_data import (
    generate_sharp_checkerboard,
    generate_surface_with_artifact,
)
from visionguard.visualization import encode_image_bytes


def test_process_live_frame_nominal():
    reset_live_session()
    nominal_bgr = generate_sharp_checkerboard(256, 256)
    jpeg_bytes = encode_image_bytes(nominal_bgr, is_rgb=False, fmt=".jpg")

    result = process_live_frame_payload(
        frame_bytes=jpeg_bytes,
        timestamp_sec=0.0,
        thresholds=InspectionThresholds(),
    )

    assert result["status"] == "success"
    assert result["timestamp_seconds"] == 0.0
    assert result["selection_reason"] == "INITIAL_FRAME"
    assert result["verdict"] == "NOMINAL_CLEAR"
    assert result["confidence"] == "HIGH"
    assert "metrics" in result
    assert result["metrics"]["laplacian_variance"] > 100.0
    assert "annotated_frame" in result
    assert result["annotated_frame"].startswith("data:image/jpeg;base64,")


def test_process_live_frame_visual_delta_and_cadence():
    reset_live_session()
    nominal_bgr = generate_sharp_checkerboard(256, 256)
    jpeg_1 = encode_image_bytes(nominal_bgr, is_rgb=False, fmt=".jpg")

    res1 = process_live_frame_payload(jpeg_1, timestamp_sec=0.0)
    assert res1["selection_reason"] == "INITIAL_FRAME"

    # Second frame with massive visual delta (surface artifact)
    artifact_bgr, _ = generate_surface_with_artifact(256, 256)
    jpeg_2 = encode_image_bytes(artifact_bgr, is_rgb=False, fmt=".jpg")

    res2 = process_live_frame_payload(jpeg_2, timestamp_sec=0.4, visual_change_threshold=10.0)
    assert res2["visual_delta"] > 10.0
    assert res2["selection_reason"] == "SIGNIFICANT_VISUAL_CHANGE"
    assert res2["candidate_count"] >= 1
    assert res2["verdict"] == "ANOMALOUS_CANDIDATE_FLAGGED"


def test_compile_live_video_report_schema_1_3_0():
    reset_live_session()
    nominal_bgr = generate_sharp_checkerboard(256, 256)
    jpeg_bytes = encode_image_bytes(nominal_bgr, is_rgb=False, fmt=".jpg")

    process_live_frame_payload(jpeg_bytes, timestamp_sec=0.0)
    process_live_frame_payload(jpeg_bytes, timestamp_sec=1.0)

    report = compile_live_video_report(
        filename="inspection_run.mp4",
        video_duration_sec=5.0,
        fps=30.0,
        total_frames_sampled=10,
        dropped_frames_count=1,
    )

    assert report["schema_version"] == "1.3.0"
    assert report["asset"]["filename"] == "inspection_run.mp4"
    assert report["asset"]["duration_seconds"] == 5.0
    assert report["playback_telemetry"]["frames_analyzed"] == 2
    assert report["playback_telemetry"]["frames_dropped_for_responsiveness"] == 1
    assert "coverage_statistics" in report
    assert report["coverage_statistics"]["total_keyframes_detected"] == 2
    assert report["coverage_statistics"]["skipped_identical_frames"] == 1
    assert report["coverage_statistics"]["timeline_coverage_percent"] > 0
    assert "executive_summary" in report
    assert report["executive_summary"]["overall_verdict"] == "VIDEO_NOMINAL_CLEAR"
    assert len(report["detected_keyframes"]) == 2


def test_request_handler_endpoints_mock():
    import io
    from unittest.mock import MagicMock
    from visionguard.server import VisionGuardRequestHandler

    # Test GET /health
    handler = VisionGuardRequestHandler.__new__(VisionGuardRequestHandler)
    handler.path = "/health"
    handler.headers = {}
    handler.rfile = io.BytesIO()
    handler.wfile = io.BytesIO()
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()

    handler.do_GET()
    handler.send_response.assert_called_with(200)
    body = handler.wfile.getvalue().decode("utf-8")
    assert "healthy" in body
    assert "opencv_version" in body

    # Test POST /api/analyze_frame with valid JPEG payload
    handler_post = VisionGuardRequestHandler.__new__(VisionGuardRequestHandler)
    handler_post.send_response = MagicMock()
    handler_post.send_header = MagicMock()
    handler_post.end_headers = MagicMock()

    nominal_bgr = generate_sharp_checkerboard(128, 128)
    jpeg_bytes = encode_image_bytes(nominal_bgr, is_rgb=False, fmt=".jpg")
    b64_str = f"data:image/jpeg;base64,{base64.b64encode(jpeg_bytes).decode('utf-8')}"
    req_body = json.dumps({"timestamp": 1.5, "frame_data": b64_str}).encode("utf-8")

    handler_post.path = "/api/analyze_frame"
    handler_post.headers = {"Content-Length": str(len(req_body))}
    handler_post.rfile = io.BytesIO(req_body)
    handler_post.wfile = io.BytesIO()

    handler_post.do_POST()
    handler_post.send_response.assert_called_with(200)
    post_res = json.loads(handler_post.wfile.getvalue().decode("utf-8"))
    assert post_res["status"] == "success"
    assert "metrics" in post_res

