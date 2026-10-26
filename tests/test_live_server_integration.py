"""Integration test verifying end-to-end HTTP communication with the VisionGuard live server."""

import base64
import json
import time
import urllib.request
import pytest

from visionguard.server import start_background_server, stop_background_server
from visionguard.synthetic_data import (
    generate_sharp_checkerboard,
    generate_surface_with_artifact,
)
from visionguard.visualization import encode_image_bytes


def test_live_server_http_communication():
    try:
        port = start_background_server(start_port=8520)
    except Exception as e:
        pytest.skip(f"Socket binding restricted in this sandbox: {e}")

    # 1. Health probe
    health_url = f"http://127.0.0.1:{port}/health"
    try:
        health_data = json.loads(urllib.request.urlopen(health_url, timeout=3).read().decode())
    except (urllib.error.URLError, OSError) as e:
        pytest.skip(f"Socket connection restricted in this sandbox: {e}")
    assert health_data["status"] == "healthy"

    # 2. Reset session
    reset_req = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/reset",
        data=b"{}",
        headers={"Content-Type": "application/json"},
    )
    reset_data = json.loads(urllib.request.urlopen(reset_req, timeout=3).read().decode())
    assert reset_data["status"] == "reset_success"

    # 3. Post nominal frame at t=0.0s
    img1 = generate_sharp_checkerboard(256, 256)
    b64_1 = "data:image/jpeg;base64," + base64.b64encode(encode_image_bytes(img1, False, ".jpg")).decode("utf-8")
    payload1 = json.dumps({"timestamp": 0.0, "frame_data": b64_1}).encode("utf-8")
    req1 = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/analyze_frame",
        data=payload1,
        headers={"Content-Type": "application/json"},
    )
    res1 = json.loads(urllib.request.urlopen(req1, timeout=3).read().decode())
    assert res1["status"] == "success"
    assert res1["selection_reason"] == "INITIAL_FRAME"
    assert res1["verdict"] == "NOMINAL_CLEAR"

    # 4. Post artifact frame at t=1.0s
    img2, _ = generate_surface_with_artifact(256, 256)
    b64_2 = "data:image/jpeg;base64," + base64.b64encode(encode_image_bytes(img2, False, ".jpg")).decode("utf-8")
    payload2 = json.dumps({"timestamp": 1.0, "frame_data": b64_2}).encode("utf-8")
    req2 = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/analyze_frame",
        data=payload2,
        headers={"Content-Type": "application/json"},
    )
    res2 = json.loads(urllib.request.urlopen(req2, timeout=3).read().decode())
    assert res2["status"] == "success"
    assert res2["verdict"] == "ANOMALOUS_CANDIDATE_FLAGGED"
    assert res2["candidate_count"] >= 1

    # 5. Compile live report
    report_payload = json.dumps({
        "filename": "live_test.mp4",
        "duration": 2.5,
        "fps": 30.0,
        "total_sampled": 2,
        "dropped_frames": 0,
    }).encode("utf-8")
    req_rep = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/compile_report",
        data=report_payload,
        headers={"Content-Type": "application/json"},
    )
    rep = json.loads(urllib.request.urlopen(req_rep, timeout=3).read().decode())
    assert rep["schema_version"] == "1.3.0"
    assert rep["executive_summary"]["overall_verdict"] == "VIDEO_ANOMALY_CANDIDATE_DETECTED"
    assert len(rep["detected_keyframes"]) == 2

    stop_background_server()
