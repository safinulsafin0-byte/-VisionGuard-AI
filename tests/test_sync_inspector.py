"""Unit and integration tests for SynchronizedVideoInspector and live video analysis."""

import os
import tempfile
import cv2
import numpy as np
import pytest

from visionguard.config import InspectionThresholds
from visionguard.video import VideoProcessingError
from visionguard.video_stream import LiveKeyframeResult, SynchronizedVideoInspector
from tests.test_video import create_synthetic_video_bytes


def create_dynamic_synthetic_video(
    num_frames: int = 30,
    fps: int = 10,
    width: int = 128,
    height: int = 128,
) -> bytes:
    """Generate a video that has an initial scene, a sudden visual change at frame 15, and static content."""
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    assert writer.isOpened()

    for i in range(num_frames):
        if i < 15:
            # Steady dark pattern
            frame = np.full((height, width, 3), 40, dtype=np.uint8)
        else:
            # Sudden high brightness transition (visual change)
            frame = np.full((height, width, 3), 200, dtype=np.uint8)
            cv2.rectangle(frame, (10, 10), (50, 50), (0, 0, 255), -1)
        writer.write(frame)

    writer.release()
    with open(path, "rb") as f:
        data = f.read()
    os.remove(path)
    return data


def test_sync_inspector_initialization():
    video_bytes = create_dynamic_synthetic_video(num_frames=10, fps=10)
    with SynchronizedVideoInspector(video_bytes, filename="dyn.mp4", sample_interval_sec=0.5) as inspector:
        assert inspector.fps > 0
        assert inspector.width == 128
        assert inspector.height == 128
        assert inspector.duration_seconds > 0
        assert len(inspector.keyframes) == 0


def test_sync_inspector_initial_and_cadence_triggers():
    video_bytes = create_dynamic_synthetic_video(num_frames=30, fps=10)
    with SynchronizedVideoInspector(video_bytes, sample_interval_sec=0.5, visual_change_threshold=50.0) as inspector:
        # First frame should be INITIAL_FRAME
        kf1 = inspector.evaluate_live_timestamp(0.0)
        assert kf1 is not None
        assert kf1.selection_reason == "INITIAL_FRAME"
        assert len(inspector.keyframes) == 1

        # Evaluating immediately at 0.05s should be skipped (too fast / redundant)
        kf_skip = inspector.evaluate_live_timestamp(0.05)
        assert kf_skip is None
        assert inspector.skipped_frames_count >= 1

        # Evaluating after sample_interval_sec (e.g. 0.6s) should trigger REGULAR_INTERVAL
        kf2 = inspector.evaluate_live_timestamp(0.6)
        assert kf2 is not None
        assert kf2.selection_reason == "REGULAR_INTERVAL"
        assert len(inspector.keyframes) == 2


def test_sync_inspector_significant_visual_change():
    video_bytes = create_dynamic_synthetic_video(num_frames=30, fps=10)  # Change happens at frame 15 (t=1.5s)
    with SynchronizedVideoInspector(video_bytes, sample_interval_sec=2.0, visual_change_threshold=20.0) as inspector:
        kf1 = inspector.evaluate_live_timestamp(0.0)
        assert kf1 is not None

        # At t=1.5s, cadence is NOT due (sample_interval_sec=2.0), but visual delta > 20
        kf_change = inspector.evaluate_live_timestamp(1.5)
        assert kf_change is not None
        assert kf_change.selection_reason == "SIGNIFICANT_VISUAL_CHANGE"
        assert kf_change.visual_delta > 20.0


def test_sync_inspector_stream_and_report_schema_1_3_0():
    video_bytes = create_dynamic_synthetic_video(num_frames=20, fps=10)
    with SynchronizedVideoInspector(video_bytes, sample_interval_sec=0.5) as inspector:
        keyframes = inspector.run_synchronized_inspection_stream()
        assert len(keyframes) >= 2

        report = inspector.compile_consolidated_report()
        assert report["schema_version"] == "1.3.0"
        assert "system" in report
        assert "asset" in report
        assert "inspection_configuration" in report
        assert "coverage_statistics" in report
        assert "executive_summary" in report
        assert "detected_keyframes" in report
        assert "disclaimer_and_limitations" in report

        exec_sum = report["executive_summary"]
        assert exec_sum["overall_verdict"] in [
            "VIDEO_NOMINAL_CLEAR",
            "VIDEO_ANOMALY_CANDIDATE_DETECTED",
            "VIDEO_QUALITY_DEFECT_DETECTED",
        ]
        assert len(report["detected_keyframes"]) == len(keyframes)


def test_sync_inspector_reset():
    video_bytes = create_dynamic_synthetic_video(num_frames=10, fps=10)
    with SynchronizedVideoInspector(video_bytes) as inspector:
        inspector.evaluate_live_timestamp(0.0)
        assert len(inspector.keyframes) == 1
        inspector.reset()
        assert len(inspector.keyframes) == 0
        assert inspector.last_inspected_timestamp == -999.0


def test_sync_inspector_cleanup_on_close():
    video_bytes = create_dynamic_synthetic_video(num_frames=10, fps=10)
    inspector = SynchronizedVideoInspector(video_bytes)
    temp_path = inspector._temp_path
    assert temp_path is not None
    assert os.path.exists(temp_path)

    inspector.close()
    assert inspector.cap is None
    assert not os.path.exists(temp_path)


def test_sync_inspector_invalid_source():
    with pytest.raises(VideoProcessingError):
        SynchronizedVideoInspector(b"corrupted_header_data_not_video")

    with pytest.raises(VideoProcessingError):
        SynchronizedVideoInspector("non_existent_file_path_1234.mp4")

    with pytest.raises(TypeError):
        SynchronizedVideoInspector(12345)  # Invalid type

