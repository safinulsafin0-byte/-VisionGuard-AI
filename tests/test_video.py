"""Comprehensive automated test suite for video ingestion, frame sampling, and sequence inspection."""

import json
import os
import tempfile
from typing import Tuple

import cv2
import numpy as np
import pytest

from visionguard.agent import VisionAgent
from visionguard.config import VideoSafetyLimits
from visionguard.report import generate_video_inspection_report, serialize_report_json
from visionguard.video import (
    SampledVideoFrame,
    VideoProcessingError,
    detect_video_container_format,
    sample_video_frames_from_bytes,
    sample_video_frames_from_file,
    validate_video_bytes,
)


def create_synthetic_video_bytes(
    codec_fourcc: str = "mp4v",
    extension: str = ".mp4",
    num_frames: int = 12,
    fps: int = 10,
    width: int = 128,
    height: int = 128,
) -> Tuple[bytes, str]:
    """Create a temporary video file, encode synthetic frames, and return raw bytes."""
    fd, path = tempfile.mkstemp(suffix=extension)
    os.close(fd)

    fourcc = cv2.VideoWriter_fourcc(*codec_fourcc)
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    assert writer.isOpened(), f"VideoWriter failed to open for codec {codec_fourcc} ({extension})"

    for i in range(num_frames):
        # Create patterned frames with distinct intensities
        frame = np.full((height, width, 3), (i * 20) % 255, dtype=np.uint8)
        # Draw a small geometric mark to provide edge structure
        cv2.rectangle(frame, (20, 20), (60, 60), (255, 255, 255), -1)
        writer.write(frame)

    writer.release()

    with open(path, "rb") as f:
        data = f.read()

    os.remove(path)
    return data, path


def test_valid_mp4_video_sampling():
    mp4_bytes, _ = create_synthetic_video_bytes(codec_fourcc="mp4v", extension=".mp4")
    frames = sample_video_frames_from_bytes(mp4_bytes, filename="test_clip.mp4", max_frames_to_sample=4)

    assert len(frames) > 0
    assert len(frames) <= 4
    for f in frames:
        assert isinstance(f, SampledVideoFrame)
        assert f.width == 128
        assert f.height == 128
        assert f.image_bgr.shape == (128, 128, 3)
        assert f.timestamp_seconds >= 0.0


def test_valid_avi_video_sampling():
    avi_bytes, _ = create_synthetic_video_bytes(codec_fourcc="MJPG", extension=".avi")
    frames = sample_video_frames_from_bytes(avi_bytes, filename="test_clip.avi", max_frames_to_sample=3)

    assert len(frames) > 0
    assert len(frames) <= 3
    for f in frames:
        assert f.width == 128
        assert f.height == 128


def test_empty_video_bytes_rejected():
    with pytest.raises(VideoProcessingError, match="empty video file"):
        sample_video_frames_from_bytes(b"", filename="empty.mp4")


def test_corrupted_video_bytes_rejected():
    # File with MP4 header signature but corrupted payload
    fake_header = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00" + b"garbage corrupted bytes" * 10
    with pytest.raises(VideoProcessingError):
        sample_video_frames_from_bytes(fake_header, filename="corrupt.mp4")


def test_unsupported_format_rejected():
    random_bytes = b"This is plain text and definitely not a valid video stream at all."
    with pytest.raises(VideoProcessingError, match="Unsupported or unrecognized video format"):
        sample_video_frames_from_bytes(random_bytes, filename="unknown.txt")


def test_video_oversized_payload_rejected(monkeypatch):
    mp4_bytes, _ = create_synthetic_video_bytes(codec_fourcc="mp4v", extension=".mp4", num_frames=3)
    # Mock limits to be smaller than the generated clip
    monkeypatch.setattr(VideoSafetyLimits, "MAX_VIDEO_FILE_BYTES", 100)
    with pytest.raises(VideoProcessingError, match="exceeds maximum allowed limit"):
        validate_video_bytes(mp4_bytes, filename="large.mp4")


def test_resource_cleanup_temporary_file():
    """Verify that temporary files are deleted from disk even after processing."""
    mp4_bytes, _ = create_synthetic_video_bytes(codec_fourcc="mp4v", extension=".mp4", num_frames=5)

    # Track temp directory files before and after
    temp_dir = tempfile.gettempdir()
    before_files = set(os.listdir(temp_dir))

    frames = sample_video_frames_from_bytes(mp4_bytes, filename="cleanup_test.mp4", max_frames_to_sample=2)
    assert len(frames) > 0

    after_files = set(os.listdir(temp_dir))
    # No new leftover temp video files
    new_files = [f for f in after_files - before_files if f.endswith(".mp4")]
    assert len(new_files) == 0, f"Temporary file leaked: {new_files}"


def test_agent_video_sequence_inspection():
    mp4_bytes, _ = create_synthetic_video_bytes(codec_fourcc="mp4v", extension=".mp4", num_frames=8)
    frames = sample_video_frames_from_bytes(mp4_bytes, filename="seq_test.mp4", max_frames_to_sample=3)

    agent = VisionAgent()
    summary, sessions = agent.inspect_video_sequence(frames, video_filename="seq_test.mp4")

    assert summary.total_frames_sampled == len(frames)
    assert len(sessions) == len(frames)
    assert summary.overall_video_verdict in (
        "VIDEO_NOMINAL_CLEAR",
        "VIDEO_ANOMALY_CANDIDATE_DETECTED",
        "VIDEO_QUALITY_DEFECT_DETECTED",
    )
    assert "no inter-frame motion" in summary.limitations_note.lower()

    # Test report serialization
    report = generate_video_inspection_report(summary, sessions)
    assert report["schema_version"] == "1.2.0"
    assert report["asset"]["modality"] == "video_keyframe_sequence"
    assert len(report["sampled_frames_analysis"]) == len(frames)

    json_str = serialize_report_json(report)
    parsed = json.loads(json_str)
    assert parsed["schema_version"] == "1.2.0"


def test_video_mime_type_resolution():
    from visionguard.video import get_video_mime_type
    assert get_video_mime_type("video.mp4") == "video/mp4"
    assert get_video_mime_type("sample.avi") == "video/x-msvideo"
    assert get_video_mime_type("clip.mov") == "video/quicktime"
    assert get_video_mime_type("stream.webm") == "video/webm"
    assert get_video_mime_type("unknown") == "video/mp4"


def test_annotated_video_overlay_generation():
    from visionguard.video import generate_annotated_video_bytes
    mp4_bytes, _ = create_synthetic_video_bytes(codec_fourcc="mp4v", extension=".mp4", num_frames=6)
    annotated = generate_annotated_video_bytes(mp4_bytes, filename="annotated_test.mp4", max_frames_to_process=6)

    # Annotated video should either succeed and return valid bytes, or return None gracefully
    if annotated is not None:
        assert isinstance(annotated, bytes)
        assert len(annotated) > 0
        assert len(annotated) >= 1000

    # Corrupt payload should return None without crashing
    bad_res = generate_annotated_video_bytes(b"invalid corrupt bytes", filename="bad.mp4")
    assert bad_res is None

