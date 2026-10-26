"""Streamlit custom bidirectional component for synchronized video playback and inspection."""

import base64
import os
from typing import Any, Dict, Optional
import streamlit.components.v1 as components

_component_dir = os.path.dirname(os.path.abspath(__file__))
_sync_video_component = components.declare_component("sync_video_player", path=_component_dir)


def render_sync_video_player(
    video_bytes: bytes,
    mime_type: str = "video/mp4",
    api_port: int = 8505,
    filename: str = "video.mp4",
    sync_interval_sec: float = 0.5,
    visual_change_thresh: float = 12.0,
    thresholds: Optional[Dict[str, float]] = None,
    key: Optional[str] = "sync_player",
    default: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Render the synchronized HTML5 video player component.
    
    Args:
        video_bytes: Raw binary bytes of uploaded video.
        mime_type: MIME container format (e.g. 'video/mp4', 'video/x-msvideo').
        api_port: Port of the background VisionGuard frame analysis HTTP server.
        filename: Name of the uploaded media file.
        sync_interval_sec: Cadence in seconds between emitted frame evaluations.
        visual_change_thresh: Intensity shift required for visual delta anomaly triggers.
        thresholds: Dictionary of active detection thresholds.
        key: Streamlit component unique identifier.
        default: Default return dictionary before first user interaction.
        
    Returns:
        Dictionary emitted from player (e.g. REPORT_COMPILED with final Schema v1.3.0 report).
    """
    b64_data = base64.b64encode(video_bytes).decode("utf-8")
    data_url = f"data:{mime_type};base64,{b64_data}"

    return _sync_video_component(
        video_src=data_url,
        mime_type=mime_type,
        api_port=api_port,
        filename=filename,
        sync_interval_sec=sync_interval_sec,
        visual_change_thresh=visual_change_thresh,
        thresholds=thresholds or {},
        key=key,
        default=default,
    )

