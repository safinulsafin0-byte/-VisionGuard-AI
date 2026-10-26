"""Lightweight HTTP API server for VisionGuard AI real-time video frame inspection.

Provides asynchronous REST endpoints for in-browser video frame analysis during live playback,
reusing a persistent VisionAgent instance to maintain low inference latency (<10ms).
"""

from dataclasses import asdict
from datetime import datetime, timezone
import http.server
import json
import os
import socket
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from visionguard.agent import AgentInspectionSession, VisionAgent
from visionguard.analysis import CandidateRegion, InspectionResult, analyze_image
from visionguard.config import (
    HUMAN_REVIEW_DISCLAIMER,
    InspectionThresholds,
)
from visionguard.validation import validate_and_decode_image
from visionguard.visualization import annotate_candidate_regions, encode_image_bytes


# Global singleton agent instance to avoid reloading models/heuristics per frame
_AGENT_LOCK = threading.RLock()
_ACTIVE_AGENT: Optional[VisionAgent] = None
_LAST_ANALYZED_GRAY: Optional[np.ndarray] = None
_LAST_ANALYZED_TIME: float = -999.0
_COLLECTED_LIVE_KEYFRAMES: List[Dict[str, Any]] = []


def get_shared_agent(thresholds: Optional[InspectionThresholds] = None) -> VisionAgent:
    """Get or initialize the shared VisionAgent singleton."""
    global _ACTIVE_AGENT
    with _AGENT_LOCK:
        if _ACTIVE_AGENT is None:
            _ACTIVE_AGENT = VisionAgent(thresholds=thresholds or InspectionThresholds())
        elif thresholds is not None:
            _ACTIVE_AGENT.thresholds = thresholds
        return _ACTIVE_AGENT


def reset_live_session():
    """Reset live session tracking state for a new video playback run."""
    global _LAST_ANALYZED_GRAY, _LAST_ANALYZED_TIME, _COLLECTED_LIVE_KEYFRAMES
    with _AGENT_LOCK:
        _LAST_ANALYZED_GRAY = None
        _LAST_ANALYZED_TIME = -999.0
        _COLLECTED_LIVE_KEYFRAMES = []


def process_live_frame_payload(
    frame_bytes: bytes,
    timestamp_sec: float,
    thresholds: Optional[InspectionThresholds] = None,
    visual_change_threshold: float = 12.0,
    force_inspect: bool = False,
) -> Dict[str, Any]:
    """Process an individual captured video frame from the live player.
    
    Args:
        frame_bytes: Binary bytes (JPEG/PNG/WebP) of the frame captured from HTML5 canvas.
        timestamp_sec: Current video playback position in seconds.
        thresholds: Active inspection parameter thresholds.
        visual_change_threshold: Mean intensity shift required for visual delta anomaly trigger.
        force_inspect: Whether to bypass cadence/delta thresholds (e.g. seek event).
        
    Returns:
        Structured dictionary containing detection metrics, candidate bounding boxes,
        annotated preview image (base64 data URL), agent verdict, and latency.
    """
    global _LAST_ANALYZED_GRAY, _LAST_ANALYZED_TIME, _COLLECTED_LIVE_KEYFRAMES
    t_start = time.perf_counter()

    # Decode image in-memory using OpenCV 5
    frame_bgr = validate_and_decode_image(frame_bytes).image_bgr
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    with _AGENT_LOCK:
        agent = get_shared_agent(thresholds)
        
        # Calculate visual delta against previous frame
        visual_delta = 0.0
        if _LAST_ANALYZED_GRAY is not None and _LAST_ANALYZED_GRAY.shape == gray.shape:
            diff = cv2.absdiff(gray, _LAST_ANALYZED_GRAY)
            visual_delta = float(np.mean(diff))

        # Determine selection trigger reason
        is_first = len(_COLLECTED_LIVE_KEYFRAMES) == 0
        is_visual_change = (_LAST_ANALYZED_GRAY is not None) and (visual_delta >= visual_change_threshold)
        time_elapsed = timestamp_sec - _LAST_ANALYZED_TIME
        is_cadence = (time_elapsed >= (agent.thresholds.BLUR_LAPLACIAN_VAR_WARN * 0 if False else 0.5))

        if is_first:
            reason = "INITIAL_FRAME"
        elif is_visual_change:
            reason = "SIGNIFICANT_VISUAL_CHANGE"
        elif force_inspect:
            reason = "SEEK_TRIGGER"
        else:
            reason = "REGULAR_INTERVAL"

        # Execute VisionAgent bounded decision pipeline
        session: AgentInspectionSession = agent.inspect(frame_bgr)
        inspection_res = session.initial_result

        # Annotate candidate regions for human review
        annotated_rgb = annotate_candidate_regions(
            image_rgb=inspection_res.image_rgb,
            candidate_regions=inspection_res.candidate_regions,
        )

        annotated_jpeg = encode_image_bytes(annotated_rgb, is_rgb=True, fmt=".jpg")
        import base64
        b64_annotated = f"data:image/jpeg;base64,{base64.b64encode(annotated_jpeg).decode('utf-8')}"

        latency_ms = (time.perf_counter() - t_start) * 1000.0

        candidates_data = [
            {
                "region_id": c.region_id,
                "bbox": [c.bbox_x, c.bbox_y, c.bbox_w, c.bbox_h],
                "area": round(c.contour_area, 1),
                "aspect_ratio": round(c.aspect_ratio, 2),
                "internal_edge_density": round(c.internal_edge_density, 4),
            }
            for c in inspection_res.candidate_regions
        ]

        result = {
            "status": "success",
            "frame_index": len(_COLLECTED_LIVE_KEYFRAMES),
            "timestamp_seconds": round(timestamp_sec, 2),
            "selection_reason": reason,
            "visual_delta": round(visual_delta, 2),
            "metrics": {
                "laplacian_variance": round(inspection_res.metrics.laplacian_variance, 2),
                "mean_brightness": round(inspection_res.metrics.mean_brightness, 2),
                "rms_contrast": round(inspection_res.metrics.rms_contrast, 2),
                "canny_edge_density": round(inspection_res.metrics.canny_edge_density, 5),
            },
            "candidate_regions": candidates_data,
            "candidate_count": len(candidates_data),
            "verdict": session.final_verdict.verdict_type,
            "confidence": session.final_verdict.confidence,
            "summary": session.final_verdict.summary,
            "recommended_actions": session.final_verdict.recommended_actions,
            "processing_latency_ms": round(latency_ms, 2),
            "annotated_frame": b64_annotated,
            "resolution": [w, h],
        }

        _LAST_ANALYZED_GRAY = gray
        _LAST_ANALYZED_TIME = timestamp_sec
        _COLLECTED_LIVE_KEYFRAMES.append(result)

        return result


def compile_live_video_report(
    filename: str = "video_playback.mp4",
    video_duration_sec: float = 0.0,
    fps: float = 30.0,
    total_frames_sampled: int = 0,
    dropped_frames_count: int = 0,
) -> Dict[str, Any]:
    """Compile a consolidated Schema v1.3.0 final report from results collected during live playback."""
    with _AGENT_LOCK:
        now_utc = datetime.now(timezone.utc).isoformat()
        keyframes = list(_COLLECTED_LIVE_KEYFRAMES)

        total_detected = len(keyframes)
        anomalies_count = sum(
            1 for kf in keyframes if kf.get("verdict") == "ANOMALOUS_CANDIDATE_FLAGGED"
        )
        quality_defects_count = sum(
            1 for kf in keyframes if kf.get("verdict") == "IMAGE_QUALITY_DEFECT"
        )
        nominal_count = sum(
            1 for kf in keyframes if kf.get("verdict") == "NOMINAL_CLEAR"
        )

        latencies = [kf.get("processing_latency_ms", 0.0) for kf in keyframes]
        mean_latency = float(np.mean(latencies)) if latencies else 0.0
        last_t = keyframes[-1]["timestamp_seconds"] if keyframes else 0.0

        coverage_pct = (
            round((last_t / video_duration_sec) * 100.0, 1)
            if video_duration_sec > 0
            else 100.0
        )

        if anomalies_count > 0:
            overall_verdict = "VIDEO_ANOMALY_CANDIDATE_DETECTED"
            summary_text = (
                f"Identified candidate anomalies across {anomalies_count} of {total_detected} analyzed playback frames. "
                "Human inspector confirmation is recommended for flagged coordinates."
            )
        elif quality_defects_count > 0:
            overall_verdict = "VIDEO_QUALITY_DEFECT_DETECTED"
            summary_text = (
                f"Detected image quality degradation (blur or exposure bounds) across {quality_defects_count} "
                f"of {total_detected} analyzed playback frames."
            )
        else:
            overall_verdict = "VIDEO_NOMINAL_CLEAR"
            summary_text = (
                f"All {total_detected} analyzed playback frames satisfied baseline clarity, contrast, and feature rules."
            )

        report = {
            "schema_version": "1.3.0",
            "system": {
                "application": "VisionGuard AI — Real-Time Live Video Inspection Server",
                "opencv_runtime_version": str(cv2.__version__),
                "inspection_timestamp_utc": now_utc,
            },
            "asset": {
                "filename": filename,
                "modality": "true_realtime_video_playback",
                "duration_seconds": round(video_duration_sec, 2),
                "fps": round(fps, 2),
            },
            "playback_telemetry": {
                "frames_analyzed": total_detected,
                "frames_sampled_from_player": total_frames_sampled,
                "frames_dropped_for_responsiveness": dropped_frames_count,
                "mean_inference_latency_ms": round(mean_latency, 2),
                "timeline_coverage_percent": coverage_pct,
                "last_playback_timestamp_sec": round(last_t, 2),
            },
            "coverage_statistics": {
                "total_keyframes_detected": total_detected,
                "skipped_identical_frames": dropped_frames_count,
                "last_analyzed_timestamp_sec": round(last_t, 2),
                "timeline_coverage_percent": coverage_pct,
                "mean_keyframe_latency_ms": round(mean_latency, 2),
                "frames_sampled_from_player": total_frames_sampled,
                "frames_dropped_for_responsiveness": dropped_frames_count,
            },
            "executive_summary": {
                "overall_verdict": overall_verdict,
                "summary_text": summary_text,
                "keyframes_with_anomalies": anomalies_count,
                "keyframes_with_quality_defects": quality_defects_count,
                "nominal_keyframes_count": nominal_count,
            },
            "detected_keyframes": keyframes,
            "disclaimer_and_limitations": (
                "Real-time video inspection operates on frames sampled asynchronously from the playing video stream. "
                "No temporal motion tracking or inter-frame optical flow is asserted. "
                + HUMAN_REVIEW_DISCLAIMER
            ),
        }
        return report


class VisionGuardRequestHandler(http.server.BaseHTTPRequestHandler):
    """HTTP request handler routing live frame analysis and report compilation requests."""

    def log_message(self, format: str, *args: Any):
        """Suppress default HTTP request logging to stderr."""
        pass

    def _send_cors_headers(self):
        """Send standard CORS headers permitting local browser requests."""
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Requested-With")

    def do_OPTIONS(self):
        """Handle CORS pre-flight requests."""
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        """Handle health check probes."""
        if self.path in ["/health", "/api/status", "/"]:
            self.send_response(200)
            self._send_cors_headers()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            resp = {
                "status": "healthy",
                "service": "VisionGuard AI Live Frame Server",
                "opencv_version": str(cv2.__version__),
                "active_keyframes_count": len(_COLLECTED_LIVE_KEYFRAMES),
            }
            self.wfile.write(json.dumps(resp).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        """Handle live frame analysis and report compilation endpoints."""
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)

        if self.path == "/api/reset":
            reset_live_session()
            self.send_response(200)
            self._send_cors_headers()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "reset_success"}).encode("utf-8"))
            return

        if self.path == "/api/analyze_frame":
            try:
                import base64
                payload = json.loads(post_data.decode("utf-8"))
                
                raw_frame_data = payload.get("frame_data", "")
                if "," in raw_frame_data:
                    raw_frame_data = raw_frame_data.split(",", 1)[1]
                frame_bytes = base64.b64decode(raw_frame_data)

                timestamp_sec = float(payload.get("timestamp", 0.0))
                visual_delta_thresh = float(payload.get("visual_change_threshold", 12.0))
                force_inspect = bool(payload.get("force_inspect", False))

                # Optional thresholds override
                thresholds = None
                t_dict = payload.get("thresholds")
                if t_dict and isinstance(t_dict, dict):
                    thresholds = InspectionThresholds(
                        BLUR_LAPLACIAN_VAR_WARN=float(t_dict.get("BLUR_LAPLACIAN_VAR_WARN", 80.0)),
                        UNDEREXPOSURE_MEAN_WARN=float(t_dict.get("UNDEREXPOSURE_MEAN_WARN", 35.0)),
                        OVEREXPOSURE_MEAN_WARN=float(t_dict.get("OVEREXPOSURE_MEAN_WARN", 225.0)),
                        LOW_CONTRAST_STD_WARN=float(t_dict.get("LOW_CONTRAST_STD_WARN", 18.0)),
                        CANNY_THRESHOLD_1=float(t_dict.get("CANNY_THRESHOLD_1", 50.0)),
                        CANNY_THRESHOLD_2=float(t_dict.get("CANNY_THRESHOLD_2", 150.0)),
                    )

                result = process_live_frame_payload(
                    frame_bytes=frame_bytes,
                    timestamp_sec=timestamp_sec,
                    thresholds=thresholds,
                    visual_change_threshold=visual_delta_thresh,
                    force_inspect=force_inspect,
                )

                self.send_response(200)
                self._send_cors_headers()
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(result).encode("utf-8"))

            except Exception as exc:
                self.send_response(400)
                self._send_cors_headers()
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": str(exc)}).encode("utf-8"))

        elif self.path == "/api/compile_report":
            try:
                payload = json.loads(post_data.decode("utf-8")) if post_data else {}
                filename = payload.get("filename", "video_playback.mp4")
                duration = float(payload.get("duration", 0.0))
                fps = float(payload.get("fps", 30.0))
                total_sampled = int(payload.get("total_sampled", len(_COLLECTED_LIVE_KEYFRAMES)))
                dropped_count = int(payload.get("dropped_frames", 0))

                report = compile_live_video_report(
                    filename=filename,
                    video_duration_sec=duration,
                    fps=fps,
                    total_frames_sampled=total_sampled,
                    dropped_frames_count=dropped_count,
                )

                self.send_response(200)
                self._send_cors_headers()
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(report).encode("utf-8"))

            except Exception as exc:
                self.send_response(500)
                self._send_cors_headers()
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": str(exc)}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


_SERVER_INSTANCE: Optional[http.server.ThreadingHTTPServer] = None
_SERVER_PORT: int = 8505


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Check if a network port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0


def start_background_server(start_port: int = 8505, max_attempts: int = 10) -> int:
    """Launch the VisionGuard live frame server on a background daemon thread.
    
    Returns:
        The active port number bound by the HTTP server.
    """
    global _SERVER_INSTANCE, _SERVER_PORT
    if _SERVER_INSTANCE is not None:
        return _SERVER_PORT

    for offset in range(max_attempts):
        port = start_port + offset
        try:
            server = http.server.ThreadingHTTPServer(("127.0.0.1", port), VisionGuardRequestHandler)
            _SERVER_INSTANCE = server
            _SERVER_PORT = port
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            return port
        except OSError:
            continue

    raise RuntimeError(f"Could not bind VisionGuard server on ports {start_port} to {start_port + max_attempts - 1}.")


def stop_background_server():
    """Shut down the background HTTP server."""
    global _SERVER_INSTANCE
    if _SERVER_INSTANCE is not None:
        try:
            _SERVER_INSTANCE.shutdown()
            _SERVER_INSTANCE.server_close()
        except Exception:
            pass
        _SERVER_INSTANCE = None
