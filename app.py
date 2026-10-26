"""VisionGuard AI — Explainable Visual Inspection Assistant.

Streamlit application designed for human visual inspection, offering
OpenCV 5 metrics, evidence-driven VisionAgent decision workflow, and reproducible audits
for both still images and video sequences.
"""

from datetime import datetime, timezone
import io
import os
import platform
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import cv2
import numpy as np

from visionguard.agent import AgentInspectionSession, VisionAgent
from visionguard.analysis import (
    CandidateRegion,
    InspectionNotice,
    InspectionResult,
    analyze_image,
)
from visionguard.config import (
    HUMAN_REVIEW_DISCLAIMER,
    InspectionThresholds,
    SafetyLimits,
    VideoSafetyLimits,
)
from visionguard.report import (
    generate_inspection_report,
    generate_video_inspection_report,
    normalize_schema_1_3_report,
    serialize_report_json,
    validate_schema_1_3_report,
)
from visionguard.synthetic_data import (
    generate_blurred_checkerboard,
    generate_low_contrast_flat,
    generate_overexposed_pattern,
    generate_sharp_checkerboard,
    generate_surface_with_artifact,
    generate_underexposed_pattern,
)
from visionguard.components.sync_player import render_sync_video_player
from visionguard.validation import ImageValidationError, validate_and_decode_image
from visionguard.video import (
    SampledVideoFrame,
    VideoProcessingError,
    generate_annotated_video_bytes,
    get_video_mime_type,
    sample_video_frames_from_bytes,
)
from visionguard.server import (
    compile_live_video_report,
    reset_live_session,
    start_background_server,
)
from visionguard.video_stream import LiveKeyframeResult, SynchronizedVideoInspector
from visionguard.visualization import (
    annotate_candidate_regions,
    encode_image_bytes,
    extract_candidate_crop,
)

# Page configuration
st.set_page_config(
    page_title="VisionGuard AI — Inspection Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling for high-contrast engineering tool aesthetics
st.markdown(
    """
    <style>
    .vg-header {
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
        color: #111827;
    }
    .vg-sub {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }
    .metric-label {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #6b7280;
    }
    .metric-val {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1f2937;
    }
    .metric-hint {
        font-size: 0.78rem;
        color: #9ca3af;
    }
    .verdict-card {
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 18px;
    }
    .verdict-nominal {
        background-color: #f0fdf4;
        border: 1px solid #86efac;
        color: #166534;
    }
    .verdict-anomaly {
        background-color: #fff7ed;
        border: 1px solid #fdba74;
        color: #9a3412;
    }
    .verdict-defect {
        background-color: #fef2f2;
        border: 1px solid #fca5a5;
        color: #991b1b;
    }
    .timeline-step {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-left: 4px solid #3b82f6;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 10px;
    }
    .tool-badge {
        font-family: monospace;
        font-size: 0.82rem;
        background-color: #eff6ff;
        color: #1d4ed8;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 600;
    }
    .warning-box {
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 12px;
    }
    .disclaimer-banner {
        background-color: #f3f4f6;
        border: 1px solid #d1d5db;
        border-radius: 6px;
        padding: 10px 14px;
        font-size: 0.85rem;
        color: #374151;
        margin-top: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_sidebar() -> Tuple[InspectionThresholds, str, float, float]:
    with st.sidebar:
        st.markdown("### 🔍 VisionGuard AI")
        st.caption(f"OpenCV Engine: **v{cv2.__version__}** | Python {platform.python_version()}")
        st.divider()

        st.markdown("#### Input Source")
        input_mode = st.radio(
            "Select input modality:",
            options=["Upload Image", "Upload Video (MP4 / AVI)", "Load Synthetic Benchmark"],
            index=0,
            help="Inspect still images, sample video keyframes, or run reproducible synthetic patterns.",
        )

        sample_interval_sec = 1.0
        visual_change_thresh = 12.0
        if input_mode == "Upload Video (MP4 / AVI)":
            st.markdown("#### Video Inspection Controls")
            sample_interval_sec = st.slider(
                "Sampling Cadence (seconds)",
                min_value=0.2,
                max_value=3.0,
                value=1.0,
                step=0.1,
                help="Cadence in seconds for periodic keyframe evaluations.",
            )
            visual_change_thresh = st.slider(
                "Visual Delta Trigger (Δ)",
                min_value=4.0,
                max_value=35.0,
                value=12.0,
                step=1.0,
                help="Mean pixel delta required to trigger immediate keyframe inspection.",
            )

        st.divider()
        st.markdown("#### Inspection Thresholds")
        st.caption("Adjust detection parameters. Measured values will be compared against these rules.")

        blur_thresh = st.slider(
            "Laplacian Sharpness Warning",
            min_value=10.0,
            max_value=300.0,
            value=float(InspectionThresholds.BLUR_LAPLACIAN_VAR_WARN),
            step=5.0,
            help="Values below this threshold trigger a focus/sharpness warning.",
        )

        col_lum1, col_lum2 = st.columns(2)
        with col_lum1:
            underexp_thresh = st.number_input(
                "Underexposure",
                min_value=5.0,
                max_value=100.0,
                value=float(InspectionThresholds.UNDEREXPOSURE_MEAN_WARN),
                step=5.0,
            )
        with col_lum2:
            overexp_thresh = st.number_input(
                "Overexposure",
                min_value=150.0,
                max_value=254.0,
                value=float(InspectionThresholds.OVEREXPOSURE_MEAN_WARN),
                step=5.0,
            )

        contrast_thresh = st.slider(
            "RMS Contrast Warning",
            min_value=5.0,
            max_value=60.0,
            value=float(InspectionThresholds.LOW_CONTRAST_STD_WARN),
            step=1.0,
        )

        st.markdown("#### Feature & Edge Tuning")
        canny_t1 = st.slider(
            "Canny Lower Hysteresis",
            min_value=10.0,
            max_value=150.0,
            value=float(InspectionThresholds.CANNY_THRESHOLD_1),
            step=5.0,
        )
        canny_t2 = st.slider(
            "Canny Upper Hysteresis",
            min_value=50.0,
            max_value=300.0,
            value=float(InspectionThresholds.CANNY_THRESHOLD_2),
            step=5.0,
        )

        thresholds = InspectionThresholds(
            BLUR_LAPLACIAN_VAR_WARN=blur_thresh,
            UNDEREXPOSURE_MEAN_WARN=underexp_thresh,
            OVEREXPOSURE_MEAN_WARN=overexp_thresh,
            LOW_CONTRAST_STD_WARN=contrast_thresh,
            CANNY_THRESHOLD_1=canny_t1,
            CANNY_THRESHOLD_2=canny_t2,
        )

        st.divider()
        st.markdown("#### Architecture & Security")
        st.caption(
            "• **In-Memory Decoding**: Safe buffers; no unverified disk leaks.\n"
            "• **Bounded Agent Workflow**: Multi-step OpenCV decisions with hard timeouts.\n"
            "• **Human-Review-First**: Algorithmic regions are suggestions, never autonomous verdicts."
        )

    return thresholds, input_mode, sample_interval_sec, visual_change_thresh


def render_inspection_details(
    inspection_res: InspectionResult,
    session: AgentInspectionSession,
    filename_label: str,
    thresholds: InspectionThresholds,
    is_video_frame: bool = False,
    timestamp_sec: Optional[float] = None,
):
    """Render metrics, side-by-side images, agent timeline, and candidate inspector."""
    annotated_rgb = annotate_candidate_regions(
        image_rgb=inspection_res.image_rgb,
        candidate_regions=inspection_res.candidate_regions,
    )

    # Visual Inspection Display
    st.markdown("---")
    col_img1, col_img2 = st.columns([1, 1])

    time_tag = f" @ {timestamp_sec:.2f}s" if timestamp_sec is not None else ""
    with col_img1:
        st.markdown(f"##### Original {'Frame' if is_video_frame else 'Image'}{time_tag}")
        st.image(
            inspection_res.image_rgb,
            caption=f"{filename_label} ({inspection_res.image_rgb.shape[1]}x{inspection_res.image_rgb.shape[0]} px)",
            use_container_width=True,
        )

    with col_img2:
        st.markdown("##### Annotated Candidate Regions (For Human Review)")
        st.image(
            annotated_rgb,
            caption=f"Highlighted {len(inspection_res.candidate_regions)} candidate region(s) — Not confirmed defects",
            use_container_width=True,
        )

    # Indicator Metrics Row
    st.markdown("#### Measured Quality Indicators")
    m = inspection_res.metrics
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)

    with m_col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Sharpness (Laplacian Var)</div>
                <div class="metric-val">{m.laplacian_variance:.1f}</div>
                <div class="metric-hint">Threshold: ≥ {thresholds.BLUR_LAPLACIAN_VAR_WARN:.1f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m_col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Mean Intensity</div>
                <div class="metric-val">{m.mean_brightness:.1f}</div>
                <div class="metric-hint">Nominal: [{thresholds.UNDEREXPOSURE_MEAN_WARN:.0f}, {thresholds.OVEREXPOSURE_MEAN_WARN:.0f}]</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m_col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">RMS Contrast</div>
                <div class="metric-val">{m.rms_contrast:.1f}</div>
                <div class="metric-hint">Threshold: ≥ {thresholds.LOW_CONTRAST_STD_WARN:.1f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m_col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Canny Edge Density</div>
                <div class="metric-val">{m.canny_edge_density * 100:.2f}%</div>
                <div class="metric-hint">Sobel Gradient: {m.mean_sobel_gradient:.1f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Agent Action Timeline
    st.markdown(f"#### ⚡ Agent Decision Timeline ({'Frame' if is_video_frame else 'Image'})")
    st.caption(f"Completed in {session.elapsed_time_ms:.1f} ms across {session.total_steps} bounded steps.")

    for action in session.action_trace:
        st.markdown(
            f"""
            <div class="timeline-step">
                <strong>Step {action.step_number}:</strong> <span class="tool-badge">{action.tool_name}</span><br/>
                <em>Rationale:</em> {action.reason}<br/>
                <details style="margin-top: 6px; font-size: 0.85rem;">
                    <summary style="cursor: pointer; color: #4b5563;">View Parameters & Observations</summary>
                    <pre style="background-color: #f3f4f6; padding: 8px; border-radius: 4px; margin-top: 4px;">Inputs: {action.input_params}\nObservations: {action.observation}</pre>
                </details>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Candidate Region Crop Inspector
    if inspection_res.candidate_regions:
        st.markdown("#### Candidate Region Inspector & Reinspection")
        st.caption("Inspect individual candidate bounding boxes up-close. Regions are highlighted for human examination.")

        region_options = [
            f"Region #{c.region_id} (Area: {c.contour_area:.0f} px, Box: {c.bbox_w}x{c.bbox_h} at ({c.bbox_x},{c.bbox_y}))"
            for c in inspection_res.candidate_regions
        ]
        selected_idx = st.selectbox("Select candidate region to inspect:", range(len(region_options)), format_func=lambda i: region_options[i])
        selected_cand = inspection_res.candidate_regions[selected_idx]

        crop_col1, crop_col2 = st.columns([1, 2])
        crop_rgb = extract_candidate_crop(inspection_res.image_rgb, selected_cand)

        with crop_col1:
            st.image(crop_rgb, caption=f"Zoomed Crop: Region #{selected_cand.region_id}", use_container_width=True)

        with crop_col2:
            st.markdown(f"**Region Identifier:** #{selected_cand.region_id}")
            st.markdown(f"**Bounding Box:** x={selected_cand.bbox_x}, y={selected_cand.bbox_y}, w={selected_cand.bbox_w}, h={selected_cand.bbox_h}")
            st.markdown(f"**Contour Area:** {selected_cand.contour_area:.1f} pixels")
            st.markdown(f"**Aspect Ratio (W/H):** {selected_cand.aspect_ratio:.2f}")
            st.markdown(f"**Internal Edge Density:** {selected_cand.internal_edge_density * 100:.2f}%")
            st.info("Status: **Candidate region for review** (Heuristic suggestion; inspector confirmation required).")

    # Edge Map & Grayscale Diagnostic View
    with st.expander("OpenCV Diagnostic Maps (Canny Edge & Grayscale)"):
        diag_col1, diag_col2 = st.columns(2)
        with diag_col1:
            st.image(inspection_res.gray_image, caption="Grayscale Representation", use_container_width=True)
        with diag_col2:
            st.image(inspection_res.edge_map, caption=f"Canny Edges ({thresholds.CANNY_THRESHOLD_1:.0f}, {thresholds.CANNY_THRESHOLD_2:.0f})", use_container_width=True)

    return annotated_rgb


def render_live_keyframe_panel(
    container: Any,
    kf: LiveKeyframeResult,
    inspector: SynchronizedVideoInspector,
    thresholds: InspectionThresholds,
):
    """Render real-time live inspection status, frames, and metrics inside a container."""
    with container.container():
        reason_color = (
            "#2563eb" if kf.selection_reason == "INITIAL_FRAME"
            else "#059669" if kf.selection_reason == "REGULAR_INTERVAL"
            else "#d97706" if kf.selection_reason == "SIGNIFICANT_VISUAL_CHANGE"
            else "#7c3aed"
        )
        reason_label = kf.selection_reason.replace("_", " ")
        if kf.visual_delta > 0:
            reason_label += f" (Δ={kf.visual_delta:.1f})"

        verdict_str = kf.agent_session.final_verdict.verdict_type
        v_class = (
            "verdict-nominal" if verdict_str == "NOMINAL_CLEAR"
            else "verdict-anomaly" if verdict_str == "ANOMALOUS_CANDIDATE_FLAGGED"
            else "verdict-defect"
        )

        st.markdown(
            f"""
            <div style="background: #1e293b; color: #f8fafc; padding: 12px 16px; border-radius: 8px; margin-bottom: 12px; border-left: 5px solid {reason_color};">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                    <div>
                        <span style="font-weight: 700; font-size: 1.1rem;">⏱️ Live Playback Timestamp: {kf.timestamp_seconds:.2f}s</span>
                        <span style="font-size: 0.85rem; color: #94a3b8; margin-left: 8px;">(Frame #{kf.frame_index})</span>
                    </div>
                    <div>
                        <span style="background: {reason_color}; color: #fff; font-size: 0.78rem; font-weight: 600; padding: 4px 8px; border-radius: 4px;">
                            TRIGGER: {reason_label}
                        </span>
                        <span style="background: #334155; color: #cbd5e1; font-size: 0.78rem; font-weight: 600; padding: 4px 8px; border-radius: 4px; margin-left: 6px;">
                            KEYFRAMES: {len(inspector.keyframes)} (Skipped: {inspector.skipped_frames_count})
                        </span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        p_col1, p_col2 = st.columns(2)
        with p_col1:
            st.markdown(f"**Latest Keyframe (t = {kf.timestamp_seconds:.2f}s)**")
            st.image(kf.inspection_res.image_rgb, use_container_width=True)
        with p_col2:
            st.markdown(f"**OpenCV Candidate Overlay ({len(kf.inspection_res.candidate_regions)} detected)**")
            st.image(kf.annotated_rgb, use_container_width=True)

        m = kf.inspection_res.metrics
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Sharpness", f"{m.laplacian_variance:.1f}")
        with c2:
            st.metric("Brightness", f"{m.mean_brightness:.1f}")
        with c3:
            st.metric("Contrast", f"{m.rms_contrast:.1f}")
        with c4:
            st.metric("Edge Density", f"{m.canny_edge_density * 100:.2f}%")
        with c5:
            st.metric("Latency", f"{kf.processing_latency_ms:.1f} ms")

        st.markdown(
            f"""
            <div class="verdict-card {v_class}" style="margin-top: 10px; margin-bottom: 8px; padding: 10px 14px;">
                <strong>Keyframe Verdict:</strong> {verdict_str.replace('_', ' ')} (Confidence: {kf.agent_session.final_verdict.confidence})<br/>
                <span style="font-size: 0.88rem;">{kf.agent_session.final_verdict.summary}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main():
    thresholds, input_mode, sample_interval_sec, visual_change_thresh = render_sidebar()

    st.markdown('<div class="vg-header">VisionGuard AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="vg-sub">Explainable computer-vision assistant with evidence-driven agent orchestration for industrial visual inspection.</div>',
        unsafe_allow_html=True,
    )

    agent = VisionAgent(thresholds=thresholds)

    # -------------------------------------------------------------
    # CASE 1: Synchronized Live Video Playback & Inspection Workflow
    # -------------------------------------------------------------
    if input_mode == "Upload Video (MP4 / AVI)":
        st.markdown("##### Synchronized Video Sequence Inspection")
        st.caption(
            "Upload an MP4 or AVI video clip (Max 50 MB). Keyframes are detected during playback via cadence and visual change triggers, analyzed with OpenCV 5 and VisionAgent."
        )

        uploaded_video = st.file_uploader(
            "Select an industrial video clip (MP4, AVI — Max 50 MB):",
            type=["mp4", "avi", "mov"],
            help="OpenCV VideoCapture will decode and sample representative frames safely.",
        )

        if uploaded_video is None:
            st.info("Upload a video clip above to begin synchronized live video inspection.")
            return

        try:
            video_bytes = uploaded_video.read()
            video_mime = get_video_mime_type(uploaded_video.name)
            video_id = f"{uploaded_video.name}_{len(video_bytes)}"
        except Exception as exc:
            st.error(f"Failed to read uploaded video: {exc}")
            return

        # Session state management for synchronized inspector
        if st.session_state.get("video_source_id") != video_id:
            if "sync_inspector" in st.session_state and st.session_state["sync_inspector"] is not None:
                try:
                    st.session_state["sync_inspector"].close()
                except Exception:
                    pass
            try:
                st.session_state["sync_inspector"] = SynchronizedVideoInspector(
                    video_source=video_bytes,
                    filename=uploaded_video.name,
                    sample_interval_sec=sample_interval_sec,
                    visual_change_threshold=visual_change_thresh,
                    thresholds=thresholds,
                    agent=agent,
                )
                st.session_state["video_source_id"] = video_id
                st.session_state["inspection_finished"] = False
                st.session_state["consolidated_report"] = None
                st.session_state["selected_kf_idx"] = 0
            except VideoProcessingError as vpe:
                st.error(f"**Video Ingestion Rejected**: {str(vpe)}")
                return
            except Exception as exc:
                st.error(f"**Initialization Error**: {str(exc)}")
                return

        inspector: SynchronizedVideoInspector = st.session_state["sync_inspector"]
        inspector.update_thresholds(thresholds)
        inspector.sample_interval_sec = sample_interval_sec
        inspector.visual_change_threshold = visual_change_thresh

        # Ensure background frame analysis server is active
        try:
            api_port = start_background_server(start_port=8505)
        except Exception:
            api_port = 8505

        thresholds_dict = {
            "BLUR_LAPLACIAN_VAR_WARN": float(thresholds.BLUR_LAPLACIAN_VAR_WARN),
            "UNDEREXPOSURE_MEAN_WARN": float(thresholds.UNDEREXPOSURE_MEAN_WARN),
            "OVEREXPOSURE_MEAN_WARN": float(thresholds.OVEREXPOSURE_MEAN_WARN),
            "LOW_CONTRAST_STD_WARN": float(thresholds.LOW_CONTRAST_STD_WARN),
            "CANNY_THRESHOLD_1": float(thresholds.CANNY_THRESHOLD_1),
            "CANNY_THRESHOLD_2": float(thresholds.CANNY_THRESHOLD_2),
        }

        # Synchronized Player Console
        st.markdown("#### 🎬 Real-Time Synchronized Video Player & Live Inspection Console")
        st.caption(
            f"Source: **{uploaded_video.name}** ({len(video_bytes) / (1024 * 1024):.2f} MB | "
            f"{inspector.width}x{inspector.height} @ {inspector.fps:.1f} fps | Duration: {inspector.duration_seconds:.2f}s)"
        )

        # Render bidirectional custom HTML5 video component with live frame streaming
        player_data = render_sync_video_player(
            video_bytes=video_bytes,
            mime_type=video_mime,
            api_port=api_port,
            filename=uploaded_video.name,
            sync_interval_sec=sample_interval_sec,
            visual_change_thresh=visual_change_thresh,
            thresholds=thresholds_dict,
            key="vg_sync_player",
        )

        # Process playback events emitted from the frontend component
        if player_data and isinstance(player_data, dict):
            action = player_data.get("action")
            if action == "CONSOLIDATED_REPORT_COMPILED":
                rep_payload = player_data.get("report")
                if rep_payload:
                    st.session_state["consolidated_report"] = normalize_schema_1_3_report(rep_payload)
                st.session_state["inspection_finished"] = True
            elif action in ["START_INSPECTION", "SYNC_TICK"]:
                curr_time = float(player_data.get("currentTime", 0.0))
                kf = inspector.evaluate_live_timestamp(curr_time)
                if kf is not None:
                    st.session_state["selected_kf_idx"] = len(inspector.keyframes) - 1
            elif action in ["FINISH_INSPECTION", "VIDEO_ENDED"]:
                st.session_state["inspection_finished"] = True
                if not st.session_state.get("consolidated_report"):
                    st.session_state["consolidated_report"] = normalize_schema_1_3_report(
                        inspector.compile_consolidated_report()
                    )

        # Action Buttons Row
        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 1, 1])
        with ctrl_col1:
            run_stream = st.button("▶ Run Synchronized Stream Inspection", type="primary", use_container_width=True)
        with ctrl_col2:
            finish_btn = st.button("🏁 Finish Inspection", use_container_width=True)
        with ctrl_col3:
            reset_btn = st.button("🔄 Reset Inspection", use_container_width=True)

        if reset_btn:
            inspector.reset()
            st.session_state["inspection_finished"] = False
            st.session_state["consolidated_report"] = None
            st.session_state["selected_kf_idx"] = 0
            st.rerun()

        if finish_btn:
            st.session_state["inspection_finished"] = True
            st.session_state["consolidated_report"] = normalize_schema_1_3_report(
                inspector.compile_consolidated_report()
            )
            st.rerun()

        if run_stream:
            inspector.reset()
            stream_prog = st.progress(0.0, text="Initializing synchronized video stream inspection...")
            stream_container = st.empty()

            def on_stream_kf(kf: LiveKeyframeResult, t_curr: float):
                pct = min(1.0, max(0.0, t_curr / inspector.duration_seconds)) if inspector.duration_seconds > 0 else 1.0
                stream_prog.progress(
                    pct,
                    text=f"Inspecting stream @ t={t_curr:.2f}s / {inspector.duration_seconds:.2f}s (Detected {len(inspector.keyframes)} keyframes)...",
                )
                render_live_keyframe_panel(stream_container, kf, inspector, thresholds)

            inspector.run_synchronized_inspection_stream(
                on_keyframe_callback=on_stream_kf,
                realtime_pacing=False,
            )
            stream_prog.progress(1.0, text="Synchronized inspection stream completed!")
            st.session_state["inspection_finished"] = True
            st.session_state["consolidated_report"] = normalize_schema_1_3_report(
                inspector.compile_consolidated_report()
            )
            st.session_state["selected_kf_idx"] = max(0, len(inspector.keyframes) - 1)
            st.rerun()

        if uploaded_video.name.lower().endswith(".avi"):
            with st.expander("ℹ️ Direct AVI Playback Note", expanded=False):
                st.info(
                    "Direct in-browser decoding of legacy AVI containers depends on client browser codec support. "
                    "VisionGuard extracts frames with native OpenCV regardless of browser playback codecs. "
                    "If the HTML5 player does not show video controls in your browser, use 'Run Synchronized Stream Inspection' above."
                )
                st.video(video_bytes, format=video_mime, autoplay=False)

        # Check if consolidated report exists in session state
        report_dict = st.session_state.get("consolidated_report")
        if report_dict is not None:
            report_dict = normalize_schema_1_3_report(report_dict)
            st.session_state["consolidated_report"] = report_dict

        # If live player compiled a report but inspector.keyframes was not populated in this session,
        # extract and inspect keyframes from container so deep inspector & gallery work seamlessly.
        if len(inspector.keyframes) == 0 and report_dict is not None:
            for kf_info in report_dict.get("detected_keyframes", []):
                ts = float(kf_info.get("timestamp_seconds", 0.0))
                try:
                    inspector.evaluate_live_timestamp(ts, force_inspect=True)
                except Exception:
                    pass

        has_keyframes = len(inspector.keyframes) > 0
        has_report = report_dict is not None

        if has_keyframes or has_report:
            # Live Inspection Status Panel
            if has_keyframes:
                st.markdown("---")
                st.markdown("#### ⚡ Live Inspection Status")
                latest_kf = inspector.keyframes[-1]
                live_panel_container = st.container()
                render_live_keyframe_panel(live_panel_container, latest_kf, inspector, thresholds)

                # Detected Keyframe Gallery & Selector
                st.markdown("---")
                st.markdown("#### 🖼️ Detected Keyframe Gallery")
                st.caption(
                    "Evidence-driven keyframes extracted during video playback via cadence and visual change triggers:"
                )

                gallery_labels = [
                    f"Frame #{kf.frame_index} (t = {kf.timestamp_seconds:.2f}s) — [{kf.selection_reason.replace('_', ' ')}] — {kf.agent_session.final_verdict.verdict_type.replace('_', ' ')}"
                    for kf in inspector.keyframes
                ]
                curr_selected = min(st.session_state.get("selected_kf_idx", 0), len(gallery_labels) - 1)
                selected_idx = st.selectbox(
                    "Select Keyframe for Deep Inspection & Bounding Box Review:",
                    range(len(gallery_labels)),
                    index=curr_selected,
                    format_func=lambda i: gallery_labels[i],
                    key="keyframe_selector",
                )
                st.session_state["selected_kf_idx"] = selected_idx
                sel_kf = inspector.keyframes[selected_idx]

                # Thumbnails row (up to 6)
                num_thumbs = min(len(inspector.keyframes), 6)
                thumb_cols = st.columns(num_thumbs)
                for i_col, kf_item in enumerate(inspector.keyframes[:num_thumbs]):
                    with thumb_cols[i_col]:
                        st.image(
                            kf_item.inspection_res.image_rgb,
                            caption=f"t={kf_item.timestamp_seconds:.2f}s\n#{kf_item.frame_index}",
                            use_container_width=True,
                        )

                # Deep inspection of selected keyframe
                annotated_rgb = render_inspection_details(
                    inspection_res=sel_kf.inspection_res,
                    session=sel_kf.agent_session,
                    filename_label=f"{uploaded_video.name} [Frame #{sel_kf.frame_index}]",
                    thresholds=thresholds,
                    is_video_frame=True,
                    timestamp_sec=sel_kf.timestamp_seconds,
                )
            else:
                sel_kf = None
                annotated_rgb = None

            # Consolidated Report & Exports
            if report_dict is None and has_keyframes:
                report_dict = normalize_schema_1_3_report(inspector.compile_consolidated_report())
                st.session_state["consolidated_report"] = report_dict

            if report_dict is not None:
                report_dict = normalize_schema_1_3_report(report_dict)
                is_valid, validation_errors = validate_schema_1_3_report(report_dict)
                if not is_valid:
                    st.warning(f"Report Schema Notice: {', '.join(validation_errors)}")

                st.markdown("---")
                st.markdown("#### 📊 Consolidated Final Inspection Report (Schema v1.3.0)")
                exec_sum = report_dict.get("executive_summary", {})
                cov_stat = report_dict.get("coverage_statistics", {})

                overall_verdict = exec_sum.get("overall_verdict", "VIDEO_NOMINAL_CLEAR")
                summary_text = exec_sum.get("summary_text", "Inspection sequence completed.")
                timeline_cov = cov_stat.get("timeline_coverage_percent", 0.0)
                total_kf = cov_stat.get("total_keyframes_detected", 0)
                skipped_frames = cov_stat.get("skipped_identical_frames", 0)
                anomalies_count = exec_sum.get("keyframes_with_anomalies", 0)
                defects_count = exec_sum.get("keyframes_with_quality_defects", 0)
                mean_latency = cov_stat.get("mean_keyframe_latency_ms", 0.0)

                v_class = (
                    "verdict-nominal" if overall_verdict == "VIDEO_NOMINAL_CLEAR"
                    else "verdict-anomaly" if overall_verdict == "VIDEO_ANOMALY_CANDIDATE_DETECTED"
                    else "verdict-defect"
                )

                st.markdown(
                    f"""
                    <div class="verdict-card {v_class}">
                        <h3 style="margin: 0 0 6px 0;">Consolidated Verdict: {overall_verdict.replace('_', ' ')}</h3>
                        <p style="margin: 0 0 8px 0; font-size: 0.95rem;">{summary_text}</p>
                        <div style="font-size: 0.88rem;">
                            • <strong>Timeline Coverage:</strong> {timeline_cov}% ({total_kf} keyframes detected, {skipped_frames} skipped identical frames)<br/>
                            • <strong>Candidate Anomalies Flagged:</strong> {anomalies_count} keyframe(s)<br/>
                            • <strong>Quality Defects (Blur/Exposure):</strong> {defects_count} keyframe(s)<br/>
                            • <strong>Mean Processing Latency:</strong> {mean_latency} ms/keyframe
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Optional Annotated Video Overlay Generation
                with st.expander("🎞️ Annotated Video Overlay (Full Sequence Processing)", expanded=False):
                    st.caption(
                        "Optionally generate an annotated MP4 video with OpenCV candidate region overlays and timestamps rendered across frames."
                    )
                    if st.button("Generate Annotated Video Overlay", key="btn_render_annotated"):
                        with st.spinner("Rendering OpenCV overlays onto video stream..."):
                            annotated_vid = generate_annotated_video_bytes(
                                video_bytes=video_bytes,
                                filename=uploaded_video.name,
                                thresholds=thresholds,
                            )
                        if annotated_vid is not None:
                            st.success("Annotated overlay video rendered successfully!")
                            st.video(annotated_vid, format="video/mp4", autoplay=False)
                            st.download_button(
                                label="📥 Download Annotated Video (MP4)",
                                data=annotated_vid,
                                file_name=f"visionguard_annotated_{os.path.splitext(uploaded_video.name)[0]}.mp4",
                                mime="video/mp4",
                                use_container_width=True,
                            )
                        else:
                            st.warning(
                                "Annotated video encoding is unavailable on this host environment's OpenCV video codecs. "
                                "Keyframe sequence inspection and frame downloads remain fully operational."
                            )

                # Export section
                st.markdown("#### Export Consolidated Audit & Media")
                report_json_str = serialize_report_json(report_dict)
                frame_png_bytes = (
                    encode_image_bytes(annotated_rgb, is_rgb=True, fmt=".png")
                    if annotated_rgb is not None
                    else None
                )

                d1, d2, d3 = st.columns(3)
                with d1:
                    st.download_button(
                        label="📥 Download Consolidated Report (JSON)",
                        data=report_json_str,
                        file_name=f"visionguard_report_{os.path.splitext(uploaded_video.name)[0]}.json",
                        mime="application/json",
                        use_container_width=True,
                    )
                with d2:
                    if frame_png_bytes is not None and sel_kf is not None:
                        st.download_button(
                            label=f"📥 Download Frame #{sel_kf.frame_index} (PNG)",
                            data=frame_png_bytes,
                            file_name=f"visionguard_frame_{sel_kf.frame_index}.png",
                            mime="image/png",
                            use_container_width=True,
                        )
                    else:
                        st.caption("Select a keyframe above to download annotated PNG.")
                with d3:
                    st.download_button(
                        label="📥 Download Original Video",
                        data=video_bytes,
                        file_name=uploaded_video.name,
                        mime=video_mime,
                        use_container_width=True,
                    )

        st.markdown(f'<div class="disclaimer-banner">{HUMAN_REVIEW_DISCLAIMER}</div>', unsafe_allow_html=True)
        return

    # -------------------------------------------------------------
    # CASE 2: Still Image Upload / Synthetic Benchmark Workflow
    # -------------------------------------------------------------
    image_data: Optional[bytes] = None
    filename_label: str = "upload"

    if input_mode == "Upload Image":
        uploaded_file = st.file_uploader(
            "Select an image file for inspection (PNG, JPEG, WebP — Max 15 MB):",
            type=["png", "jpg", "jpeg", "webp"],
            help="Images are decoded safely in memory using OpenCV 5.",
        )
        if uploaded_file is not None:
            image_data = uploaded_file.read()
            filename_label = uploaded_file.name
        else:
            st.info("Upload an image above to begin inspection, or switch modes in the sidebar.")
            return

    else:  # Load Synthetic Benchmark
        st.markdown("##### Select Synthetic Benchmark Pattern")
        sample_choice = st.selectbox(
            "Select reproducible benchmark pattern:",
            options=[
                "surface_with_artifact (Clean surface with intentional scratch anomaly)",
                "sharp_checkerboard (High-frequency sharp edges - Nominal)",
                "blurred_checkerboard (Gaussian-blurred focus loss defect)",
                "underexposed_pattern (Low-light underexposure defect)",
                "overexposed_pattern (Saturated highlight overexposure defect)",
                "low_contrast_flat (Uniform gray low-contrast field)",
            ],
            index=0,
        )

        if "surface_with_artifact" in sample_choice:
            raw_img, _ = generate_surface_with_artifact(512, 512)
        elif "sharp_checkerboard" in sample_choice:
            raw_img = generate_sharp_checkerboard(512, 512)
        elif "blurred_checkerboard" in sample_choice:
            raw_img = generate_blurred_checkerboard(512, 512)
        elif "underexposed_pattern" in sample_choice:
            raw_img = generate_underexposed_pattern(512, 512)
        elif "overexposed_pattern" in sample_choice:
            raw_img = generate_overexposed_pattern(512, 512)
        else:
            raw_img = generate_low_contrast_flat(512, 512)

        image_data = encode_image_bytes(raw_img, is_rgb=False, fmt=".png")
        filename_label = f"synthetic_{sample_choice.split()[0]}.png"

    # Process and validate image data
    try:
        with st.spinner("Decoding and executing OpenCV 5 VisionAgent decision workflow..."):
            validation_res = validate_and_decode_image(image_data, filename=filename_label)
            session: AgentInspectionSession = agent.inspect(validation_res.image_bgr)
            inspection_res = session.initial_result
    except ImageValidationError as val_err:
        st.error(f"**Validation Rejected**: {str(val_err)}")
        return
    except Exception as exc:
        st.error(f"**Processing Error**: Unexpected failure during analysis: {str(exc)}")
        return

    # Agent Final Verdict Card
    verdict = session.final_verdict
    card_class = (
        "verdict-nominal"
        if verdict.verdict_type == "NOMINAL_CLEAR"
        else "verdict-anomaly"
        if verdict.verdict_type == "ANOMALOUS_CANDIDATE_FLAGGED"
        else "verdict-defect"
    )

    st.markdown(
        f"""
        <div class="verdict-card {card_class}">
            <h3 style="margin: 0 0 6px 0;">Agent Verdict: {verdict.verdict_type.replace('_', ' ')} (Confidence: {verdict.confidence})</h3>
            <p style="margin: 0 0 8px 0; font-size: 0.95rem;">{verdict.summary}</p>
            <strong>Recommended Actions:</strong>
            <ul style="margin: 4px 0 0 0; padding-left: 20px;">
                {''.join(f'<li>{r}</li>' for r in verdict.recommended_actions)}
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    annotated_rgb = render_inspection_details(
        inspection_res=inspection_res,
        session=session,
        filename_label=filename_label,
        thresholds=thresholds,
        is_video_frame=False,
    )

    # Export & Downloads for Still Image Mode
    st.markdown("#### Export Reproducible Audit")
    report_dict = generate_inspection_report(
        inspection_res,
        filename=filename_label,
        agent_session=session,
    )
    report_json_str = serialize_report_json(report_dict)
    annotated_png_bytes = encode_image_bytes(annotated_rgb, is_rgb=True, fmt=".png")

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        st.download_button(
            label="📥 Download Complete Audit Report (JSON)",
            data=report_json_str,
            file_name=f"visionguard_audit_{filename_label}.json",
            mime="application/json",
            use_container_width=True,
        )
    with btn_col2:
        st.download_button(
            label="📥 Download Annotated Image (PNG)",
            data=annotated_png_bytes,
            file_name=f"visionguard_annotated_{filename_label}.png",
            mime="image/png",
            use_container_width=True,
        )

    st.markdown(
        f'<div class="disclaimer-banner">{HUMAN_REVIEW_DISCLAIMER}</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
