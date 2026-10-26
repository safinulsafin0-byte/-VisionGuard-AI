# VisionGuard AI — Project Audit & Implementation Blueprint

## 1. Verified Environment & Baseline Findings
- **Date & Platform:** 2026-10-27 | Windows 11 (AMD64)
- **Python Version:** 3.14.7 | Pip 26.2.1
- **OpenCV Version:** `5.0.0` (`opencv-python==5.0.0.93` verified via `cv2.__version__`)
- **Key Dependencies Installed:**
  - `streamlit==1.65.0`
  - `numpy==2.5.3`
  - `pillow==12.3.0`
  - `scipy==1.18.1`
  - `pytest==9.1.1`
- **Application Entry Point:** `app.py` executed via `streamlit run app.py`
- **Current Test Status:** 15 unit tests passing in 0.52s (`tests/test_validation.py`, `tests/test_analysis.py`, `tests/test_report.py`, `tests/test_synthetic.py`, `tests/test_app.py`).

## 2. Component-by-Component Audit

| Component | Current State | Strengths | Gaps & Next Steps |
| :--- | :--- | :--- | :--- |
| **Input Validation** (`validation.py`) | Implemented & Tested | Magic byte inspection, 15MB file cap, 4096px dimension cap, in-memory decoding. | Add video frame / sequence decoding utility. |
| **OpenCV 5 Core Pipeline** (`analysis.py`) | Implemented & Tested | Laplacian variance, RMS contrast, mean/median brightness, Canny edge density, Sobel gradients, contour extraction. | Add local grid sharpness, regional exposure masks for targeted operations. |
| **Candidate Extraction & Visualization** (`visualization.py`) | Implemented & Tested | Bounding box rendering with adaptive scaling, context-padded crop extraction, in-memory PNG encoding. | Add side-by-side comparison stitching and crop reinspection overlays. |
| **Report Serialization** (`report.py`) | Implemented & Tested | Schema version 1.0.0, system metadata, JSON export, disclaimer banner. | Extend schema with Agent Action Trace and multi-step findings. |
| **Agentic Vision Engine** (`agent.py`) | **Missing (Need to Implement)** | Initial single-pass rule check exists. | **Priority P1:** Build bounded, auditable decision engine (`VisionAgent`) that runs follow-up operations based on preliminary metrics. |
| **Evaluation Suite** (`evaluate.py`) | Implemented | 6 deterministic synthetic categories, latency measurement, threshold sweep. | Add classification metrics (Precision, Recall, F1, Confusion Matrix) on labeled synthetic benchmark. |
| **User Interface** (`app.py`) | Implemented & Clean | Minimalist design, metric cards, synthetic selector, JSON/PNG export. | Add visible Agent Action Timeline, crop comparison panel, video frame tab. |
| **Cloud & Deployment** | Scaffolded | `.gitignore` and `.dockerignore` present. | Create `Dockerfile`, `.env.example`, `docs/DEPLOYMENT_AWS.md`, mark live AWS as BLOCKED/NOT RUN. |
| **Documentation & Compliance** | Initial worklog created | `docs/WORKLOG.md` active. | Complete `README.md`, `docs/ARCHITECTURE.md`, `docs/EVALUATION.md`, `docs/SECURITY.md`, `docs/DEMO_SCRIPT.md`, `docs/SUBMISSION_CHECKLIST.md`, `FINAL_VERIFICATION.md`. |

## 3. Prioritized Implementation Plan

- **Phase 1 (P1): Genuine Agentic Vision Engine (`visionguard/agent.py`)**
  - Implement `VisionAgent` with a tool-based architecture:
    - Step 1: Global survey (`analyze_global_image`).
    - Step 2: Anomaly / ambiguity detection logic.
    - Step 3: Targeted tool execution (e.g. `reinspect_candidate_crop`, `compute_local_sharpness_grid`, `inspect_regional_exposure`).
    - Step 4: Evidence comparison against global baseline.
    - Step 5: Final auditable verdict with step-by-step reasoning log.
- **Phase 2 (P1): Extended OpenCV Operations & Video Frame Support**
  - Add `compute_local_sharpness_grid` (measuring Laplacian variance across a 4x4 spatial grid).
  - Add video frame sampler in `visionguard/video.py` for processing video clips (MP4/AVI) frame-by-frame.
- **Phase 3 (P2): Evaluation Benchmark & Confusion Matrix**
  - Create a labeled synthetic evaluation benchmark with 20 test assets across 4 classes: Sharp-Nominal, Blurred-Defect, Exposure-Abnormal, Surface-Defect.
  - Calculate true positive, false positive, true negative, false negative, precision, recall, and F1 score.
- **Phase 4 (P3): Streamlit UI Enhancement**
  - Integrate Agent Action Timeline showing each step taken, rationale, and tool outputs.
  - Add crop reinspection comparison view.
- **Phase 5 (P4 & P5): Cloud Artifacts, Security, and Final Documentation**
  - Create `Dockerfile`, `.env.example`, `docs/DEPLOYMENT_AWS.md`.
  - Create `README.md`, `docs/ARCHITECTURE.md`, `docs/EVALUATION.md`, `docs/SECURITY.md`, `docs/DEMO_SCRIPT.md`, `docs/SUBMISSION_CHECKLIST.md`.
  - Execute full test suite and compile `FINAL_VERIFICATION.md`.

