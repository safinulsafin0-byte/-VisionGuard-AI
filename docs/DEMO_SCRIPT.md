# VisionGuard AI — Live Demonstration Script (2–4 Minutes)

This script provides an exact, reproducible demonstration narrative for video recording, live competition pitching, or judge evaluation.

---

## Act 1: Introduction & Vision (0:00 – 0:45)

- **Speaker Action:** Open browser to `http://localhost:8501`. Display clean VisionGuard AI header and sidebar.
- **Narrative:**
  > *"Welcome to VisionGuard AI — an explainable visual inspection assistant powered by OpenCV 5 and an evidence-driven vision agent. In industrial quality control, automated black-box classifiers often fail silently or overwhelm operators with opaque alarms. VisionGuard AI takes a human-review-first approach: computing deterministic image-quality metrics, dynamically dispatching targeted micro-inspections, and providing fully auditable action traces."*
- **Visible on Screen:**
  - Sidebar showing `OpenCV Engine: v5.0.0` and Python runtime.
  - Initial clean state prompting for image upload or benchmark selection.

---

## Act 2: Baseline Benchmark & The VisionAgent Feedback Loop (0:45 – 2:00)

- **Speaker Action:** In the sidebar, select **"Load Synthetic Benchmark"** -> Choose `surface_with_artifact`.
- **Narrative:**
  > *"Let's test an industrial brushed metal surface with an intentional geometric scratch anomaly. Within 7 milliseconds, our VisionAgent executes a multi-step inspection workflow."*
- **Visible on Screen:**
  - **Verdict Card:** Displays `ANOMALOUS CANDIDATE FLAGGED` with `HIGH` confidence.
  - **Side-by-Side Images:** Original surface on the left; annotated candidate bounding box with `#1 Candidate` badge on the right.
  - **Quality Metrics:** Laplacian variance (182.3), RMS contrast (13.6), mean intensity (158.4), and Canny edge density.
  - **Agent Decision Timeline:** Point out the 3 sequential steps taken:
    1. `survey_global_image`: Measured baseline metrics across the image.
    2. `reinspect_candidate_crop`: Detected candidate region #1, extracted high-resolution crop, and calculated local contrast ratio.
    3. `conclude_inspection`: Verified that the region exhibits salient edge contrast relative to background, prioritizing it for human review.

---

## Act 3: Candidate Crop Reinspection & Deep Examination (2:00 – 2:45)

- **Speaker Action:** Scroll down to the **"Candidate Region Inspector & Reinspection"** section.
- **Narrative:**
  > *"Rather than relying on low-resolution global thumbnails, the inspector can zoom directly into candidate bounding boxes. The crop inspector displays exact pixel coordinates, aspect ratio, internal edge density, and context padding. Notice our explicit disclaimer: this is flagged as a candidate for human review, not an autonomous verdict."*
- **Visible on Screen:**
  - High-resolution zoomed crop of the scratch anomaly.
  - Metrics breakdown: Bounding box coordinates, contour area, internal edge density (5.2%).
  - Diagnostic expander showing raw Canny edges and grayscale conversions.

---

## Act 4: Focus Defect & Audit Export (2:45 – 3:30)

- **Speaker Action:** In the sidebar, switch to `blurred_checkerboard`.
- **Narrative:**
  > *"What happens when camera focus or motion blur degrades capture quality? Let's select a blurred pattern. Immediately, the agent flags an `IMAGE QUALITY DEFECT`. Notice the agent's adaptation: because global sharpness fell below 100.0, the agent executed the `compute_local_sharpness_grid` tool, testing a 4x4 spatial grid to prove that blur was uniform across all quadrants rather than localized to a textureless plane."*
- **Speaker Action:** Click **"Download Complete Audit Report (JSON)"** and **"Download Annotated Image (PNG)"**.
- **Narrative:**
  > *"Every session produces a reproducible JSON audit containing complete environment metadata, raw OpenCV 5 measurements, and the full step-by-step agent action trace. VisionGuard AI delivers explainable, reliable, human-centered vision inspection."*
- **Visible on Screen:**
  - Generated JSON report downloaded to workstation.

