# VisionGuard AI — Security Policy & Responsible Operation

## 1. Security Architecture & Threat Model

VisionGuard AI is designed as a secure, local-first decision support tool for visual inspection. The architecture adheres to defensive principles to prevent arbitrary code execution, denial-of-service, and unauthorized data leakage.

### Input Validation & Defensive Ingestion
- **Strict Byte Caps**: Ingestion enforces a hard 15 MB limit (`SafetyLimits.MAX_FILE_BYTES`). Excessive payloads are terminated before memory allocation.
- **Magic Byte Verification**: File extension alone is never trusted. Headers are inspected for explicit magic sequences (`\x89PNG\r\n\x1a\n`, `\xff\xd8\xff`, `RIFF...WEBP`).
- **Geometric Bounds**: Image dimensions are capped at 4096 x 4096 px (16 megapixels max) to eliminate decompression bombs (Zip bombs / pixel bombs).
- **In-Memory Decoding**: Uploaded media are decoded directly in RAM using OpenCV's `imdecode` and are **never** persisted to local disk or remote storage by default.

### Agentic Execution Safety
- **Bounded Step Caps**: The `VisionAgent` enforces a hard limit of `max_steps = 5` and a timeout of 5.0 seconds. Infinite recursion and cyclic tool calls are structurally prevented.
- **Strict Tool Whitelist**: Tools are strictly registered in an internal whitelist (`survey_global_image`, `compute_local_sharpness_grid`, `analyze_regional_exposure`, `reinspect_candidate_crop`, `conclude_inspection`). Unregistered operations raise an immediate `ValueError`.
- **Zero Shell / Eval Execution**: The agent possesses no access to OS shells, `eval()`, `exec()`, or external network endpoints.

### Cloud & Secrets Management
- **Zero Hardcoded Credentials**: No AWS keys, session tokens, passwords, or personal credentials are present in the codebase.
- **Local-First Posture**: The application is fully testable and operational offline without requiring cloud connections or cloud credentials.
- **Non-Privileged Execution**: The Docker container runs under an unprivileged user (`appuser`, UID 1000) with no root permissions.

## 2. Responsible AI & Operational Limitations

### Human-Review-First Guarantee
- **Not a Black-Box Classifier**: VisionGuard AI does not claim autonomous defect classification.
- **Heuristics vs. Truth**: Mathematical measurements (Laplacian variance, intensity distributions, edge densities) are exposed as indicators, not absolute proof of physical defects.
- **Candidate Labeling**: Discontinuities are explicitly labeled as *"Candidate region for review"*, requiring trained human inspector verification.

### Life-Critical Disclaimer
VisionGuard AI is a decision-support prototype developed for research and competitive demonstration. It is **NOT** certified for life-critical, medical diagnosis, aviation, or high-risk industrial safety applications where defect omission could lead to bodily injury or structural failure.

