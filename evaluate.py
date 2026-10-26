"""VisionGuard AI — Comprehensive and Reproducible Evaluation Harness.

Executes controlled, evidence-based benchmark evaluation:
- Labeled synthetic test suite with ground-truth classifications
- Quantitative OpenCV 5 metrics and processing latencies
- Confusion matrix, precision, recall, and F1 calculations
- Repeatability / determinism verification across duplicate runs
- Defensive error rejection verification on malformed/boundary inputs
"""

from datetime import datetime, timezone
import json
import os
import platform
import time
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from visionguard.agent import VisionAgent
from visionguard.analysis import analyze_image
from visionguard.config import InspectionThresholds
from visionguard.synthetic_data import (
    generate_blurred_checkerboard,
    generate_low_contrast_flat,
    generate_overexposed_pattern,
    generate_sharp_checkerboard,
    generate_surface_with_artifact,
    generate_underexposed_pattern,
)
from visionguard.validation import ImageValidationError, validate_and_decode_image


def generate_labeled_benchmark_suite() -> List[Dict[str, Any]]:
    """Build a deterministic, labeled synthetic benchmark callset."""
    items = []

    # 1. Nominal Sharp Controls (6 items)
    for i in range(6):
        seed = 100 + i
        sq_size = 16 + (i * 8)
        img = generate_sharp_checkerboard(width=384, height=384, square_size=sq_size)
        items.append({
            "asset_id": f"NOMINAL_SHARP_{i+1:02d}",
            "ground_truth_class": "NOMINAL",
            "image": img,
            "description": f"Sharp high-contrast checkerboard (square size {sq_size}px)",
        })

    # 2. Defocus / Motion Blur Quality Defects (6 items)
    for i in range(6):
        kernel = 15 + (i * 4)  # 15, 19, 23, 27, 31, 35
        img = generate_blurred_checkerboard(width=384, height=384, blur_kernel=kernel)
        items.append({
            "asset_id": f"BLUR_DEFECT_{i+1:02d}",
            "ground_truth_class": "IMAGE_QUALITY_DEFECT",
            "image": img,
            "description": f"Gaussian-blurred pattern (kernel {kernel}x{kernel})",
        })

    # 3. Illumination Quality Defects (6 items: 3 under, 3 over)
    for i in range(3):
        mean_tgt = 15 + (i * 5)
        img = generate_underexposed_pattern(width=384, height=384, mean_target=mean_tgt, seed=200+i)
        items.append({
            "asset_id": f"UNDEREXP_DEFECT_{i+1:02d}",
            "ground_truth_class": "IMAGE_QUALITY_DEFECT",
            "image": img,
            "description": f"Low-illumination surface (target mean ~{mean_tgt})",
        })
    for i in range(3):
        mean_tgt = 230 + (i * 5)
        img = generate_overexposed_pattern(width=384, height=384, mean_target=mean_tgt, seed=300+i)
        items.append({
            "asset_id": f"OVEREXP_DEFECT_{i+1:02d}",
            "ground_truth_class": "IMAGE_QUALITY_DEFECT",
            "image": img,
            "description": f"High-illumination saturated surface (target mean ~{mean_tgt})",
        })

    # 4. Surface Discontinuity / Candidate Anomalies (6 items)
    for i in range(6):
        img, coords = generate_surface_with_artifact(width=384, height=384, seed=400+i)
        items.append({
            "asset_id": f"SURFACE_ANOMALY_{i+1:02d}",
            "ground_truth_class": "ANOMALOUS_CANDIDATE",
            "image": img,
            "ground_truth_box": coords,
            "description": "Brushed surface with intentional geometric scratch artifact",
        })

    return items


def run_benchmark_evaluation() -> Dict[str, Any]:
    """Execute evaluation across the benchmark suite and compute formal statistics."""
    agent = VisionAgent()
    benchmark_items = generate_labeled_benchmark_suite()

    eval_records: List[Dict[str, Any]] = []
    latencies_ms: List[float] = []

    # Confusion matrix counters for Binary Image Quality Defect Detection
    # Positive = IMAGE_QUALITY_DEFECT, Negative = NOMINAL or ANOMALOUS_CANDIDATE
    tp_quality = 0
    fp_quality = 0
    tn_quality = 0
    fn_quality = 0

    # Confusion matrix counters for Candidate Anomaly Detection
    # Positive = ANOMALOUS_CANDIDATE, Negative = NOMINAL
    tp_anomaly = 0
    fp_anomaly = 0
    tn_anomaly = 0
    fn_anomaly = 0

    for item in benchmark_items:
        img = item["image"]
        gt_class = item["ground_truth_class"]

        t0 = time.perf_counter()
        session = agent.inspect(img)
        lat = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(lat)

        verdict_type = session.final_verdict.verdict_type
        m = session.initial_result.metrics

        # Quality defect evaluation
        is_gt_quality_defect = (gt_class == "IMAGE_QUALITY_DEFECT")
        is_pred_quality_defect = (verdict_type == "IMAGE_QUALITY_DEFECT")
        if is_gt_quality_defect and is_pred_quality_defect:
            tp_quality += 1
        elif not is_gt_quality_defect and is_pred_quality_defect:
            fp_quality += 1
        elif not is_gt_quality_defect and not is_pred_quality_defect:
            tn_quality += 1
        elif is_gt_quality_defect and not is_pred_quality_defect:
            fn_quality += 1

        # Anomaly candidate detection evaluation
        if gt_class in ("ANOMALOUS_CANDIDATE", "NOMINAL"):
            is_gt_anomaly = (gt_class == "ANOMALOUS_CANDIDATE")
            is_pred_anomaly = (verdict_type == "ANOMALOUS_CANDIDATE_FLAGGED")
            if is_gt_anomaly and is_pred_anomaly:
                tp_anomaly += 1
            elif not is_gt_anomaly and is_pred_anomaly:
                fp_anomaly += 1
            elif not is_gt_anomaly and not is_pred_anomaly:
                tn_anomaly += 1
            elif is_gt_anomaly and not is_pred_anomaly:
                fn_anomaly += 1

        eval_records.append({
            "asset_id": item["asset_id"],
            "ground_truth": gt_class,
            "predicted_verdict": verdict_type,
            "confidence": session.final_verdict.confidence,
            "steps_taken": session.total_steps,
            "latency_ms": round(lat, 2),
            "laplacian_var": round(m.laplacian_variance, 2),
            "mean_brightness": round(m.mean_brightness, 2),
            "rms_contrast": round(m.rms_contrast, 2),
            "candidates_found": len(session.initial_result.candidate_regions),
        })

    def calc_metrics(tp: int, fp: int, tn: int, fn: int) -> Dict[str, float]:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        return {
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
        }

    quality_metrics = calc_metrics(tp_quality, fp_quality, tn_quality, fn_quality)
    anomaly_metrics = calc_metrics(tp_anomaly, fp_anomaly, tn_anomaly, fn_anomaly)

    # Repeatability test: run first 5 items twice and verify zero divergence
    divergences = 0
    for item in benchmark_items[:5]:
        s1 = agent.inspect(item["image"])
        s2 = agent.inspect(item["image"])
        if s1.final_verdict.verdict_type != s2.final_verdict.verdict_type:
            divergences += 1
        if s1.initial_result.metrics.laplacian_variance != s2.initial_result.metrics.laplacian_variance:
            divergences += 1

    # Malformed / Boundary input testing
    malformed_tests = [
        {"test": "empty_bytes", "payload": b"", "expected_error": True},
        {"test": "text_payload", "payload": b"NOT AN IMAGE", "expected_error": True},
        {"test": "zero_size", "payload": b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR", "expected_error": True},
    ]
    rejection_count = 0
    for mt in malformed_tests:
        try:
            validate_and_decode_image(mt["payload"])
        except ImageValidationError:
            rejection_count += 1

    report = {
        "evaluation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "opencv_version": str(cv2.__version__),
            "python_version": platform.python_version(),
            "os": f"{platform.system()} {platform.release()}",
            "machine": platform.machine(),
        },
        "benchmark_summary": {
            "total_benchmark_assets": len(benchmark_items),
            "classes_tested": ["NOMINAL", "IMAGE_QUALITY_DEFECT", "ANOMALOUS_CANDIDATE"],
            "latency_stats_ms": {
                "mean": round(float(np.mean(latencies_ms)), 2),
                "median": round(float(np.median(latencies_ms)), 2),
                "min": round(float(np.min(latencies_ms)), 2),
                "max": round(float(np.max(latencies_ms)), 2),
                "p95": round(float(np.percentile(latencies_ms, 95)), 2),
            },
        },
        "quality_defect_classification": quality_metrics,
        "candidate_anomaly_classification": anomaly_metrics,
        "reproducibility_audit": {
            "repeated_run_divergences": divergences,
            "is_fully_deterministic": (divergences == 0),
        },
        "defensive_security_testing": {
            "total_malformed_probes": len(malformed_tests),
            "successfully_rejected_safely": rejection_count,
            "rejection_rate": f"{(rejection_count / len(malformed_tests)) * 100:.1f}%",
        },
        "per_asset_details": eval_records,
    }

    return report


if __name__ == "__main__":
    report_data = run_benchmark_evaluation()
    print(json.dumps(report_data, indent=2))
