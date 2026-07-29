#!/usr/bin/env python3
"""
End-to-end pipeline demo for the Agricultural Safety AI project.

Exercises the real HarvesterSafetyEngine + HarvesterSafetyVisualizer
on a synthetic video. Uses scripted "detections" (moving humans approaching
the harvester) so the demo runs without the YOLO/torch stack while still
demonstrating the risk-assessment + visualization logic end-to-end.

Outputs:
  - sample_video.mp4              (input synthetic video)
  - annotated_pipeline_output.mp4 (annotated output)
  - pipeline_summary.json         (frame-by-frame risk events)
"""

import json
import time
import logging
import cv2
import numpy as np

from harvester_safety import HarvesterSafetyEngine
from harvester_visualizer import HarvesterSafetyVisualizer
from generate_sample_video import generate_sample_video

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


def scripted_detections(frame_idx, total_frames, w, h):
    """
    Three humans moving on different trajectories. As the index grows they
    approach the harvester position (centered at the bottom), so the risk
    levels should escalate from SAFE -> LOW_WARNING -> ... -> CRITICAL.
    """
    t = frame_idx / max(total_frames - 1, 1)

    # Person 1: walking diagonally toward the harvester from the upper-left
    p1_x = int(80 + t * 220)
    p1_y = int(80 + t * 280)
    p1 = ((p1_x, p1_y, p1_x + 50, p1_y + 110), 0.92)

    # Person 2: drifting in from the right
    p2_x = int(550 - t * 160)
    p2_y = int(150 + t * 180)
    p2 = ((p2_x, p2_y, p2_x + 45, p2_y + 100), 0.88)

    # Person 3: stationary far in the field (should stay SAFE)
    p3 = ((300, 90, 320, 130), 0.81)

    return [p1, p2, p3]


def main():
    video_in = "sample_video.mp4"
    video_out = "annotated_pipeline_output.mp4"
    summary_out = "pipeline_summary.json"

    # 1) Build the input video using the project's helper.
    log.info("Generating synthetic input video ...")
    generate_sample_video(video_in, num_frames=30, width=640, height=480)

    # 2) Wire up the actual safety + visualization modules.
    safety = HarvesterSafetyEngine()
    viz = HarvesterSafetyVisualizer()

    cap = cv2.VideoCapture(video_in)
    if not cap.isOpened():
        raise SystemExit(f"could not open {video_in}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_out, fourcc, fps, (w, h))

    risk_counts = {"CRITICAL": 0, "HIGH_WARNING": 0, "WARNING": 0, "LOW_WARNING": 0, "SAFE": 0}
    per_frame = []
    latencies_ms = []

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t0 = time.perf_counter()

        detections = scripted_detections(frame_idx, total, w, h)
        risks = []
        for bbox, _conf in detections:
            risks.append(safety.compute_risk_level(bbox, frame.shape))

        zones = safety.get_danger_zones_visualization(frame.shape)
        annotated = viz.annotate_frame(frame, detections, risks, zones)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(latency_ms)

        for r in risks:
            risk_counts[r["risk_level"]] = risk_counts.get(r["risk_level"], 0) + 1

        per_frame.append(
            {
                "frame": frame_idx,
                "latency_ms": round(latency_ms, 2),
                "risks": [
                    {"level": r["risk_level"], "score": round(float(r.get("risk_score", 0)), 3)}
                    for r in risks
                ],
            }
        )

        writer.write(annotated)
        frame_idx += 1

    cap.release()
    writer.release()

    summary = {
        "frames_processed": frame_idx,
        "input_video": video_in,
        "output_video": video_out,
        "risk_level_counts": risk_counts,
        "avg_latency_ms": round(float(np.mean(latencies_ms)), 2) if latencies_ms else 0.0,
        "p95_latency_ms": round(float(np.percentile(latencies_ms, 95)), 2) if latencies_ms else 0.0,
    }
    with open(summary_out, "w") as f:
        json.dump({"summary": summary, "per_frame": per_frame}, f, indent=2)

    print()
    print("=" * 60)
    print("AGRICULTURAL SAFETY AI - PIPELINE DEMO")
    print("=" * 60)
    print(f"Frames processed     : {summary['frames_processed']}")
    print(f"Avg latency / frame  : {summary['avg_latency_ms']} ms")
    print(f"P95 latency / frame  : {summary['p95_latency_ms']} ms")
    print("Risk-level event counts (across all detections):")
    for k in ("CRITICAL", "HIGH_WARNING", "WARNING", "LOW_WARNING", "SAFE"):
        print(f"  {k:13s}: {risk_counts.get(k, 0)}")
    print(f"\nOutputs:\n  {video_out}\n  {summary_out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
