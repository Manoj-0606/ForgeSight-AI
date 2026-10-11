"""Headless end-to-end check: run the full pipeline on a video and verify the outputs.

Usage:  python scripts/smoke_test.py path/to/video.mp4 [--seconds 20] [--fps 6]
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore", category=FutureWarning)

import cv2  # noqa: E402

from forgesight.config import Settings, ensure_dirs  # noqa: E402
from forgesight.detector import load_model  # noqa: E402
from forgesight.pipeline import analyze_video  # noqa: E402
from forgesight.results import load_run  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--seconds", type=int, default=20)
    ap.add_argument("--fps", type=float, default=6.0)
    ap.add_argument("--no-tracking", action="store_true")
    ap.add_argument("--ppe", action="store_true", help="enable the PPE beta (downloads its model once)")
    ap.add_argument("--hazard", action="store_true", help="enable the fire/smoke beta")
    ap.add_argument("--email", action="store_true", help="send alert emails using FORGESIGHT_SMTP_* env vars")
    ap.add_argument("--zone", help="zone polygon in %% of frame, e.g. \"0,75;30,75;30,100;0,100\"")
    args = ap.parse_args()

    ensure_dirs()
    settings = Settings(max_seconds=args.seconds, analysis_fps=args.fps,
                        tracking_enabled=not args.no_tracking, ppe_enabled=args.ppe)
    if args.zone:
        settings.zone_points = [[float(v) for v in pt.split(",")] for pt in args.zone.split(";")]
        settings.zone_name = "Test zone"
    t0 = time.perf_counter()
    model = load_model(settings.model_name)
    print(f"model loaded in {time.perf_counter() - t0:.1f}s")

    last = [0.0]

    def progress(p):
        if time.perf_counter() - last[0] > 2:
            last[0] = time.perf_counter()
            print(f"  {p.done}/{p.total} frames  t={p.t:.1f}s  people={p.people}  events={p.events}")

    ppe_model = None
    if args.ppe:
        from forgesight.ppe import load_ppe_model
        ppe_model = load_ppe_model()
    hazard_model = dispatcher = None
    if args.hazard:
        from forgesight.hazard import load_hazard_model
        from forgesight.notify import AlertDispatcher, load_email_config
        settings.hazard_enabled, settings.email_enabled = True, args.email
        hazard_model = load_hazard_model()
        cfg, msg = load_email_config()
        print("email:", msg)
        dispatcher = AlertDispatcher(cfg, enabled=args.email, cooldown_s=settings.email_cooldown_s,
                                     max_per_run=settings.email_max_per_run)
    run_dir = analyze_video(args.video, settings, model, on_progress=progress, ppe_model=ppe_model,
                            hazard_model=hazard_model, alert_dispatcher=dispatcher)
    run = load_run(run_dir)
    s = run.summary
    print("run:", run_dir)
    print("processing:", s["processing"])
    print("tracking:", s["tracking"])
    print("metrics:", s["metrics"])
    if run.ppe_active:
        print("ppe:", s["ppe"]["counts"], "check frames:", s["ppe"]["check_frames"])
    if run.hazard_active:
        print("hazard:", {k: s["hazard"][k] for k in ("checks", "events")}, "email:", s["hazard"]["email"]["outcomes"])
        if run.hazards is not None and len(run.hazards):
            print(run.hazards[["event_id", "start_time_s", "confirmed_time_s", "end_time_s", "max_score",
                               "email_status"]].to_string(index=False))

    failures = []
    cap = cv2.VideoCapture(str(run.video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if n != s["processing"]["frames_analyzed"]:
        failures.append(f"annotated video has {n} frames, expected {s['processing']['frames_analyzed']}")
    if run.frames["people"].sum() != s["metrics"]["total_detections"]:
        failures.append("total_detections does not match frames.csv")
    for rel in run.events["evidence_image"].tolist():
        img = cv2.imread(str(run.path(rel)))
        if img is None:
            failures.append(f"evidence image unreadable: {rel}")
    if len(run.events) and run.tracking_active:
        dup = run.events.groupby("track_id")["entry_time_s"].apply(lambda x: x.sort_values().diff().min())
        print("min gap between repeat entries per track:", dup.dropna().to_dict())
    print("events:\n", run.events[["event_id", "track_id", "entry_time_s", "exit_time_s", "dwell_s",
                                   "exit_reason"]].to_string(index=False) if len(run.events) else "none")
    if failures:
        print("FAIL:", *failures, sep="\n  ")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
