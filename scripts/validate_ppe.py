"""Person-level validation of the PPE beta against labelled images (end to end, real models).

Builds a slideshow video from a YOLO-format labelled split (each image held for a few seconds), runs the full
ForgeSight pipeline on it (YOLO11 people + ByteTrack + PPE model + multi-check voting), matches each tracked
person to a ground-truth person box, and compares the final per-person status with the labels.

Dataset used for the published numbers: Ultralytics Construction-PPE (test split), classes
0 helmet, 2 vest, 6 Person, 7 no_helmet. Download:
  https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip

usage:  python scripts/validate_ppe.py <dataset_root> [--split test] [--seconds 3] [--out validation_ppe]
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore", category=FutureWarning)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import forgesight.pipeline as pipeline  # noqa: E402
from forgesight import ppe  # noqa: E402
from forgesight.config import Settings  # noqa: E402
from forgesight.detector import load_model  # noqa: E402
from forgesight.results import load_run  # noqa: E402

CANVAS = (1280, 720)
GT = {"helmet": 0, "vest": 2, "person": 6, "no_helmet": 7}


def letterbox(img):
    h, w = img.shape[:2]
    r = min(CANVAS[0] / w, CANVAS[1] / h)
    nw, nh = int(w * r), int(h * r)
    out = np.full((CANVAS[1], CANVAS[0], 3), 20, np.uint8)
    x0, y0 = (CANVAS[0] - nw) // 2, (CANVAS[1] - nh) // 2
    out[y0:y0 + nh, x0:x0 + nw] = cv2.resize(img, (nw, nh))
    return out, r, x0, y0


def load_labels(path, w, h, r, x0, y0):
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            p = line.split()
            if len(p) == 5:
                c, cx, cy, bw, bh = int(p[0]), *map(float, p[1:])
                b = np.array([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
                rows.append((c, b * r + [x0, y0, x0, y0]))
    return rows


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def gt_status(person, labels, item):
    """'wearing', 'not wearing' or None (unknown) from the ground-truth boxes."""
    region = ppe.ITEMS[item].region
    def linked(cls):
        boxes = np.array([b for c, b in labels if c == cls]).reshape(-1, 4)
        return len(boxes) and ppe.associate(person, boxes, np.ones(len(boxes)), region) >= 0
    if item == "hardhat":
        if linked(GT["helmet"]):
            return "wearing"
        return "not wearing" if linked(GT["no_helmet"]) else None
    return "wearing" if linked(GT["vest"]) else "not wearing"  # dataset has no explicit no-vest class


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--split", default="test")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--gap", type=float, default=3.0, help="blank seconds between images so tracks end")
    ap.add_argument("--out", default="validation_ppe")
    args = ap.parse_args()
    root = Path(args.dataset)
    images = sorted((root / "images" / args.split).glob("*.*"))
    fps, hold, gap = 10, int(args.seconds * 10), int(args.gap * 10)
    blank = np.full((CANVAS[1], CANVAS[0], 3), 20, np.uint8)
    period = args.seconds + args.gap

    tmp = Path(tempfile.mkdtemp(prefix="fs_ppe_val_"))
    video = tmp / "slideshow.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), fps, CANVAS)
    segments = []
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        canvas, r, x0, y0 = letterbox(img)
        labels = load_labels(root / "labels" / args.split / f"{img_path.stem}.txt", img.shape[1], img.shape[0],
                             r, x0, y0)
        if not any(c == GT["person"] for c, _ in labels):
            continue
        segments.append((img_path.name, labels))
        for _ in range(hold):
            writer.write(canvas)
        for _ in range(gap):
            writer.write(blank)
    writer.release()
    print(f"slideshow: {len(segments)} images with people, {len(segments) * period:.0f} s")

    runs = tmp / "runs"
    runs.mkdir()
    pipeline.RUNS_DIR = runs
    settings = Settings(analysis_fps=5, zone_enabled=False, ppe_enabled=True, ppe_rate_fps=2.0,
                        max_width=CANVAS[0], camera_label="PPE validation")
    run = load_run(pipeline.analyze_video(video, settings, load_model(settings.model_name),
                                          ppe_model=ppe.load_ppe_model()))
    print(f"pipeline: {run.summary['processing']['elapsed_s']:.0f} s, PPE checks {run.summary['ppe']['check_frames']}")

    rows = []
    res = run.ppe.assign(seg=(run.ppe["first_check_s"] // period).astype(int))
    for seg, (name, labels) in enumerate(segments):
        gt_people = [b for c, b in labels if c == GT["person"]]
        preds = res[res["seg"] == seg]
        for item in ppe.ITEMS:
            p_item = preds[preds["item"] == item]
            boxes = {r.track_id: np.array([r.box_x1, r.box_y1, r.box_x2, r.box_y2]) * [*CANVAS, *CANVAS]
                     for r in p_item.itertuples()}
            used = set()
            for gb in gt_people:
                truth = gt_status(gb, labels, item)
                if truth is None:
                    continue
                best = max(((iou(gb, b), t) for t, b in boxes.items() if t not in used), default=(0, None))
                if best[0] >= 0.5:
                    used.add(best[1])
                    status = p_item[p_item["track_id"] == best[1]].iloc[0]["status"]
                else:
                    status = "no result (person not tracked/judged)"
                rows.append({"image": name, "item": item, "truth": truth, "predicted": status})

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    df.to_csv(out / "person_level_results.csv", index=False)
    lines = []
    for item in ppe.ITEMS:
        t = pd.crosstab(df[df["item"] == item]["truth"], df[df["item"] == item]["predicted"], margins=True)
        lines += [f"\n{ppe.ITEMS[item].label} (rows: ground truth, columns: ForgeSight status)", t.to_string()]
    report = "\n".join(lines)
    (out / "summary.txt").write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
