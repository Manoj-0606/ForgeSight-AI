"""Validate the fire/smoke classifier shipped with ForgeSight on labelled images and on videos.

Images (YOLO-format split; default convention D-Fire: 0=smoke, 1=fire): image-level precision/recall per class
and the false-alarm rate on images with neither fire nor smoke.
Videos: share of checks above threshold and confirmed episodes (consecutive-check rule), as in the app.

D-Fire: https://github.com/gaiasd/DFireDataset (CC BY 4.0); mirror huggingface.co/datasets/hanvithSai/gavin-dfire-raw
usage:  python scripts/validate_fire.py --images <split_dir_with_images_and_labels> [--limit 1500]
        python scripts/validate_fire.py --videos clip1.mp4 clip2.mp4 [--threshold 0.45]
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from forgesight.hazard import HazardMonitor, load_hazard_model  # noqa: E402

THRESHOLDS = (0.40, 0.45, 0.50, 0.55)


def eval_images(model, root: Path, limit: int) -> None:
    imgs = sorted(p for p in root.rglob("*.jpg") if not p.name.startswith("._"))
    random.Random(0).shuffle(imgs)
    imgs = imgs[:limit] if limit else imgs
    rows, t0 = [], time.perf_counter()
    for p in imgs:
        fr = cv2.imread(str(p))
        if fr is None:
            continue
        sc = model.scores(fr)
        lab = p.parent.parent / "labels" / f"{p.stem}.txt"
        cls = {ln.split()[0] for ln in lab.read_text(errors="ignore").splitlines() if ln.strip()} if lab.exists() else set()
        rows.append({"fire": "1" in cls, "smoke": "0" in cls, **sc})
    print(f"{len(rows)} images (seeded sample), {(time.perf_counter() - t0) / max(len(rows), 1) * 1000:.0f} ms/image")
    for c in ("fire", "smoke"):
        gt = np.array([r[c] for r in rows])
        sc = np.array([r[c] if c in r else 0.0 for r in rows])
        for t in THRESHOLDS:
            pp = sc >= t
            tp = int((pp & gt).sum())
            print(f"  {c:5s} p>={t:.2f}: precision {tp / max(pp.sum(), 1):.2f}  recall {tp / max(gt.sum(), 1):.2f}"
                  f"  (positive images {int(gt.sum())})")
    neg = [r for r in rows if not r["fire"] and not r["smoke"]]
    for t in THRESHOLDS:
        fa = np.mean([r["fire"] >= t for r in neg]) if neg else 0
        print(f"  false alarms (fire) on {len(neg)} images without fire/smoke, p>={t:.2f}: {fa:.1%}")


def eval_videos(model, paths: list[str], threshold: float, rate: float, confirm: int, clear: int) -> None:
    for path in paths:
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        step = max(1, int(round(fps / rate)))
        mon, n, hits, i = HazardMonitor(confirm, clear, ["fire"]), 0, 0, 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if i % step == 0:
                p = model.scores(fr)["fire"]
                n += 1
                hits += p >= threshold
                mon.update(i / fps, {"fire": p if p >= threshold else 0.0})
            i += 1
        mon.finish()
        eps = [f"{e.start_t:.1f}-{e.end_t:.1f}s (max {e.max_conf:.2f})" for e in mon.episodes]
        print(f"{Path(path).name}: {n} checks, fire p>={threshold} in {hits / max(n, 1):.0%}; "
              f"confirmed fire episodes: {eps or 'none'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images")
    ap.add_argument("--limit", type=int, default=1500)
    ap.add_argument("--videos", nargs="*")
    ap.add_argument("--threshold", type=float, default=0.45)
    ap.add_argument("--rate", type=float, default=2.0)
    ap.add_argument("--confirm", type=int, default=3)
    ap.add_argument("--clear", type=int, default=4)
    a = ap.parse_args()
    model = load_hazard_model()
    if a.images:
        eval_images(model, Path(a.images), a.limit)
    if a.videos:
        eval_videos(model, a.videos, a.threshold, a.rate, a.confirm, a.clear)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
