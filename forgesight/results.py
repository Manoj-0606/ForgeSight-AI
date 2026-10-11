"""Loading saved analysis runs and deriving interval analytics from them."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import events as evt
from .config import RUNS_DIR


@dataclass
class RunResult:
    run_dir: Path
    summary: dict
    frames: pd.DataFrame
    events: pd.DataFrame
    tracks: pd.DataFrame
    heatmap: np.ndarray
    ppe: pd.DataFrame  # empty unless the PPE beta ran
    heatmap_dwell: np.ndarray | None = None  # stationary-presence layer (runs from v1.1 with tracking)
    hazards: pd.DataFrame | None = None  # confirmed fire/smoke events (runs from v1.2 with the beta on)
    hazard_checks: pd.DataFrame | None = None  # per-check fire/smoke scores
    alerts: pd.DataFrame | None = None  # email delivery log

    @property
    def hazard_active(self) -> bool:
        return bool(self.summary.get("hazard", {}).get("active"))

    @property
    def ppe_active(self) -> bool:
        return bool(self.summary.get("ppe", {}).get("active"))

    @property
    def run_id(self) -> str:
        return self.summary["run_id"]

    @property
    def metrics(self) -> dict:
        return self.summary["metrics"]

    @property
    def video_path(self) -> Path:
        return self.run_dir / "annotated.mp4"

    @property
    def background_path(self) -> Path | None:
        p = self.run_dir / "background.jpg"
        return p if p.exists() else None

    @property
    def tracking_active(self) -> bool:
        return bool(self.summary["tracking"]["active"])

    def path(self, rel: str) -> Path:
        return self.run_dir / rel


def _csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path, keep_default_na=True) if path.exists() else None


def load_run(run_dir: str | Path) -> RunResult:
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    events = pd.read_csv(run_dir / "events.csv")
    if "track_id" in events:
        events["track_id"] = events["track_id"].astype("Int64")
    if "entry_context" not in events:  # runs created before this column existed
        events.insert(2, "entry_context", "entered zone")
    frames = pd.read_csv(run_dir / "frames.csv")
    tracks = pd.read_csv(run_dir / "tracks.csv")
    summary, frames, events = evt.upgrade_legacy_run(summary, frames, events, tracks)
    if "dwell_is_lower_bound" in events:
        events["dwell_is_lower_bound"] = events["dwell_is_lower_bound"].fillna(False).astype(bool)
    return RunResult(
        run_dir=run_dir,
        summary=summary,
        frames=frames,
        events=events,
        tracks=tracks,
        heatmap=np.load(run_dir / "heatmap.npy"),
        ppe=pd.read_csv(run_dir / "ppe.csv", keep_default_na=True) if (run_dir / "ppe.csv").exists()
        else pd.DataFrame(),
        heatmap_dwell=np.load(run_dir / "heatmap_dwell.npy") if (run_dir / "heatmap_dwell.npy").exists() else None,
        hazards=_csv(run_dir / "hazards.csv"),
        hazard_checks=_csv(run_dir / "hazard_checks.csv"),
        alerts=_csv(run_dir / "alerts.csv"),
    )


def list_runs() -> list[dict]:
    """Completed runs, newest first, with a light summary for selection lists."""
    runs = []
    if not RUNS_DIR.exists():
        return runs
    for d in RUNS_DIR.iterdir():
        f = d / "summary.json"
        if d.is_dir() and not d.name.endswith(".partial") and f.exists():
            try:
                s = json.loads(f.read_text(encoding="utf-8"))
                runs.append({"run_dir": d, "run_id": s["run_id"], "name": s["original_name"],
                             "created_at": s["created_at"], "events": s["metrics"]["zone_entries"]})
            except (OSError, ValueError, KeyError):
                continue
    return sorted(runs, key=lambda r: r["created_at"], reverse=True)


def choose_bin_seconds(duration_s: float, target_bins: int = 24) -> int:
    for b in (1, 2, 5, 10, 15, 30, 60, 120, 300, 600):
        if duration_s / b <= target_bins:
            return b
    return 900


def interval_activity(frames: pd.DataFrame, events: pd.DataFrame, bin_s: int) -> pd.DataFrame:
    """Per time interval: detections, mean/peak occupancy and zone entries."""
    if frames.empty:
        return pd.DataFrame(columns=["bin_start_s", "label", "detections", "avg_occupancy",
                                     "peak_occupancy", "zone_entries"])
    f = frames.assign(bin=(frames["time_s"] // bin_s).astype(int))
    g = f.groupby("bin").agg(detections=("people", "sum"), avg_occupancy=("people", "mean"),
                             peak_occupancy=("people", "max"))
    last_bin = int(f["bin"].max())
    g = g.reindex(range(0, last_bin + 1), fill_value=0)
    if "event_type" in events:
        events = evt.intrusions(events)  # people present at the start are not entries
    if not events.empty:
        ev_bins = (events["entry_time_s"] // bin_s).astype(int).value_counts()
        g["zone_entries"] = ev_bins.reindex(g.index, fill_value=0)
    else:
        g["zone_entries"] = 0
    g = g.reset_index().rename(columns={"bin": "bin_index"})
    g["bin_start_s"] = g["bin_index"] * bin_s
    g["label"] = g["bin_start_s"].map(lambda s: f"{int(s // 60):02d}:{int(s % 60):02d}")
    g["avg_occupancy"] = g["avg_occupancy"].round(2)
    return g


def events_export(run: RunResult) -> pd.DataFrame:
    """Event log formatted for CSV export."""
    cols = ["event_id", "event_type", "entry_context", "zone_name", "track_id", "entry_time_s", "exit_time_s",
            "dwell_s", "dwell_is_lower_bound", "exit_reason", "frame_index", "confidence", "people_in_zone",
            "evidence_image", "video_reference"]
    df = run.events.reindex(columns=cols).copy() if not run.events.empty else pd.DataFrame(columns=cols)
    df.insert(0, "source_video", run.summary["original_name"])
    df.insert(1, "camera", run.summary["camera_label"])
    df.insert(2, "run_id", run.run_id)
    df["evidence_image"] = df["evidence_image"].map(lambda p: str(run.run_dir / p) if isinstance(p, str) else "")
    return df
