"""Paths, default settings and settings persistence."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
RUNS_DIR = OUTPUT_DIR / "runs"
UPLOADS_DIR = OUTPUT_DIR / "uploads"
SETTINGS_FILE = OUTPUT_DIR / "settings.json"

SUPPORTED_EXTENSIONS = ("mp4", "mov", "avi", "mkv", "m4v")
MAX_RECOMMENDED_SECONDS = 300

MODEL_CHOICES = {
    "yolo11n.onnx": "YOLO11n — fastest (recommended for CPU)",
    "yolo11s.onnx": "YOLO11s — more accurate, ~2–3× slower",
}

# Zone presets in percent of frame width/height, clockwise from top-left.
ZONE_PRESETS: dict[str, list[list[float]]] = {
    "Right-side robot cell": [[62, 35], [96, 35], [96, 92], [58, 92]],
    "Left-side machinery area": [[4, 35], [38, 35], [42, 92], [4, 92]],
    "Center aisle crossing": [[38, 45], [62, 45], [66, 95], [34, 95]],
    "Lower-half hazard band": [[5, 60], [95, 60], [95, 97], [5, 97]],
}
DEFAULT_ZONE_PRESET = "Right-side robot cell"


@dataclass
class Settings:
    # Detection
    model_name: str = "yolo11n.onnx"
    confidence: float = 0.35
    analysis_fps: float = 6.0
    max_width: int = 1280
    max_seconds: int = 0  # 0 = analyze the whole video
    # Tracking
    tracking_enabled: bool = True
    min_track_frames: int = 3  # tracks shorter than this are not counted as unique people
    # Restricted zone
    zone_enabled: bool = True
    zone_name: str = "Robot Cell 3"
    zone_points: list[list[float]] = field(
        default_factory=lambda: [p[:] for p in ZONE_PRESETS[DEFAULT_ZONE_PRESET]]
    )
    entry_frames: int = 2  # consecutive analyzed frames inside before an entry is confirmed
    exit_frames: int = 3  # consecutive analyzed frames outside before an exit is confirmed
    # Analytics (applied to saved runs instantly; no re-analysis needed)
    trend_window_s: float = 2.0
    occ_alert_enabled: bool = True
    occ_alert_scope: str = "zone"  # "zone" or "frame"
    occ_alert_threshold: int = 3
    occ_alert_min_s: float = 3.0
    # Severity rules (transparent, configurable)
    sev_critical_dwell_s: float = 5.0  # intrusion dwell ≥ this → Critical
    sev_brief_dwell_s: float = 0.5  # intrusion dwell < this → Medium (brief boundary incursion)
    # PPE (beta, optional; requires tracking)
    ppe_enabled: bool = False
    ppe_items: list[str] = field(default_factory=lambda: ["hardhat", "vest"])
    ppe_confidence: float = 0.50
    ppe_rate_fps: float = 2.0
    # Fire / smoke (beta, optional)
    hazard_enabled: bool = False
    hazard_classes: list[str] = field(default_factory=lambda: ["fire"])  # smoke: low recall, opt-in only
    hazard_threshold: float = 0.45  # minimum class score for a positive check
    hazard_rate_fps: float = 2.0  # checks per second of video
    hazard_confirm_checks: int = 3  # consecutive positive checks before an event is confirmed
    hazard_clear_checks: int = 4  # consecutive negative checks before an episode closes
    # Email alerts for confirmed fire/smoke events. Credentials never live here: environment variables or
    # .streamlit/secrets.toml only (see forgesight/notify.py).
    email_enabled: bool = False
    email_cooldown_s: float = 120.0  # per hazard type, in video time
    email_max_per_run: int = 3
    # Presentation
    camera_label: str = "CAM-01 · Body Shop Line 2"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Settings":
        known = {f.name for f in fields(cls)}
        s = cls(**{k: v for k, v in data.items() if k in known})
        if s.model_name not in MODEL_CHOICES:
            s.model_name = cls.model_name
        return s


def ensure_dirs() -> None:
    for d in (MODELS_DIR, RUNS_DIR, UPLOADS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def load_settings() -> Settings:
    try:
        return Settings.from_dict(json.loads(SETTINGS_FILE.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError):
        return Settings()


def save_settings(settings: Settings) -> None:
    ensure_dirs()
    SETTINGS_FILE.write_text(json.dumps(settings.to_dict(), indent=2), encoding="utf-8")
