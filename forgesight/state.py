"""Streamlit session state, cached resources and settings binding."""
from __future__ import annotations

import base64
from pathlib import Path

import cv2
import streamlit as st

from .config import Settings, load_settings, save_settings
from .detector import load_model
from .results import RunResult, load_run

# Page objects registered by app.py so views can link to each other.
PAGES: dict = {}


@st.cache_resource(show_spinner=False)
def get_model(model_name: str):
    """Load the detector once per model per server process."""
    return load_model(model_name)


@st.cache_resource(show_spinner=False)
def get_ppe_model():
    """PPE beta model (downloaded once, SHA-256 verified)."""
    from .ppe import load_ppe_model

    return load_ppe_model()


@st.cache_resource(show_spinner=False)
def get_hazard_model():
    """Fire/smoke beta classifier (downloaded once, SHA-256 verified)."""
    from .hazard import load_hazard_model

    return load_hazard_model()


def email_config():
    """(EmailConfig | None, message). Environment variables first, then .streamlit/secrets.toml [email]."""
    from .notify import load_email_config

    try:
        secrets = dict(st.secrets.get("email", {}))
    except Exception:  # no secrets.toml, or it cannot be parsed
        secrets = {}
    return load_email_config(secrets=secrets or None)


def get_settings() -> Settings:
    if "settings" not in st.session_state:
        st.session_state.settings = load_settings()
    return st.session_state.settings


def update_settings(**changes) -> None:
    s = get_settings()
    for k, v in changes.items():
        setattr(s, k, v)
    save_settings(s)


def bound(widget, label: str, attr: str, key_suffix: str = "", **kwargs):
    """Render a widget whose value is persisted to a Settings attribute.

    The same setting rendered on two pages must use different ``key_suffix`` values: Streamlit cleans up widget
    state per page, and a key shared between pages can come back with the widget's default value, which would
    then be auto-saved (this reset the zone name/toggle in v1.0).
    """
    s = get_settings()
    key = f"set_{attr}" + (f"__{key_suffix}" if key_suffix else "")
    if key not in st.session_state:
        st.session_state[key] = getattr(s, attr)
    value = widget(label, key=key, **kwargs)
    if value != getattr(s, attr):
        update_settings(**{attr: value})
    return value


def reset_settings() -> None:
    st.session_state.settings = Settings()
    save_settings(st.session_state.settings)
    for k in [k for k in st.session_state if str(k).startswith("set_") or str(k).startswith("zone_")]:
        del st.session_state[k]


def get_run() -> RunResult | None:
    return st.session_state.get("run")


def set_run(run_dir: str | Path | None) -> None:
    st.session_state.run = load_run(run_dir) if run_dir else None
    st.session_state.pop("events_table", None)


@st.cache_data(show_spinner=False, max_entries=512)
def image_data_uri(path: str, mtime: float, max_side: int = 220) -> str:
    img = cv2.imread(path)
    if img is None:
        return ""
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else ""


def thumb(path: Path, max_side: int = 220) -> str:
    try:
        return image_data_uri(str(path), path.stat().st_mtime, max_side)
    except OSError:
        return ""
