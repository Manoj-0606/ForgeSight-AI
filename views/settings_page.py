"""Settings: detection, tracking, restricted zone and processing options."""
from __future__ import annotations

import cv2
import streamlit as st

from forgesight import __version__, ui
from forgesight.analytics import OCC_GAP_TOLERANCE_S, SCOPES
from forgesight.config import MODEL_CHOICES, MODELS_DIR, OUTPUT_DIR
from forgesight.detector import model_path
from forgesight.incidents import severity_rules
from forgesight.ppe import DETECT_RATIO as PPE_DETECT_RATIO
from forgesight.ppe import ITEMS as PPE_ITEMS
from forgesight.ppe import MIN_EVALS as PPE_MIN_EVALS
from forgesight.ppe import MISSING_RATIO as PPE_MISSING_RATIO
from forgesight.ppe import PPE_MODEL_FILE, PPE_MODEL_REPO
from forgesight.ppe import VALIDATION as PPE_VALIDATION
from forgesight.hazard import HAZARD_MODEL_FILE, HAZARD_MODEL_REPO
from forgesight.state import bound, email_config, get_run, get_settings, reset_settings
from views.zone_editor import render_zone_editor


def _send_test_email(cfg) -> None:
    """Send one clearly-labelled test message (with a generated image) and show the outcome."""
    import tempfile
    from pathlib import Path

    import numpy as np

    from forgesight.notify import AlertDispatcher

    img = np.full((360, 640, 3), (26, 18, 11), np.uint8)
    cv2.putText(img, "FORGESIGHT AI - TEST EMAIL", (40, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (238, 211, 34), 2)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "forgesight_test_evidence.jpg"
        cv2.imwrite(str(path), img)
        row = AlertDispatcher(cfg, enabled=True).notify(
            {"event_id": "TEST-000", "hazard": "fire (TEST - not a real event)", "video_time_s": 0.0,
             "confidence": 0.0, "camera": "Settings page", "source_video": "-", "run_id": "-"}, path)
    if row["status"] == "sent":
        st.success(f"Test email sent to {row['recipients']} ({row['duration_s']} s).")
    else:
        st.error(f"Test email failed: {row['detail']}")


def render() -> None:
    s = get_settings()
    ui.page_header("Settings", "Configuration",
                   "Changes are saved automatically and apply to the next analysis.",
                   [ui.chip("Auto-saved", "ok")])

    left, right = st.columns(2, gap="medium")
    with left:
        with ui.card("detect"):
            ui.section("Detection", "YOLO11 · COCO person class")
            bound(st.selectbox, "Model", "model_name", options=list(MODEL_CHOICES),
                  format_func=lambda k: MODEL_CHOICES[k])
            bound(st.slider, "Detection confidence threshold", "confidence", min_value=0.10, max_value=0.90,
                  step=0.05, help="Higher means fewer false detections but more missed people. 0.30–0.45 suits "
                                  "most CCTV footage. With tracking on, an existing track may continue through "
                                  "lower-confidence boxes, which is how ByteTrack handles partial occlusion.")
            bound(st.slider, "Analysis frame rate (frames per second)", "analysis_fps", min_value=1.0,
                  max_value=15.0, step=1.0, format="%.0f fps",
                  help="How many frames per second of video are analyzed. Lower is faster; 5–8 fps is a good "
                       "CPU balance for walking-speed movement.")
    with right:
        with ui.card("process"):
            ui.section("Processing options", "CPU performance")
            bound(st.select_slider, "Max processing width (px)", "max_width", options=[640, 960, 1280, 1920],
                  help="Frames are downscaled to this width for annotation and output. Detection always runs "
                       "at 640 px.")
            bound(st.number_input, "Analyze only the first N seconds (0 = full video)", "max_seconds",
                  min_value=0, max_value=3600, step=10)
            bound(st.text_input, "Camera label (shown on video overlay)", "camera_label", max_chars=48)

    left, right = st.columns(2, gap="medium")
    with left:
        with ui.card("tracking"):
            ui.section("Tracking", "ByteTrack multi-object tracker")
            bound(st.toggle, "Enable ByteTrack tracking", "tracking_enabled",
                  help="Assigns persistent IDs so entries are counted per person and unique people can be "
                       "estimated. Without tracking, zone events are raised per zone occupancy change.")
            bound(st.number_input, "Minimum track length to count as a person (frames)", "min_track_frames",
                  min_value=1, max_value=30, step=1, disabled=not s.tracking_enabled)
    with right:
        with ui.card("logic"):
            ui.section("Zone event logic", "Hysteresis against flicker")
            bound(st.number_input, "Frames inside before an entry is confirmed", "entry_frames",
                  min_value=1, max_value=15, step=1,
                  help="Consecutive analyzed frames with the foot point inside the zone.")
            bound(st.number_input, "Frames outside before an exit is confirmed", "exit_frames",
                  min_value=1, max_value=30, step=1,
                  help="Prevents a brief step out of the zone from creating a duplicate re-entry event.")

    left, right = st.columns(2, gap="medium")
    with left:
        with ui.card("analytics"):
            ui.section("Analytics & occupancy alerts", "Applied to saved analyses instantly")
            bound(st.slider, "Trend smoothing window", "trend_window_s", min_value=0.0, max_value=10.0, step=0.5,
                  format="%.1f s", help="Rolling average drawn over the per-frame counts. 0 shows raw counts only.")
            bound(st.toggle, "Sustained-occupancy alerts", "occ_alert_enabled",
                  help="Raise an alert when occupancy stays at or above a threshold for a minimum time.")
            c1, c2, c3 = st.columns([1.3, 1, 1])
            with c1:
                bound(st.selectbox, "Count", "occ_alert_scope", options=list(SCOPES),
                      format_func=lambda k: "People in zone" if k == "zone" else "People in view",
                      disabled=not s.occ_alert_enabled)
            with c2:
                bound(st.number_input, "At least (people)", "occ_alert_threshold", min_value=1, max_value=200,
                      step=1, disabled=not s.occ_alert_enabled)
            with c3:
                bound(st.number_input, "For at least (s)", "occ_alert_min_s", min_value=0.5, max_value=600.0,
                      step=0.5, disabled=not s.occ_alert_enabled)
            st.caption(f"Dips shorter than {OCC_GAP_TOLERANCE_S:g} s (detection flicker) do not split an alert.")
    with right:
        with ui.card("severity"):
            ui.section("Severity rules", "Transparent · applied at review time")
            c1, c2 = st.columns(2)
            with c1:
                bound(st.number_input, "Critical if intrusion dwell ≥ (s)", "sev_critical_dwell_s",
                      min_value=0.5, max_value=600.0, step=0.5)
            with c2:
                bound(st.number_input, "Brief if intrusion dwell < (s)", "sev_brief_dwell_s",
                      min_value=0.0, max_value=60.0, step=0.1)
            st.markdown(ui.key_values([(rule, sev) for rule, sev in severity_rules(get_settings())]),
                        unsafe_allow_html=True)

    with ui.card("ppe"):
        ui.section("PPE detection · BETA", "Hard hat & hi-vis vest · optional, off by default")
        c1, c2 = st.columns([1, 1], gap="large")
        with c1:
            bound(st.toggle, "Enable PPE checks", "ppe_enabled",
                  help="Runs a second, PPE-specific detector about twice per second and links hard hats / vests to "
                       "tracked people. Requires tracking. Adds roughly 0.1 s of CPU time per check.")
            bound(st.multiselect, "PPE items to check", "ppe_items", options=list(PPE_ITEMS),
                  format_func=lambda k: PPE_ITEMS[k].label, disabled=not s.ppe_enabled)
            bound(st.slider, "PPE detection confidence", "ppe_confidence", min_value=0.30, max_value=0.80,
                  step=0.05, disabled=not s.ppe_enabled,
                  help="0.50 keeps false hard-hat detections rare (see validation).")
            bound(st.slider, "PPE checks per second", "ppe_rate_fps", min_value=0.5, max_value=5.0, step=0.5,
                  format="%.1f /s", disabled=not s.ppe_enabled)
            if s.ppe_enabled and not s.tracking_enabled:
                st.warning("PPE checks need tracking to combine results per person. Enable ByteTrack tracking.")
            mp_ppe = MODELS_DIR / PPE_MODEL_FILE
            st.caption(f"Model: `{PPE_MODEL_REPO}` (pinned revision, SHA-256 verified) · "
                       f"{'downloaded' if mp_ppe.exists() else 'downloads (12 MB) on first PPE analysis'}")
        with c2:
            pl = PPE_VALIDATION["person_level"]
            st.markdown(
                f"""
**How statuses are decided** (per tracked person and item, over ≥ {PPE_MIN_EVALS} clear-view checks)
- **Detected**: item found in ≥ {PPE_DETECT_RATIO:.0%} of checks
- **Not detected (verify)**: found in ≤ {PPE_MISSING_RATIO:.0%} of checks. A lead to verify, not a confirmed violation
- **Uncertain**: too few usable checks, or mixed results

**Measured on labelled construction images** (full pipeline, person level):
"Detected" was correct {pl['hardhat']['detected_precision']:.0%} (hard hat) / {pl['vest']['detected_precision']:.0%} (vest)
of the time; "Not detected" only {pl['hardhat']['not_detected_precision']:.0%} / {pl['vest']['not_detected_precision']:.0%}.
Not validated on automotive footage. See `docs/PPE_VALIDATION.md`.
"""
            )

    left, right = st.columns(2, gap="medium")
    with left:
        with ui.card("hazard"):
            ui.section("Fire & smoke detection · BETA", "Frame classifier · optional, off by default")
            bound(st.toggle, "Enable fire/smoke checks", "hazard_enabled",
                  help="Runs a dedicated fire/smoke image classifier a few times per second. Adds about 0.3 s "
                       "of CPU time per check.")
            bound(st.multiselect, "Classes to monitor", "hazard_classes", options=["fire", "smoke"],
                  format_func=str.capitalize, disabled=not s.hazard_enabled)
            if "smoke" in s.hazard_classes:
                st.warning("Smoke is unreliable with this model (recall 0.08 on labelled test images; dense indoor "
                           "smoke was missed). Do not rely on it.")
            c1, c2 = st.columns(2)
            with c1:
                bound(st.slider, "Score threshold", "hazard_threshold", min_value=0.35, max_value=0.80, step=0.05,
                      disabled=not s.hazard_enabled,
                      help="Minimum class probability for a positive check. 0.45: fire recall 0.72, precision "
                           "0.64 on D-Fire test images; no positives on the automotive sample clips.")
                bound(st.number_input, "Confirm after N consecutive checks", "hazard_confirm_checks",
                      min_value=1, max_value=20, step=1, disabled=not s.hazard_enabled)
            with c2:
                bound(st.slider, "Checks per second", "hazard_rate_fps", min_value=0.5, max_value=5.0, step=0.5,
                      format="%.1f /s", disabled=not s.hazard_enabled)
                bound(st.number_input, "Close episode after N clear checks", "hazard_clear_checks",
                      min_value=1, max_value=30, step=1, disabled=not s.hazard_enabled)
            mp_hz = MODELS_DIR / HAZARD_MODEL_FILE
            st.caption(f"Model: `{HAZARD_MODEL_REPO}` (Apache-2.0, pinned, SHA-256 verified) · "
                       f"{'downloaded' if mp_hz.exists() else 'downloads (87 MB) on first use'}. "
                       "Process flames (furnaces, welding) are detected as fire, too. See "
                       "docs/FIRE_SMOKE_VALIDATION.md.")
    with right:
        with ui.card("email"):
            ui.section("Email alerts", "For confirmed fire/smoke events")
            cfg, cfg_msg = email_config()
            (st.success if cfg else st.info)(cfg_msg, icon=":material/mail:" if cfg else ":material/info:")
            bound(st.toggle, "Send email when fire/smoke is confirmed", "email_enabled",
                  help="One email per confirmed event, with the evidence frame attached. Every attempt is logged "
                       "in the run's alerts.csv: sent, failed (with the error), not configured or suppressed.")
            c1, c2 = st.columns(2)
            with c1:
                bound(st.number_input, "Cooldown per hazard type (s of video)", "email_cooldown_s",
                      min_value=0.0, max_value=3600.0, step=10.0, disabled=not s.email_enabled)
            with c2:
                bound(st.number_input, "Max emails per analysis", "email_max_per_run", min_value=1,
                      max_value=50, step=1, disabled=not s.email_enabled)
            if st.button("Send test email", icon=":material/send:", disabled=cfg is None, width="stretch"):
                _send_test_email(cfg)
            with st.expander("How to configure (credentials never stored in settings)"):
                st.markdown(
                    "Set **environment variables** before starting the app (preferred):\n\n"
                    "```powershell\n$env:FORGESIGHT_SMTP_HOST=\"smtp.office365.com\"; $env:FORGESIGHT_SMTP_PORT=\"587\"\n"
                    "$env:FORGESIGHT_SMTP_USER=\"alerts@company.com\"; $env:FORGESIGHT_SMTP_PASSWORD=\"<app password>\"\n"
                    "$env:FORGESIGHT_ALERT_FROM=\"alerts@company.com\"; $env:FORGESIGHT_ALERT_TO=\"ehs@company.com\"\n"
                    ".venv\\Scripts\\python -m streamlit run app.py\n```\n"
                    "or copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` (git-ignored). "
                    "Restart the app after changing either.\n\n"
                    "`$env:` variables exist only in **that** PowerShell window: start the app from the same window.\n\n"
                    "**Gmail:** host `smtp.gmail.com`, port `587` (STARTTLS) or `465` (SSL), user = the full Gmail "
                    "address, password = a 16-character **App Password** (Google Account > Security > 2-Step "
                    "Verification > App passwords). Your normal Gmail password is rejected with 535 5.7.8.")

    with ui.card("zone"):
        ui.section("Restricted zone", "One polygon · coordinates in % of frame")
        upload = st.session_state.get("upload")
        frame = None
        if upload and not upload.get("error"):
            cached = st.session_state.get("first_frame")
            frame = cached[1] if cached and cached[0] == upload["sha"] else None
            if frame is None:
                from forgesight.video_io import read_first_frame
                frame = read_first_frame(upload["path"], 1.0)
        if frame is None and get_run() and get_run().background_path:
            frame = cv2.imread(str(get_run().background_path))
        render_zone_editor(frame, key="settings", layout="side")

    with ui.card("system"):
        ui.section("System")
        c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
        mp = model_path(s.model_name)
        c1.markdown(ui.key_values([
            ("Version", f"ForgeSight AI {__version__}"),
            ("Inference", "ONNX Runtime · CPU"),
            ("Model file", f"{mp.name} · {'downloaded' if mp.exists() else 'downloads on first analysis'}"),
            ("Output folder", str(OUTPUT_DIR)),
        ]), unsafe_allow_html=True)
        if c2.button("Reset to defaults", icon=":material/restart_alt:", width="stretch"):
            reset_settings()
            st.rerun()
