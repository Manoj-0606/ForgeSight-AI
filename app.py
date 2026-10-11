"""ForgeSight AI — Industrial Vision & Safety Intelligence (Streamlit entry point).

Run with:  streamlit run app.py
"""
from __future__ import annotations

import warnings

import streamlit as st

warnings.filterwarnings("ignore", category=FutureWarning, module="supervision")

# Load every native library the app needs up front: a Windows Smart App Control block then shows a clear message
# here instead of a traceback halfway through a demo, and nothing native is loaded later in the session.
try:
    import cv2  # noqa: E402,F401
    import numpy  # noqa: E402,F401
    import onnxruntime  # noqa: E402,F401
    import pandas  # noqa: E402,F401
    import pyarrow  # noqa: E402,F401  (Streamlit dataframes)
except ImportError as _exc:  # pragma: no cover - environment problem, not app logic
    st.set_page_config(page_title="ForgeSight AI - setup problem", layout="centered")
    st.error(f"A required native library could not be loaded:\n\n`{_exc}`")
    if "Application Control" in str(_exc) or "4551" in str(_exc):
        st.markdown(
            "**Windows Smart App Control blocked an unsigned Python extension.** It checks the reputation of "
            "unsigned files online, so this typically happens while the network is unavailable or changing.\n\n"
            "1. Make sure the laptop is online (stable Wi-Fi).\n"
            "2. Stop the app (Ctrl+C) and run `.venv\\Scripts\\python scripts\\check_environment.py`.\n"
            "3. Start the app again once every check passes.\n\n"
            "Do not turn Smart App Control off for this: it cannot be re-enabled without resetting Windows.")
    st.stop()

from forgesight import __version__, ui  # noqa: E402
from forgesight.annotate import fmt_ts  # noqa: E402
from forgesight.config import ASSETS_DIR, ensure_dirs  # noqa: E402
from forgesight.results import list_runs  # noqa: E402
from forgesight.state import PAGES, get_run, set_run  # noqa: E402
from views import command_center, safety_events, settings_page, video_intelligence  # noqa: E402

st.set_page_config(page_title="ForgeSight AI", page_icon=":material/visibility:", layout="wide",
                   initial_sidebar_state="expanded")
ensure_dirs()
ui.inject_css()

PAGES.update({
    "command": st.Page(command_center.render, title="Command Center", icon=":material/space_dashboard:",
                       url_path="command-center", default=True),
    "video": st.Page(video_intelligence.render, title="Video Intelligence", icon=":material/videocam:",
                     url_path="video-intelligence"),
    "events": st.Page(safety_events.render, title="Safety Events", icon=":material/warning:",
                      url_path="safety-events"),
    "settings": st.Page(settings_page.render, title="Settings", icon=":material/tune:", url_path="settings"),
})
nav = st.navigation({
    "Operations": [PAGES["command"], PAGES["video"], PAGES["events"]],
    "Configuration": [PAGES["settings"]],
})

st.logo(str(ASSETS_DIR / "logo.svg"), size="large", icon_image=str(ASSETS_DIR / "icon.svg"))

with st.sidebar:
    runs = list_runs()
    run = get_run()
    st.markdown('<div class="fs-side-label">Active analysis</div>', unsafe_allow_html=True)
    if runs:
        ids = [None] + [r["run_id"] for r in runs]
        labels = {r["run_id"]: f"{r['name']} · {r['created_at'][5:16].replace('T', ' ')}" for r in runs}
        current = run.run_id if run and run.run_id in labels else None
        choice = st.selectbox("Analysis", ids, index=ids.index(current), label_visibility="collapsed",
                              format_func=lambda r: "— none loaded —" if r is None else labels[r],
                              key=f"run_picker_{current}")
        if choice != current:
            set_run(next((r["run_dir"] for r in runs if r["run_id"] == choice), None))
            st.rerun()
    else:
        st.caption("No analyses yet")
    if run:
        m = run.metrics
        st.markdown(
            f'<div class="fs-side-card" style="margin-top:6px">'
            f'<div class="r"><span>Status</span><span style="color:#12B76A">● complete</span></div>'
            f'<div class="r"><span>Duration</span><span>{fmt_ts(run.summary["processing"]["analyzed_duration_s"])}</span></div>'
            f'<div class="r"><span>Peak occupancy</span><span>{m["peak_occupancy"]}</span></div>'
            f'<div class="r"><span>Zone intrusions</span><span style="color:{"#FF8A80" if m["zone_entries"] else "#9AA4B2"}">'
            f'{m["zone_entries"]}</span></div></div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        f'<div class="fs-foot">ForgeSight AI v{__version__} · YOLO11 + ByteTrack · CPU<br>'
        'Video-based safety monitoring prototype. Not a certified industrial safety system.</div>',
        unsafe_allow_html=True,
    )

nav.run()
