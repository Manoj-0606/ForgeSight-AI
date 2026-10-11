"""Pre-demo check: Python environment, native libraries, models, sample videos, email configuration.

Run from the project folder before starting the app or recording a demo:
    .venv\\Scripts\\python scripts\\check_environment.py

It never prints secrets. Exit code 0 = everything required is OK.
"""
from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REQUIRED = ["numpy", "pandas", "pyarrow", "cv2", "onnxruntime", "streamlit", "supervision", "plotly",
            "imageio_ffmpeg", "scipy"]
OK, FAIL, WARN = "OK  ", "FAIL", "WARN"


def line(status: str, what: str, detail: str = "") -> None:
    print(f"[{status}] {what:34s} {detail}")


def recent_blocks(hours: int = 24) -> list[str]:
    """Smart App Control / WDAC block events (Code Integrity 3077) for files of this project."""
    query = (f"*[System[(EventID=3077) and TimeCreated[timediff(@SystemTime) <= {hours * 3600 * 1000}]]]")
    try:
        out = subprocess.run(["wevtutil", "qe", "Microsoft-Windows-CodeIntegrity/Operational", f"/q:{query}",
                              "/f:text", "/rd:true", "/c:50"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    hits = []
    for ln in out.splitlines():
        if "attempted to load" in ln and str(ROOT.name) in ln:
            hits.append(ln.split("attempted to load", 1)[1].split(" that did not", 1)[0].strip())
    return hits


def main() -> int:
    failed = False
    in_venv = Path(sys.prefix).resolve() == (ROOT / ".venv").resolve()
    line(OK if in_venv else WARN, "Python", f"{sys.version.split()[0]} at {sys.prefix}"
         + ("" if in_venv else "  (not the project .venv: use .venv\\Scripts\\python)"))

    for mod in REQUIRED:
        try:
            m = importlib.import_module(mod)
            line(OK, f"import {mod}", getattr(m, "__version__", ""))
        except Exception as exc:  # ImportError, OSError from blocked DLLs
            failed = True
            line(FAIL, f"import {mod}", f"{type(exc).__name__}: {exc}")

    try:
        from forgesight.config import MODELS_DIR
        from forgesight.hazard import HAZARD_MODEL_FILE
        from forgesight.ppe import PPE_MODEL_FILE
        req = MODELS_DIR / "yolo11n.onnx"
        line(OK if req.exists() else WARN, "person model yolo11n.onnx",
             "present" if req.exists() else "missing: downloads on first analysis (needs internet)")
        for name, label in ((PPE_MODEL_FILE, "PPE beta model"), (HAZARD_MODEL_FILE, "fire beta model")):
            p = MODELS_DIR / name
            line(OK if p.exists() else WARN, label, "present" if p.exists() else "missing: downloads when enabled")
        import onnxruntime as ort
        ort.InferenceSession(str(req), providers=["CPUExecutionProvider"])
        line(OK, "person model loads in ONNX Runtime")
    except Exception as exc:
        failed = True
        line(FAIL, "models", f"{type(exc).__name__}: {exc}")

    try:
        from forgesight.video_io import probe_video
        vids = sorted((ROOT / "sample_videos").rglob("*.mp4"))
        bad = [v.name for v in vids if not probe_video(v)]
        line(OK if vids and not bad else WARN, "sample videos", f"{len(vids)} found" + (f", unreadable: {bad}" if bad else ""))
    except Exception as exc:
        line(WARN, "sample videos", f"{type(exc).__name__}: {exc}")

    try:
        from forgesight.notify import load_email_config
        secrets = {}
        sec_file = ROOT / ".streamlit" / "secrets.toml"
        if sec_file.exists():
            import tomllib
            secrets = tomllib.loads(sec_file.read_text(encoding="utf-8")).get("email", {})
        cfg, msg = load_email_config(secrets=secrets or None)
        line(OK if cfg else WARN, "email alerts", msg)  # message contains host/port and masked recipients only
    except Exception as exc:
        line(WARN, "email alerts", f"{type(exc).__name__}: {exc}")

    blocks = recent_blocks()
    used = [b for b in blocks if not any(x in b for x in ("\\torch\\", "_tkagg", "\\_parquet", "_cpp_tests"))]
    if used:
        line(WARN, "Smart App Control (24 h)", f"{len(used)} block(s) of files the app uses, e.g. {used[0]}")
        print("       These usually happen while offline. If every import above is OK, they no longer apply.")
    else:
        line(OK, "Smart App Control (24 h)", "no blocks of files the app uses"
             + (f" ({len(blocks)} block(s) of unused files: torch/matplotlib-Tk/parquet)" if blocks else ""))

    print("\nResult:", "NOT READY - fix the FAIL lines above" if failed else "ready - start the app with:\n"
          "  .venv\\Scripts\\python -m streamlit run app.py")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
