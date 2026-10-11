"""Self-contained HTML report and ZIP data package for one analysis run (no server, no internet needed)."""
from __future__ import annotations

import base64
import html
import io
import json
import zipfile
from datetime import datetime

import cv2
import numpy as np
import pandas as pd

from . import __version__, analytics, charts
from . import events as evt
from . import incidents as inc
from . import ppe
from .annotate import fmt_ts
from .config import Settings
from .results import RunResult, choose_bin_seconds, events_export, interval_activity

E = html.escape


def _img_uri(path, max_side: int = 320, quality: int = 80) -> str:
    img = cv2.imread(str(path)) if path else None
    if img is None:
        return ""
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else ""


def heatmap_image(run: RunResult, heat: np.ndarray) -> str:
    """Static heatmap: cyan density over the dimmed median background, zone outlined in amber."""
    bg = cv2.imread(str(run.background_path)) if run.background_path else None
    w, h = run.summary["processing"]["output_size"]
    if bg is None:
        bg = np.full((h // 2, w // 2, 3), 30, np.uint8)
    base = (bg.astype(np.float32) * 0.55)
    norm = charts.normalized_heatmap(heat)
    norm = cv2.resize(norm.astype(np.float32), (bg.shape[1], bg.shape[0]), interpolation=cv2.INTER_CUBIC)
    alpha = np.where(norm < 0.12, 0.0, np.clip(norm, 0, 1) * 0.85)[..., None]
    cyan = np.array([238, 211, 34], np.float32)
    out = (base * (1 - alpha) + cyan * alpha).astype(np.uint8)
    pts = run.summary["settings"].get("zone_points")
    if run.metrics.get("zone_enabled") and pts:
        poly = np.array([[p[0] / 100 * bg.shape[1], p[1] / 100 * bg.shape[0]] for p in pts], np.int32)
        cv2.polylines(out, [poly.reshape(-1, 1, 2)], True, (36, 165, 245), 2, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def _light(fig):
    """Restyle a dashboard figure for a white document."""
    fig.update_layout(margin=dict(l=56, r=16, t=fig.layout.margin.t or 12, b=56),
                      paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", font=dict(color="#334155"),
                      legend=dict(font=dict(color="#334155")),
                      hoverlabel=dict(bgcolor="#0F172A", font=dict(color="#F8FAFC")))
    fig.update_xaxes(tickfont=dict(color="#475569"), title_font=dict(color="#475569"), linecolor="#CBD5E1")
    fig.update_yaxes(gridcolor="#E2E8F0", tickfont=dict(color="#475569"), title_font=dict(color="#475569"))
    return fig


def _kv(rows) -> str:
    return "<table class='kv'>" + "".join(f"<tr><th>{E(str(k))}</th><td>{E(str(v))}</td></tr>" for k, v in rows) + "</table>"


def key_findings(run: RunResult, incidents: pd.DataFrame, alerts: pd.DataFrame) -> list[str]:
    """Plain-language findings, each derived directly from the data."""
    m, p = run.metrics, run.summary["processing"]
    out = [f"{fmt_ts(p['analyzed_duration_s'])} of footage analyzed; peak occupancy {m['peak_occupancy']} people at "
           f"{fmt_ts(m['peak_occupancy_time_s'])}, average {m['avg_occupancy']:.1f}."]
    if m.get("zone_enabled"):
        ints = evt.intrusions(run.events)
        if len(ints):
            crit = int((incidents["severity"] == "Critical").sum())
            longest = ints.loc[ints["dwell_s"].idxmax()] if ints["dwell_s"].notna().any() else None
            s = (f"{len(ints)} restricted-zone intrusion{'s' if len(ints) != 1 else ''} into "
                 f"{run.summary['settings']['zone_name']}")
            if longest is not None:
                s += (f"; longest dwell {evt.format_dwell(longest.dwell_s, longest.dwell_is_lower_bound)} "
                      f"({longest.event_id})")
            out.append(s + (f"; {crit} rated Critical." if crit else "."))
        else:
            out.append(f"No restricted-zone intrusions into {run.summary['settings']['zone_name']} were detected.")
        if m.get("zone_initial_occupants"):
            out.append(f"{m['zone_initial_occupants']} person(s) were already inside the zone when the footage began "
                       "(listed, not counted as intrusions).")
    if len(alerts):
        top = alerts.loc[alerts["duration_s"].idxmax()]
        out.append(f"{len(alerts)} sustained-occupancy alert(s); longest {top.duration_s:.1f} s from "
                   f"{fmt_ts(top.start_s)} (peak {top.peak}).")
    if run.ppe_active:
        for item, c in run.summary["ppe"]["counts"].items():
            out.append(f"PPE beta, {ppe.ITEMS[item].label.lower()}: {c['detected']} detected, "
                       f"{c['not detected']} not detected (verify), {c['uncertain']} uncertain.")
    if run.hazard_active:
        hz = run.hazards if run.hazards is not None else pd.DataFrame()
        if len(hz):
            mails = hz["email_status"].fillna("").value_counts().to_dict()
            out.append(f"Fire/smoke beta: {len(hz)} confirmed event(s) ("
                       + ", ".join(f"{e.event_id} at {fmt_ts(e.start_time_s)}" for e in hz.itertuples())
                       + "); email: " + ", ".join(f"{n} {k}" for k, n in mails.items() if k) + ".")
        else:
            out.append(f"Fire/smoke beta: no event confirmed in {run.summary['hazard']['checks']} checks.")
    n_open = int((incidents["status"] == inc.OPEN).sum())
    out.append(f"{len(incidents)} incident(s) in total; {n_open} open, {len(incidents) - n_open} reviewed.")
    return out


CSS = """
:root{--ink:#0F172A;--ink2:#334155;--ink3:#64748B;--line:#E2E8F0;--cyan:#0891B2;--amber:#B45309;--red:#DC2626}
*{box-sizing:border-box}body{margin:0;font-family:Inter,'Segoe UI',system-ui,sans-serif;color:var(--ink);background:#F1F5F9}
.page{max-width:1100px;margin:0 auto;background:#fff;padding:0 0 40px}
header{background:#0A0F1A;color:#E6EAF0;padding:28px 40px;border-bottom:3px solid #22D3EE}
header .brand{font-family:'Cascadia Mono',Consolas,monospace;font-size:12px;letter-spacing:.14em;color:#22D3EE}
header h1{margin:6px 0 4px;font-size:26px}header p{margin:0;color:#9AA4B2;font-size:13px}
section{padding:22px 40px 4px}h2{font-size:16px;margin:0 0 12px;padding-bottom:6px;border-bottom:1px solid var(--line)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.kpi{border:1px solid var(--line);border-left:3px solid var(--cyan);border-radius:8px;padding:10px 12px}
.kpi.red{border-left-color:var(--red)}.kpi.amber{border-left-color:var(--amber)}
.kpi b{display:block;font-family:Consolas,monospace;font-size:24px;margin-top:4px}.kpi span{font-size:12px;color:var(--ink3)}
ul.findings li{margin:4px 0;font-size:14px}
table{border-collapse:collapse;width:100%;font-size:12.5px}
table.kv th{text-align:left;color:var(--ink3);font-weight:500;width:40%;padding:4px 8px 4px 0}
table.kv td{font-family:Consolas,monospace;padding:4px 0}
table.grid th{background:#F8FAFC;text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);font-weight:600}
table.grid td{padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}
table.grid img{width:72px;height:72px;object-fit:cover;border-radius:6px}
.sev{display:inline-block;font:600 10px Consolas,monospace;padding:2px 6px;border-radius:4px;letter-spacing:.05em}
.sev.Critical{background:var(--red);color:#fff}.sev.High{background:#FEE2E2;color:#991B1B}
.sev.Medium{background:#FEF3C7;color:#92400E}.sev.Low{background:#F1F5F9;color:var(--ink2)}
.st{font:600 10px Consolas,monospace}.st.Reviewed{color:#047857}.st.Open{color:var(--ink3)}
.two{display:grid;grid-template-columns:1fr 1fr;gap:20px}.muted{color:var(--ink3);font-size:12px}
.note{background:#F8FAFC;border:1px solid var(--line);border-radius:8px;padding:10px 12px;font-size:12.5px;color:var(--ink2)}
.beta{background:#FEF3C7;color:#92400E;font:600 10px Consolas,monospace;padding:2px 6px;border-radius:4px}
img.heat{width:100%;border-radius:8px;border:1px solid var(--line)}
footer{padding:20px 40px 0;font-size:11.5px;color:var(--ink3)}
@media print{body{background:#fff}.page{max-width:none}section{break-inside:avoid-page}header{-webkit-print-color-adjust:exact;print-color-adjust:exact}}
"""


def build_report_html(run: RunResult, cfg: Settings, incidents: pd.DataFrame | None = None) -> str:
    s, m, p = run.summary, run.metrics, run.summary["processing"]
    incidents = incidents if incidents is not None else inc.build_incidents(run, cfg)
    alerts = inc.occupancy_alerts(run, cfg)
    occ_scope = cfg.occ_alert_scope if m.get("zone_enabled") else "frame"
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    occ = _light(charts.occupancy_chart(run.frames, run.events, m.get("zone_enabled", False), height=340,
                                        trend_window_s=cfg.trend_window_s, alerts=alerts,
                                        threshold=cfg.occ_alert_threshold if cfg.occ_alert_enabled else None,
                                        threshold_scope=occ_scope))
    bin_s = choose_bin_seconds(p["analyzed_duration_s"])
    act = _light(charts.activity_chart(interval_activity(run.frames, run.events, bin_s), bin_s, height=260))
    occ_html = occ.to_html(full_html=False, include_plotlyjs="inline", config={"displaylogo": False})
    act_html = act.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False})

    n_open = int((incidents["status"] == inc.OPEN).sum())
    kpis = [
        ("Person detections", f"{m['total_detections']:,}", "boxes over all analyzed frames", ""),
        ("Est. unique people", "—" if m["unique_people_est"] is None else m["unique_people_est"], "track IDs (upper bound)", ""),
        ("Peak occupancy", m["peak_occupancy"], f"at {fmt_ts(m['peak_occupancy_time_s'])}", ""),
        ("Zone intrusions", m["zone_entries"] if m.get("zone_enabled") else "—",
         f"+{m.get('zone_initial_occupants', 0)} present at start", "red" if m["zone_entries"] else ""),
        ("Occupancy alerts", len(alerts) if cfg.occ_alert_enabled else "—",
         f"≥{cfg.occ_alert_threshold} for ≥{cfg.occ_alert_min_s:g} s", "amber" if len(alerts) else ""),
        ("Open incidents", f"{n_open}/{len(incidents)}", "review status", "amber" if n_open else ""),
    ]
    kpi_html = "".join(f"<div class='kpi {t}'><span>{E(a)}</span><b>{E(str(b))}</b><span>{E(c)}</span></div>"
                       for a, b, c, t in kpis)

    rows = []
    for r in incidents.itertuples():
        img = _img_uri(run.path(r.evidence_crop) if isinstance(r.evidence_crop, str) and r.evidence_crop else None)
        who = f"ID {r.track_id}" if pd.notna(r.track_id) else "—"
        rows.append(
            f"<tr><td>{'<img src=' + chr(34) + img + chr(34) + '>' if img else ''}</td>"
            f"<td><b>{E(r.incident_id)}</b><br><span class='sev {E(r.severity)}'>{E(r.severity.upper())}</span></td>"
            f"<td>{E(r.type)}<br><span class='muted'>{E(str(r.context))}</span></td>"
            f"<td>{fmt_ts(r.time_s) if pd.notna(r.time_s) else '—'}</td><td>{E(who)}</td><td>{E(str(r.dwell))}</td>"
            f"<td>{E(r.severity_reason)}</td>"
            f"<td><span class='st {E(r.status)}'>{E(r.status.upper())}</span>"
            f"{'<br><span class=muted>' + E(r.note) + '</span>' if r.note else ''}</td></tr>")
    incident_table = ("<table class='grid'><tr><th>Evidence</th><th>Incident</th><th>Type</th><th>Time</th>"
                      "<th>Track</th><th>Dwell / duration</th><th>Severity reason</th><th>Status</th></tr>"
                      + "".join(rows) + "</table>") if rows else "<p class='muted'>No incidents.</p>"

    ppe_html = ""
    if s.get("ppe", {}).get("enabled"):
        if run.ppe_active and len(run.ppe):
            prow = "".join(
                f"<tr><td>ID {r.track_id}</td><td>{E(r.item_label)}</td><td>{E(ppe.STATUS_LABELS[r.status])}</td>"
                f"<td>{r.hits}/{r.checks}</td><td>{fmt_ts(r.first_check_s)}–{fmt_ts(r.last_check_s)}</td></tr>"
                for r in run.ppe.sort_values(["status", "track_id"]).itertuples())
            pl = ppe.VALIDATION["person_level"]
            ppe_html = (
                "<section><h2>PPE checks <span class='beta'>BETA</span></h2>"
                "<p class='note'>Statuses combine several PPE checks per tracked person. <b>Not detected</b> means the "
                "person was clearly visible but the item was found in almost no checks. It is a lead to verify on "
                "video, not a confirmed violation. On labelled construction images the full pipeline's “detected” "
                f"status was correct {pl['hardhat']['detected_precision']:.0%} (hard hat) / "
                f"{pl['vest']['detected_precision']:.0%} (vest) of the time, “not detected” only "
                f"{pl['hardhat']['not_detected_precision']:.0%} / {pl['vest']['not_detected_precision']:.0%}. "
                "Not validated on automotive footage.</p>"
                "<table class='grid'><tr><th>Person</th><th>Item</th><th>Status</th><th>Found in checks</th>"
                f"<th>Checked</th></tr>{prow}</table></section>")
        else:
            ppe_html = (f"<section><h2>PPE checks <span class='beta'>BETA</span></h2><p class='note'>"
                        f"{E(s['ppe'].get('note') or 'No people could be judged.')}</p></section>")

    hazard_html = ""
    if s.get("hazard", {}).get("enabled"):
        hzs = s["hazard"]
        if run.hazard_active:
            hz_fig = _light(charts.hazard_chart(run.hazard_checks, run.hazards, hzs["threshold"],
                                                hzs.get("classes", ["fire"]), height=240))
            rows_h = "".join(
                f"<tr><td><b>{E(h.event_id)}</b></td><td>{fmt_ts(h.start_time_s)}</td>"
                f"<td>{fmt_ts(h.confirmed_time_s)}</td><td>{h.max_score:.2f}</td>"
                f"<td>{E(str(h.email_status))}</td><td>{E(str(h.email_detail))}</td></tr>"
                for h in (run.hazards.itertuples() if run.hazards is not None else []))
            table = ("<table class='grid'><tr><th>Event</th><th>First seen</th><th>Confirmed</th><th>Max score</th>"
                     f"<th>Email</th><th>Delivery detail</th></tr>{rows_h}</table>") if rows_h else \
                "<p class='muted'>No fire/smoke event was confirmed.</p>"
            hazard_html = (
                "<section><h2>Fire &amp; smoke checks <span class='beta'>BETA</span></h2>"
                + hz_fig.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False})
                + table +
                f"<p class='note'>Frame-level classifier ({E(hzs.get('model') or '')}). An event is confirmed after "
                f"{hzs['confirm_checks']} consecutive checks with score ≥ {hzs['threshold']:.2f}. Measured on labelled "
                "D-Fire test images: fire recall 0.72, precision 0.64; smoke is unreliable (recall 0.08). It also "
                "flags process flames (furnaces, welding). Verify every alert on video. This is not a certified "
                "fire detection system.</p></section>")
        else:
            hazard_html = (f"<section><h2>Fire &amp; smoke checks <span class='beta'>BETA</span></h2>"
                           f"<p class='note'>{E(hzs.get('note') or 'Not active.')}</p></section>")

    heat_uri = heatmap_image(run, run.heatmap)
    hs = analytics.heatmap_stats(run.heatmap, s["settings"].get("zone_points") if m.get("zone_enabled") else None)
    heat_caption = "Where people's feet were detected (presence)."
    if hs["zone_share"] is not None:
        heat_caption += f" {hs['zone_share']:.0%} of presence inside the restricted zone (amber outline)."

    settings_rows = [
        ("Model", f"{s['model']['name']} · {s['model']['runtime']}"),
        ("Person confidence", f"{s['settings']['confidence']:.2f}"),
        ("Analysis rate", f"{p['analysis_fps']:.2f} fps (every {p['sampling_step']} frames)"),
        ("Tracking", "ByteTrack" if run.tracking_active else "off"),
        ("Zone", f"{s['settings']['zone_name']} · {len(s['settings']['zone_points'])} points"
         if m.get("zone_enabled") else "disabled"),
        ("Entry / exit confirmation", f"{s['settings']['entry_frames']} / {s['settings']['exit_frames']} frames"),
        ("Occupancy alert rule", f"{'zone' if occ_scope == 'zone' else 'view'} ≥ {cfg.occ_alert_threshold} for ≥ "
         f"{cfg.occ_alert_min_s:g} s" if cfg.occ_alert_enabled else "disabled"),
        ("Trend smoothing", f"{cfg.trend_window_s:g} s"),
        ("PPE beta", "on" if run.ppe_active else "off"),
        ("Fire/smoke beta", "on" if run.hazard_active else "off"),
    ]
    clip_rows = [
        ("Source video", s["original_name"]), ("Camera", s["camera_label"]),
        ("Resolution", f"{s['video']['width']}×{s['video']['height']} @ {s['video']['fps']:.1f} fps"),
        ("Analyzed duration", fmt_ts(p["analyzed_duration_s"])), ("Frames analyzed", f"{p['frames_analyzed']:,}"),
        ("Processing time", f"{p['elapsed_s']:.1f} s on CPU"), ("Analyzed at", s["created_at"].replace("T", " ")),
        ("Run ID", run.run_id),
    ]
    rules = "".join(f"<tr><td>{E(a)}</td><td><span class='sev {E(b)}'>{E(b.upper())}</span></td></tr>"
                    for a, b in inc.severity_rules(cfg))
    findings = "".join(f"<li>{E(x)}</li>" for x in key_findings(run, incidents, alerts))

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>ForgeSight Report · {E(s['original_name'])}</title>
<style>{CSS}</style></head><body><div class="page">
<header><div class="brand">FORGESIGHT AI · SAFETY & WORKFORCE REPORT</div>
<h1>{E(s['original_name'])}</h1><p>{E(s['camera_label'])} · analyzed {E(s['created_at'].replace('T', ' '))} ·
report generated {generated}</p></header>
<section><h2>Summary</h2><div class="kpis">{kpi_html}</div><ul class="findings">{findings}</ul></section>
<section><h2>Occupancy over time</h2>{occ_html}<p class="muted">Thin steps: people per analyzed frame. Solid lines:
{cfg.trend_window_s:g} s rolling average. ◆ zone intrusion, ○ present at start, shaded = sustained-occupancy alert.</p></section>
<section class="two"><div><h2>Movement heatmap</h2><img class="heat" src="{heat_uri}" alt="heatmap">
<p class="muted">{E(heat_caption)}</p></div><div><h2>Detection activity</h2>{act_html}</div></section>
<section><h2>Incidents</h2>{incident_table}</section>
{ppe_html}
{hazard_html}
<section class="two"><div><h2>Clip</h2>{_kv(clip_rows)}</div><div><h2>Processing settings</h2>{_kv(settings_rows)}</div></section>
<section><h2>Severity rules</h2><table class="grid">{rules}</table></section>
<section><h2>Method & limitations</h2><div class="note">
People are detected with YOLO11n (COCO, ONNX Runtime on CPU) on sampled frames and tracked with ByteTrack. Zone
membership uses each person's foot point (bottom-centre of the box). An intrusion is confirmed after
{s['settings']['entry_frames']} consecutive frames inside; re-entry requires a confirmed exit. Dwell marked “≥”
was cut off by the start or end of the footage or by a lost track. Unique-people counts are upper-bound estimates,
because identity switches after occlusion can split one person into several IDs. Accuracy depends on camera
angle, occlusion and lighting. All figures in this report are derived from the detections of this analysis.
<b>Video-based safety monitoring prototype, not a certified industrial safety system.</b></div></section>
<footer>ForgeSight AI v{__version__} · evidence images and the annotated video are stored in the run folder
<code>{E(str(run.run_dir))}</code></footer>
</div></body></html>"""


def incidents_export(run: RunResult, incidents: pd.DataFrame) -> pd.DataFrame:
    df = incidents.drop(columns=["sev_rank", "review_key"], errors="ignore").copy()
    df["evidence_image"] = df["evidence_image"].map(lambda x: str(run.run_dir / x) if isinstance(x, str) and x else "")
    df = df.drop(columns=["evidence_crop"])
    df.insert(0, "run_id", run.run_id)
    df.insert(1, "source_video", run.summary["original_name"])
    return df


def build_data_package(run: RunResult, cfg: Settings, incidents: pd.DataFrame, report_html: str) -> bytes:
    """ZIP with the report, all tables, review status and evidence images (the annotated video is not included)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("report.html", report_html)
        z.writestr("incidents.csv", incidents_export(run, incidents).to_csv(index=False))
        z.writestr("zone_events.csv", events_export(run).to_csv(index=False))
        for name in ("summary.json", "frames.csv", "tracks.csv", "ppe.csv", "review.json", "hazards.csv",
                     "hazard_checks.csv", "alerts.csv"):
            f = run.run_dir / name
            if f.exists():
                z.write(f, name)
        z.writestr("review_settings.json", json.dumps({
            k: v for k, v in cfg.to_dict().items()
            if k.startswith(("occ_", "sev_", "trend_"))}, indent=2))
        for f in sorted((run.run_dir / "evidence").glob("*.jpg")):
            z.write(f, f"evidence/{f.name}")
        z.writestr("README.txt", f"ForgeSight AI data package for run {run.run_id}.\n"
                                 f"The annotated video is not included (size); it is at {run.video_path}\n")
    return buf.getvalue()
