"""Theme CSS and small HTML components for the Streamlit interface."""
from __future__ import annotations

import html

import streamlit as st

CSS = """
<style>

:root {
  --fs-bg: #0A0F1A;
  --fs-surface: #111827;
  --fs-surface-2: #0D1422;
  --fs-elevated: #162033;
  --fs-border: #1E293B;
  --fs-border-strong: #2B3A52;
  --fs-text: #E6EAF0;
  --fs-text-2: #9AA4B2;
  --fs-text-3: #64748B;
  --fs-cyan: #22D3EE;
  --fs-cyan-soft: rgba(34, 211, 238, 0.10);
  --fs-amber: #F5A524;
  --fs-amber-soft: rgba(245, 165, 36, 0.12);
  --fs-red: #F04438;
  --fs-red-soft: rgba(240, 68, 56, 0.12);
  --fs-green: #12B76A;
  --fs-green-soft: rgba(18, 183, 106, 0.12);
  --fs-mono: 'JetBrains Mono', 'Cascadia Mono', Consolas, monospace;
  --fs-sans: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
}

html, body, [class*="css"], .stApp, .stMarkdown, button, input, textarea, select {
  font-family: var(--fs-sans);
}
.stApp {
  background:
    radial-gradient(1200px 500px at 85% -10%, rgba(34,211,238,0.06), transparent 60%),
    var(--fs-bg);
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stMainBlockContainer"], .block-container {
  padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1480px;
}
h1, h2, h3, h4 { letter-spacing: -0.01em; color: var(--fs-text); }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0B1220 0%, #0A0F1A 100%);
  border-right: 1px solid var(--fs-border);
}
[data-testid="stSidebarNav"] a, [data-testid="stSidebarNavLink"] { border-radius: 8px; }
[data-testid="stSidebarNavLink"][aria-current="page"], [data-testid="stSidebarNav"] a[aria-current="page"] {
  background: var(--fs-cyan-soft);
  box-shadow: inset 2px 0 0 var(--fs-cyan);
}
[data-testid="stSidebarNavSeparator"] { border-color: var(--fs-border); }
[data-testid="stSidebarLogo"], [data-testid="stLogo"] { height: 44px !important; max-width: 230px !important; }

/* ---------- cards: st.container(border=True, key="fs-card-...") ---------- */
[class*="st-key-fs-card"] {
  background: var(--fs-surface);
  border-radius: 12px;
}

/* ---------- buttons ---------- */
.stButton > button, .stDownloadButton > button, [data-testid="stPageLink"] a {
  border-radius: 8px; font-weight: 500;
  border: 1px solid var(--fs-border-strong);
  transition: border-color .15s ease, background .15s ease, transform .1s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover { border-color: var(--fs-cyan); color: var(--fs-cyan); }
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] {
  background: linear-gradient(180deg, #22D3EE, #0EA5C6); color: #04131A; border: none; font-weight: 600;
  box-shadow: 0 0 0 1px rgba(34,211,238,.35), 0 8px 24px -8px rgba(34,211,238,.45);
}
.stButton > button[kind="primary"]:hover, .stButton > button[data-testid="stBaseButton-primary"]:hover {
  color: #04131A; filter: brightness(1.08);
}
.stButton > button:active { transform: translateY(1px); }

/* ---------- inputs ---------- */
[data-testid="stFileUploaderDropzone"] {
  background: var(--fs-surface-2); border: 1px dashed var(--fs-border-strong); border-radius: 12px;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--fs-cyan); }
.stProgress > div > div > div > div { background: linear-gradient(90deg, #0EA5C6, #22D3EE); }
[data-testid="stExpander"] details { border-color: var(--fs-border); border-radius: 10px; background: var(--fs-surface-2); }
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--fs-border); }
.stTabs [data-baseweb="tab"] { padding: 6px 12px; }
[data-testid="stDataFrame"], [data-testid="stDataEditor"] { border: 1px solid var(--fs-border); border-radius: 10px; }
[data-testid="stImage"] img, [data-testid="stVideo"] video, video { border-radius: 10px; }

/* ---------- custom components ---------- */
.fs-header { display:flex; align-items:flex-end; justify-content:space-between; gap:16px; flex-wrap:wrap;
  margin-bottom: 18px; padding-bottom: 16px; border-bottom: 1px solid var(--fs-border); }
.fs-eyebrow { font-family: var(--fs-mono); font-size: 11px; letter-spacing: .14em; text-transform: uppercase;
  color: var(--fs-cyan); margin-bottom: 6px; }
.fs-title { font-size: 28px; font-weight: 650; color: var(--fs-text); line-height: 1.15; margin: 0; }
.fs-subtitle { color: var(--fs-text-2); font-size: 14px; margin-top: 6px; }
.fs-chips { display:flex; gap:8px; flex-wrap:wrap; }
.fs-chip { display:inline-flex; align-items:center; gap:6px; padding: 4px 10px; border-radius: 999px;
  border: 1px solid var(--fs-border-strong); background: var(--fs-surface-2); color: var(--fs-text-2);
  font-size: 12px; font-family: var(--fs-mono); white-space: nowrap; }
.fs-chip .d { width:7px; height:7px; border-radius:50%; background: var(--fs-text-3); }
.fs-chip.ok .d { background: var(--fs-green); box-shadow: 0 0 8px var(--fs-green); }
.fs-chip.live .d { background: var(--fs-cyan); box-shadow: 0 0 8px var(--fs-cyan); animation: fs-pulse 1.8s infinite; }
.fs-chip.warn .d { background: var(--fs-amber); }
.fs-chip.crit { border-color: rgba(240,68,56,.45); color: #FCA5A0; }
.fs-chip.crit .d { background: var(--fs-red); box-shadow: 0 0 8px var(--fs-red); }
@keyframes fs-pulse { 0%,100% { opacity: 1 } 50% { opacity: .35 } }

.fs-kpi { position:relative; background: var(--fs-surface); border: 1px solid var(--fs-border); border-radius: 12px;
  padding: 16px 16px 14px; height: 100%; min-height: 128px; overflow:hidden;
  transition: border-color .2s ease, transform .2s ease; animation: fs-rise .45s ease both; }
.fs-kpi:hover { border-color: var(--fs-border-strong); transform: translateY(-1px); }
.fs-kpi::before { content:""; position:absolute; left:0; top:0; bottom:0; width:3px; background: var(--fs-cyan); opacity:.85; }
.fs-kpi.amber::before { background: var(--fs-amber); }
.fs-kpi.red::before { background: var(--fs-red); }
.fs-kpi.green::before { background: var(--fs-green); }
.fs-kpi.muted::before { background: var(--fs-text-3); }
.fs-kpi-label { color: var(--fs-text-2); font-size: 12.5px; font-weight: 500; display:flex; align-items:center; gap:6px; }
.fs-kpi-label .i { color: var(--fs-text-3); font-size: 11px; cursor: help; border:1px solid var(--fs-border-strong);
  border-radius: 50%; width: 15px; height: 15px; display:inline-flex; align-items:center; justify-content:center; }
.fs-kpi-value { font-family: var(--fs-mono); font-size: 32px; font-weight: 600; color: var(--fs-text);
  margin-top: 8px; line-height: 1.1; font-variant-numeric: tabular-nums; }
.fs-kpi-value small { font-size: 14px; color: var(--fs-text-3); font-weight: 500; margin-left: 4px; }
.fs-kpi.red .fs-kpi-value { color: #FF8A80; }
.fs-kpi-sub { color: var(--fs-text-3); font-size: 12px; margin-top: 6px; }
@keyframes fs-rise { from { opacity:0; transform: translateY(4px) } to { opacity:1; transform:none } }

.fs-section { display:flex; align-items:baseline; justify-content:space-between; gap: 12px; margin: 2px 0 10px; }
.fs-section h3 { font-size: 15px; font-weight: 600; margin: 0; padding: 0; color: var(--fs-text); }
.fs-section span { color: var(--fs-text-3); font-size: 12px; }

.fs-empty { border: 1px dashed var(--fs-border-strong); border-radius: 16px; padding: 44px 36px; text-align:center;
  background: linear-gradient(180deg, rgba(34,211,238,0.04), transparent 70%), var(--fs-surface-2); }
.fs-empty h2 { font-size: 22px; margin: 14px 0 8px; }
.fs-empty p { color: var(--fs-text-2); max-width: 620px; margin: 0 auto; font-size: 14px; }
.fs-steps { display:flex; justify-content:center; gap: 14px; margin-top: 26px; flex-wrap:wrap; }
.fs-step { background: var(--fs-surface); border:1px solid var(--fs-border); border-radius: 12px; padding: 14px 16px;
  width: 210px; text-align:left; }
.fs-step b { font-family: var(--fs-mono); color: var(--fs-cyan); font-size: 12px; }
.fs-step div { color: var(--fs-text); font-size: 14px; font-weight: 600; margin-top: 4px; }
.fs-step span { color: var(--fs-text-3); font-size: 12px; }

.fs-event { display:flex; gap: 12px; align-items:center; padding: 10px 10px; border-radius: 10px;
  border: 1px solid var(--fs-border); background: var(--fs-surface-2); margin-bottom: 8px; position:relative; }
.fs-event::before { content:""; position:absolute; left:0; top:8px; bottom:8px; width:3px; border-radius: 3px; background: var(--fs-red); }
.fs-event.amber::before { background: var(--fs-amber); }
.fs-event.muted::before { background: var(--fs-text-3); }
.fs-sev { display:inline-block; font-family: var(--fs-mono); font-size: 10px; font-weight: 600; letter-spacing: .06em;
  padding: 1px 6px; border-radius: 4px; margin-right: 4px; vertical-align: 1px; border: 1px solid transparent; }
.fs-sev.critical { background: var(--fs-red); color: #fff; }
.fs-sev.high { background: var(--fs-red-soft); color: #FCA5A0; border-color: rgba(240,68,56,.45); }
.fs-sev.medium { background: var(--fs-amber-soft); color: #FCD34D; border-color: rgba(245,165,36,.45); }
.fs-sev.low { background: transparent; color: var(--fs-text-2); border-color: var(--fs-border-strong); }
.fs-sev.reviewed { background: var(--fs-green-soft); color: #6EE7B7; border-color: rgba(18,183,106,.45); }
.fs-sev.open { background: transparent; color: var(--fs-text-2); border-color: var(--fs-border-strong); }
.fs-alert.hazard { border-color: rgba(230,84,0,.6); background: rgba(230,84,0,.14); color: #FFD8C2; }
.fs-alert.amber { border-color: rgba(245,165,36,.45); background: var(--fs-amber-soft); color: #FDE7C2; }
.fs-event img { width: 54px; height: 54px; object-fit: cover; border-radius: 8px; border:1px solid var(--fs-border); margin-left: 6px; }
.fs-event .t { font-size: 13px; font-weight: 600; color: var(--fs-text); }
.fs-event .m { font-family: var(--fs-mono); font-size: 11.5px; color: var(--fs-text-2); margin-top: 3px; }

.fs-kv { display:grid; grid-template-columns: auto 1fr; gap: 7px 16px; font-size: 13px; }
.fs-kv dt { color: var(--fs-text-3); }
.fs-kv dd { margin: 0; color: var(--fs-text); font-family: var(--fs-mono); font-size: 12.5px; text-align: right; }

.fs-brand { display:flex; align-items:center; gap:10px; padding: 4px 2px 14px; border-bottom: 1px solid var(--fs-border); margin-bottom: 12px; }
.fs-brand .n { font-weight: 700; font-size: 16px; color: var(--fs-text); letter-spacing: -.01em; }
.fs-brand .n em { font-style: normal; color: var(--fs-cyan); }
.fs-brand .s { font-size: 10.5px; color: var(--fs-text-3); font-family: var(--fs-mono); letter-spacing: .08em; text-transform: uppercase; }
.fs-side-label { color: var(--fs-text-3); font-size: 10.5px; letter-spacing: .1em; text-transform: uppercase;
  font-family: var(--fs-mono); margin: 4px 0 -4px; }
.fs-side-card { border:1px solid var(--fs-border); border-radius: 10px; padding: 12px; background: var(--fs-surface-2); font-size: 12.5px; }
.fs-side-card .h { color: var(--fs-text-3); font-size: 10.5px; letter-spacing: .1em; text-transform: uppercase; font-family: var(--fs-mono); margin-bottom: 8px; }
.fs-side-card .v { color: var(--fs-text); font-weight: 600; word-break: break-all; }
.fs-side-card .r { display:flex; justify-content:space-between; color: var(--fs-text-2); margin-top: 6px; font-family: var(--fs-mono); font-size: 11.5px; }
.fs-foot { color: var(--fs-text-3); font-size: 11px; line-height: 1.5; margin-top: 14px; }

.fs-alert { display:flex; gap:12px; align-items:center; padding: 12px 14px; border-radius: 10px;
  border: 1px solid rgba(240,68,56,.45); background: var(--fs-red-soft); color: #FECACA; font-size: 13.5px; margin-bottom: 14px; }
.fs-alert b { color: #fff; letter-spacing: .06em; font-family: var(--fs-mono); font-size: 12.5px; }
.fs-note { padding: 10px 14px; border-radius: 10px; border: 1px solid var(--fs-border-strong); background: var(--fs-surface-2);
  color: var(--fs-text-2); font-size: 13px; margin-bottom: 12px; }
</style>
"""

LOGO_SVG = """<svg width="34" height="34" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M32 4 56 18v28L32 60 8 46V18L32 4Z" stroke="#22D3EE" stroke-width="3" fill="rgba(34,211,238,0.08)"/>
<path d="M14 32c4.5-7 10.5-11 18-11s13.5 4 18 11c-4.5 7-10.5 11-18 11s-13.5-4-18-11Z" stroke="#E6EAF0" stroke-width="2.6"/>
<circle cx="32" cy="32" r="6.5" fill="#22D3EE"/><circle cx="32" cy="32" r="2.4" fill="#0A0F1A"/>
<path d="M44 12l6 3.5" stroke="#F5A524" stroke-width="3" stroke-linecap="round"/></svg>"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def esc(value) -> str:
    return html.escape(str(value))


def chip(text: str, tone: str = "") -> str:
    return f'<span class="fs-chip {tone}"><span class="d"></span>{esc(text)}</span>'


def page_header(eyebrow: str, title: str, subtitle: str = "", chips: list[str] | None = None) -> None:
    chips_html = f'<div class="fs-chips">{"".join(chips)}</div>' if chips else ""
    st.markdown(
        f'<div class="fs-header"><div><div class="fs-eyebrow">{esc(eyebrow)}</div>'
        f'<h1 class="fs-title">{esc(title)}</h1>'
        f'{f"<div class=fs-subtitle>{esc(subtitle)}</div>" if subtitle else ""}</div>{chips_html}</div>',
        unsafe_allow_html=True,
    )


def kpi(label: str, value, sub: str = "", tone: str = "", help_text: str = "", unit: str = "") -> str:
    info = f'<span class="i" title="{esc(help_text)}">i</span>' if help_text else ""
    unit_html = f"<small>{esc(unit)}</small>" if unit else ""
    return (f'<div class="fs-kpi {tone}"><div class="fs-kpi-label">{esc(label)}{info}</div>'
            f'<div class="fs-kpi-value">{esc(value)}{unit_html}</div><div class="fs-kpi-sub">{esc(sub)}</div></div>')


def kpi_row(cards: list[str]) -> None:
    cols = st.columns(len(cards), gap="small")
    for col, card in zip(cols, cards):
        col.markdown(card, unsafe_allow_html=True)


def section(title: str, caption: str = "") -> None:
    st.markdown(f'<div class="fs-section"><h3>{esc(title)}</h3><span>{esc(caption)}</span></div>',
                unsafe_allow_html=True)


def empty_state(title: str, body: str, steps: list[tuple[str, str]] | None = None) -> None:
    steps_html = ""
    if steps:
        steps_html = '<div class="fs-steps">' + "".join(
            f'<div class="fs-step"><b>STEP {i:02d}</b><div>{esc(t)}</div><span>{esc(d)}</span></div>'
            for i, (t, d) in enumerate(steps, start=1)) + "</div>"
    st.markdown(f'<div class="fs-empty">{LOGO_SVG.replace("34", "52")}<h2>{esc(title)}</h2>'
                f'<p>{esc(body)}</p>{steps_html}</div>', unsafe_allow_html=True)


def key_values(rows: list[tuple[str, str]]) -> str:
    return '<dl class="fs-kv">' + "".join(f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in rows) + "</dl>"


def hazard_alert(run) -> None:
    """Red-orange strip summarising confirmed fire/smoke events of a run (nothing if none)."""
    hz = getattr(run, "hazards", None)
    if not getattr(run, "hazard_active", False) or hz is None or not len(hz):
        return
    from .annotate import fmt_ts

    first = hz.sort_values("start_time_s").iloc[0]
    kinds = " + ".join(sorted({h.upper() for h in hz["hazard"]}))
    mails = hz["email_status"].fillna("").value_counts().to_dict()
    mail_txt = ", ".join(f"{n} {k}" for k, n in mails.items() if k) or "no email attempted"
    st.markdown(f'<div class="fs-alert hazard"><b>POTENTIAL {esc(kinds)} DETECTED</b> {len(hz)} confirmed '
                f'event{"s" if len(hz) != 1 else ""} · first {esc(first.event_id)} at {fmt_ts(first.start_time_s)} '
                f'(score {first.max_score:.2f}) · email: {esc(mail_txt)} · beta, verify on video</div>',
                unsafe_allow_html=True)


def sev_badge(severity: str) -> str:
    """Small uppercase badge for a severity (Critical/High/Medium/Low) or a review status (Open/Reviewed)."""
    return f'<span class="fs-sev {esc(severity).lower()}">{esc(severity).upper()}</span>'


def card(key: str, **kwargs):
    """Bordered surface container (styled through its st-key-fs-card-* class)."""
    return st.container(border=True, key=f"fs-card-{key}", **kwargs)


def note(text: str) -> None:
    st.markdown(f'<div class="fs-note">{esc(text)}</div>', unsafe_allow_html=True)
