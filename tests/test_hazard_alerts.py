"""Fire/smoke confirmation, email alerts (mocked + a real SMTP round trip on localhost), and pipeline integration."""
import json
import socket
import socketserver
import threading
from email import message_from_bytes
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

import forgesight.pipeline as pipeline
from forgesight import incidents as inc
from forgesight import notify
from forgesight.config import Settings
from forgesight.hazard import HazardMonitor
from forgesight.results import load_run
from test_pipeline_events import FPS, H, N, W, ZONE, ScriptedModel, script

SECRET = "s3cr3t-pa55"


def cfg(**kw):
    base = dict(host="smtp.example.com", sender="alerts@example.com", recipients=["ops@example.com"],
                username="user", password=SECRET)
    base.update(kw)
    return notify.EmailConfig(**base)


# ------------------------------------------------------------------ confirmation logic

def feed(mon, seq, cls="fire", dt=0.5):
    out = []
    for i, v in enumerate(seq):
        out += mon.update(i * dt, {cls: v})
    return out


def test_single_and_double_blips_are_not_confirmed():
    assert feed(HazardMonitor(3, 4), [0.9, 0, 0.8, 0.9, 0, 0, 0.7, 0]) == []


def test_persistent_fire_is_one_event():
    mon = HazardMonitor(3, 4)
    eps = feed(mon, [0.6] * 40)
    mon.finish()
    assert len(eps) == 1 and eps[0].start_t == 0.0 and eps[0].confirmed_t == 1.0
    assert mon.episodes[0].end_t == 19.5 and mon.episodes[0].positive_checks == 40


def test_short_gap_keeps_episode_long_gap_starts_new_one():
    mon = HazardMonitor(3, 4)
    eps = feed(mon, [0.7] * 5 + [0] * 3 + [0.7] * 5)  # 3 negatives < clear_checks
    assert len(eps) == 1
    eps += feed(HazardMonitor(3, 4), [0.7] * 5 + [0] * 4 + [0.7] * 5)
    assert len(eps) == 1 + 2


def test_classes_are_independent():
    mon = HazardMonitor(2, 2)
    new = []
    for i in range(4):
        new += mon.update(i, {"fire": 0.8, "smoke": 0.8 if i >= 2 else 0})
    assert sorted(e.cls for e in new) == ["fire", "smoke"]


# ------------------------------------------------------------------ configuration & secrets

ENV_OK = {"FORGESIGHT_SMTP_HOST": "smtp.example.com", "FORGESIGHT_SMTP_PORT": "465",
          "FORGESIGHT_SMTP_USER": "user", "FORGESIGHT_SMTP_PASSWORD": SECRET,
          "FORGESIGHT_ALERT_FROM": "alerts@example.com", "FORGESIGHT_ALERT_TO": "a@x.com, b@y.org"}


def test_config_from_environment():
    c, msg = notify.load_email_config(env=ENV_OK)
    assert c.port == 465 and c.security == "ssl" and c.recipients == ["a@x.com", "b@y.org"]
    assert SECRET not in msg and SECRET not in repr(c)
    assert "a***@x.com" in msg


def test_environment_takes_precedence_over_secrets():
    c, _ = notify.load_email_config(secrets={"host": "other", "sender": "s@x", "recipients": ["r@x"]}, env=ENV_OK)
    assert c.host == "smtp.example.com" and c.source == "environment variables"


def test_config_from_streamlit_secrets_and_missing_values():
    c, _ = notify.load_email_config(secrets={"host": "h", "port": 587, "sender": "s@x.com",
                                             "recipients": ["r@x.com"], "password": SECRET}, env={})
    assert c.source == "Streamlit secrets" and c.security == "starttls"
    none, msg = notify.load_email_config(secrets=None, env={"FORGESIGHT_SMTP_HOST": "h"})
    assert none is None and "FORGESIGHT_ALERT_FROM" in msg and "FORGESIGHT_ALERT_TO" in msg
    bad, msg = notify.load_email_config(env={**ENV_OK, "FORGESIGHT_SMTP_PORT": "abc"})
    assert bad is None and "port" in msg.lower()


def test_alert_message_content(tmp_path):
    img = tmp_path / "ev.jpg"
    cv2.imwrite(str(img), np.zeros((20, 20, 3), np.uint8))
    m = notify.build_alert_message(cfg(), {"event_id": "FIRE-001", "hazard": "fire", "video_time_s": 12.4,
                                          "confidence": 0.71, "camera": "CAM-01", "source_video": "x.mp4",
                                          "run_id": "r1", "detected_at": "2026-10-10T10:00:00"}, img)
    assert "Potential FIRE" in m["Subject"] and "00:12.4" in m["Subject"]
    body = m.get_body(("plain",)).get_content()
    assert "FIRE-001" in body and "0.71" in body and "00:12.4" in body and "verify" in body.lower()
    att = list(m.iter_attachments())
    assert len(att) == 1 and att[0].get_content_type() == "image/jpeg" and att[0].get_filename() == "ev.jpg"


# ------------------------------------------------------------------ dispatcher (mocked delivery)

def event(eid="FIRE-001", hazard="fire", t=10.0):
    return {"event_id": eid, "hazard": hazard, "video_time_s": t, "confidence": 0.7}


def test_dispatcher_statuses():
    sent = []
    d = notify.AlertDispatcher(cfg(), enabled=True, cooldown_s=60, max_per_run=2,
                               sender=lambda c, m: sent.append(m))
    assert d.notify(event("FIRE-001", t=10), None)["status"] == notify.SENT
    assert d.notify(event("FIRE-002", t=40), None)["status"] == notify.COOLDOWN  # 30 s < 60 s
    assert d.notify(event("SMOKE-001", "smoke", t=41), None)["status"] == notify.SENT  # other class
    assert d.notify(event("FIRE-003", t=200), None)["status"] == notify.CAP  # 2 already sent
    assert len(sent) == 2 and [r["status"] for r in d.log] == ["sent", notify.COOLDOWN, "sent", notify.CAP]


def test_failure_is_recorded_without_secret_and_does_not_start_cooldown():
    calls = {"n": 0}

    def flaky(c, m):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError(f"auth failed for password {SECRET}")

    d = notify.AlertDispatcher(cfg(), enabled=True, cooldown_s=60, sender=flaky)
    r1 = d.notify(event(t=10), None)
    assert r1["status"] == notify.FAILED and SECRET not in r1["detail"] and "***" in r1["detail"]
    assert d.notify(event("FIRE-002", t=12), None)["status"] == notify.SENT  # failure did not set cooldown


def test_not_configured_and_disabled():
    assert notify.AlertDispatcher(None, enabled=True).notify(event(), None)["status"] == notify.NOT_CONFIGURED
    assert notify.AlertDispatcher(cfg(), enabled=False).notify(event(), None)["status"] == notify.DISABLED


# ------------------------------------------------------------------ real SMTP round trip (localhost only)

class _SMTPHandler(socketserver.StreamRequestHandler):
    def handle(self):
        def say(line):
            self.wfile.write((line + "\r\n").encode())
        say("220 test ESMTP")
        while True:
            line = self.rfile.readline().decode(errors="replace").strip()
            cmd = line.upper()
            if not line:
                return
            if cmd.startswith(("EHLO", "HELO")):
                say("250 test")
            elif cmd.startswith(("MAIL FROM", "RCPT TO")):
                self.server.envelope.append(line)
                say("250 OK")
            elif cmd == "DATA":
                say("354 end with .")
                data = b""
                while True:
                    chunk = self.rfile.readline()
                    if chunk in (b".\r\n", b".\n", b""):
                        break
                    data += chunk[1:] if chunk.startswith(b"..") else chunk
                self.server.messages.append(data)
                say("250 queued")
            elif cmd == "QUIT":
                say("221 bye")
                return
            else:
                say("250 OK")


def test_real_smtp_round_trip_on_localhost(tmp_path):
    srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), _SMTPHandler)
    srv.messages, srv.envelope = [], []
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        img = tmp_path / "FIRE-001.jpg"
        cv2.imwrite(str(img), np.full((40, 40, 3), 200, np.uint8))
        c = notify.EmailConfig(host="127.0.0.1", port=srv.server_address[1], security="none",
                               sender="alerts@forgesight.local", recipients=["ops@forgesight.local"], timeout_s=5)
        d = notify.AlertDispatcher(c, enabled=True)
        row = d.notify({"event_id": "FIRE-001", "hazard": "fire", "video_time_s": 3.5, "confidence": 0.66}, img)
        assert row["status"] == notify.SENT, row
        msg = message_from_bytes(srv.messages[0])
        assert "Potential FIRE" in msg["Subject"] and msg["To"] == "ops@forgesight.local"
        att = [p for p in msg.walk() if p.get_content_type() == "image/jpeg"]
        assert len(att) == 1 and att[0].get_payload(decode=True) == img.read_bytes()
        assert any("ops@forgesight.local" in e for e in srv.envelope)
    finally:
        srv.shutdown()
        srv.server_close()


def test_unreachable_smtp_server_fails_cleanly():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()  # nothing listens here now
    c = notify.EmailConfig(host="127.0.0.1", port=port, security="none", sender="a@b.c", recipients=["d@e.f"],
                           timeout_s=2)
    row = notify.AlertDispatcher(c, enabled=True).notify(event(), None)
    assert row["status"] == notify.FAILED and row["detail"].startswith("connect:")


# ------------------------------------------------------------------ pipeline integration

class ScriptedHazard:
    """Fire on frames 10-40 and 46-55, smoke blip on 20-21 (never confirmed) and smoke on 50-59."""

    def __init__(self, person_model):
        self.pm = person_model

    def scores(self, frame):
        f = self.pm.i - 1
        fire = 0.8 if (10 <= f <= 40 or 46 <= f <= 55) else 0.1
        smoke = 0.7 if (20 <= f <= 21 or 50 <= f <= 59) else 0.1
        return {"fire": fire, "smoke": smoke, "normal": max(0.0, 1 - fire - smoke)}


@pytest.fixture(scope="module")
def hz_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("hazard")
    video = tmp / "synthetic.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for _ in range(N):
        w.write(np.full((H, W, 3), 60, np.uint8))
    w.release()
    (tmp / "runs").mkdir()
    mp = pytest.MonkeyPatch()
    mp.setattr(pipeline, "RUNS_DIR", tmp / "runs")
    sent = []
    settings = Settings(analysis_fps=FPS, zone_points=ZONE, hazard_enabled=True, hazard_rate_fps=FPS,
                        hazard_classes=["fire", "smoke"],
                        hazard_threshold=0.5, hazard_confirm_checks=3, hazard_clear_checks=4,
                        email_enabled=True, email_cooldown_s=120, email_max_per_run=5)
    dispatcher = notify.AlertDispatcher(cfg(), enabled=True, cooldown_s=settings.email_cooldown_s,
                                        max_per_run=settings.email_max_per_run, sender=lambda c, m: sent.append(m))
    pm = ScriptedModel(script())
    run = load_run(pipeline.analyze_video(video, settings, pm, hazard_model=ScriptedHazard(pm),
                                          alert_dispatcher=dispatcher))
    mp.undo()
    return run, sent


def test_hazard_events_confirmed_once_per_episode(hz_run):
    run, _ = hz_run
    h = run.hazards.set_index("event_id")
    assert list(h.index) == ["FIRE-001", "FIRE-002", "SMOKE-001"]  # in order of confirmation
    f1 = h.loc["FIRE-001"]
    assert (f1.start_time_s, f1.confirmed_time_s, f1.end_time_s) == (1.0, 1.2, 4.0)
    assert h.loc["FIRE-002"].start_time_s == pytest.approx(4.6)
    assert h.loc["SMOKE-001"].start_time_s == pytest.approx(5.0)  # the 20-21 blip was never confirmed
    for r in h.itertuples():
        assert cv2.imread(str(run.path(r.evidence_image))) is not None


def test_email_outcomes_recorded_per_event(hz_run):
    run, sent = hz_run
    status = dict(zip(run.hazards["event_id"], run.hazards["email_status"]))
    assert status == {"FIRE-001": notify.SENT, "FIRE-002": notify.COOLDOWN, "SMOKE-001": notify.SENT}
    assert len(sent) == 2 and all(len(list(m.iter_attachments())) == 1 for m in sent)
    body = sent[0].get_body(("plain",)).get_content()
    assert "FIRE-001" in body and "Confirmed at:    00:01.2" in body and "00:01.0" in body
    log = run.alerts.set_index("event_id")
    assert dict(log["status"]) == status
    assert run.summary["hazard"]["email"]["outcomes"] == {notify.SENT: 2, notify.COOLDOWN: 1}


def test_secrets_never_persisted(hz_run):
    run, _ = hz_run
    for f in run.run_dir.rglob("*"):
        if f.suffix in (".csv", ".json", ".txt"):
            assert SECRET not in f.read_text(encoding="utf-8", errors="ignore"), f
    assert "password" not in json.dumps(Settings().to_dict()).lower()


def test_hazard_incidents_and_existing_features_intact(hz_run):
    run, _ = hz_run
    df = inc.build_incidents(run, Settings(), with_evidence=False).set_index("incident_id")
    assert df.loc["FIRE-001", "severity"] == "Critical" and df.loc["SMOKE-001", "severity"] == "High"
    assert df.loc["FIRE-001", "notification"] == notify.SENT
    # zone events from the same scripted people are unchanged by the hazard feature
    assert run.metrics["zone_entries"] == 2 and run.metrics["zone_initial_occupants"] == 1
    assert len(run.hazard_checks) == N  # one check per analyzed frame at this rate


def test_runs_without_hazard_files_still_load(tmp_path):
    from test_pipeline_events import _write_legacy_run
    _write_legacy_run(tmp_path / "old")
    run = load_run(tmp_path / "old")
    assert run.hazards is None and not run.hazard_active
    assert "notification" in inc.build_incidents(run, Settings(), with_evidence=False).columns


def test_smoke_off_by_default_ignores_smoke_scores():
    assert Settings().hazard_classes == ["fire"]
    mon = HazardMonitor(2, 2, classes=["fire"])
    eps = []
    for i in range(5):
        eps += mon.update(i, {"fire": 0.0, "smoke": 0.9})
    assert eps == [] and not mon.active("smoke")


def test_report_and_package_include_fire_events(hz_run):
    import io
    import zipfile

    from forgesight import report
    run, _ = hz_run
    html = report.build_report_html(run, Settings())
    for eid in ("FIRE-001", "FIRE-002", "SMOKE-001"):
        assert eid in html
    assert "Fire &amp; smoke checks" in html and notify.COOLDOWN in html
    assert SECRET not in html
    z = zipfile.ZipFile(io.BytesIO(report.build_data_package(run, Settings(), inc.build_incidents(run, Settings()),
                                                             html)))
    assert {"hazards.csv", "hazard_checks.csv", "alerts.csv"} <= set(z.namelist())
