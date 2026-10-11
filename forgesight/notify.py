"""Email alerts for confirmed fire/smoke events (SMTP, standard library only).

Credentials are read from environment variables (preferred) or from Streamlit secrets
(``.streamlit/secrets.toml``, section ``[email]``). They are never written to settings, logs or reports.

Environment variables:
  FORGESIGHT_SMTP_HOST, FORGESIGHT_SMTP_PORT (default 587), FORGESIGHT_SMTP_SECURITY (starttls | ssl | none),
  FORGESIGHT_SMTP_USER, FORGESIGHT_SMTP_PASSWORD, FORGESIGHT_ALERT_FROM, FORGESIGHT_ALERT_TO (comma-separated)
"""
from __future__ import annotations

import csv
import os
import re
import smtplib
import socket
import ssl
import time
from dataclasses import dataclass, field
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from typing import Callable, Mapping

ENV = {
    "host": "FORGESIGHT_SMTP_HOST", "port": "FORGESIGHT_SMTP_PORT", "security": "FORGESIGHT_SMTP_SECURITY",
    "username": "FORGESIGHT_SMTP_USER", "password": "FORGESIGHT_SMTP_PASSWORD",
    "sender": "FORGESIGHT_ALERT_FROM", "recipients": "FORGESIGHT_ALERT_TO",
}
REQUIRED = ("host", "sender", "recipients")

SENT, FAILED = "sent", "failed"
NOT_CONFIGURED, DISABLED = "not configured", "disabled"
COOLDOWN, CAP = "suppressed (cooldown)", "suppressed (per-run cap)"

LOG_FIELDS = ["event_id", "hazard", "video_time_s", "attempted_at", "status", "detail", "recipients", "duration_s"]


@dataclass
class EmailConfig:
    host: str
    sender: str
    recipients: list[str]
    port: int = 587
    security: str = "starttls"  # starttls | ssl | none
    username: str = ""
    password: str = field(default="", repr=False)  # never printed
    timeout_s: float = 20.0
    source: str = "env"

    def masked_recipients(self) -> str:
        return ", ".join(mask_address(r) for r in self.recipients)


def mask_address(addr: str) -> str:
    name, _, domain = addr.partition("@")
    return f"{name[:1]}***@{domain}" if domain else "***"


def load_email_config(secrets: Mapping | None = None, env: Mapping | None = None) -> tuple[EmailConfig | None, str]:
    """Return (config, message). Environment variables take precedence over Streamlit secrets."""
    env = os.environ if env is None else env
    values, source = {k: env.get(v, "") for k, v in ENV.items()}, "environment variables"
    if not any(values[k] for k in REQUIRED) and secrets:
        values = {k: str(secrets.get(k, "") or "") for k in ENV}
        if isinstance(secrets.get("recipients"), (list, tuple)):
            values["recipients"] = ",".join(secrets["recipients"])
        source = "Streamlit secrets"
    values = {k: (v or "").strip() for k, v in values.items()}  # stray spaces/newlines from copy-paste
    missing = [ENV[k] for k in REQUIRED if not values.get(k)]
    if missing:
        return None, ("Email alerts are not configured (missing " + ", ".join(missing) + "). Environment variables "
                      "must be set in the same terminal that starts the app.")
    try:
        port = int(values.get("port") or 587)
    except ValueError:
        return None, f"Invalid SMTP port: {values.get('port')!r}."
    security = (values.get("security") or ("ssl" if port == 465 else "starttls")).lower()
    if security not in ("starttls", "ssl", "none"):
        return None, f"Invalid SMTP security mode: {security!r} (use starttls, ssl or none)."
    if port == 465 and security != "ssl":
        return None, "Port 465 uses implicit TLS: set FORGESIGHT_SMTP_SECURITY=ssl (or use port 587 with starttls)."
    if port == 587 and security == "ssl":
        return None, "Port 587 uses STARTTLS: set FORGESIGHT_SMTP_SECURITY=starttls (or use port 465 with ssl)."
    password = values.get("password", "")
    if _is_gmail(values["host"]):
        # Google shows App Passwords as 'abcd efgh ijkl mnop'; they contain no real spaces.
        password = password.replace(" ", "")
    recipients = [r.strip() for r in values["recipients"].split(",") if r.strip()]
    cfg = EmailConfig(host=values["host"], port=port, security=security, username=values.get("username", ""),
                      password=password, sender=values["sender"], recipients=recipients, source=source)
    return cfg, f"Configured from {source}: {cfg.host}:{cfg.port} ({cfg.security}) to {cfg.masked_recipients()}"


def build_alert_message(cfg: EmailConfig, event: dict, image_path: Path | None) -> EmailMessage:
    """``event`` keys: event_id, hazard, video_time_s, confidence, camera, source_video, run_id, detected_at."""
    from .annotate import fmt_ts

    msg = EmailMessage()
    hz = event["hazard"].upper()
    msg["Subject"] = (f"[ForgeSight AI] Potential {hz} detected - {event.get('camera', 'camera')} "
                      f"at {fmt_ts(event['video_time_s'])}")
    msg["From"] = cfg.sender
    msg["To"] = ", ".join(cfg.recipients)
    msg.set_content(
        f"ForgeSight AI confirmed a potential {event['hazard']} event during video analysis.\n\n"
        f"Event ID:        {event['event_id']}\n"
        f"Event type:      Potential {event['hazard']} (BETA detector)\n"
        f"Video timestamp: {fmt_ts(event['video_time_s'])} ({event['video_time_s']:.2f} s into the footage)\n"
        + (f"Confirmed at:    {fmt_ts(event['confirmed_time_s'])} (after consecutive positive checks)\n"
           if event.get("confirmed_time_s") is not None else "") +
        f"Detected at:     {event.get('detected_at', '')} (analysis time)\n"
        f"Confidence:      {event['confidence']:.2f} (max over the confirming checks)\n"
        f"Camera:          {event.get('camera', '')}\n"
        f"Source video:    {event.get('source_video', '')}\n"
        f"Run ID:          {event.get('run_id', '')}\n\n"
        "The attached evidence frame shows the detection. This is an automated alert from a prototype "
        "detector analysing recorded footage: verify on video before acting. It is not a certified fire "
        "detection or alarm system.\n"
    )
    if image_path and Path(image_path).exists():
        msg.add_attachment(Path(image_path).read_bytes(), maintype="image", subtype="jpeg",
                           filename=Path(image_path).name)
    return msg


def _is_gmail(host: str) -> bool:
    return host.lower().endswith(("gmail.com", "googlemail.com"))


class EmailDeliveryError(Exception):
    """A delivery failure with the SMTP step where it happened and an actionable explanation."""

    def __init__(self, stage: str, explanation: str, original: Exception):
        super().__init__(f"{stage}: {explanation}")
        self.stage, self.explanation, self.original = stage, explanation, original


def _authenticate(server: smtplib.SMTP, username: str, password: str) -> None:
    """Log in with ONE explicitly chosen mechanism.

    ``smtplib.SMTP.login`` silently tries every mechanism in turn. Gmail answers a bad password on AUTH PLAIN with
    ``535 5.7.8`` and then drops the connection when AUTH LOGIN is tried, so ``login`` raises
    ``SMTPServerDisconnected: Connection unexpectedly closed`` and the real 535 is lost. Using a single mechanism
    keeps the server's actual answer.
    """
    server.ehlo_or_helo_if_needed()
    if not server.has_extn("auth"):
        raise smtplib.SMTPNotSupportedError("The server did not offer AUTH on this connection.")
    offered = server.esmtp_features["auth"].upper().split()
    server.user, server.password = username, password
    for mech in ("PLAIN", "LOGIN"):
        if mech in offered:
            server.auth(mech, getattr(server, "auth_" + mech.lower()), initial_response_ok=True)
            return
    server.login(username, password)  # e.g. only CRAM-MD5 offered: let smtplib handle it


def _explain(stage: str, exc: Exception, cfg: EmailConfig) -> str:
    """Turn a low-level SMTP/socket error into a hint the user can act on."""
    gmail = _is_gmail(cfg.host)
    if isinstance(exc, smtplib.SMTPAuthenticationError):
        server_says = exc.smtp_error.decode(errors="replace").replace("\n", " ") if isinstance(
            exc.smtp_error, bytes) else str(exc.smtp_error)
        hint = ("Gmail rejected the username/password. Use a 16-character Google App Password (Google Account > "
                "Security > 2-Step Verification > App passwords), not your normal password, and set "
                "FORGESIGHT_SMTP_USER to the full Gmail address." if gmail else
                "The server rejected the username/password. Check FORGESIGHT_SMTP_USER and FORGESIGHT_SMTP_PASSWORD.")
        return f"authentication failed ({exc.smtp_code} {server_says[:140]}). {hint}"
    if isinstance(exc, smtplib.SMTPNotSupportedError):
        return (f"{exc} Port 587 needs FORGESIGHT_SMTP_SECURITY=starttls; port 465 needs ssl."
                if stage in ("starttls", "login") else str(exc))
    if isinstance(exc, ssl.SSLError):
        return (f"TLS error ({exc}). Port/security mismatch? Use 587 with starttls or 465 with ssl. An antivirus "
                "'mail shield' that intercepts TLS can also cause this.")
    if isinstance(exc, smtplib.SMTPServerDisconnected):
        return (f"the server closed the connection during {stage}. Check that host, port and security match "
                "(587 starttls / 465 ssl) and that no antivirus or firewall intercepts SMTP.")
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return f"timed out after {cfg.timeout_s:g} s during {stage} (network, firewall or wrong port)."
    if isinstance(exc, smtplib.SMTPRecipientsRefused):
        return "the server refused the recipient address(es): check FORGESIGHT_ALERT_TO."
    if isinstance(exc, smtplib.SMTPSenderRefused):
        return ("the server refused the sender address: for Gmail, FORGESIGHT_ALERT_FROM must be the account itself "
                "or a verified alias." if gmail else "the server refused the sender (FORGESIGHT_ALERT_FROM).")
    if isinstance(exc, OSError):
        return f"network error during {stage}: {exc}"
    return f"{type(exc).__name__}: {exc}"


def send_message(cfg: EmailConfig, msg: EmailMessage) -> None:
    """Send via SMTP: 587 → SMTP + EHLO + STARTTLS (default SSL context) + EHLO + login; 465 → SMTP_SSL.

    Raises EmailDeliveryError naming the failed step (connect / ehlo / starttls / login / send).
    """
    stage, server = "connect", None
    try:
        if cfg.security == "ssl":
            server = smtplib.SMTP_SSL(cfg.host, cfg.port, timeout=cfg.timeout_s,
                                      context=ssl.create_default_context())
        else:
            server = smtplib.SMTP(cfg.host, cfg.port, timeout=cfg.timeout_s)
        stage = "ehlo"
        server.ehlo()
        if cfg.security == "starttls":
            stage = "starttls"
            if not server.has_extn("starttls"):
                raise smtplib.SMTPNotSupportedError("The server does not offer STARTTLS on this port.")
            server.starttls(context=ssl.create_default_context())
            server.ehlo()  # capabilities (incl. AUTH) change after TLS
        if cfg.username:
            stage = "login"
            _authenticate(server, cfg.username, cfg.password)
        stage = "send"
        server.send_message(msg)
    except Exception as exc:
        raise EmailDeliveryError(stage, _explain(stage, exc, cfg), exc) from exc
    finally:
        if server is not None:
            try:
                server.quit()
            except Exception:
                pass


def _safe_error(exc: Exception, cfg: EmailConfig | None) -> str:
    text = str(exc) if isinstance(exc, EmailDeliveryError) else f"{type(exc).__name__}: {exc}"
    secret = cfg.password.replace(" ", "") if cfg else ""
    if secret:
        # also matches the password as typed, e.g. Google's spaced 'abcd efgh ijkl mnop' form
        text = re.sub(r"\s*".join(re.escape(ch) for ch in secret), "***", text)
    return text[:600]


class AlertDispatcher:
    """Decides whether to email each confirmed event (cooldown, per-run cap) and records every outcome."""

    def __init__(self, cfg: EmailConfig | None, enabled: bool, cooldown_s: float = 60.0, max_per_run: int = 5,
                 sender: Callable[[EmailConfig, EmailMessage], None] = send_message):
        self.cfg, self.enabled = cfg, enabled
        self.cooldown_s, self.max_per_run = float(cooldown_s), int(max_per_run)
        self._send = sender
        self._last_sent: dict[str, float] = {}
        self.sent = 0
        self.log: list[dict] = []

    def notify(self, event: dict, image_path: Path | None) -> dict:
        t, hz = float(event["video_time_s"]), event["hazard"]
        row = {"event_id": event["event_id"], "hazard": hz, "video_time_s": round(t, 2),
               "attempted_at": datetime.now().isoformat(timespec="seconds"), "status": "", "detail": "",
               "recipients": self.cfg.masked_recipients() if self.cfg else "", "duration_s": 0.0}
        last = self._last_sent.get(hz)
        if not self.enabled:
            row.update(status=DISABLED, detail="Email alerts are switched off in Settings.")
        elif self.cfg is None:
            row.update(status=NOT_CONFIGURED, detail="No SMTP configuration (environment variables or secrets).")
        elif last is not None and t - last < self.cooldown_s:
            row.update(status=COOLDOWN, detail=f"Last {hz} email {t - last:.1f} s earlier (cooldown {self.cooldown_s:g} s).")
        elif self.sent >= self.max_per_run:
            row.update(status=CAP, detail=f"Already sent {self.sent} emails in this run (cap {self.max_per_run}).")
        else:
            start = time.perf_counter()
            try:
                self._send(self.cfg, build_alert_message(self.cfg, event, image_path))
                row.update(status=SENT, detail=f"Delivered to SMTP server {self.cfg.host}:{self.cfg.port}.")
                self.sent += 1
                self._last_sent[hz] = t
            except Exception as exc:  # delivery failure must never stop the analysis
                row.update(status=FAILED, detail=_safe_error(exc, self.cfg))
            row["duration_s"] = round(time.perf_counter() - start, 2)
        self.log.append(row)
        return row

    def write_log(self, path: Path) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            w.writeheader()
            w.writerows(self.log)
