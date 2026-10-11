"""Mocked SMTP flow: connection method per port, EHLO/STARTTLS/EHLO order, single-mechanism login, and error
reporting. Includes the Gmail case where smtplib.login() hid a 535 behind 'Connection unexpectedly closed'."""
import smtplib
import ssl
from email.message import EmailMessage

import pytest

from forgesight import notify

SECRET = "abcd efgh ijkl mnop"  # how Google displays an App Password


class FakeSMTP:
    """Records the calls made by notify.send_message; behaviour is configured per test."""

    instances: list = []
    features = {"starttls": "", "auth": " LOGIN PLAIN XOAUTH2"}  # features after STARTTLS
    pre_tls_features = {"starttls": ""}
    fail_at: dict = {}  # method name -> exception to raise
    implicit_tls = False

    def __init__(self, host, port, timeout=None, context=None):
        if "connect" in self.fail_at:
            raise self.fail_at["connect"]
        self.host, self.port, self.timeout, self.context = host, port, timeout, context
        self.calls = [("connect", host, port)]
        self.tls = self.implicit_tls
        self.esmtp_features = {}
        self.user = self.password = None
        FakeSMTP.instances.append(self)

    def _maybe_fail(self, name):
        if name in self.fail_at:
            raise self.fail_at[name]

    def ehlo(self, name=""):
        self.calls.append(("ehlo",))
        self._maybe_fail("ehlo")
        self.esmtp_features = dict(self.features if self.tls else self.pre_tls_features)
        return 250, b"ok"

    def ehlo_or_helo_if_needed(self):
        if not self.esmtp_features:
            self.ehlo()

    def has_extn(self, opt):
        return opt.lower() in self.esmtp_features

    def starttls(self, context=None):
        self.calls.append(("starttls", context))
        self._maybe_fail("starttls")
        self.tls, self.esmtp_features = True, {}
        return 220, b"ready"

    def auth_plain(self, challenge=None):
        return f"\0{self.user}\0{self.password}"

    def auth_login(self, challenge=None):
        return self.user

    def auth(self, mechanism, authobject, *, initial_response_ok=True):
        self.calls.append(("auth", mechanism, self.user, self.password))
        self._maybe_fail("auth")
        return 235, b"accepted"

    def login(self, user, password):
        self.calls.append(("login", user))
        self._maybe_fail("login")
        return 235, b"accepted"

    def send_message(self, msg):
        self.calls.append(("send_message", msg["To"]))
        self._maybe_fail("send_message")

    def quit(self):
        self.calls.append(("quit",))


class FakeSMTP_SSL(FakeSMTP):
    implicit_tls = True


@pytest.fixture(autouse=True)
def fake_smtplib(monkeypatch):
    FakeSMTP.instances, FakeSMTP.fail_at = [], {}
    FakeSMTP.features = {"starttls": "", "auth": " LOGIN PLAIN XOAUTH2"}
    monkeypatch.setattr(notify.smtplib, "SMTP", FakeSMTP)
    monkeypatch.setattr(notify.smtplib, "SMTP_SSL", FakeSMTP_SSL)


def gmail_cfg(port=587):
    env = {"FORGESIGHT_SMTP_HOST": "smtp.gmail.com", "FORGESIGHT_SMTP_PORT": str(port),
           "FORGESIGHT_SMTP_USER": "me@gmail.com", "FORGESIGHT_SMTP_PASSWORD": f"  {SECRET} \n",
           "FORGESIGHT_ALERT_FROM": "me@gmail.com", "FORGESIGHT_ALERT_TO": "ehs@example.com"}
    cfg, msg = notify.load_email_config(env=env)
    assert cfg is not None, msg
    return cfg


def message():
    m = EmailMessage()
    m["To"], m["Subject"] = "ehs@example.com", "t"
    m.set_content("x")
    return m


def names(server):
    return [c[0] for c in server.calls]


def test_port_587_uses_smtp_then_ehlo_starttls_ehlo_login():
    notify.send_message(gmail_cfg(587), message())
    (s,) = FakeSMTP.instances
    assert type(s) is FakeSMTP and (s.host, s.port) == ("smtp.gmail.com", 587)
    assert names(s) == ["connect", "ehlo", "starttls", "ehlo", "auth", "send_message", "quit"]
    starttls = next(c for c in s.calls if c[0] == "starttls")
    assert isinstance(starttls[1], ssl.SSLContext) and starttls[1].check_hostname
    auth = next(c for c in s.calls if c[0] == "auth")
    assert auth[1] == "PLAIN" and auth[2] == "me@gmail.com"
    assert auth[3] == "abcdefghijklmnop"  # app-password spaces and surrounding whitespace removed


def test_port_465_uses_smtp_ssl_without_starttls():
    notify.send_message(gmail_cfg(465), message())
    (s,) = FakeSMTP.instances
    assert type(s) is FakeSMTP_SSL and isinstance(s.context, ssl.SSLContext)
    assert names(s) == ["connect", "ehlo", "auth", "send_message", "quit"]


def test_gmail_535_is_reported_not_masked_as_disconnect():
    """Only ONE auth attempt: a second mechanism would make Gmail hang up and hide the 535."""
    FakeSMTP.fail_at = {"auth": smtplib.SMTPAuthenticationError(
        535, b"5.7.8 Username and Password not accepted. BadCredentials")}
    with pytest.raises(notify.EmailDeliveryError) as ei:
        notify.send_message(gmail_cfg(), message())
    e = ei.value
    assert e.stage == "login" and isinstance(e.original, smtplib.SMTPAuthenticationError)
    assert "535" in str(e) and "App Password" in str(e) and "unexpectedly closed" not in str(e)
    s = FakeSMTP.instances[0]
    assert names(s).count("auth") == 1 and "login" not in names(s) and names(s)[-1] == "quit"


def test_dispatcher_records_explained_failure_without_secret():
    FakeSMTP.fail_at = {"auth": smtplib.SMTPAuthenticationError(535, f"bad creds for {SECRET}".encode())}
    cfg = gmail_cfg()
    row = notify.AlertDispatcher(cfg, enabled=True).notify(
        {"event_id": "FIRE-001", "hazard": "fire", "video_time_s": 1.0, "confidence": 0.7}, None)
    assert row["status"] == notify.FAILED and row["detail"].startswith("login: authentication failed (535")
    assert SECRET not in row["detail"] and SECRET.replace(" ", "") not in row["detail"]


@pytest.mark.parametrize("where,exc,expect", [
    ("connect", TimeoutError("timed out"), "connect: timed out"),
    ("connect", ConnectionRefusedError(10061, "refused"), "connect: network error"),
    ("ehlo", smtplib.SMTPServerDisconnected("Connection unexpectedly closed"), "closed the connection during ehlo"),
    ("starttls", ssl.SSLError("WRONG_VERSION_NUMBER"), "TLS error"),
    ("send_message", smtplib.SMTPRecipientsRefused({"x": (550, b"no")}), "refused the recipient"),
    ("send_message", smtplib.SMTPSenderRefused(553, b"not yours", "a@b"), "refused the sender address"),
])
def test_each_failure_names_its_step(where, exc, expect):
    FakeSMTP.fail_at = {where: exc}
    with pytest.raises(notify.EmailDeliveryError) as ei:
        notify.send_message(gmail_cfg(), message())
    assert expect in str(ei.value)
    for s in FakeSMTP.instances:  # connection always closed after a failure
        assert names(s)[-1] == "quit"


def test_missing_starttls_or_auth_is_explained():
    FakeSMTP.pre_tls_features = {}
    try:
        with pytest.raises(notify.EmailDeliveryError) as ei:
            notify.send_message(gmail_cfg(), message())
        assert ei.value.stage == "starttls" and "STARTTLS" in str(ei.value)
    finally:
        FakeSMTP.pre_tls_features = {"starttls": ""}
    FakeSMTP.features = {"starttls": ""}  # no AUTH even after TLS
    with pytest.raises(notify.EmailDeliveryError) as ei:
        notify.send_message(gmail_cfg(), message())
    assert ei.value.stage == "login" and "AUTH" in str(ei.value)


def test_no_username_skips_login():
    cfg = notify.EmailConfig(host="relay.local", port=587, sender="a@b.c", recipients=["d@e.f"])
    notify.send_message(cfg, message())
    assert "auth" not in names(FakeSMTP.instances[0])


@pytest.mark.parametrize("port,security,ok", [
    ("587", "", True), ("465", "", True), ("587", "starttls", True), ("465", "ssl", True),
    ("465", "starttls", False), ("587", "ssl", False), ("25", "none", True),
])
def test_port_security_combinations(port, security, ok):
    env = {"FORGESIGHT_SMTP_HOST": "smtp.gmail.com", "FORGESIGHT_SMTP_PORT": port, "FORGESIGHT_SMTP_SECURITY": security,
           "FORGESIGHT_ALERT_FROM": "a@gmail.com", "FORGESIGHT_ALERT_TO": "b@x.com"}
    cfg, msg = notify.load_email_config(env=env)
    assert (cfg is not None) == ok, msg


def test_non_gmail_password_spaces_are_kept():
    env = {"FORGESIGHT_SMTP_HOST": "smtp.office365.com", "FORGESIGHT_SMTP_PASSWORD": "pass with spaces",
           "FORGESIGHT_ALERT_FROM": "a@x.com", "FORGESIGHT_ALERT_TO": "b@x.com"}
    cfg, _ = notify.load_email_config(env=env)
    assert cfg.password == "pass with spaces"


def test_not_configured_message_explains_terminal_scope():
    cfg, msg = notify.load_email_config(env={})
    assert cfg is None and "same terminal" in msg
