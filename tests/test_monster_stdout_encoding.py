import io

import fastgc.monster as monster


class RestrictedStdout(io.StringIO):
    @property
    def encoding(self):
        return "cp1252"

    def write(self, text):
        text.encode(self.encoding, errors="strict")
        return super().write(text)


def _dashboard(monkeypatch, *, tty):
    stream = RestrictedStdout()
    monkeypatch.setattr(monster.sys, "stdout", stream)

    dashboard = monster.ProgressDashboard(
        "FAST_GC TLS",
        total=10,
        unit="step",
    )

    dashboard._native_win = False
    dashboard._tty = tty
    return dashboard, stream


def _unicode_progress_line():
    return (
        "FAST-GC "
        + "\u2588" * 5
        + "\u2591" * 2
        + " \u2192 TLS \u2713"
    )


def test_progress_dashboard_redirected_stdout_survives_restricted_encoding(monkeypatch):
    dashboard, stream = _dashboard(monkeypatch, tty=False)

    dashboard._last_snapshot_done = -1
    dashboard._write_line(_unicode_progress_line())

    value = stream.getvalue()

    assert "FAST-GC" in value
    assert "#####" in value
    assert "\n" in value

    # The restricted stream itself proves every emitted character
    # was encodable under cp1252.
    value.encode("cp1252", errors="strict")


def test_progress_dashboard_tty_survives_restricted_encoding(monkeypatch):
    dashboard, stream = _dashboard(monkeypatch, tty=True)

    dashboard._write_line(_unicode_progress_line())

    value = stream.getvalue()

    assert "FAST-GC" in value
    assert "#####" in value
    assert value.startswith("\r\x1b[2K")

    value.encode("cp1252", errors="strict")
