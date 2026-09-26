from fastgc.monster import (
    ProgressDashboard,
    compact_progress_name,
    progress_plot_context,
)


def test_compact_progress_name_preserves_plot_identity():
    assert compact_progress_name(
        r"E:\segtest\qualified\NSpruce_plot1.las"
    ) == "NSpruce_plot1"

    assert compact_progress_name("NSpruce_plot2") == "NSpruce_plot2"

    # Long names retain the identifying suffix rather than only the prefix.
    compacted = compact_progress_name(
        "VeryLongForestPlotName_027",
        max_len=14,
    )
    assert len(compacted) <= 14
    assert compacted.endswith("_027")
    assert "..." in compacted


def test_dashboard_inherits_scoped_plot_context():
    with progress_plot_context("NSpruce_plot2"):
        dashboard = ProgressDashboard(
            stage_name="FAST-GC | TLS",
            total=6,
            unit="tile",
            enabled=False,
        )

    assert dashboard.plot_name == "NSpruce_plot2"


def test_plot_context_resets_after_scope():
    with progress_plot_context("NSpruce_plot3"):
        inside = ProgressDashboard(
            stage_name="FAST-GC | TLS",
            total=1,
            unit="tile",
            enabled=False,
        )

    outside = ProgressDashboard(
        stage_name="FAST-GC | TLS",
        total=1,
        unit="tile",
        enabled=False,
    )

    assert inside.plot_name == "NSpruce_plot3"
    assert outside.plot_name == ""


def test_plot_context_resets_after_exception():
    try:
        with progress_plot_context("LPine_plot1"):
            raise RuntimeError("test failure")
    except RuntimeError:
        pass

    dashboard = ProgressDashboard(
        stage_name="FAST-GC | TLS",
        total=1,
        unit="tile",
        enabled=False,
    )

    assert dashboard.plot_name == ""


def test_render_keeps_plot_context_at_narrow_console(monkeypatch):
    with progress_plot_context("NSpruce_plot2"):
        dashboard = ProgressDashboard(
            stage_name="FAST-GC | TLS",
            total=100,
            unit="tile",
            enabled=False,
        )

    dashboard.done = 50

    monkeypatch.setattr(
        dashboard,
        "_console_width",
        lambda: 80,
    )

    rendered = dashboard._render_line()

    assert "plot=NSpruce_plot2" in rendered
    assert len(rendered) <= 80


def test_render_without_plot_context_remains_supported(monkeypatch):
    dashboard = ProgressDashboard(
        stage_name="FAST-GC | TLS",
        total=100,
        unit="tile",
        enabled=False,
    )

    dashboard.done = 50

    monkeypatch.setattr(
        dashboard,
        "_console_width",
        lambda: 80,
    )

    rendered = dashboard._render_line()

    assert "plot=" not in rendered
    assert len(rendered) <= 80
