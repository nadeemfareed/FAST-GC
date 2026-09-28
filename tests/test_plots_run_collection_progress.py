"""Tests for collection-level plots-run progress formatting."""

from __future__ import annotations


def test_plots_run_startup_format_contract():
    sensor_mode = "TLS"
    total_plots = 12
    resolved_products = ["FAST_GC", "FAST_DEM", "FAST_CHM"]
    in_path = r"E:\segtest\qualified"
    output = r"E:\segtest\qualified_FAST_GC\TLS_plots"

    product_label = ",".join(resolved_products)

    header = (
        f"FAST-GC PLOTS | {sensor_mode} | "
        f"{total_plots} plots | products={product_label}"
    )

    assert header == (
        "FAST-GC PLOTS | TLS | 12 plots | "
        "products=FAST_GC,FAST_DEM,FAST_CHM"
    )
    assert f"Input  : {in_path}" == r"Input  : E:\segtest\qualified"
    assert (
        f"Output : {output}"
        == r"Output : E:\segtest\qualified_FAST_GC\TLS_plots"
    )


def test_plots_run_current_plot_format_contract():
    index = 3
    total_plots = 12
    plot_name = "NSpruce_plot1"
    published_count = 2
    skipped_count = 0

    line = (
        f"[PLOT {index}/{total_plots}] {plot_name} | "
        f"completed={published_count} | skipped={skipped_count}"
    )

    assert line == (
        "[PLOT 3/12] NSpruce_plot1 | "
        "completed=2 | skipped=0"
    )


def test_plots_run_skipped_format_contract():
    index = 4
    total_plots = 12
    plot_name = "NSpruce_plot2"
    published_count = 2
    skipped_count = 2

    line = (
        f"[SKIPPED {index}/{total_plots}] {plot_name} | "
        f"completed={published_count} | skipped={skipped_count}"
    )

    assert line == (
        "[SKIPPED 4/12] NSpruce_plot2 | "
        "completed=2 | skipped=2"
    )


def test_plots_run_published_format_contract():
    index = 3
    total_plots = 12
    plot_name = "NSpruce_plot1"
    published_count = 3
    skipped_count = 0
    plots_elapsed = 135.4
    eta_text = "406s"

    line = (
        f"[PUBLISHED {index}/{total_plots}] {plot_name} | "
        f"completed={published_count} | skipped={skipped_count} | "
        f"elapsed={plots_elapsed:.1f}s | ETA={eta_text}"
    )

    assert line == (
        "[PUBLISHED 3/12] NSpruce_plot1 | "
        "completed=3 | skipped=0 | "
        "elapsed=135.4s | ETA=406s"
    )


def test_plots_run_eta_uses_only_processed_plots():
    published_count = 2
    skipped_count = 7
    total_plots = 12
    processed_seconds = 100.0

    attempted = published_count + skipped_count
    remaining = max(0, total_plots - attempted)

    mean_processed_seconds = processed_seconds / published_count
    eta_seconds = mean_processed_seconds * remaining

    assert attempted == 9
    assert remaining == 3
    assert mean_processed_seconds == 50.0
    assert eta_seconds == 150.0


def test_plots_run_final_summary_contract():
    completed = ["A", "B", "C", "D"]
    total_plots = 4
    published_count = 3
    skipped_count = 1
    elapsed = 91.25

    line = (
        f"[TIME] WORKFLOW plots-run: "
        f"{elapsed:.2f}s | "
        f"completed={len(completed)}/{total_plots} | "
        f"published={published_count} | skipped={skipped_count}"
    )

    assert line == (
        "[TIME] WORKFLOW plots-run: 91.25s | "
        "completed=4/4 | published=3 | skipped=1"
    )
