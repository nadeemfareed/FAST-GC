"""FAST-GIS progress adapter.

All GIS stages use FAST-GC's existing progress backend so PowerShell rendering,
CPU/RAM reporting, elapsed time, ETA, throughput, and fallback behavior remain
consistent with the production pipeline.
"""
from __future__ import annotations

from ..monster import progress_bar


def gis_progress(stage: str, total: int, *, unit: str = "item", enabled: bool = True):
    name = str(stage).strip().upper()
    if not name.startswith("FAST-GIS"):
        name = f"FAST-GIS {name}"
    return progress_bar(
        total=max(0, int(total)),
        desc=name,
        unit=str(unit),
        dynamic_ncols=True,
        leave=False,
        disable=not bool(enabled),
    )
