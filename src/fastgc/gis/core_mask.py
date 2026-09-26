"""Core-plot raster masking for FAST-GIS plots-run."""

from __future__ import annotations

import json
import os
from pathlib import Path


_RASTER_SUFFIXES = {".tif", ".tiff"}


def load_core_mask_metadata(metadata_path: str | Path) -> tuple[dict, str]:
    """Load authoritative core geometry and processing CRS from plot metadata."""
    path = Path(metadata_path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    if payload.get("schema") != "fastgc.gis.plot":
        raise ValueError(f"Unsupported FAST-GIS plot metadata schema: {path}")

    if payload.get("point_extent") != "core_plus_buffer":
        raise ValueError(
            "Core raster masking requires point_extent=core_plus_buffer: "
            f"{path}"
        )

    geometry = payload.get("core_geometry")
    if not isinstance(geometry, dict) or not geometry.get("type"):
        raise ValueError(f"Missing core_geometry in plot metadata: {path}")

    processing_crs = payload.get("processing_crs")
    if not processing_crs:
        raise ValueError(f"Missing processing_crs in plot metadata: {path}")

    return geometry, str(processing_crs)


def mask_raster_to_core(
    raster_path: str | Path,
    *,
    core_geometry: dict,
    processing_crs: str,
) -> Path:
    """Mask and crop one raster in place to the authoritative plot core."""
    import rasterio
    from rasterio.crs import CRS
    from rasterio.mask import mask

    path = Path(raster_path)
    expected_crs = CRS.from_user_input(processing_crs)
    temp = path.with_name(f".{path.name}.coremask.tmp{path.suffix}")

    try:
        with rasterio.open(path) as src:
            raster_crs = src.crs if src.crs is not None else expected_crs

            if raster_crs != expected_crs:
                raise ValueError(
                    "Raster CRS does not match FAST-GIS processing CRS: "
                    f"{path} has {raster_crs}; expected {expected_crs}"
                )

            out_img, out_transform = mask(
                src,
                [core_geometry],
                crop=True,
                filled=True,
            )

            profile = src.profile.copy()
            profile["crs"] = expected_crs
            profile.update(
                height=out_img.shape[1],
                width=out_img.shape[2],
                transform=out_transform,
            )

            with rasterio.open(temp, "w", **profile) as dst:
                dst.write(out_img)

        os.replace(temp, path)
        return path
    finally:
        if temp.exists():
            temp.unlink()


def mask_plot_rasters_to_core(
    *,
    work_root: str | Path,
    metadata_path: str | Path,
) -> list[Path]:
    """Mask all private plots-run raster products to the core geometry."""
    work = Path(work_root)
    if not work.is_dir():
        return []

    core_geometry, processing_crs = load_core_mask_metadata(metadata_path)

    rasters = sorted(
        p
        for p in work.rglob("*")
        if p.is_file() and p.suffix.lower() in _RASTER_SUFFIXES
    )

    masked: list[Path] = []
    for raster_path in rasters:
        masked.append(
            mask_raster_to_core(
                raster_path,
                core_geometry=core_geometry,
                processing_crs=processing_crs,
            )
        )

    return masked

