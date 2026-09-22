"""
Read-only discovery of completed FAST-GC products.

This module inspects files that already exist on disk.
It does not participate in classification, merging, raster creation,
or any other scientific FAST-GC computation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .manifest import ViewManifest


POINT_EXTENSIONS = {".las", ".laz", ".copc"}
RASTER_EXTENSIONS = {".tif", ".tiff"}

KNOWN_PRODUCTS = (
    "FAST_GC",
    "FAST_DEM",
    "FAST_NORMALIZED",
    "FAST_DSM",
    "FAST_CHM",
    "FAST_TERRAIN",
    "FAST_CHANGE",
    "FAST_ITD",
    "FAST_STRUCTURE",
    "FAST_TREECLOUDS",
)


def _files(root: Path, extensions: set[str]) -> list[Path]:
    if not root.exists():
        return []

    return sorted(
        (
            p.resolve()
            for p in root.rglob("*")
            if p.is_file()
            and p.suffix.lower() in extensions
        ),
        key=lambda p: str(p).lower(),
    )


def _path_has_product(
    path: Path,
    product: str,
) -> bool:
    """
    Match an existing output file to a FAST-GC product.

    Normal FAST-GC products are identified by an exact
    case-insensitive path component such as FAST_GC,
    FAST_DEM, FAST_CHM, or FAST_NORMALIZED.

    A conservative filename fallback is retained for merged
    outputs that may live outside a canonical product folder.

    Visualization metadata only; this function performs no
    scientific computation.
    """
    target = str(product).upper()

    parts = [
        part.upper()
        for part in path.parts
    ]

    # Canonical FAST-GC product-directory contract.
    if target in parts:
        return True

    # Conservative fallback for files whose own filename
    # explicitly begins with the product token.
    stem = path.stem.upper()

    if stem == target:
        return True

    if stem.startswith(target + "_"):
        return True

    return False


def _classify_source(path: Path) -> str:
    """
    Conservative path-based source classification.

    This affects visualization metadata only.
    It never changes FAST-GC processing.
    """

    parts = [part.lower() for part in path.parts]

    tile_tokens = (
        "tile",
        "tiles",
        "tiled",
        "workspace",
    )

    merge_tokens = (
        "merge",
        "merged",
        "final",
    )

    if any(
        any(token in part for token in merge_tokens)
        for part in parts
    ):
        return "merged"

    if any(
        any(token in part for token in tile_tokens)
        for part in parts
    ):
        return "tile"

    return "completed"


def _product_kind(product: str) -> str:
    if product in {
        "FAST_GC",
        "FAST_NORMALIZED",
        "FAST_TREECLOUDS",
    }:
        return "pointcloud"

    if product in {
        "FAST_DEM",
        "FAST_DSM",
        "FAST_CHM",
        "FAST_TERRAIN",
        "FAST_CHANGE",
    }:
        return "raster"

    return "mixed"


def discover_product_files(
    root,
    *,
    products: Iterable[str] = KNOWN_PRODUCTS,
) -> dict[str, dict[str, list[str]]]:
    """
    Discover existing FAST-GC outputs beneath *root*.

    Results are visualization metadata only.
    """

    root = Path(root).resolve()

    point_files = _files(root, POINT_EXTENSIONS)
    raster_files = _files(root, RASTER_EXTENSIONS)

    candidates = point_files + raster_files

    result = {}

    for product in products:
        product = str(product).upper()

        matched = [
            p for p in candidates
            if _path_has_product(p, product)
        ]

        if not matched:
            continue

        grouped = {
            "merged": [],
            "tiles": [],
            "completed": [],
        }

        for path in matched:
            source = _classify_source(path)

            if source == "merged":
                grouped["merged"].append(str(path))
            elif source == "tile":
                grouped["tiles"].append(str(path))
            else:
                grouped["completed"].append(str(path))

        result[product] = grouped

    return result


def build_view_manifest_from_output(
    root,
    *,
    dataset: str | None = None,
    sensor: str = "UNKNOWN",
) -> ViewManifest:
    """
    Build an in-memory visualization manifest from completed outputs.

    Nothing is written unless the caller explicitly calls manifest.write().
    """

    root = Path(root).resolve()

    if dataset is None:
        dataset = root.name

    discovered = discover_product_files(root)

    manifest = ViewManifest(
        dataset=str(dataset),
        sensor=str(sensor).upper(),
    )

    for product, sources in discovered.items():

        merged = None

        # We deliberately refuse to guess between multiple merged files.
        if len(sources["merged"]) == 1:
            merged = sources["merged"][0]

        tiles = list(sources["tiles"])

        # Files whose role cannot be established safely remain viewable.
        # They are included as tiles/completed resources rather than being
        # falsely declared the authoritative merged product.
        tiles.extend(sources["completed"])

        manifest.register_product(
            product,
            kind=_product_kind(product),
            merged=merged,
            tiles=tiles,
            status="complete",
        )

    return manifest
