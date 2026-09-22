"""
Viewer-neutral FAST-GC visualization recipe.

This module reads visualization metadata already produced by FAST-GC and
creates a declarative recipe for downstream viewers.

It performs no scientific computation and is never consumed by FAST-GC
classification.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


RECIPE_SCHEMA = "fastgc-view-recipe"
RECIPE_SCHEMA_VERSION = 1


def _load_manifest(path) -> tuple[Path, dict]:
    path = Path(path).resolve()

    if not path.is_file():
        raise FileNotFoundError(path)

    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    if data.get("schema") != "fastgc-view-manifest":
        raise ValueError(
            "Input is not a FAST-GC view manifest."
        )

    if data.get("schema_version") != 1:
        raise ValueError(
            "Unsupported FAST-GC view-manifest version: "
            f"{data.get('schema_version')!r}"
        )

    products = data.get("products")

    if not isinstance(products, dict):
        raise ValueError(
            "Manifest products field must be a dictionary."
        )

    return path, data


def _existing_paths(record: dict) -> list[str]:
    """
    Return unique existing product files in deterministic order.

    A merged product is preferred when present. Tile/completed paths are
    retained as additional sources only when they are distinct.
    """
    paths = []

    merged = record.get("merged")

    if merged:
        p = Path(merged).resolve()

        if p.is_file():
            paths.append(str(p))

    tiles = record.get("tiles", [])

    if not isinstance(tiles, list):
        raise ValueError(
            "Product tiles field must be a list."
        )

    for value in tiles:
        p = Path(value).resolve()

        if p.is_file():
            value = str(p)

            if value not in paths:
                paths.append(value)

    return paths


def _source_kind(path: str) -> str:
    suffix = Path(path).suffix.lower()

    if suffix in {".las", ".laz", ".copc"}:
        return "pointcloud"

    if suffix in {".tif", ".tiff"}:
        return "raster"

    return "other"


def build_view_recipe(
    manifest_path,
    *,
    products: Iterable[str] | None = None,
) -> dict:
    """
    Build a viewer-neutral recipe from FASTGC_VIEW_MANIFEST.json.

    ``products`` optionally limits the recipe to selected FAST-GC products.
    Product names are case-insensitive.

    No FAST-GC scientific output is created or modified.
    """
    manifest_path, manifest = _load_manifest(
        manifest_path
    )

    requested = None

    if products is not None:
        requested = {
            str(product).upper()
            for product in products
        }

    layers = []

    for product, record in sorted(
        manifest["products"].items()
    ):
        product = str(product).upper()

        if (
            requested is not None
            and product not in requested
        ):
            continue

        if not isinstance(record, dict):
            continue

        paths = _existing_paths(record)

        if not paths:
            continue

        sources = [
            {
                "path": path,
                "kind": _source_kind(path),
            }
            for path in paths
        ]

        layers.append(
            {
                "id": product.lower(),
                "product": product,
                "kind": record.get(
                    "kind",
                    sources[0]["kind"],
                ),
                "visible": True,
                "sources": sources,
            }
        )

    return {
        "schema": RECIPE_SCHEMA,
        "schema_version": RECIPE_SCHEMA_VERSION,
        "dataset": manifest.get("dataset"),
        "sensor": manifest.get("sensor"),
        "manifest": str(manifest_path),
        "layers": layers,
    }


def write_view_recipe(
    manifest_path,
    output_path=None,
    *,
    products: Iterable[str] | None = None,
) -> Path:
    """
    Build and atomically write a FAST-GC visualization recipe.
    """
    manifest_path = Path(manifest_path).resolve()

    if output_path is None:
        output_path = (
            manifest_path.parent /
            "FASTGC_VIEW_RECIPE.json"
        )

    output_path = Path(output_path).resolve()

    recipe = build_view_recipe(
        manifest_path,
        products=products,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = output_path.with_name(
        output_path.name + ".tmp"
    )

    tmp.write_text(
        json.dumps(recipe, indent=2),
        encoding="utf-8",
    )

    tmp.replace(output_path)

    return output_path
