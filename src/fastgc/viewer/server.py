"""
Local HTTP serving boundary for FAST-GC visualization.

This module serves existing FAST-GC visualization products to a browser.
It performs no classification, terrain processing, product derivation,
or modification of scientific outputs.
"""

from __future__ import annotations

import argparse
import functools
import json
import mimetypes
import re
import shutil
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Optional
from urllib.parse import quote

from .recipe import build_view_recipe, write_view_recipe


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


def _safe_relative(path: Path, root: Path) -> str:
    path = path.resolve()
    root = root.resolve()

    try:
        rel = path.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"Viewer source is outside served root: {path}"
        ) from exc

    return rel.as_posix()



def _inspect_spatial_metadata(recipe: dict) -> dict:
    """
    Inspect existing visualization products for CRS and bounds.

    Header/metadata access only. Point coordinates are not loaded into
    memory and no FAST-GC scientific output is modified.
    """
    result = {
        "crs": None,
        "bounds": None,
        "point_count": None,
        "sources": [],
    }

    xmin = None
    ymin = None
    zmin = None
    xmax = None
    ymax = None
    zmax = None

    detected_crs = []

    for layer in recipe.get("layers", []):
        for source in layer.get("sources", []):
            path = Path(source["path"]).resolve()

            if not path.is_file():
                continue

            suffix = path.suffix.lower()

            info = {
                "product": layer.get("product"),
                "path": str(path),
                "kind": source.get("kind"),
            }

            if suffix in {".las", ".laz"}:
                try:
                    import laspy

                    with laspy.open(path) as reader:
                        header = reader.header

                        mins = [
                            float(v)
                            for v in header.mins
                        ]

                        maxs = [
                            float(v)
                            for v in header.maxs
                        ]

                        count = int(header.point_count)

                        info["bounds"] = {
                            "xmin": mins[0],
                            "ymin": mins[1],
                            "zmin": mins[2],
                            "xmax": maxs[0],
                            "ymax": maxs[1],
                            "zmax": maxs[2],
                        }

                        info["point_count"] = count

                        if (
                            layer.get("product")
                            == "FAST_GC"
                        ):
                            result["point_count"] = count

                        crs = header.parse_crs()

                        if crs is not None:
                            authority = crs.to_authority()

                            crs_info = {
                                "authority": (
                                    authority[0]
                                    if authority
                                    else None
                                ),
                                "code": (
                                    int(authority[1])
                                    if authority
                                    and str(
                                        authority[1]
                                    ).isdigit()
                                    else (
                                        authority[1]
                                        if authority
                                        else None
                                    )
                                ),
                                "name": crs.name,
                                "wkt": crs.to_wkt(),
                            }

                            if authority:
                                crs_info["id"] = (
                                    f"{authority[0]}:"
                                    f"{authority[1]}"
                                )
                            else:
                                crs_info["id"] = None

                            info["crs"] = crs_info
                            detected_crs.append(
                                crs_info
                            )

                        xmin = (
                            mins[0]
                            if xmin is None
                            else min(xmin, mins[0])
                        )

                        ymin = (
                            mins[1]
                            if ymin is None
                            else min(ymin, mins[1])
                        )

                        zmin = (
                            mins[2]
                            if zmin is None
                            else min(zmin, mins[2])
                        )

                        xmax = (
                            maxs[0]
                            if xmax is None
                            else max(xmax, maxs[0])
                        )

                        ymax = (
                            maxs[1]
                            if ymax is None
                            else max(ymax, maxs[1])
                        )

                        zmax = (
                            maxs[2]
                            if zmax is None
                            else max(zmax, maxs[2])
                        )

                except Exception as exc:
                    info["metadata_error"] = (
                        f"{type(exc).__name__}: {exc}"
                    )

            elif suffix in {".tif", ".tiff"}:
                try:
                    import rasterio

                    with rasterio.open(path) as ds:
                        b = ds.bounds

                        info["bounds"] = {
                            "xmin": float(b.left),
                            "ymin": float(b.bottom),
                            "xmax": float(b.right),
                            "ymax": float(b.top),
                        }

                        if ds.crs is not None:
                            epsg = ds.crs.to_epsg()

                            crs_info = {
                                "authority": (
                                    "EPSG"
                                    if epsg is not None
                                    else None
                                ),
                                "code": epsg,
                                "id": (
                                    f"EPSG:{epsg}"
                                    if epsg is not None
                                    else None
                                ),
                                "name": str(ds.crs),
                                "wkt": ds.crs.to_wkt(),
                            }

                            info["crs"] = crs_info
                            detected_crs.append(
                                crs_info
                            )

                        xmin = (
                            float(b.left)
                            if xmin is None
                            else min(
                                xmin,
                                float(b.left),
                            )
                        )

                        ymin = (
                            float(b.bottom)
                            if ymin is None
                            else min(
                                ymin,
                                float(b.bottom),
                            )
                        )

                        xmax = (
                            float(b.right)
                            if xmax is None
                            else max(
                                xmax,
                                float(b.right),
                            )
                        )

                        ymax = (
                            float(b.top)
                            if ymax is None
                            else max(
                                ymax,
                                float(b.top),
                            )
                        )

                except Exception as exc:
                    info["metadata_error"] = (
                        f"{type(exc).__name__}: {exc}"
                    )

            result["sources"].append(info)

    # --------------------------------------------------------
    # CRS agreement
    # --------------------------------------------------------

    ids = {
        item.get("id")
        for item in detected_crs
        if item.get("id")
    }

    if len(ids) > 1:
        raise ValueError(
            "Viewer products contain conflicting CRS "
            f"definitions: {sorted(ids)}"
        )

    if detected_crs:
        result["crs"] = detected_crs[0]

    # --------------------------------------------------------
    # Combined bounds
    # --------------------------------------------------------

    if (
        xmin is not None
        and ymin is not None
        and xmax is not None
        and ymax is not None
    ):
        result["bounds"] = {
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
            "zmin": zmin,
            "zmax": zmax,
        }

        result["center"] = {
            "x": (xmin + xmax) / 2.0,
            "y": (ymin + ymax) / 2.0,
            "z": (
                (zmin + zmax) / 2.0
                if (
                    zmin is not None
                    and zmax is not None
                )
                else 0.0
            ),
        }

    return result



def _viewer_raster_display_stats(path: Path) -> dict | None:
    """
    Lightweight viewer-only raster statistics.

    The scientific raster is never modified.  A bounded sample is read
    solely to establish useful display/color-map limits.
    """
    try:
        import numpy as np
        import rasterio
        from rasterio.enums import Resampling

        with rasterio.open(path) as ds:
            if ds.count < 1:
                return None

            total = max(1, int(ds.width) * int(ds.height))
            target = 1_000_000

            scale = min(
                1.0,
                (target / total) ** 0.5,
            )

            out_w = max(
                1,
                int(round(ds.width * scale)),
            )
            out_h = max(
                1,
                int(round(ds.height * scale)),
            )

            arr = ds.read(
                1,
                out_shape=(out_h, out_w),
                masked=True,
                resampling=Resampling.nearest,
            )

            values = arr.compressed()

            if values.size == 0:
                return None

            values = values[
                np.isfinite(values)
            ]

            if values.size == 0:
                return None

            vmin = float(np.min(values))
            vmax = float(np.max(values))
            p02 = float(np.percentile(values, 2))
            p98 = float(np.percentile(values, 98))

            if not np.isfinite(p02):
                p02 = vmin

            if not np.isfinite(p98):
                p98 = vmax

            if p98 <= p02:
                p02 = vmin
                p98 = vmax

            return {
                "band": 1,
                "min": vmin,
                "max": vmax,
                "p02": p02,
                "p98": p98,
            }

    except Exception:
        # Viewer metadata must never prevent FAST-GC outputs
        # from being served.
        return None


def build_runtime_recipe(
    root,
    *,
    manifest_name: str = "FASTGC_VIEW_MANIFEST.json",
    products=None,
) -> dict:
    """
    Build browser-safe metadata from a FAST-GC view manifest.

    Scientific files remain unchanged. Absolute filesystem paths are
    replaced only in this runtime representation with HTTP-relative URLs.
    """
    root = Path(root).resolve()

    if not root.is_dir():
        raise NotADirectoryError(root)

    manifest = root / manifest_name

    if not manifest.is_file():
        raise FileNotFoundError(manifest)

    recipe = build_view_recipe(
        manifest,
        products=products,
    )

    spatial = _inspect_spatial_metadata(recipe)

    runtime = {
        "schema": "fastgc-view-runtime",
        "schema_version": 2,
        "dataset": recipe.get("dataset"),
        "sensor": recipe.get("sensor"),
        "crs": spatial.get("crs"),
        "bounds": spatial.get("bounds"),
        "center": spatial.get("center"),
        "point_count": spatial.get("point_count"),
        "spatial_sources": spatial.get("sources", []),
        "layers": [],
    }

    for layer in recipe.get("layers", []):
        out_layer = {
            "id": layer["id"],
            "product": layer["product"],
            "kind": layer["kind"],
            "visible": layer.get("visible", True),
            "sources": [],
        }

        for source in layer.get("sources", []):
            path = Path(source["path"]).resolve()

            if not path.is_file():
                continue

            rel = _safe_relative(path, root)

            source_entry = {
                "kind": source["kind"],
                "url": "/data/" + quote(
                    rel,
                    safe="/",
                ),
                "filename": path.name,
            }

            if source["kind"] == "raster":
                display = _viewer_raster_display_stats(path)

                if display is not None:
                    source_entry["display"] = display

            out_layer["sources"].append(
                source_entry
            )

        #
        # Scientific sources above remain authoritative.
        # A viewer-only COPC may be advertised as an additional
        # visualization source.  It never replaces or modifies the
        # scientific LAS/LAZ product.
        if out_layer["kind"] == "pointcloud":
            for item in out_layer["sources"]:
                item["role"] = "authoritative"
                item["preferred"] = False

            viewer_copc = (
                root
                / ".fastgc_viewer"
                / "pointcloud"
                / f"{out_layer['product']}.copc.laz"
            )

            if viewer_copc.is_file():
                viewer_rel = _safe_relative(
                    viewer_copc.resolve(),
                    root,
                )

                out_layer["sources"].insert(
                    0,
                    {
                        "kind": "pointcloud",
                        "url": "/data/" + quote(
                            viewer_rel,
                            safe="/",
                        ),
                        "filename": viewer_copc.name,
                        "role": "visualization",
                        "preferred": True,
                    },
                )

        if out_layer["sources"]:
            runtime["layers"].append(out_layer)

    return runtime


class FastGCViewerHandler(SimpleHTTPRequestHandler):
    """
    HTTP handler for one FAST-GC output root.

    Routes:
      /                  -> FAST-GC viewer
      /viewer/*          -> packaged viewer assets
      /runtime.json      -> browser-safe runtime metadata
      /data/*             -> completed FAST-GC products
    """

    server_version = "FASTGCViewer/1"

    def end_headers(self):
        # Useful for LAS/LAZ/COPC range requests and local viewer assets.
        self.send_header(
            "Access-Control-Allow-Origin",
            "*",
        )
        self.send_header(
            "Accept-Ranges",
            "bytes",
        )
        super().end_headers()

    def do_GET(self):
        clean_path = self.path.split("?", 1)[0]

        if clean_path == "/":
            self.send_response(302)
            self.send_header(
                "Location",
                "/viewer/index.html",
            )
            self.end_headers()
            return

        if clean_path == "/runtime.json":
            payload = self.server.runtime_recipe
            body = json.dumps(
                payload,
                indent=2,
            ).encode("utf-8")

            self.send_response(200)
            self.send_header(
                "Content-Type",
                "application/json; charset=utf-8",
            )
            self.send_header(
                "Content-Length",
                str(len(body)),
            )
            self.end_headers()
            self.wfile.write(body)
            return

        if clean_path.startswith("/viewer/"):
            original = self.path

            relative = clean_path[len("/viewer/"):]

            if (
                not relative
                or ".." in Path(relative).parts
            ):
                self.send_error(404)
                return

            asset = (
                Path(__file__).resolve().parent
                / "web"
                / relative
            ).resolve()

            web_root = (
                Path(__file__).resolve().parent
                / "web"
            ).resolve()

            try:
                asset.relative_to(web_root)
            except ValueError:
                self.send_error(404)
                return

            if not asset.is_file():
                self.send_error(404)
                return

            content_types = {
                ".html": "text/html; charset=utf-8",
                ".js": "text/javascript; charset=utf-8",
                ".css": "text/css; charset=utf-8",
                ".json": "application/json; charset=utf-8",
            }

            body = asset.read_bytes()

            self.send_response(200)
            self.send_header(
                "Content-Type",
                content_types.get(
                    asset.suffix.lower(),
                    "application/octet-stream",
                ),
            )
            self.send_header(
                "Content-Length",
                str(len(body)),
            )
            self.end_headers()
            self.wfile.write(body)
            return

        if clean_path.startswith("/data/"):
            #
            # COPCSource performs HTTP byte-range requests.  Serve
            # scientific/viewer files directly from the FAST-GC output
            # root with real 206 Partial Content responses.
            from urllib.parse import unquote

            relative = unquote(
                clean_path[len("/data/"):]
            )

            if not relative or ".." in Path(relative).parts:
                self.send_error(404)
                return

            root = Path(self.server.fastgc_root).resolve()
            path = (root / relative).resolve()

            try:
                path.relative_to(root)
            except ValueError:
                self.send_error(404)
                return

            if not path.is_file():
                self.send_error(404)
                return

            size = path.stat().st_size
            range_header = self.headers.get("Range")

            content_type = (
                mimetypes.guess_type(str(path))[0]
                or "application/octet-stream"
            )

            if not range_header:
                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    content_type,
                )
                self.send_header(
                    "Content-Length",
                    str(size),
                )
                self.end_headers()

                with path.open("rb") as src:
                    shutil.copyfileobj(
                        src,
                        self.wfile,
                        length=1024 * 1024,
                    )
                return

            match = re.fullmatch(
                r"bytes=(\d*)-(\d*)",
                range_header.strip(),
            )

            if match is None:
                self.send_response(416)
                self.send_header(
                    "Content-Range",
                    f"bytes */{size}",
                )
                self.end_headers()
                return

            start_text, end_text = match.groups()

            if start_text:
                start = int(start_text)
                end = (
                    int(end_text)
                    if end_text
                    else size - 1
                )
            else:
                suffix_length = int(end_text or "0")

                if suffix_length <= 0:
                    self.send_response(416)
                    self.send_header(
                        "Content-Range",
                        f"bytes */{size}",
                    )
                    self.end_headers()
                    return

                start = max(0, size - suffix_length)
                end = size - 1

            if (
                start < 0
                or start >= size
                or end < start
            ):
                self.send_response(416)
                self.send_header(
                    "Content-Range",
                    f"bytes */{size}",
                )
                self.end_headers()
                return

            end = min(end, size - 1)
            length = end - start + 1

            self.send_response(206)
            self.send_header(
                "Content-Type",
                content_type,
            )
            self.send_header(
                "Content-Range",
                f"bytes {start}-{end}/{size}",
            )
            self.send_header(
                "Content-Length",
                str(length),
            )
            self.end_headers()

            with path.open("rb") as src:
                src.seek(start)
                remaining = length

                while remaining:
                    block = src.read(
                        min(1024 * 1024, remaining)
                    )
                    if not block:
                        break

                    self.wfile.write(block)
                    remaining -= len(block)

            return

        self.send_error(404)


def create_server(
    root,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    products=None,
):
    """
    Create a local FAST-GC visualization HTTP server.

    Creating the server does not start processing or modify outputs.
    """
    root = Path(root).resolve()

    runtime = build_runtime_recipe(
        root,
        products=products,
    )

    handler = functools.partial(
        FastGCViewerHandler,
        directory=str(root),
    )

    httpd = ThreadingHTTPServer(
        (host, int(port)),
        handler,
    )

    httpd.runtime_recipe = runtime
    httpd.fastgc_root = root

    return httpd


def serve(
    root,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    products=None,
):
    """
    Serve a FAST-GC output root until interrupted.
    """
    httpd = create_server(
        root,
        host=host,
        port=port,
        products=products,
    )

    address, actual_port = httpd.server_address[:2]

    print("FAST-GC Viewer Server")
    print(f"Root    : {httpd.fastgc_root}")
    print(f"Runtime : http://{address}:{actual_port}/runtime.json")
    print(f"Data    : http://{address}:{actual_port}/data/")
    print()
    print("Press Ctrl+C to stop.")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Serve completed FAST-GC products for local visualization."
        )
    )

    parser.add_argument(
        "root",
        help=(
            "FAST-GC output root containing "
            "FASTGC_VIEW_MANIFEST.json."
        ),
    )

    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"Bind host. Default: {DEFAULT_HOST}",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Bind port. Default: {DEFAULT_PORT}",
    )

    parser.add_argument(
        "--product",
        action="append",
        dest="products",
        help=(
            "Limit viewer to a product. Repeat for multiple products."
        ),
    )

    args = parser.parse_args(argv)

    serve(
        args.root,
        host=args.host,
        port=args.port,
        products=args.products,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
