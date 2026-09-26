"""Read-only, reusable FAST-GIS survey inventory; no FAST-GC pipeline changes."""
from __future__ import annotations

import json
import re
from pathlib import Path

import laspy
from pyproj import CRS


def _crs_hint_from_path(path: Path):
    """Return an explicit EPSG CRS encoded in the file/path, if present.

    Only explicit EPSG identifiers are promoted to authoritative CRS values.
    Ambiguous hints such as UTM56S are intentionally not converted here
    because the datum is not specified.
    """
    for part in reversed(path.parts):
        match = re.search(
            r"(?i)(?:^|[^A-Z0-9])EPSG[ _-]?(\d{4,6})(?:$|[^0-9])",
            part,
        )
        if match:
            code = int(match.group(1))
            try:
                return CRS.from_epsg(code)
            except Exception as exc:
                raise ValueError(
                    f"Invalid EPSG code in path metadata: EPSG:{code} ({path})"
                ) from exc
    return None


def _utm_hint_from_path(path: Path):
    """Return a non-authoritative UTM zone/hemisphere hint from a path.

    A bare UTM zone and hemisphere do not identify the datum, so this helper
    never returns or assigns a CRS.
    """
    patterns = (
        r"(?i)(?:^|[^A-Z0-9])UTM[ _-]?(?:ZONE[ _-]?)?([1-9]|[1-5][0-9]|60)[ _-]?([NS])(?:$|[^A-Z0-9])",
        r"(?i)(?:^|[^A-Z0-9])ZONE[ _-]?([1-9]|[1-5][0-9]|60)[ _-]?([NS])(?:$|[^A-Z0-9])",
    )

    for part in reversed(path.parts):
        for pattern in patterns:
            match = re.search(pattern, part)
            if match:
                return {
                    "type": "utm",
                    "zone": int(match.group(1)),
                    "hemisphere": match.group(2).upper(),
                    "datum": None,
                    "source": "path_hint",
                }

    return None


def _crs(header, path: Path, override=None):
    embedded = header.parse_crs()
    sidecar = path.with_suffix('.prj')
    external = CRS.from_wkt(sidecar.read_text(encoding='utf-8-sig')) if sidecar.is_file() else None
    supplied = CRS.from_user_input(override) if override is not None else None
    path_hint = _crs_hint_from_path(path)

    candidates = [
        ('las_header', embedded),
        ('sidecar_prj', external),
        ('user', supplied),
        ('path_epsg', path_hint),
    ]
    candidates = [
        (name, CRS.from_user_input(value))
        for name, value in candidates
        if value is not None
    ]

    if candidates and any(
        not candidates[0][1].equals(value)
        for _, value in candidates[1:]
    ):
        details = ", ".join(
            f"{name}={value.to_string()}"
            for name, value in candidates
        )
        raise ValueError(
            f"Conflicting CRS metadata: {path} ({details})"
        )

    return (
        (candidates[0][1].to_wkt(), candidates[0][0])
        if candidates
        else (None, 'unresolved')
    )


def build_survey_catalog(input_path, *, source_crs=None, recursive=True):
    """Inventory one LAS/LAZ or a directory. Never modifies original files.

    Unresolved CRS or unreadable sources remain in the inventory with status.
    Spatial region discovery must exclude them until reconciled.
    """
    source = Path(input_path).resolve()
    if source.is_file():
        if source.suffix.lower() not in {'.las', '.laz'}:
            raise ValueError('Expected LAS/LAZ input')
        paths = [source]
    elif source.is_dir():
        paths = [p for p in (source.rglob('*') if recursive else source.iterdir())
                 if p.is_file() and p.suffix.lower() in {'.las', '.laz'}]
        paths.sort(key=lambda p: str(p).casefold())
    else:
        raise FileNotFoundError(source)
    if not paths:
        raise ValueError('No LAS/LAZ files found')
    entries = []
    for path in paths:
        stat = path.stat()
        entry = {'tile_id': f'tile_{len(entries)+1:06d}', 'path': str(path),
                 'size_bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
        try:
            with laspy.open(path) as reader:
                h = reader.header
                wkt, origin = _crs(h, path, source_crs)
                crs_hint = _utm_hint_from_path(path)
                entry.update(las_version=str(h.version), point_format=int(h.point_format.id),
                             dimensions=list(h.point_format.dimension_names),
                             scales=[float(x) for x in h.scales],
                             offsets=[float(x) for x in h.offsets],
                             point_count=int(h.point_count),
                             bounds=[float(h.mins[0]), float(h.mins[1]),
                                     float(h.maxs[0]), float(h.maxs[1])],
                             crs_wkt=wkt, crs_source=origin,
                             crs_hint=crs_hint,
                             status='ready' if wkt else 'unresolved_crs')
        except Exception as exc:
            entry.update(status='error', error=f'{type(exc).__name__}: {exc}')
        entries.append(entry)
    return {'schema_version': 1, 'input_path': str(source), 'tiles': entries,
            'tile_count': len(entries), 'ready_count': sum(e['status']=='ready' for e in entries)}


def save_survey_catalog(catalog, output_path):
    path = Path(output_path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(catalog, indent=2, allow_nan=False), encoding='utf-8')
    return path
