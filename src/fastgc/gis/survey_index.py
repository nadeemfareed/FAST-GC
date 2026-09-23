"""Read-only survey-tile spatial index for FAST-GIS."""
from __future__ import annotations

from pathlib import Path
from pyproj import CRS
from shapely.geometry import box

from .spatial_index import GeometrySpatialIndex


class SurveyTileIndex:
    """Query ready survey-catalog tiles by 2-D geometry."""

    def __init__(self, catalog):
        ready = [t for t in catalog.get("tiles", []) if t.get("status") == "ready"]
        self.tiles = tuple(ready)
        if ready:
            crs = CRS.from_wkt(ready[0]["crs_wkt"])
            if any(not crs.equals(CRS.from_wkt(t["crs_wkt"])) for t in ready[1:]):
                raise ValueError("Ready survey tiles must share one CRS before indexing")
            self.crs = crs
        else:
            self.crs = None
        self._index = GeometrySpatialIndex([box(*t["bounds"]) for t in ready])

    def query(self, geometry):
        indices = self._index.query_intersects(geometry)
        return [self.tiles[int(i)] for i in indices]

    def paths(self, geometry):
        return [Path(t["path"]) for t in self.query(geometry)]
