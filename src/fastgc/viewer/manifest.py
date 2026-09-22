"""Product registry for optional FAST-GC visualization."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Optional


@dataclass
class ProductRecord:
    product: str
    kind: str
    merged: Optional[str] = None
    tiles: list[str] = field(default_factory=list)
    status: str = "complete"


@dataclass
class ViewManifest:
    dataset: str
    sensor: str
    products: dict[str, ProductRecord] = field(default_factory=dict)

    def register_product(
        self,
        product: str,
        *,
        kind: str,
        merged: Optional[str] = None,
        tiles: Optional[list[str]] = None,
        status: str = "complete",
    ) -> None:

        self.products[product] = ProductRecord(
            product=product,
            kind=kind,
            merged=merged,
            tiles=list(tiles or []),
            status=status,
        )

    def as_dict(self) -> dict:
        return {
            "schema": "fastgc-view-manifest",
            "schema_version": 1,
            "dataset": self.dataset,
            "sensor": self.sensor,
            "products": {
                key: asdict(value)
                for key, value in self.products.items()
            },
        }

    def register_completed_output(
        self,
        *,
        product: str,
        output,
        source: str = "tile",
        kind: str | None = None,
    ) -> None:
        """
        Register an output path already produced successfully by FAST-GC.

        This method performs no scientific computation and never creates,
        modifies, or validates the product itself.
        """

        product = str(product).upper()
        path = str(Path(output).resolve())

        if kind is None:
            suffix = Path(path).suffix.lower()

            if suffix in {".las", ".laz", ".copc"}:
                kind = "pointcloud"
            elif suffix in {".tif", ".tiff"}:
                kind = "raster"
            else:
                kind = "other"

        record = self.products.get(product)

        if record is None:
            record = ProductRecord(
                product=product,
                kind=kind,
                merged=None,
                tiles=[],
                status="complete",
            )
            self.products[product] = record

        if source == "merged":
            record.merged = path

        elif source == "tile":
            if path not in record.tiles:
                record.tiles.append(path)

        else:
            raise ValueError(
                "source must be 'tile' or 'merged'"
            )

        record.status = "complete"


    def register_result(
        self,
        result,
        *,
        source: str = "tile",
    ) -> bool:
        """
        Register an existing FAST-GC result dictionary.

        Expected existing contract:
            {
                "status": "ok",
                "product": "...",
                "output": "..."
            }

        Returns True only when a completed output was registered.
        """

        if not isinstance(result, dict):
            return False

        if str(result.get("status", "")).lower() != "ok":
            return False

        product = result.get("product")
        output = result.get("output")

        if not product or not output:
            return False

        self.register_completed_output(
            product=str(product),
            output=output,
            source=source,
        )

        return True


    def register_merged_outputs(
        self,
        outputs: dict,
    ) -> int:
        """
        Register the exact output dictionary returned by FAST-GC merging.

        Returns the number of merged products registered.
        """

        count = 0

        for product, output in outputs.items():
            if not output:
                continue

            # CHM merge keys may carry the method:
            # FAST_CHM_p2r, FAST_CHM_pitfree, etc.
            canonical_product = str(product)

            if canonical_product.upper().startswith("FAST_CHM_"):
                canonical_product = "FAST_CHM"

            self.register_completed_output(
                product=canonical_product,
                output=output,
                source="merged",
            )

            count += 1

        return count


    def write(self, path) -> Path:
        """
        Write visualization metadata atomically.

        This file is non-authoritative and is never consumed by
        FAST-GC scientific classification.
        """

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        tmp = path.with_name(path.name + ".tmp")

        tmp.write_text(
            json.dumps(self.as_dict(), indent=2),
            encoding="utf-8",
        )

        tmp.replace(path)

        return path
