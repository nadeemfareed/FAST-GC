from fastgc.core import _resolve_products
from fastgc.io_las import (
    PRODUCT_DEM,
    PRODUCT_GC,
    PRODUCT_NORMALIZED,
    PRODUCT_TERRAIN,
)


def test_fast_dem_does_not_require_normalized():
    resolved = _resolve_products([PRODUCT_DEM])

    assert PRODUCT_DEM in resolved
    assert PRODUCT_NORMALIZED not in resolved


def test_terrain_requires_dem_but_not_normalized():
    resolved = _resolve_products([PRODUCT_TERRAIN])

    assert PRODUCT_DEM in resolved
    assert PRODUCT_TERRAIN in resolved
    assert PRODUCT_NORMALIZED not in resolved


def test_normalized_requires_dem():
    resolved = _resolve_products([PRODUCT_NORMALIZED])

    assert PRODUCT_DEM in resolved
    assert PRODUCT_NORMALIZED in resolved


def test_dem_and_gc_do_not_add_normalized():
    resolved = _resolve_products(
        [PRODUCT_GC, PRODUCT_DEM]
    )

    assert PRODUCT_GC in resolved
    assert PRODUCT_DEM in resolved
    assert PRODUCT_NORMALIZED not in resolved
