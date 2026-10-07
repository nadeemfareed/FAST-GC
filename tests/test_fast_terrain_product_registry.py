from fastgc.terrain import (
    FAST_TERRAIN_ALL_PRODUCTS,
    FAST_TERRAIN_CONTINUOUS_HYDROLOGY_PRODUCTS,
    FAST_TERRAIN_LEGACY_PRODUCTS,
    HYDROLOGY_TERRAIN_PRODUCTS,
    _resolve_terrain_products,
)


def test_all_is_local_core_and_excludes_hydrology():
    resolved = _resolve_terrain_products(["all"])

    assert resolved == list(FAST_TERRAIN_ALL_PRODUCTS)

    assert not (
        set(resolved)
        & set(FAST_TERRAIN_CONTINUOUS_HYDROLOGY_PRODUCTS)
    )


def test_all_excludes_legacy_compatibility_products():
    resolved = set(_resolve_terrain_products(["all"]))

    assert resolved.isdisjoint(
        FAST_TERRAIN_LEGACY_PRODUCTS
    )


def test_legacy_products_remain_explicitly_selectable():
    for product in sorted(FAST_TERRAIN_LEGACY_PRODUCTS):
        assert _resolve_terrain_products(
            [product]
        ) == [product]


def test_continuous_hydrology_remains_explicitly_selectable():
    for product in sorted(
        FAST_TERRAIN_CONTINUOUS_HYDROLOGY_PRODUCTS
    ):
        assert _resolve_terrain_products(
            [product]
        ) == [product]


def test_hydrology_registries_are_identical():
    assert HYDROLOGY_TERRAIN_PRODUCTS == set(
        FAST_TERRAIN_CONTINUOUS_HYDROLOGY_PRODUCTS
    )


def test_duplicate_explicit_products_are_removed():
    assert _resolve_terrain_products(
        [
            "slope_percent",
            "slope_percent",
            "topographic_wetness_index",
            "topographic_wetness_index",
        ]
    ) == [
        "slope_percent",
        "topographic_wetness_index",
    ]
