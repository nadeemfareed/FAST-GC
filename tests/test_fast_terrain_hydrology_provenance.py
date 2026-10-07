from pathlib import Path

from fastgc.terrain import (
    FAST_TERRAIN_SCHEMA,
    HYDROLOGY_TERRAIN_PRODUCTS,
    _terrain_provenance,
)


def _item(product):
    return {
        "product": product,
        "dem_fp": "merged_FAST_DEM.tif",
        "analytical_domain": "continuous_merged_dem",
        "hillshade_azimuth": 315.0,
        "hillshade_altitude": 45.0,
        "hillshade_z_factor": 1.0,
        "tpi_radius": 3,
        "multiscale_tpi_radius_m": None,
        "openness_radius_m": None,
        "twi_eps": 1.0e-6,
        "dtw_max_distance": None,
        "stream_threshold_area_m2": 1234.0,
        "mfd_exponent": 1.1,
        "twi_min_slope_radians": 1.0e-6,
        "ls_m": 0.4,
        "ls_n": 1.3,
    }


def test_hydrology_provenance_names_current_flat_resolver():
    tags = _terrain_provenance(
        _item("d8_flow_direction")
    )

    assert tags[
        "FASTGC_HYDRO_D8_FLAT_RESOLUTION"
    ] == "barnes_style_flat_mask_open_boundary"


def test_hydrology_provenance_records_scientific_parameters():
    tags = _terrain_provenance(
        _item("topographic_wetness_index")
    )

    assert tags["FASTGC_HYDRO_CONDITIONING"] == "priority_flood"
    assert tags["FASTGC_HYDRO_CONNECTIVITY"] == "8"

    assert float(
        tags["FASTGC_HYDRO_STREAM_THRESHOLD_M2"]
    ) == 1234.0

    assert float(
        tags["FASTGC_HYDRO_MFD_EXPONENT"]
    ) == 1.1

    assert float(
        tags["FASTGC_HYDRO_TWI_MIN_SLOPE_RAD"]
    ) == 1.0e-6

    assert float(tags["FASTGC_HYDRO_LS_M"]) == 0.4
    assert float(tags["FASTGC_HYDRO_LS_N"]) == 1.3


def test_hydrology_provenance_records_continuous_domain():
    tags = _terrain_provenance(
        _item("d8_flow_accumulation")
    )

    assert (
        tags["FASTGC_ANALYTICAL_DOMAIN"]
        == "continuous_merged_dem"
    )

    assert tags["FASTGC_SOURCE_DEM"] == "merged_FAST_DEM.tif"
    assert tags["FASTGC_TERRAIN_SCHEMA"] == FAST_TERRAIN_SCHEMA


def test_all_hydrology_products_receive_hydrology_provenance():
    for product in HYDROLOGY_TERRAIN_PRODUCTS:
        tags = _terrain_provenance(_item(product))

        assert "FASTGC_HYDRO_CONDITIONING" in tags
        assert "FASTGC_HYDRO_CONNECTIVITY" in tags
        assert "FASTGC_HYDRO_D8_FLAT_RESOLUTION" in tags


def test_stale_deterministic_bfs_provenance_absent():
    source = Path("src/fastgc/terrain.py").read_text(
        encoding="utf-8"
    )

    assert "deterministic_bfs" not in source
