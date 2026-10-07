import numpy as np
import pytest

from fastgc.terrain import _prepare_terrain_raster_output


@pytest.mark.parametrize(
    "product,dtype,nodata",
    [
        ("d8_flow_direction", np.dtype("uint8"), 255),
        ("stream_mask", np.dtype("uint8"), 255),
        ("watershed_boundary", np.dtype("uint8"), 255),
        ("strahler_stream_order", np.dtype("uint16"), 65535),
        ("stream_link_id", np.dtype("int32"), -1),
        ("basin_id", np.dtype("int32"), -1),
        ("subcatchment_id", np.dtype("int32"), -1),
    ],
)
def test_categorical_hydrology_storage(product, dtype, nodata):
    arr = np.array(
        [[0.0, 1.0], [2.0, 3.0]],
        dtype=np.float64,
    )
    valid = np.array(
        [[True, True], [False, True]],
        dtype=bool,
    )

    out, out_nodata = _prepare_terrain_raster_output(
        product,
        arr,
        valid,
        np.nan,
    )

    assert out.dtype == dtype
    assert out_nodata == nodata
    assert out[1, 0] == nodata
    assert out[0, 0] == 0
    assert out[0, 1] == 1


def test_continuous_hydrology_remains_float32():
    arr = np.array(
        [[1.25, 2.5], [3.75, 4.0]],
        dtype=np.float64,
    )
    valid = np.array(
        [[True, True], [False, True]],
        dtype=bool,
    )

    out, nodata = _prepare_terrain_raster_output(
        "d8_contributing_area",
        arr,
        valid,
        np.nan,
    )

    assert out.dtype == np.dtype("float32")
    assert np.isnan(out[1, 0])
    assert np.isnan(nodata)


def test_categorical_rejects_fractional_values():
    arr = np.array([[0.0, 1.5]], dtype=np.float64)
    valid = np.ones(arr.shape, dtype=bool)

    with pytest.raises(ValueError, match="non-integer"):
        _prepare_terrain_raster_output(
            "stream_link_id",
            arr,
            valid,
            np.nan,
        )


def test_categorical_rejects_reserved_nodata_collision():
    arr = np.array([[255.0]], dtype=np.float64)
    valid = np.ones(arr.shape, dtype=bool)

    with pytest.raises(OverflowError, match="reserved nodata"):
        _prepare_terrain_raster_output(
            "d8_flow_direction",
            arr,
            valid,
            np.nan,
        )


def test_write_raster_preserves_uint8_storage(tmp_path):
    import rasterio
    from rasterio.transform import from_origin

    from fastgc.terrain import _write_raster

    out_fp = tmp_path / "d8.tif"

    arr = np.array(
        [[1, 2], [255, 4]],
        dtype=np.uint8,
    )

    profile = {
        "driver": "GTiff",
        "height": 2,
        "width": 2,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:32617",
        "transform": from_origin(0, 2, 1, 1),
    }

    _write_raster(
        arr,
        profile,
        str(out_fp),
        nodata=255,
    )

    with rasterio.open(out_fp) as src:
        assert src.dtypes[0] == "uint8"
        assert src.nodata == 255
        got = src.read(1)

    assert got.dtype == np.uint8
    assert got[1, 0] == 255


def test_write_raster_preserves_int32_storage(tmp_path):
    import rasterio
    from rasterio.transform import from_origin

    from fastgc.terrain import _write_raster

    out_fp = tmp_path / "basin.tif"

    arr = np.array(
        [[1, 2], [-1, 3]],
        dtype=np.int32,
    )

    profile = {
        "driver": "GTiff",
        "height": 2,
        "width": 2,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:32617",
        "transform": from_origin(0, 2, 1, 1),
    }

    _write_raster(
        arr,
        profile,
        str(out_fp),
        nodata=-1,
    )

    with rasterio.open(out_fp) as src:
        assert src.dtypes[0] == "int32"
        assert src.nodata == -1
        got = src.read(1)

    assert got.dtype == np.int32
    assert got[1, 0] == -1
