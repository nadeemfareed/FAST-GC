from pathlib import Path
import pytest

from fastgc.gis import gis_cli


def test_random_does_not_require_dsm(monkeypatch, tmp_path):
    source = tmp_path / "survey"
    source.mkdir()
    seen = {}
    monkeypatch.setattr(gis_cli, "sample_random_and_clip", lambda *a, **k: seen.update(kwargs=k) or tmp_path / "out")
    gis_cli.main(["--in_path", str(source), "--out_dir", str(tmp_path / "out"),
                  "--sensor_mode", "ALS", "--sampling", "random", "--plots", "6"])
    assert seen["kwargs"]["count"] == 6
    assert seen["kwargs"]["sensor_mode"] == "ALS"


def test_rough_requires_dsm(tmp_path):
    source = tmp_path / "x.laz"
    source.write_bytes(b"")
    with pytest.raises(SystemExit):
        gis_cli.main(["--in_path", str(source), "--out_dir", str(tmp_path / "out"),
                      "--sensor_mode", "ALS", "--sampling", "rough_height_stratified"])


def test_imported_requires_plot_file(tmp_path):
    source = tmp_path / "survey"
    source.mkdir()
    with pytest.raises(SystemExit):
        gis_cli.main(["--in_path", str(source), "--out_dir", str(tmp_path / "out"),
                      "--sensor_mode", "ALS", "--sampling", "imported"])


def test_random_defaults_hexagon_15m_core_5m_buffer(monkeypatch, tmp_path):
    source = tmp_path / "survey"; source.mkdir(); seen = {}
    monkeypatch.setattr(gis_cli, "sample_random_and_clip", lambda *a, **k: seen.update(k) or tmp_path/"out")
    gis_cli.main(["--in_path", str(source), "--out_dir", str(tmp_path/"out"),
                  "--sensor_mode", "ALS", "--sampling", "random"])
    assert seen["shape"] == "hexagon"
    assert seen["radius"] == 15.0
    assert seen["buffer"] == 5.0
