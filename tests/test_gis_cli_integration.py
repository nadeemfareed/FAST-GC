import pytest
from fastgc import cli

def test_gis_dispatch(monkeypatch):
    seen={}
    import fastgc.gis.gis_cli as g
    monkeypatch.setattr(g, "main", lambda argv: seen.setdefault("argv", argv))
    cli.main(["gis","--x"])
    assert seen["argv"] == ["--x"]

def test_normal_cli_not_dispatched(monkeypatch):
    import fastgc.gis.gis_cli as g
    monkeypatch.setattr(g, "main", lambda argv: pytest.fail("GIS dispatch used"))
    with pytest.raises(SystemExit):
        cli.main(["--help"])
