from fastgc.gis import progress as gp


def test_gis_progress_reuses_fastgc_backend(monkeypatch):
    seen = {}
    sentinel = object()
    def fake_progress_bar(*args, **kwargs):
        seen.update(kwargs)
        return sentinel
    monkeypatch.setattr(gp, "progress_bar", fake_progress_bar)
    out = gp.gis_progress("clipping", 30, unit="plot")
    assert out is sentinel
    assert seen["desc"] == "FAST-GIS CLIPPING"
    assert seen["total"] == 30
    assert seen["unit"] == "plot"
    assert seen["disable"] is False


def test_gis_progress_does_not_duplicate_prefix(monkeypatch):
    seen = {}
    monkeypatch.setattr(gp, "progress_bar", lambda *a, **k: seen.update(k) or object())
    gp.gis_progress("FAST-GIS SURVEY", 4, unit="file", enabled=False)
    assert seen["desc"] == "FAST-GIS SURVEY"
    assert seen["disable"] is True
