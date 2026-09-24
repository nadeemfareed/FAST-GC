from pathlib import Path
from fastgc.gis import publish


def test_publish_falls_back_after_permission_error(tmp_path, monkeypatch):
    stage=tmp_path/"stage"; stage.mkdir(); (stage/"x.txt").write_text("ok")
    target=tmp_path/"final"
    real=publish.os.replace; calls={"n":0}
    def flaky(a,b):
        calls["n"] += 1
        if Path(a) == stage: raise PermissionError(5, "denied")
        return real(a,b)
    monkeypatch.setattr(publish.os,"replace",flaky)
    publish.publish_directory(stage,target,retries=2,delay=0)
    assert (target/"x.txt").read_text()=="ok"
