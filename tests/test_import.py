import fastgc
import fastgc.cli
import fastgc.core
import fastgc.io_las
import fastgc.chm
import fastgc.terrain
import fastgc.change
import fastgc.itd
import fastgc.merge
import fastgc.monster


def test_import():
    assert hasattr(fastgc, "__version__")


def test_core_module_imports():
    assert hasattr(fastgc.cli, "main")
    assert hasattr(fastgc.core, "run_fastgc")
    assert hasattr(fastgc.io_las, "process_fastgc_path")
    assert hasattr(fastgc.chm, "build_chm_from_normalized_root")
    assert hasattr(fastgc.terrain, "run_terrain_from_processed_root")
    assert hasattr(fastgc.change, "run_change_from_processed_root")
    assert hasattr(fastgc.itd, "run_itd_from_processed_root")
    assert hasattr(fastgc.merge, "merge_processed_tiles")
    assert hasattr(fastgc.monster, "run_stage")