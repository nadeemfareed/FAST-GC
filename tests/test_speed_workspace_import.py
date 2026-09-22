def test_io_las_imports_workspace_builder():
    import fastgc.io_las as io_las
    assert callable(io_las.build_surface_consensus_workspace)
