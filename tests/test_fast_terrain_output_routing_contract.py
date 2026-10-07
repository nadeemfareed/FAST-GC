import ast
from pathlib import Path


CORE = Path("src/fastgc/core.py")


def _tree():
    return ast.parse(CORE.read_text(encoding="utf-8"))


def _calls(node):
    out = []
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            if isinstance(n.func, ast.Name):
                out.append((n.func.id, n))
            elif isinstance(n.func, ast.Attribute):
                out.append((n.func.attr, n))
    return out


def _find_if_with_mode(mode):
    tree = _tree()

    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue

        text = ast.unparse(node.test)

        if (
            "terrain_output" in text
            and mode in text
        ):
            return node

    raise AssertionError(
        f"No terrain_output routing branch found for {mode}"
    )


def test_raster_branch_calls_raster_runner():
    node = _find_if_with_mode("raster")
    names = [name for name, _ in _calls(node)]

    assert "run_terrain_from_dem" in names


def test_vector_branch_calls_vector_runner():
    node = _find_if_with_mode("vector")
    names = [name for name, _ in _calls(node)]

    assert "run_hydrology_vector_from_dem" in names


def test_raster_runner_receives_stream_threshold():
    node = _find_if_with_mode("raster")

    calls = [
        call
        for name, call in _calls(node)
        if name == "run_terrain_from_dem"
    ]

    assert calls

    kw = {
        item.arg: ast.unparse(item.value)
        for item in calls[0].keywords
    }

    assert (
        kw["stream_threshold_area_m2"]
        == "stream_threshold_area_m2"
    )


def test_vector_runner_receives_same_stream_threshold():
    node = _find_if_with_mode("vector")

    calls = [
        call
        for name, call in _calls(node)
        if name == "run_hydrology_vector_from_dem"
    ]

    assert calls

    kw = {
        item.arg: ast.unparse(item.value)
        for item in calls[0].keywords
    }

    assert (
        kw["stream_threshold_area_m2"]
        == "stream_threshold_area_m2"
    )


def test_stream_min_order_only_sent_to_vector_runner():
    tree = _tree()

    raster_calls = [
        call
        for name, call in _calls(tree)
        if name == "run_terrain_from_dem"
    ]

    vector_calls = [
        call
        for name, call in _calls(tree)
        if name == "run_hydrology_vector_from_dem"
    ]

    assert raster_calls
    assert vector_calls

    raster_kw = {
        item.arg
        for call in raster_calls
        for item in call.keywords
    }

    vector_kw = {
        item.arg
        for call in vector_calls
        for item in call.keywords
    }

    assert "stream_min_order" not in raster_kw
    assert "stream_min_order" in vector_kw


def test_public_output_modes_are_exactly_three():
    source = Path(
        "src/fastgc/terrain.py"
    ).read_text(encoding="utf-8")

    tree = ast.parse(source)

    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name)
                and t.id == "TERRAIN_OUTPUT_CHOICES"
                for t in node.targets
            )
        ):
            values = ast.literal_eval(node.value)
            assert values == (
                "raster",
                "vector",
                "both",
            )
            return

    raise AssertionError(
        "TERRAIN_OUTPUT_CHOICES not found"
    )
