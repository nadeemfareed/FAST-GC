import ast
from pathlib import Path


def test_core_internal_keyword_contracts():
    text = Path("src/fastgc/core.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(text)

    functions = {}

    for node in tree.body:
        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            args = (
                list(node.args.posonlyargs)
                + list(node.args.args)
                + list(node.args.kwonlyargs)
            )

            functions[node.name] = {
                "args": {a.arg for a in args},
                "varkw": node.args.kwarg is not None,
            }

    problems = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        if not isinstance(node.func, ast.Name):
            continue

        name = node.func.id

        if name not in functions:
            continue

        if functions[name]["varkw"]:
            continue

        supplied = {
            kw.arg
            for kw in node.keywords
            if kw.arg is not None
        }

        bad = sorted(
            supplied - functions[name]["args"]
        )

        if bad:
            problems.append(
                (node.lineno, name, bad)
            )

    assert problems == []


def test_processing_helper_accepts_terrain_scale_parameters():
    text = Path("src/fastgc/core.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(text)

    func = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name == "_run_processing_with_optional_fpfix"
    )

    names = {
        a.arg
        for a in (
            list(func.args.posonlyargs)
            + list(func.args.args)
            + list(func.args.kwonlyargs)
        )
    }

    assert "multiscale_tpi_radii_m" in names
    assert "openness_radii_m" in names
