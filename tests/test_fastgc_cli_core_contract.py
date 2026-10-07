import ast
from pathlib import Path


def test_cli_run_fastgc_keyword_contract():
    cli_tree = ast.parse(
        Path("src/fastgc/cli.py").read_text(
            encoding="utf-8"
        )
    )

    core_tree = ast.parse(
        Path("src/fastgc/core.py").read_text(
            encoding="utf-8"
        )
    )

    call = next(
        n for n in ast.walk(cli_tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "run_fastgc"
    )

    cli_kwargs = {
        kw.arg
        for kw in call.keywords
        if kw.arg is not None
    }

    func = next(
        n for n in core_tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name == "run_fastgc"
    )

    core_args = {
        a.arg
        for a in (
            list(func.args.posonlyargs)
            + list(func.args.args)
            + list(func.args.kwonlyargs)
        )
    }

    assert cli_kwargs <= core_args


def test_openness_radii_public_argument_exists():
    tree = ast.parse(
        Path("src/fastgc/core.py").read_text(
            encoding="utf-8"
        )
    )

    func = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name == "run_fastgc"
    )

    names = {
        a.arg
        for a in (
            list(func.args.posonlyargs)
            + list(func.args.args)
            + list(func.args.kwonlyargs)
        )
    }

    assert "openness_radii_m" in names
