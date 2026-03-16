
from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

from .monster import DEFAULT_BACKEND, log_info, run_stage

PRODUCT_ITD = "FAST_ITD"
ALGORITHM_PACKAGE = "fastgc.itd_algorithms"

SUPPORTED_ITD_METHODS = {
    "placeholder",
    "lmf",
    "watershed",
    "dalponte2016",
    "li2012",
    "csp",
}


def _existing_output(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_file():
        return True
    return any(path.iterdir())


def _resolve_chm_root(processed_root: Path) -> Path:
    chm_root = processed_root / "FAST_CHM"
    if not chm_root.exists():
        raise FileNotFoundError(f"FAST_CHM folder not found: {chm_root}")
    return chm_root


def _resolve_source_chm_variant(chm_root: Path, source_chm: str | None) -> Path:
    if source_chm is not None:
        source_dir = chm_root / source_chm
        if not source_dir.exists():
            raise FileNotFoundError(f"Requested CHM variant not found: {source_dir}")
        if not source_dir.is_dir():
            raise NotADirectoryError(f"Requested CHM variant is not a folder: {source_dir}")
        return source_dir

    variants = sorted([p for p in chm_root.iterdir() if p.is_dir()])
    if not variants:
        raise FileNotFoundError(f"No CHM method folders found in: {chm_root}")

    return variants[0]


def _list_chm_rasters(chm_variant_root: Path) -> list[Path]:
    rasters = sorted([p for p in chm_variant_root.glob("*.tif") if p.is_file()])
    if not rasters:
        raise FileNotFoundError(f"No CHM rasters found in: {chm_variant_root}")
    return rasters


def _sanitize_name(name: str) -> str:
    safe = "".join(c if c.isalnum() or c in {"_", "-", "."} else "_" for c in name)
    return safe.strip("._") or "unnamed"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _write_placeholder_record(
    out_root: Path,
    *,
    method: str,
    processed_root: Path,
    chm_variant_name: str,
    chm_raster: Path,
) -> str:
    stem = _sanitize_name(chm_raster.stem)
    out_fp = out_root / f"{stem}_itd_placeholder.json"
    payload = {
        "status": "placeholder",
        "module": PRODUCT_ITD,
        "method": method,
        "source_root": str(processed_root),
        "source_chm_variant": chm_variant_name,
        "source_raster": str(chm_raster),
        "message": (
            "FAST_ITD placeholder created. "
            "This raster has been routed through the FAST_ITD pipeline, "
            "but the selected algorithm is not implemented yet."
        ),
    }
    _write_json(out_fp, payload)
    return str(out_fp)


def _load_algorithm_module(method: str):
    """
    Future-ready import hook.

    Expected future pattern in fastgc/itd_algorithms/<method>.py:
        def run_itd_on_chm(chm_raster: str | Path, out_root: str | Path, **kwargs) -> dict | str:
            ...
    """
    if method == "placeholder":
        return None

    module_name = f"{ALGORITHM_PACKAGE}.{method}"
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        return None


def _algorithm_runner(
    *,
    chm_raster: Path,
    out_root: Path,
    processed_root: Path,
    method: str,
    chm_variant_name: str,
    extra_kwargs: dict[str, Any],
) -> str:
    module = _load_algorithm_module(method)

    if module is None:
        return _write_placeholder_record(
            out_root,
            method=method,
            processed_root=processed_root,
            chm_variant_name=chm_variant_name,
            chm_raster=chm_raster,
        )

    fn = getattr(module, "run_itd_on_chm", None)
    if fn is None:
        return _write_placeholder_record(
            out_root,
            method=method,
            processed_root=processed_root,
            chm_variant_name=chm_variant_name,
            chm_raster=chm_raster,
        )

    result = fn(
        chm_raster=chm_raster,
        out_root=out_root,
        processed_root=processed_root,
        method=method,
        source_chm_variant=chm_variant_name,
        **extra_kwargs,
    )

    if isinstance(result, dict):
        stem = _sanitize_name(chm_raster.stem)
        out_fp = out_root / f"{stem}_itd_result.json"
        _write_json(out_fp, result)
        return str(out_fp)

    return str(result)


def _write_manifest(
    out_root: Path,
    *,
    processed_root: Path,
    method: str,
    source_chm_variant: str,
    source_rasters: list[Path],
    outputs: list[str],
    implementation_state: str,
) -> None:
    payload = {
        "module": PRODUCT_ITD,
        "method": method,
        "algorithm_package": ALGORITHM_PACKAGE,
        "implementation_state": implementation_state,
        "source_root": str(processed_root),
        "source_chm_variant": source_chm_variant,
        "source_rasters": [str(p) for p in source_rasters],
        "outputs": outputs,
        "message": (
            "FAST_ITD has been routed through a future-ready algorithm interface. "
            "If a matching module exists under fastgc.itd_algorithms and exposes "
            "run_itd_on_chm(...), it will be used. Otherwise placeholder records are written."
        ),
    }
    _write_json(out_root / "itd_manifest.json", payload)


def run_itd_from_processed_root(
    processed_root: str | Path,
    *,
    method: str = "placeholder",
    source_chm: str | None = None,
    skip_existing: bool = False,
    overwrite: bool = False,
    n_jobs: int = 1,
    joblib_backend: str = DEFAULT_BACKEND,
    joblib_batch_size: int | str = 1,
    joblib_pre_dispatch: str = "2*n_jobs",
    **kwargs: Any,
) -> str:
    processed_root = Path(processed_root)
    if not processed_root.exists():
        raise FileNotFoundError(f"Processed root not found: {processed_root}")

    method = str(method).strip().lower()
    if method not in SUPPORTED_ITD_METHODS:
        raise ValueError(f"Unsupported ITD method: {method}")

    chm_root = _resolve_chm_root(processed_root)
    chm_variant_root = _resolve_source_chm_variant(chm_root, source_chm)
    chm_variant_name = chm_variant_root.name
    source_rasters = _list_chm_rasters(chm_variant_root)

    out_root = processed_root / PRODUCT_ITD / method / chm_variant_name
    if skip_existing and _existing_output(out_root) and not overwrite:
        log_info(f"[SKIP] Existing FAST_ITD output found: {out_root}")
        return str(out_root)

    out_root.mkdir(parents=True, exist_ok=True)

    log_info(f"FAST_ITD method: {method}")
    log_info(f"FAST_ITD CHM variant: {chm_variant_name}")
    log_info(f"FAST_ITD source rasters: {len(source_rasters)}")

    implementation_state = "implemented" if _load_algorithm_module(method) is not None else "placeholder"

    def _task(chm_raster: Path):
        return _algorithm_runner(
            chm_raster=chm_raster,
            out_root=out_root,
            processed_root=processed_root,
            method=method,
            chm_variant_name=chm_variant_name,
            extra_kwargs=kwargs,
        )

    summary = run_stage(
        items=source_rasters,
        worker_fn=_task,
        stage_name=f"FAST-GC derive ITD [{method}]",
        unit="raster",
        current_name_fn=lambda p: p.name,
        n_jobs=n_jobs,
        backend=joblib_backend,
        batch_size=joblib_batch_size,
        pre_dispatch=joblib_pre_dispatch,
    )

    outputs: list[str] = []
    for rec in summary.get("records", []):
        result = rec.get("result")
        if result is not None:
            outputs.append(str(result))

    _write_manifest(
        out_root,
        processed_root=processed_root,
        method=method,
        source_chm_variant=chm_variant_name,
        source_rasters=source_rasters,
        outputs=outputs,
        implementation_state=implementation_state,
    )

    return str(out_root)


__all__ = [
    "PRODUCT_ITD",
    "ALGORITHM_PACKAGE",
    "SUPPORTED_ITD_METHODS",
    "run_itd_from_processed_root",
]
