from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

from .monster import DEFAULT_BACKEND, log_info, run_stage

PRODUCT_ITD = "FAST_ITD"
PRODUCT_CHM = "FAST_CHM"
PRODUCT_DSM = "FAST_DSM"
ALGORITHM_PACKAGE = "fastgc.itd_algorithms"

SUPPORTED_ITD_METHODS = {
    "placeholder",
    "lmf",
    "watershed",
    "yun2021",
    "dalponte2016",
    "li2012",
    "csp",
}

_RASTER_SUFFIXES = {".tif", ".tiff"}


def _existing_output(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_file():
        return True
    return any(path.iterdir())


def _is_raster_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in _RASTER_SUFFIXES


def _sanitize_name(name: str) -> str:
    safe = "".join(c if c.isalnum() or c in {"_", "-", "."} else "_" for c in name)
    return safe.strip("._") or "unnamed"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _infer_surface_from_filename(raster: Path) -> tuple[str, str]:
    stem = raster.stem

    if "_FAST_CHM_" in stem:
        tail = stem.split("_FAST_CHM_", 1)[1].strip()
        return "CHM", _sanitize_name(tail) or "unknown_chm"

    if stem.endswith("_FAST_CHM"):
        return "CHM", "FAST_CHM"

    if "_FAST_DSM_" in stem:
        tail = stem.split("_FAST_DSM_", 1)[1].strip()
        return "DSM", _sanitize_name(tail) or "dsm"

    if stem.endswith("_FAST_DSM"):
        return "DSM", "dsm"

    return "CHM", "direct_surface"


def _looks_like_chm_variant_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    parent = path.parent
    if parent.name != PRODUCT_CHM:
        return False
    tif_count = sum(1 for p in path.glob("*.tif") if p.is_file())
    return tif_count > 0


def _looks_like_dsm_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if path.name != PRODUCT_DSM:
        return False
    tif_count = sum(1 for p in path.glob("*.tif") if p.is_file())
    return tif_count > 0


def _find_processed_root_from_chm_variant_dir(variant_dir: Path) -> Path:
    if variant_dir.parent.name != PRODUCT_CHM:
        raise ValueError(f"Not a FAST_CHM variant folder: {variant_dir}")
    return variant_dir.parent.parent


def _find_processed_root_from_dsm_dir(dsm_dir: Path) -> Path:
    if dsm_dir.name != PRODUCT_DSM:
        raise ValueError(f"Not a FAST_DSM folder: {dsm_dir}")
    return dsm_dir.parent


def _list_rasters(folder: Path) -> list[Path]:
    rasters = sorted([p for p in folder.glob("*.tif") if p.is_file()])
    if not rasters:
        raise FileNotFoundError(f"No rasters found in: {folder}")
    return rasters


def _resolve_from_processed_root(
    processed_root: Path,
    source_chm: str | None,
) -> tuple[list[Path], str, str, Path, str]:
    chm_root = processed_root / PRODUCT_CHM
    if not chm_root.exists():
        raise FileNotFoundError(f"FAST_CHM folder not found: {chm_root}")

    if source_chm is not None and str(source_chm).strip():
        chm_variant_root = chm_root / str(source_chm).strip()
        if not chm_variant_root.exists():
            raise FileNotFoundError(f"Requested CHM variant not found: {chm_variant_root}")
        if not chm_variant_root.is_dir():
            raise NotADirectoryError(f"Requested CHM variant is not a folder: {chm_variant_root}")
    else:
        variants = sorted([p for p in chm_root.iterdir() if p.is_dir()])
        if not variants:
            raise FileNotFoundError(f"No CHM method folders found in: {chm_root}")
        chm_variant_root = variants[0]

    source_rasters = _list_rasters(chm_variant_root)
    surface_type = "CHM"
    surface_variant = chm_variant_root.name
    out_base_root = processed_root
    return source_rasters, surface_type, surface_variant, out_base_root, "processed_root"


def _resolve_from_chm_variant_dir(
    variant_dir: Path,
) -> tuple[list[Path], str, str, Path, str]:
    source_rasters = _list_rasters(variant_dir)
    surface_type = "CHM"
    surface_variant = variant_dir.name
    processed_root = _find_processed_root_from_chm_variant_dir(variant_dir)
    out_base_root = processed_root
    return source_rasters, surface_type, surface_variant, out_base_root, "chm_variant_dir"


def _resolve_from_dsm_dir(
    dsm_dir: Path,
) -> tuple[list[Path], str, str, Path, str]:
    source_rasters = _list_rasters(dsm_dir)
    surface_type = "DSM"
    surface_variant = "dsm"
    processed_root = _find_processed_root_from_dsm_dir(dsm_dir)
    out_base_root = processed_root
    return source_rasters, surface_type, surface_variant, out_base_root, "dsm_dir"


def _resolve_from_direct_raster(
    raster: Path,
    source_chm: str | None,
) -> tuple[list[Path], str, str, Path, str]:
    inferred_surface_type, inferred_surface_variant = _infer_surface_from_filename(raster)

    if source_chm is not None and str(source_chm).strip():
        surface_type = "CHM"
        surface_variant = str(source_chm).strip()
    else:
        surface_type = inferred_surface_type
        surface_variant = inferred_surface_variant

    out_base_root = raster.parent
    return [raster], surface_type, surface_variant, out_base_root, "direct_raster"




def _looks_like_merged_root(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(path.glob("*_FAST_CHM_*.tif")) or any(path.glob("*_FAST_CHM_*.tiff"))


def _resolve_from_merged_root(
    merged_root: Path,
    source_chm: str | None,
) -> tuple[list[Path], str, str, Path, str]:
    rasters = sorted([p for p in merged_root.glob("*_FAST_CHM_*.tif") if p.is_file()])
    rasters += sorted([p for p in merged_root.glob("*_FAST_CHM_*.tiff") if p.is_file()])
    if not rasters:
        dsm = sorted([p for p in merged_root.glob("*_FAST_DSM*.tif") if p.is_file()])
        dsm += sorted([p for p in merged_root.glob("*_FAST_DSM*.tiff") if p.is_file()])
        if dsm:
            return dsm, "DSM", "dsm", merged_root, "merged_root"
        raise FileNotFoundError(f"No merged CHM rasters found under: {merged_root}")

    if source_chm is not None and str(source_chm).strip():
        variant = _sanitize_name(str(source_chm).strip())
        selected = [p for p in rasters if _infer_surface_from_filename(p)[1] == variant]
        if not selected:
            raise FileNotFoundError(f"Merged CHM variant not found: {variant} under {merged_root}")
        return selected, "CHM", variant, merged_root, "merged_root"

    variants = sorted({_infer_surface_from_filename(p)[1] for p in rasters})
    if len(variants) > 1:
        raise FileNotFoundError(
            f"Multiple merged CHM variants found under {merged_root}; specify --itd_source_chm from {variants}"
        )
    variant = variants[0]
    selected = [p for p in rasters if _infer_surface_from_filename(p)[1] == variant]
    return selected, "CHM", variant, merged_root, "merged_root"

def _resolve_surface_inputs(
    in_path: Path,
    source_chm: str | None,
) -> tuple[list[Path], str, str, Path, str]:
    if not in_path.exists():
        raise FileNotFoundError(f"Input path not found: {in_path}")

    if _is_raster_file(in_path):
        return _resolve_from_direct_raster(in_path, source_chm)

    if _looks_like_chm_variant_dir(in_path):
        return _resolve_from_chm_variant_dir(in_path)

    if _looks_like_dsm_dir(in_path):
        return _resolve_from_dsm_dir(in_path)

    if _looks_like_merged_root(in_path):
        return _resolve_from_merged_root(in_path, source_chm)

    if in_path.is_dir():
        return _resolve_from_processed_root(in_path, source_chm)

    raise ValueError(f"Unsupported ITD input path: {in_path}")


def _write_placeholder_record(
    out_root: Path,
    *,
    method: str,
    source_root: Path,
    surface_type: str,
    surface_variant: str,
    source_raster: Path,
    input_mode: str,
) -> str:
    stem = _sanitize_name(source_raster.stem)
    out_fp = out_root / f"{stem}_itd_placeholder.json"
    payload = {
        "status": "placeholder",
        "module": PRODUCT_ITD,
        "method": method,
        "input_mode": input_mode,
        "surface_type": surface_type,
        "surface_variant": surface_variant,
        "source_root": str(source_root),
        "source_raster": str(source_raster),
        "message": (
            "FAST_ITD placeholder created. "
            "This raster was routed through the FAST_ITD pipeline, "
            "but the selected algorithm is not implemented yet."
        ),
    }
    _write_json(out_fp, payload)
    return str(out_fp)


def _load_algorithm_module(method: str):
    if method == "placeholder":
        return None

    module_name = f"{ALGORITHM_PACKAGE}.{method}"
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        return None


def _algorithm_runner(
    *,
    surface_raster: Path,
    out_root: Path,
    source_root: Path,
    method: str,
    surface_type: str,
    surface_variant: str,
    input_mode: str,
    extra_kwargs: dict[str, Any],
) -> str:
    module = _load_algorithm_module(method)

    if module is None:
        return _write_placeholder_record(
            out_root,
            method=method,
            source_root=source_root,
            surface_type=surface_type,
            surface_variant=surface_variant,
            source_raster=surface_raster,
            input_mode=input_mode,
        )

    fn_surface = getattr(module, "run_itd_on_surface", None)
    fn_chm = getattr(module, "run_itd_on_chm", None)

    if fn_surface is not None:
        result = fn_surface(
            surface_raster=surface_raster,
            out_root=out_root,
            source_root=source_root,
            method=method,
            surface_type=surface_type,
            surface_variant=surface_variant,
            input_mode=input_mode,
            **extra_kwargs,
        )
    elif fn_chm is not None:
        if surface_type != "CHM":
            return _write_placeholder_record(
                out_root,
                method=method,
                source_root=source_root,
                surface_type=surface_type,
                surface_variant=surface_variant,
                source_raster=surface_raster,
                input_mode=input_mode,
            )

        result = fn_chm(
            chm_raster=surface_raster,
            out_root=out_root,
            source_root=source_root,
            method=method,
            source_chm_variant=surface_variant,
            input_mode=input_mode,
            **extra_kwargs,
        )
    else:
        return _write_placeholder_record(
            out_root,
            method=method,
            source_root=source_root,
            surface_type=surface_type,
            surface_variant=surface_variant,
            source_raster=surface_raster,
            input_mode=input_mode,
        )

    if isinstance(result, dict):
        stem = _sanitize_name(surface_raster.stem)
        out_fp = out_root / f"{stem}_itd_result.json"
        _write_json(out_fp, result)
        return str(out_fp)

    return str(result)


def _write_manifest(
    out_root: Path,
    *,
    source_root: Path,
    method: str,
    surface_type: str,
    surface_variant: str,
    source_rasters: list[Path],
    outputs: list[str],
    implementation_state: str,
    input_mode: str,
) -> None:
    payload = {
        "module": PRODUCT_ITD,
        "method": method,
        "algorithm_package": ALGORITHM_PACKAGE,
        "implementation_state": implementation_state,
        "input_mode": input_mode,
        "surface_type": surface_type,
        "surface_variant": surface_variant,
        "source_root": str(source_root),
        "source_rasters": [str(p) for p in source_rasters],
        "outputs": outputs,
        "message": (
            "FAST_ITD accepts a processed root, a FAST_CHM variant folder, "
            "a FAST_DSM folder, or a direct merged raster. Matching algorithm "
            "modules are loaded from fastgc.itd_algorithms.<method> via "
            "run_itd_on_surface(...) when available, with backward-compatible "
            "support for run_itd_on_chm(...)."
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
    joblib_batch_size: int | str = "auto",
    joblib_pre_dispatch: str | int = "2*n_jobs",
    **kwargs: Any,
) -> str:
    in_path = Path(processed_root)
    if not in_path.exists():
        raise FileNotFoundError(f"ITD input not found: {in_path}")

    method = str(method).strip().lower()
    if method not in SUPPORTED_ITD_METHODS:
        raise ValueError(f"Unsupported ITD method: {method}")

    source_rasters, surface_type, surface_variant, out_base_root, input_mode = _resolve_surface_inputs(
        in_path=in_path,
        source_chm=source_chm,
    )

    out_root = out_base_root / PRODUCT_ITD / method / surface_type.lower() / surface_variant
    if skip_existing and _existing_output(out_root) and not overwrite:
        log_info(f"[SKIP] Existing FAST_ITD output found: {out_root}")
        return str(out_root)

    out_root.mkdir(parents=True, exist_ok=True)

    log_info(f"FAST_ITD method: {method}")
    log_info(f"FAST_ITD input mode: {input_mode}")
    log_info(f"FAST_ITD surface type: {surface_type}")
    log_info(f"FAST_ITD surface variant: {surface_variant}")
    log_info(f"FAST_ITD source rasters: {len(source_rasters)}")

    implementation_state = "implemented" if _load_algorithm_module(method) is not None else "placeholder"

    def _task(surface_raster: Path):
        return _algorithm_runner(
            surface_raster=surface_raster,
            out_root=out_root,
            source_root=in_path,
            method=method,
            surface_type=surface_type,
            surface_variant=surface_variant,
            input_mode=input_mode,
            extra_kwargs=kwargs,
        )

    summary = run_stage(
        stage_name=f"FAST-GC derive ITD [{method}]",
        items=source_rasters,
        worker=_task,
        item_name_fn=lambda p: p.name,
        n_jobs=n_jobs,
        backend=joblib_backend,
        batch_size=joblib_batch_size,
        pre_dispatch=joblib_pre_dispatch,
        source=str(in_path),
        unit="raster",
    )

    outputs: list[str] = []
    for rec in summary.records:
        if rec.result is not None:
            outputs.append(str(rec.result))

    _write_manifest(
        out_root,
        source_root=in_path,
        method=method,
        surface_type=surface_type,
        surface_variant=surface_variant,
        source_rasters=source_rasters,
        outputs=outputs,
        implementation_state=implementation_state,
        input_mode=input_mode,
    )

    return str(out_root)


__all__ = [
    "PRODUCT_ITD",
    "PRODUCT_CHM",
    "PRODUCT_DSM",
    "ALGORITHM_PACKAGE",
    "SUPPORTED_ITD_METHODS",
    "run_itd_from_processed_root",
]