from __future__ import annotations

import math
import os
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

try:
    import psutil
except Exception:
    psutil = None

try:
    from joblib import Parallel, delayed
except Exception:
    Parallel = None
    delayed = None

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


DEFAULT_BACKEND = "loky"
BACKEND_CHOICES = ("loky", "multiprocessing", "threading", "sequential")


@dataclass(slots=True)
class StageRecord:
    index: int
    name: str
    status: str
    elapsed_sec: float
    result: Any = None
    error: str | None = None


@dataclass(slots=True)
class StageSummary:
    stage: str
    total: int
    ok: int
    skipped: int
    failed: int
    elapsed_sec: float
    avg_sec: float
    records: list[StageRecord]

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "total": self.total,
            "ok": self.ok,
            "skipped": self.skipped,
            "failed": self.failed,
            "elapsed_sec": self.elapsed_sec,
            "avg_sec": self.avg_sec,
            "records": [
                {
                    "index": r.index,
                    "name": r.name,
                    "status": r.status,
                    "elapsed_sec": r.elapsed_sec,
                    "result": r.result,
                    "error": r.error,
                }
                for r in self.records
            ],
        }


# ---------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------

def format_seconds(seconds: float | int | None) -> str:
    if seconds is None:
        return "n/a"
    seconds = float(seconds)
    if not math.isfinite(seconds):
        return "n/a"

    if seconds < 60:
        return f"{seconds:.2f}s"

    m, s = divmod(int(round(seconds)), 60)
    if m < 60:
        return f"{m:02d}:{s:02d}"

    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

def log_info(message: str) -> None:
    print(f"[INFO] {message}")


def log_skip(message: str) -> None:
    print(f"[SKIP] {message}")


def log_fail(message: str) -> None:
    print(f"[FAIL] {message}")


def log_tile_stage(message: str) -> None:
    print(f"    [tile-stage] {message}")


# ---------------------------------------------------------
# Stage banner
# ---------------------------------------------------------

def stage_banner(
    stage_name: str,
    *,
    source: str | None = None,
    total: int | None = None,
    unit: str = "item",
    **_: Any,
) -> None:

    log_info(f"Stage: {stage_name}")

    if source:
        log_info(f"Source: {source}")

    if total is not None:
        label = unit if unit else "item"
        if total == 1:
            log_info(f"Items: {total} {label}")
        else:
            log_info(f"Items: {total} {label}s")

    if psutil is not None:
        try:
            vm = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=None)

            log_info(
                "Resources: "
                f"cpu={cpu:.1f}% | "
                f"ram_used={vm.percent:.1f}% | "
                f"ram_free={vm.available / (1024**3):.1f} GB"
            )
        except Exception:
            pass


# ---------------------------------------------------------
# CPU helpers
# ---------------------------------------------------------

def resolve_n_jobs(
    n_jobs: int | None = None,
    *,
    reserve_cores: int = 1,
    max_jobs: int | None = None,
) -> int:

    cpu_total = os.cpu_count() or 1

    if n_jobs is None or n_jobs == 0:
        jobs = max(1, cpu_total - max(0, reserve_cores))

    elif n_jobs < 0:
        jobs = max(1, cpu_total + 1 + n_jobs)

    else:
        jobs = int(n_jobs)

    if max_jobs is not None:
        jobs = min(jobs, int(max_jobs))

    return max(1, min(jobs, cpu_total))


# ---------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------

def _infer_status(result: Any) -> str:
    if isinstance(result, dict):
        status = str(result.get("status", "ok")).lower()
        if status in {"ok", "skipped", "failed"}:
            return status
    return "ok"


def _extract_name(item: Any, item_name_fn: Callable[[Any], str] | None = None) -> str:

    if item_name_fn is not None:
        try:
            return str(item_name_fn(item))
        except Exception:
            pass

    if isinstance(item, dict):
        for key in ("name", "path", "tile", "file", "id"):
            if key in item:
                return os.path.basename(str(item[key]))

    if isinstance(item, (str, os.PathLike)):
        return os.path.basename(str(item))

    return str(item)


def _execute_callable(
    index: int,
    item: Any,
    func: Callable[[Any], Any],
    item_name_fn: Callable[[Any], str] | None = None,
) -> StageRecord:

    name = _extract_name(item, item_name_fn=item_name_fn)

    t0 = time.perf_counter()

    try:
        result = func(item)

        status = _infer_status(result)

        elapsed = time.perf_counter() - t0

        return StageRecord(
            index=index,
            name=name,
            status=status,
            elapsed_sec=elapsed,
            result=result,
            error=None,
        )

    except Exception as exc:

        elapsed = time.perf_counter() - t0

        tb = traceback.format_exc(limit=10)

        return StageRecord(
            index=index,
            name=name,
            status="failed",
            elapsed_sec=elapsed,
            result=None,
            error=f"{exc}\n{tb}",
        )


# ---------------------------------------------------------
# Progress bar
# ---------------------------------------------------------

def _make_progress_bar(stage_name: str, total: int, unit: str):

    if tqdm is None:
        return None

    return tqdm(
        total=total,
        desc=stage_name,
        unit=unit,
        dynamic_ncols=True,
        leave=True,
    )


def _update_bar(
    bar: Any,
    *,
    current_name: str,
    idx_done: int,
    total: int,
    elapsed_item_sec: float,
    start_time: float,
    unit: str,
) -> None:

    if bar is None:
        return

    elapsed_total = time.perf_counter() - start_time

    avg = elapsed_total / max(1, idx_done)

    remaining = max(0, total - idx_done)

    eta = avg * remaining

    postfix = (
        f"{idx_done}/{total} | "
        f"{elapsed_item_sec:.2f}s/{unit} | "
        f"eta={format_seconds(eta)} | "
        f"current={current_name}"
    )

    try:
        bar.set_postfix_str(postfix)
    except Exception:
        pass

    bar.update(1)


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

def summarize_stage(
    stage_name: str,
    records: Sequence[StageRecord],
    elapsed_sec: float,
) -> StageSummary:

    total = len(records)

    ok = sum(1 for r in records if r.status == "ok")

    skipped = sum(1 for r in records if r.status == "skipped")

    failed = sum(1 for r in records if r.status == "failed")

    avg = elapsed_sec / total if total else 0.0

    return StageSummary(
        stage=stage_name,
        total=total,
        ok=ok,
        skipped=skipped,
        failed=failed,
        elapsed_sec=elapsed_sec,
        avg_sec=avg,
        records=list(records),
    )


def finish_stage(stage_name: str, summary: StageSummary, *, unit: str = "tile") -> None:

    print(
        f"[TIME] {stage_name}: {summary.total} {unit}s | "
        f"total={format_seconds(summary.elapsed_sec)} | "
        f"avg={summary.avg_sec:.2f}s/{unit} | "
        f"ok={summary.ok} | skipped={summary.skipped} | failed={summary.failed}"
    )


# ---------------------------------------------------------
# MAIN EXECUTION ENGINE
# ---------------------------------------------------------

def run_stage(
    stage_name: str,
    items: Iterable[Any],
    func: Callable[[Any], Any] | None = None,
    *,
    worker: Callable[[Any], Any] | None = None,
    item_name_fn: Callable[[Any], str] | None = None,
    n_jobs: int = 1,
    backend: str = DEFAULT_BACKEND,
    batch_size: int | str = "auto",
    pre_dispatch: str | int = "2*n_jobs",
    source: str | None = None,
    unit: str = "tile",
    show_banner: bool = True,
    show_progress: bool = True,
    **_: Any,
) -> StageSummary:

    # ---- backwards compatibility ----

    if func is None and worker is None:
        raise TypeError("run_stage requires 'func' or 'worker'")

    if func is not None and worker is not None:
        raise TypeError("Provide only one of 'func' or 'worker'")

    callable_fn = func if func is not None else worker

    # ---------------------------------

    item_list = list(items)

    total = len(item_list)

    if show_banner:
        stage_banner(stage_name, source=source, total=total, unit=unit)

    if total == 0:
        summary = summarize_stage(stage_name, [], 0.0)
        finish_stage(stage_name, summary, unit=unit)
        return summary

    start_time = time.perf_counter()

    records: list[StageRecord] = []

    bar = _make_progress_bar(stage_name, total, unit) if show_progress else None

    jobs = resolve_n_jobs(n_jobs)

    backend = (backend or DEFAULT_BACKEND).lower()

    if backend not in BACKEND_CHOICES:
        raise ValueError(f"Unsupported backend: {backend}")

    use_parallel = (
        backend != "sequential"
        and jobs > 1
        and Parallel is not None
        and delayed is not None
    )

    if use_parallel:

        parallel_backend = "loky" if backend == "multiprocessing" else backend

        parallel = Parallel(
            n_jobs=jobs,
            backend=parallel_backend,
            batch_size=batch_size,
            pre_dispatch=pre_dispatch,
        )

        outputs = parallel(
            delayed(_execute_callable)(i, item, callable_fn, item_name_fn)
            for i, item in enumerate(item_list, start=1)
        )

        for idx_done, rec in enumerate(outputs, start=1):

            records.append(rec)

            _update_bar(
                bar,
                current_name=rec.name,
                idx_done=idx_done,
                total=total,
                elapsed_item_sec=rec.elapsed_sec,
                start_time=start_time,
                unit=unit,
            )

            if rec.status == "failed" and rec.error:
                log_fail(f"{rec.name}: {rec.error.splitlines()[0]}")

    else:

        for idx, item in enumerate(item_list, start=1):

            rec = _execute_callable(idx, item, callable_fn, item_name_fn)

            records.append(rec)

            _update_bar(
                bar,
                current_name=rec.name,
                idx_done=idx,
                total=total,
                elapsed_item_sec=rec.elapsed_sec,
                start_time=start_time,
                unit=unit,
            )

            if rec.status == "failed" and rec.error:
                log_fail(f"{rec.name}: {rec.error.splitlines()[0]}")

    if bar is not None:
        try:
            bar.close()
        except Exception:
            pass

    elapsed = time.perf_counter() - start_time

    summary = summarize_stage(stage_name, records, elapsed)

    finish_stage(stage_name, summary, unit=unit)

    return summary


__all__ = [
    "BACKEND_CHOICES",
    "DEFAULT_BACKEND",
    "StageRecord",
    "StageSummary",
    "finish_stage",
    "format_seconds",
    "log_fail",
    "log_info",
    "log_skip",
    "log_tile_stage",
    "resolve_n_jobs",
    "run_stage",
    "stage_banner",
    "summarize_stage",
]