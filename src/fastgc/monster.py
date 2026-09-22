from __future__ import annotations

import math
import os
import re
import sys
import time
import traceback
from collections import deque
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


def format_rate(rate_per_sec: float, unit: str = "item") -> str:
    """Human-readable throughput without changing timing/ETA semantics."""
    rate = max(0.0, float(rate_per_sec))
    singular = str(unit or "item").rstrip("s") or "item"
    plural = _plural_unit(singular, 2)

    if rate >= 1.0:
        return f"{rate:.2f} {plural}/s"

    if rate > 0.0:
        sec_per_item = 1.0 / rate
        if sec_per_item >= 60.0:
            minutes = int(sec_per_item // 60)
            seconds = int(round(sec_per_item - 60 * minutes))
            if seconds >= 60:
                minutes += 1
                seconds = 0
            return f"{minutes}m {seconds:02d}s/{singular}"
        return f"{sec_per_item:.1f} s/{singular}"

    return f"0.00 {plural}/s"


def _fastgc_debug_enabled() -> bool:
    """Return True only when internal FAST-GC diagnostics are requested."""
    return os.environ.get("FASTGC_DEBUG", "").strip().lower() in {
        "1", "true", "yes", "on"
    }


def log_info(message: str) -> None:
    """Emit routine internal diagnostics only in FASTGC_DEBUG mode."""
    if _fastgc_debug_enabled():
        print(f"[INFO] {message}")


def log_skip(message: str) -> None:
    print(f"[SKIP] {message}")


def log_fail(message: str) -> None:
    print(f"[FAIL] {message}")


def log_tile_stage(message: str) -> None:
    print(f"    [tile-stage] {message}")


def stage_banner(
    stage_name: str,
    *,
    source: str | None = None,
    total: int | None = None,
    unit: str = "item",
    **_: Any,
) -> None:
    """Compatibility hook; public progress is shown by the live stage line."""
    return

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


def configure_rayon_threads(jobs: int) -> int:
    """Coordinate inner Rayon threads with outer FAST-GC tile workers."""
    cpu_total = os.cpu_count() or 1
    jobs = max(1, int(jobs))
    threads = max(1, cpu_total // jobs)

    # FAST-GC owns this setting for its native kernels so a previous
    # shell value cannot accidentally oversubscribe tile workers.
    os.environ["RAYON_NUM_THREADS"] = str(threads)
    return threads


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



def _public_progress_identity(stage_name: str) -> tuple[str, str | None]:
    """Map internal stage descriptions to stable public workflow/product labels."""
    raw = str(stage_name or "").strip()
    upper = raw.upper().replace("_", "-")

    sensor = None
    for candidate in ("ALS", "ULS", "TLS"):
        if re.search(rf"(?<![A-Z0-9]){candidate}(?![A-Z0-9])", upper):
            sensor = candidate
            break

    # Tiling is a first-class public stage.
    if "TILING" in upper or upper.startswith("TILE "):
        return "TILING", sensor

    # Resolve the public product first.
    product = None
    product_patterns = (
        ("FAST-POINTCLOUDS", ("FAST-TREECLOUDS", "FAST-POINTCLOUDS", "TREECLOUD")),
        ("FAST-STRUCTURE", ("FAST-STRUCTURE",)),
        ("FAST-TERRAIN", ("FAST-TERRAIN",)),
        ("FAST-CHANGE", ("FAST-CHANGE",)),
        ("FAST-ITD", ("FAST-ITD",)),
        ("FAST-CHM", ("FAST-CHM", "CHM")),
        ("FAST-DSM", ("FAST-DSM", "DSM")),
        ("FAST-DEM", ("FAST-DEM", "DEM")),
        ("FAST-NORMALIZED", ("FAST-NORMALIZED", "NORMALIZED")),
        ("FAST-GC", ("FAST-GC", "CLASSIFY", "GROUND")),
    )
    for public_name, patterns in product_patterns:
        if any(pattern in upper for pattern in patterns):
            product = public_name
            break

    # Merge is public, but internal trim/crop preparation is intentionally
    # presented as part of the product merge rather than as another stage.
    if "MERGE" in upper or "TRIM" in upper:
        return f"MERGE {product or 'PRODUCT'}", sensor

    if product is not None:
        return product, sensor

    if "WORKFLOW" in upper:
        return "WORKFLOW", sensor

    return "PROCESSING", sensor


def _plural_unit(unit: str, count: int) -> str:
    unit = str(unit or "item")
    if count == 1:
        return unit
    return unit if unit.endswith("s") else f"{unit}s"


class ProgressDashboard:
    """Shared single-line FAST-family progress dashboard.

    The dashboard deliberately exposes only the public product/sensor identity,
    progress, throughput, elapsed/remaining time, resources, and outcome counts.
    Internal algorithm/stage names remain private to the processing pipeline.
    """

    def __init__(
        self,
        stage_name: str,
        total: int,
        *,
        unit: str = "tile",
        enabled: bool = True,
        bar_width: int = 24,
    ):
        self.stage_name = str(stage_name)
        self.product, self.sensor = _public_progress_identity(self.stage_name)
        self.total = max(0, int(total))
        self.unit = str(unit or "item")
        self.enabled = bool(enabled)
        self.bar_width = max(12, int(bar_width))
        self.start_time = time.perf_counter()
        self.done = 0
        self.current_name = ""
        self.current_file = ""
        self.current_item = ""
        self.ok = 0
        self.skipped = 0
        self.failed = 0
        self._outcomes_supplied = False

        # Recent completion samples stabilize throughput/ETA without changing
        # any processing behavior.  Each entry is (done_count, perf_counter).
        self._rate_samples = deque(maxlen=24)
        self._rate_samples.append((0, self.start_time))

        self._drawn = False
        self._last_snapshot_done = -1
        self._last_snapshot_time = 0.0

        self._native_win = False
        self._win_handle = None
        self._win_kernel32 = None
        self._win_COORD = None
        self._win_row = 0

        if os.name == "nt":
            self._init_windows_console()

        try:
            self._tty = self._native_win or bool(
                getattr(sys.stdout, "isatty", lambda: False)()
            )
        except Exception:
            self._tty = self._native_win

    def _init_windows_console(self) -> None:
        try:
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            handle = kernel32.GetStdHandle(-11)
            if handle in (0, -1):
                return

            class _COORD(ctypes.Structure):
                _fields_ = [("X", ctypes.c_short), ("Y", ctypes.c_short)]

            class _SMALL_RECT(ctypes.Structure):
                _fields_ = [
                    ("Left", ctypes.c_short),
                    ("Top", ctypes.c_short),
                    ("Right", ctypes.c_short),
                    ("Bottom", ctypes.c_short),
                ]

            class _CSBI(ctypes.Structure):
                _fields_ = [
                    ("dwSize", _COORD),
                    ("dwCursorPosition", _COORD),
                    ("wAttributes", wintypes.WORD),
                    ("srWindow", _SMALL_RECT),
                    ("dwMaximumWindowSize", _COORD),
                ]

            csbi = _CSBI()
            if not kernel32.GetConsoleScreenBufferInfo(handle, ctypes.byref(csbi)):
                return

            self._native_win = True
            self._win_handle = handle
            self._win_kernel32 = kernel32
            self._win_COORD = _COORD
            self._win_CSBI = _CSBI
            self._win_row = int(csbi.dwCursorPosition.Y)
        except Exception:
            self._native_win = False

    def _resources(self) -> tuple[str, str]:
        if psutil is None:
            return "n/a", "n/a"
        try:
            cpu = float(psutil.cpu_percent(interval=None))
            vm = psutil.virtual_memory()
            used_gb = float(vm.total - vm.available) / (1024**3)
            total_gb = float(vm.total) / (1024**3)
            return f"{cpu:.0f}%", f"{used_gb:.1f}/{total_gb:.1f}G"
        except Exception:
            return "n/a", "n/a"

    def _console_width(self) -> int:
        if self._native_win:
            try:
                import ctypes

                csbi = self._win_CSBI()
                if self._win_kernel32.GetConsoleScreenBufferInfo(
                    self._win_handle, ctypes.byref(csbi)
                ):
                    width = int(csbi.srWindow.Right - csbi.srWindow.Left + 1)
                    return max(40, width - 1)
            except Exception:
                pass
        try:
            import shutil as _shutil
            return max(40, int(_shutil.get_terminal_size((160, 24)).columns) - 1)
        except Exception:
            return 159

    def _identity(self) -> str:
        return f"{self.product} | {self.sensor}" if self.sensor else self.product

    def _rate(self, now: float) -> float:
        if self.done <= 0:
            return 0.0

        # Keep a short recent window; it reacts to changing tile complexity but
        # avoids the unstable ETA produced by a single early tile.
        self._rate_samples.append((self.done, now))
        newest_done, newest_time = self._rate_samples[-1]

        oldest_done, oldest_time = self._rate_samples[0]
        # Prefer a sample at least ~8 seconds old when one exists.
        for sample_done, sample_time in self._rate_samples:
            if newest_time - sample_time >= 8.0:
                oldest_done, oldest_time = sample_done, sample_time
                break

        delta_done = newest_done - oldest_done
        delta_time = newest_time - oldest_time
        if delta_done > 0 and delta_time > 0:
            return delta_done / delta_time

        elapsed = max(1e-9, now - self.start_time)
        return self.done / elapsed

    def _render_line(self, *, elapsed_item_sec: float | None = None) -> str:
        now = time.perf_counter()
        elapsed = max(0.0, now - self.start_time)
        done = max(0, min(self.done, self.total))
        frac = (done / self.total) if self.total else 1.0
        filled = int(round(frac * self.bar_width))
        filled = max(0, min(self.bar_width, filled))
        bar = "\u2588" * filled + "\u2591" * (self.bar_width - filled)

        rate = self._rate(now)
        remaining = max(0, self.total - done)
        eta = (remaining / rate) if rate > 0 else None
        cpu, ram = self._resources()
        units = _plural_unit(self.unit, 2)

        if rate > 0:
            rate_text = format_rate(rate, self.unit)
        elif elapsed_item_sec is not None and elapsed_item_sec > 0:
            rate_text = format_rate(1.0 / float(elapsed_item_sec), self.unit)
        else:
            rate_text = format_rate(0.0, self.unit)

        line = (
            f"{self._identity()} | [{bar}] | {frac*100:5.1f}% | {done}/{self.total}"
            f" | {rate_text}"
            f" | elapsed {format_seconds(elapsed)}"
            f" | ETA {format_seconds(eta)}"
            f" | CPU {cpu}"
            f" | RAM {ram}"
        )
        if self._outcomes_supplied:
            line += (
                f" | ok {self.ok}"
                f" | skipped {self.skipped}"
                f" | failed {self.failed}"
            )

        width = self._console_width()
        if len(line) > width:
            line = line[: max(1, width - 1)]
        return line

    def _write_native_windows(self, line: str) -> bool:
        if not self._native_win:
            return False

        try:
            import ctypes
            from ctypes import wintypes

            width = self._console_width()
            line = line[:width]
            padded = line.ljust(width)

            coord = self._win_COORD(0, int(self._win_row))

            if not self._win_kernel32.SetConsoleCursorPosition(
                self._win_handle, coord
            ):
                return False

            # Write complete line normally first.
            written = wintypes.DWORD()
            ok = self._win_kernel32.WriteConsoleW(
                self._win_handle,
                ctypes.c_wchar_p(padded),
                len(padded),
                ctypes.byref(written),
                None,
            )
            if not ok:
                return False

            # Color only the completed part of the progress bar green.
            bar_start = line.find("[")
            bar_end = line.find("]", bar_start + 1)

            if bar_start >= 0 and bar_end > bar_start:
                bar_text = line[bar_start + 1:bar_end]
                filled_count = len(bar_text) - len(bar_text.lstrip("\u2588"))

                if filled_count > 0:
                    green_coord = self._win_COORD(
                        int(bar_start + 1),
                        int(self._win_row),
                    )

                    # FOREGROUND_GREEN | FOREGROUND_INTENSITY
                    green_attr = 0x0002 | 0x0008

                    self._win_kernel32.FillConsoleOutputAttribute(
                        self._win_handle,
                        green_attr,
                        filled_count,
                        green_coord,
                        ctypes.byref(written),
                    )

            self._win_kernel32.SetConsoleCursorPosition(
                self._win_handle, coord
            )
            self._drawn = True
            return True

        except Exception:
            return False
    def _write_line(self, line: str) -> None:
        if not self.enabled:
            return

        if self._native_win and self._write_native_windows(line):
            return

        if self._tty:
            #
            # Native Windows consoles are handled above through WriteConsoleW.
            # For any remaining TTY, keep the dynamic single-line dashboard,
            # but never allow the terminal encoding to terminate processing.
            encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
            try:
                safe_line = line.encode(encoding, errors="strict").decode(encoding)
            except (UnicodeEncodeError, LookupError):
                safe_line = (
                    line
                    .replace("\u2588", "#")
                    .replace("\u2591", "-")
                )

            sys.stdout.write("\r\x1b[2K" + safe_line)
            sys.stdout.flush()
            self._drawn = True
            return

        # Redirected/captured output cannot redraw. Emit sparse snapshots only.
        now = time.perf_counter()
        stride = max(1, int(math.ceil(max(1, self.total) * 0.05)))
        should_emit = (
            self.done >= self.total
            or self._last_snapshot_done < 0
            or (self.done - self._last_snapshot_done) >= stride
            or (now - self._last_snapshot_time) >= 10.0
        )
        if should_emit:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()
            self._last_snapshot_done = self.done
            self._last_snapshot_time = now
            self._drawn = True

    def update(
        self,
        n: int = 1,
        *,
        current_name: str | None = None,
        current_file: str | None = None,
        current_item: str | None = None,
        elapsed_item_sec: float | None = None,
        ok_count: int | None = None,
        skipped_count: int | None = None,
        failed_count: int | None = None,
    ) -> None:
        if not self.enabled:
            return

        self.done = min(self.total, self.done + max(0, int(n)))
        if current_name is not None:
            self.current_name = str(current_name)
        if current_file is not None:
            self.current_file = str(current_file)
        if current_item is not None:
            self.current_item = str(current_item)
        if any(v is not None for v in (ok_count, skipped_count, failed_count)):
            self._outcomes_supplied = True

        if ok_count is not None:
            self.ok = int(ok_count)
        if skipped_count is not None:
            self.skipped = int(skipped_count)
        if failed_count is not None:
            self.failed = int(failed_count)

        self._write_line(self._render_line(elapsed_item_sec=elapsed_item_sec))

    def set_context(
        self,
        *,
        current_file: str | None = None,
        current_item: str | None = None,
    ) -> None:
        if current_file is not None:
            self.current_file = str(current_file)
        if current_item is not None:
            self.current_item = str(current_item)
        if self.enabled:
            self._write_line(self._render_line())

    def close(self) -> None:
        if not self.enabled or not self._drawn:
            return

        if self._native_win:
            try:
                import ctypes
                from ctypes import wintypes

                width = self._console_width()
                coord = self._win_COORD(0, int(self._win_row))
                self._win_kernel32.SetConsoleCursorPosition(self._win_handle, coord)
                written = wintypes.DWORD()
                self._win_kernel32.WriteConsoleW(
                    self._win_handle,
                    ctypes.c_wchar_p(" " * width),
                    width,
                    ctypes.byref(written),
                    None,
                )
                self._win_kernel32.SetConsoleCursorPosition(self._win_handle, coord)
                final_line = self._render_line()
                self._win_kernel32.WriteConsoleW(
                    self._win_handle,
                    ctypes.c_wchar_p(final_line),
                    len(final_line),
                    ctypes.byref(written),
                    None,
                )
                self._win_kernel32.WriteConsoleW(
                    self._win_handle,
                    ctypes.c_wchar_p("\n"),
                    1,
                    ctypes.byref(written),
                    None,
                )
                return
            except Exception:
                pass

        if self._tty:
            sys.stdout.write("\n")
            sys.stdout.flush()


def _make_progress_bar(stage_name: str, total: int, unit: str):
    dashboard_cls = globals().get('ProgressDashboard')
    if dashboard_cls is not None:
        return dashboard_cls(stage_name, total, unit=unit, enabled=True)
    if tqdm is not None:
        return tqdm(total=int(total), desc=str(stage_name), unit=str(unit), dynamic_ncols=True, leave=False)
    return None



def _update_bar(
    bar: Any,
    *,
    current_name: str,
    idx_done: int,
    total: int,
    elapsed_item_sec: float,
    start_time: float,
    unit: str,
    ok_count: int,
    skipped_count: int,
    failed_count: int,
) -> None:
    if bar is None:
        return

    # One unified dashboard update.  The dashboard owns speed/ETA/resource
    # calculations so individual product stages do not create nested bars.
    bar.update(
        1,
        current_name=current_name,
        current_item=current_name,
        elapsed_item_sec=elapsed_item_sec,
        ok_count=ok_count,
        skipped_count=skipped_count,
        failed_count=failed_count,
    )


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
    product, sensor = _public_progress_identity(stage_name)
    public_name = f"{product} | {sensor}" if sensor else product
    count_label = _plural_unit(unit, summary.total)
    rate = (summary.total / summary.elapsed_sec) if summary.elapsed_sec > 0 else 0.0
    print(
        f"[DONE] {public_name}: {summary.total} {count_label} | "
        f"elapsed={format_seconds(summary.elapsed_sec)} | "
        f"rate={format_rate(rate, unit)} | "
        f"ok={summary.ok} | skipped={summary.skipped} | failed={summary.failed}"
    )


def _make_parallel_iterator(
    *,
    jobs: int,
    backend: str,
    batch_size: int | str,
    pre_dispatch: str | int,
    item_list: list[Any],
    callable_fn: Callable[[Any], Any],
    item_name_fn: Callable[[Any], str] | None,
):
    parallel_backend = "loky" if backend == "multiprocessing" else backend

    # Best: as-completed streaming
    try:
        parallel = Parallel(
            n_jobs=jobs,
            backend=parallel_backend,
            batch_size=batch_size,
            pre_dispatch=pre_dispatch,
            return_as="generator_unordered",
        )
        return parallel(
            delayed(_execute_callable)(i, item, callable_fn, item_name_fn)
            for i, item in enumerate(item_list, start=1)
        )
    except TypeError:
        pass

    # Good: ordered streaming
    try:
        parallel = Parallel(
            n_jobs=jobs,
            backend=parallel_backend,
            batch_size=batch_size,
            pre_dispatch=pre_dispatch,
            return_as="generator",
        )
        return parallel(
            delayed(_execute_callable)(i, item, callable_fn, item_name_fn)
            for i, item in enumerate(item_list, start=1)
        )
    except TypeError:
        pass

    # Fallback: blocking collection
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
    return iter(outputs)


def _print_failure_summary(records: Sequence[StageRecord], *, max_examples: int = 12) -> None:
    """Print a compact, de-duplicated failure summary after a stage finishes.

    Per-item failures are intentionally not printed while a progress bar is active;
    doing so corrupts the single-bar display and, on PowerShell, can look like a
    native-command failure when stderr is redirected.
    """
    failed = [r for r in records if r.status == "failed"]
    if not failed:
        return

    groups: dict[str, list[str]] = {}
    for rec in failed:
        first = (rec.error or "unknown error").splitlines()[0].strip() or "unknown error"
        groups.setdefault(first, []).append(rec.name)

    print(f"[FAILURES] {len(failed)} item(s) failed across {len(groups)} unique error(s).")
    shown = 0
    for message, names in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        if shown >= max_examples:
            break
        examples = ", ".join(names[:3])
        extra = len(names) - min(3, len(names))
        suffix = f" (+{extra} more)" if extra > 0 else ""
        print(f"  - {len(names)}x {message} | example: {examples}{suffix}")
        shown += 1

    remaining = len(groups) - shown
    if remaining > 0:
        print(f"  - ... {remaining} additional unique error type(s) omitted")


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
    if func is None and worker is None:
        raise TypeError("run_stage requires 'func' or 'worker'")

    if func is not None and worker is not None:
        raise TypeError("Provide only one of 'func' or 'worker'")

    callable_fn = func if func is not None else worker
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

    ok_count = 0
    skipped_count = 0
    failed_count = 0

    jobs = resolve_n_jobs(n_jobs)
    configure_rayon_threads(jobs)
    backend = (backend or DEFAULT_BACKEND).lower()

    if backend not in BACKEND_CHOICES:
        raise ValueError(f"Unsupported backend: {backend}")

    use_parallel = (
        backend != "sequential"
        and jobs > 1
        and Parallel is not None
        and delayed is not None
    )

    if not use_parallel:
        try:
            for idx, item in enumerate(item_list, start=1):
                rec = _execute_callable(idx, item, callable_fn, item_name_fn=item_name_fn)
                records.append(rec)

                if rec.status == "ok":
                    ok_count += 1
                elif rec.status == "skipped":
                    skipped_count += 1
                else:
                    failed_count += 1

                _update_bar(
                    bar,
                    current_name=rec.name,
                    idx_done=idx,
                    total=total,
                    elapsed_item_sec=rec.elapsed_sec,
                    start_time=start_time,
                    unit=unit,
                    ok_count=ok_count,
                    skipped_count=skipped_count,
                    failed_count=failed_count,
                )

        finally:
            if bar is not None:
                try:
                    bar.close()
                except Exception:
                    pass

        elapsed = time.perf_counter() - start_time
        summary = summarize_stage(stage_name, records, elapsed)
        finish_stage(stage_name, summary, unit=unit)
        _print_failure_summary(records)
        return summary

    iterator = _make_parallel_iterator(
        jobs=jobs,
        backend=backend,
        batch_size=batch_size,
        pre_dispatch=pre_dispatch,
        item_list=item_list,
        callable_fn=callable_fn,
        item_name_fn=item_name_fn,
    )

    try:
        for idx_done, rec in enumerate(iterator, start=1):
            records.append(rec)

            if rec.status == "ok":
                ok_count += 1
            elif rec.status == "skipped":
                skipped_count += 1
            else:
                failed_count += 1

            _update_bar(
                bar,
                current_name=rec.name,
                idx_done=idx_done,
                total=total,
                elapsed_item_sec=rec.elapsed_sec,
                start_time=start_time,
                unit=unit,
                ok_count=ok_count,
                skipped_count=skipped_count,
                failed_count=failed_count,
            )

    finally:
        if bar is not None:
            try:
                bar.close()
            except Exception:
                pass

    elapsed = time.perf_counter() - start_time
    summary = summarize_stage(stage_name, records, elapsed)
    finish_stage(stage_name, summary, unit=unit)
    _print_failure_summary(records)
    return summary



def progress_bar(*args, **kwargs):
    """Backward-compatible API for older FAST-GC modules."""

    dashboard_cls = globals().get("ProgressDashboard")

    if dashboard_cls is not None:

        class _CompatBar:
            def __init__(self):
                total = int(kwargs.get("total", 0))
                desc = str(kwargs.get("desc", ""))
                unit = str(kwargs.get("unit", "item"))
                disable = bool(kwargs.get("disable", False))

                self._dash = dashboard_cls(
                    desc,
                    total,
                    unit=unit,
                    enabled=not disable,
                )
                self.total = total
                self.n = 0

            def update(self, n=1):
                self.n += int(n)
                self._dash.update(
                    int(n),
                    current_item=f"step {self.n}/{self.total}",
                )

            def set_description(self, *a, **k):
                pass

            def set_description_str(self, *a, **k):
                pass

            def set_postfix_str(self, *a, **k):
                pass

            def refresh(self):
                pass

            def close(self):
                self._dash.close()

        return _CompatBar()

    try:
        from tqdm.auto import tqdm
        return tqdm(*args, **kwargs)
    except Exception:

        class DummyBar:
            def update(self, *a, **k): pass
            def set_description(self, *a, **k): pass
            def set_description_str(self, *a, **k): pass
            def set_postfix_str(self, *a, **k): pass
            def refresh(self): pass
            def close(self): pass

        return DummyBar()


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
    "ProgressDashboard",
    "run_stage",
    "stage_banner",
    "summarize_stage",
]


