"""Windows-safe transactional directory publication for FAST-GIS."""
from __future__ import annotations

import gc
import os
import shutil
import tempfile
import time
from pathlib import Path


def publish_directory(staging, target, *, retries=6, delay=0.25):
    """Publish a completed staging directory without overwriting a target.

    First attempts an atomic same-volume rename.  On Windows, transient file
    handles from readers/AV/indexers can make directory rename fail with
    WinError 5; retries are followed by a copy-to-sibling + atomic rename
    fallback.  The caller still owns cleanup of the original staging path.
    """
    staging = Path(staging).resolve()
    target = Path(target).resolve()
    if not staging.is_dir():
        raise FileNotFoundError(staging)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)

    last = None
    for attempt in range(max(1, int(retries))):
        try:
            gc.collect()
            os.replace(staging, target)
            return target
        except PermissionError as exc:
            last = exc
            time.sleep(float(delay) * (attempt + 1))

    sibling = Path(tempfile.mkdtemp(prefix=f".{target.name}_publish_", dir=target.parent))
    shutil.rmtree(sibling)
    try:
        shutil.copytree(staging, sibling, copy_function=shutil.copy2)
        if target.exists():
            raise FileExistsError(target)
        os.replace(sibling, target)
        return target
    except Exception:
        if sibling.exists():
            shutil.rmtree(sibling, ignore_errors=True)
        if last is not None:
            raise RuntimeError(
                f"FAST-GIS completed staging but could not publish {target}: {last}"
            ) from last
        raise
