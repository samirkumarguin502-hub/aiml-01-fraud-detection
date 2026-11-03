"""Runtime/memory facts recorded with every training run (no external services)."""
from __future__ import annotations

import os
import platform
import sys


def peak_memory_mb() -> float | None:
    """Peak resident memory of this process in MB, or None if the platform gives no way to read it."""
    try:
        import resource  # POSIX only
        v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(v / (1024 * 1024) if sys.platform == "darwin" else v / 1024, 1)
    except ImportError:
        pass
    try:
        import psutil
        mi = psutil.Process().memory_info()
        peak = getattr(mi, "peak_wset", None)  # Windows peak working set
        return round((peak or mi.rss) / 1e6, 1)
    except Exception:  # noqa: BLE001
        return None


def environment_info() -> dict:
    return {"python": sys.version.split()[0], "platform": platform.platform(), "cpu_count": os.cpu_count()}
