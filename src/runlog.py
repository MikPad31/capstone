"""
Run-log helpers shared by every exploratory runner: timestamped progress, [verify] and [report] lines.

Promoted from `exploratory/target_pipeline.py` unchanged in behaviour.
"""

from __future__ import annotations

import sys
import time

try:
    import resource
except ImportError:  # Windows has no `resource`; rss_gb falls back to the Win32 API
    resource = None

_failures: list[str] = []


def log(msg: str) -> None:
    """Prints a timestamped progress line."""

    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def check(ok: bool, msg: str) -> bool:
    """
    Records a pass/fail as a [verify] line without aborting the run.
    Failures accumulate and are turned into a non-zero exit by `finish`.
    """

    print(f"[verify] {'PASS' if ok else 'FAIL'} — {msg}", flush=True)
    if not ok:
        _failures.append(msg)
    return ok


def report(msg: str) -> None:
    """Prints a measured quantity as a [report] line. Never asserted."""

    print(f"[report] {msg}", flush=True)


def _windows_peak_bytes() -> int:
    """Peak working set of this process in bytes, via psapi.GetProcessMemoryInfo."""

    import ctypes
    from ctypes import wintypes

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32")
    psapi = ctypes.WinDLL("psapi")
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError()
    return int(counters.PeakWorkingSetSize)


def rss_gb() -> float:
    """Peak resident set size of this process in gigabytes."""

    if resource is None:
        return _windows_peak_bytes() / 1e9
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on Darwin, KB on Linux
    return peak / 1e9 if sys.platform == "darwin" else peak / 1e6


def finish(start: float) -> None:
    """Prints elapsed time and peak memory, then exits non-zero if any `check()` failed."""

    log(f"done in {time.time() - start:.1f}s. peak rss={rss_gb():.2f}GB")
    if _failures:
        log(f"{len(_failures)} check(s) failed")
        sys.exit(1)
