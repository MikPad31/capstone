"""
Run-log helpers shared by every exploratory runner: timestamped progress, [verify] and [report] lines.

Promoted from `exploratory/target_pipeline.py` unchanged in behaviour.
"""

from __future__ import annotations

import resource
import sys
import time

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


def rss_gb() -> float:
    """Peak resident set size of this process in gigabytes."""

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on Darwin, KB on Linux
    return peak / 1e9 if sys.platform == "darwin" else peak / 1e6


def finish(start: float) -> None:
    """Prints elapsed time and peak memory, then exits non-zero if any `check()` failed."""

    log(f"done in {time.time() - start:.1f}s. peak rss={rss_gb():.2f}GB")
    if _failures:
        log(f"{len(_failures)} check(s) failed")
        sys.exit(1)
