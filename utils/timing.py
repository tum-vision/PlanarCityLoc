"""Compute-only timers (exclude I/O and host-device transfers from reported times)."""
from __future__ import annotations

import time

import torch


class GpuTimer:
    """Measure elapsed GPU kernel time via CUDA events."""

    def __init__(self) -> None:
        self.elapsed_s = 0.0
        self._use_cuda = torch.cuda.is_available()

    def __enter__(self) -> "GpuTimer":
        if self._use_cuda:
            self._start = torch.cuda.Event(enable_timing=True)
            self._end = torch.cuda.Event(enable_timing=True)
            self._start.record()
        else:
            self._cpu_t0 = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        if self._use_cuda:
            self._end.record()
            torch.cuda.synchronize()
            self.elapsed_s = self._start.elapsed_time(self._end) / 1000.0
        else:
            self.elapsed_s = time.perf_counter() - self._cpu_t0


class CpuTimer:
    """Wall time for CPU compute (caller must not perform file I/O in the block)."""

    def __init__(self) -> None:
        self.elapsed_s = 0.0

    def __enter__(self) -> "CpuTimer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        self.elapsed_s = time.perf_counter() - self._t0
