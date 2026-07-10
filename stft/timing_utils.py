"""Utilities for lightweight host and CUDA phase timing."""

from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter

import torch


class Timer:
    """Accumulate timings from named context-managed phases.

    Host timings use :func:`time.perf_counter`. CUDA timings use events and are
    synchronized only by :meth:`flush`, avoiding a synchronization per phase.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self._host_seconds: dict[str, float] = defaultdict(float)
        self._cuda_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = (
            defaultdict(list)
        )

    @contextmanager
    def measure(self, name: str, *, cuda: bool = False) -> Iterator[None]:
        """Time the code in a ``with`` block under ``name``.

        Set ``cuda=True`` to also collect elapsed GPU-stream time when the
        configured device is CUDA.
        """
        host_start = perf_counter()
        use_cuda = cuda and self.device.type == "cuda"

        if use_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

        try:
            yield
        finally:
            self._host_seconds[name] += perf_counter() - host_start
            if use_cuda:
                end.record()
                self._cuda_events[name].append((start, end))

    def flush(self) -> dict[str, float]:
        """Return accumulated seconds and reset the timer.

        CUDA event elapsed times are reported in seconds after one device
        synchronization. Host timings do not synchronize the device.
        """
        if self._cuda_events:
            torch.cuda.synchronize(self.device)

        metrics = {
            f"timing/{name}_host_s": seconds
            for name, seconds in self._host_seconds.items()
        }
        for name, events in self._cuda_events.items():
            milliseconds = sum(start.elapsed_time(end) for start, end in events)
            metrics[f"timing/{name}_cuda_s"] = milliseconds / 1_000

        self._host_seconds.clear()
        self._cuda_events.clear()
        return metrics
