"""Graceful stop on SIGUSR1 (SLURM pre-walltime warning) and SIGTERM
(preemption / scancel): the current question finishes, files are closed, and
the process exits with :data:`EXIT_PENDING` so the batch script can requeue."""

from __future__ import annotations

import logging
import signal

log = logging.getLogger("energy_llm.signals")

EXIT_DONE = 0
EXIT_PENDING = 75   # EX_TEMPFAIL: work remains (signal or retryable failures)


class GracefulStop:
    def __init__(self, install: bool = True) -> None:
        self.requested = False
        self.signal_name: str | None = None
        if install:
            for sig in (signal.SIGUSR1, signal.SIGTERM):
                try:
                    signal.signal(sig, self._handler)
                except ValueError:  # not the main thread
                    pass

    def _handler(self, signum, _frame) -> None:
        self.requested = True
        self.signal_name = signal.Signals(signum).name
        log.warning("received %s: stopping after the current question", self.signal_name)
