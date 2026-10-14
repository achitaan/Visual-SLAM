"""Qualified successor candidate: deferred observation is not writer failure.

The frozen observer still performs identity, truncation and committed-prefix
checks. Only retry admission differs. This module never signals a writer.
Blocking I/O requires a separately owned process deadline watchdog.
"""
import errno
import math
import time

from native_observation import CommittedPrefixObserver as FrozenPrefixObserver


class ObservationDeferred(Exception):
    """Retry admission window expired; caller must re-poll the same writer."""


class ObservationBudgetExhausted(Exception):
    """The shared finite ENODATA allowance has been consumed."""


class DeliveryBudget:
    def __init__(self, *, window_s=.5, retry_s=.02, max_errors=8,
                 clock=time.monotonic, sleep=time.sleep):
        if not math.isfinite(window_s) or window_s <= 0:
            raise ValueError('finite positive observation window required')
        if not math.isfinite(retry_s) or retry_s <= 0:
            raise ValueError('finite positive retry interval required')
        if type(max_errors) is not int or max_errors < 0:
            raise ValueError('nonnegative integer ENODATA allowance required')
        self.window_s, self.retry_s, self.max_errors = window_s, retry_s, max_errors
        self.clock, self.sleep = clock, sleep
        self.transient_errors = 0
        self.events = []
        self.delayed_returns = 0
        self.max_read_s = 0.0
        self.deferrals = 0
        self._last_clock = -math.inf

    def _now(self):
        now = self.clock()
        if not math.isfinite(now) or now < self._last_clock:
            raise ValueError('observation clock must be finite and monotonic')
        self._last_clock = now
        return now

    def call(self, operation, *, deadline, poll=lambda: None):
        if not math.isfinite(deadline):
            raise ValueError('finite absolute arm deadline required')
        began = self._now()
        stop = min(deadline, began + self.window_s)
        while True:
            now = self._now()
            if now >= deadline:
                raise TimeoutError('absolute native arm deadline exceeded')
            if self.transient_errors > self.max_errors:
                raise ObservationBudgetExhausted('shared ENODATA allowance exhausted')
            if now >= stop:
                self.deferrals += 1
                raise ObservationDeferred('retry admission deferred; re-poll same writer')
            read_start = now
            try:
                result = operation()
            except OSError as error:
                if error.errno != getattr(errno, 'ENODATA', 61):
                    raise
                self.transient_errors += 1
                now = self._now()
                if len(self.events) <= self.max_errors:
                    self.events.append(dict(monotonic_s=now, errno=error.errno,
                                            process_exit_code=poll(), attempt=self.transient_errors))
                if now >= deadline:
                    raise TimeoutError('absolute native arm deadline exceeded') from error
                if self.transient_errors > self.max_errors:
                    raise ObservationBudgetExhausted('shared ENODATA allowance exhausted') from error
                if now >= stop:
                    self.deferrals += 1
                    raise ObservationDeferred('retry admission deferred; re-poll same writer') from error
                self.sleep(min(self.retry_s, stop - now))
                continue
            ended = self._now()
            self.max_read_s = max(self.max_read_s, ended - read_start)
            if ended >= deadline:
                raise TimeoutError('absolute native arm deadline exceeded')
            if ended >= stop:
                self.delayed_returns += 1
            # Successful bytes still undergo the original identity/history checks.
            # A slow successful read is recorded, never described as low latency.
            return result


class CommittedPrefixObserver(FrozenPrefixObserver):
    def __init__(self, *, read_snapshot=None, **budget_options):
        # Reuse frozen snapshot and commitment code without copying/weakening it.
        super().__init__(**({} if read_snapshot is None else {'read_snapshot': read_snapshot}))
        self.budget = DeliveryBudget(**budget_options)

    def statistics(self):
        result = super().statistics()
        result.update(delayed_returns=self.budget.delayed_returns,
                      max_read_s=self.budget.max_read_s, deferrals=self.budget.deferrals,
                      semantics='retry_admission_window_with_absolute_arm_deadline')
        return result
