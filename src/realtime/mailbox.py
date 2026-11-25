"""Thread-safe FrameMailbox supporting non-blocking latest-frame drop and bounded FIFO replay."""

from __future__ import annotations

from collections import deque
from dataclasses import replace
import threading
import time
from typing import Literal

from .frame_source import StereoFrame


class FrameMailbox:
    """Producer-consumer mailbox between camera capture and SLAM tracking threads.

    Modes:
    - ``"latest"`` (default for live cameras): non-blocking overwrite of unconsumed
      frame slot so queue latency never grows unbounded; tracks ``dropped_frames``.
    - ``"fifo"`` (for lossless dataset replay): bounded FIFO queue with backpressure
      and zero dropped frames.
    """

    def __init__(
        self,
        mode: Literal["latest", "fifo"] = "latest",
        maxsize: int = 2,
        *,
        clock=time.perf_counter,
    ) -> None:
        if mode not in ("latest", "fifo"):
            raise ValueError(f"FrameMailbox mode must be 'latest' or 'fifo', got {mode!r}")
        if int(maxsize) < 1:
            raise ValueError("FrameMailbox maxsize must be >= 1")
        self._mode: Literal["latest", "fifo"] = mode
        self._maxsize = int(maxsize)
        self._clock = clock
        self._cond = threading.Condition()
        self._slot: tuple[StereoFrame, float] | None = None
        self._pending_drops_for_next: int = 0
        self._queue: deque[tuple[StereoFrame, float]] = deque()
        self._closed = False
        self._dropped_frames = 0
        self._received_frames = 0
        self._delivered_frames = 0
        self._last_queue_age_ms = 0.0
        self._max_queue_age_ms = 0.0

    @property
    def mode(self) -> Literal["latest", "fifo"]:
        return self._mode

    @property
    def maxsize(self) -> int:
        return self._maxsize

    @property
    def closed(self) -> bool:
        with self._cond:
            return self._closed

    @property
    def dropped_frames(self) -> int:
        with self._cond:
            return self._dropped_frames

    @property
    def received_frames(self) -> int:
        with self._cond:
            return self._received_frames

    @property
    def delivered_frames(self) -> int:
        with self._cond:
            return self._delivered_frames

    @property
    def queue_age_ms(self) -> float:
        with self._cond:
            return self._last_queue_age_ms

    @property
    def size(self) -> int:
        with self._cond:
            if self._mode == "latest":
                return 1 if self._slot is not None else 0
            return len(self._queue)

    def put(self, frame: StereoFrame, timeout: float | None = None) -> bool:
        """Enqueue a frame. Returns False only if closed or timed out in FIFO mode."""
        now = self._clock()
        with self._cond:
            if self._closed:
                return False
            if self._mode == "latest":
                if self._slot is not None:
                    self._dropped_frames += 1
                    self._pending_drops_for_next += 1
                self._slot = (frame, now)
                self._received_frames += 1
                self._cond.notify_all()
                return True

            # Bounded FIFO with backpressure
            deadline = None if timeout is None else (self._clock() + max(0.0, float(timeout)))
            while len(self._queue) >= self._maxsize and not self._closed:
                if deadline is not None:
                    remaining = deadline - self._clock()
                    if remaining <= 0.0:
                        return False
                    self._cond.wait(timeout=remaining)
                else:
                    self._cond.wait()
            if self._closed:
                return False
            self._queue.append((frame, now))
            self._received_frames += 1
            self._cond.notify_all()
            return True

    def get(self, timeout: float | None = None) -> StereoFrame | None:
        """Consume the next available frame, or None on timeout / closed+empty."""
        with self._cond:
            deadline = None if timeout is None else (self._clock() + max(0.0, float(timeout)))
            while True:
                if self._mode == "latest":
                    if self._slot is not None:
                        frame, enqueued_at = self._slot
                        drops = self._pending_drops_for_next
                        self._slot = None
                        self._pending_drops_for_next = 0
                        break
                else:
                    if self._queue:
                        frame, enqueued_at = self._queue.popleft()
                        drops = 0
                        self._cond.notify_all()
                        break
                if self._closed:
                    return None
                if deadline is not None:
                    remaining = deadline - self._clock()
                    if remaining <= 0.0:
                        return None
                    self._cond.wait(timeout=remaining)
                else:
                    self._cond.wait()

            age_ms = max(0.0, (self._clock() - enqueued_at) * 1000.0)
            self._last_queue_age_ms = age_ms
            if age_ms > self._max_queue_age_ms:
                self._max_queue_age_ms = age_ms
            self._delivered_frames += 1

        if drops > 0:
            return replace(frame, dropped_before=frame.dropped_before + drops)
        return frame

    def clear(self) -> int:
        """Discard any currently queued frames and return count discarded."""
        with self._cond:
            if self._mode == "latest":
                count = 1 if self._slot is not None else 0
                self._slot = None
                self._pending_drops_for_next = 0
            else:
                count = len(self._queue)
                self._queue.clear()
                self._cond.notify_all()
            return count

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def reopen(self) -> None:
        with self._cond:
            self._closed = False
            self._cond.notify_all()

    def stats(self) -> dict[str, float | int | str]:
        with self._cond:
            return {
                "mode": self._mode,
                "size": (1 if self._slot is not None else 0) if self._mode == "latest" else len(self._queue),
                "received_frames": self._received_frames,
                "delivered_frames": self._delivered_frames,
                "dropped_frames": self._dropped_frames,
                "queue_age_ms": round(self._last_queue_age_ms, 3),
                "max_queue_age_ms": round(self._max_queue_age_ms, 3),
            }
