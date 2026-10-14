"""Separate loopback control handshake and telemetry readiness adapters."""
from __future__ import annotations

import asyncio
import json
import math
import threading
from typing import Any

import websockets

from native_application_stop_state_v4 import ApplicationStopState
from native_fifo_telemetry import NativeFifoTelemetryServer


class NativeApplicationControlServer:
    def __init__(self, host: str, port: int, state: ApplicationStopState):
        if host not in ("127.0.0.1", "localhost") or not 0 <= port <= 65535:
            raise ValueError("loopback bounded control endpoint required")
        self.host, self.port, self.state = host, port, state
        self.hello_delivered = threading.Event()
        self._ready = threading.Event()
        self._stopped = threading.Event()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._clients: set[Any] = set()
        self._error: BaseException | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("one control server start only")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if not self._ready.wait(5):
            raise TimeoutError("control server startup deadline")
        if self._error is not None:
            raise RuntimeError("control server failed") from self._error

    def _run(self) -> None:
        try:
            asyncio.run(self._run_async())
        except BaseException as exc:
            self._error = exc
            self._ready.set()

    async def _run_async(self) -> None:
        self._loop = asyncio.get_running_loop()
        async with websockets.serve(self._handler, self.host, self.port,
                                    max_size=1024, max_queue=4,
                                    origins=["http://127.0.0.1:4301"]) as server:
            self.port = server.sockets[0].getsockname()[1]
            self._ready.set()
            while not self._stopped.is_set():
                await asyncio.sleep(.01)

    async def _handler(self, websocket) -> None:
        self._clients.add(websocket)
        try:
            await websocket.send(json.dumps(self.state.hello(), allow_nan=False))
            self.hello_delivered.set()
            async for message in websocket:
                if not isinstance(message, str) or len(message.encode("utf8")) > 512:
                    response = self.state.snapshot(status="rejected_control")
                else:
                    response = self.state.receive(message.encode("utf8"))
                await websocket.send(json.dumps(response, allow_nan=False))
        finally:
            self._clients.discard(websocket)

    def publish_state(self) -> None:
        if self._loop is None or self._stopped.is_set():
            return
        payload = json.dumps(self.state.snapshot(), allow_nan=False)

        async def send() -> None:
            await asyncio.gather(*(client.send(payload) for client in list(self._clients)),
                                 return_exceptions=True)
        asyncio.run_coroutine_threadsafe(send(), self._loop)

    def stop(self) -> None:
        self._stopped.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                raise RuntimeError("control server did not stop")
        if self._error is not None:
            raise RuntimeError("control server failed") from self._error


class NativeApplicationTelemetryServer(NativeFifoTelemetryServer):
    """Expose real client readiness without adding a synthetic pose."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client_ready = threading.Event()

    async def _handler(self, websocket):
        self.client_ready.set()
        await super()._handler(websocket)

    def await_client(self, timeout_s: float) -> bool:
        if not math.isfinite(timeout_s) or timeout_s < 0:
            raise ValueError("finite telemetry readiness deadline required")
        return self.client_ready.wait(timeout_s)
