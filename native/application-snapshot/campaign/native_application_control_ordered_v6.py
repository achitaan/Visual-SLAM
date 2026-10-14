"""Ordered multi-client Stop publication; requires the frozen V4 runtime.

Component successor only. The retained launchers are not rewired here.
"""
import asyncio
import json

from websockets.exceptions import ConnectionClosed

from native_application_control_v4 import NativeApplicationControlServer as FrozenServer


class NativeApplicationControlServer(FrozenServer):
    def __init__(self, host, port, state):
        super().__init__(host, port, state)
        self._send_lock = asyncio.Lock()

    async def _deliver(self, *, actor=None, broadcast=True, rejected=False):
        # Snapshot at delivery, not on the publisher's thread. All state sends
        # share this lock so an older queued request cannot regress the UI.
        async with self._send_lock:
            current = self.state.snapshot(status="rejected_control" if rejected else None)
            payload = json.dumps(current, allow_nan=False)
            if broadcast:
                await asyncio.gather(*(peer.send(payload) for peer in list(self._clients)),
                                     return_exceptions=True)
            if actor is not None:
                try:
                    await actor.send(payload)
                except ConnectionClosed:
                    # The accepted Stop remains valid if its requester leaves;
                    # other subscribers have already received the snapshot.
                    pass

    def _queue_publication(self):
        return asyncio.run_coroutine_threadsafe(self._deliver(), self._loop)

    def publish_state(self):
        if self._loop is not None and not self._stopped.is_set():
            self._queue_publication()

    async def _handler(self, websocket):
        self._clients.add(websocket)
        try:
            async with self._send_lock:
                await websocket.send(json.dumps(self.state.hello(), allow_nan=False))
                self.hello_delivered.set()
            async for message in websocket:
                if not isinstance(message, str) or len(message.encode("utf8")) > 512:
                    rejected = True
                    broadcast = False
                else:
                    response = self.state.receive(message.encode("utf8"))
                    rejected = response.get("status") == "rejected_control"
                    broadcast = not rejected and response.get("phase") in (
                        "stopping", "cancelled", "completed")
                await self._deliver(actor=websocket, broadcast=broadcast, rejected=rejected)
        finally:
            self._clients.discard(websocket)
