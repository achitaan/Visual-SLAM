import asyncio
import base64
import json
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import cv2 as cv
import numpy as np
import websockets

SCHEMA_VERSION = 1


@dataclass
class TelemetryState:
    stream_enabled: bool = True
    mode: str = "slam"
    overlay_enabled: bool = True


def encode_image(image: np.ndarray, encoding: str = "jpg") -> Optional[Dict[str, Any]]:
    if image is None:
        return None
    if encoding not in ("jpg", "png"):
        encoding = "jpg"
    ext = ".jpg" if encoding == "jpg" else ".png"
    success, buffer = cv.imencode(ext, image)
    if not success:
        return None
    return {
        "encoding": encoding,
        "data_base64": base64.b64encode(buffer).decode("ascii"),
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
    }


def make_frame_message(
    *,
    frame_index: int,
    timestamp: float,
    pose_T_wc: np.ndarray,
    tracking: Dict[str, Any],
    map_state: Dict[str, int],
    pose_graph: Optional[Dict[str, Any]] = None,
    expected_pose_T_wc: Optional[np.ndarray] = None,
    map_points: Optional[Iterable[Iterable[float]]] = None,
    image_payload: Optional[Dict[str, Any]] = None,
    features: Optional[Iterable[Dict[str, Any]]] = None,
    fps: Optional[float] = None,
    velocity: Optional[Iterable[float]] = None,
    events: Optional[Iterable[Dict[str, Any]]] = None,
    state: Optional[TelemetryState] = None,
) -> Dict[str, Any]:
    state = state or TelemetryState()
    return {
        "schema_version": SCHEMA_VERSION,
        "frame_index": frame_index,
        "timestamp": timestamp,
        "mode": state.mode,
        "stream_enabled": state.stream_enabled,
        "pose_T_wc": pose_T_wc.tolist(),
        "velocity": list(velocity) if velocity is not None else None,
        "fps": fps,
        "tracking": tracking,
        "map": map_state,
        "map_points": list(map_points) if map_points is not None else None,
        "pose_graph": pose_graph,
        "expected_pose_T_wc": expected_pose_T_wc.tolist() if expected_pose_T_wc is not None else None,
        "image": image_payload,
        "features": list(features) if features is not None else None,
        "events": list(events) if events is not None else [],
    }


class TelemetryServer:
    def __init__(self, host: str, port: int, state: Optional[TelemetryState] = None) -> None:
        self.host = host
        self.port = port
        self.state = state or TelemetryState()
        self._clients: set[websockets.WebSocketServerProtocol] = set()
        self._queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def publish(self, payload: Dict[str, Any]) -> None:
        self._queue.put(payload)

    def set_stream_enabled(self, enabled: bool) -> None:
        self.state.stream_enabled = enabled

    def set_mode(self, mode: str) -> None:
        self.state.mode = mode

    def set_overlay_enabled(self, enabled: bool) -> None:
        self.state.overlay_enabled = enabled

    def _run(self) -> None:
        asyncio.run(self._run_async())

    async def _run_async(self) -> None:
        async with websockets.serve(self._handler, self.host, self.port):
            await self._broadcast_loop()

    async def _handler(self, websocket: websockets.WebSocketServerProtocol) -> None:
        self._clients.add(websocket)
        try:
            async for message in websocket:
                self._handle_message(message)
        finally:
            self._clients.discard(websocket)

    def _handle_message(self, message: str) -> None:
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            return
        if data.get("type") != "control":
            return
        action = data.get("action")
        if action == "start":
            self.set_stream_enabled(True)
        elif action == "stop":
            self.set_stream_enabled(False)
        elif action == "set_mode":
            mode = data.get("mode")
            if mode in ("vo", "slam"):
                self.set_mode(mode)
        elif action == "toggle_overlay":
            enabled = data.get("enabled")
            if isinstance(enabled, bool):
                self.set_overlay_enabled(enabled)

    async def _broadcast_loop(self) -> None:
        while True:
            payload = await asyncio.to_thread(self._queue.get)
            if not self.state.stream_enabled:
                continue
            if not self._clients:
                continue
            message = json.dumps(payload)
            await asyncio.gather(
                *(client.send(message) for client in list(self._clients)),
                return_exceptions=True,
            )


def now() -> float:
    return time.time()
