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
from websockets.asyncio.server import ServerConnection

SCHEMA_VERSION = 1


@dataclass
class TelemetryState:
    stream_enabled: bool = True
    mode: str = "slam"
    overlay_enabled: bool = True
    mode_locked: bool = False


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
    translation_scale: str = "unspecified",
    sequence: Optional[str] = None,
    total_frames: Optional[int] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    state = state or TelemetryState()
    return {
        "schema_version": SCHEMA_VERSION,
        "frame_index": frame_index,
        "timestamp": timestamp,
        "mode": state.mode,
        "mode_locked": state.mode_locked,
        "translation_scale": translation_scale,
        "sequence": sequence,
        "total_frames": total_frames,
        "run_id": run_id,
        "overlay_enabled": state.overlay_enabled,
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
        self._clients: set[ServerConnection] = set()
        self._queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ready = threading.Event()
        self._stopped = threading.Event()
        self._error: Optional[Exception] = None
        self._latest: Optional[Dict[str, Any]] = None
        self._queue = queue.Queue(maxsize=2)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=5):
            raise RuntimeError("Telemetry server startup timed out")
        if self._error is not None:
            raise RuntimeError(f"Telemetry server failed: {self._error}") from self._error

    def stop(self) -> None:
        self._stopped.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                raise RuntimeError("Telemetry server did not stop")

    def publish(self, payload: Dict[str, Any]) -> None:
        self._latest = payload
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(payload)

    def set_stream_enabled(self, enabled: bool) -> None:
        self.state.stream_enabled = enabled

    def set_mode(self, mode: str) -> None:
        if not self.state.mode_locked:self.state.mode = mode

    def set_overlay_enabled(self, enabled: bool) -> None:
        self.state.overlay_enabled = enabled

    def _run(self) -> None:
        try:
            asyncio.run(self._run_async())
        except Exception as exc:
            self._error = exc
            self._ready.set()

    async def _run_async(self) -> None:
        async with websockets.serve(self._handler, self.host, self.port) as server:
            self.port = server.sockets[0].getsockname()[1]
            self._ready.set()
            await self._broadcast_loop()

    async def _handler(self, websocket: ServerConnection) -> None:
        self._clients.add(websocket)
        try:
            if self._latest is not None:
                await websocket.send(json.dumps(self._snapshot()))
            async for message in websocket:
                self._handle_message(message)
                if self._latest is not None:
                    # Controls remain acknowledged while streaming is paused.
                    await websocket.send(json.dumps(self._snapshot()))
        finally:
            self._clients.discard(websocket)

    def _handle_message(self, message: str) -> None:
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            return
        if not isinstance(data, dict) or data.get("type") != "control":
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

    def _snapshot(self) -> Dict[str, Any]:
        return {**(self._latest or {}), "mode": self.state.mode, "stream_enabled": self.state.stream_enabled,
                "overlay_enabled": self.state.overlay_enabled}

    async def _broadcast_loop(self) -> None:
        while not self._stopped.is_set():
            try:
                payload = self._queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.01)
                continue
            if not self.state.stream_enabled:
                continue
            if not self._clients:
                continue
            message = json.dumps(payload)
            await asyncio.gather(
                *(client.send(message) for client in list(self._clients)),
                return_exceptions=True,
            )
        # Deliver the final pose before closing the server at end of sequence.
        if self._latest is not None and self._clients:
            await asyncio.gather(*(client.send(json.dumps(self._snapshot())) for client in list(self._clients)), return_exceptions=True)


def now() -> float:
    return time.time()
