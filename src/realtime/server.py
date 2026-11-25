"""FastAPI + Uvicorn server exposing REST controls and coalesced WebSockets for live SLAM."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .engine import RealtimeSlamEngine
from .frame_source import (
    DepthAIStereoSource,
    FrameSource,
    OpenCVStereoCameraSource,
    PacedKittiSource,
    RealSenseStereoSource,
    StereoVideoFileSource,
)


class RateRequest(BaseModel):
    rate: float


class DropFramesRequest(BaseModel):
    drop_frames: bool


class SaveTrajectoryRequest(BaseModel):
    path: str


class SourceRequest(BaseModel):
    kind: str
    data_root: str | None = None
    sequence: str | None = "00"
    stereo: bool = True
    rate: float = 1.0
    drop_stale: bool = True
    max_frames: int | None = None
    loop: bool = False
    video_left: str | None = None
    video_right: str | None = None
    calib: str | None = None
    layout: str = "auto"
    camera_id: int = 0
    right_camera_id: int | None = None
    width: int | None = None
    height: int | None = None
    fps: float = 30.0
    auto_start: bool = True


def build_source_from_spec(spec: dict[str, Any], *, gpu_accelerator: Any = None) -> FrameSource:
    """Construct a FrameSource adapter from a dictionary / REST SourceRequest."""
    kind = str(spec.get("kind", "")).strip().lower()
    if kind == "kitti":
        data_root = spec.get("data_root")
        if not data_root:
            raise ValueError("KITTI source requires 'data_root'")
        return PacedKittiSource(
            data_root=data_root,
            sequence=spec.get("sequence", "00"),
            stereo=bool(spec.get("stereo", True)),
            rate=float(spec.get("rate", 1.0)),
            drop_stale=bool(spec.get("drop_stale", True)),
            max_frames=spec.get("max_frames"),
            loop=bool(spec.get("loop", False)),
        )
    if kind == "video":
        video_left = spec.get("video_left")
        if not video_left:
            raise ValueError("Video source requires 'video_left'")
        return StereoVideoFileSource(
            left_path=video_left,
            right_path=spec.get("video_right"),
            calibration=spec.get("calib"),
            layout=str(spec.get("layout", "auto")),
            rate=float(spec.get("rate", 1.0)),
            drop_stale=bool(spec.get("drop_stale", False)),
            max_frames=spec.get("max_frames"),
            loop=bool(spec.get("loop", False)),
            gpu_accelerator=gpu_accelerator,
        )
    if kind == "uvc":
        return OpenCVStereoCameraSource(
            device_id=int(spec.get("camera_id", 0)),
            right_device_id=spec.get("right_camera_id"),
            calibration=spec.get("calib"),
            layout=str(spec.get("layout", "auto")),
            width=spec.get("width"),
            height=spec.get("height"),
            target_fps=float(spec.get("fps", 30.0)),
            gpu_accelerator=gpu_accelerator,
        )
    if kind == "realsense":
        return RealSenseStereoSource(
            width=int(spec.get("width") or 848),
            height=int(spec.get("height") or 480),
            target_fps=int(spec.get("fps") or 30),
        )
    if kind == "depthai":
        return DepthAIStereoSource(
            width=int(spec.get("width") or 640),
            height=int(spec.get("height") or 400),
            target_fps=float(spec.get("fps") or 30.0),
        )
    raise ValueError(f"Unsupported source kind {kind!r}; expected kitti, video, uvc, realsense, or depthai")


def create_app(
    engine: RealtimeSlamEngine,
    *,
    telemetry_hz: float = 25.0,
    map_hz: float = 4.0,
    video_hz: float = 18.0,
) -> FastAPI:
    """Create the FastAPI application wired to a ``RealtimeSlamEngine`` instance."""
    app = FastAPI(title="Visual-SLAM Real-Time API", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.engine = engine

    @app.get("/api/status")
    async def api_status() -> dict[str, Any]:
        return engine.get_telemetry()

    @app.get("/api/snapshot")
    async def api_snapshot() -> dict[str, Any]:
        return engine.get_map_snapshot()

    @app.post("/api/control/start")
    async def api_control_start() -> dict[str, Any]:
        try:
            engine.start()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "status": engine.get_telemetry()}

    @app.post("/api/control/pause")
    async def api_control_pause() -> dict[str, Any]:
        engine.pause()
        return {"ok": True, "status": engine.get_telemetry()}

    @app.post("/api/control/resume")
    async def api_control_resume() -> dict[str, Any]:
        engine.resume()
        return {"ok": True, "status": engine.get_telemetry()}

    @app.post("/api/control/reset")
    async def api_control_reset() -> dict[str, Any]:
        try:
            engine.reset_map()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "status": engine.get_telemetry()}

    @app.post("/api/control/rate")
    async def api_control_rate(req: RateRequest) -> dict[str, Any]:
        engine.set_playback_rate(req.rate)
        return {"ok": True, "playback_rate": req.rate}

    @app.post("/api/control/drop_frames")
    async def api_control_drop_frames(req: DropFramesRequest) -> dict[str, Any]:
        engine.set_drop_frames(req.drop_frames)
        return {"ok": True, "drop_frames": req.drop_frames}

    @app.post("/api/control/save_trajectory")
    async def api_control_save_trajectory(req: SaveTrajectoryRequest) -> dict[str, Any]:
        try:
            saved = engine.save_trajectory(Path(req.path))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "path": str(saved)}

    @app.post("/api/source")
    async def api_switch_source(req: SourceRequest) -> dict[str, Any]:
        try:
            source = build_source_from_spec(req.model_dump(), gpu_accelerator=engine.gpu)
            engine.switch_source(source, auto_start=req.auto_start)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "status": engine.get_telemetry()}

    @app.websocket("/ws/telemetry")
    async def ws_telemetry(websocket: WebSocket) -> None:
        await websocket.accept()
        interval = 1.0 / max(float(telemetry_hz), 1.0)
        try:
            while True:
                payload = engine.get_telemetry()
                await websocket.send_json(payload)
                await asyncio.sleep(interval)
        except (WebSocketDisconnect, asyncio.CancelledError):
            return
        except Exception:
            return

    @app.websocket("/ws/map")
    async def ws_map(websocket: WebSocket) -> None:
        await websocket.accept()
        interval = 1.0 / max(float(map_hz), 0.5)
        try:
            snap = engine.get_map_snapshot()
            await websocket.send_json(snap)
            since_kf = int(snap.get("max_kf_id", -1))
            since_pose_rev = int(snap.get("pose_revision", 0))
            since_loops = len(snap.get("loop_edges") or ())
            since_traj = len(snap.get("trajectory") or ())
            while True:
                await asyncio.sleep(interval)
                delta, since_kf, since_pose_rev, since_loops, since_traj = engine.get_map_delta(
                    since_kf_id=since_kf,
                    since_pose_revision=since_pose_rev,
                    since_loop_count=since_loops,
                    since_traj_len=since_traj,
                )
                if delta is not None:
                    await websocket.send_json(delta)
        except (WebSocketDisconnect, asyncio.CancelledError):
            return
        except Exception:
            return

    @app.websocket("/ws/video")
    async def ws_video(websocket: WebSocket) -> None:
        await websocket.accept()
        interval = 1.0 / max(float(video_hz), 1.0)
        last_seq = -1
        try:
            while True:
                seq, jpeg = engine.get_latest_video_jpeg(last_seq=last_seq)
                if jpeg is not None:
                    last_seq = seq
                    await websocket.send_bytes(jpeg)
                await asyncio.sleep(interval)
        except (WebSocketDisconnect, asyncio.CancelledError):
            return
        except Exception:
            return

    return app
