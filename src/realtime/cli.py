"""Single-command CLI launcher for Real-Time Visual-SLAM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_SRC_DIR = str(Path(__file__).resolve().parents[1])
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from realtime.engine import RealtimeSlamEngine
from realtime.server import build_source_from_spec, create_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.realtime.cli",
        description="Launch Real-Time Visual-SLAM engine and FastAPI WebSocket streaming server.",
    )
    parser.add_argument(
        "--source",
        choices=("kitti", "video", "uvc", "realsense", "depthai"),
        default="kitti",
        help="Input frame source adapter (default: kitti)",
    )
    parser.add_argument("--kitti-root", type=str, default="KITTI_sequence_1", help="KITTI dataset root or sequence directory")
    parser.add_argument("--sequence", type=str, default="00", help="KITTI two-digit sequence ID (default: 00)")
    parser.add_argument("--mono", action="store_true", help="Run monocular mode instead of stereo")
    parser.add_argument("--video-left", type=str, default=None, help="Stereo video path (SBS) or left video path")
    parser.add_argument("--video-right", type=str, default=None, help="Optional right video path for dual-file stereo")
    parser.add_argument("--calib", type=str, default=None, help="Calibration file (.yaml/.json/KITTI calib.txt)")
    parser.add_argument("--layout", choices=("auto", "sbs", "dual", "mono"), default="auto")
    parser.add_argument("--camera-id", type=int, default=0, help="UVC camera index (left or SBS)")
    parser.add_argument("--right-camera-id", type=int, default=None, help="Optional right UVC camera index")
    parser.add_argument("--width", type=int, default=None, help="Requested camera capture width")
    parser.add_argument("--height", type=int, default=None, help="Requested camera capture height")
    parser.add_argument("--fps", type=float, default=30.0, help="Requested camera capture FPS")
    parser.add_argument("--rate", type=float, default=1.0, help="Playback rate multiplier (0 = unpaced)")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame cap for replay sources")
    parser.add_argument("--loop-playback", action="store_true", help="Loop dataset/video playback")
    drop_group = parser.add_mutually_exclusive_group()
    drop_group.add_argument("--drop-frames", dest="drop_frames", action="store_true", default=True, help="Drop stale frames to bound latency (default)")
    drop_group.add_argument("--no-drop-frames", dest="drop_frames", action="store_false", help="Bounded FIFO lossless replay")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto", help="Compute backend")
    parser.add_argument("--sync-ba", action="store_true", help="Run synchronous local BA instead of async worker")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="HTTP/WebSocket bind host")
    parser.add_argument("--port", type=int, default=8080, help="HTTP/WebSocket bind port")
    parser.add_argument("--no-start", action="store_true", help="Launch server paused until started via REST/UI")
    parser.add_argument("--headless", action="store_true", help="Run engine to completion without starting HTTP server")
    parser.add_argument("--save-trajectory", type=str, default=None, help="Optional trajectory output path on exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = {
        "kind": args.source,
        "data_root": args.kitti_root,
        "sequence": args.sequence,
        "stereo": not args.mono,
        "rate": args.rate,
        "drop_stale": args.drop_frames,
        "max_frames": args.max_frames,
        "loop": args.loop_playback,
        "video_left": args.video_left,
        "video_right": args.video_right,
        "calib": args.calib,
        "layout": args.layout,
        "camera_id": args.camera_id,
        "right_camera_id": args.right_camera_id,
        "width": args.width,
        "height": args.height,
        "fps": args.fps,
    }
    engine = RealtimeSlamEngine(
        source=None,
        drop_frames=args.drop_frames,
        device=args.device,
        async_local_ba=not args.sync_ba,
    )
    source = build_source_from_spec(spec, gpu_accelerator=engine.gpu)
    engine.switch_source(source, auto_start=not args.no_start)

    if args.headless:
        try:
            engine.wait_finished()
            telemetry = engine.get_telemetry()
            if args.save_trajectory:
                engine.save_trajectory(args.save_trajectory)
            print(json.dumps(telemetry, indent=2))
        finally:
            engine.stop(finish=True)
        return 0

    import uvicorn

    app = create_app(engine)
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        if args.save_trajectory:
            try:
                engine.save_trajectory(args.save_trajectory)
            except Exception:
                pass
        engine.stop(finish=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
