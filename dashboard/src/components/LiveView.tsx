"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Icon } from "@/components/Icon";
import { SwitchSourceSpec, useRealtimeSlam } from "@/src/hooks/useRealtimeSlam";

const DEFAULT_API_URL =
  process.env.NEXT_PUBLIC_REALTIME_API_URL ?? "http://127.0.0.1:8080";

const INITIAL_VIEW = { yaw: 0.55, pitch: -0.62, zoom: 1.0 };

function stateTone(state: string): string {
  if (state === "TRACKING") return "teal";
  if (state === "LOST" || state === "RELOCALIZING") return "danger";
  return "amber";
}

function transformPoint(
  pose: number[][],
  lx: number,
  ly: number,
  lz: number
): [number, number, number] {
  return [
    pose[0][0] * lx + pose[0][1] * ly + pose[0][2] * lz + pose[0][3],
    pose[1][0] * lx + pose[1][1] * ly + pose[1][2] * lz + pose[1][3],
    pose[2][0] * lx + pose[2][1] * ly + pose[2][2] * lz + pose[2][3],
  ];
}

export function LiveView() {
  const [apiUrl, setApiUrl] = useState(DEFAULT_API_URL);
  const [sourceKind, setSourceKind] =
    useState<SwitchSourceSpec["kind"]>("kitti");
  const [sourcePath, setSourcePath] = useState("KITTI_sequence_1");
  const [sequenceId, setSequenceId] = useState("00");
  const [cameraId, setCameraId] = useState(0);
  const [rateValue, setRateValue] = useState(1.0);
  const [hasVideo, setHasVideo] = useState(false);
  const [view, setView] = useState(INITIAL_VIEW);

  const videoCanvasRef = useRef<HTMLCanvasElement>(null);
  const mapCanvasRef = useRef<HTMLCanvasElement>(null);
  const dragRef = useRef<{ x: number; y: number } | null>(null);
  const paintingVideoRef = useRef(false);

  const handleVideoFrame = useCallback((blob: Blob) => {
    const canvas = videoCanvasRef.current;
    if (!canvas || paintingVideoRef.current) return;
    paintingVideoRef.current = true;
    const drawBitmap = (img: ImageBitmap | HTMLImageElement, w: number, h: number) => {
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }
      const ctx = canvas.getContext("2d");
      if (ctx) ctx.drawImage(img, 0, 0, w, h);
      setHasVideo(true);
      paintingVideoRef.current = false;
    };
    if (typeof createImageBitmap === "function") {
      createImageBitmap(blob)
        .then((bmp) => {
          drawBitmap(bmp, bmp.width, bmp.height);
          bmp.close();
        })
        .catch(() => {
          paintingVideoRef.current = false;
        });
    } else {
      const url = URL.createObjectURL(blob);
      const img = new Image();
      img.onload = () => {
        drawBitmap(img, img.naturalWidth, img.naturalHeight);
        URL.revokeObjectURL(url);
      };
      img.onerror = () => {
        URL.revokeObjectURL(url);
        paintingVideoRef.current = false;
      };
      img.src = url;
    }
  }, []);

  const {
    connected,
    telemetry,
    mapState,
    mapVersion,
    controlError,
    start,
    pause,
    resume,
    resetMap,
    setRate,
    setDropFrames,
    switchSource,
  } = useRealtimeSlam(apiUrl, handleVideoFrame);

  // Render live 3D map: keyframe-anchored Float32 point cloud, trajectory trail,
  // 3D camera frustum, and flashing loop-closure arcs.
  useEffect(() => {
    const canvas = mapCanvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;

    const w = 780;
    const h = 360;
    canvas.width = w;
    canvas.height = h;
    ctx.fillStyle = "#0f1722";
    ctx.fillRect(0, 0, w, h);

    const cosY = Math.cos(view.yaw);
    const sinY = Math.sin(view.yaw);
    const cosP = Math.cos(view.pitch);
    const sinP = Math.sin(view.pitch);

    const rotate = (x: number, y: number, z: number): [number, number, number] => {
      const rx = cosY * x + sinY * z;
      const rz = -sinY * x + cosY * z;
      const ry = cosP * y - sinP * rz;
      const depth = sinP * y + cosP * rz;
      return [rx, ry, depth];
    };

    const traj = mapState.trajectory;
    const pose = telemetry?.pose_T_wc;
    const kfs = Array.from(mapState.keyframes.values());

    // Gather bounds across trajectory + sampled world points
    let minX = 0,
      maxX = 0,
      minY = 0,
      maxY = 0;
    let hasBounds = false;
    const expand = (x: number, y: number, z: number) => {
      const [rx, ry] = rotate(x, y, z);
      if (!hasBounds) {
        minX = maxX = rx;
        minY = maxY = ry;
        hasBounds = true;
      } else {
        if (rx < minX) minX = rx;
        if (rx > maxX) maxX = rx;
        if (ry < minY) minY = ry;
        if (ry > maxY) maxY = ry;
      }
    };

    for (const p of traj) expand(p[0], p[1], p[2]);
    if (pose) expand(pose[0][3], pose[1][3], pose[2][3]);

    const worldPoints: [number, number, number][] = [];
    for (const kf of kfs) {
      const lp = kf.localPoints;
      const stride = lp.length > 900 ? 6 : 3;
      for (let i = 0; i + 2 < lp.length; i += stride) {
        const wp = transformPoint(kf.pose, lp[i], lp[i + 1], lp[i + 2]);
        worldPoints.push(wp);
        if (worldPoints.length < 4000) expand(wp[0], wp[1], wp[2]);
      }
    }

    const spanX = Math.max(4, maxX - minX);
    const spanY = Math.max(4, maxY - minY);
    const scale = Math.min((w - 110) / spanX, (h - 70) / spanY) * view.zoom;
    const cx = (minX + maxX) * 0.5;
    const cy = (minY + maxY) * 0.5;

    const project = (x: number, y: number, z: number): [number, number] => {
      const [rx, ry] = rotate(x, y, z);
      return [w * 0.5 + (rx - cx) * scale, h * 0.5 + (ry - cy) * scale];
    };

    // Subtle spatial grid
    ctx.strokeStyle = "#1c2a3a";
    ctx.lineWidth = 1;
    for (let i = 1; i < 8; i++) {
      ctx.beginPath();
      ctx.moveTo((i * w) / 8, 0);
      ctx.lineTo((i * w) / 8, h);
      ctx.stroke();
    }
    for (let i = 1; i < 6; i++) {
      ctx.beginPath();
      ctx.moveTo(0, (i * h) / 6);
      ctx.lineTo(w, (i * h) / 6);
      ctx.stroke();
    }

    // 1. Keyframe-anchored 3D point cloud
    ctx.fillStyle = "rgba(130, 198, 232, 0.55)";
    for (let i = 0; i < worldPoints.length; i++) {
      const p = worldPoints[i];
      const [sx, sy] = project(p[0], p[1], p[2]);
      if (sx >= 0 && sx < w && sy >= 0 && sy < h) {
        ctx.fillRect(sx, sy, 2, 2);
      }
    }

    // 2. Trajectory trail
    if (traj.length > 1) {
      ctx.strokeStyle = "#5b8aff";
      ctx.lineWidth = 2.2;
      ctx.beginPath();
      traj.forEach((p, idx) => {
        const [sx, sy] = project(p[0], p[1], p[2]);
        if (idx === 0) ctx.moveTo(sx, sy);
        else ctx.lineTo(sx, sy);
      });
      ctx.stroke();
    }

    // 3. Flashing loop-closure arcs
    const now = Date.now();
    for (const edge of mapState.loopEdges) {
      const kfA = mapState.keyframes.get(edge.from_kf);
      const kfB = mapState.keyframes.get(edge.to_kf);
      if (!kfA || !kfB) continue;
      const [ax, ay] = project(kfA.pose[0][3], kfA.pose[1][3], kfA.pose[2][3]);
      const [bx, by] = project(kfB.pose[0][3], kfB.pose[1][3], kfB.pose[2][3]);
      const isFlashing = edge.flashUntil > now;
      ctx.save();
      ctx.strokeStyle = isFlashing ? "#2ee6b8" : "rgba(245, 180, 72, 0.8)";
      ctx.lineWidth = isFlashing ? 3.2 : 1.6;
      ctx.setLineDash(isFlashing ? [] : [5, 4]);
      const mx = (ax + bx) * 0.5;
      const my = (ay + by) * 0.5 - 26;
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.quadraticCurveTo(mx, my, bx, by);
      ctx.stroke();
      ctx.restore();
    }

    // 4. Live 3D camera frustum
    if (pose) {
      const fSize = Math.max(0.5, Math.min(2.5, Math.max(spanX, spanY) * 0.04));
      const o = transformPoint(pose, 0, 0, 0);
      const tl = transformPoint(pose, -fSize * 0.6, -fSize * 0.4, fSize);
      const tr = transformPoint(pose, fSize * 0.6, -fSize * 0.4, fSize);
      const br = transformPoint(pose, fSize * 0.6, fSize * 0.4, fSize);
      const bl = transformPoint(pose, -fSize * 0.6, fSize * 0.4, fSize);

      const pO = project(o[0], o[1], o[2]);
      const corners = [tl, tr, br, bl].map((c) => project(c[0], c[1], c[2]));

      ctx.strokeStyle =
        telemetry?.tracking_state === "TRACKING" ? "#26d9b0" : "#f06a6a";
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (const c of corners) {
        ctx.moveTo(pO[0], pO[1]);
        ctx.lineTo(c[0], c[1]);
      }
      ctx.moveTo(corners[0][0], corners[0][1]);
      ctx.lineTo(corners[1][0], corners[1][1]);
      ctx.lineTo(corners[2][0], corners[2][1]);
      ctx.lineTo(corners[3][0], corners[3][1]);
      ctx.closePath();
      ctx.stroke();

      ctx.fillStyle = "#26d9b0";
      ctx.beginPath();
      ctx.arc(pO[0], pO[1], 4, 0, Math.PI * 2);
      ctx.fill();
    } else if (traj.length === 0) {
      ctx.fillStyle = "#7e91a6";
      ctx.font = "12px Segoe UI, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(
        "Connect to Real-Time SLAM server (python -m src.realtime.cli --port 8080)",
        w / 2,
        h / 2
      );
    }

    // Orientation triad
    [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ].forEach((axis, i) => {
      const [rx, ry] = rotate(axis[0], axis[1], axis[2]);
      const colors = ["#e06c75", "#26d9b0", "#5b8aff"];
      ctx.strokeStyle = colors[i];
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      ctx.moveTo(w - 46, 44);
      ctx.lineTo(w - 46 + rx * 24, 44 + ry * 24);
      ctx.stroke();
      ctx.fillStyle = colors[i];
      ctx.font = "10px Segoe UI, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("XYZ"[i], w - 46 + rx * 33, 48 + ry * 33);
    });
  }, [telemetry, mapState, mapVersion, view]);

  const handleApplySource = () => {
    const spec: SwitchSourceSpec = {
      kind: sourceKind,
      rate: rateValue,
      drop_stale: telemetry?.drop_frames ?? true,
      auto_start: true,
    };
    if (sourceKind === "kitti") {
      spec.data_root = sourcePath;
      spec.sequence = sequenceId;
    } else if (sourceKind === "video") {
      spec.video_left = sourcePath;
    } else if (sourceKind === "uvc") {
      spec.camera_id = cameraId;
    }
    void switchSource(spec);
  };

  const trackState = telemetry?.tracking_state ?? "INIT";
  const dropEnabled = telemetry?.drop_frames ?? true;

  return (
    <div className="realtimeContainer">
      <div className="sessionBar realtimeControls">
        <div className="sessionIdentity">
          <span className={`pill ${stateTone(trackState)}`}>{trackState}</span>
          <strong>{telemetry?.source ?? "Disconnected"}</strong>
          <span className="divider" />
          <span className="pill">{telemetry?.backend_device ?? "cpu"}</span>
          <span className={`connectionBadge ${connected ? "connected" : ""}`}>
            <span className={`statusDot ${connected ? "live" : ""}`} />
            {connected ? "WS Live" : "Offline"}
          </span>
        </div>

        <div className="toolbar">
          <button
            className="button primary"
            onClick={() =>
              telemetry?.running && !telemetry.paused
                ? void pause()
                : telemetry?.paused
                ? void resume()
                : void start()
            }
            disabled={!connected}
          >
            <Icon
              name={telemetry?.running && !telemetry.paused ? "pause" : "play"}
              size={14}
            />
            {telemetry?.running && !telemetry.paused
              ? "Pause"
              : telemetry?.paused
              ? "Resume"
              : "Start"}
          </button>
          <button
            className="button"
            onClick={() => void resetMap()}
            disabled={!connected}
          >
            <Icon name="reset" size={14} />
            Reset
          </button>
          <select
            aria-label="Playback rate"
            className="button"
            value={rateValue}
            onChange={(e) => {
              const r = Number(e.target.value);
              setRateValue(r);
              void setRate(r);
            }}
          >
            <option value={0.5}>0.5× rate</option>
            <option value={1.0}>1.0× real-time</option>
            <option value={2.0}>2.0× fast</option>
            <option value={0}>Max speed</option>
          </select>
          <button
            className={`button toggle ${dropEnabled ? "active" : ""}`}
            onClick={() => void setDropFrames(!dropEnabled)}
            disabled={!connected}
            title="Toggle real-time stale frame dropping vs FIFO lossless replay"
          >
            {dropEnabled ? "Drop Stale: ON" : "FIFO Lossless"}
          </button>
        </div>
      </div>

      <div className="sessionBar sourceSwitcher">
        <div className="toolbar">
          <label className="inlineField">
            <span>Server</span>
            <input
              type="text"
              value={apiUrl}
              onChange={(e) => setApiUrl(e.target.value)}
              className="textInput"
              aria-label="Real-time API base URL"
            />
          </label>
          <label className="inlineField">
            <span>Source</span>
            <select
              value={sourceKind}
              onChange={(e) =>
                setSourceKind(e.target.value as SwitchSourceSpec["kind"])
              }
              className="button"
              aria-label="Frame source type"
            >
              <option value="kitti">KITTI Dataset</option>
              <option value="video">Stereo Video</option>
              <option value="uvc">UVC Stereo Camera</option>
              <option value="realsense">Intel RealSense</option>
              <option value="depthai">OAK-D DepthAI</option>
            </select>
          </label>
          {(sourceKind === "kitti" || sourceKind === "video") && (
            <input
              type="text"
              value={sourcePath}
              onChange={(e) => setSourcePath(e.target.value)}
              placeholder={
                sourceKind === "kitti" ? "KITTI root path" : "Path to .mp4/.mkv"
              }
              className="textInput"
              aria-label="Source path"
            />
          )}
          {sourceKind === "kitti" && (
            <input
              type="text"
              value={sequenceId}
              onChange={(e) => setSequenceId(e.target.value)}
              placeholder="00"
              className="textInput shortInput"
              aria-label="KITTI sequence"
            />
          )}
          {sourceKind === "uvc" && (
            <input
              type="number"
              value={cameraId}
              onChange={(e) => setCameraId(Number(e.target.value))}
              className="textInput shortInput"
              aria-label="Camera device index"
            />
          )}
          <button
            className="button"
            onClick={handleApplySource}
            disabled={!connected}
          >
            Switch Source
          </button>
        </div>
        {controlError && <span className="pill danger">{controlError}</span>}
      </div>

      <section className="metricStrip">
        <div className="statCard positive">
          <div className="statLabel">Tracking / Capture FPS</div>
          <div className="statValue">
            {telemetry ? telemetry.tracking_fps.toFixed(1) : "—"}
            <span>/ {telemetry ? telemetry.capture_fps.toFixed(1) : "—"}</span>
          </div>
          <div className="statNote">
            Frame #{telemetry?.frame_id ?? 0} ·{" "}
            {telemetry?.processed_frames ?? 0} processed
          </div>
        </div>

        <div
          className={`statCard ${
            (telemetry?.track_p95_ms ?? 0) > 35 ? "warning" : "positive"
          }`}
        >
          <div className="statLabel">Track Latency (p50 / p95)</div>
          <div className="statValue">
            {telemetry ? telemetry.track_p50_ms.toFixed(1) : "—"}
            <span>
              / {telemetry ? telemetry.track_p95_ms.toFixed(1) : "—"} ms
            </span>
          </div>
          <div className="statNote">
            Async BA p50: {telemetry ? telemetry.ba_p50_ms.toFixed(1) : "—"} ms
          </div>
        </div>

        <div className="statCard">
          <div className="statLabel">Inliers &amp; Landmarks</div>
          <div className="statValue">
            {telemetry?.inliers ?? 0}
            <span>/ {telemetry?.active_landmarks ?? 0} pts</span>
          </div>
          <div className="statNote">
            Reproj:{" "}
            {telemetry?.reprojection_error != null
              ? `${telemetry.reprojection_error.toFixed(2)} px`
              : "—"}
          </div>
        </div>

        <div className="statCard">
          <div className="statLabel">Keyframes &amp; Loops</div>
          <div className="statValue">
            {telemetry?.keyframes ?? 0}
            <span>· {telemetry?.verified_loops_applied ?? 0} loops</span>
          </div>
          <div className="statNote">
            Loop v2a p95:{" "}
            {telemetry ? telemetry.loop_v2a_p95_ms.toFixed(0) : "0"} ms
          </div>
        </div>

        <div className="statCard">
          <div className="statLabel">Mailbox Health</div>
          <div className="statValue">
            {telemetry?.dropped_frames ?? 0}
            <span>dropped</span>
          </div>
          <div className="statNote">
            Queue age: {telemetry ? telemetry.queue_age_ms.toFixed(1) : "0.0"}{" "}
            ms
          </div>
        </div>

        <div className="statCard">
          <div className="statLabel">Compute Backend</div>
          <div className="statValue" style={{ fontSize: 16 }}>
            {telemetry?.backend_device ?? "offline"}
          </div>
          <div className="statNote">
            Pose rev #{telemetry?.pose_revision ?? 0}
          </div>
        </div>
      </section>

      <div className="visualGrid">
        <section className="panel">
          <div className="panelHeading">
            <div>
              <span className="eyebrow">ANNOTATED STEREO FEED</span>
              <h2>Live camera stream (/ws/video)</h2>
            </div>
            <span className="pill">
              {telemetry ? `${telemetry.inliers} inliers` : "Waiting"}
            </span>
          </div>
          <div className={`cameraStage ${hasVideo ? "hasImage" : ""}`}>
            <canvas
              ref={videoCanvasRef}
              hidden={!hasVideo}
              aria-label="Live annotated camera feed"
            />
            {!hasVideo && (
              <div className="emptyState">
                <strong>No live video frames yet</strong>
                <span>
                  Start the backend with{" "}
                  <code>python -m src.realtime.cli --source kitti --port 8080</code>
                </span>
              </div>
            )}
            {hasVideo && (
              <div className="cameraTag">
                {trackState} · F#{telemetry?.frame_id ?? 0} ·{" "}
                {telemetry?.tracking_fps.toFixed(1) ?? "0.0"} FPS
              </div>
            )}
          </div>
          <div className="frameFooter">
            <span>Per-client coalesced JPEG stream (non-blocking)</span>
            <span>Green: PnP inliers · Red: outliers</span>
          </div>
        </section>

        <section className="panel mapPanel">
          <div className="panelHeading">
            <div>
              <span className="eyebrow">KEYFRAME-ANCHORED 3D MAP</span>
              <h2>Frustum, point cloud &amp; loop arcs</h2>
            </div>
            <button
              className="button"
              onClick={() => setView(INITIAL_VIEW)}
            >
              Reset view
            </button>
          </div>
          <div className="chartBody">
            <canvas
              ref={mapCanvasRef}
              role="img"
              aria-label="Interactive live 3D camera frustum, point cloud, and loop closure map"
              style={{ touchAction: "none" }}
              onPointerDown={(e) => {
                dragRef.current = { x: e.clientX, y: e.clientY };
                e.currentTarget.setPointerCapture(e.pointerId);
              }}
              onPointerMove={(e) => {
                if (!dragRef.current) return;
                const dx = e.clientX - dragRef.current.x;
                const dy = e.clientY - dragRef.current.y;
                dragRef.current = { x: e.clientX, y: e.clientY };
                setView((v) => ({
                  ...v,
                  yaw: v.yaw + dx * 0.006,
                  pitch: Math.max(-1.5, Math.min(1.5, v.pitch + dy * 0.006)),
                }));
              }}
              onPointerUp={() => {
                dragRef.current = null;
              }}
              onPointerCancel={() => {
                dragRef.current = null;
              }}
            />
          </div>
          <div className="chartFooter mapToolbar">
            <span>
              Drag to orbit · {mapState.keyframes.size} KF chunks ·{" "}
              {mapState.loopEdges.length} loop arcs
            </span>
            <div>
              <button
                className="button"
                aria-label="Zoom out 3D map"
                onClick={() =>
                  setView((v) => ({ ...v, zoom: Math.max(0.3, v.zoom / 1.3) }))
                }
              >
                −
              </button>
              <button
                className="button"
                aria-label="Zoom in 3D map"
                onClick={() =>
                  setView((v) => ({ ...v, zoom: Math.min(8, v.zoom * 1.3) }))
                }
              >
                +
              </button>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
