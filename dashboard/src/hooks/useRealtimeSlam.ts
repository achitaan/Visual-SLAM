"use client";

import { useCallback, useEffect, useRef, useState } from "react";

export interface RealtimeTelemetry {
  running: boolean;
  paused: boolean;
  finished: boolean;
  source: string;
  total_frames: number | null;
  playback_rate: number;
  drop_frames: boolean;
  frame_id: number;
  processed_frames: number;
  timestamp_s: number;
  pose_T_wc: number[][];
  capture_fps: number;
  tracking_fps: number;
  track_p50_ms: number;
  track_p95_ms: number;
  ba_p50_ms: number;
  loop_v2a_p95_ms: number;
  dropped_frames: number;
  queue_age_ms: number;
  tracking_state: "INIT" | "TRACKING" | "LOST" | "RELOCALIZING" | string;
  state: string;
  inliers: number;
  matches: number;
  reprojection_error: number | null;
  active_landmarks: number;
  keyframes: number;
  verified_loops_applied: number;
  pose_revision: number;
  map_revision: number;
  backend_device: string;
  last_error: string | null;
}

export interface KeyframeChunk {
  kfId: number;
  frameId: number;
  pose: number[][];
  localPoints: Float32Array;
}

export interface LoopEdge {
  from_kf: number;
  to_kf: number;
  timestamp_s: number;
  flashUntil: number;
}

export interface LiveMapState {
  keyframes: Map<number, KeyframeChunk>;
  trajectory: [number, number, number][];
  loopEdges: LoopEdge[];
  poseRevision: number;
  version: number;
}

export interface SwitchSourceSpec {
  kind: "kitti" | "video" | "uvc" | "realsense" | "depthai";
  data_root?: string;
  sequence?: string;
  video_left?: string;
  video_right?: string;
  calib?: string;
  camera_id?: number;
  right_camera_id?: number | null;
  rate?: number;
  drop_stale?: boolean;
  auto_start?: boolean;
}

function decodeBase64Float32(b64: string): Float32Array {
  if (!b64) return new Float32Array(0);
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return new Float32Array(bytes.buffer);
}

function toWsBase(httpBase: string): string {
  const trimmed = httpBase.replace(/\/+$/, "");
  if (trimmed.startsWith("https://")) return "wss://" + trimmed.slice(8);
  if (trimmed.startsWith("http://")) return "ws://" + trimmed.slice(7);
  if (trimmed.startsWith("ws://") || trimmed.startsWith("wss://")) return trimmed;
  return "ws://" + trimmed;
}

export function useRealtimeSlam(
  apiBaseUrl: string,
  onVideoFrame?: (blob: Blob) => void
) {
  const [connected, setConnected] = useState(false);
  const [telemetry, setTelemetry] = useState<RealtimeTelemetry | null>(null);
  const [controlError, setControlError] = useState<string | null>(null);
  const [mapVersion, setMapVersion] = useState(0);

  const mapRef = useRef<LiveMapState>({
    keyframes: new Map(),
    trajectory: [],
    loopEdges: [],
    poseRevision: 0,
    version: 0,
  });

  const onVideoRef = useRef(onVideoFrame);
  onVideoRef.current = onVideoFrame;

  const baseHttp = apiBaseUrl.replace(/\/+$/, "");
  const baseWs = toWsBase(baseHttp);

  useEffect(() => {
    let disposed = false;
    let telWs: WebSocket | null = null;
    let mapWs: WebSocket | null = null;
    let vidWs: WebSocket | null = null;
    const timers: ReturnType<typeof setTimeout>[] = [];

    const scheduleReconnect = (fn: () => void) => {
      if (disposed) return;
      timers.push(setTimeout(fn, 1500));
    };

    const connectTelemetry = () => {
      if (disposed) return;
      try {
        const ws = new WebSocket(`${baseWs}/ws/telemetry`);
        telWs = ws;
        ws.onopen = () => {
          if (!disposed) setConnected(true);
        };
        ws.onmessage = (ev) => {
          try {
            const msg = JSON.parse(ev.data) as RealtimeTelemetry;
            if (!disposed && msg && typeof msg.frame_id === "number") {
              setTelemetry(msg);
            }
          } catch {
            /* ignore malformed frames */
          }
        };
        ws.onerror = () => {
          if (!disposed) setConnected(false);
        };
        ws.onclose = () => {
          if (!disposed) {
            setConnected(false);
            scheduleReconnect(connectTelemetry);
          }
        };
      } catch {
        scheduleReconnect(connectTelemetry);
      }
    };

    const connectMap = () => {
      if (disposed) return;
      try {
        const ws = new WebSocket(`${baseWs}/ws/map`);
        mapWs = ws;
        ws.onmessage = (ev) => {
          try {
            const msg = JSON.parse(ev.data);
            const state = mapRef.current;
            const now = Date.now();
            if (msg?.type === "map_snapshot") {
              const nextKfs = new Map<number, KeyframeChunk>();
              for (const kf of msg.keyframes ?? []) {
                nextKfs.set(kf.kf_id, {
                  kfId: kf.kf_id,
                  frameId: kf.frame_id,
                  pose: kf.pose_T_wc,
                  localPoints: decodeBase64Float32(kf.points_f32_b64 || ""),
                });
              }
              state.keyframes = nextKfs;
              state.trajectory = Array.isArray(msg.trajectory) ? msg.trajectory : [];
              state.loopEdges = (msg.loop_edges ?? []).map((e: any) => ({
                from_kf: e.from_kf,
                to_kf: e.to_kf,
                timestamp_s: e.timestamp_s ?? 0,
                flashUntil: 0,
              }));
              state.poseRevision = msg.pose_revision ?? 0;
              state.version += 1;
              if (!disposed) setMapVersion(state.version);
            } else if (msg?.type === "map_delta") {
              for (const kf of msg.new_keyframes ?? []) {
                state.keyframes.set(kf.kf_id, {
                  kfId: kf.kf_id,
                  frameId: kf.frame_id,
                  pose: kf.pose_T_wc,
                  localPoints: decodeBase64Float32(kf.points_f32_b64 || ""),
                });
              }
              if (msg.kf_pose_batch) {
                for (const [idStr, pose] of Object.entries(msg.kf_pose_batch)) {
                  const existing = state.keyframes.get(Number(idStr));
                  if (existing) {
                    existing.pose = pose as number[][];
                  }
                }
              }
              if (Array.isArray(msg.trajectory)) {
                state.trajectory = msg.trajectory;
              } else if (Array.isArray(msg.trajectory_append)) {
                state.trajectory.push(...msg.trajectory_append);
              }
              if (Array.isArray(msg.loop_edges) && msg.loop_edges.length > 0) {
                for (const e of msg.loop_edges) {
                  state.loopEdges.push({
                    from_kf: e.from_kf,
                    to_kf: e.to_kf,
                    timestamp_s: e.timestamp_s ?? 0,
                    flashUntil: now + 2600,
                  });
                }
              }
              state.poseRevision = msg.pose_revision ?? state.poseRevision;
              state.version += 1;
              if (!disposed) setMapVersion(state.version);
            }
          } catch {
            /* ignore malformed map payloads */
          }
        };
        ws.onclose = () => scheduleReconnect(connectMap);
      } catch {
        scheduleReconnect(connectMap);
      }
    };

    const connectVideo = () => {
      if (disposed) return;
      try {
        const ws = new WebSocket(`${baseWs}/ws/video`);
        ws.binaryType = "blob";
        vidWs = ws;
        ws.onmessage = (ev) => {
          if (ev.data instanceof Blob && onVideoRef.current) {
            onVideoRef.current(ev.data);
          }
        };
        ws.onclose = () => scheduleReconnect(connectVideo);
      } catch {
        scheduleReconnect(connectVideo);
      }
    };

    connectTelemetry();
    connectMap();
    connectVideo();

    return () => {
      disposed = true;
      timers.forEach(clearTimeout);
      telWs?.close();
      mapWs?.close();
      vidWs?.close();
    };
  }, [baseWs]);

  const postJson = useCallback(
    async (path: string, body?: unknown) => {
      setControlError(null);
      try {
        const res = await fetch(`${baseHttp}${path}`, {
          method: "POST",
          headers: body !== undefined ? { "Content-Type": "application/json" } : undefined,
          body: body !== undefined ? JSON.stringify(body) : undefined,
        });
        if (!res.ok) {
          const text = await res.text();
          throw new Error(text || `HTTP ${res.status}`);
        }
        return await res.json();
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        setControlError(msg);
        return null;
      }
    },
    [baseHttp]
  );

  const start = useCallback(() => postJson("/api/control/start"), [postJson]);
  const pause = useCallback(() => postJson("/api/control/pause"), [postJson]);
  const resume = useCallback(() => postJson("/api/control/resume"), [postJson]);
  const resetMap = useCallback(async () => {
    const res = await postJson("/api/control/reset");
    mapRef.current = {
      keyframes: new Map(),
      trajectory: [],
      loopEdges: [],
      poseRevision: 0,
      version: mapRef.current.version + 1,
    };
    setMapVersion(mapRef.current.version);
    return res;
  }, [postJson]);

  const setRate = useCallback(
    (rate: number) => postJson("/api/control/rate", { rate }),
    [postJson]
  );

  const setDropFrames = useCallback(
    (drop_frames: boolean) => postJson("/api/control/drop_frames", { drop_frames }),
    [postJson]
  );

  const switchSource = useCallback(
    async (spec: SwitchSourceSpec) => {
      const res = await postJson("/api/source", spec);
      if (res?.ok) {
        mapRef.current = {
          keyframes: new Map(),
          trajectory: [],
          loopEdges: [],
          poseRevision: 0,
          version: mapRef.current.version + 1,
        };
        setMapVersion(mapRef.current.version);
      }
      return res;
    },
    [postJson]
  );

  return {
    connected,
    telemetry,
    mapState: mapRef.current,
    mapVersion,
    controlError,
    start,
    pause,
    resume,
    resetMap,
    setRate,
    setDropFrames,
    switchSource,
  };
}
