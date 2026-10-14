export type TelemetryPose = number[][];

export type TelemetryFeature = {
  x: number;
  y: number;
  inlier: boolean;
};

export type TelemetryEvent = {
  timestamp: number;
  type: string;
  message: string;
  severity?: "info" | "warn" | "error";
};

export type TelemetryFrame = {
  trajectory?: number[][];
  schema_version: number;
  frame_index: number;
  timestamp: number;
  mode: "vo" | "slam";
  mode_locked?: boolean;
  translation_scale?: "metric" | "arbitrary" | "unspecified";
  sequence?: string | null;
  total_frames?: number | null;
  run_id?: string | null;
  overlay_enabled?: boolean;
  stream_enabled: boolean;
  pose_T_wc: TelemetryPose;
  velocity?: number[] | null;
  fps?: number | null;
  tracking: {
    num_matches?: number | null;
    num_inliers?: number | null;
    inlier_ratio?: number | null;
    reprojection_error?: number | null;
    tracking_ok?: boolean;
    state?: "initializing" | "tracking" | "lost" | "relocalized";
  };
  map: {
    keyframes: number;
    map_points: number;
    revision?: number;
  };
  map_points?: number[][] | null;
  pose_graph?: {
    optimized_pose_T_wc?: TelemetryPose | null;
    optimized_poses_count?: number | null;
    optimized_poses?: TelemetryPose[] | null;
  } | null;
  expected_pose_T_wc?: TelemetryPose | null;
  image?: {
    encoding: "jpg" | "png";
    data_base64: string;
    width: number;
    height: number;
  } | null;
  features?: TelemetryFeature[] | null;
  events?: TelemetryEvent[];
};

export type ControlMessage =
  | { type: "control"; action: "start" | "stop" }
  | { type: "control"; action: "set_mode"; mode: "vo" | "slam" }
  | { type: "control"; action: "toggle_overlay"; enabled: boolean };

export type BenchmarkStatus = {
  schema_version: 1;
  kind: "benchmark_status";
  timestamp: number;
  running: boolean;
  paused?: boolean;
  finished: boolean;
  completed: number;
  total: number;
  stream_available: boolean;
  active: { sequence: string; sensor: string; run_id: string; progress: {
    frames: number; total_frames: number; state: string; landmarks: number; updated_at: number;
  } | null } | null;
  rows: { sequence: string; sensor: string; status: string; frames?: number; lost_frames?: number;
    ate_rmse_m?: number | null; alignment?: string; translation_percent?: number | null }[];
};
