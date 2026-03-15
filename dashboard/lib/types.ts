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
  schema_version: number;
  frame_index: number;
  timestamp: number;
  mode: "vo" | "slam";
  stream_enabled: boolean;
  pose_T_wc: TelemetryPose;
  velocity?: number[] | null;
  fps?: number | null;
  tracking: {
    num_matches?: number | null;
    num_inliers?: number | null;
    inlier_ratio?: number | null;
    reprojection_error?: number | null;
  };
  map: {
    keyframes: number;
    map_points: number;
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
