from dataclasses import dataclass


@dataclass(frozen=True)
class TelemetryConfig:
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 8765
    stream_images: bool = True  # Camera view
    stream_features: bool = True
    stream_fps: bool = True
    max_features: int = 300
    stream_map_points: bool = True
    max_map_points: int = 1000
    frame_delay_ms: int = 50  # Pacing between frames


TELEMETRY = TelemetryConfig()

# SLAM feature flags (incremental rollout)
ENABLE_KEYFRAMES = True
ENABLE_LOCAL_MAP = True
ENABLE_POSE_GRAPH = True
ENABLE_LOOP_CLOSURE = True

# VO/SLAM tuning (safe defaults)
USE_RELATIVE_SCALE_FIX = False
KEYFRAME_INTERVAL = 15  # Less frequent keyframes
MIN_KEYFRAME_TRANSLATION = 0.5  # Larger threshold

# Pose graph / loop closure tuning
VOCAB_BUILD_MIN_FRAMES = 5
VOCAB_NUM_CLUSTERS = 30  # Fewer clusters = faster BoW
LOOP_CLOSURE_THRESHOLD = 0.3
POSE_GRAPH_OPT_EVERY = 20  # Less frequent optimization

# Local map tuning
LOCAL_MAP_MAX_POINTS = 2000
LOCAL_MAP_MIN_DEPTH = 0.1
LOCAL_MAP_MAX_RANGE = 30.0

# Stereo PnP tuning
STEREO_MIN_MATCHES = 15
STEREO_MIN_VALID_3D = 15
STEREO_PNP_ITER = 200
STEREO_PNP_REPROJ = 2.0
STEREO_PNP_CONF = 0.995
STEREO_OWNED_IMAGE_BUNDLE = False

# Relocalization tuning
ENABLE_RELOCALIZATION = False
LOST_INLIER_RATIO = 0.2
LOST_MIN_INLIERS = 30
LOST_CONSEC_FRAMES = 5
RELOCALIZE_MIN_MATCHES = 40
RELOCALIZE_PNP_MIN_INLIERS = 20
RELOCALIZE_PNP_REPROJ = 4.0
