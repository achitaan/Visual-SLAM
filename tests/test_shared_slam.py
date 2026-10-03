import numpy as np
import cv2 as cv
import pytest
from slam_state import MapState, MappingKeyframe, Observation
from shared_slam import SharedSlam, MappingConfig, StereoCamera
from mapping_geometry import (
    project,
    initialize_monocular,
    triangulate,
    estimate_pose,
    estimate_stereo_reference,
)
from reconstruction import fit_depth_scale, DepthPrediction, fuse_reconstruction
import json
from local_bundle import local_bundle_adjustment
from live_loops import optimize_similarities, verify_similarity, LiveLoopWorker
from loop_geometry import StereoLoopFrame, verify_loop

K = np.array([[250.0, 0, 320], [0, 250, 240], [0, 0, 1]])


def scene():
    return np.random.default_rng(2).uniform([-5, -4, 4], [5, 4, 14], (400, 3))


def pose(x):
    result = np.eye(4)
    result[0, 3] = x
    return result


def test_monocular_initialization_and_degenerate_rotation():
    cv.setRNGSeed(0)
    world = scene()
    a, _ = project(world, pose(0), K)
    b, _ = project(world, pose(1), K)
    result = initialize_monocular(
        a.astype(np.float32), b.astype(np.float32), K, (640, 480)
    )
    assert result is not None
    estimated, points, ids = result
    assert len(points) > 200
    assert np.allclose(estimated, pose(1), atol=1e-4)
    assert (
        initialize_monocular(a.astype(np.float32), a.astype(np.float32), K, (640, 480))
        is None
    )
    flat = world.copy()
    flat[:, 2] = 8
    a, _ = project(flat, pose(0), K)
    b, _ = project(flat, pose(1), K)
    assert (
        initialize_monocular(a.astype(np.float32), b.astype(np.float32), K, (640, 480))
        is None
    )


def test_low_parallax_cannot_create_depth():
    world = scene()
    first, _ = project(world, pose(0), K)
    second, _ = project(world, pose(0.001), K)
    points, ids = triangulate(first, second, pose(0), pose(0.001), K)
    assert not len(points) and not len(ids)
    assert initialize_monocular(first, second, K, (640, 480)) is None


def test_keyframe_keeps_flow_observations_without_detected_features():
    slam = SharedSlam(K)
    lid = slam.map.add_landmark([0, 0, 5], np.ones(128, np.float32), 0, {})
    slam.accepted_tracks = [(lid, np.array([320.0, 240.0]))]
    key = slam._keyframe(
        0,
        np.eye(4),
        np.empty((0, 2)),
        np.empty((0, 128), np.float32),
        np.empty((0, 3)),
        np.empty(0),
        {},
    )
    assert slam.map.keyframes[key].landmark_ids.tolist() == [lid]
    assert np.array_equal(
        slam.map.landmarks[lid].observations[key].pixel, [320.0, 240.0]
    )
    slam.close()


def test_nearby_detector_pixel_does_not_replace_verified_observation():
    slam = SharedSlam(K)
    lid = slam.map.add_landmark([0, 0, 5], np.ones(128, np.float32), 0, {})
    slam.accepted_tracks = [(lid, np.array([320.0, 240.0]))]
    key = slam._keyframe(
        0,
        np.eye(4),
        np.array([[322.0, 240.0]]),
        np.ones((1, 128), np.float32),
        np.full((1, 3), np.nan),
        np.array([300.0]),
        {0: lid},
    )
    observation = slam.map.landmarks[lid].observations[key]
    assert np.array_equal(observation.pixel, [320.0, 240.0])
    assert observation.right_u is None
    slam.close()


def test_stereo_flow_observation_keeps_its_own_metric_constraint():
    baseline = 0.54
    q = np.array(
        [[1, 0, 0, -320], [0, 1, 0, -240], [0, 0, 0, 250], [0, 0, 1 / baseline, 0]],
        float,
    )
    slam = SharedSlam(K, stereo=StereoCamera(None, q, baseline))
    slam.current_disparity = np.full((480, 640), -1.0)
    slam.current_disparity[240, 320] = 27.0
    slam.current_disparity[240, 322] = 10.0
    lid = slam.map.add_landmark([0, 0, 5], np.ones(128, np.float32), 0, {})
    slam.accepted_tracks = [(lid, np.array([320.0, 240.0]))]
    key = slam._keyframe(
        0,
        np.eye(4),
        np.array([[322.0, 240.0]]),
        np.ones((1, 128), np.float32),
        np.full((1, 3), np.nan),
        np.array([312.0]),
        {0: lid},
    )
    observation = slam.map.landmarks[lid].observations[key]
    assert np.array_equal(observation.pixel, [320.0, 240.0])
    assert observation.right_u == 293.0
    points, right = slam._measure_stereo_pixels(
        np.array([[320.0, 240.0], [-1, 0], [322, 241]])
    )
    assert np.allclose(points[0], [0, 0, 5])
    assert np.isnan(points[1:]).all() and np.isnan(right[1:]).all()
    slam.close()


def test_monocular_extraction_does_not_read_right_camera():
    class ForbiddenCamera:
        def __getattribute__(self, name):
            raise AssertionError("Monocular estimator accessed a second camera")

    slam = SharedSlam(K)
    slam._extract(np.zeros((480, 640), np.uint8), ForbiddenCamera())
    slam.close()


def test_conflicting_flow_does_not_overwrite_mutual_descriptor_match(monkeypatch):
    world = scene()
    descriptors = (
        np.random.default_rng(44).normal(size=(len(world), 128)).astype(np.float32)
    )
    old, _ = project(world, pose(0), K)
    observed, _ = project(world, pose(0.3), K)
    slam = SharedSlam(K)
    slam.map.keyframes[0] = MappingKeyframe(
        0, 0, pose(0), old, descriptors, np.arange(len(world))
    )
    for i, point in enumerate(world):
        slam.map.add_landmark(point, descriptors[i], 0, {0: Observation(old[i])})
    slam.map.record(pose(0), "tracking", 0)
    slam.previous_gray = slam.current_gray = np.zeros((480, 640), np.uint8)
    slam.previous_tracks = [(i, p) for i, p in enumerate(old)]
    wrong = observed.copy()
    wrong[:, 0] += 10
    calls = []

    def flow(*args, **kwargs):
        calls.append(1)
        pixels = wrong if len(calls) == 1 else old
        return (
            pixels.astype(np.float32).reshape(-1, 1, 2),
            np.ones((len(world), 1), np.uint8),
            None,
        )

    monkeypatch.setattr(cv, "calcOpticalFlowPyrLK", flow)
    result, stats = slam._track(observed, descriptors, (640, 480))
    assert result is not None and np.allclose(result[0], pose(0.3), atol=1e-3)
    assert stats["flow_descriptor_conflicts"] > 100
    slam.close()


def test_flow_cannot_reintroduce_landmark_predicted_behind_camera(monkeypatch):
    import shared_slam as module

    def forward(z):
        p = np.eye(4)
        p[2, 3] = z
        return p

    world = scene()
    world[:, 2] += 5
    world = np.vstack([world, [0, 0, 1.2]])
    descriptors = (
        np.random.default_rng(71).normal(size=(len(world), 128)).astype(np.float32)
    )
    old, _ = project(world, forward(0.8), K)
    observed, _ = project(world[:-1], forward(1.6), K)
    observed = np.vstack([observed, old[-1]])  # A look-alike remains at the old pixel.
    slam = SharedSlam(K)
    slam.map.keyframes[0] = MappingKeyframe(
        0, 0, forward(0), old, descriptors, np.arange(len(world))
    )
    for i, point in enumerate(world):
        slam.map.add_landmark(point, descriptors[i], 0, {0: Observation(old[i])})
    slam.map.record(forward(0), "tracking", 0)
    slam.map.record(forward(0.8), "tracking", 0)
    slam.previous_gray = slam.current_gray = np.zeros((480, 640), np.uint8)
    slam.previous_tracks = [(i, pixel) for i, pixel in enumerate(old)]
    calls = []

    def flow(*args, **kwargs):
        calls.append(1)
        pixels = observed if len(calls) == 1 else old
        return (
            pixels.astype(np.float32).reshape(-1, 1, 2),
            np.ones((len(world), 1), np.uint8),
            None,
        )

    original = module.estimate_pose

    def inspect(points, *args, **kwargs):
        assert np.all(points[:, 2] > 5), "An invisible landmark entered PnP via flow"
        return original(points, *args, **kwargs)

    monkeypatch.setattr(cv, "calcOpticalFlowPyrLK", flow)
    monkeypatch.setattr(module, "estimate_pose", inspect)
    result, stats = slam._track(observed, descriptors, (640, 480))
    assert result is not None and np.allclose(result[0], forward(1.6), atol=1e-3)
    assert stats["flow_visibility_rejections"] >= 1
    assert all(lid != len(world) - 1 for lid, _ in slam.accepted_tracks)
    slam.close()


def test_stereo_temporal_recovery_requires_bidirectional_geometry():
    world = scene()
    descriptors = (
        np.random.default_rng(31).normal(size=(len(world), 128)).astype(np.float32)
    )
    slam = SharedSlam(K, stereo=StereoCamera(None, np.eye(4), 0.54))
    current = {"x": 0.0}

    def extract(*_):
        pixels, depth = project(world, pose(current["x"]), K)
        return (
            pixels.astype(np.float32),
            descriptors.copy(),
            world - pose(current["x"])[:3, 3],
            pixels[:, 0] - K[0, 0] * 0.54 / depth,
        )

    slam._extract = extract
    slam.process(0, np.zeros((480, 640), np.uint8))
    slam._track = lambda *_: (None, {})
    slam._relocalize = lambda *_: (None, {})
    current["x"] = 0.3
    estimated, info = slam.process(1, np.zeros((480, 640), np.uint8))
    assert info["pose_source"] == "stereo_tracking_reference"
    assert np.allclose(estimated, pose(0.3), atol=1e-3)
    before = (len(slam.map.landmarks), len(slam.map.keyframes))
    descriptors[:] = np.random.default_rng(32).normal(size=descriptors.shape)
    _, info = slam.process(2, np.zeros((480, 640), np.uint8))
    assert info["state"] == "lost"
    assert before == (len(slam.map.landmarks), len(slam.map.keyframes))
    slam.close()


def test_stereo_rejects_self_consistent_map_pose_that_conflicts_with_metric_motion():
    world = scene()
    descriptors = (
        np.random.default_rng(52).normal(size=(len(world), 128)).astype(np.float32)
    )
    slam = SharedSlam(K, stereo=StereoCamera(None, np.eye(4), 0.54))
    current = {"x": 0.0}

    def extract(*_):
        pixels, depth = project(world, pose(current["x"]), K)
        return (
            pixels.astype(np.float32),
            descriptors.copy(),
            world - pose(current["x"])[:3, 3],
            pixels[:, 0] - K[0, 0] * 0.54 / depth,
        )

    slam._extract = extract
    slam.process(0, np.zeros((480, 640), np.uint8))
    # An internally distorted map can claim excellent left-image residuals.
    slam._track = lambda *_: (
        (pose(1.5), {}),
        {"reprojection_error": 0.01, "num_inliers": 200},
    )
    current["x"] = 0.3
    estimated, info = slam.process(1, np.zeros((480, 640), np.uint8))
    assert info["map_pose_rejected_for_stereo_conflict"]
    assert info["map_reference_translation_error_m"] > 1.0
    assert info["pose_source"] == "stereo_tracking_reference"
    assert np.allclose(estimated, pose(0.3), atol=1e-3)
    assert len(slam.map.keyframes) == 2
    slam.close()


def test_bidirectional_stereo_uses_each_directions_available_depth():
    world = scene()
    descriptors = (
        np.random.default_rng(61).normal(size=(len(world), 128)).astype(np.float32)
    )
    first_pixels, _ = project(world, pose(0), K)
    second_pixels, _ = project(world, pose(0.3), K)
    first_depth = world.copy()
    second_depth = world - pose(0.3)[:3, 3]
    # No correspondence has depth in both images, but both PnP directions
    # independently have ample source geometry and target image observations.
    first_depth[1::2] = np.nan
    second_depth[::2] = np.nan
    first = StereoLoopFrame(first_pixels, first_depth, descriptors, (640, 480))
    second = StereoLoopFrame(
        second_pixels, second_depth, descriptors.copy(), (640, 480)
    )
    measured = verify_loop(first, second, K)
    assert measured is not None and np.allclose(
        measured["measurement"], pose(0.3), atol=1e-3
    )
    assert measured["matches"] == 200 and measured["reverse_depth_support"] == 200
    # A single direction alone cannot authorize a loop constraint.
    second.points[:] = np.nan
    assert verify_loop(first, second, K) is None


def test_tracking_reference_uses_source_depth_without_relaxing_loop_verification():
    world = scene()
    descriptors = (
        np.random.default_rng(102).normal(size=(len(world), 128)).astype(np.float32)
    )
    a, _ = project(world, pose(0), K)
    b, _ = project(world, pose(0.3), K)
    first = StereoLoopFrame(a, world, descriptors, (640, 480))
    second = StereoLoopFrame(
        b, np.full_like(world, np.nan), descriptors.copy(), (640, 480)
    )
    result = estimate_stereo_reference(first, second, K, initial_pose=np.eye(4))
    assert result is not None and np.allclose(
        result["measurement"], pose(0.3), atol=1e-3
    )
    assert not result["reverse_checked"]
    assert verify_loop(first, second, K) is None
    # Tracking still enforces the existing 15-inlier minimum.
    short = StereoLoopFrame(a[:14], world[:14], descriptors[:14], (640, 480))
    assert estimate_stereo_reference(short, second, K) is None


def test_tracking_reference_rejects_contradictory_reverse_stereo_geometry():
    cv.setRNGSeed(0)
    world = scene()
    descriptors = (
        np.random.default_rng(104).normal(size=(len(world), 128)).astype(np.float32)
    )
    first_pixels, _ = project(world, pose(0), K)
    second_pixels, _ = project(world, pose(0.3), K)
    first = StereoLoopFrame(first_pixels, world, descriptors, (640, 480))
    # Consistent image observations but incompatible metric depth in the target.
    second = StereoLoopFrame(
        second_pixels, 5 * (world - pose(0.3)[:3, 3]), descriptors.copy(), (640, 480)
    )
    assert estimate_pose(first.points, second.pixels, K, second.image_size) is not None
    assert estimate_pose(second.points, first.pixels, K, first.image_size) is not None
    assert estimate_stereo_reference(first, second, K, initial_pose=np.eye(4)) is None


def test_stereo_keyframe_recovery_does_not_depend_on_corrupted_world_landmarks():
    world = scene()
    descriptors = (
        np.random.default_rng(93).normal(size=(len(world), 128)).astype(np.float32)
    )
    slam = SharedSlam(K, stereo=StereoCamera(None, np.eye(4), 0.54))
    current = {"x": 0.0}

    def extract(*_):
        pixels, depth = project(world, pose(current["x"]), K)
        return (
            pixels.astype(np.float32),
            descriptors.copy(),
            world - pose(current["x"])[:3, 3],
            pixels[:, 0] - K[0, 0] * 0.54 / depth,
        )

    slam._extract = extract
    slam.process(0, np.zeros((480, 640), np.uint8))
    # Raw keyframe camera geometry remains valid when global point estimates fail.
    for landmark in slam.map.landmarks.values():
        landmark.position[:] = [-1000, -1000, -1000]
    slam.previous_stereo_geometry = None
    current["x"] = 0.3
    estimated, info = slam.process(1, np.zeros((480, 640), np.uint8))
    assert np.allclose(estimated, pose(0.3), atol=1e-3)
    assert info["pose_source"] == "keyframe_stereo_reference"
    assert info["state"] == "relocalized"
    assert np.array_equal(slam.map.poses[0], np.eye(4))
    assert len(slam.map.keyframes) == 2
    # Appearance retrieval without valid stereo geometry cannot recover a pose.
    slam.map.keyframes[0].depth_points[:] = np.nan
    slam.map.keyframes[1].depth_points[:] = np.nan
    result, _ = slam._keyframe_stereo_reference(*extract()[:3], (640, 480))
    assert result is None
    slam.close()


def test_stale_loop_job_retries_only_with_fresh_revision():
    from concurrent.futures import Future

    state = MapState(metric=True)
    for i in range(3):
        state.keyframes[i] = MappingKeyframe(
            i, i * 100, pose(i), np.empty((0, 2)), np.empty((0, 128)), np.empty(0, int)
        )
    state.revision = 1
    worker = LiveLoopWorker(K, True)

    def result(revision):
        return {
            "revision": revision,
            "loops": {},
            "correction": {i: pose(i) for i in range(3)},
            "scales": {i: 1.0 for i in range(3)},
        }

    worker.future = Future()
    worker.future.set_result(result(0))

    def schedule(current):
        assert current.revision == 1
        worker.future = Future()
        worker.future.set_result(result(current.revision))

    worker.schedule = schedule
    worker.close(state)
    assert [e["type"] for e in worker.events] == ["loop_discarded", "loop_applied"]
    assert state.revision == 2


def test_stale_stereo_correction_keeps_independent_loop_measurement():
    from concurrent.futures import Future

    state = MapState(metric=True)
    for i in range(3):
        state.keyframes[i] = MappingKeyframe(
            i, i * 100, pose(i), np.empty((0, 2)), np.empty((0, 128)), np.empty(0, int)
        )
    state.revision = 1
    worker = LiveLoopWorker(K, True)
    measurement = {"measurement": pose(2), "inliers": 40}
    worker.future = Future()
    worker.future.set_result(
        {
            "revision": 0,
            "loops": {(0, 2): measurement},
            "correction": {i: pose(i) for i in range(3)},
            "scales": {i: 1.0 for i in range(3)},
        }
    )
    assert not worker.poll(state)
    assert state.revision == 1 and not worker.verified
    assert (0, 2) in worker.pending_loops
    worker.close(state)
    assert state.revision == 2
    assert (0, 2) in worker.verified
    assert not worker.pending_loops


def test_variable_speed_tracks_map_scale_and_loss_holds_without_new_geometry():
    cv.setRNGSeed(0)
    world = scene()
    desc = np.random.default_rng(4).normal(size=(len(world), 128)).astype(np.float32)
    slam = SharedSlam(K, config=MappingConfig(keyframe_interval=20))
    camera_positions = [0, 1, 1.1, 1.8]
    current = {"frame": 0}

    def extract(image, right):
        pixels, _ = project(world, pose(camera_positions[current["frame"]]), K)
        return (
            pixels.astype(np.float32),
            desc.copy(),
            np.full((len(world), 3), np.nan),
            np.full(len(world), np.nan),
        )

    slam._extract = extract
    for i, x in enumerate(camera_positions):
        current["frame"] = i
        estimated, stats = slam.process(i, np.zeros((480, 640), np.uint8))
        if i:
            assert np.allclose(estimated, pose(x), atol=1e-3)
    assert slam.map.statuses == ["initializing", "tracking", "tracking", "tracking"]
    before = (len(slam.map.keyframes), len(slam.map.landmarks), slam.map.revision)
    slam._extract = lambda *_: (
        np.empty((0, 2), np.float32),
        np.empty((0, 128), np.float32),
        np.empty((0, 3)),
        np.empty(0),
    )
    held, stats = slam.process(4, np.zeros((480, 640), np.uint8))
    assert stats["state"] == "lost"
    assert np.array_equal(held, pose(camera_positions[-1])) or np.allclose(
        held, pose(camera_positions[-1]), atol=1e-3
    )
    assert before == (
        len(slam.map.keyframes),
        len(slam.map.landmarks),
        slam.map.revision,
    )


def test_atomic_corrections_and_stale_revision():
    state = MapState()
    for i in range(2):
        state.keyframes[i] = MappingKeyframe(
            i, i, pose(i), np.empty((0, 2)), np.empty((0, 128)), np.empty(0, int)
        )
    lid = state.add_landmark(
        [2, 0, 5], np.ones(128), 1, {1: Observation(np.array([320, 240]))}
    )
    state.record(pose(1.5), "tracking", 1)
    assert not state.apply_corrections(1, {0: pose(0), 1: pose(2)})
    assert state.apply_corrections(0, {0: pose(0), 1: pose(2)}, {0: 1.0, 1: 2.0})
    assert np.allclose(state.landmarks[lid].position, [4, 0, 10])
    assert np.allclose(state.poses[0][:3, 3], [3, 0, 0])
    assert state.revision == 1


def test_pnp_rejects_outliers_and_negative_depth():
    cv.setRNGSeed(0)
    world = scene()
    pixels, _ = project(world, pose(0.3), K)
    pixels[:100] = np.random.default_rng(5).uniform([0, 0], [640, 480], (100, 2))
    result = estimate_pose(world, pixels, K, (640, 480))
    assert result is not None
    assert np.allclose(result[0], pose(0.3), atol=1e-3)
    assert len(result[1]) >= 290


def test_bad_motion_prior_cannot_override_independent_pnp_geometry():
    rng = np.random.default_rng(70)
    world = rng.uniform([-3, -2, 5], [3, 2, 9], (180, 3))
    expected = pose(0.3)
    pixels, _ = project(world, expected, K)
    pixels += rng.normal(0, 0.25, pixels.shape)
    pixels[:45] = rng.uniform([20, 20], [620, 460], (45, 2))
    bad_prior = pose(5)
    bad_prior[:3, :3] = cv.Rodrigues(np.array([0.0, np.pi / 2, 0.0]))[0]
    cv.setRNGSeed(0)
    diagnostics = {}
    result = estimate_pose(
        world,
        pixels,
        K,
        (640, 480),
        initial_pose=bad_prior,
        diagnostics=diagnostics,
    )
    assert result is not None, "A bad prior hid valid correspondence geometry"
    estimated, inliers, error = result
    assert np.linalg.norm(estimated[:3, 3] - expected[:3, 3]) < 0.02
    assert np.all(inliers >= 45)
    assert len(inliers) >= 120 and error < 0.5
    assert diagnostics["pnp_method"] == "epnp_ransac"
    assert diagnostics["seeded_rejection_reason"] == "negative_depth"
    assert "pose_rejection_reason" not in diagnostics


def test_validated_seeded_pose_is_preserved_and_nonfinite_prior_recovers():
    world = scene()
    expected = pose(0.3)
    pixels, _ = project(world, expected, K)
    diagnostics = {}
    cv.setRNGSeed(0)
    result = estimate_pose(
        world,
        pixels,
        K,
        (640, 480),
        initial_pose=expected,
        diagnostics=diagnostics,
    )
    assert np.allclose(result[0], expected, atol=1e-5)
    assert diagnostics["pnp_method"] == "iterative_ransac"
    assert "seeded_rejection_reason" not in diagnostics
    diagnostics = {}
    result = estimate_pose(
        world,
        pixels,
        K,
        (640, 480),
        initial_pose=np.full((4, 4), np.nan),
        diagnostics=diagnostics,
    )
    assert np.allclose(result[0], expected, atol=1e-5)
    assert diagnostics["seeded_rejection_reason"] == "invalid_motion_prior"


def test_depth_scale_robust_positive_and_insufficient():
    reference = np.arange(1, 31, dtype=float)
    predicted = reference / 3
    predicted[:3] *= 10
    assert abs(fit_depth_scale(predicted, reference) - 3) < 1e-10
    assert fit_depth_scale(predicted[:5], reference[:5]) is None
    assert fit_depth_scale(np.full(30, np.nan), reference) is None


def test_dense_fusion_uses_saved_camera_units_and_skips_invalid(tmp_path):
    folder = tmp_path / "run"
    folder.mkdir()
    image = np.full((40, 60, 3), 120, np.uint8)
    cv.imwrite(str(folder / "image.png"), image)
    matrix = np.array([[30.0, 0, 30], [0, 30, 20], [0, 0, 1]])
    observations = [
        {"pixel": [x, y], "position": [(x - 30) / 30 * 6, (y - 20) / 30 * 6, 6]}
        for x in range(5, 55, 5)
        for y in range(5, 35, 5)
    ]
    run = {
        "revision": 3,
        "translation_scale": "arbitrary",
        "camera_matrix": matrix.tolist(),
        "keyframes": [
            {
                "id": 0,
                "pose": np.eye(4).tolist(),
                "image": "image.png",
                "observations": observations,
            }
        ],
    }
    (folder / "run.json").write_text(json.dumps(run))
    (folder / "preview.json").write_text(
        json.dumps(
            {
                "sparse": [],
                "trajectory": [],
                "revision": 3,
                "translation_scale": "arbitrary",
            }
        )
    )

    class Provider:
        provenance = {"model": "synthetic test fixture"}

        def predict(self, image):
            return DepthPrediction(
                np.full(image.shape[:2], 2.0),
                np.ones(image.shape[:2], bool),
                "predicted_meters",
                self.provenance,
            )

    result = fuse_reconstruction(folder, Provider(), tmp_path / "dense", voxel=0.2)
    assert result["points"] > 100
    assert result["keyframes"][0]["scale_to_map"] == 3
    assert result["translation_scale"] == "arbitrary"
    assert (tmp_path / "dense/dense.ply").exists()


def test_local_bundle_reduces_reprojection_and_preserves_origin():
    state = MapState(metric=True)
    world = scene()[:60]
    baseline = 0.54
    for k, x in enumerate([0.0, 0.3, 0.6]):
        p = pose(x)
        pixels, _ = project(world, p, K)
        estimated = p.copy()
        if k == 2:
            estimated[1, 3] = 0.05
        state.keyframes[k] = MappingKeyframe(
            k, k, estimated, pixels, np.ones((len(world), 128)), np.arange(len(world))
        )
    for i, p in enumerate(world):
        observations = {}
        for k in state.keyframes:
            pixels, z = project(p[None], pose(0.3 * k), K)
            observations[k] = Observation(
                pixels[0], float(pixels[0, 0] - K[0, 0] * baseline / z[0])
            )
        state.add_landmark(
            p + np.array([0.005, 0.002, 0]), np.ones(128), 0, observations
        )
    state.record(pose(0.6), "tracking", 2)
    report = local_bundle_adjustment(state, K, baseline, max_landmarks=60)
    assert report["applied"]
    assert report["final_cost"] < report["initial_cost"] * 0.1
    assert np.array_equal(state.keyframes[0].pose, np.eye(4))
    assert abs(state.keyframes[2].pose[1, 3]) < 0.01


@pytest.mark.parametrize("metric", [True, False])
def test_bundle_anchors_components_disconnected_from_origin(metric):
    state = MapState(metric=metric)
    world = scene()[:60]
    baseline = 0.54 if metric else 0.0
    for k in range(5):
        true_pose = pose(0.3 * k)
        pixels, _ = project(world, true_pose, K)
        estimate = true_pose.copy()
        if k == 4:
            estimate[1, 3] += 0.05
        state.keyframes[k] = MappingKeyframe(
            k, k, estimate, pixels, np.ones((len(world), 128)), np.arange(len(world))
        )
    # The optimized landmarks connect only cameras 2, 3 and 4. Fixing the
    # oldest window cameras 0/1 alone leaves this component's gauge free.
    for position in world:
        observations = {}
        for k in (2, 3, 4):
            pixels, depth = project(position[None], pose(0.3 * k), K)
            observations[k] = Observation(
                pixels[0],
                float(pixels[0, 0] - K[0, 0] * baseline / depth[0]) if metric else None,
            )
        state.add_landmark(position, np.ones(128), 2, observations)
    state.record(state.keyframes[4].pose.copy(), "tracking", 4)
    before = {k: frame.pose.copy() for k, frame in state.keyframes.items()}
    report = local_bundle_adjustment(state, K, baseline, max_landmarks=60)
    assert report["applied"]
    assert report["observation_components"] == 1
    assert report["final_cost"] < report["initial_cost"] * 0.1
    for k in (0, 1, 2) if metric else (0, 1, 2, 3):
        assert np.array_equal(state.keyframes[k].pose, before[k])
    assert abs(state.keyframes[4].pose[1, 3]) < 0.01
    if not metric:
        assert np.isclose(
            np.linalg.norm(
                state.keyframes[3].pose[:3, 3] - state.keyframes[2].pose[:3, 3]
            ),
            0.3,
        )


def test_sim3_graph_corrects_independent_loop_and_fixes_origin():
    poses = [pose(x) for x in [0.0, 1.2, 2.4]]
    edges = [
        (0, 1, np.eye(3), 1.0, np.array([1.2, 0, 0]), 1.0),
        (1, 2, np.eye(3), 1.0, np.array([1.2, 0, 0]), 1.0),
        (0, 2, np.eye(3), 1.0, np.array([2.0, 0, 0]), 5.0),
    ]
    corrected, scales = optimize_similarities(poses, edges)
    assert np.array_equal(corrected[0], poses[0])
    assert scales[0] == 1
    assert abs(corrected[-1][0, 3] - 2) < abs(poses[-1][0, 3] - 2)


def test_sim3_loop_geometry_and_false_candidate():
    points = scene()
    source = points / 1.3
    first_pixels, _ = project(points, np.eye(4), K)
    second_pixels, _ = project(source, np.eye(4), K)
    descriptors = (
        np.random.default_rng(11).normal(size=(len(points), 128)).astype(np.float32)
    )
    first = MappingKeyframe(
        0,
        0,
        np.eye(4),
        first_pixels,
        descriptors,
        np.arange(len(points)),
        depth_points=points,
    )
    second = MappingKeyframe(
        1,
        200,
        np.eye(4),
        second_pixels,
        descriptors.copy(),
        np.arange(len(points)),
        depth_points=source,
    )
    verified = verify_similarity(first, second, K)
    assert verified is not None and abs(verified["scale"] - 1.3) < 1e-5
    second.depth_points = np.random.default_rng(13).uniform(
        [-5, -4, 4], [5, 4, 14], source.shape
    )
    assert verify_similarity(first, second, K) is None
