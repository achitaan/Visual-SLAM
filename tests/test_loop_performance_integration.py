from concurrent.futures import Future
from threading import RLock
from types import SimpleNamespace

import numpy as np

from keyframe_retrieval import KeyframeRetrieval
from live_loops import LiveLoopWorker
from performance import PerformanceConfig
from slam_state import MappingKeyframe


K = np.array([[250.0, 0, 320], [0, 250, 240], [0, 0, 1]])


class CaptureExecutor:
    def __init__(self):
        self.args = None
        self.shutdown_called = False

    def submit(self, function, *args):
        self.args = (function, *args)
        return Future()

    def shutdown(self, **_kwargs):
        self.shutdown_called = True


def keyframe(ident, frame, *, read_only_features=False, retrieval_count=3):
    rng = np.random.default_rng(ident + 12)
    pixels = rng.uniform([20, 20], [600, 440], (12, 2)).astype(np.float32)
    descriptors = rng.normal(size=(12, 128)).astype(np.float32)
    points = rng.uniform([-2, -2, 4], [2, 2, 12], (12, 3)).astype(np.float32)
    image = np.full((8, 9), ident, dtype=np.uint8)
    result = MappingKeyframe(
        ident,
        frame,
        np.eye(4),
        pixels,
        descriptors,
        np.arange(12),
        image=image,
        depth_points=points,
        image_size=(640, 480),
    )
    result.retrieval_descriptors = descriptors[:retrieval_count]
    if read_only_features:
        for array in (pixels, descriptors, image, result.retrieval_descriptors):
            array.setflags(write=False)
    return result


def state_with_keyframes():
    frames = [keyframe(0, 0, read_only_features=True), keyframe(1, 60), keyframe(2, 200)]
    return SimpleNamespace(
        lock=RLock(),
        keyframes={frame.id: frame for frame in frames},
        revision=4,
        geometry_revision=2,
        poses=[np.eye(4) for _ in frames],
    )


def test_snapshot_copies_mutable_arrays_and_geometry_but_shares_readonly_features():
    state = state_with_keyframes()
    frozen, mutable = state.keyframes[0], state.keyframes[1]
    worker = LiveLoopWorker(K, True, performance=PerformanceConfig())
    executor = CaptureExecutor()
    worker.executor = executor

    worker.schedule(state)
    snapshot = executor.args[1]
    frozen_copy = snapshot[0]
    mutable_copy = snapshot[1]

    assert frozen_copy.descriptors is frozen.descriptors
    assert frozen_copy.pixels is frozen.pixels
    assert frozen_copy.image is frozen.image
    assert frozen_copy.retrieval_descriptors is frozen.retrieval_descriptors
    assert mutable_copy.descriptors is not mutable.descriptors
    assert mutable_copy.pixels is not mutable.pixels
    assert mutable_copy.image is not mutable.image
    assert mutable_copy.retrieval_descriptors is not mutable.retrieval_descriptors
    for name in ("pose", "landmark_ids", "depth_points"):
        assert not np.shares_memory(getattr(frozen_copy, name), getattr(frozen, name))
        assert not np.shares_memory(getattr(mutable_copy, name), getattr(mutable, name))

    mutable.descriptors[0, 0] += 100
    mutable.depth_points[0, 0] += 100
    assert mutable_copy.descriptors[0, 0] != mutable.descriptors[0, 0]
    assert mutable_copy.depth_points[0, 0] != mutable.depth_points[0, 0]
    assert frozen.descriptors.flags.writeable is False
    assert mutable.descriptors.flags.writeable is True
    assert worker.snapshot_geometry_revision == state.geometry_revision
    np.testing.assert_array_equal(worker.snapshot_poses[0], frozen.pose)
    executor.shutdown()


def test_off_mode_does_not_schedule_and_current_retrieval_keeps_sketch_api(monkeypatch):
    state = state_with_keyframes()
    off = LiveLoopWorker(K, True, mode="off")
    off_executor = CaptureExecutor()
    off.executor = off_executor
    off.schedule(state)
    assert off_executor.args is None
    assert off.future is None
    off.close(state, finish=False)
    assert off_executor.shutdown_called

    worker = LiveLoopWorker(K, True)
    assert worker.retrieval_mode == "current"
    assert isinstance(worker.retrieval, KeyframeRetrieval)
    assert worker.retrieval_index is None
    queried = {}

    def query(_descriptors, count, allowed=None):
        queried.update(count=count, allowed=list(allowed))
        return [0]

    monkeypatch.setattr(worker.retrieval, "query", query)
    monkeypatch.setattr("live_loops.verify_loop", lambda *_: None)
    worker.matcher = lambda *_: np.array([[0, 0], [1, 1]])
    result = worker._solve(state.keyframes, state.revision, {})
    assert queried == {"count": 8, "allowed": [0]}
    assert [attempt["first_keyframe"] for attempt in result["attempts"]] == [0]
    assert result["attempts"][0]["geometrically_verified"] is False
    worker.executor.shutdown(wait=True)


def test_offline_mode_retains_verified_loops_until_finalization(monkeypatch):
    state = state_with_keyframes()
    worker = LiveLoopWorker(K, True, mode="offline")
    monkeypatch.setattr(worker.retrieval, "query", lambda *_args, **_kwargs: [0])
    monkeypatch.setattr(
        "live_loops.verify_loop", lambda *_: {"measurement": np.eye(4)}
    )
    worker.matcher = lambda *_: np.array([[0, 0], [1, 1]])

    result = worker._solve(state.keyframes, state.revision, {})
    assert result["correction"] is None
    assert set(result["loops"]) == {(0, 2)}
    np.testing.assert_array_equal(result["loops"][(0, 2)]["measurement"], np.eye(4))
    worker.future = Future()
    worker.future.set_result(result)
    assert worker.poll(state) is False
    assert (0, 2) in worker.verified
    worker.executor.shutdown(wait=True)


def test_indexed_proposals_use_detected_view_and_false_match_is_not_a_loop(monkeypatch):
    frames = [keyframe(0, 0, retrieval_count=3), keyframe(1, 200, retrieval_count=5),
              keyframe(2, 300, retrieval_count=7)]
    keyframes = {frame.id: frame for frame in frames}
    worker = LiveLoopWorker(K, True, performance=PerformanceConfig(retrieval="indexed"))

    class IndexStub:
        def __init__(self):
            self.upserts = []
            self.query_args = None

        def upsert(self, ident, frame, descriptors):
            self.upserts.append((ident, frame, len(descriptors)))

        def query(self, descriptors, eligible, limit=20):
            self.query_args = (len(descriptors), list(eligible), limit)
            return [0]

    index = IndexStub()
    worker.retrieval_index = index
    worker.matcher = lambda *_: np.array([[0, 0], [1, 1], [2, 2]])
    verification_calls = []

    def reject_geometry(*_):
        verification_calls.append(True)
        return None

    monkeypatch.setattr("live_loops.verify_loop", reject_geometry)
    result = worker._solve(keyframes, 5, {})

    assert index.query_args == (7, [0], 20)
    assert [item[2] for item in index.upserts] == [3, 5, 7]
    assert verification_calls == [True]
    assert result["attempts"] == [
        {
            "first_keyframe": 0,
            "second_keyframe": 2,
            "appearance_matches": 3,
            "geometrically_verified": False,
        }
    ]
    assert result["loops"] == {}
    assert result["correction"] is None
    assert worker.verified == {}
    worker.executor.shutdown(wait=True)
