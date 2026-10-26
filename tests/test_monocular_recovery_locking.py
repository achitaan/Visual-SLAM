"""Deterministic narrow recovery transaction/lock contracts, not stress tests.

Pose probes are manufactured. One controlled competing real map correction tests
RLock exclusion; it does not establish general concurrent-correction safety.
"""
import copy
from threading import Event, RLock, Thread, current_thread, local

import numpy as np
import pytest

from shared_slam import MappingConfig, SharedSlam
from slam_state import MappingKeyframe, Observation


K = np.array([[250., 0., 320.], [0., 250., 240.], [0., 0., 1.]])
SIZE = (640, 480)
PIXELS = np.array([[100., 100.], [400., 350.]], np.float32)
DESCRIPTORS = np.arange(2*128, dtype=np.float32).reshape(2, 128)


@pytest.fixture
def slam():
    obj = SharedSlam(K, config=MappingConfig(
        loop_mode='off', bundle_enabled=False, monocular_recovery_fallback=True,
    ))
    for ident in (0, 1):
        pose = np.eye(4)
        pose[0, 3] = float(ident)
        lid = obj.map.add_landmark(
            [float(ident), 0., 5.], DESCRIPTORS[ident], ident,
            {ident: Observation(PIXELS[ident].copy())},
        )
        obj.map.landmarks[lid].misses = 7
        obj.map.keyframes[ident] = MappingKeyframe(
            ident, ident, pose, PIXELS[ident:ident+1].copy(),
            DESCRIPTORS[ident:ident+1].copy(), np.array([lid]),
        )
    obj.accepted_tracks = [('initial', np.array([1., 2.]))]
    try:
        yield obj
    finally:
        obj.close(finish=False)


@pytest.mark.parametrize('outcome', ['failure', 'exception'])
def test_rollback_does_not_overwrite_replaced_landmark_object(slam, monkeypatch, outcome):
    original = slam.map.landmarks[0]
    untouched = slam.map.landmarks[1]
    initial_tracks = slam.accepted_tracks
    replacement = copy.deepcopy(original)
    replacement.misses = 83
    replacement.position += np.array([2., 0., 0.])

    def probe(*args, **kwargs):
        # This is a deliberately reentrant mutation, not a concurrent-map
        # correction. It isolates the object-identity guard in rollback.
        with slam.map.lock:
            original.misses = 91
            untouched.misses = 92
            slam.map.landmarks[original.id] = replacement
            slam.accepted_tracks = [('probe', np.array([4., 5.]))]
        if outcome == 'exception':
            raise RuntimeError('manufactured replacement failure')
        return None, {'num_matches': 20}

    monkeypatch.setattr(slam, '_track', probe)
    if outcome == 'exception':
        with pytest.raises(RuntimeError, match='manufactured replacement failure'):
            slam._recover_ranked(PIXELS, DESCRIPTORS, SIZE, None, [(20, 0)])
    else:
        result, stats = slam._recover_ranked(PIXELS, DESCRIPTORS, SIZE, None, [(20, 0)])
        assert result is None and stats == {}
    assert slam.map.landmarks[0] is replacement
    assert replacement.misses == 83
    assert np.array_equal(replacement.position, [2., 0., 5.])
    assert original.misses == 91  # detached object is not mistaken for current ID 0
    assert untouched.misses == 7
    assert slam.accepted_tracks is initial_tracks


class _ObservedRLock:
    """Real RLock with observable acquisition attempts, without polling sleeps."""
    def __init__(self, events, before_owner_release):
        self.mutex = RLock()
        self.depth = local()
        self.owner = current_thread()
        self.events = events
        self.before_owner_release = before_owner_release
        self.failed_nonblocking = Event()
        self.attempting = Event()
        self.correction_entered = Event()

    def held_here(self):
        return getattr(self.depth, 'value', 0) > 0

    def __enter__(self):
        depth = getattr(self.depth, 'value', 0)
        if current_thread() is not self.owner and depth == 0:
            # The actual nonblocking failure establishes contention while the
            # owner is in the probe. Mere absence of a signal would not.
            already_acquired = self.mutex.acquire(blocking=False)
            if not already_acquired:
                self.failed_nonblocking.set()
            self.attempting.set()
            if not already_acquired:
                self.mutex.acquire()
            self.events.append('correction_entered')
            self.correction_entered.set()
        else:
            self.mutex.acquire()
            if current_thread() is self.owner and depth == 0:
                self.events.append('recovery_entered')
        self.depth.value = depth + 1
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        depth = self.depth.value
        try:
            if current_thread() is self.owner and depth == 1:
                self.before_owner_release()
                self.events.append('recovery_commit_before_release')
        finally:
            self.depth.value = depth - 1
            self.mutex.release()


class _SnapshotObservedDict(dict):
    def __init__(self, values, lock, events):
        super().__init__(values)
        self.lock, self.events = lock, events

    def items(self):
        if current_thread() is self.lock.owner:
            assert self.lock.held_here(), 'landmark snapshot occurred outside map lock'
            self.events.append('snapshot_under_lock')
        return super().items()


def test_competing_correction_waits_until_winner_commit_and_lock_release(slam, monkeypatch):
    events, corrections, errors = [], [], []
    initial_tracks = slam.accepted_tracks
    thread = None

    def before_owner_release():
        assert not lock.correction_entered.is_set()
        assert all(landmark.misses == 10 for landmark in slam.map.landmarks.values())
        assert slam.accepted_tracks[0][0] == 0  # winner, not the final losing probe
        assert slam.map.geometry_revision == 0

    lock = _ObservedRLock(events, before_owner_release)
    slam.map.lock = lock
    slam.map.landmarks = _SnapshotObservedDict(slam.map.landmarks, lock, events)
    corrected = {ident: frame.pose.copy() for ident, frame in slam.map.keyframes.items()}
    corrected[1][0, 3] += 1.

    def correction():
        try:
            with lock:
                # Winner commit must already be visible before this real
                # correction is permitted to inspect or mutate the map.
                assert all(landmark.misses == 10 for landmark in slam.map.landmarks.values())
                assert slam.accepted_tracks[0][0] == 0
                assert slam.map.apply_corrections(0, corrected)
                corrections.append(True)
        except BaseException as error:
            errors.append(error)

    def probe(*args, candidate_ids, **kwargs):
        nonlocal thread
        assert lock.held_here()
        assert 'snapshot_under_lock' in events
        assert all(landmark.misses == 7 for landmark in slam.map.landmarks.values())
        assert slam.accepted_tracks is initial_tracks
        ident = next(iter(candidate_ids))
        if thread is None:
            thread = Thread(target=correction, name='controlled-map-correction', daemon=True)
            thread.start()
            assert lock.attempting.wait(timeout=2), 'correction did not attempt acquisition'
            assert lock.failed_nonblocking.is_set(), 'real lock did not exclude correction'
        assert not lock.correction_entered.is_set()
        assert slam.map.geometry_revision == 0
        for landmark in slam.map.landmarks.values():
            landmark.misses = 10 + ident
        slam.accepted_tracks = [(ident, np.array([float(ident), 0.]))]
        events.append('probe_' + str(ident))
        return (np.eye(4), {}), {'num_inliers': 30 if ident == 0 else 25}

    monkeypatch.setattr(slam, '_track', probe)
    try:
        result, stats = slam._recover_ranked(
            PIXELS, DESCRIPTORS, SIZE, None, [(30, 0), (29, 1)],
        )
    finally:
        if thread is not None:
            thread.join(timeout=2)
    assert thread is not None and not thread.is_alive()
    assert not errors, repr(errors)
    assert result is not None and stats['relocalization_keyframe'] == 0
    assert corrections == [True]
    assert events == [
        'recovery_entered', 'snapshot_under_lock', 'probe_0', 'probe_1',
        'recovery_commit_before_release', 'correction_entered',
    ]
    assert slam.map.revision == slam.map.geometry_revision == 1
    assert np.array_equal(slam.map.keyframes[1].pose, corrected[1])
    assert np.array_equal(slam.map.landmarks[1].position, [2., 0., 5.])
    assert all(landmark.misses == 10 for landmark in slam.map.landmarks.values())
