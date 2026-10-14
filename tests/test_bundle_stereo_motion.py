"""Local corrections must preserve independently verified stereo motion."""
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from local_bundle import local_bundle_adjustment
from mapping_geometry import project, right_pixel
from slam_state import MapState, MappingKeyframe, Observation


@pytest.mark.parametrize('change,accepted', [('translation', False), ('rotation', False), ('common', True)])
@pytest.mark.parametrize('previous_frame', [2, 3])
def test_lower_bundle_cost_does_not_override_verified_relative_motion(monkeypatch, change, accepted, previous_frame):
    import local_bundle as module
    matrix = np.array([[250., 0, 320], [0, 250, 240], [0, 0, 1.]])
    world = np.random.default_rng(941).uniform([-2., -2., 6.], [2., 2., 12.], (24, 3))
    state = MapState(metric=True)
    proposed = {}
    for ident, x in enumerate([0., .5, 1.]):
        pose = np.eye(4); pose[0, 3] = x
        state.keyframes[ident] = MappingKeyframe(ident, 2*ident, pose.copy(),
            np.empty((0, 2)), np.empty((0, 128)), np.empty(0, int))
        candidate = pose.copy()
        if ident == 2 or (change == 'common' and ident == 1):
            if change == 'rotation':
                candidate[:3, :3] = Rotation.from_euler('y', 2., degrees=True).as_matrix()
            else:
                candidate[0, 3] += .7
        proposed[ident] = candidate
    for frame in range(5):
        pose = np.eye(4); pose[0, 3] = .25 * frame
        state.record(pose, 'tracking', frame // 2)
    for point in world:
        observations = {}
        for ident, pose in proposed.items():
            pixel, z = project(point[None], pose, matrix)
            observations[ident] = Observation(pixel[0], float(right_pixel(pixel[0, 0], z[0], 250., .2)))
        state.add_landmark(point, np.ones(128), 0, observations)
    measurement = np.linalg.inv(state.poses[previous_frame]) @ state.poses[4]
    state.add_stereo_motion(previous_frame, 4, measurement)

    def solve(_residual, initial, **_kwargs):
        value = initial.copy()
        for offset, ident in [(0, 1), (6, 2)]:
            value[offset:offset+3] = Rotation.from_matrix(proposed[ident][:3, :3]).as_rotvec()
            value[offset+3:offset+6] = proposed[ident][:3, 3]
        return SimpleNamespace(x=value, nfev=1, success=True)
    monkeypatch.setattr(module, 'least_squares', solve)
    original = np.array(state.poses)
    positions = np.array([p.position.copy() for p in state.landmarks.values()])
    report = local_bundle_adjustment(state, matrix, .2, window=3)
    assert report['affected_final_cost'] < report['affected_initial_cost'] * 1e-5
    assert report['applied'] == accepted
    if not accepted:
        assert report['reason'] == 'independent_stereo_motion_inconsistency'
        assert state.revision == 0 and state.geometry_revision == 0
        assert np.array_equal(state.poses, original)
        assert np.array_equal([p.position for p in state.landmarks.values()], positions)
    else:
        # Both world poses move beyond the disagreement limit, but their motion agrees.
        assert report['independent_stereo_motion_checks'] == 1
        assert report['max_stereo_motion_translation_error_m'] < 1e-8
        assert np.allclose(state.keyframes[2].pose, proposed[2])


@pytest.mark.parametrize('status', ['lost', 'initializing'])
def test_failed_frames_cannot_record_stereo_constraints(status):
    state = MapState(metric=True)
    state.record(np.eye(4), 'tracking')
    state.record(np.eye(4), status)
    with pytest.raises(ValueError, match='accepted metric frames'):
        state.add_stereo_motion(0, 1, np.eye(4))
    assert not state.stereo_motion


def test_stereo_constraints_own_the_measurement_and_reject_duplicates():
    state = MapState(metric=True)
    state.record(np.eye(4), 'tracking')
    state.record(np.eye(4), 'relocalized')
    measurement = np.eye(4)
    state.add_stereo_motion(0, 1, measurement)
    measurement[0, 3] = 100.
    assert state.stereo_motion[(0, 1)][0, 3] == 0.
    with pytest.raises(ValueError, match='already recorded'):
        state.add_stereo_motion(0, 1, np.eye(4))
