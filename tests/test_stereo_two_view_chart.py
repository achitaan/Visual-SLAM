"""A common world-frame change must reconstruct the same fixed-source chart."""
import numpy as np

from stereo_two_view_bundle import build_two_view_stereo_problem
from test_stereo_two_view_bundle import _scene, _pose


def test_common_se3_world_change_preserves_source_chart_and_image_objective():
    data, truth = _scene()
    original = build_two_view_stereo_problem(**data)
    world_change = _pose(phi=(-.3, .2, .1), center=(2., -1., 3.))
    world_source = world_change
    world_target_seed = world_change @ data['seed_pose']
    world_target_measurement = world_change @ truth
    world_points = data['source_points'] @ world_change[:3, :3].T + world_change[:3, 3]
    # The API owns source-camera XYZ, not arbitrary world XYZ. Reconstruct its
    # fixed-source chart; simply transforming x would introduce a false gauge.
    recovered = dict(data)
    recovered['seed_pose'] = np.linalg.inv(world_source) @ world_target_seed
    recovered['reverse_pose'] = np.linalg.inv(world_target_measurement) @ world_source
    recovered['source_points'] = (world_points - world_source[:3, 3]) @ world_source[:3, :3]
    recovered['target_points'] = (world_points - world_target_measurement[:3, 3]) @ world_target_measurement[:3, :3]
    transformed = build_two_view_stereo_problem(**recovered)
    np.testing.assert_array_equal(transformed.selected_pairs, original.selected_pairs)
    np.testing.assert_allclose(transformed.x0, original.x0, atol=2e-14, rtol=0.)
    np.testing.assert_allclose(transformed.residual(transformed.x0), original.residual(original.x0), atol=2e-12, rtol=0.)
    np.testing.assert_allclose(transformed.jacobian(transformed.x0).toarray(), original.jacobian(original.x0).toarray(), atol=2e-11, rtol=0.)
    assert transformed.observability(transformed.x0)['rank'] == 6
