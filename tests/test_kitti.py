import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from kitti import load_poses_txt, save_poses_txt, validate_sequence
from metrics import evaluate_trajectory, segment_errors, umeyama_alignment


def trajectory(count=902, scale=1.0):
    poses = []
    for i in range(count):
        pose = np.eye(4)
        pose[0, 3] = i * scale
        poses.append(pose)
    return poses


def test_pose_roundtrip_and_rejection(tmp_path):
    path = tmp_path / '00.txt'
    poses = trajectory(3)
    save_poses_txt(path, poses)
    assert len(path.read_text().splitlines()) == 3
    assert np.allclose(load_poses_txt(path), poses)
    path.write_text('1 2 3\n')
    with pytest.raises(ValueError, match='12 finite'):
        load_poses_txt(path)
    invalid = np.eye(4)
    invalid[0, 0] = -1
    with pytest.raises(ValueError, match='SO'):
        save_poses_txt(path, [invalid])


def test_segment_endpoint_matches_devkit_float32_accumulation():
    poses = trajectory(2200, scale=.1)
    errors = segment_errors(poses, poses, lengths=[100])
    segment = next(error for error in errors if error['first_frame'] == 1000)
    # Official float32 accumulation ends here; float64 would choose frame 2001.
    assert segment['last_frame'] == 2000
    assert segment['translation_percent'] == pytest.approx(0.)


def test_perfect_kitti_trajectory_has_zero_error():
    report = evaluate_trajectory(trajectory(), trajectory())
    assert report['segment_count'] > 0
    assert report['translation_percent'] == 0
    assert report['rotation_deg_per_m'] == 0
    assert report['ate_rmse_m'] < 1e-10


def test_scale_alignment_does_not_hide_drift():
    report = evaluate_trajectory(trajectory(), trajectory(scale=1.1), 'sim3')
    assert report['ate_rmse_m'] < 1e-9
    assert report['alignment_scale'] == pytest.approx(1 / 1.1)
    # The endpoint is strictly beyond the requested 100m, per KITTI devkit.
    error = segment_errors(trajectory(112), trajectory(112, 1.1), [100])[0]
    assert error['last_frame'] == 101
    assert error['translation_percent'] == pytest.approx(10.1)


def test_rotation_units():
    gt, est = trajectory(112), trajectory(112)
    theta = np.deg2rad(10)
    est[101][:3, :3] = [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
    assert segment_errors(gt, est, [100])[0]['rotation_deg_per_m'] == pytest.approx(0.1)


def test_umeyama_recovers_rotation_scale_translation():
    src = np.random.default_rng(42).normal(size=(100, 3))
    rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    dst = (2.5 * (rot @ src.T)).T + [1, 2, 3]
    r, scale, t = umeyama_alignment(src, dst)
    assert np.allclose(r, rot)
    assert scale == pytest.approx(2.5)
    assert np.allclose(t, [1, 2, 3])


def test_reflections_are_not_allowed():
    src = np.random.default_rng(12).normal(size=(100, 3))
    dst = src * [-1, 1, 1]
    r, scale, t = umeyama_alignment(src, dst)
    assert np.linalg.det(r) == pytest.approx(1)
    assert np.linalg.norm((scale * (r @ src.T)).T + t - dst) > 0.1


def test_short_and_incomplete_trajectories():
    report = evaluate_trajectory(trajectory(20), trajectory(20))
    assert report['segment_count'] == 0
    assert report['translation_percent'] is None
    with pytest.raises(ValueError, match='equal trajectories'):
        evaluate_trajectory(trajectory(3), trajectory(2))
    with pytest.raises(ValueError, match='stationary'):
        umeyama_alignment(np.zeros((4, 3)), np.zeros((4, 3)))


def test_dataset_checks_pairs(tmp_path):
    seq = tmp_path / 'sequences' / '00'
    (seq / 'image_0').mkdir(parents=True)
    (seq / 'image_1').mkdir()
    (seq / 'calib.txt').write_text('placeholder')
    for index in range(2):
        (seq / 'image_0' / f'{index:06}.png').touch()
        (seq / 'image_1' / f'{index + 1:06}.png').touch()
    with pytest.raises(ValueError, match='names do not match'):
        validate_sequence(tmp_path, '00', stereo=True)


def test_saved_trajectory_cli(tmp_path):
    ground, estimates, output = tmp_path / 'gt', tmp_path / 'est', tmp_path / 'out'
    save_poses_txt(ground / '00.txt', trajectory(20))
    save_poses_txt(estimates / '00.txt', trajectory(20))
    result = subprocess.run([sys.executable, 'src/eval_kitti.py', '--poses-root', str(ground),
                             '--estimates-root', str(estimates), '--output-root', str(output)],
                            text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert json.loads((output / 'summary.json').read_text())['sequences']['00']['frames'] == 20
