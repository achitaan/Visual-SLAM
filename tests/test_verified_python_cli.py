"""The original application must expose the already-verified Python profile."""
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
from shared_slam import MappingConfig

ROOT = Path(__file__).resolve().parents[1]


def test_application_config_matches_confirmed_original_python_case():
    spec = importlib.util.spec_from_file_location('original_verified_main', ROOT / 'src/main.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = json.loads((ROOT / 'docs/benchmark/original-python-verified.json').read_text())['configuration']
    actual = asdict(module._mapping_config_from_args(SimpleNamespace(**expected)))
    assert actual == expected


@pytest.mark.parametrize('flag,message', [
    ('--stereo-descriptor-association-retry', 'requires --slam --stereo --stereo-pose-arbitration'),
    ('--stereo-verified-reference-retry', 'requires --slam --stereo'),
])
def test_stereo_profile_options_reject_incompatible_modes(flag, message):
    result = subprocess.run([sys.executable, '-B', str(ROOT / 'src/main.py'), flag],
                            cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    assert message in result.stderr


def test_main_cli_preserves_improved_stereo_preset_and_overrides():
    import cv2 as cv
    import numpy as np
    from shared_slam import SharedSlam, StereoCamera

    spec = importlib.util.spec_from_file_location('improved_preset_main', ROOT / 'src/main.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    _, args = module.parse_main_args(['--slam', '--stereo', '--preset', 'improved_stereo'])
    cfg = module._mapping_config_from_args(args)
    expected = MappingConfig.from_preset('improved_stereo', stereo=True)
    assert cfg == expected
    assert cfg.enhancement_dict() == expected.enhancement_dict()
    assert cfg.use_grid_bucketing is True
    assert cfg.use_analytic_ba_jacobians is True
    assert cfg.use_incremental_map_index is True
    assert cfg.max_live_loop_keyframes == 0
    assert cfg.use_weighted_pgo is True
    assert cfg.use_vlad_loop_retrieval is True
    assert cfg.loop_mode == 'live'

    # Explicit CLI overrides must layer on top of the preset without dropping enhancement flags
    _, override_args = module.parse_main_args([
        '--slam', '--stereo', '--preset', 'improved_stereo',
        '--features', '1200', '--loop-mode', 'off',
    ])
    override_cfg = module._mapping_config_from_args(override_args)
    assert override_cfg.features == 1200
    assert override_cfg.loop_mode == 'off'
    assert override_cfg.use_grid_bucketing is True
    assert override_cfg.use_analytic_ba_jacobians is True
    assert override_cfg.max_live_loop_keyframes == 0

    # Verify SharedSlam receives the full preset configuration end-to-end
    K = np.array([[700.0, 0.0, 620.0], [0.0, 700.0, 188.0], [0.0, 0.0, 1.0]])
    stereo = StereoCamera(cv.StereoSGBM_create(numDisparities=96, blockSize=5), np.eye(4), 0.54)
    slam = SharedSlam(K, stereo=stereo, config=cfg)
    try:
        assert slam._grid_detector is not None
        assert slam._fast_stereo_sgbm is not None
        assert slam.loop_worker.max_keyframes == 0
        assert slam.loop_worker.use_weighted_pgo is True
        assert slam.loop_worker.use_vlad_retrieval is True
    finally:
        slam.close()
