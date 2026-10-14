"""A completed initialization failure must still produce an honest overview."""
import importlib.util
import json
from pathlib import Path


def test_uninitialized_monocular_plot_has_no_alignment_or_keyframe_lookup(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / 'scripts/plot_shared_slam.py'
    spec = importlib.util.spec_from_file_location('plot_initialization_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / 'run.json').write_text(json.dumps({
        'translation_scale': 'arbitrary', 'revision': 0, 'keyframes': [],
        'tracking': [{'frame': i, 'state': 'initializing', 'tracking_ok': False}
                     for i in range(2)],
    }))
    (tmp_path / 'preview.json').write_text(json.dumps({
        'trajectory': [[0, 0, 0], [0, 0, 0]], 'sparse': [], 'dense': [],
    }))
    reference = tmp_path / 'poses.txt'
    reference.write_text('1 0 0 0 0 1 0 0 0 0 1 0\n' * 2)
    monkeypatch.setattr(module, 'umeyama_alignment',
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError('Uninitialized scale fit')))
    monkeypatch.setattr(module.sys, 'argv', ['plot', '--run', str(tmp_path), '--reference', str(reference)])
    labels = []
    save = module.plt.Figure.savefig

    def capture(figure, *args, **kwargs):
        labels.extend(text.get_text() for axis in figure.axes for text in axis.texts)
        return save(figure, *args, **kwargs)

    monkeypatch.setattr(module.plt.Figure, 'savefig', capture)
    module.main()
    assert (tmp_path / 'overview.png').is_file()
    assert 'Accuracy unavailable: initialization failed' in labels
    assert 'No initialized keyframe image' in labels
