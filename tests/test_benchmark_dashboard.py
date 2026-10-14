"""Display observers must preserve estimator inputs and current-run identity."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from slam_state import MapState


def script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_observer_samples_actual_map_without_mutation(tmp_path):
    module = script("benchmark_telemetry")
    state = MapState(metric=False)
    for i in range(5100):
        pose = np.eye(4)
        pose[0, 3] = i
        state.poses.append(pose)
    state.landmarks = {
        i: SimpleNamespace(position=np.array([i, 1, 2.0])) for i in range(4100)
    }
    state.revision = 7
    slam = SimpleNamespace(
        map=state, map_state=lambda: dict(keyframes=3, map_points=4100, revision=7)
    )
    target = tmp_path / "frame.json"
    writer = module.SnapshotWriter(target, "04", "test-mono", 5100, interval=1000)
    writer.publish(
        slam,
        5099,
        np.zeros((500, 1200), np.uint8),
        {"state": "tracking", "feature_points": [[600, 100]], "inlier_mask": [True]},
    )
    payload = json.loads(target.read_text())
    assert payload["pose_T_wc"][0][3] == 5099
    assert payload["trajectory"][0] == [0, 0, 0]
    assert payload["trajectory"][-1] == [5099, 0, 0]
    assert len(payload["trajectory"]) <= 5000 and len(payload["map_points"]) <= 2000
    assert payload["features"][0]["x"] == 480
    assert payload["image"]["width"] == 960
    assert payload["map"]["revision"] == state.revision == 7
    assert payload["expected_pose_T_wc"] is None
    assert payload["translation_scale"] == "arbitrary"
    first = target.read_bytes()
    writer.publish(slam, 5099, np.zeros((10, 10), np.uint8), {})
    assert target.read_bytes() == first
    # A final corrected snapshot replaces the entire displayed path.
    state.poses[-1][0, 3] = 5000
    writer.publish(slam, 5099, np.zeros((10, 10), np.uint8), {}, force=True)
    assert json.loads(target.read_text())["trajectory"][-1][0] == 5000


def test_snapshot_io_failure_does_not_fail_tracking(tmp_path):
    module = script("benchmark_telemetry")
    state = MapState()
    state.poses.append(np.eye(4))
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    writer = module.SnapshotWriter(blocker / "frame.json", "04", "run", 1)
    writer.publish(
        SimpleNamespace(map=state, map_state=lambda: {}),
        0,
        np.zeros((10, 10), np.uint8),
        {},
    )
    assert writer.error
    assert len(state.poses) == 1


def test_bridge_rejects_snapshot_from_another_run(tmp_path, monkeypatch):
    module = script("benchmark_dashboard")
    monkeypatch.setattr(module, "REPO", tmp_path)
    root = tmp_path / "results/batch"
    root.mkdir(parents=True)
    current = tmp_path / "current.json"
    current.write_text(
        json.dumps(
            dict(batch_root="results/batch", sequence_order=["01"], modes=["stereo"])
        )
    )
    (root / "batch.json").write_text(
        json.dumps(
            dict(
                runs=[
                    dict(
                        sequence="01",
                        mode="stereo",
                        status="running",
                        output="kitti01-stereo",
                    )
                ]
            )
        )
    )
    (root / "01-stereo.log").write_text("01 51/1101 tracking landmarks=23\n")
    feed = root / "dashboard-frame.json"
    feed.write_text(json.dumps(dict(schema_version=1, run_id="kitti04-mono")))
    status, frame = module.batch_snapshot(current, True)
    assert frame is None and not status["stream_available"]
    assert status["active"]["progress"]["frames"] == 51
    assert "pose_T_wc" not in status
    feed.write_text(json.dumps(dict(schema_version=1, run_id="kitti01-stereo")))
    status, frame = module.batch_snapshot(current, False)
    assert frame["run_id"] == "kitti01-stereo"
    assert not status["running"]

    batch = json.loads((root / "batch.json").read_text())
    batch["paused"] = True
    batch["runs"][0]["status"] = "interrupted_user_pause"
    (root / "batch.json").write_text(json.dumps(batch))
    status, frame = module.batch_snapshot(current, False)
    assert status["paused"] and not status["running"]
    assert status["active"] is None
    assert frame["run_id"] == "kitti01-stereo"


def test_loaded_evaluator_provenance_changes_no_numerical_results(tmp_path):
    import hashlib

    module = script("benchmark_dashboard")
    source = b"# evaluator loaded by the running worker\n"
    (tmp_path / "before.py").write_bytes(source)
    expected = hashlib.sha256(source).hexdigest()
    (tmp_path / "evaluator-provenance.json").write_text(
        json.dumps(
            dict(
                snapshot="before.py",
                actual_evaluator_sha256=expected,
                loaded_before_edit=[dict(output="run")],
            )
        )
    )
    folder = tmp_path / "run"
    folder.mkdir()
    report_path = folder / "evaluation.json"
    original = dict(
        evaluator_sha256="hash-of-edited-file",
        source_sha256={"tracker.py": "unchanged"},
        metrics={"ate_rmse_m": 1.23},
        frames=1101,
    )
    report_path.write_text(json.dumps(original))
    module.retain_active_evaluator_provenance(tmp_path)
    report = json.loads(report_path.read_text())
    assert report["metrics"] == original["metrics"]
    assert report["source_sha256"] == original["source_sha256"]
    assert report["frames"] == 1101
    assert report["evaluator_sha256"] == expected
    assert (
        report["evaluator_sha256_reported_at_shutdown"] == original["evaluator_sha256"]
    )
    assert (folder / "evaluator.py").read_bytes() == source
    first = report_path.read_bytes()
    module.retain_active_evaluator_provenance(tmp_path)
    assert report_path.read_bytes() == first
