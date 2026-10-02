"""Low-disk runs must retain processed frames without claiming full coverage."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest


def evaluator():
    path = Path(__file__).resolve().parents[1] / "scripts/evaluate_shared_slam.py"
    spec = importlib.util.spec_from_file_location("storage_evaluator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_low_space_refuses_to_start_tracking(tmp_path, monkeypatch):
    module = evaluator()
    monkeypatch.setattr(module.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    monkeypatch.setattr(
        module, "validate_sequence", lambda *_: pytest.fail("Read input")
    )
    monkeypatch.setattr(
        module.sys,
        "argv",
        ["evaluate", "--data-root", str(tmp_path), "--output", str(tmp_path / "run")],
    )
    with pytest.raises(OSError, match="Insufficient space"):
        module.main()
    assert not (tmp_path / "run/evaluation.json").exists()


def test_low_space_exports_only_processed_frames(tmp_path, monkeypatch):
    module = evaluator()
    capacity = iter([2 * 1024**3, 2 * 1024**3, 500 * 1024**2])
    monkeypatch.setattr(
        module.shutil, "disk_usage", lambda _: SimpleNamespace(free=next(capacity))
    )
    paths = [tmp_path / f"{i}.png" for i in range(120)]

    class Images:
        def __len__(self):
            return len(paths)

        def __getitem__(self, index):
            return np.zeros((10, 10), np.uint8)

    images = Images()
    images.paths = paths
    monkeypatch.setattr(module, "validate_sequence", lambda *_: tmp_path)
    monkeypatch.setattr(
        module,
        "VisualOdometry",
        lambda *_, **__: SimpleNamespace(Images=images, K=np.eye(3)),
    )

    class Tracker:
        def __init__(self, *_, **__):
            self.map = SimpleNamespace(
                poses=[], statuses=[], keyframes={}, landmarks={}
            )
            self.config = SimpleNamespace()
            self.loop_worker = SimpleNamespace(verified={}, events=[])
            self.diagnostics = []
            self.bundle_reports = []
            self.closed = False
            from stage_profile import StageProfile
            self.profile = StageProfile()
            self.matcher = SimpleNamespace(metadata=lambda: {'requested': 'cpu', 'cuda_calls': 0})

        def process(self, index, image, right):
            self.map.poses.append(np.eye(4))
            self.map.statuses.append("tracking")
            return None, {"tracking_ok": True, "state": "tracking"}

        def close(self, finish=True):
            self.closed = True

    exported = []

    def export(tracker, folder, saved_paths, **_):
        assert tracker.closed
        folder.mkdir(exist_ok=True)
        exported.extend(saved_paths)

    monkeypatch.setattr(module, "SharedSlam", Tracker)
    monkeypatch.setattr(module, "export_run", export)
    output = tmp_path / "run"
    monkeypatch.setattr(
        module.sys,
        "argv",
        ["evaluate", "--data-root", str(tmp_path), "--output", str(output)],
    )
    module.main()
    report = json.loads((output / "evaluation.json").read_text())
    assert exported == paths[:50]
    assert report["frames"] == 50
    assert report["states"] == {"tracking": 50}
    assert report["coverage"] == "partial"
    assert report["status"] == "interrupted_low_disk_space"
    assert report["storage_guard"]["interruption"]["next_frame"] == 50


@pytest.mark.parametrize("interrupted", [False, True])
def test_local_batch_preserves_inputs_and_does_not_complete_interruption(
    tmp_path, monkeypatch, interrupted
):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "local_batch", scripts / "run_shared_benchmark.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = tmp_path / "borrowed-data"
    sequence = data / "sequences" / "04"
    sequence.mkdir(parents=True)
    (sequence / "calib.txt").write_text("P0: synthetic calibration\n", encoding="utf-8")
    (sequence / "times.txt").write_text("0.0\n0.1\n", encoding="utf-8")
    for camera in (0, 1):
        image_folder = sequence / f"image_{camera}"
        image_folder.mkdir()
        for index in range(2):
            ok, image = cv2.imencode(".png", np.full((8, 10), index + camera, np.uint8))
            assert ok
            (image_folder / f"{index:06d}.png").write_bytes(image.tobytes())
    sentinel = data / "image.png"
    sentinel.write_bytes(b"retain original input")
    scratch = tmp_path / "scratch"
    output = tmp_path / "results"
    poses = tmp_path / "poses"
    poses.mkdir()
    reference = poses / "04.txt"
    reference.write_text(
        "1 0 0 0 0 1 0 0 0 0 1 0\n1 0 0 1 0 1 0 0 0 0 1 0\n",
        encoding="ascii",
    )
    repo = Path(__file__).resolve().parents[1]
    contract = module.source_contract(repo)
    mapping, performance, _threads = module._configuration(repo)
    monkeypatch.setattr(
        module, "RangeFile", lambda *_: pytest.fail("Local input downloaded")
    )

    def run(command, **_):
        if Path(command[1]).name == "evaluate_shared_slam.py":
            assert command[command.index("--data-root") + 1] == str(data.resolve())
            run_output = Path(command[command.index("--output") + 1])
            evaluator_bytes = Path(command[1]).read_bytes()
            (run_output / "evaluator.py").write_bytes(evaluator_bytes)
            (run_output / "run.json").write_text(json.dumps({
                "configuration": mapping,
                "performance_configuration": performance,
                "matching_backend": {"requested": "cpu"},
            }), encoding="utf-8")
            (run_output / "preview.json").write_text(
                json.dumps({"trajectory": [[0, 0, 0]]}), encoding="utf-8")
            (run_output / "poses.txt").write_text(
                "1 0 0 0 0 1 0 0 0 0 1 0\n1 0 0 1 0 1 0 0 0 0 1 0\n",
                encoding="ascii",
            )
            (run_output / "sparse.ply").write_text(
                "ply\nformat ascii 1.0\nelement vertex 0\nproperty float x\nproperty float y\n"
                "property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n",
                encoding="ascii",
            )
            (run_output / "evaluation.json").write_text(
                json.dumps(
                    {
                        "sequence": "04", "dataset": "kitti", "stereo": True,
                        "frames": 2, "coverage": "partial",
                        "status": "interrupted_low_disk_space" if interrupted else "completed",
                        "ground_truth_used_for_estimation": False,
                        "configuration": mapping,
                        "performance_configuration": performance,
                        "opencv_threads": 1,
                        "source_sha256": module._report_source_hashes(contract),
                        "evaluator_sha256": contract["sources"]["scripts/evaluate_shared_slam.py"],
                        "diagnostic_overrides": {"disable_bundle": False, "loop_mode": None},
                        "feature_cache": {"enabled": False},
                        "matching_backend": {"requested": "cpu"},
                        "input_source": "local_images",
                        "telemetry": {"enabled": False},
                    }
                )
            )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", run)
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "batch",
            "--data-root",
            str(data),
            "--poses-root",
            str(poses),
            "--sequences",
            "04",
            "--modes",
            "stereo",
            "--output",
            str(output),
            "--cache-root",
            str(scratch),
            "--max-frames",
            "2",
        ],
    )
    if interrupted:
        with pytest.raises(RuntimeError, match="failed exact reuse validation"):
            module.main()
    else:
        module.main()
    batch = json.loads(next(output.rglob("batch.json")).read_text())
    assert batch["runs"][0]["status"] == ("invalid_report" if interrupted else "completed")
    assert batch["runs"][0]["coverage"] == "partial"
    assert batch["runs"][0]["frames_expected"] == 2
    assert sentinel.read_bytes() == b"retain original input"
    assert not (data / "owner.json").exists()
    assert not scratch.exists()
