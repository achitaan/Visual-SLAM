import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"


def load_identity(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "benchmark_identity_under_test", SCRIPTS / "benchmark_identity.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tiny_kitti(root, sequence="04"):
    directory = root / "sequences" / sequence
    directory.mkdir(parents=True)
    (directory / "calib.txt").write_text("P0: synthetic\n", encoding="utf-8")
    (directory / "times.txt").write_text("0.0\n0.1\n", encoding="utf-8")
    for camera in (0, 1):
        folder = directory / f"image_{camera}"
        folder.mkdir()
        for index in range(2):
            image = np.full((8, 10), index + camera, np.uint8)
            ok, encoded = cv2.imencode(".png", image)
            assert ok
            (folder / f"{index:06d}.png").write_bytes(encoded.tobytes())


def test_source_contract_is_named_versioned_and_changes_with_dependencies(tmp_path, monkeypatch):
    module = load_identity(monkeypatch)
    for folder in ("src", "scripts"):
        (tmp_path / folder).mkdir()
    (tmp_path / "src" / "model.py").write_bytes(b"x=1\n")
    for name in ("run_shared_benchmark.py", "benchmark_identity.py", "evaluate_shared_slam.py",
                 "data_preflight.py", "download_kitti_sequence.py", "run_kitti_stream.py",
                 "benchmark_telemetry.py", "test_budget.py"):
        (tmp_path / "scripts" / name).write_bytes(name.encode())
    (tmp_path / "requirements-lock.txt").write_text("numpy==1.0\n", encoding="utf-8")
    first = module.source_contract(tmp_path)
    assert first["version"] == module.CONTRACT_VERSION
    assert first["sources"]["src/model.py"]
    assert first["dependencies"]["files"]["requirements-lock.txt"]
    assert module._framed_hash((("ab", b"c"),)) != module._framed_hash((("a", b"bc"),))

    (tmp_path / "requirements-lock.txt").write_text("numpy==2.0\n", encoding="utf-8")
    second = module.source_contract(tmp_path)
    assert second["contract_sha256"] != first["contract_sha256"]


def test_resolved_config_comes_from_current_mapping_and_performance_defaults(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "shared_benchmark_under_test", SCRIPTS / "run_shared_benchmark.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    mapping, performance, threads = module._configuration(REPO)
    monkeypatch.syspath_prepend(str(REPO / "src"))
    from shared_slam import MappingConfig
    from performance import PerformanceConfig
    assert mapping == MappingConfig().__dict__
    assert performance == PerformanceConfig().__dict__
    assert threads == {"opencv": 1, "openblas": 1, "omp": 1}


def test_cli_rejects_max_frames_below_two_before_starting_work(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "shared_benchmark_args_under_test", SCRIPTS / "run_shared_benchmark.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.sys, "argv", [
        "benchmark", "--poses-root", str(tmp_path), "--max-frames", "1",
    ])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2


def test_input_identity_separates_camera_calibration_timestamps_and_reference(tmp_path, monkeypatch):
    module = load_identity(monkeypatch)
    tiny_kitti(tmp_path)
    reference = tmp_path / "poses" / "04.txt"
    reference.parent.mkdir()
    reference.write_text("ground truth A\n", encoding="utf-8")

    local = module.hash_sequence_inputs(tmp_path, "04", 2, True, reference)
    assert local["cameras"] == [0, 1]
    assert local["camera_image_sha256"]["0"] != local["camera_image_sha256"]["1"]
    assert local["calibration_sha256"] and local["timestamps_sha256"]

    archive_path = tmp_path / "data.zip"
    import zipfile
    with zipfile.ZipFile(archive_path, "w") as archive:
        for path in (tmp_path / "sequences" / "04").rglob("*"):
            if path.is_file():
                archive.write(path, Path("dataset/sequences/04") / path.relative_to(tmp_path / "sequences" / "04"))
    with zipfile.ZipFile(archive_path) as archive:
        assert module.hash_sequence_inputs(archive, "04", 2, True, reference) == local

    reference.write_text("ground truth B\n", encoding="utf-8")
    changed_reference = module.hash_sequence_inputs(tmp_path, "04", 2, True, reference)
    assert changed_reference["reference_sha256"] != local["reference_sha256"]
    assert changed_reference["image_sha256"] == local["image_sha256"]


def test_preflight_timeout_never_returns_a_partial_identity(tmp_path, monkeypatch):
    module = load_identity(monkeypatch)
    tiny_kitti(tmp_path)
    reference = tmp_path / "04.txt"
    reference.write_bytes(b"reference")

    class ExpiringBudget:
        checks = 0

        @property
        def remaining(self):
            self.checks += 1
            return 1 if self.checks < 4 else 0

    with pytest.raises(TimeoutError, match="no hash was authorized"):
        module.hash_sequence_inputs(tmp_path, "04", 2, True, reference, ExpiringBudget())


def test_ply_export_vertex_count_must_match_finite_rows(tmp_path, monkeypatch):
    module = load_identity(monkeypatch)
    ply = tmp_path / "sparse.ply"
    header = "ply\nformat ascii 1.0\nelement vertex {count}\nend_header\n"
    ply.write_text(header.format(count=1) + "1 2 3 4 5 6\n", encoding="ascii")
    assert module._finite_ply(ply)
    ply.write_text(header.format(count=2) + "1 2 3 4 5 6\n", encoding="ascii")
    assert not module._finite_ply(ply)
    ply.write_text(header.format(count=0) + "1 2 3 4 5 6\n", encoding="ascii")
    assert not module._finite_ply(ply)


def test_legacy_partial_report_retries_only_with_matching_manifest_identity(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "shared_benchmark_retry_under_test", SCRIPTS / "run_shared_benchmark.py"
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    output = tmp_path / "kitti04-stereo"
    output.mkdir()
    report_path = output / "evaluation.json"
    report_path.write_text(json.dumps({
        "sequence": "04", "dataset": "kitti", "stereo": True,
        "frames": 1, "coverage": "partial", "status": "interrupted_low_disk_space",
    }), encoding="utf-8")
    identity = {"version": 1, "sequence": "04", "mode": "stereo", "frames": 2}
    kwargs = dict(
        sequence="04", mode="stereo", frames=2, coverage="partial",
        mapping={}, performance={}, contract={"sources": {"scripts/evaluate_shared_slam.py": "hash"}},
    )
    with pytest.raises(ValueError, match="legacy"):
        runner._try_reuse(output, identity, **kwargs)
    assert runner._try_reuse(output, identity, manifest_identity=identity, **kwargs) == "incomplete"

    report_path.write_text(json.dumps({
        "sequence": "04", "dataset": "kitti", "stereo": True,
        "frames": 2, "coverage": "partial", "status": "completed",
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="legacy"):
        runner._try_reuse(output, identity, manifest_identity=identity, **kwargs)


def _load_runner(monkeypatch, name):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / "run_shared_benchmark.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner


def test_non_object_json_report_is_classified_as_incomplete(tmp_path, monkeypatch):
    runner = _load_runner(monkeypatch, "shared_benchmark_non_object_under_test")
    output = tmp_path / "report"
    output.mkdir()
    (output / "evaluation.json").write_text("[]", encoding="utf-8")
    assert runner._try_reuse(
        output, {"version": 1}, sequence="04", mode="stereo", frames=2,
        coverage="partial", mapping={}, performance={}, contract={},
    ) == "incomplete"


def test_complete_initialization_failure_resumes_and_allows_next_mode(tmp_path, monkeypatch):
    runner = _load_runner(monkeypatch, "shared_benchmark_init_failure_under_test")
    monkeypatch.setattr(runner, "artifacts_complete", lambda *args, **kwargs: True)
    monkeypatch.setattr(runner, "report_reusable", lambda *args, **kwargs: True)
    contract = {"contract_sha256": "contract", "sources": {
        "scripts/evaluate_shared_slam.py": "evaluator-hash",
    }}
    status = {"runs": []}
    identity_by_mode = {}
    for mode in ("stereo", "mono"):
        identity = runner._run_identity({}, contract, {}, "04", mode, 2, "partial",
                                        {}, {}, {}, "local_images")
        identity_by_mode[mode] = identity
        output = runner._report_path(tmp_path, "04", mode)
        output.mkdir()
        (output / "evaluation.json").write_text(json.dumps({
            "benchmark_identity": identity, "status": "initialization_failed",
            "frames": 2, "coverage": "partial", "initialization_elapsed_s": None,
            "initialized": False, "tracking_outcome": "initialization_failed",
        }), encoding="utf-8")
        runner._run_mode(
            tmp_path, tmp_path, status, lambda: None, {}, tmp_path, True,
            "04", mode, 2, "partial", tmp_path / "04.txt", contract, {}, {}, {}, {}, {},
        )

    assert [row["mode"] for row in status["runs"]] == ["stereo", "mono"]
    for row in status["runs"]:
        assert row["status"] == "completed"
        assert row["evaluation_status"] == "initialization_failed"
        assert row["experiment_complete"] is True
        assert row["initialized"] is False
        assert row["tracking_outcome"] == "initialization_failed"

    changed = {**identity_by_mode["stereo"], "inputs": {"changed": True}}
    with pytest.raises(ValueError, match="different exact identity"):
        runner._try_reuse(
            runner._report_path(tmp_path, "04", "stereo"), changed,
            sequence="04", mode="stereo", frames=2, coverage="partial",
            mapping={}, performance={}, contract=contract,
        )


def test_source_change_after_evaluator_return_invalidates_attempt(tmp_path, monkeypatch):
    from types import SimpleNamespace

    runner = _load_runner(monkeypatch, "shared_benchmark_source_drift_under_test")
    expected_contract = {"contract_sha256": "before", "sources": {}}
    monkeypatch.setattr(runner, "source_contract", lambda _repo: {
        "contract_sha256": "after", "sources": {},
    })
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0))
    root = tmp_path / "out"
    root.mkdir()
    saved = []
    status = {"runs": []}
    with pytest.raises(RuntimeError, match="sources changed during evaluation"):
        runner._run_mode(
            tmp_path, root, status, lambda: saved.append(json.loads(json.dumps(status)),),
            {}, tmp_path, True, "04", "stereo", 2, "partial", tmp_path / "04.txt",
            expected_contract, {}, {}, {}, {}, {},
        )
    assert status["runs"][0]["status"] == "invalid_source"
    assert saved[-1]["runs"][0]["status"] == "invalid_source"


@pytest.mark.parametrize(("error_type", "expected_status"), [
    (RuntimeError, "failed"), (KeyboardInterrupt, "interrupted"),
])
def test_main_persists_stopped_status_and_preserves_completed_rows(
        tmp_path, monkeypatch, error_type, expected_status):
    runner = _load_runner(monkeypatch, "shared_benchmark_failure_manifest_under_test")
    completed = {"sequence": "00", "mode": "mono", "status": "completed"}
    monkeypatch.setattr(runner.sys, "argv", [
        "benchmark", "--poses-root", str(tmp_path), "--data-root", str(tmp_path),
        "--output", str(tmp_path / "results"), "--sequences", "04", "--modes", "stereo",
    ])
    monkeypatch.setattr(runner, "resume_manifest", lambda *args, **kwargs: {"runs": [completed.copy()]})
    def fail_sequence(*args, **kwargs):
        args[2]["runs"].append({"sequence": "04", "mode": "stereo", "status": "running"})
        raise error_type("sequence exploded")
    monkeypatch.setattr(runner, "_run_sequence", fail_sequence)
    with pytest.raises(error_type, match="sequence exploded"):
        runner.main()
    manifest = next((tmp_path / "results").rglob("batch.json"))
    saved = json.loads(manifest.read_text(encoding="utf-8"))
    assert saved["status"] == expected_status
    assert saved["finished_utc"]
    assert f"{error_type.__name__}: sequence exploded" in saved["error"]
    assert saved["runs"] == [completed, {
        "sequence": "04", "mode": "stereo", "status": "interrupted",
        "error": f"{error_type.__name__}: sequence exploded",
    }]
    assert "owner" not in saved


@pytest.mark.parametrize("liveness", [True, None])
def test_active_or_unknown_owner_blocks_resume(tmp_path, monkeypatch, liveness):
    module = load_identity(monkeypatch)
    identity = {"version": 1, "config": "exact"}
    path = tmp_path / "batch.json"
    path.write_text(json.dumps({
        "identity_version": 1,
        "identity": identity,
        "status": "running",
        "owner": {"host": "test-host", "pid": 42},
        "runs": [],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="active or liveness is unknown"):
        module.resume_manifest(path, {"identity_version": 1, "identity": identity}, True,
                               current_host="test-host", probe=lambda _pid: liveness)


def test_resume_rejects_legacy_or_changed_contract_and_allows_dead_owner(tmp_path, monkeypatch):
    module = load_identity(monkeypatch)
    initial = {"identity_version": 1, "identity": {"exact": "yes"}}
    path = tmp_path / "batch.json"
    path.write_text(json.dumps({"source_sha256": "old", "runs": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="legacy"):
        module.resume_manifest(path, initial, True)

    path.write_text(json.dumps({
        **initial, "status": "running", "owner": {"host": "host", "pid": 43},
        "runs": [{"status": "completed"}],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="changed code"):
        module.resume_manifest(path, {**initial, "identity": {"exact": "no"}}, True,
                               current_host="host", probe=lambda _pid: False)
    resumed = module.resume_manifest(path, initial, True, current_host="host",
                                     probe=lambda _pid: False)
    assert resumed["interruptions"][0]["status"] == "owner_exited"
    assert resumed["runs"] == [{"status": "completed"}]


def test_report_reuse_requires_exact_complete_finite_artifacts(tmp_path, monkeypatch):
    module = load_identity(monkeypatch)
    output = tmp_path / "kitti04-stereo"
    output.mkdir()
    configuration = {"loop_mode": "live", "bundle_enabled": True}
    performance = {"retrieval": "current", "matching_backend": "cpu",
                   "cpu_optimizations": True, "profile": False}
    evaluator_source = b"exact evaluator"
    evaluator_hash = __import__("hashlib").sha256(evaluator_source).hexdigest()
    (output / "evaluator.py").write_bytes(evaluator_source)
    (output / "run.json").write_text(json.dumps({
        "configuration": configuration,
        "performance_configuration": performance,
        "matching_backend": {"requested": "cpu"},
    }), encoding="utf-8")
    (output / "preview.json").write_text(json.dumps({"trajectory": [[0, 0, 0]]}), encoding="utf-8")
    (output / "poses.txt").write_text(
        "1 0 0 0 0 1 0 0 0 0 1 0\n1 0 0 1 0 1 0 0 0 0 1 0\n",
        encoding="ascii",
    )
    (output / "sparse.ply").write_text(
        "ply\nformat ascii 1.0\nelement vertex 0\nproperty float x\nproperty float y\n"
        "property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n",
        encoding="ascii",
    )
    identity = {"version": 1, "sequence": "04", "mode": "stereo", "frames": 2}
    identity["artifacts"] = module.artifact_hashes(output)
    report = {
        "benchmark_identity": identity,
        "sequence": "04", "dataset": "kitti", "stereo": True, "frames": 2,
        "coverage": "partial", "status": "completed",
        "ground_truth_used_for_estimation": False,
        "configuration": configuration, "performance_configuration": performance,
        "opencv_threads": 1, "source_sha256": {"shared_slam.py": "source"},
        "evaluator_sha256": evaluator_hash,
        "diagnostic_overrides": {"disable_bundle": False, "loop_mode": None},
        "feature_cache": {"enabled": False},
        "matching_backend": {"requested": "cpu"},
        "input_source": "local_images", "telemetry": {"enabled": False},
    }
    (output / "evaluation.json").write_text(json.dumps(report), encoding="utf-8")
    kwargs = dict(sequence="04", mode="stereo", frames=2, coverage="partial",
                  configuration=configuration, performance=performance,
                  evaluator_sha256=evaluator_hash, source_hashes={"shared_slam.py": "source"},
                  evaluator_input_source="local_images")
    assert module.report_reusable(output, {key: value for key, value in identity.items()
                                           if key != "artifacts"}, **kwargs)

    report["frames"] = 1
    (output / "evaluation.json").write_text(json.dumps(report), encoding="utf-8")
    assert not module.report_reusable(output, identity, **kwargs)

    report["frames"] = 2
    report["diagnostic_overrides"] = {"disable_bundle": True, "loop_mode": None}
    (output / "evaluation.json").write_text(json.dumps(report), encoding="utf-8")
    assert not module.report_reusable(output, identity, **kwargs)

    report["diagnostic_overrides"] = {"disable_bundle": False, "loop_mode": None}
    (output / "preview.json").write_text('{"depth": NaN}', encoding="utf-8")
    identity["artifacts"] = module.artifact_hashes(output)
    report["benchmark_identity"] = identity
    (output / "evaluation.json").write_text(json.dumps(report), encoding="utf-8")
    assert not module.report_reusable(output, identity, **kwargs)
