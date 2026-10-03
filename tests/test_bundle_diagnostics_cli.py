import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]


def run_cli(script, *arguments):
    return subprocess.run(
        [sys.executable, str(REPO / script), *map(str, arguments)],
        cwd=REPO, capture_output=True, text=True, timeout=30,
    )


@pytest.mark.parametrize("script,extra", [
    ("src/main.py", ("--slam", "--stereo")),
    ("scripts/evaluate_shared_slam.py", ("--stereo", "--output", "unused")),
])
def test_diagnostic_directory_and_frames_must_be_supplied_together(script, extra, tmp_path):
    result = run_cli(script, *extra, "--bundle-diagnostics-frames", "127")

    assert result.returncode == 2
    assert "--bundle-diagnostics-dir and --bundle-diagnostics-frames must be supplied together" in result.stderr


@pytest.mark.parametrize("arguments", [
    ("--bundle-diagnostics-dir", "results/poses-map/bundle-diagnostics",
     "--bundle-diagnostics-frames", "127"),
    ("--slam", "--bundle-diagnostics-dir", "results/poses-map/bundle-diagnostics",
     "--bundle-diagnostics-frames", "127"),
])
def test_main_diagnostics_require_slam_and_stereo(arguments):
    result = run_cli("src/main.py", *arguments)

    assert result.returncode == 2
    assert "--bundle-diagnostics requires --slam --stereo" in result.stderr


def test_main_accepts_diagnostic_selection_when_slam_and_stereo_are_selected(tmp_path):
    output = tmp_path / "poses.txt"
    diagnostics = tmp_path / "poses-map" / "bundle-diagnostics"
    result = run_cli("src/main.py", "--slam", "--stereo", "--output", output,
                     "--bundle-diagnostics-dir", diagnostics,
                     "--bundle-diagnostics-frames", "127")

    assert result.returncode == 2
    assert "--bundle-diagnostics requires --slam --stereo" not in result.stderr
    assert "--stereo requires --data-root" in result.stderr


def test_evaluator_accepts_diagnostics_inside_owned_run_output(tmp_path):
    output = tmp_path / "run"
    result = run_cli("scripts/evaluate_shared_slam.py", "--stereo", "--output", output,
                     "--bundle-diagnostics-dir", output / "bundle-diagnostics",
                     "--bundle-diagnostics-frames", "127")

    assert result.returncode == 2
    assert "--bundle-diagnostics-dir must be inside the run output directory" not in result.stderr
    assert "--data-root is required for local input" in result.stderr


def test_evaluator_rejects_diagnostic_directory_outside_run_output(tmp_path):
    result = run_cli("scripts/evaluate_shared_slam.py", "--stereo", "--output", tmp_path / "run",
                     "--bundle-diagnostics-dir", tmp_path / "outside",
                     "--bundle-diagnostics-frames", "127")

    assert result.returncode == 2
    assert "--bundle-diagnostics-dir must be inside the run output directory" in result.stderr


def test_diagnostic_frame_selection_rejects_negative_and_duplicate_ids(tmp_path):
    output = tmp_path / "run"
    for frames, expected in ((["-1"], "nonnegative frame IDs"),
                             (["127", "127"], "must not contain duplicates")):
        result = run_cli("scripts/evaluate_shared_slam.py", "--stereo", "--output", output,
                         "--bundle-diagnostics-dir", output / "bundle-diagnostics",
                         "--bundle-diagnostics-frames", *frames)
        assert result.returncode == 2
        assert expected in result.stderr


def load_runner():
    scripts = str(REPO / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    path = REPO / "scripts" / "run_development_tests.py"
    spec = importlib.util.spec_from_file_location("diagnostics_development_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_manifest(output, *, status="complete", frames=(127,)):
    root = output / "bundle-diagnostics"
    phases = {}
    if status == "complete":
        for phase in ("prepared", "solved", "finished"):
            relative = f"frame-00127/{phase}.json"
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"frame": 127, "phase": phase}), encoding="utf-8")
            phases[phase] = {
                "status": "written", "path": relative,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    return {
        "schema_version": 1,
        "enabled": True,
        "selected_frames": list(frames),
        "max_frames": 32,
        "max_snapshots": 96,
        "frames": [{
            "frame": 127, "status": status,
            **({"reason": "tracking_not_accepted"} if status == "skipped" else {}),
            "image_size": [1242, 375], "phases": phases, "errors": [],
        }],
    }


def test_reuse_requires_finite_hash_checked_prepared_solved_finished_snapshots(tmp_path):
    runner = load_runner()
    manifest = make_manifest(tmp_path)
    identity = {"bundle_diagnostics_enabled": True,
                "bundle_diagnostics_frames": [127]}
    report = {"bundle_diagnostics": manifest, "bundle_diagnostic_errors": []}
    run = {"bundle_diagnostics": manifest, "bundle_diagnostic_errors": []}

    assert runner._bundle_diagnostics_reusable(tmp_path, identity, report, run)

    snapshot = tmp_path / "bundle-diagnostics" / "frame-00127" / "solved.json"
    snapshot.write_text('{"frame":127,"phase":"solved","cost":NaN}', encoding="utf-8")
    assert not runner._bundle_diagnostics_reusable(tmp_path, identity, report, run)


def test_reuse_requires_explicit_skip_record_and_exact_requested_frames(tmp_path):
    runner = load_runner()
    identity = {"bundle_diagnostics_enabled": True,
                "bundle_diagnostics_frames": [127]}
    manifest = make_manifest(tmp_path, status="skipped")
    assert runner._bundle_diagnostics_reusable(
        tmp_path, identity,
        {"bundle_diagnostics": manifest, "bundle_diagnostic_errors": []},
        {"bundle_diagnostics": manifest, "bundle_diagnostic_errors": []})

    incomplete = dict(manifest, selected_frames=[14])
    assert not runner._bundle_diagnostics_reusable(
        tmp_path, identity,
        {"bundle_diagnostics": incomplete, "bundle_diagnostic_errors": []},
        {"bundle_diagnostics": incomplete, "bundle_diagnostic_errors": []})


def test_default_identity_cannot_reuse_diagnostic_capture(tmp_path):
    runner = load_runner()
    manifest = make_manifest(tmp_path)
    identity = {"bundle_diagnostics_enabled": False,
                "bundle_diagnostics_frames": []}
    assert not runner._bundle_diagnostics_reusable(
        tmp_path, identity, {"bundle_diagnostics": manifest},
        {"bundle_diagnostics": manifest})
