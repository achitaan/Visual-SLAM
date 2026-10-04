"""Atomic evaluation result writes handle brief Windows sharing locks."""
import importlib.util
import json
from pathlib import Path

import pytest


def load_test_budget():
    path = Path(__file__).resolve().parents[1] / "scripts" / "test_budget.py"
    spec = importlib.util.spec_from_file_location("atomic_result_test_budget", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_transient_replace_permission_error_retries_then_succeeds(tmp_path, monkeypatch):
    module = load_test_budget()
    output = tmp_path / "evaluation.json"
    pending = output.with_suffix(output.suffix + ".part")
    original_replace = Path.replace
    attempts = []
    now = [0.0]
    sleeps = []

    def transient_then_replace(self, target):
        attempts.append((self, target))
        if len(attempts) < 3:
            raise PermissionError("temporary sharing violation")
        return original_replace(self, target)

    def fake_sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    monkeypatch.setattr(Path, "replace", transient_then_replace)
    module.write_json(output, {"status": "complete"}, retry_timeout=0.5,
                      retry_interval=0.1, clock=lambda: now[0], sleep=fake_sleep)

    assert len(attempts) == 3
    assert all(source == pending and target == output for source, target in attempts)
    assert sum(sleeps) == pytest.approx(0.2)
    assert json.loads(output.read_text(encoding="utf-8")) == {"status": "complete"}
    assert not pending.exists()


def test_persistent_replace_lock_is_bounded_and_preserves_pending_and_target(
    tmp_path, monkeypatch
):
    module = load_test_budget()
    output = tmp_path / "evaluation.json"
    pending = output.with_suffix(output.suffix + ".part")
    output.write_text('{"status": "previous"}\n', encoding="utf-8")
    original_target = output.read_bytes()
    now = [0.0]
    sleeps = []
    attempts = []
    failure = PermissionError("persistent sharing violation")

    def always_locked(self, target):
        attempts.append((self, target))
        raise failure

    def fake_sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    monkeypatch.setattr(Path, "replace", always_locked)
    with pytest.raises(PermissionError) as caught:
        module.write_json(output, {"status": "pending"}, retry_timeout=0.5,
                          retry_interval=0.2, clock=lambda: now[0], sleep=fake_sleep)

    assert caught.value is failure
    assert len(attempts) == 4
    assert sum(sleeps) == pytest.approx(0.5)
    assert output.read_bytes() == original_target
    assert json.loads(pending.read_text(encoding="utf-8")) == {"status": "pending"}


def test_non_permission_oserror_is_not_retried(tmp_path, monkeypatch):
    module = load_test_budget()
    output = tmp_path / "evaluation.json"
    pending = output.with_suffix(output.suffix + ".part")
    output.write_text('{"status": "previous"}\n', encoding="utf-8")
    original_target = output.read_bytes()
    attempts = []

    def disk_error(self, target):
        attempts.append((self, target))
        raise OSError(28, "no space left on device")

    monkeypatch.setattr(Path, "replace", disk_error)
    with pytest.raises(OSError, match="no space left"):
        module.write_json(output, {"status": "pending"})

    assert len(attempts) == 1
    assert output.read_bytes() == original_target
    assert json.loads(pending.read_text(encoding="utf-8")) == {"status": "pending"}


def test_nonfinite_json_is_rejected_without_changing_existing_target(tmp_path, monkeypatch):
    module = load_test_budget()
    output = tmp_path / "evaluation.json"
    output.write_text('{"status": "previous"}\n', encoding="utf-8")
    original_target = output.read_bytes()
    attempts = []
    monkeypatch.setattr(Path, "replace", lambda *args: attempts.append(args))

    with pytest.raises(ValueError, match="Out of range float values"):
        module.write_json(output, {"invalid": float("nan")})

    assert attempts == []
    assert output.read_bytes() == original_target
    assert not output.with_suffix(output.suffix + ".part").exists()


def test_successful_write_remains_atomic_json_and_removes_part_file(tmp_path):
    module = load_test_budget()
    output = tmp_path / "evaluation.json"
    pending = output.with_suffix(output.suffix + ".part")

    module.write_json(output, {"status": "complete", "frames": 12})

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "status": "complete", "frames": 12
    }
    assert not pending.exists()
