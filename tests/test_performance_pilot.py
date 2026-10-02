import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location("pilot", Path(__file__).resolve().parents[1] / "scripts/run_performance_pilot.py")
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def report(ate=1.0, lost=(10, 11)):
    return {"frames": 300, "lost_frames": 2, "initializing_frames": 0, "loops": 1,
            "lost_intervals": [{"first_frame": lost[0], "last_frame": lost[1], "recovered": True}],
            "metrics": {"ate_rmse_m": ate}}


def test_quality_gate_rejects_new_losses_even_when_counts_match():
    assert pilot.quality_passed(report(), report())
    assert not pilot.quality_passed(report(), report(lost=(11, 12)))
    candidate = report()
    candidate["loops"] = 0
    assert not pilot.quality_passed(report(), candidate)


def test_quality_gate_requires_measured_accuracy_within_allowance():
    assert pilot.quality_passed(report(), report(1.10))
    assert not pilot.quality_passed(report(), report(1.101))
    assert not pilot.quality_passed(report(), report(None))
