import pytest

import windows_affinity as affinity


def _core(mask, *, group=0, efficiency=0):
    size = affinity._CORE_HEADER.size + affinity._GROUP_AFFINITY.size
    return (affinity._CORE_HEADER.pack(0, size, 1, efficiency, 1)
            + affinity._GROUP_AFFINITY.pack(mask, group))


def test_core_masks_respect_process_availability_and_pair_smt_siblings():
    topology = _core(48) + _core(3) + _core(12) + _core(192)
    assert affinity._physical_core_masks(topology, 0b11001110) == [2, 12, 192]


@pytest.mark.parametrize("topology", [_core(3, group=1), _core(3) + _core(12, efficiency=1), b"\xff"])
def test_core_masks_decline_unsupported_topology(topology):
    with pytest.raises(ValueError):
        affinity._physical_core_masks(topology, 255)


def test_default_and_non_windows_placements_do_not_call_win32(monkeypatch):
    def unexpected_api():
        raise AssertionError("Windows API should only load on supported hosts when requested")

    monkeypatch.setattr(affinity, "_windows_api", unexpected_api)
    monkeypatch.setattr(affinity.sys, "platform", "linux")
    default = affinity.WindowsCorePlacement("default")
    default.bind("tracking")
    assert default.report["fallback_reason"] is None
    split = affinity.WindowsCorePlacement("split")
    split.bind("tracking")
    assert split.report["fallback_reason"] == "Core placement requires 64-bit Windows"
    assert split.report["applied_threads"] == {}
