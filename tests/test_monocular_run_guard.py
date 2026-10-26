"""Pure arithmetic checks; no evaluator, dataset or worker is launched."""
import importlib.util
import json
from pathlib import Path

import pytest


path = Path(__file__).resolve().parents[1] / 'scripts' / 'monocular_run_guard.py'
spec = importlib.util.spec_from_file_location('monocular_run_guard', path)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


def request(**changes):
    values = dict(remaining_seconds=900., expected_launch_seconds=520.385951437,
                  uncertainty_seconds=50., shutdown_export_reserve_seconds=30.,
                  terminal_receipt_reserve_seconds=5., disk_free_bytes=guard.MIN_FREE_BYTES)
    values.update(changes)
    return guard.check_sequence_start(**values)


def test_old_sixty_second_room_cutoff_is_not_sufficient():
    result = request(remaining_seconds=60.)
    assert result['planning_fits'] is False
    assert result['required_seconds'] == pytest.approx(605.385951437)
    assert len(result['blockers']) == 1


def test_observed_room_time_uncertainty_and_both_reserves_fit_inside_grant():
    # Hypothetical allowances exercise arithmetic, not a justified reserve/grant.
    result = request(remaining_seconds=900. - 288.900008358)
    assert result['planning_fits'] is True
    assert result['slack_seconds'] == pytest.approx(5.714040205)
    assert result['execution_authorized'] is False


def test_source_frozen_pair_estimate_does_not_fit_when_reserve_is_underfunded():
    result = request(expected_launch_seconds=809.297743843,
                     uncertainty_seconds=60., shutdown_export_reserve_seconds=30.,
                     terminal_receipt_reserve_seconds=5.)
    assert result['planning_fits'] is False
    assert result['required_seconds'] == pytest.approx(904.297743843)


def test_exact_time_edge_abstains_instead_of_using_zero_slack():
    needed = request()['required_seconds']
    assert request(remaining_seconds=needed)['planning_fits'] is False
    assert request(remaining_seconds=needed + .001)['planning_fits'] is True


def test_disk_floor_is_inclusive_and_rechecked_independently():
    assert request(disk_free_bytes=guard.MIN_FREE_BYTES)['planning_fits'] is True
    result = request(disk_free_bytes=guard.MIN_FREE_BYTES - 1, remaining_seconds=60.)
    assert len(result['blockers']) == 2


def test_a_stricter_evaluator_disk_floor_can_only_raise_the_minimum():
    required = guard.MIN_FREE_BYTES + 256*1024**2
    result = request(minimum_free_bytes=required)
    assert result['planning_fits'] is False and result['minimum_free_bytes'] == required
    assert request(minimum_free_bytes=required, disk_free_bytes=required)['planning_fits'] is True
    with pytest.raises(ValueError, match='at least 5 GiB'):
        request(minimum_free_bytes=guard.MIN_FREE_BYTES - 1)


@pytest.mark.parametrize('name', [
    'remaining_seconds', 'expected_launch_seconds', 'uncertainty_seconds',
    'shutdown_export_reserve_seconds', 'terminal_receipt_reserve_seconds',
])
@pytest.mark.parametrize('value', [True, -1., float('nan'), float('inf'), '60'])
def test_invalid_time_inputs_cannot_create_a_planning_pass(name, value):
    with pytest.raises(ValueError, match='must be a finite'):
        request(**{name: value})


@pytest.mark.parametrize('name', [
    'expected_launch_seconds', 'uncertainty_seconds',
    'shutdown_export_reserve_seconds', 'terminal_receipt_reserve_seconds',
])
def test_no_implicit_zero_uncertainty_or_shutdown_bound(name):
    with pytest.raises(ValueError, match='positive'):
        request(**{name: 0.})


@pytest.mark.parametrize('value', [True, -1, 5.1, '5368709120'])
def test_disk_input_is_an_exact_byte_count(value):
    with pytest.raises(ValueError, match='nonnegative integer'):
        request(disk_free_bytes=value)


def test_zero_remaining_is_a_blocked_state_not_an_input_error():
    assert request(remaining_seconds=0.)['planning_fits'] is False


def test_cli_reports_planning_only_and_requires_all_explicit_allowances(capsys):
    arguments = ['--remaining-seconds', '900', '--expected-launch-seconds', '520',
                 '--uncertainty-seconds', '50', '--shutdown-export-reserve-seconds', '30',
                 '--terminal-receipt-reserve-seconds', '5',
                 '--disk-free-bytes', str(guard.MIN_FREE_BYTES)]
    assert guard.main(arguments) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['planning_fits'] and result['execution_authorized'] is False
    with pytest.raises(SystemExit) as missing:
        guard.main(arguments[:-2])
    assert missing.value.code == 2


def test_cli_returns_nonzero_for_an_underbudget_sequence(capsys):
    assert guard.main([
        '--remaining-seconds', '60', '--expected-launch-seconds', '520',
        '--uncertainty-seconds', '50', '--shutdown-export-reserve-seconds', '30',
        '--terminal-receipt-reserve-seconds', '5',
        '--disk-free-bytes', str(guard.MIN_FREE_BYTES),
    ]) == 2
    assert json.loads(capsys.readouterr().out)['planning_fits'] is False
