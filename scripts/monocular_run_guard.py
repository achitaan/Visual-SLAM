"""Conservative budget arithmetic for a separately admitted monocular run.

This checker never launches work, grants CPU ownership, authenticates a release,
or proves a cancellation bound. The owner must supply and justify every time
allowance. Expected launch time includes normal startup, tracking, worker drain,
export and native scoring; the two reserves are additional contingency inside
the grant. No fixed 60-second room cutoff or assumed shutdown bound is used.
"""
import argparse
import json
import math


MIN_FREE_BYTES = 5 * 1024**3


def _seconds(name, value, *, positive=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or (positive and value == 0)):
        qualifier = 'positive' if positive else 'nonnegative'
        raise ValueError(f'{name} must be a finite {qualifier} number')
    return float(value)


def check_sequence_start(*, remaining_seconds, expected_launch_seconds,
                         uncertainty_seconds, shutdown_export_reserve_seconds,
                         terminal_receipt_reserve_seconds, disk_free_bytes,
                         minimum_free_bytes=MIN_FREE_BYTES):
    """Return a planning decision, not permission to execute.

    Call again immediately before *each* sequence with actual remaining grant
    time and current free disk. A completed desk result cannot justify silently
    shortening room, extending the deadline, or borrowing a later grant.
    """
    remaining = _seconds('remaining_seconds', remaining_seconds)
    expected = _seconds('expected_launch_seconds', expected_launch_seconds, positive=True)
    uncertainty = _seconds('uncertainty_seconds', uncertainty_seconds, positive=True)
    shutdown = _seconds('shutdown_export_reserve_seconds',
                        shutdown_export_reserve_seconds, positive=True)
    receipt = _seconds('terminal_receipt_reserve_seconds',
                       terminal_receipt_reserve_seconds, positive=True)
    if (isinstance(disk_free_bytes, bool) or not isinstance(disk_free_bytes, int)
            or disk_free_bytes < 0):
        raise ValueError('disk_free_bytes must be a nonnegative integer')
    if (isinstance(minimum_free_bytes, bool) or not isinstance(minimum_free_bytes, int)
            or minimum_free_bytes < MIN_FREE_BYTES):
        raise ValueError('minimum_free_bytes must be an integer at least 5 GiB')
    required = math.fsum((expected, uncertainty, shutdown, receipt))
    if not math.isfinite(required):
        raise ValueError('combined time requirement must be finite')
    blockers = []
    # Strict slack avoids authorizing a start exactly on the termination edge.
    if remaining <= required:
        blockers.append('insufficient remaining grant for estimate, uncertainty and reserves')
    if disk_free_bytes < minimum_free_bytes:
        blockers.append(f'less than required free disk floor ({minimum_free_bytes} bytes)')
    return {
        'planning_fits': not blockers,
        'remaining_seconds': remaining,
        'expected_launch_seconds': expected,
        'uncertainty_seconds': uncertainty,
        'shutdown_export_reserve_seconds': shutdown,
        'terminal_receipt_reserve_seconds': receipt,
        'required_seconds': required,
        'slack_seconds': remaining - required,
        'disk_free_bytes': disk_free_bytes,
        'minimum_free_bytes': minimum_free_bytes,
        'blockers': blockers,
        'execution_authorized': False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('remaining-seconds', 'expected-launch-seconds', 'uncertainty-seconds',
                 'shutdown-export-reserve-seconds', 'terminal-receipt-reserve-seconds'):
        parser.add_argument('--' + name, type=float, required=True)
    parser.add_argument('--disk-free-bytes', type=int, required=True)
    args = vars(parser.parse_args(argv))
    try:
        result = check_sequence_start(**args)
    except (ValueError, OverflowError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result['planning_fits'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
