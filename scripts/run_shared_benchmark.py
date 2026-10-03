"""Run fixed-configuration KITTI tests under an owned total wall-clock deadline."""

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import uuid
import zipfile

from benchmark_identity import (
    CONTRACT_VERSION,
    COMPLETE_EXPERIMENT_STATUSES,
    FULL_FRAMES,
    REUSABLE_STATUSES,
    artifact_hashes,
    artifacts_complete,
    hash_sequence_inputs,
    _load_finite_json,
    report_reusable,
    resume_manifest as _resume_manifest,
    same_run_identity,
    source_contract,
)
from download_kitti_sequence import RangeFile, URL
from test_budget import Budget


MAX_BATCH_BUDGET_SECONDS = 3600
MIN_BATCH_BUDGET_SECONDS = 100
CLEANUP_RESERVE_SECONDS = 45


def _cleanup_reserve(seconds):
    return min(CLEANUP_RESERVE_SECONDS, max(0.1, seconds * 0.1))


class DeadlineBudget:
    """A shared absolute monotonic deadline passed from supervisor to worker."""
    def __init__(self, deadline, clock=time.monotonic):
        self.deadline, self.clock = float(deadline), clock

    @property
    def remaining(self):
        return max(0.0, self.deadline - self.clock())

    def permits(self, estimate, reserve=0):
        return estimate + reserve < self.remaining


class CombinedBudget:
    def __init__(self, sequence_budget, batch_budget, reserve):
        self.sequence_budget, self.batch_budget, self.reserve = sequence_budget, batch_budget, reserve

    @property
    def remaining(self):
        total_remaining = max(0.0, self.batch_budget.remaining - self.reserve)
        return min(self.sequence_budget.remaining, total_remaining)


class BudgetDeferred(Exception):
    def __init__(self, sequence, mode, phase, estimated_seconds):
        self.next_case = {"sequence": sequence, "mode": mode, "phase": phase,
                          "estimated_seconds": estimated_seconds}
        super().__init__(f"Deferred {sequence} {mode or ''} at {phase} to stay within batch budget")


def _mark_worker_timeout(output, worker_pid, owner_token):
    """Mark only the manifest owned by the timed-out child; keep its reports intact."""
    output = Path(output).resolve()
    changed = []
    for manifest_path in output.glob("*/*/batch.json"):
        try:
            status = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        if not isinstance(status, dict):
            continue
        owner = status.get("owner")
        if not isinstance(owner, dict):
            continue
        identity = status.get("identity") if isinstance(status, dict) else None
        contract = identity.get("code_contract_sha256") if isinstance(identity, dict) else None
        if (status.get("status") != "running"
                or owner.get("pid") != worker_pid
                or owner.get("host") != socket.gethostname()
                or owner.get("supervisor_token") != owner_token
                or not isinstance(contract, str)
                or status.get("source_sha256") != contract
                or manifest_path.parent.parent.name != contract[:12]):
            continue
        for row in status.get("runs", []):
            if row.get("status") == "running":
                output_name = row.get("output")
                if isinstance(output_name, str) and Path(output_name).name == output_name:
                    report_path = manifest_path.parent / output_name / "evaluation.json"
                    try:
                        partial = json.loads(report_path.read_text(encoding="utf-8"))
                    except (OSError, ValueError, TypeError):
                        partial = None
                    if (isinstance(partial, dict)
                            and not isinstance(partial.get("benchmark_identity"), dict)):
                        _annotate_partial_report(report_path, row.get("identity", {}))
                row.update(status="interrupted_total_budget",
                           error="Owned worker exceeded total batch wall-clock budget")
        status.update(status="interrupted_total_budget",
                      finished_utc=datetime.now(timezone.utc).isoformat(),
                      error="Owned worker exceeded total batch wall-clock budget")
        status.pop("owner", None)
        _write_json(manifest_path, status)
        changed.append(manifest_path)
    return changed


def _supervise_batch(output, budget_seconds, argv, *, run_owned_fn=None,
                     clock=time.monotonic):
    """Run the whole official worker under an owned process-tree hard deadline."""
    started = clock()
    deadline = started + budget_seconds
    reserve = _cleanup_reserve(budget_seconds)
    worker_deadline = deadline - reserve - 10.0
    owner_token = uuid.uuid4().hex
    if run_owned_fn is None:
        from run_development_tests import run_owned as run_owned_fn
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    worker_pid = None

    def started_worker(pid):
        nonlocal worker_pid
        worker_pid = pid

    command = [sys.executable, "-u", str(Path(__file__).resolve()), *argv,
               "--_deadline-worker", "--_deadline-at", str(worker_deadline),
               "--_owner-token", owner_token]
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
           "PYTHONIOENCODING": "utf-8"}
    log = output / f"batch-supervisor-{os.getpid()}-{time.time_ns()}.log"
    hard_timeout = max(0.05, deadline - reserve - 5.0 - clock())
    print(f"Official benchmark worker log: {log}", flush=True)
    result = run_owned_fn(command, log, hard_timeout, env, on_start=started_worker)
    if result.get("timed_out"):
        if worker_pid is not None:
            _mark_worker_timeout(output, worker_pid, owner_token)
        print("Official benchmark worker exceeded its total wall-clock budget.", flush=True)
        return 2
    print(f"Official benchmark worker exited {result.get('exit_code')} "
          f"after {result.get('elapsed_s', 0):.1f}s; log: {log}", flush=True)
    return int(result.get("exit_code") or 0)


def _absolute_path_argv(argv, args):
    """Preserve caller-relative paths despite the owned worker using repository cwd."""
    result = list(argv)
    single = {
        "--poses-root": args.poses_root.resolve(),
        "--output": args.output.resolve(),
        "--cache-root": args.cache_root.resolve(),
    }
    if args.data_root is not None:
        single["--data-root"] = args.data_root.resolve()
    for option, value in single.items():
        if option in result:
            index = len(result) - 1 - result[::-1].index(option)
            if index + 1 < len(result):
                result[index + 1] = str(value)
        else:
            result.extend((option, str(value)))
    if args.timing_history:
        option = "--timing-history"
        absolute = [str(path.resolve()) for path in args.timing_history]
        if option in result:
            index = len(result) - 1 - result[::-1].index(option)
            end = index + 1
            while end < len(result) and not result[end].startswith("--"):
                end += 1
            result[index + 1:end] = absolute
        else:
            result.extend((option, *absolute))
    return result


def _bounded_network_timeout(budget):
    """Temporarily cap RangeFile HTTP socket timeouts to the remaining batch time."""
    import urllib.request
    original = urllib.request.urlopen

    def bounded_urlopen(request, *args, timeout=None, **kwargs):
        remaining = budget.remaining - _cleanup_reserve(budget.remaining)
        if remaining <= 0:
            raise TimeoutError("Batch budget expired before the next HTTP range request")
        timeout = remaining if timeout is None else min(float(timeout), remaining)
        return original(request, *args, timeout=max(0.05, timeout), **kwargs)

    urllib.request.urlopen = bounded_urlopen
    return original


def _restore_network_timeout(original):
    import urllib.request
    urllib.request.urlopen = original


def _require_budget(budget, sequence, mode, phase, estimate):
    if budget is not None and not budget.permits(estimate, reserve=_cleanup_reserve(budget.remaining)):
        raise BudgetDeferred(sequence, mode, phase, estimate)


def resume_manifest(path, initial, resume, *, current_host=None, probe=None):
    """Public wrapper retained for callers and focused tests."""
    return _resume_manifest(path, initial, resume, current_host=current_host, probe=probe)


def acquire_batch_lock(path):
    """Hold a cross-process, OS-released lock for this batch directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+b")
    stream.seek(0, os.SEEK_END)
    if stream.tell() == 0:
        stream.write(b"\0")
        stream.flush()
    stream.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        stream.close()
        raise ValueError("Another benchmark owner holds the active batch lock") from error
    return stream


def release_batch_lock(stream):
    try:
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    finally:
        stream.close()


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _annotate_partial_report(path, run_identity):
    """Record the attempted identity on any parseable report, even after failure."""
    path = Path(path)
    if not path.is_file():
        return False
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    if not isinstance(report, dict):
        return False
    report["benchmark_identity"] = run_identity
    temporary = path.with_suffix(path.suffix + ".identity-part")
    # Preserve parseable NaN diagnostics if present; resume will reject them as incomplete.
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return True


def _configuration(repo):
    source_path = str(repo / "src")
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    from shared_slam import MappingConfig
    from performance import PerformanceConfig

    mapping = MappingConfig().__dict__.copy()
    performance = PerformanceConfig().__dict__.copy()
    return mapping, performance, {
        "opencv": 1,
        "openblas": 1,
        "omp": 1,
    }


def _run_identity(batch_identity, contract, inputs, sequence, mode, frames,
                  coverage, mapping, performance, threads, evaluator_input_source):
    return {
        "version": CONTRACT_VERSION,
        "batch": batch_identity,
        "code_contract_sha256": contract["contract_sha256"],
        "sequence": sequence,
        "mode": mode,
        "frames": frames,
        "coverage": coverage,
        "inputs": inputs,
        "configuration": mapping,
        "performance_configuration": performance,
        "threads": threads,
        "evaluator_input_source": evaluator_input_source,
        "diagnostic_cache": None,
    }


def _report_source_hashes(contract):
    # The evaluator's report uses basenames for its source snapshot keys.
    return {
        Path(name).name: digest
        for name, digest in contract["sources"].items()
        if name.startswith("src/")
    }


def _tracking_outcome_fields(evaluation):
    outcome = evaluation.get("status")
    initialized = (outcome in REUSABLE_STATUSES
                   or evaluation.get("initialization_elapsed_s") is not None)
    tracking_outcome = {
        "completed": "tracking_success",
        "completed_with_tracking_loss": "tracking_with_loss",
        "initialization_failed": "initialization_failed",
    }.get(outcome, outcome)
    return {
        "evaluation_status": outcome,
        "experiment_complete": outcome in COMPLETE_EXPERIMENT_STATUSES,
        "initialized": initialized,
        "tracking_outcome": tracking_outcome,
    }


def _estimate_mode_seconds(frames, run_identity, status, history_paths=()):
    """Estimate only runtime; history is accepted only for an identical run identity."""
    samples = []
    accepted = []
    rejected = []
    for row in status.get("runs", []):
        elapsed = row.get("elapsed_s")
        if (same_run_identity(row.get("identity", {}), run_identity)
                and row.get("status") == "completed"
                and isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool)
                and math.isfinite(elapsed) and elapsed > 0):
            samples.append(float(elapsed))
            accepted.append("manifest:" + str(row.get("output", "unknown")))
    for path in history_paths:
        try:
            report = _load_finite_json(path)
            identity = report.get("benchmark_identity") if isinstance(report, dict) else None
            elapsed = report.get("elapsed_s") if isinstance(report, dict) else None
            compatible = (
                isinstance(identity, dict)
                and same_run_identity(identity, run_identity)
                and report.get("frames") == frames
                and report.get("status") in COMPLETE_EXPERIMENT_STATUSES
                and isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool)
                and math.isfinite(elapsed) and elapsed > 0
            )
        except (OSError, ValueError, TypeError):
            compatible = False
        if compatible:
            samples.append(float(elapsed))
            accepted.append(str(Path(path).resolve()))
        else:
            rejected.append(str(Path(path).resolve()))
    fallback = 4.0 * frames
    measured = max(samples, default=0.0)
    estimate = math.ceil((measured if samples else fallback) * 1.25 + 30.0)
    return {"estimated_seconds": estimate, "fallback_seconds": fallback,
            "compatible_history_seconds": samples, "accepted_history": accepted,
            "rejected_history": rejected, "basis": "compatible_history" if samples else "fallback",
            "use": "cost_estimate_only"}


def _estimate_preflight_seconds(frames, stereo, remote):
    images = frames * (2 if stereo else 1)
    base = images * 0.25 + (120 if remote else 10)
    return math.ceil(base * 1.25 + 15)


def _run_owned(command, log, seconds, env):
    from run_development_tests import run_owned
    return run_owned(command, log, seconds, env)


def _prepare_archive(archive, cache, sequence, frames, budget):
    prefix = f"dataset/sequences/{sequence}/"
    entries = [
        entry for entry in archive.infolist()
        if entry.filename.startswith(prefix)
        and (entry.filename.endswith(".png")
             or Path(entry.filename).name in ("calib.txt", "times.txt"))
        and (not entry.filename.endswith(".png")
             or int(Path(entry.filename).stem) < frames)
    ]
    required = sum(entry.file_size for entry in entries)
    if shutil.disk_usage(cache).free <= required + 750 * 1024**2:
        return False
    for number, entry in enumerate(sorted(entries, key=lambda item: item.header_offset)):
        if budget.remaining <= 0:
            raise TimeoutError("Archive preflight exceeded its budget; no input hash was authorized")
        relative = Path(*Path(entry.filename).parts[1:])
        target = (cache / relative).resolve()
        if not target.is_relative_to(cache):
            raise ValueError("Archive member escapes owned cache")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archive.read(entry))
        if number % 200 == 0:
            print(f"{sequence}: cached {number + 1}/{len(entries)} files", flush=True)
    return True


def _report_path(root, sequence, mode, attempt=0):
    stem = f"kitti{sequence}-{mode}"
    return root / (stem if attempt == 0 else f"{stem}-retry-{attempt}")


def _try_reuse(output, expected_identity, *, sequence, mode, frames, coverage,
               mapping, performance, contract, manifest_identity=None):
    if not (output / "evaluation.json").exists():
        return "missing"
    try:
        candidate = _load_finite_json(output / "evaluation.json")
    except (OSError, ValueError, TypeError):
        return "incomplete"
    if not isinstance(candidate, dict):
        return "incomplete"
    stored_identity = candidate.get("benchmark_identity")
    if not isinstance(stored_identity, dict) or stored_identity.get("version") != CONTRACT_VERSION:
        is_partial = (candidate.get("status") not in COMPLETE_EXPERIMENT_STATUSES
                      or candidate.get("frames") != frames
                      or candidate.get("coverage") != coverage)
        if is_partial and manifest_identity is not None and same_run_identity(
                manifest_identity, expected_identity):
            return "incomplete"
        raise ValueError(f"Existing {sequence}-{mode} report is legacy and cannot be resumed")
    if (candidate.get("status") not in COMPLETE_EXPERIMENT_STATUSES
            or candidate.get("frames") != frames or candidate.get("coverage") != coverage):
        return "incomplete"
    if not same_run_identity(stored_identity, expected_identity):
        raise ValueError(f"Existing completed {sequence}-{mode} report has a different exact identity")
    evaluator_hash = contract["sources"]["scripts/evaluate_shared_slam.py"]
    if not artifacts_complete(output, frames, evaluator_hash):
        return "incomplete"
    valid = report_reusable(
        output,
        expected_identity,
        sequence=sequence,
        mode=mode,
        frames=frames,
        coverage=coverage,
        configuration=mapping,
        performance=performance,
        evaluator_sha256=evaluator_hash,
        source_hashes=_report_source_hashes(contract),
        opencv_threads=1,
        evaluator_input_source=expected_identity.get("evaluator_input_source"),
    )
    if not valid:
        raise ValueError(
            f"Existing completed {sequence}-{mode} report has mismatched configuration "
            "or invalid artifacts; preserve it and use a fresh --output"
        )
    return "reusable"


def _run_mode(repo, root, status, save, env, cache, use_local_cache, sequence, mode,
              frames, coverage, reference, contract, batch_identity, inputs,
              mapping, performance, threads, budget=None, timing_history_paths=()):
    evaluator_input_source = "local_images" if use_local_cache else "official_remote_archive"
    run_identity = _run_identity(
        batch_identity, contract, inputs, sequence, mode, frames, coverage,
        mapping, performance, threads, evaluator_input_source)

    previous = [row for row in status["runs"]
                if row.get("sequence") == sequence and row.get("mode") == mode]
    for row in previous:
        if row.get("identity") is not None and not same_run_identity(
                row["identity"], run_identity):
            raise ValueError(f"Cannot resume changed inputs for {sequence}-{mode}")
    manifest_identity = previous[-1].get("identity") if previous else None

    base = _report_path(root, sequence, mode)
    decision = _try_reuse(base, run_identity, sequence=sequence, mode=mode, frames=frames,
                          coverage=coverage, mapping=mapping, performance=performance,
                          contract=contract, manifest_identity=manifest_identity)
    if decision == "reusable":
        evaluation = _load_finite_json(base / "evaluation.json")
        status["runs"].append({
            "sequence": sequence, "mode": mode, "status": "completed",
            "reused": True, "output": base.name, "frames": frames,
            "coverage": coverage, "identity": run_identity,
            **_tracking_outcome_fields(evaluation),
        })
        save()
        print(f"{sequence} {mode}: reused exact completed experiment "
              f"({evaluation.get('status')})", flush=True)
        return

    if base.exists():
        attempts = 1
        while _report_path(root, sequence, mode, attempts).exists():
            candidate = _report_path(root, sequence, mode, attempts)
            decision = _try_reuse(candidate, run_identity, sequence=sequence, mode=mode,
                                  frames=frames, coverage=coverage, mapping=mapping,
                                  performance=performance, contract=contract,
                                  manifest_identity=manifest_identity)
            if decision == "reusable":
                evaluation = _load_finite_json(candidate / "evaluation.json")
                status["runs"].append({
                    "sequence": sequence, "mode": mode, "status": "completed",
                    "reused": True, "output": candidate.name, "frames": frames,
                    "coverage": coverage, "identity": run_identity,
                    **_tracking_outcome_fields(evaluation),
                })
                save()
                return
            attempts += 1
        output = _report_path(root, sequence, mode, attempts)
    else:
        output = base

    estimate = None
    evaluator_timeout = None
    if budget is not None:
        estimate = _estimate_mode_seconds(frames, run_identity, status, timing_history_paths)
        _require_budget(budget, sequence, mode, "evaluation", estimate["estimated_seconds"])
        status.setdefault("timing_estimates", []).append({
            "sequence": sequence, "mode": mode, **estimate,
        })
        save()
        available = budget.remaining - _cleanup_reserve(budget.remaining)
        evaluator_timeout = min(float(estimate["estimated_seconds"]), available - 10.0)
        if evaluator_timeout <= 1:
            raise BudgetDeferred(sequence, mode, "evaluation", estimate["estimated_seconds"])
    output.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable,
        str(repo / "scripts/evaluate_shared_slam.py"),
        "--sequence", sequence,
        "--poses-root", str(reference.parent),
        "--output", str(output),
        "--loop-mode", "live",
        "--matching-backend", "cpu",
        "--retrieval", "current",
        "--opencv-threads", "1",
    ]
    command += ["--data-root", str(cache)] if use_local_cache else ["--remote"]
    if mode == "stereo":
        command.append("--stereo")
    if coverage != "full":
        command += ["--max-frames", str(frames)]
    if evaluator_timeout is not None:
        command += ["--max-wall-seconds", str(max(1.0, evaluator_timeout))]

    row = {
        "sequence": sequence, "mode": mode, "status": "running",
        "output": output.name, "frames_expected": frames,
        "coverage": coverage, "identity": run_identity,
    }
    status["runs"].append(row)
    save()
    if budget is None:
        with (root / f"{sequence}-{mode}.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(command, env=env, cwd=repo, stdout=log,
                                    stderr=subprocess.STDOUT, check=False)
        run_result = {"timed_out": False, "elapsed_s": None}
    else:
        run_timeout = min(float(estimate["estimated_seconds"]) + 10.0,
                         budget.remaining - _cleanup_reserve(budget.remaining))
        run_result = _run_owned(command, root / f"{sequence}-{mode}.log",
                                run_timeout, env)
        result = type("RunResult", (), {"returncode": run_result.get("exit_code")})()
    row["exit_code"] = result.returncode
    if run_result.get("elapsed_s") is not None:
        row["elapsed_s"] = run_result["elapsed_s"]
    if run_result.get("timed_out"):
        _annotate_partial_report(output / "evaluation.json", run_identity)
        row.update(status="interrupted_total_budget",
                   error="Evaluator exceeded its owned wall-clock limit")
        save()
        raise BudgetDeferred(sequence, mode, "evaluation", estimate["estimated_seconds"])
    try:
        source_unchanged = source_contract(repo) == contract
    except Exception:
        source_unchanged = False
    if not source_unchanged:
        _annotate_partial_report(output / "evaluation.json", run_identity)
        row.update(status="invalid_source",
                   error="Benchmark code or dependencies changed during evaluation")
        save()
        raise RuntimeError(f"{sequence} {mode}: benchmark sources changed during evaluation")
    if result.returncode != 0:
        _annotate_partial_report(output / "evaluation.json", run_identity)
        row["status"] = "failed"
        save()
        raise RuntimeError(f"{sequence} {mode}: evaluator exited {result.returncode}")

    evaluation_path = output / "evaluation.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if evaluation.get("status") == "interrupted_time_budget":
        _annotate_partial_report(evaluation_path, run_identity)
        row.update(status="interrupted_total_budget",
                   error="Evaluator reached its cooperative wall-clock limit")
        save()
        raise BudgetDeferred(sequence, mode, "evaluation", estimate["estimated_seconds"]
                             if estimate else frames * 4)
    tracking_fields = _tracking_outcome_fields(evaluation)
    evaluation.update({key: value for key, value in tracking_fields.items()
                       if key != "experiment_complete"})
    run_identity["artifacts"] = artifact_hashes(output)
    evaluation["benchmark_identity"] = run_identity
    _write_json(evaluation_path, evaluation)
    try:
        valid_report = _try_reuse(
            output, run_identity, sequence=sequence, mode=mode, frames=frames,
            coverage=coverage, mapping=mapping, performance=performance,
            contract=contract,
        )
    except ValueError:
        valid_report = False
    if valid_report != "reusable":
        row["status"] = "invalid_report"
        save()
        raise RuntimeError(f"{sequence} {mode}: evaluator report failed exact reuse validation")
    row.update(
        status="completed",
        frames=frames,
        **tracking_fields,
    )
    save()
    print(f"{sequence} {mode}: completed experiment ({evaluation['status']})", flush=True)
    plot_command = [sys.executable, str(repo / "scripts/plot_shared_slam.py"),
                    "--run", str(output), "--reference", str(reference)]
    if budget is None:
        subprocess.run(plot_command, env=env, cwd=repo, check=False)
    else:
        if budget.permits(15, reserve=_cleanup_reserve(budget.remaining)):
            plot_result = _run_owned(plot_command, root / f"{sequence}-{mode}-plot.log",
                                     min(15.0, budget.remaining - _cleanup_reserve(budget.remaining)),
                                     env)
            row["plot"] = plot_result
        else:
            row["plot"] = {"status": "deferred_budget", "estimated_seconds": 15}
        save()


def _run_sequence(repo, root, status, save, env, args, sequence, contract,
                  batch_identity, mapping, performance, threads,
                  batch_budget=None, timing_history_paths=()):
    frames = min(args.max_frames or FULL_FRAMES[sequence], FULL_FRAMES[sequence])
    report_coverage = "full" if frames == FULL_FRAMES[sequence] else "partial"
    reference = (args.poses_root / f"{sequence}.txt").resolve()
    remote_input = args.data_root is None
    preflight_estimate = _estimate_preflight_seconds(
        frames, "stereo" in args.modes, remote_input)
    completed_modes = {
        row.get("mode") for row in status.get("runs", [])
        if row.get("sequence") == sequence and row.get("status") == "completed"
    }
    needs_evaluation = any(mode not in completed_modes for mode in args.modes)
    first_evaluation_estimate = (math.ceil(4.0 * frames * 1.25 + 30.0)
                                 if needs_evaluation else 0)
    preparation_plan = preflight_estimate + first_evaluation_estimate
    _require_budget(batch_budget, sequence, None, "preflight", preparation_plan)
    status.setdefault("timing_estimates", []).append({
        "sequence": sequence, "mode": None, "phase": "preflight",
        "estimated_seconds": preflight_estimate,
        "followup_evaluation_seconds": first_evaluation_estimate,
        "planned_total_seconds": preparation_plan,
        "use": "cost_estimate_only",
    })
    save()
    _require_budget(batch_budget, sequence, None, "preflight", preparation_plan)
    sequence_budget = Budget(args.preflight_budget_seconds)
    preflight_budget = (CombinedBudget(sequence_budget, batch_budget,
                                       _cleanup_reserve(batch_budget.remaining))
                        if batch_budget is not None else sequence_budget)
    with ExitStack() as resources:
        archive = None
        network_original = None
        try:
            if args.data_root is not None:
                cache = args.data_root.resolve()
                use_local_cache = True
            else:
                temporary = resources.enter_context(
                    tempfile.TemporaryDirectory(prefix=f"{sequence}-", dir=args.cache_root))
                cache = Path(temporary).resolve()
                (cache / "owner.json").write_text(json.dumps({
                    "purpose": "temporary shared SLAM benchmark images",
                    "sequence": sequence, "source": URL, "process": os.getpid(),
                }), encoding="utf-8")
                if batch_budget is not None:
                    network_original = _bounded_network_timeout(batch_budget)
                remote = resources.enter_context(RangeFile(URL))
                archive = resources.enter_context(zipfile.ZipFile(remote))
                use_local_cache = _prepare_archive(archive, cache, sequence, frames, preflight_budget)

            identity_source = cache if use_local_cache else archive
            inputs = hash_sequence_inputs(
                identity_source, sequence, frames, stereo="stereo" in args.modes,
                reference_path=reference, budget=preflight_budget,
            )
        except TimeoutError:
            if (batch_budget is not None
                    and preflight_budget.remaining <= 0
                    and batch_budget.remaining <= preflight_budget.reserve + 1):
                raise BudgetDeferred(sequence, None, "preflight", preflight_estimate)
            raise
        finally:
            if network_original is not None:
                _restore_network_timeout(network_original)
        status.setdefault("preflight", []).append({
            "sequence": sequence, "frames": frames,
            "input_identity": inputs,
        })
        save()
        if source_contract(repo) != contract:
            raise RuntimeError("Benchmark code or dependencies changed during the batch")
        for mode in args.modes:
            if source_contract(repo) != contract:
                raise RuntimeError("Benchmark code or dependencies changed during the batch")
            _run_mode(
                repo, root, status, save, env, cache, use_local_cache,
                sequence, mode, frames, report_coverage, reference, contract,
                batch_identity, inputs, mapping, performance, threads,
                batch_budget, timing_history_paths,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", nargs="+", default=list(FULL_FRAMES))
    parser.add_argument("--modes", nargs="+", choices=["stereo", "mono"],
                        default=["stereo", "mono"])
    parser.add_argument("--poses-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path,
                        help="Read existing KITTI sequences without downloading or deleting images")
    parser.add_argument("--output", type=Path, default=Path("results/shared-benchmark"))
    parser.add_argument("--cache-root", type=Path, default=Path(".datasets/shared-benchmark-scratch"))
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--preflight-budget-seconds", type=float, default=3600,
                        help="Maximum time to validate and hash each sequence before evaluation")
    parser.add_argument("--budget-seconds", type=float, default=3600,
        help="Total batch wall-clock budget (100–3600 seconds)")
    parser.add_argument("--timing-history", type=Path, nargs="*", default=[],
                        help="Exact-identity evaluation reports used only for runtime estimates")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--_deadline-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_deadline-at", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--_owner-token", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if any(sequence not in FULL_FRAMES for sequence in args.sequences):
        parser.error("Evaluation sequences must be KITTI odometry 00–10")
    if len(set(args.sequences)) != len(args.sequences):
        parser.error("Sequences must not be repeated")
    if args.max_frames is not None and args.max_frames < 2:
        parser.error("--max-frames must be at least 2")
    if args.preflight_budget_seconds <= 0:
        parser.error("--preflight-budget-seconds must be positive")
    if not MIN_BATCH_BUDGET_SECONDS <= args.budget_seconds <= MAX_BATCH_BUDGET_SECONDS:
        parser.error("--budget-seconds must be within 100–3600 seconds")
    if not args._deadline_worker:
        return _supervise_batch(
            args.output, args.budget_seconds, _absolute_path_argv(sys.argv[1:], args))
    if args._deadline_at is None or not args._owner_token:
        parser.error("Internal deadline worker is missing its supervisor metadata")

    worker_started = time.monotonic()
    batch_budget = DeadlineBudget(args._deadline_at)
    repo = Path(__file__).resolve().parents[1]
    args.poses_root = args.poses_root.resolve()
    contract = source_contract(repo)
    mapping, performance, threads = _configuration(repo)
    effective_full = all(
        args.max_frames is None or args.max_frames >= FULL_FRAMES[sequence]
        for sequence in args.sequences
    )
    coverage = "full" if effective_full else f"prefix{args.max_frames}"
    batch_identity = {
        "version": CONTRACT_VERSION,
        "code_contract_sha256": contract["contract_sha256"],
        "code_contract": contract,
        "coverage": coverage,
        "max_frames": args.max_frames,
        "requested_sequences": list(args.sequences),
        "requested_modes": list(args.modes),
        "input_source": "local_images" if args.data_root is not None else "official_remote_archive",
        "configuration": mapping,
        "performance_configuration": performance,
        "loop_mode": "live",
        "threads": threads,
        "diagnostic_cache": None,
    }
    root = args.output.resolve() / contract["contract_sha256"][:12] / coverage
    root.mkdir(parents=True, exist_ok=True)
    if args.data_root is None:
        args.cache_root = args.cache_root.resolve()
        args.cache_root.mkdir(parents=True, exist_ok=True)
    initial = {
        "identity_version": CONTRACT_VERSION,
        "identity": batch_identity,
        "source_sha256": contract["contract_sha256"],
        "coverage": coverage,
        "input_source": batch_identity["input_source"],
        "requested_sequences": list(args.sequences),
        "requested_modes": list(args.modes),
        "configuration_policy": "Versioned fixed MappingConfig, PerformanceConfig, loop mode and thread contract",
        "budget_seconds": args.budget_seconds,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "runs": [],
    }
    lock = acquire_batch_lock(root / "batch.lock")
    status = None
    save = None
    try:
        status = resume_manifest(root / "batch.json", initial, args.resume)
        status.update(
            identity_version=CONTRACT_VERSION,
            identity=batch_identity,
            source_sha256=contract["contract_sha256"],
            coverage=coverage,
            input_source=batch_identity["input_source"],
            requested_sequences=list(args.sequences),
            requested_modes=list(args.modes),
            budget_seconds=args.budget_seconds,
            owner={"host": socket.gethostname(), "pid": os.getpid(),
                   "supervisor_token": args._owner_token},
            status="running",
        )
        status.pop("next_case", None)

        def save():
            _write_json(root / "batch.json", status)

        env = {
            **os.environ,
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "PYTHONIOENCODING": "utf-8",
        }
        save()
        for sequence in args.sequences:
            _run_sequence(
                repo, root, status, save, env, args, sequence, contract,
                batch_identity, mapping, performance, threads,
                batch_budget, args.timing_history,
            )
        status.update(status="completed", finished_utc=datetime.now(timezone.utc).isoformat(),
                      elapsed_s=time.monotonic() - worker_started)
        status.pop("owner", None)
        status.pop("next_case", None)
        save()
        print(root / "batch.json")
        return 0
    except BudgetDeferred as error:
        if status is not None and save is not None:
            status.update(status="deferred_budget", next_case=error.next_case,
                          finished_utc=datetime.now(timezone.utc).isoformat(),
                          elapsed_s=time.monotonic() - worker_started)
            status.pop("owner", None)
            save()
            print(json.dumps(error.next_case), flush=True)
        return 2
    except BaseException as error:
        if status is not None and save is not None:
            for row in status.get("runs", []):
                if row.get("status") == "running":
                    row.update(status="interrupted",
                               error=f"{type(error).__name__}: {error}")
            status.update(
                status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                finished_utc=datetime.now(timezone.utc).isoformat(),
                error=f"{type(error).__name__}: {error}",
            )
            status.pop("owner", None)
            save()
        raise
    finally:
        release_batch_lock(lock)


if __name__ == "__main__":
    raise SystemExit(main())
