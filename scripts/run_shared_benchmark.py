"""Run fixed-configuration KITTI tests with bounded caches.

Use the bounded development runner for focused validation first; official worker
processes launched here currently have no independent wall-clock deadline.
"""

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import zipfile

from benchmark_identity import (
    CONTRACT_VERSION,
    FULL_FRAMES,
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
    stored_identity = candidate.get("benchmark_identity")
    if not isinstance(stored_identity, dict) or stored_identity.get("version") != CONTRACT_VERSION:
        is_partial = (candidate.get("status") not in ("completed", "completed_with_tracking_loss")
                      or candidate.get("frames") != frames
                      or candidate.get("coverage") != coverage)
        if is_partial and manifest_identity is not None and same_run_identity(
                manifest_identity, expected_identity):
            return "incomplete"
        raise ValueError(f"Existing {sequence}-{mode} report is legacy and cannot be resumed")
    if (candidate.get("status") not in ("completed", "completed_with_tracking_loss")
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
              mapping, performance, threads):
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
        status["runs"].append({
            "sequence": sequence, "mode": mode, "status": "completed",
            "reused": True, "output": base.name, "frames": frames,
            "coverage": coverage, "identity": run_identity,
        })
        save()
        print(f"{sequence} {mode}: reused exact completed report", flush=True)
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
                status["runs"].append({
                    "sequence": sequence, "mode": mode, "status": "completed",
                    "reused": True, "output": candidate.name, "frames": frames,
                    "coverage": coverage, "identity": run_identity,
                })
                save()
                return
            attempts += 1
        output = _report_path(root, sequence, mode, attempts)
    else:
        output = base
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

    row = {
        "sequence": sequence, "mode": mode, "status": "running",
        "output": output.name, "frames_expected": frames,
        "coverage": coverage, "identity": run_identity,
    }
    status["runs"].append(row)
    save()
    with (root / f"{sequence}-{mode}.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(command, env=env, cwd=repo, stdout=log,
                                stderr=subprocess.STDOUT, check=False)
    row["exit_code"] = result.returncode
    if result.returncode != 0:
        _annotate_partial_report(output / "evaluation.json", run_identity)
        row["status"] = "failed"
        save()
        raise RuntimeError(f"{sequence} {mode}: evaluator exited {result.returncode}")

    evaluation_path = output / "evaluation.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
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
        evaluation_status=evaluation["status"],
    )
    save()
    print(f"{sequence} {mode}: completed", flush=True)
    subprocess.run(
        [sys.executable, str(repo / "scripts/plot_shared_slam.py"),
         "--run", str(output), "--reference", str(reference)],
        env=env, cwd=repo, check=False,
    )


def _run_sequence(repo, root, status, save, env, args, sequence, contract,
                  batch_identity, mapping, performance, threads):
    frames = min(args.max_frames or FULL_FRAMES[sequence], FULL_FRAMES[sequence])
    report_coverage = "full" if frames == FULL_FRAMES[sequence] else "partial"
    reference = (args.poses_root / f"{sequence}.txt").resolve()
    preflight_budget = Budget(args.preflight_budget_seconds)
    with ExitStack() as resources:
        archive = None
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
            remote = resources.enter_context(RangeFile(URL))
            archive = resources.enter_context(zipfile.ZipFile(remote))
            use_local_cache = _prepare_archive(archive, cache, sequence, frames, preflight_budget)

        identity_source = cache if use_local_cache else archive
        inputs = hash_sequence_inputs(
            identity_source, sequence, frames, stereo="stereo" in args.modes,
            reference_path=reference, budget=preflight_budget,
        )
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
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if any(sequence not in FULL_FRAMES for sequence in args.sequences):
        parser.error("Evaluation sequences must be KITTI odometry 00–10")
    if len(set(args.sequences)) != len(args.sequences):
        parser.error("Sequences must not be repeated")
    if args.max_frames is not None and args.max_frames < 2:
        parser.error("--max-frames must be at least 2")
    if args.preflight_budget_seconds <= 0:
        parser.error("--preflight-budget-seconds must be positive")

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
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "runs": [],
    }
    lock = acquire_batch_lock(root / "batch.lock")
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
            owner={"host": socket.gethostname(), "pid": os.getpid()},
            status="running",
        )

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
            )
        status.update(status="completed", finished_utc=datetime.now(timezone.utc).isoformat())
        status.pop("owner", None)
        save()
        print(root / "batch.json")
    finally:
        release_batch_lock(lock)


if __name__ == "__main__":
    main()
