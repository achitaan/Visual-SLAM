"""Versioned identities and safe reuse checks for official KITTI runs."""

from importlib import metadata
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import socket
import sys


CONTRACT_VERSION = 1
FULL_FRAMES = {
    "00": 4541,
    "01": 1101,
    "02": 4661,
    "03": 801,
    "04": 271,
    "05": 2761,
    "06": 1101,
    "07": 1101,
    "08": 4071,
    "09": 1591,
    "10": 1201,
}
REUSABLE_STATUSES = {"completed", "completed_with_tracking_loss"}


class _FramedHasher:
    def __init__(self):
        self.digest = hashlib.sha256()

    def update(self, name, content):
        name = name.encode("utf-8") if isinstance(name, str) else bytes(name)
        content = content.encode("utf-8") if isinstance(content, str) else bytes(content)
        self.digest.update(len(name).to_bytes(8, "big"))
        self.digest.update(name)
        self.digest.update(len(content).to_bytes(8, "big"))
        self.digest.update(content)

    def hexdigest(self):
        return self.digest.hexdigest()


def _framed_hash(records):
    digest = _FramedHasher()
    for name, content in records:
        digest.update(name, content)
    return digest.hexdigest()


def _dependency_names(repo):
    names = set()
    for filename in ("requirement.txt", "requirements-lock.txt", "requirements-depth.txt",
                     "requirements-performance-gpu.txt"):
        path = repo / filename
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            token = line.split(";", 1)[0].strip().split("[", 1)[0]
            name = token.split("==", 1)[0].split(">=", 1)[0].split("~=", 1)[0].strip()
            if name:
                names.add(name)
    return sorted(names, key=str.lower)


def source_contract(repo):
    """Hash named implementation/dependency inputs with unambiguous framing."""
    repo = Path(repo).resolve()
    # Match the evaluator's persisted source snapshot exactly: direct src modules.
    source_paths = list((repo / "src").glob("*.py"))
    source_paths += [repo / "scripts" / name for name in (
        "run_shared_benchmark.py", "benchmark_identity.py", "evaluate_shared_slam.py",
        "data_preflight.py", "download_kitti_sequence.py", "run_kitti_stream.py",
        "benchmark_telemetry.py", "test_budget.py")]
    source_records = []
    source_hashes = {}
    for path in sorted({p.resolve() for p in source_paths if p.is_file()}):
        name = path.relative_to(repo).as_posix()
        data = path.read_bytes()
        source_records.append((name, data))
        source_hashes[name] = hashlib.sha256(data).hexdigest()

    dependency_files = {}
    dependency_records = []
    for name in ("requirement.txt", "requirements-lock.txt", "requirements-depth.txt",
                 "requirements-performance-gpu.txt"):
        path = repo / name
        if path.is_file():
            data = path.read_bytes()
            dependency_files[name] = hashlib.sha256(data).hexdigest()
            dependency_records.append((name, data))
    packages = {}
    for name in _dependency_names(repo):
        try:
            version = metadata.version(name)
        except metadata.PackageNotFoundError:
            version = None
        packages[name] = version
    dependencies = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": sys.platform,
        "machine": platform.machine(),
        "files": dependency_files,
        "packages": packages,
    }
    dependency_records.append(("runtime-environment.json", json.dumps(
        dependencies, sort_keys=True, separators=(",", ":"))))
    contract = {
        "version": CONTRACT_VERSION,
        "sources": source_hashes,
        "source_sha256": _framed_hash(source_records),
        "dependencies": dependencies,
        "dependencies_sha256": _framed_hash(dependency_records),
    }
    contract["contract_sha256"] = _framed_hash((
        ("contract.json", json.dumps(contract, sort_keys=True, separators=(",", ":"))),
    ))
    return contract


def hash_sequence_inputs(data_root, sequence, frames, stereo, reference_path, budget=None):
    """Validate and hash exact ordered KITTI inputs; timeout yields no partial identity."""
    import cv2 as cv
    import numpy as np
    import io

    if sequence not in FULL_FRAMES:
        raise ValueError(f"Unsupported KITTI odometry sequence: {sequence}")
    if not isinstance(frames, int) or frames < 2 or frames > FULL_FRAMES[sequence]:
        raise ValueError("Input frame count must be between 2 and the KITTI sequence length")
    from zipfile import ZipFile

    archive = data_root if isinstance(data_root, ZipFile) else None
    directory = None if archive else Path(data_root) / "sequences" / sequence
    prefix = f"dataset/sequences/{sequence}/"

    def read_member(name):
        if budget is not None and budget.remaining <= 0:
            raise TimeoutError("Input preflight exceeded its budget; no hash was authorized")
        return archive.read(name) if archive else (directory / name).read_bytes()

    if archive:
        names = set(archive.namelist())
        calibration_name = prefix + "calib.txt"
        timestamp_name = prefix + "times.txt"
        if calibration_name not in names:
            raise FileNotFoundError(calibration_name)
        calibration = read_member(calibration_name)
        timestamps = read_member(timestamp_name) if timestamp_name in names else None
    else:
        calibration = read_member("calib.txt")
        timestamps_path = directory / "times.txt"
        timestamps = read_member("times.txt") if timestamps_path.is_file() else None
    if timestamps is not None:
        values = np.loadtxt(io.StringIO(timestamps.decode("utf-8")))
        values = np.asarray(values, dtype=float).reshape(-1)
        if len(values) < frames or not np.isfinite(values).all() or np.any(np.diff(values) <= 0):
            raise ValueError("Invalid frame timestamps")

    cameras = (0, 1) if stereo else (0,)
    camera_hashes = {}
    overall_digest = _FramedHasher()
    shape = None
    for camera in cameras:
        if archive:
            paths = sorted(name for name in archive.namelist()
                           if name.startswith(prefix + f"image_{camera}/")
                           and name.endswith(".png"))
            path_names = paths[:frames]
        else:
            folder = directory / f"image_{camera}"
            paths = sorted(path for path in folder.glob("*.png") if path.is_file())
            path_names = paths[:frames]
        if len(paths) < frames:
            raise ValueError(f"Camera {camera}: expected at least {frames} images")
        camera_digest = _FramedHasher()
        for index, path in enumerate(path_names):
            if budget is not None and budget.remaining <= 0:
                raise TimeoutError("Input preflight exceeded its budget; no hash was authorized")
            image_name = Path(path).name
            if Path(path).stem != f"{index:06d}":
                raise ValueError(f"Camera {camera}: image numbering has a gap at {image_name}")
            data = archive.read(path) if archive else path.read_bytes()
            image = cv.imdecode(np.frombuffer(data, np.uint8), cv.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Unreadable camera {camera} image {image_name}")
            if shape is None:
                shape = image.shape
            if image.shape != shape:
                raise ValueError("Image dimensions changed across cameras or frames")
            named = f"camera-{camera}/{image_name}"
            camera_digest.update(named, data)
            overall_digest.update(named, data)
        camera_hashes[str(camera)] = camera_digest.hexdigest()

    reference = Path(reference_path).read_bytes()
    if budget is not None and budget.remaining <= 0:
        raise TimeoutError("Input preflight exceeded its budget; no hash was authorized")
    return {
        "version": CONTRACT_VERSION,
        "sequence": sequence,
        "frames": frames,
        "cameras": list(cameras),
        "image_sha256": overall_digest.hexdigest(),
        "camera_image_sha256": camera_hashes,
        "calibration_sha256": hashlib.sha256(calibration).hexdigest(),
        "timestamps_sha256": hashlib.sha256(timestamps).hexdigest() if timestamps is not None else None,
        "reference_sha256": hashlib.sha256(reference).hexdigest(),
        "shape": list(shape),
    }


def _owner_alive(owner, current_host=None, probe=None):
    if not isinstance(owner, dict):
        return None
    current_host = current_host or socket.gethostname()
    if owner.get("host") != current_host:
        return None
    pid = owner.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        return None
    if probe is not None:
        return probe(pid)
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return False if ctypes.get_last_error() == 87 else None
        code = wintypes.DWORD()
        try:
            return code.value == 259 if kernel.GetExitCodeProcess(handle, ctypes.byref(code)) else None
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return None


def resume_manifest(path, initial, resume, *, current_host=None, probe=None):
    path = Path(path)
    if not path.exists():
        return initial
    if not resume:
        raise ValueError("Existing batch is preserved; use --resume or a fresh output")
    existing = json.loads(path.read_text(encoding="utf-8"))
    if existing.get("identity_version") != CONTRACT_VERSION:
        raise ValueError("Cannot resume a legacy benchmark manifest")
    if existing.get("identity") != initial.get("identity"):
        raise ValueError("Cannot resume a benchmark with changed code, dependencies, inputs, or configuration")
    if existing.get("status") == "running":
        alive = _owner_alive(existing.get("owner"), current_host=current_host, probe=probe)
        if alive is not False:
            raise ValueError("Batch owner is active or liveness is unknown; refusing concurrent resume")
        existing.setdefault("interruptions", []).append({
            "status": "owner_exited", "owner": existing.get("owner")})
    existing.setdefault("resumed_utc", []).append(datetime.now(timezone.utc).isoformat())
    existing.pop("finished_utc", None)
    return existing


def _finite(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(v) for v in value)
    return True


def _load_finite_json(path):
    def reject(value):
        raise ValueError(f"Non-finite JSON number: {value}")
    value = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)
    if not _finite(value):
        raise ValueError("Non-finite JSON artifact")
    return value


def _finite_numeric_file(path, *, pose_rows=None):
    rows = []
    for line in Path(path).read_text(encoding="ascii").splitlines():
        if not line.strip():
            continue
        try:
            row = [float(token) for token in line.split()]
        except ValueError:
            return False
        if not all(math.isfinite(value) for value in row):
            return False
        rows.append(row)
    if pose_rows is not None:
        return len(rows) == pose_rows and all(len(row) == 12 for row in rows)
    return bool(rows)


def report_reusable(output, expected_identity, *, sequence, mode, frames, coverage,
                    configuration, performance, evaluator_sha256, source_hashes,
                    opencv_threads=1, evaluator_input_source=None):
    """Accept only complete, exact-identity reports with finite exported artifacts."""
    output = Path(output)
    path = output / "evaluation.json"
    if not path.is_file():
        return False
    try:
        report = _load_finite_json(path)
        stored_identity = report.get("benchmark_identity")
        if not isinstance(stored_identity, dict) or not same_run_identity(
                stored_identity, expected_identity):
            return False
        if stored_identity.get("artifacts") != artifact_hashes(output):
            return False
        if (report.get("sequence") != sequence or report.get("dataset") != "kitti"
                or report.get("stereo") is not (mode == "stereo")
                or report.get("frames") != frames or report.get("coverage") != coverage
                or report.get("status") not in REUSABLE_STATUSES
                or report.get("ground_truth_used_for_estimation") is not False):
            return False
        if report.get("configuration") != configuration:
            return False
        if report.get("performance_configuration") != performance:
            return False
        if report.get("opencv_threads") != opencv_threads:
            return False
        if report.get("source_sha256") != source_hashes:
            return False
        if report.get("evaluator_sha256") != evaluator_sha256:
            return False
        if report.get("diagnostic_overrides") != {"disable_bundle": False, "loop_mode": None}:
            return False
        if report.get("feature_cache", {}).get("enabled") is not False:
            return False
        if report.get("performance_configuration", {}).get("profile") is not False:
            return False
        if report.get("matching_backend", {}).get("requested") != performance.get("matching_backend"):
            return False
        if (report.get("input_source") not in ("local_images", "official_remote_archive")
                or (evaluator_input_source is not None
                    and report.get("input_source") != evaluator_input_source)):
            return False
        run = _load_finite_json(output / "run.json")
        preview = _load_finite_json(output / "preview.json")
        if not isinstance(run, dict) or not isinstance(preview, dict):
            return False
        if (run.get("configuration") != configuration
                or run.get("performance_configuration") != performance
                or run.get("matching_backend", {}).get("requested")
                != performance.get("matching_backend")):
            return False
        if report.get("telemetry", {}).get("enabled") is not False:
            return False
        if not _finite_numeric_file(output / "poses.txt", pose_rows=frames):
            return False
        if not _finite_ply(output / "sparse.ply"):
            return False
        if hashlib.sha256((output / "evaluator.py").read_bytes()).hexdigest() != evaluator_sha256:
            return False
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    return True


def artifacts_complete(output, frames, evaluator_sha256):
    """Check the finite, complete exports needed for a report to be reusable."""
    output = Path(output)
    required = ("run.json", "preview.json", "poses.txt", "sparse.ply", "evaluator.py")
    if any(not (output / name).is_file() for name in required):
        return False
    try:
        run = _load_finite_json(output / "run.json")
        preview = _load_finite_json(output / "preview.json")
        if not isinstance(run, dict) or not isinstance(preview, dict):
            return False
        if not _finite_numeric_file(output / "poses.txt", pose_rows=frames):
            return False
        if not _finite_ply(output / "sparse.ply"):
            return False
        return hashlib.sha256((output / "evaluator.py").read_bytes()).hexdigest() == evaluator_sha256
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _finite_numeric_file_from_line(line):
    try:
        values = [float(token) for token in line.split()]
    except ValueError:
        return False
    return len(values) == 6 and all(math.isfinite(value) for value in values)


def _finite_ply(path):
    lines = Path(path).read_text(encoding="ascii").splitlines()
    try:
        end = lines.index("end_header")
    except ValueError:
        return False
    header = lines[:end]
    if "format ascii 1.0" not in header:
        return False
    counts = [line.split()[-1] for line in header
              if line.startswith("element vertex ") and len(line.split()) == 3]
    if len(counts) != 1:
        return False
    try:
        expected = int(counts[0])
    except ValueError:
        return False
    if expected < 0:
        return False
    rows = [line for line in lines[end + 1:] if line.strip()]
    return len(rows) == expected and all(_finite_numeric_file_from_line(line) for line in rows)


def artifact_hashes(output):
    output = Path(output)
    names = ("run.json", "preview.json", "poses.txt", "sparse.ply", "evaluator.py")
    return {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
            for name in names}


def same_run_identity(first, second):
    return ({key: value for key, value in first.items() if key != "artifacts"}
            == {key: value for key, value in second.items() if key != "artifacts"})
