"""Decode requested inputs and fingerprint their contents before estimator startup."""
import hashlib
from pathlib import Path
import cv2 as cv
import numpy as np


def preflight(root, sequence, frames, stereo=True, budget=None):
    directory = Path(root) / "sequences" / sequence
    digest = hashlib.sha256()
    calibration = (directory / "calib.txt").read_bytes()
    digest.update(calibration)
    timings = directory / "times.txt"
    if timings.exists():
        data = timings.read_bytes()
        values = np.loadtxt(timings)
        if len(values) < frames or not np.isfinite(values).all() or np.any(np.diff(values) <= 0):
            raise ValueError("Invalid frame timestamps")
        digest.update(data)
    shape = None
    cameras = [0, 1] if stereo else [0]
    for camera in cameras:
        paths = sorted((directory / f"image_{camera}").glob("*.png"))
        if len(paths) < frames:
            raise ValueError(f"Camera {camera}: expected at least {frames} images")
        for i, path in enumerate(paths[:frames]):
            if budget and budget.remaining <= 0:
                raise TimeoutError("Input preflight exceeded budget")
            if path.stem != f"{i:06d}":
                raise ValueError("Image numbering has gaps")
            data = path.read_bytes()
            image = cv.imdecode(np.frombuffer(data, np.uint8), cv.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Unreadable camera {camera} image {path.name}")
            if shape is None:
                shape = image.shape
            if image.shape != shape:
                raise ValueError("Image dimensions changed")
            digest.update(f"{camera}/{path.name}".encode())
            digest.update(data)
    return {"sha256": digest.hexdigest(), "frames": frames, "cameras": cameras, "shape": list(shape)}
