"""EuRoC stereo calibration and timestamp association; never reads reference poses.

T_A_B maps coordinates in B into A. Camera T_BS is body-from-camera.
Both estimators must consume the same prepared rectified image files.
"""
from dataclasses import dataclass
from pathlib import Path
import csv
import hashlib
import json
import re
import cv2 as cv
import numpy as np
import yaml
from kitti import validate_pose


@dataclass(frozen=True)
class CameraCalibration:
    matrix: np.ndarray
    distortion: np.ndarray
    body_from_camera: np.ndarray
    size: tuple[int, int]


def read_transform(value):
    if isinstance(value, dict):
        if value.get('rows') != 4 or value.get('cols') != 4:
            raise ValueError('Expected a 4x4 sensor transform')
        value = value['data']
    transform = np.asarray(value, dtype=float).reshape(4, 4)
    validate_pose(transform)
    return transform


def read_camera(path):
    data = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if data.get('camera_model') != 'pinhole' or data.get('distortion_model') != 'radial-tangential':
        raise ValueError('EuRoC input requires calibrated pinhole/radial-tangential cameras')
    intrinsics = np.asarray(data['intrinsics'], float)
    distortion = np.asarray(data['distortion_coefficients'], float)
    size = tuple(data['resolution'])
    if (intrinsics.shape != (4,) or distortion.shape != (4,)
            or not np.isfinite(intrinsics).all() or not np.isfinite(distortion).all()
            or np.any(intrinsics[:2] <= 0) or len(size) != 2
            or any(not isinstance(n, int) or n <= 0 for n in size)):
        raise ValueError('Invalid camera intrinsics, distortion or resolution')
    fx, fy, cx, cy = intrinsics
    return CameraCalibration(np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]]),
                             distortion, read_transform(data['T_BS']), size)


def read_catalog(camera_folder):
    folder = Path(camera_folder)
    entries = []
    with (folder/'data.csv').open(newline='', encoding='utf-8') as handle:
        for row in csv.reader(handle):
            if not row or row[0].lstrip().startswith('#'):
                continue
            if len(row) != 2:
                raise ValueError('Camera catalog rows must contain timestamp and filename')
            timestamp = int(row[0].strip())  # Preserve nanoseconds; never parse as float.
            filename = row[1].strip()
            if (timestamp < 0 or Path(filename).name != filename
                    or not filename.endswith('.png') or '\\' in filename):
                raise ValueError('Invalid camera timestamp or filename')
            if entries and timestamp <= entries[-1][0]:
                raise ValueError('Camera timestamps must be strictly increasing and unique')
            path = folder/'data'/filename
            if not path.is_file():
                raise FileNotFoundError(path)
            entries.append((timestamp, path))
    if len(entries) < 2:
        raise ValueError('Need at least two camera frames')
    return entries


def associate_stereo(left, right, tolerance_ns=1_000_000):
    """One-to-one monotonic pairs; unmatched inputs are returned, never hidden."""
    if not isinstance(tolerance_ns, int) or tolerance_ns < 0:
        raise ValueError('Timestamp tolerance must be nonnegative integer nanoseconds')
    for rows in (left, right):
        if not rows or any(b[0] <= a[0] for a, b in zip(rows, rows[1:])):
            raise ValueError('Catalog timestamps must be strictly increasing')
        if any(b[0]-a[0] <= 2*tolerance_ns for a, b in zip(rows, rows[1:])):
            raise ValueError('Timestamp tolerance permits ambiguous stereo association')
    pairs, missing_left, missing_right = [], [], []
    i = j = 0
    while i < len(left) and j < len(right):
        delta = right[j][0]-left[i][0]
        if abs(delta) <= tolerance_ns:
            pairs.append((left[i], right[j])); i += 1; j += 1
        elif delta > 0:
            missing_left.append(left[i]); i += 1
        else:
            missing_right.append(right[j]); j += 1
    return pairs, missing_left+left[i:], missing_right+right[j:]


class StereoRectification:
    def __init__(self, left, right):
        if left.size != right.size:
            raise ValueError('Stereo cameras must share an image resolution')
        right_from_left = np.linalg.inv(right.body_from_camera) @ left.body_from_camera
        if np.linalg.norm(right_from_left[:3, 3]) <= 1e-6:
            raise ValueError('Stereo baseline must be nonzero')
        self.left, self.right = left, right
        self.R_left, self.R_right, self.P_left, self.P_right, self.Q, _, _ = cv.stereoRectify(
            left.matrix, left.distortion, right.matrix, right.distortion, left.size,
            np.ascontiguousarray(right_from_left[:3, :3]),
            np.ascontiguousarray(right_from_left[:3, 3].reshape(3, 1)),
            flags=cv.CALIB_ZERO_DISPARITY, alpha=0)
        self.baseline = float(-self.P_right[0, 3]/self.P_right[0, 0])
        if self.baseline <= 0 or abs(self.P_right[1, 3]) > 1e-8:
            raise ValueError('Tracker requires horizontal positive-disparity stereo')
        self.maps = [cv.initUndistortRectifyMap(c.matrix, c.distortion, rotation,
                        projection[:3, :3], c.size, cv.CV_32FC1)
                     for c, rotation, projection in ((left,self.R_left,self.P_left),
                                                       (right,self.R_right,self.P_right))]
        camera_from_rectified = np.eye(4)
        camera_from_rectified[:3, :3] = self.R_left.T
        self.body_from_rectified_left = left.body_from_camera @ camera_from_rectified

    def rectify(self, left, right):
        expected = self.left.size[::-1]
        if left is None or right is None or left.shape != expected or right.shape != expected:
            raise ValueError('Stereo images do not match calibration resolution')
        return tuple(cv.remap(image, *maps, interpolation=cv.INTER_LINEAR,
                              borderMode=cv.BORDER_CONSTANT)
                     for image, maps in zip((left,right), self.maps))


def load_prepared(root, sequence, max_frames=None):
    """Verify shared prepared inputs before either estimator consumes them."""
    if not re.fullmatch(r'[A-Za-z0-9_-]+',sequence):
        raise ValueError('Invalid EuRoC sequence name')
    folder = Path(root)/'sequences'/sequence
    manifest = json.loads((folder/'euroc.json').read_text(encoding='utf-8'))
    if (manifest.get('dataset') != 'euroc' or manifest.get('status') != 'completed'
            or manifest.get('sequence') != sequence or manifest.get('format_version') != 1):
        raise ValueError('Invalid or incomplete EuRoC preparation')
    count = manifest['frames']
    timestamps = np.asarray(manifest['timestamps_ns'],np.int64)
    if count < 2 or len(timestamps) != count or np.any(np.diff(timestamps) <= 0):
        raise ValueError('Invalid prepared frame timestamps')
    if max_frames is not None and (max_frames < 2 or max_frames > count):
        raise ValueError('Requested prefix exceeds prepared input')
    if max_frames is None and not manifest['complete_sequence']:
        raise ValueError('Prepared input is partial; request an explicit diagnostic prefix')
    requested = max_frames or count
    files = ['calib.txt','times.txt'] + [f'image_{camera}/{i:06d}.png'
                for i in range(requested) for camera in range(2)]
    for filename in files:
        actual = hashlib.sha256((folder/filename).read_bytes()).hexdigest()
        if actual != manifest['prepared_sha256'].get(filename):
            raise ValueError(f'Prepared input changed: {filename}')
    for camera in range(2):
        expected = [f'{i:06d}.png' for i in range(count)]
        if sorted(p.name for p in (folder/f'image_{camera}').glob('*.png')) != expected:
            raise ValueError('Prepared image catalog changed')
    return folder,manifest
