"""Compare retained KITTI reference poses with the official public pose archive."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen
import zipfile
import numpy as np
from json_output import write_json

URL = 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_poses.zip'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=Path('results/benchmark-batch'))
    args = parser.parse_args()
    with urlopen(URL, timeout=30) as response:
        body = response.read(8 * 1024 * 1024 + 1)
    if len(body) > 8 * 1024 * 1024:
        raise ValueError('Public ground-truth download exceeded its expected size bound')
    results = []
    with zipfile.ZipFile(io.BytesIO(body)) as archive:
        for sequence in [f'{index:02d}' for index in range(11)]:
            entries = [entry for entry in archive.infolist() if entry.filename.split('/')[-2:] == ['poses', f'{sequence}.txt']]
            if len(entries) != 1:
                raise ValueError(f'Missing or ambiguous public poses for {sequence}')
            entry = entries[0]
            if entry.file_size > 16 * 1024 * 1024:
                raise ValueError('Unexpected public pose member size')
            data = archive.read(entry)  # Validates CRC.
            official = np.loadtxt(io.BytesIO(data), ndmin=2)
            local_path = args.output_root / 'reference/poses' / f'{sequence}.txt'
            local = np.loadtxt(local_path, ndmin=2)
            same_shape = official.shape == local.shape and official.shape[1] == 12
            difference = float(np.max(np.abs(local - official))) if same_shape else None
            results.append(dict(sequence=sequence, matching_shape=same_shape, official_frames=len(official),
                                max_pose_element_difference=difference, within_tolerance=bool(same_shape and difference <= 1e-10),
                                official_member_sha256=hashlib.sha256(data).hexdigest(), local_file_sha256=hashlib.sha256(local_path.read_bytes()).hexdigest(),
                                official_member_crc32=f'{entry.CRC:08x}'))
    reference = args.output_root / 'reference'
    reference.mkdir(parents=True, exist_ok=True)
    (reference / 'official-poses.zip').write_bytes(body)
    report = dict(url=URL, archive_bytes=len(body), archive_sha256=hashlib.sha256(body).hexdigest(),
                  pose_element_tolerance=1e-10, passed=all(row['within_tolerance'] for row in results), results=results)
    write_json(args.output_root / 'ground-truth-verification.json', report)
    print(json.dumps(report, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
