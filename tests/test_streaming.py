"""Bounded remote reads must survive interleaved stereo access and reject bad data."""
from collections import OrderedDict
import io
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile
import cv2 as cv
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from download_kitti_sequence import RangeFile
from run_kitti_stream import RemoteImages, GeometryStore, write_json, reuse_complete
from kitti import save_poses_txt
from loop_geometry import StereoLoopFrame


def test_range_cache_interleaving_cross_boundary_and_eviction(monkeypatch):
    payload = bytes(range(251)) * 50
    requests = []
    class Response(io.BytesIO):
        status = 206
        def __init__(self, start, end):
            super().__init__(payload[start:end + 1])
            self.headers = {'Content-Range': f'bytes {start}-{end}/{len(payload)}'}
    def open_range(request, timeout):
        start, end = map(int, request.headers['Range'].split('=')[1].split('-'))
        requests.append((start, end))
        return Response(start, end)
    monkeypatch.setattr('urllib.request.urlopen', open_range)
    remote = RangeFile('https://example.test/data.zip')
    remote.block_size, remote.max_blocks = 256, 2
    for position in [250, 800, 251, 801]:
        remote.seek(position)
        assert remote.read(3) == payload[position:position + 3]
    assert len(requests) == 3  # Initial size probe and two stereo blocks.
    remote.seek(255)
    assert remote.read(5) == payload[255:260]
    assert len(remote.cache) == 2
    remote.seek(-4, 2)
    assert remote.read(10) == payload[-4:]
    assert remote.read() == b''
    with pytest.raises(ValueError, match='Negative'):
        remote.seek(-1)


def test_full_archive_response_rejected(monkeypatch):
    remote = RangeFile.__new__(RangeFile)
    remote.url, remote.position, remote.size = 'https://example.test/data.zip', 0, 1000
    remote.cache, remote.block_size, remote.max_blocks, remote.bytes_downloaded = OrderedDict(), 256, 4, 0
    remote.prefetched = OrderedDict()
    class Response(io.BytesIO):
        status = 200
        headers = {}
    monkeypatch.setattr('urllib.request.urlopen', lambda *a, **k: Response(b'x' * 1000))
    with pytest.raises(ValueError, match='refusing full-archive'):
        remote.read(4)


def test_prefetched_member_header_and_body_use_one_request(monkeypatch):
    payload = b'a' * 10000
    calls = []
    class Response(io.BytesIO):
        status = 206
        def __init__(self, start, end):
            super().__init__(payload[start:end + 1])
            self.headers = {'Content-Range': f'bytes {start}-{end}/{len(payload)}'}
    def fetch(request, timeout):
        start, end = map(int, request.headers['Range'].split('=')[1].split('-'))
        calls.append((start, end))
        return Response(start, end)
    monkeypatch.setattr('urllib.request.urlopen', fetch)
    remote = RangeFile('https://example.test/data.zip')
    remote.prefetch(7000, 500)
    remote.seek(7000)
    assert remote.read(30) == payload[7000:7030]
    assert remote.read(470) == payload[7030:7500]
    assert calls == [(0, 3), (7000, 7499)]


def test_decoded_image_sequence_and_corrupt_member():
    memory = io.BytesIO()
    originals = [np.full((12, 20), index, np.uint8) for index in range(8)]
    with zipfile.ZipFile(memory, 'w') as archive:
        for index, original in enumerate(originals):
            archive.writestr(f'{index:06d}.png', cv.imencode('.png', original)[1].tobytes())
        archive.writestr('bad.png', b'not an image')
    memory.seek(0)
    with zipfile.ZipFile(memory) as archive:
        images = RemoteImages(archive, archive.infolist()[:-1])
        for index in range(8):
            assert np.array_equal(images[index], originals[index])
        assert len(images.cache) == 4
        assert np.array_equal(images[-1], originals[-1])
        with pytest.raises(IndexError):
            images[8]
        with pytest.raises(ValueError, match='Cannot decode'):
            RemoteImages(archive, [archive.infolist()[-1]])[0]


def test_geometry_spill_roundtrip_and_cache_bound(tmp_path):
    store = GeometryStore(tmp_path)
    for index in range(6):
        store.append(StereoLoopFrame(np.ones((5, 2), np.float32) * index,
                                    np.ones((5, 3), np.float32), np.ones((5, 128), np.float32), (1241, 376)))
    for index, frame in enumerate(store):
        assert np.all(frame.pixels == index)
        assert frame.image_size == (1241, 376)
    assert len(store.cache) == 4


def test_status_write_recovers_transient_windows_reader_lock(tmp_path, monkeypatch):
    replace = Path.replace
    calls = []
    def locked_twice(self, target):
        calls.append(target)
        if len(calls) < 3:
            raise PermissionError('temporary sharing violation')
        return replace(self, target)
    monkeypatch.setattr(Path, 'replace', locked_twice)
    monkeypatch.setattr('run_kitti_stream.time.sleep', lambda _: None)
    output = tmp_path / 'status.json'
    write_json(output, {'status': 'running'})
    assert len(calls) == 3
    assert output.read_text().find('running') >= 0
    assert not output.with_suffix('.json.part').exists()


def test_reuse_rejects_partial_coverage_or_changed_constraints(tmp_path):
    poses = [np.eye(4), np.eye(4)]
    poses[1][0, 3] = 1
    gt_root = tmp_path / 'reference'
    save_poses_txt(gt_root / '04.txt', poses)
    for kind in ('stereo', 'graph'):
        save_poses_txt(tmp_path / kind / '04.txt', poses)
        write_json(tmp_path / kind / '04.json', {'status': 'complete', 'frames': 2})
    digest = hashlib.sha256((tmp_path / 'stereo/04.txt').read_bytes()).hexdigest()
    cache = tmp_path / 'graph/04-constraints.json'
    write_json(cache, {'raw_trajectory_sha256': digest})
    args = SimpleNamespace(output_root=tmp_path, poses_root=gt_root, sequence='04', force_retest=False, max_frames=None, graph_from_raw=False)
    assert reuse_complete(args)
    write_json(cache, {'raw_trajectory_sha256': 'changed'})
    assert not reuse_complete(args)
    write_json(cache, {'raw_trajectory_sha256': digest})
    save_poses_txt(tmp_path / 'graph/04.txt', poses[:1])
    assert not reuse_complete(args)


def test_reuse_cli_finishes_status_without_downloading(tmp_path, monkeypatch):
    import json
    import run_kitti_stream
    poses = [np.eye(4), np.eye(4)]
    poses[1][0, 3] = 1
    gt_root = tmp_path / 'reference'
    save_poses_txt(gt_root / '04.txt', poses)
    for kind in ('stereo', 'graph'):
        save_poses_txt(tmp_path / kind / '04.txt', poses)
        write_json(tmp_path / kind / '04.json', {'status': 'complete', 'frames': 2})
    digest = hashlib.sha256((tmp_path / 'stereo/04.txt').read_bytes()).hexdigest()
    write_json(tmp_path / 'graph/04-constraints.json', {'raw_trajectory_sha256': digest})
    status = tmp_path / '04-status.json'
    write_json(status, {'status': 'running', 'phase': 'graph'})
    monkeypatch.setattr(run_kitti_stream, 'REPO', tmp_path)
    monkeypatch.setattr(run_kitti_stream, 'RangeFile', lambda *args: pytest.fail('A complete cached run must not download'))
    monkeypatch.setattr('sys.argv', ['run_kitti_stream.py', '--sequence', '04', '--poses-root', str(gt_root), '--output-root', str(tmp_path)])
    run_kitti_stream.main()
    report = json.loads(status.read_text())
    assert report['status'] == 'complete' and report['phase'] == 'finished'
    assert report['reused_validated_reports']
    assert not (tmp_path / '.datasets').exists()
