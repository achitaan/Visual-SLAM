"""Download selected grayscale KITTI ZIP members using HTTP ranges, with a disk-space bound."""
import argparse
from collections import OrderedDict
import io
import json
from pathlib import Path
import re
import shutil
import time
import urllib.error
import urllib.request
import zipfile
import zlib

URL = 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_gray.zip'


class RangeFile(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        self.position = 0
        self.cache = OrderedDict()
        self.prefetched = OrderedDict()
        self.block_size = 4 * 1024 * 1024
        self.max_blocks = 4
        self.bytes_downloaded = 0
        with urllib.request.urlopen(urllib.request.Request(url, headers={'Range': 'bytes=0-3'}), timeout=30) as response:
            if response.status != 206:
                raise ValueError('Server must support HTTP range requests')
            self.size = int(response.headers['Content-Range'].split('/')[-1])

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if self.position < 0:
            raise ValueError('Negative seek')
        return self.position

    def read(self, size=-1):
        size = min(size if size >= 0 else self.size - self.position, self.size - self.position)
        if size > 64 * 1024 * 1024:
            raise ValueError('Refusing a range larger than 64 MiB')
        if size <= 0:
            return b''
        for start, data in list(self.prefetched.items()):
            offset = self.position - start
            if offset >= 0 and offset + size <= len(data):
                self.prefetched.move_to_end(start)
                self.position += size
                return data[offset:offset + size]
        # ZIP central directories are contiguous; fetch them in one request.
        if size > self.block_size:
            data = self._fetch(self.position, size)
            self.position += len(data)
            return data
        chunks = []
        remaining = size
        while remaining:
            start = self.position // self.block_size * self.block_size
            if start not in self.cache:
                length = min(self.block_size, self.size - start)
                self.cache[start] = self._fetch(start, length)
                if len(self.cache) > self.max_blocks:
                    self.cache.popitem(last=False)
            self.cache.move_to_end(start)
            offset = self.position - start
            chunk = self.cache[start][offset:offset + remaining]
            chunks.append(chunk)
            self.position += len(chunk)
            remaining -= len(chunk)
        return b''.join(chunks)

    def prefetch(self, start, length):
        """Warm a single ZIP member, avoiding block read-ahead for shuffled images."""
        length = min(length, self.size - start)
        if length > 64 * 1024 * 1024 or start < 0 or length <= 0:
            raise ValueError('Invalid prefetch range')
        self.prefetched[start] = self._fetch(start, length)
        self.prefetched.move_to_end(start)
        while len(self.prefetched) > 4:
            self.prefetched.popitem(last=False)

    def _fetch(self, start, length):
        request = urllib.request.Request(self.url, headers={'Range': f'bytes={start}-{start + length - 1}'})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    expected = f'bytes {start}-{start + length - 1}/{self.size}'
                    if response.status != 206 or response.headers.get('Content-Range') != expected:
                        raise ValueError('Unexpected range response; refusing full-archive download')
                    data = response.read(length)
                    if len(data) != length:
                        raise OSError('Truncated HTTP range')
                    self.bytes_downloaded += length
                    return data
            except (urllib.error.URLError, OSError):
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence', required=True)
    parser.add_argument('--output-root', type=Path, default=Path('.datasets/kitti'))
    parser.add_argument('--inspect', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'(0[0-9]|1[0-9]|2[01])', args.sequence):
        parser.error('Sequence must be 00–21')
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(rf'dataset/sequences/{args.sequence}/(image_[01]/[0-9]{{6}}\.png|calib\.txt|times\.txt)')
    with zipfile.ZipFile(RangeFile(URL)) as archive:
        entries = sorted((entry for entry in archive.infolist() if pattern.fullmatch(entry.filename)), key=lambda entry: entry.header_offset)
        if not entries:
            raise ValueError('No selected sequence entries found')
        required = sum(entry.file_size for entry in entries)
        free = shutil.disk_usage(root).free
        print(json.dumps({'sequence': args.sequence, 'files': len(entries), 'uncompressed_bytes': required, 'free_bytes': free, 'source': URL}), flush=True)
        if args.inspect:
            return
        if free < required + 250 * 1024 * 1024:
            raise ValueError('Not enough free space for this sequence plus 250 MiB reserve')
        for i, entry in enumerate(entries):
            relative = Path(*Path(entry.filename).parts[1:])
            target = (root / relative).resolve()
            if not target.is_relative_to(root):
                raise ValueError('Archive member escapes output root')
            if target.exists() and target.stat().st_size == entry.file_size and zlib.crc32(target.read_bytes()) == entry.CRC:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            data = archive.read(entry)  # ZipFile verifies member CRC before saving.
            pending = target.with_suffix(target.suffix + '.part')
            pending.write_bytes(data)
            pending.replace(target)
            if (i + 1) % 100 == 0 or i + 1 == len(entries):
                print(f'Downloaded {i + 1}/{len(entries)} files', flush=True)
    (root / f'source-{args.sequence}.json').write_text(json.dumps({'url': URL, 'sequence': args.sequence, 'files': len(entries), 'uncompressed_bytes': required}, indent=2))


if __name__ == '__main__':
    main()
