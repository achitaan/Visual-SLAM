"""Check KITTI availability before a long run, including OneDrive placeholders."""
import argparse
import json
from pathlib import Path


def check_sequence(root, sequence, limit):
    folder = root / 'sequences' / sequence
    result = {'sequence': sequence, 'ready': True, 'errors': []}
    names = []
    for camera in ('image_0', 'image_1'):
        paths = sorted((folder / camera).glob('*.png'))
        selected = paths[:limit] if limit else paths
        placeholders = [p for p in selected if getattr(p.stat(), 'st_file_attributes', 0) & (0x1000 | 0x400000)]
        result[camera] = {'frames': len(paths), 'requested': len(selected),
                          'cloud_only': len(placeholders), 'first_cloud_file': str(placeholders[0]) if placeholders else None}
        if not paths:
            result['errors'].append(f'No PNG images in {folder / camera}')
        names.append([p.name for p in paths])
        # Open the boundary images to check the provider, rather than assuming metadata means readable.
        for path in dict.fromkeys(selected[:1] + selected[-1:]):
            try:
                with path.open('rb') as handle:
                    if handle.read(8) != b'\x89PNG\r\n\x1a\n':
                        raise ValueError('Invalid PNG signature')
            except (OSError, ValueError) as error:
                result['errors'].append(f'{path}: {error}')
    if names[0] != names[1]:
        result['errors'].append('Left/right image names do not match')
    for path in (folder / 'calib.txt', root / 'poses' / f'{sequence}.txt'):
        try:
            if not path.read_text().strip():
                raise ValueError('Empty file')
        except (OSError, ValueError) as error:
            result['errors'].append(f'{path}: {error}')
    if any(result[camera]['cloud_only'] for camera in ('image_0', 'image_1')):
        result['errors'].append('Requested images include cloud-only files. Make this sequence available locally before benchmarking.')
    result['ready'] = not result['errors']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--sequences', nargs='+', default=['00', '03', '04'])
    parser.add_argument('--max-frames', type=int, default=300, help='0 checks full sequence; default checks first 300')
    args = parser.parse_args()
    if args.max_frames < 0 or any(len(s) != 2 or not s.isdigit() or int(s) > 21 for s in args.sequences):
        parser.error('Use sequence IDs 00–21 and a nonnegative frame limit')
    results = [check_sequence(args.data_root, s, args.max_frames) for s in args.sequences]
    print(json.dumps(results, indent=2))
    return 0 if all(result['ready'] for result in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
