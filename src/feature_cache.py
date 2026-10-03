"""Optional bounded diagnostic cache; cache keys include exact extraction and image identity."""
import hashlib
import json
import inspect
import platform
from dataclasses import asdict
import stereo_depth
from pathlib import Path
import numpy as np


def extraction_signature(slam, opencv):
    """Fingerprint extraction dependencies without invalidating features for pose changes."""
    def parameters(algorithm):
        return {'type': type(algorithm).__module__+'.'+type(algorithm).__qualname__,
                'parameters': {name: getattr(algorithm, name)() for name in sorted(dir(algorithm))
                if name.startswith('get') and callable(getattr(algorithm, name))}}

    settings = {
        'schema': 2, 'opencv': opencv.__version__, 'numpy': np.__version__,
        'machine': platform.machine(), 'processor': platform.processor(),
        'detector': parameters(slam.detector),
        'stereo': parameters(slam.stereo.stereo) if slam.stereo is not None else None,
        'stereo_depth_policy': slam.config.stereo_depth_policy,
        'stereo_pose_arbitration': bool(getattr(slam.config, 'stereo_pose_arbitration', False)),
        'stereo_raw_reference_retry': bool(getattr(slam.config, 'stereo_raw_reference_retry', False)),
        'stereo_search_config': asdict(slam.stereo_search_config),
    }
    if slam.stereo is not None:
        settings['stereo_calibration'] = {
            'baseline': float(slam.stereo.baseline),
            'disparity_offset': float(slam.stereo.disparity_offset),
        }
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
    digest.update(opencv.getBuildInformation().encode())
    digest.update(Path(stereo_depth.__file__).read_bytes())
    scorer_source = Path(__file__).resolve().with_name('stereo_pose_arbitration.py')
    if scorer_source.is_file():
        digest.update(b'stereo_pose_arbitration.py\0')
        digest.update(scorer_source.read_bytes())
    method_names = ('__init__', '_extract', '_measure_stereo_pixels',
                    '_measure_supported_stereo_pixels', '_capture_supported_stereo',
                    '_prepare_stereo_arbitration', '_prepare_frame_images', 'process')
    for name in method_names:
        method = getattr(type(slam), name, None)
        if method is not None:
            digest.update(name.encode()+b'\0')
            digest.update(inspect.getsource(method).encode())
    for value in (slam.K, slam.inverse_K):
        digest.update(value.dtype.str.encode());digest.update(value.tobytes())
    if slam.stereo is not None:
        digest.update(slam.stereo.Q.dtype.str.encode());digest.update(slam.stereo.Q.tobytes())
    return digest.hexdigest()


class FeatureCache:
    def __init__(self, folder, signature, max_bytes=512*1024**2):
        self.root=Path(folder).resolve()
        self.signature=signature
        self.folder=self.root/signature
        self.folder.mkdir(parents=True,exist_ok=True)
        (self.folder/'owner.json').write_text(json.dumps({'purpose':'shared-slam-diagnostic-feature-cache','signature':signature}),encoding='utf-8')
        self.max_bytes=max_bytes;self.hits=0;self.misses=0

    def key(self,image,right):
        digest=hashlib.sha256()
        for array in (image,right):
            if array is None:digest.update(b'none');continue
            digest.update(str((array.shape,array.dtype.str)).encode());digest.update(array.tobytes())
        return digest.hexdigest()

    def get(self,key):
        path=self.folder/(key+'.npz')
        if not path.exists():self.misses+=1;return None
        try:
            with np.load(path,allow_pickle=False) as data:
                values=tuple(data[name].copy() for name in ('pixels','descriptors','points','right_u','disparity'))
            self.hits+=1;path.touch();return values
        except (OSError,ValueError,KeyError):
            self.misses+=1;return None

    def put(self,key,values):
        path=self.folder/(key+'.npz');temporary=path.with_suffix('.part')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream,**dict(zip(('pixels','descriptors','points','right_u','disparity'),values)))
        temporary.replace(path)
        files=[]
        for marker in self.root.glob('*/owner.json'):
            try:
                owner=json.loads(marker.read_text(encoding='utf-8'))
            except (ValueError,OSError):continue
            directory=marker.parent.resolve()
            if directory.is_relative_to(self.root) and owner.get('purpose')=='shared-slam-diagnostic-feature-cache':
                files.extend(directory.glob('*.npz'))
        files.sort(key=lambda p:p.stat().st_mtime)
        size=sum(p.stat().st_size for p in files)
        for old in files:
            if size<=self.max_bytes:break
            size-=old.stat().st_size;old.unlink()

    def metadata(self):return {'enabled':True,'signature':self.signature,'hits':self.hits,'misses':self.misses,'runtime_is_cached_diagnostic':True}
