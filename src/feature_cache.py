"""Optional bounded diagnostic cache; cache keys include exact extraction and image identity."""
import hashlib
import json
import inspect
import platform
from pathlib import Path
import numpy as np


def extraction_signature(slam, opencv):
    """Fingerprint extraction dependencies without invalidating features for pose changes."""
    def parameters(algorithm):
        return {'type': type(algorithm).__module__+'.'+type(algorithm).__qualname__,
                'parameters': {name: getattr(algorithm, name)() for name in sorted(dir(algorithm))
                if name.startswith('get') and callable(getattr(algorithm, name))}}

    settings = {
        'schema': 1, 'opencv': opencv.__version__, 'numpy': np.__version__,
        'machine': platform.machine(), 'processor': platform.processor(),
        'detector': parameters(slam.detector),
        'stereo': parameters(slam.stereo.stereo) if slam.stereo is not None else None,
    }
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
    digest.update(opencv.getBuildInformation().encode())
    for method in (type(slam).__init__, type(slam)._extract, type(slam)._measure_stereo_pixels):
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
