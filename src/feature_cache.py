"""Optional bounded diagnostic cache; cache keys include exact extraction and image identity."""
import hashlib
import json
from pathlib import Path
import numpy as np


class FeatureCache:
    def __init__(self, folder, signature, max_bytes=512*1024**2):
        self.root=Path(folder).resolve()
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

    def metadata(self):return {'enabled':True,'hits':self.hits,'misses':self.misses,'runtime_is_cached_diagnostic':True}
