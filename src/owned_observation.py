"""Committed pixel VALUES, never shared mutable NumPy array metadata.

Supported numeric reads expose independent read-only snapshots. Ordinary
Observation and unsupported storage keep their original mutable fallback API.
"""
import copy
import numpy as np
from slam_state import Observation


class OwnedObservation(Observation):
    def __init__(self,pixel,right_u=None):
        self._pixel_state=None
        self._legacy_pixel=None
        self._identity_owned=False
        self._identity_key=None
        self.pixel=pixel
        self.right_u=right_u

    @property
    def pixel(self):
        state=self._pixel_state
        if state is None:return self._legacy_pixel
        buffer,dtype,shape=state
        # Neither this ndarray nor its ndarray bases are stored in the object.
        # All exposed array metadata is disposable; only bytes are shared.
        return np.frombuffer(buffer,dtype=dtype).reshape(shape)

    @pixel.setter
    def pixel(self,value):
        raw=np.asarray(value)
        if not np.issubdtype(raw.dtype,np.number) or np.iscomplexobj(raw):
            self._identity_owned=False
            self._legacy_pixel=value
            self._pixel_state=None
            self._identity_key=None
            return
        normalized=np.asarray(raw,np.float32).reshape(-1)
        key=tuple(normalized.tolist()) if normalized.shape==(2,) and np.isfinite(normalized).all() else None
        state=(np.ascontiguousarray(raw).tobytes(),raw.dtype,tuple(raw.shape))
        self._identity_owned=False
        self._pixel_state=state
        self._identity_key=key
        self._identity_owned=True
        self._legacy_pixel=None

    def __copy__(self):return type(self)(self.pixel,self.right_u)

    def __deepcopy__(self,memo):
        result=type(self)(copy.deepcopy(self.pixel,memo),copy.deepcopy(self.right_u,memo))
        memo[id(self)]=result
        return result

    def __reduce__(self):return type(self),(self.pixel,self.right_u)
