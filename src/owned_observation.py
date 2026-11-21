"""Committed pixel VALUES, never shared mutable NumPy array metadata.

Supported numeric reads expose independent read-only snapshots. Ordinary
Observation and unsupported storage keep their original mutable fallback API.
"""
import copy
import math
import struct
import numpy as np
from slam_state import Observation

_FLOAT32_DTYPE = np.dtype(np.float32)
_UNPACK_2I = struct.Struct("<ii").unpack


class OwnedObservation(Observation):
    def __init__(self,pixel,right_u=None):
        self._pixel_state=None
        self._legacy_pixel=None
        self._identity_owned=False
        self._identity_key=None
        self._pixel_id_i32=None
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
        if (type(value) is np.ndarray and value.dtype == _FLOAT32_DTYPE
                and value.shape == (2,) and value.flags.c_contiguous):
            x = float(value[0])
            y = float(value[1])
            buf = value.tobytes()
            if math.isfinite(x) and math.isfinite(y):
                key = (x, y)
                pid_i32 = _UNPACK_2I(buf)
            else:
                key = None
                pid_i32 = None
            self._identity_owned = False
            self._pixel_state = (buf, _FLOAT32_DTYPE, (2,))
            self._identity_key = key
            self._pixel_id_i32 = pid_i32
            self._identity_owned = True
            self._legacy_pixel = None
            return
        raw=np.asarray(value)
        if not np.issubdtype(raw.dtype,np.number) or np.iscomplexobj(raw):
            self._identity_owned=False
            self._legacy_pixel=value
            self._pixel_state=None
            self._identity_key=None
            self._pixel_id_i32=None
            return
        normalized=np.asarray(raw,np.float32).reshape(-1)
        if normalized.shape==(2,) and np.isfinite(normalized).all():
            key=tuple(normalized.tolist())
            pid_i32=_UNPACK_2I(np.ascontiguousarray(normalized, _FLOAT32_DTYPE).tobytes())
        else:
            key=None
            pid_i32=None
        state=(np.ascontiguousarray(raw).tobytes(),raw.dtype,tuple(raw.shape))
        self._identity_owned=False
        self._pixel_state=state
        self._identity_key=key
        self._pixel_id_i32=pid_i32
        self._identity_owned=True
        self._legacy_pixel=None

    def __copy__(self):return type(self)(self.pixel,self.right_u)

    def __deepcopy__(self,memo):
        result=type(self)(copy.deepcopy(self.pixel,memo),copy.deepcopy(self.right_u,memo))
        memo[id(self)]=result
        return result

    def __reduce__(self):return type(self),(self.pixel,self.right_u)
