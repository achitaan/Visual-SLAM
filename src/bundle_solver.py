"""Equivalent sparse BA products with local solver dispatch specialization.

SciPy functions and installed files are never mutated. Unsupported kernels or
operand types use the original implementation. Inner/outer solve rules stay
identical; this only removes nested LinearOperator calls for real CSR vectors.
"""
import hashlib
import importlib
import inspect
from pathlib import Path
from types import FunctionType

import numpy as np
import scipy
from scipy.optimize import least_squares as original_least_squares
from scipy.sparse import issparse
from scipy.sparse.linalg import LinearOperator, aslinearoperator

_trf = importlib.import_module('scipy.optimize._lsq.trf')
_common = importlib.import_module('scipy.optimize._lsq.common')
_util = importlib.import_module('scipy._lib._util')
_original_factory = _trf.right_multiplied_operator
EXPECTED_KERNEL_HASHES = {'scipy.optimize._lsq.least_squares': 'b768b647932fea1329a9e44df3b3b1cc69635dc79a43ad36e402f41799dbe7b8', 'scipy.optimize._lsq.trf': 'a174a6e79b67067823804666c9e97413dd0d432c7ac7cd148c0c9ee5cfa9732b', 'scipy.optimize._lsq.common': '538075ef250c1936456aeab1be8f5697196a777b4fcc9f4c4b68e5fb4064cb1e', 'scipy.sparse.linalg._interface': '21c9a35a714f63113d6fcebe24632b543462feb650f6003fda28ef5fb2ad07f1', 'scipy._lib._util': 'f817a92102229b51a3f1a2c0435f1347250ffe4812f4e5406a10e8a0e3c0eac2'}


class ScaledSparseOperator(LinearOperator):
    def __init__(self,matrix,scale):
        super().__init__(dtype=matrix.dtype,shape=matrix.shape)
        self.matrix = matrix
        self.operator = aslinearoperator(matrix)
        self.transpose = None
        self.scale = scale
        self.fallback = _original_factory(self.operator,scale)

    def matvec(self,vector):
        if type(vector) is np.ndarray and vector.shape==(self.shape[1],) and vector.dtype==np.float64:
            return self.matrix @ (vector * self.scale)
        return self.fallback.matvec(vector)

    def rmatvec(self,vector):
        if type(vector) is np.ndarray and vector.shape==(self.shape[0],) and vector.dtype==np.float64:
            if self.transpose is None:
                # Share SciPy's lazy conjugate snapshot with every fallback
                # path, including matrix inputs and mutable CSR operands.
                self.transpose = self.operator.H.A
            return self.scale * (self.transpose @ vector)
        return self.fallback.rmatvec(vector)

    def _matvec(self,vector):
        return self.matvec(vector)

    def _rmatvec(self,vector):
        return self.rmatvec(vector)

    def _matmat(self,matrix):
        return self.fallback.matmat(matrix)

    def _rmatmat(self,matrix):
        return self.fallback.rmatmat(matrix)


def scaled_sparse_operator(matrix,scale):
    if (issparse(matrix) and matrix.format=='csr' and matrix.dtype==np.float64
            and type(scale) is np.ndarray and scale.shape==(matrix.shape[1],)
            and scale.dtype==np.float64):
        return ScaledSparseOperator(matrix,scale)
    return _original_factory(matrix,scale)


def _clone(function,replacements):
    result=FunctionType(function.__code__,{**function.__globals__,**replacements},
                        name=function.__name__,argdefs=function.__defaults__,
                        closure=function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    result.__annotations__ = function.__annotations__
    return result


def _specialized_solver():
    if scipy.__version__!='1.18.1' or not EXPECTED_KERNEL_HASHES:
        return original_least_squares
    for name,digest in EXPECTED_KERNEL_HASHES.items():
        module=importlib.import_module(name)
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()!=digest:
            return original_least_squares
    no_bounds=_clone(_trf.trf_no_bounds,{'right_multiplied_operator':scaled_sparse_operator})
    trf=_clone(_trf.trf,{'trf_no_bounds':no_bounds})
    least=_clone(inspect.unwrap(original_least_squares),{'trf':trf})
    # Preserve SciPy's worker-map lifetime wrapper as well as its solver code.
    return _util._workers_wrapper(least)


least_squares = _specialized_solver()
