"""Mod-p matmul and rank on top of BLIS (flint/libffblas.so, built with flint/build_ffblas.sh); numpy/FLINT fallback.

Set FORTUITY_FFBLAS to the library path to override; FF_THREADS sets the OpenMP thread count.
"""
import ctypes, os
import numpy as np
from fortuity.bmn.modp import P, matmul_mod

_path = os.environ.get('FORTUITY_FFBLAS', os.path.join(os.path.dirname(__file__), 'flint', 'libffblas.so'))
try:
    _lib = ctypes.CDLL(_path)
    HAVE_FFBLAS = True
except OSError:
    _lib = None
    HAVE_FFBLAS = False

_LP = ctypes.POINTER(ctypes.c_int64)
_I32 = ctypes.POINTER(ctypes.c_int32)
if HAVE_FFBLAS:
    _lib.ff_matmul.argtypes = [_LP, ctypes.c_long, ctypes.c_long, _LP, ctypes.c_long, _LP, ctypes.c_int64]
    _lib.ff_rank2.restype = ctypes.c_long
    _lib.ff_rank2.argtypes = [_LP, ctypes.c_long, ctypes.c_long, ctypes.c_long, ctypes.c_int64, ctypes.c_long,
                              ctypes.POINTER(ctypes.c_long)]
    _lib.ff_matmul_b32c32.argtypes = [_LP, ctypes.c_long, ctypes.c_long, _I32, ctypes.c_long, ctypes.c_long, _I32,
                                      ctypes.c_long, ctypes.c_int64]
    _lib.ff_rank_pair_i32.restype = ctypes.c_long
    _lib.ff_rank_pair_i32.argtypes = [_I32, ctypes.c_long, ctypes.c_long, _I32, ctypes.c_long, ctypes.c_long,
                                      ctypes.c_long, ctypes.c_int64, ctypes.c_long, ctypes.POINTER(ctypes.c_long)]
    _lib.ff_nullspace.restype = ctypes.c_long
    _lib.ff_nullspace.argtypes = [_LP, ctypes.c_long, ctypes.c_long, ctypes.c_int64, ctypes.c_long, _LP]
    _lib.ff_basis_new.restype = ctypes.c_void_p
    _lib.ff_basis_new.argtypes = [ctypes.c_long, ctypes.c_int64]
    _lib.ff_basis_add.restype = ctypes.c_long
    _lib.ff_basis_add.argtypes = [ctypes.c_void_p, _LP, ctypes.c_long]
    _lib.ff_basis_rank.restype = ctypes.c_long
    _lib.ff_basis_rank.argtypes = [ctypes.c_void_p]
    _lib.ff_basis_nullspace.restype = ctypes.c_long
    _lib.ff_basis_nullspace.argtypes = [ctypes.c_void_p, _LP, ctypes.c_long]
    _lib.ff_basis_free.argtypes = [ctypes.c_void_p]
    _lib.ff_set_threads.argtypes = [ctypes.c_int]
    # built against MKL (build_ffblas_mkl.sh): MKL_NUM_THREADS takes precedence over omp_set_num_threads, so set MKL's
    # own count too (dlsym on the handle also finds symbols of its dependencies)
    try:
        _mkl_set = _lib.MKL_Set_Num_Threads
        _mkl_set.argtypes = [ctypes.c_int]
    except AttributeError:
        _mkl_set = None


def set_threads(t):
    if HAVE_FFBLAS:
        _lib.ff_set_threads(int(t))
        if _mkl_set is not None:
            _mkl_set(int(t))


if HAVE_FFBLAS:
    set_threads(int(os.environ.get('FF_THREADS', os.environ.get('SLURM_CPUS_PER_TASK', '1'))))


def _c(A, p):
    A = np.asarray(A)
    if A.dtype != np.int64 or not A.flags.c_contiguous or (A.size and (A.min() < 0 or A.max() >= p)):
        A = np.ascontiguousarray(np.asarray(A, dtype=np.int64) % p)
    return A


def matmul(A, B, p=P):
    if not HAVE_FFBLAS:
        return matmul_mod(np.asarray(A, dtype=np.int64) % p, np.asarray(B, dtype=np.int64) % p, p)
    A, B = _c(A, p), _c(B, p)
    m, k = A.shape
    n = B.shape[1]
    C = np.empty((m, n), dtype=np.int64)
    if m and n:
        _lib.ff_matmul(A.ctypes.data_as(_LP), m, k, B.ctypes.data_as(_LP), n, C.ctypes.data_as(_LP), p)
    return C


def matmul_into32(A, B32, C32, p=P):
    """C32[...] = A @ B32 mod p (B32, C32 int32 2-d views with unit column stride)."""
    if not HAVE_FFBLAS:
        C32[...] = matmul_mod(np.asarray(A, dtype=np.int64) % p, np.asarray(B32, dtype=np.int64), p).astype(np.int32)
        return
    A = _c(A, p)
    m, k = A.shape
    n = B32.shape[1]
    assert B32.dtype == np.int32 and C32.dtype == np.int32 and B32.shape[0] == k and C32.shape == (m, n)
    assert B32.strides[1] == 4 and C32.strides[1] == 4
    if m and n:
        _lib.ff_matmul_b32c32(A.ctypes.data_as(_LP), m, k, B32.ctypes.data_as(_I32), B32.strides[0] // 4, n,
                              C32.ctypes.data_as(_I32), C32.strides[0] // 4, p)


def rank_pair32(A1, A2=None, p=P, blk=4096):
    """(rank A1, rank [A1; A2]) for int32 2-d arrays with the same number of columns."""
    c = A1.shape[1]
    if A2 is None or A2.shape[0] == 0:
        A2 = None
    if A1.shape[0] == 0 and A2 is not None:        # rank [0 rows; A2] = rank A2
        return 0, rank_pair32(A2, None, p, blk)[0]
    if c == 0 or A1.shape[0] == 0:
        return 0, 0
    if A2 is None:
        A2 = A1[:1]                                  # non-empty dummy with valid strides, zero rows used
        r2 = 0
    else:
        r2 = A2.shape[0]
    if not HAVE_FFBLAS:
        from . import ffla
        r1 = ffla.rank(np.asarray(A1, dtype=np.int64))
        if r2 == 0:
            return r1, r1
        return r1, ffla.rank(np.concatenate([np.asarray(A1, dtype=np.int64), np.asarray(A2, dtype=np.int64)]))
    for X in (A1, A2):
        assert X.dtype == np.int32 and X.strides[1] == 4 and X.shape[1] == c
    r1 = ctypes.c_long(0)
    tot = _lib.ff_rank_pair_i32(A1.ctypes.data_as(_I32), max(A1.strides[0] // 4, c), A1.shape[0],
                                A2.ctypes.data_as(_I32), max(A2.strides[0] // 4, c), r2, c, p, blk,
                                ctypes.byref(r1))
    return int(r1.value), int(tot)


def nullspace_right(A, p=P, blk=4096):
    """Rows spanning {x : A x = 0 mod p} (BLIS kernel; FLINT/numpy fallback)."""
    if not HAVE_FFBLAS:
        from . import ffla
        return ffla.nullspace_right(A, p)
    A = _c(A, p)
    r, c = A.shape
    if c == 0:
        return np.zeros((0, 0), dtype=np.int64)
    if r == 0:
        return np.eye(c, dtype=np.int64)
    out = np.empty((c, c), dtype=np.int64)
    nul = _lib.ff_nullspace(A.ctypes.data_as(_LP), r, c, p, blk, out.ctypes.data_as(_LP))
    return out[:nul].copy()


class StreamBasis:
    """Row-echelon basis fed by row blocks (BLIS kernel); null_space() returns the rows x with v.x = 0 for all fed v."""

    def __init__(self, ncols, p=P):
        assert HAVE_FFBLAS
        self.c, self.p = ncols, p
        self.h = _lib.ff_basis_new(ncols, p)

    def add(self, R):
        R = _c(R, self.p)
        if R.shape[0]:
            _lib.ff_basis_add(self.h, R.ctypes.data_as(_LP), R.shape[0])

    @property
    def rank(self):
        return int(_lib.ff_basis_rank(self.h))

    def null_space(self):
        nul = self.c - self.rank
        out = np.empty((max(nul, 1), self.c), dtype=np.int64)
        got = _lib.ff_basis_nullspace(self.h, out.ctypes.data_as(_LP), nul)
        assert got == nul, (got, nul)
        return out[:nul].copy()

    def __del__(self):
        if getattr(self, 'h', None):
            _lib.ff_basis_free(self.h)
            self.h = None
