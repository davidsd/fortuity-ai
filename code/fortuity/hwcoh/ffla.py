"""Dense linear algebra mod p: FLINT 3 (flint/libffla.so, built with flint/build.sh) if available, else numpy.

Set FORTUITY_FFLA to the library path to override; FLINT_THREADS sets the FLINT thread count.
"""
import ctypes, os
import numpy as np
from fortuity.bmn.modp import P, RREFBasis

_path = os.environ.get('FORTUITY_FFLA', os.path.join(os.path.dirname(__file__), 'flint', 'libffla.so'))
try:
    _lib = ctypes.CDLL(_path)
    HAVE_FLINT = True
except OSError:
    _lib = None
    HAVE_FLINT = False

if HAVE_FLINT:
    _LP = ctypes.POINTER(ctypes.c_long)
    _lib.ff_rank.restype = ctypes.c_long
    _lib.ff_rank.argtypes = [_LP, ctypes.c_long, ctypes.c_long, ctypes.c_ulong]
    _lib.ff_nullspace.restype = ctypes.c_long
    _lib.ff_nullspace.argtypes = [_LP, ctypes.c_long, ctypes.c_long, ctypes.c_ulong, _LP]
    _lib.ff_set_threads.argtypes = [ctypes.c_int]
    # default 1 thread; forked workers must never call FLINT when the parent has a FLINT thread pool
    _lib.ff_set_threads(int(os.environ.get('FLINT_THREADS', '1')))


def _prep(A, p):
    A = np.ascontiguousarray(np.asarray(A, dtype=np.int64) % p)
    return A, A.ctypes.data_as(_LP)


def rank(A, p=P):
    A = np.asarray(A)
    if A.size == 0:
        return 0
    if not HAVE_FLINT:
        b = RREFBasis(A.shape[1], p)
        b.add_batched(A % p, 512)
        return b.rank
    A, ptr = _prep(A, p)
    return int(_lib.ff_rank(ptr, A.shape[0], A.shape[1], p))


def nullspace_right(A, p=P):
    """Rows spanning {x : A x = 0}."""
    A = np.asarray(A)
    r, c = A.shape
    if not HAVE_FLINT:
        b = RREFBasis(c, p)
        if r:
            b.add_batched(A % p, 512)
        piv = np.array(b.piv, dtype=np.int64)
        free = np.setdiff1d(np.arange(c), piv)
        X = np.zeros((len(free), c), dtype=np.int64)
        X[np.arange(len(free)), free] = 1
        if len(piv):
            X[:, piv] = (-b.B[:b.rank].astype(np.int64)[:, free].T) % p
        return X
    A, ptr = _prep(A, p)
    out = np.zeros((c, c), dtype=np.int64)
    nul = _lib.ff_nullspace(ptr, r, c, p, out.ctypes.data_as(_LP))
    return out[:nul].copy()
