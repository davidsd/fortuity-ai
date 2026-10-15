"""Explicit non-graviton representatives from the SU(3)_R highest-weight complex (T-028, result R-016).

hw_nongraviton_reps(N, group, lam, n, dims, GA): closed highest-weight vectors of the sector (2J = n; lam) that are
independent modulo Q-exact + graviton, returned as formal multi-trace expressions (coefficients mod p).  Same linear
algebra as fortuity.hwcoh.restrict (left kernel of the lower Q rows = closed combinations; rank increase over the
exact + graviton rows = non-graviton), with the per-degree faithfulness check of Sector.run (rank of the hw vectors at
the chosen functionals equals the exact multiplicity from the Haar integral); the functionals are doubled if not.
"""
import time
import numpy as np

from fortuity.bmn.modp import P, matmul_mod
from . import ffla
from .sector import Sector
from .hw import left_kernel
from .restrict import _lower_q_rows, _rank_increase


def _faithful(S, A):
    col, rk, ok = 0, {}, True
    for d in sorted(S.m_d):
        w = sum(fr.ncols for dd, fr in S.colblocks if dd == d)
        rows = np.flatnonzero(A[:, col:col + w].any(axis=1))
        rk[d] = ffla.rank(A[rows, col:col + w]) if rows.size else 0
        ok &= rk[d] == S.m_d[d]
        col += w
    return ok, rk


def hw_nongraviton_reps(N, group, lam, n, dims, GA, seed=1, maxlen=None, log=print):
    """Returns (exprs, info).  exprs: non-graviton representatives (formal expressions, mod p).  info: m (hw
    multiplicity), closed, r_exact, h = closed - r_exact, grav (graviton classes mod exact), nongrav, timing."""
    t0 = time.time()
    S = Sector(N, lam, n, dims, seed=seed, log=log, graviton_algebra=GA, maxlen=maxlen, group=group)
    S.prepare()
    for attempt, factor in enumerate((1, 2, 4)):
        A, B, G = S.evaluate(factor, attempt)
        ok, rk = _faithful(S, A)
        log(f"  faithfulness {'OK' if ok else 'FAIL'}: rank per degree {rk} vs exact {S.m_d}")
        if ok:
            break
    else:
        raise RuntimeError('functionals not faithful')
    L = _lower_q_rows(S, dims) if n > 0 else np.zeros((A.shape[0], 0), dtype=np.int64)
    nA = A.shape[0]
    Kc = left_kernel(L) if L.shape[1] else np.eye(nA, dtype=np.int64)
    closed_vals = matmul_mod(Kc % P, A % P) if Kc.shape[0] else np.zeros((0, A.shape[1]), dtype=np.int64)
    r_exact = ffla.rank(B) if B.shape[0] else 0
    EG = np.concatenate([B, G], axis=0) if (B.size or G.size) else np.zeros((0, A.shape[1]), dtype=np.int64)
    r_eg = ffla.rank(EG) if EG.shape[0] else 0
    closed = ffla.rank(closed_vals) if closed_vals.shape[0] else 0
    idx = _rank_increase(EG, closed_vals)
    R = Kc[idx]
    exprs = []
    for row in R:
        rep = {}
        r0 = 0
        for widx, H in S.hw:
            zb = row[r0:r0 + H.shape[0]] % P
            r0 += H.shape[0]
            if not zb.any():
                continue
            c = matmul_mod(zb[None, :], H % P)[0]
            for wi, cc in zip(widx, c):
                if cc:
                    w = S.ws[int(wi)]
                    rep[w] = (rep.get(w, 0) + int(cc)) % P
        exprs.append({m: c for m, c in rep.items() if c})
    info = dict(group=group, N=N, lam=list(lam), n=n, m=int(S.m), closed=int(closed), r_exact=int(r_exact),
                grav=int(r_eg - r_exact), h=int(closed - r_exact), nongrav=len(exprs),
                rep_terms=[len(e) for e in exprs], time=round(time.time() - t0, 1))
    log(f"  sector {group} N={N} lam={lam} 2J={n}: m={S.m} closed={closed} exact={r_exact} h={closed - r_exact} "
        f"grav={r_eg - r_exact} non-graviton={len(exprs)} ({time.time() - t0:.0f}s)")
    return exprs, info
