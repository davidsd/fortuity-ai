"""T-033: the SU(2) "fused channels" of an order-3 partner of R-007.

A C-odd partner w of R-007 at order 3 lies in T_3 = (2J1,2J2;2q) = (4,4;14,8,8), Y = 8, d_BMN = 3, [3,0] hw.  Under
SU(3) > SU(2)xU(1) its restriction is sum_alpha alpha (x) rho_alpha(w), with alpha a monomial in abelian letters,
|alpha| odd (C), and rho_alpha(w) an SU(2) class with odd Y, hence non-graviton or zero (Theorem A; summary/fortuity-summary.pdf, sec. 4.6).
Non-graviton SU(2) classes need j' >= 24, so j(alpha) <= 6.  The channels with j' <= 25 are closed by Chang-Lin's
complete SU(2) tables (only O_0 at j' = 24, only E = 10 classes at j' = 25).  The open channels are
    alpha = x_m (j'=28), lambda_hat (j'=27), psi_hat_m (j'=26),
and SU(3)_R covariance ([3,0] in rep(alpha) (x) R') fixes the SU(2) hw sectors computed here.  h = 0 in all of them
means: every C-odd class of T_3 restricts to zero, i.e. is an SU(3) core.
Control: (4,4;8,8,8) Y=5, O_0's own order-3 partner sector (h = 0).

Method = stage 'reps' of scripts/oneloop/su-threshold-classes/oneloop_test.py (imports of the full-letter code): hw vectors of the sector, exact multiplicities per fermion degree (spanning check),
closed = left kernel of Q_0 into the next sector, h = closed - rank(Q_0 images of the previous sector).
Random evaluation can only lower ranks; spanning is asserted, so a computed h = 0 is a proof (closed is an upper bound
only if rank(Q_0 out) is not underestimated: we evaluate it at 1.3x margin, as in oneloop_test.py).
Run (from a scratch directory, caches are written to ./cache):
    PYTHONPATH=<repo>/code python3 <repo>/scripts/oneloop/protection/code/su2_channels.py [label ...]
"""
import os, sys, time, math, pickle
import numpy as np
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path[:0] = [os.path.join(ROOT, 'scripts', 'oneloop', 'su-threshold-classes'), os.path.join(ROOT, 'code')]
from fastcontents import prime_cache
from fortuity.bmn.modp import RREFBasis, matmul_mod, P
from fortuity.hwcoh import ffla
from fortuity.full.sector import words_of, hw_mult_by_degree, frame_k
from fortuity.full.words import Q_multi
from fortuity.full.evalg import Frame, nferm
from fortuity.full.hw import nullspace_left, hw_vectors

N, ML, MARGIN = 2, int(os.environ.get('ML', '3')), 64
KMIN_EXTRA = int(os.environ.get('KMIN_EXTRA', '0'))
F0_FACTOR = float(os.environ.get('F0_FACTOR', '1'))   # more evaluation points in the sector itself (spanning)
SECTORS = {   # label: (SEC = (2J1, 2J2, 2q1, 2q2, 2q3), Y)
    'control_x3': ((4, 4, 8, 8, 8), 5),       # alpha = x^3: O_0's order-3 partner sector (expected h = 0)
    'x_20': ((4, 4, 12, 8, 8), 7),            # alpha = x_m, R' = [2,0]
    'x_31': ((4, 4, 14, 8, 6), 7),            # alpha = x_m, R' = [3,1]
    'psi_40': ((3, 3, 14, 6, 6), 7),          # alpha = psi_hat_m, R' = [4,0]
    'psi_21': ((3, 3, 12, 8, 6), 7),          # alpha = psi_hat_m, R' = [2,1]
    'lam_30': ((5, 3, 13, 7, 7), 7),          # alpha = lambda_hat (SU(2)_Rbar doublet), R' = [3,0]
    'lam_30b': ((3, 5, 13, 7, 7), 7),         # the other SU(2)_Rbar component (cross-check)
    # channels with j' <= 25, closed by Chang-Lin's SU(2) tables; recomputed here as a cross-check
    'x3_11': ((4, 4, 10, 8, 6), 5),           # alpha = x^3 in [3,0], R' = [1,1]
    'x3_22': ((4, 4, 12, 8, 4), 5),           # R' = [2,2]
    'x3_33': ((4, 4, 14, 8, 2), 5),           # R' = [3,3]
    'f_30': ((2, 2, 12, 6, 6), 7),            # alpha = f_hat (singlet), R' = [3,0]
    'dlam_30': ((3, 3, 12, 6, 6), 7),         # alpha = D lambda_hat (SU(2)_Rbar triplet; m = 0 component), R' = [3,0]
    'dx_20': ((2, 4, 11, 7, 7), 7),           # alpha = D x_m (doublet), R' = [2,0]
    'dx_31': ((2, 4, 13, 7, 5), 7),           # alpha = D x_m, R' = [3,1]
}
log = lambda *a: print(time.strftime('%H:%M:%S'), *a, flush=True)


def hw_blocks(sec, Yv):
    prime_cache(sec, Yv)
    os.makedirs('cache', exist_ok=True)
    fn = f'cache/hw_{"_".join(map(str, sec))}_Y{Yv}_ml{ML}_N{N}.pkl'
    ws = words_of(sec, Yv, ML)
    if os.path.exists(fn):
        nw, blocks = pickle.load(open(fn, 'rb'))
        if nw == len(ws):
            return ws, [(i, H.astype(np.int64)) for i, H in blocks]
    blocks = hw_vectors(ws)
    pickle.dump((len(ws), [(i, H.astype(np.int32)) for i, H in blocks]), open(fn, 'wb'), protocol=4)
    return ws, blocks


def frames_for(m_d, letters, factor=1.0, seed=11):
    out = []
    for d, md in sorted(m_d.items()):
        need = int(factor * (md + MARGIN))
        k = max(frame_k(N, d), d + KMIN_EXTRA) if d else 0    # KMIN_EXTRA > 0: more Grassmann generators than frame_k's cap 3(N^2-1)
        out.append((d, Frame(N, max(1, math.ceil(need / math.comb(k, d))), k, [d], letters, seed=seed + 7919 * d)))
    return out


def evaluate(words, frames):
    X = np.zeros((len(words), sum(fr.ncols for _, fr in frames)), dtype=np.int64)
    deg = np.array([nferm(w) for w in words], dtype=np.int64)
    col = 0
    for d, fr in frames:
        idx = np.flatnonzero(deg == d)
        if idx.size:
            X[idx, col:col + fr.ncols] = fr.eval_words([words[i] for i in idx])
        col += fr.ncols
    return X


def eval_dicts(dicts, frames, chunk=20000):
    ncols = sum(fr.ncols for _, fr in frames)
    allw = sorted({w for d in dicts for w in d})
    out = np.zeros((len(dicts), ncols), dtype=np.int64)
    for s in range(0, len(allw), chunk):
        part = allw[s:s + chunk]
        Xc = evaluate(part, frames)
        pidx = {w: i for i, w in enumerate(part)}
        for r, d in enumerate(dicts):
            rows = [pidx[w] for w in d if w in pidx]
            if rows:
                cs = np.array([d[part[i]] for i in rows], dtype=np.int64) % P
                out[r] = (out[r] + matmul_mod(cs[None, :], Xc[np.array(rows)])[0]) % P
    return out


def hw_dicts(ws, blocks):
    out = []
    for widx, H in blocks:
        for r in range(H.shape[0]):
            nz = np.flatnonzero(H[r])
            out.append({ws[int(widx[c])]: int(H[r, c]) for c in nz})
    return out


def Q0(v):
    out = defaultdict(int)
    for w, c in v.items():
        for m2, c2 in Q_multi(w).items():
            out[m2] = (out[m2] + c * c2) % P
    return {m: c for m, c in out.items() if c}


def letters_of(*colls):
    return sorted({L for coll in colls for w in coll for n in w for L in n})


def cohomology(label, SEC, Y):
    T0 = time.time()
    UP, LOW = (SEC[0] + 1, SEC[1] + 1) + SEC[2:], (SEC[0] - 1, SEC[1] - 1) + SEC[2:]
    ws, blocks = hw_blocks(SEC, Y)
    m_d = hw_mult_by_degree(N, SEC, Y)
    m = sum(m_d.values())
    if m == 0:
        log(f"[{label}] SEC {SEC} Y={Y}: hw multiplicity 0 -> h = 0")
        return dict(label=label, SEC=SEC, Y=Y, m=0, h=0)
    ws_up, blocks_up = hw_blocks(UP, Y - 1) if Y > 1 else ([], [])
    Ed = [Q0(v) for v in hw_dicts(ws_up, blocks_up)] if ws_up else []
    F0 = frames_for(m_d, sorted(set(letters_of(ws)) | set(letters_of([w for d in Ed for w in d]))), factor=F0_FACTOR)
    A = np.zeros((sum(H.shape[0] for _, H in blocks), sum(fr.ncols for _, fr in F0)), dtype=np.int64)
    r0 = 0
    for widx, H in blocks:
        A[r0:r0 + H.shape[0]] = matmul_mod(H % P, evaluate([ws[int(i)] for i in widx], F0))
        r0 += H.shape[0]
    basis = RREFBasis(A.shape[1])
    indep = np.zeros(A.shape[0], dtype=bool)
    for s in range(0, A.shape[0], 256):
        indep[s:s + 256] = basis.add(A[s:s + 256])
    sel = np.flatnonzero(indep)
    log(f"[{label}] SEC {SEC} Y={Y}: words {len(ws)}, hw rows {A.shape[0]}, independent {len(sel)} vs exact m {m} {m_d}"
        f" -> {'SPAN' if len(sel) == m else 'NO SPAN'}")
    assert len(sel) == m, "hw vectors do not span"
    Hd = hw_dicts(ws, blocks)
    Hsel, Asel = [Hd[i] for i in sel], A[sel]
    ebasis = RREFBasis(A.shape[1])
    if Ed:
        ebasis.add_batched(eval_dicts(Ed, F0))
    r_in = ebasis.rank
    Qd = [Q0(v) for v in Hsel]
    m_low = hw_mult_by_degree(N, LOW, Y + 1)
    if any(Qd) and sum(m_low.values()):
        prime_cache(LOW, Y + 1)
        F1 = frames_for(m_low, letters_of([w for d in Qd for w in d]), factor=1.3, seed=23)
        M = eval_dicts(Qd, F1)
        K = nullspace_left(M)
        r_out = int(ffla.rank(M))
    else:
        K = np.eye(len(Hsel), dtype=np.int64)
        r_out = 0
    Cm = matmul_mod(K % P, Asel)
    h = int(sum(ebasis.add(Cm[r:r + 1])[0] for r in range(Cm.shape[0])))
    log(f"[{label}] m {m}, rank in (Q_0 from {UP} Y={Y - 1}) {r_in}, rank out (to {LOW} Y={Y + 1}) {r_out}, "
        f"closed {K.shape[0]} -> h = {h}  ({time.time() - T0:.0f}s)")
    return dict(label=label, SEC=SEC, Y=Y, m=m, m_d=m_d, words=len(ws), r_in=r_in, r_out=r_out, closed=int(K.shape[0]), h=h)


if __name__ == '__main__':
    labels = sys.argv[1:] or list(SECTORS)
    res = [cohomology(l, *SECTORS[l]) for l in labels]
    log("SUMMARY", [(r['label'], r['m'], r['h']) for r in res])
