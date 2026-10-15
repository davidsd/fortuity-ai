"""Second-order differential d_2 on a derivative-free Q_0-cocycle x that survives one loop (summary/fortuity-summary.pdf):

    d_2[x] = [Q_2 x + Q_1 y],   Q_0 y = -Q_1 x,     Q_2 = K_2 U on derivative-free words (q2bmn.py, K_2 = -1/6).

Input: the prep of the one-loop test of x (fullpar.py CASE --prep-dir PREP: rows of B = Q_0 of the compressed hw rows
h_i of the preimage sector at the functionals, V = [controls; Q_1 x]) and the hw cache of that case.
  1. left-kernel vectors (c | a) of [B; V_reps], c B + sum_k a_k V_k = 0: from kernel_vec_aug.py (PREP/xker_aug.npy,
     zker_aug.npy with F = the V rows of the representatives, --x/--z) or by integer elimination here (small cases).
     x = sum_k a_k v_k is in the kernel of d_1 on the span of the representatives (--reps; one representative: x is
     a multiple of it), and y = sum_i c_i h_i solves Q_0 y = -Q_1 x;
  2. check Q_0 y + Q_1 x = 0 at fresh random points (words are not independent at rank N, so not word by word);
     control: Q_0 y + 2 Q_1 x must not vanish;
  3. z = K_2 U x + Q_1 y; check Q_0 z = 0 at fresh points; control: the same with 2 K_2 must not vanish;
  4. test rows for the exactness of z in T_2:
       z, [t_k from --tdir (cocycles of T_2, e.g. the Coulomb lowest-weight classes: sensitivity controls)], z + noise
     with exact controls Q_0 u (u = compressed hw rows of the preimage sector of T_2, --in-cache), written
       --fullpar T2CASE : classes/<T2CASE>/q1.pkl, ctrl.pkl for fullpar.py (small cases), and/or
       OUT/vecs_z.pkl   : dict(words, coef, n_ctrl) for veval.py (stored preps).
y is fixed up to Q_0-closed vectors y', which change z by Q_1 y': the class of z is defined modulo the image of d_1.
With --nclosed M, M random combinations of the kernel vectors with a = 0 are used as random Q_0-closed vectors y'_j
of the preimage sector, and Q_1 y'_j are added as test rows 'd1_j' after z: their increments give the rank of d_1
from that sector into T_2 on a random M-dimensional space of cocycles (the whole image of d_1 if it is < M).
The checks use plain evaluations (no charge-conjugation projection): the identities hold for every C-parity.
usage: python3 d2_make.py --case su2_j24 --N 2 --prep PREP --fullpar rank:2:5,5,8,8,8:6 [--nclosed 6]
       python3 d2_make.py --case su2_j36 --N 2 --prep PREP --reps 0,1 --fullpar rank:2:9,9,12,12,12:8
       python3 d2_make.py --case su3_j30 --N 3 --prep prep/su3_j30 --x xker_aug.npy --z zker_aug.npy
                          --in-cache cache/fullpar_hw_r007_o2_in_codd.pkl --tdir classes/r007_d1 --out classes/r007_d2"""
import argparse, json, os, pickle, sys, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--case', default='su2_j24')
ap.add_argument('--N', type=int, default=2)
ap.add_argument('--prep', required=True)
ap.add_argument('--cache', default=None, help='hw cache of the one-loop case (default cache/fullpar_hw_CASE.pkl)')
ap.add_argument('--x', default=None, help='kernel_vec_aug output in PREP (x part); default: solve here')
ap.add_argument('--z', default='zker_aug.npy')
ap.add_argument('--rep', type=int, default=0, help='index of x in classes/CASE/reps.pkl')
ap.add_argument('--reps', default=None, help='comma list of representatives: x = the combination(s) of them whose Q_1 '
                'image is Q_0-exact (kernel of d_1 on their span); default: the single --rep')
ap.add_argument('--kdir', type=int, default=0, help='which kernel direction of d_1 to use when there are several')
ap.add_argument('--fullpar', default=None, help='T_2 case for fullpar.py (writes classes/<case>/q1.pkl, ctrl.pkl)')
ap.add_argument('--in-cache', default=None, help='hw cache of the preimage sector of T_2 (exact controls)')
ap.add_argument('--tdir', default=None, help='directory with twords.pkl, tvec.npz: cocycles of T_2 added as test rows')
ap.add_argument('--out', default=None)
ap.add_argument('--q1-stored', default=None, help='pickle [Q_1 v_k] that made the V rows of PREP: must equal Q_1 x')
ap.add_argument('--nclosed', type=int, default=0, help='number of random closed vectors y\'')
ap.add_argument('--libnull', action='store_true', help='left kernel of [B; V] with the library kernel (FLINT / BLAS) '
                'instead of plain integer elimination: for large B on Linux (see nullspace_left below)')
ap.add_argument('--npts', type=int, default=24)
ap.add_argument('--workers', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', '8')))
a = ap.parse_args()
sys.argv = ['x']
import fullpar
import build_cand as bc
import q1apply, q2bmn
try:
    from fortuity.full.words import Q_multi, sector_of
    from fortuity.full.hw import E_multi
except ImportError:            # older clones: the original fullsector modules
    from fortuity.full.words import Q_multi, sector_of
    from fortuity.full.hw import E_multi
from fortuity.bmn.modp import P, matmul_mod

fullpar.CSEC = 0
K2 = (-pow(6, P - 2, P)) % P
log = lambda *s: print(time.strftime('%H:%M:%S'), *s, flush=True)


def nullspace_left(M):
    """Rows v with v M = 0 mod p, by plain integer elimination (no BLAS or FLINT calls before the worker pools are
    forked: on macOS a fork after threaded library calls can hang the children)."""
    M = np.asarray(M, dtype=np.int64) % P
    r, c = M.shape
    A = np.concatenate([M, np.eye(r, dtype=np.int64)], axis=1)
    row = 0
    for col in range(c):
        if row == r:
            break
        nz = np.flatnonzero(A[row:, col])
        if nz.size == 0:
            continue
        piv = row + int(nz[0])
        if piv != row:
            A[[row, piv]] = A[[piv, row]]
        A[row] = A[row] * pow(int(A[row, col]), P - 2, P) % P
        rows = np.flatnonzero(A[:, col])
        rows = rows[rows != row]
        if rows.size:
            A[rows] = (A[rows] - A[rows, col:col + 1] * A[row:row + 1]) % P
        row += 1
    return np.ascontiguousarray(A[row:, c:])


def rank_small(M):
    M = np.asarray(M, dtype=np.int64)
    return M.shape[0] - nullspace_left(M).shape[0] if M.size else 0


def mm_int(A, B):
    """(A @ B) mod p with integer arithmetic only (A split into 13-bit halves; small matrices)"""
    A, B = np.asarray(A, dtype=np.int64) % P, np.asarray(B, dtype=np.int64) % P
    out = np.zeros((A.shape[0], B.shape[1]), dtype=np.int64)
    for k0 in range(0, A.shape[1], 2048):          # 2048 terms < 2^39 each: no overflow
        Ak, Bk = A[:, k0:k0 + 2048], B[k0:k0 + 2048]
        out = (out + (((Ak >> 13) @ Bk) % P) * (1 << 13) + (Ak & 0x1FFF) @ Bk) % P
    return out


def lin(op, vec, chunk=2000, what=''):
    """op (word -> {word: coeff}; None = Q_1) applied to the rows of vec = (words, K x n): (image words, K x n')"""
    words, C = vec
    t = time.time()
    out = q1apply.apply(words, C, N=a.N, workers=a.workers, chunk=chunk, op=op)
    if what:
        log(f"{what}: {len(out[0])} words ({time.time()-t:.0f}s)")
    return out


def stack(parts, K):
    """parts = [(vec, [(source row, target row, scalar), ...])]: one word list with the K x n matrix of the sums"""
    ks, vs = [], []
    for (words, C), rows in parts:
        if not len(words):
            continue
        V = np.zeros((K, len(words)), dtype=np.int64)
        for r_src, r_dst, lam in rows:
            V[r_dst] = (V[r_dst] + lam % P * (C[r_src] % P)) % P
        ks.append(q1apply.encode(words))
        vs.append(V)
    keys, V = q1apply._merge(list(zip(ks, vs)))
    keep = np.flatnonzero(V.any(axis=0))
    return q1apply.decode(keys[keep]), np.ascontiguousarray(V[:, keep])


def nonzero_at_points(vec, seed):
    """per row: (nonzero values, values) at fresh random points (0 nonzero = the row vanishes as an operator, w.h.p.)"""
    words, C = vec
    K = C.shape[0]
    if not len(words):
        return [0] * K, 0
    vals = bc.eval_combo(words, C, a.N, a.npts, seed, False, a.workers)
    return [int(sum(np.count_nonzero(v[k]) for v in vals.values())) for k in range(K)], int(sum(v.shape[1] for v in vals.values()))


T0 = time.time()
cdir = f'classes/{a.case}'
out = a.out or f'classes/d2_{a.case}'
os.makedirs(out, exist_ok=True)
rep = dict(case=a.case, N=a.N, prep=a.prep, K2='-1/6', p=P, x_from=a.x or 'dense solve')
ridx = [int(t) for t in a.reps.split(',')] if a.reps else [a.rep]
vecs = [{w: c % P for w, c in pickle.load(open(f'{cdir}/reps.pkl', 'rb'))['vecs'][k].items() if c % P} for k in ridx]
meta = json.load(open(os.path.join(a.prep, 'meta.json')))
nB, ncols, nV, nc = meta['nB'], meta['ncols'], meta['nV'], meta['n_ctrl']
# left-kernel vectors (c | a) of [B; V_reps]: c B + sum_k a_k V_k = 0, i.e. Q_0 (sum_i c_i h_i) = -Q_1 (sum_k a_k v_k)
if a.x:
    KVc, KVa = np.ascontiguousarray(np.load(os.path.join(a.prep, a.x)).T % P), np.load(os.path.join(a.prep, a.z)) % P
    assert KVc.shape[1] == nB and KVa.shape == (KVc.shape[0], len(ridx)), (KVc.shape, KVa.shape)
else:
    B = np.fromfile(os.path.join(a.prep, 'B.i32'), dtype=np.int32).reshape(nB, ncols).astype(np.int64)
    V = np.fromfile(os.path.join(a.prep, 'V.i32'), dtype=np.int32).reshape(nV, ncols).astype(np.int64)
    A = np.ascontiguousarray(np.vstack([B, V[[nc + k for k in ridx]]]))
    if a.libnull:
        from fortuity.hwcoh.hw import nullspace_left as lib_nullspace_left
        Nl = np.asarray(lib_nullspace_left(A % P), dtype=np.int64) % P
    else:
        Nl = np.asarray(nullspace_left(A), dtype=np.int64) % P
    assert not (matmul_mod(Nl, A % P) if a.libnull else mm_int(Nl, A)).any()
    KVc, KVa = np.ascontiguousarray(Nl[:, :nB]), np.ascontiguousarray(Nl[:, nB:])
    log(f"left kernel of [B; V] ({nB} + {len(ridx)} rows, {ncols} columns): dimension {Nl.shape[0]}")
    del B, V, A, Nl
ra = rank_small(KVa)
assert ra > 0, "no combination of the representatives has a Q_0-exact Q_1 image: all of them lift at one loop"
sel = []
for j in range(KVa.shape[0]):
    if rank_small(KVa[sel + [j]]) > len(sel):
        sel.append(j)
    if len(sel) == ra:
        break
avec, c = KVa[sel[a.kdir]], KVc[sel[a.kdir]]
log(f"kernel of d_1 on the {len(ridx)} representative(s): dimension {ra}; using direction {a.kdir}: a = {avec.tolist()}")
rep.update(reps=ridx, d1_kernel_dim=ra, a=[int(t) for t in avec])
closed = []
if a.nclosed:
    T = np.asarray(nullspace_left(KVa), dtype=np.int64) % P            # combinations of kernel vectors with a = 0
    G = np.random.default_rng(777).integers(0, P, (min(a.nclosed, T.shape[0]), T.shape[0]), dtype=np.int64)
    closed = list(mm_int(mm_int(G, T), KVc))
    log(f"{len(closed)} random closed combinations (a = 0) of the {KVa.shape[0]} kernel vectors")
del KVc, KVa
xd = {}
for ak, v in zip(avec.tolist(), vecs):
    for w, cw in v.items():
        xd[w] = (xd.get(w, 0) + ak * cw) % P
xd = {w: cw for w, cw in xd.items() if cw}
assert all(L[0] == 0 and L[1] == 0 for w in xd for t in w for L in t), "x must be derivative-free"
xw = sorted(xd)
x = (xw, np.array([[xd[w] for w in xw]], dtype=np.int64))
log(f"x: {len(xw)} words, sector {sector_of(xw[0])}, Y = {sum(len(t) for t in xw[0])}")
q1x = lin(None, x, 500, 'Q_1 x')
if a.q1_stored and len(ridx) == 1:
    st = {w: cw * int(avec[0]) % P for w, cw in pickle.load(open(a.q1_stored, 'rb'))[ridx[0]].items() if cw % P}
    same = st == {w: int(cw) for w, cw in zip(q1x[0], q1x[1][0])}
    log(f"Q_1 x {'equals' if same else 'DIFFERS FROM'} a times row {ridx[0]} of {a.q1_stored} (the fresh-point check below decides)")
    rep['q1x_equals_stored'] = bool(same)
    del st
M = len(closed)
ws, groups, ms, stats = pickle.load(open(a.cache or f'cache/fullpar_hw_{a.case}.pkl', 'rb'))
assert sum(g['k'] for g in groups) == nB, (sum(g['k'] for g in groups), nB)
Call = np.ascontiguousarray(np.vstack([c] + closed))                    # rows: y, y'_1 .. y'_M
coef = np.zeros((1 + M, len(ws)), dtype=np.int64)
g0 = 0
for g in groups:
    cg = np.ascontiguousarray(Call[:, g0:g0 + g['k']])
    assert len(g['parts'][0]) == 3, "hw cache without the per-part random matrices"
    for widx, H, R in g['parts']:
        coef[:, widx] = (coef[:, widx] + matmul_mod(matmul_mod(cg, R % P), H.astype(np.int64) % P)) % P
    g0 += g['k']
del groups, Call
nzw = np.flatnonzero(coef.any(axis=0))
y = ([ws[int(i)] for i in nzw], np.ascontiguousarray(coef[:, nzw]))
del ws, coef
log(f"y, y': {1 + M} rows on {len(y[0])} words ({int(np.count_nonzero(y[1][0]))} in y), sector {sector_of(y[0][0])}, "
    f"Y = {sum(len(t) for t in y[0][0])} ({time.time()-T0:.0f}s)")
# check 1: Q_0 y + Q_1 x = 0 (row 0), control (row 1), Q_0 y'_j = 0 (rows 2..)
q0y = lin(Q_multi, y, 4000, 'Q_0 (y, y\')')
nz1, tot1 = nonzero_at_points(stack([(q0y, [(0, 0, 1), (0, 1, 1)] + [(1 + j, 2 + j, 1) for j in range(M)]),
                                     (q1x, [(0, 0, 1), (0, 1, 2)])], 2 + M), 901)
log(f"check Q_0 y + Q_1 x = 0 at fresh points: {nz1[0]} nonzero of {tot1} (control Q_0 y + 2 Q_1 x: {nz1[1]} nonzero)"
    + (f"; Q_0 y'_j nonzero: {nz1[2:]}" if M else ''))
assert nz1[0] == 0 and nz1[1] > 0 and not any(nz1[2:])
del q0y
# z
ux = lin(lambda m: q2bmn.U_multi(m, a.N), x, 500, 'U x')
q1y = lin(None, y, 2000, 'Q_1 (y, y\')')
rep['closed_rows'] = M
rep['q1yprime_nonzero_words'] = [int(np.count_nonzero(q1y[1][1 + j])) for j in range(M)]
zz = stack([(ux, [(0, 0, K2), (0, 1, 2 * K2)]), (q1y, [(0, 0, 1), (0, 1, 1)])], 2)
zsec = sector_of(zz[0][0])
log(f"z = K_2 U x + Q_1 y: {int(np.count_nonzero(zz[1][0]))} words, sector {zsec}, Y = {sum(len(t) for t in zz[0][0])}")
q0z = lin(Q_multi, zz, 4000, 'Q_0 z')
nz2, tot2 = nonzero_at_points(q0z, 902)
log(f"check Q_0 z = 0 at fresh points: {nz2[0]} nonzero of {tot2} (control with 2 K_2: {nz2[1]} nonzero)")
assert nz2[0] == 0 and nz2[1] > 0
del q0z
keep = np.flatnonzero(zz[1][0])
z = ([zz[0][int(i)] for i in keep], np.ascontiguousarray(zz[1][:1, keep]))
del zz
nzz, totz = nonzero_at_points(z, 903)
nze = [nonzero_at_points(lin(lambda m, i=i: E_multi(m, i), z, 4000), 904 + i)[0][0] for i in (1, 2)]
log(f"z itself: {nzz[0]} nonzero values of {totz}; E_1 z, E_2 z nonzero values: {nze} (0 = hw)")
rep.update(x_words=len(xw), q1x_words=len(q1x[0]), y_words=int(np.count_nonzero(y[1][0])), Ux_words=len(ux[0]),
           q1y_words=int(np.count_nonzero(q1y[1][0])),
           z_words=len(z[0]), z_sector=list(zsec), check_Q0y_plus_Q1x_nonzero=nz1[0], control_Q0y_plus_2Q1x_nonzero=nz1[1],
           check_values=tot1, check_Q0z_nonzero=nz2[0], control_2K2_nonzero=nz2[1], check_Q0z_values=tot2,
           z_nonzero_values=nzz[0], E_z_nonzero=nze)
pickle.dump(dict(xw=xw, xC=x[1], yw=y[0], yC=y[1], zw=z[0], zC=z[1]), open(f'{out}/d2_vectors.pkl', 'wb'), protocol=4)
del y, ux
# test rows: z, d1_j = Q_1 y'_j, t_k, z + noise
noise = np.zeros_like(z[1])
noise[0, len(z[0]) // 2] = 12345
parts, rows = [(z, [(0, 0, 1)])], ['z']
if M:
    parts.append((q1y, [(1 + j, 1 + j, 1) for j in range(M)]))
    rows += [f'd1_{j}' for j in range(M)]
if a.tdir:
    tw = pickle.load(open(os.path.join(a.tdir, 'twords.pkl'), 'rb'))
    tC = np.load(os.path.join(a.tdir, 'tvec.npz'))['coef'] % P
    assert tC.shape[1] == len(tw) and sector_of(tw[0]) == zsec
    parts.append(((tw, tC), [(k, len(rows) + k, 1) for k in range(tC.shape[0])]))
    rows += [f't_{k}' for k in range(tC.shape[0])]
nt = len(rows)
parts += [(z, [(0, nt, 1)]), ((z[0], noise), [(0, nt, 1)])]
rows.append('z + noise')
tests = stack(parts, nt + 1)
del q1y
rep['test_rows'] = rows
log(f"test rows {rows}: {len(tests[0])} words")
ctrl = []
if a.in_cache and os.path.exists(a.in_cache):
    wi, gi, _, _ = pickle.load(open(a.in_cache, 'rb'))
    for g in sorted(gi, key=lambda g: g['rows'])[:3]:
        u = {}
        for widx, H, R in g['parts']:
            row = matmul_mod(R[:1] % P, H.astype(np.int64) % P)[0]
            for j in np.flatnonzero(row):
                u[wi[int(widx[j])]] = int(row[j])
        uw = sorted(u)
        ctrl.append(lin(Q_multi, (uw, np.array([[u[w] for w in uw]], dtype=np.int64)), 2000))
    del wi, gi
    rep['ctrl_words'] = [len(cv[0]) for cv in ctrl]
    log(f"exact controls Q_0 u: sizes {rep['ctrl_words']}")
if a.fullpar:
    tdir = 'classes/' + a.fullpar.replace(':', '_').replace(',', '-')
    os.makedirs(tdir, exist_ok=True)
    tovec = lambda vec, r: {vec[0][int(i)]: int(vec[1][r, i]) for i in np.flatnonzero(vec[1][r])}
    pickle.dump([tovec(tests, r) for r in range(nt + 1)], open(f'{tdir}/q1.pkl', 'wb'), protocol=4)
    if ctrl:
        pickle.dump([tovec(cv, 0) for cv in ctrl], open(f'{tdir}/ctrl.pkl', 'wb'), protocol=4)
    log(f"wrote {tdir}/q1.pkl ({nt + 1} rows)" + (f" and ctrl.pkl ({len(ctrl)})" if ctrl else ''))
allv = stack([(cv, [(0, r, 1)]) for r, cv in enumerate(ctrl)] + [(tests, [(r, len(ctrl) + r, 1) for r in range(nt + 1)])],
             len(ctrl) + nt + 1)
pickle.dump(dict(words=allv[0], coef=allv[1], n_ctrl=len(ctrl), rows=rows), open(f'{out}/vecs_z.pkl', 'wb'), protocol=4)
rep.update(vec_words=len(allv[0]), n_ctrl=len(ctrl), time=round(time.time() - T0))
json.dump(rep, open(f'{out}/d2_make.json', 'w'), indent=1)
log(json.dumps(rep))
