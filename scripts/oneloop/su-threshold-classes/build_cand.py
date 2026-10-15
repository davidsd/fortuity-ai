"""R-007 order 2: representatives of the non-Coulomb C-odd class of T_2 = (4,4;14,8,8), Y = 9, and the SU(2)_R test.

T_2 has h = 1,899 C-odd [3,0] classes, 1,898 of Coulomb type, so at most one non-Coulomb class (summary/fortuity-summary.pdf).  A random
left-kernel vector x of the T_2 'out' prep [B_out | Cartan] (kernel_vec.py) gives t = sum_i x_i h_i in
ker Q_0 n ker(Cartan), i.e. lambda * (non-Coulomb class) + (exact), lambda random.

R-007 is an SU(2)_R singlet (no derivatives, no gauginos) and Q(g) commutes with SU(2)_R (it acts on z+, z-), so every
d_n[R-007] is a singlet.  D: (a, b, S) -> (b + 1) (a - 1, b + 1, S) (the vector field z- d/dz+ on the superfield's
Taylor coefficients) lowers J_R by one and commutes with Q_0 (checked on random multitraces; a wrong coefficient fails).
A weight-0 class [t] is a singlet iff [D t] = 0.  So if D t is not Q_0-exact in (2,6;14,8,8), Y = 9, then [t] is not a
singlet, the non-Coulomb part of T_2 has no singlet, and d_2[R-007] (non-Coulomb under the Coulomb-branch assumption)
vanishes.  That exactness question is a fullpar run of case rank:3:3,7,14,8,8:9 with V = [D t_1, D t_2]: expected
increments [1, 0] if [t] is not a singlet (the second vector is a multiple of the first modulo exact vectors).

This script: builds t_k from x and the hw cache, checks at fresh random points that t_k != 0, that t_k vanishes on the
Cartan and that Q_0 t_k = 0 (and Q_0 D t_k = 0), then writes classes/<dcase>/q1.pkl = [D t_k] and ctrl.pkl = [D Q_0 u]
(u hw vectors of (5,5;14,8,8), Y = 8: D Q_0 u = Q_0 D u is exact, so the controls must not raise the rank).
"""
import argparse, json, math, os, pickle, sys, time
import multiprocessing as mp
from collections import defaultdict
import numpy as np
import scipy.sparse as sp
import fullpar
from fullpar import frame_k, eval_csec, Frame, nferm
from fortuity.bmn.modp import P, matmul_mod
try:
    from fortuity.full.words import canon_word, canon_multi, Q_multi, word_parity
    from fortuity.full.letters import admissible, is_ferm, theta_sign
except ImportError:      # clones older than fortuity.full: the original fullsector modules (on the path via fullpar)
    from fortuity.full.words import canon_word, canon_multi, Q_multi, word_parity
    from fortuity.full.letters import admissible, is_ferm, theta_sign

log = lambda *a: print(time.strftime('%H:%M:%S'), *a, flush=True)


def D_multi(multi):
    """SU(2)_R lowering z- d/dz+ on a canonical multitrace: (a, b, S) -> (b + 1) (a - 1, b + 1, S), a derivation."""
    out = defaultdict(int)
    for i, w in enumerate(multi):
        for pos, (a, b, S) in enumerate(w):
            if a == 0:
                continue
            s, cw = canon_word(w[:pos] + ((a - 1, b + 1, S),) + w[pos + 1:])
            if not s:
                continue
            sm, cm = canon_multi(multi[:i] + (cw,) + multi[i + 1:])
            if sm:
                out[cm] += (b + 1) * s * sm
    return {m: c for m, c in out.items() if c}


def Kp_multi(multi):
    """Special conformal K_+ = z_+ E (E = z.d_z + theta.d_theta, the projective vector field of C^{2|3}, divergence
    free): (a, b, S) -> (a + b - 1 + |S|) (a - 1, b, S), a derivation.  [Q_0, K_+] = 0 on random multitraces (a
    coefficient that is not a vector field fails)."""
    out = defaultdict(int)
    for i, w in enumerate(multi):
        for pos, (a, b, S) in enumerate(w):
            c = a + b - 1 + len(S)
            if a == 0 or c == 0:
                continue
            s, cw = canon_word(w[:pos] + ((a - 1, b, S),) + w[pos + 1:])
            if not s:
                continue
            sm, cm = canon_multi(multi[:i] + (cw,) + multi[i + 1:])
            if sm:
                out[cm] += c * s * sm
    return {m: c for m, c in out.items() if c}


def _odd_derivation(letter_map):
    """Odd derivation from a letter map L -> (coeff, L') with the sign conventions of Q_0 (fortuity.full.words): a
    factor (-1)^(fermions before the letter, across the multitrace in written order)."""
    def op(multi):
        out = defaultdict(int)
        par_before = 0
        for i, w in enumerate(multi):
            nf = 0
            for pos, L in enumerate(w):
                r = letter_map(L)
                if r is not None:
                    c, nl = r
                    s, cw = canon_word(w[:pos] + (nl,) + w[pos + 1:])
                    if s:
                        sm, cm = canon_multi(multi[:i] + (cw,) + multi[i + 1:])
                        if sm:
                            out[cm] += (-1 if (par_before + nf) % 2 else 1) * c * s * sm
                if is_ferm(L):
                    nf += 1
            par_before += word_parity(w)
        return {m: c for m, c in out.items() if c}
    return op


def _S_letter(a_th):
    """S_a = theta_a E on fields: Psi_{m,S} -> (-1)^|S| (|m|+|S|-1) sgn(theta_a theta^{S-a}) Psi_{m,S-a} (a in S)."""
    def f(L):
        a, b, S = L
        if a_th not in S:
            return None
        T = tuple(x for x in S if x != a_th)
        c = (a + b + len(S) - 1) * theta_sign((a_th,), T) * (-1) ** len(S)
        return (c, (a, b, T)) if c and admissible(a, b, T) else None
    return f


def _T_letter(al, a_th):
    """T_{al,a} = z_al d/dtheta_a on fields: Psi_{m,S} -> (-1)^|S| sgn(theta_a theta^S) Psi_{m-e_al, S+a} (a not in S)."""
    def f(L):
        a, b, S = L
        if a_th in S or (a, b)[al] == 0:
            return None
        T = tuple(sorted(S + (a_th,)))
        na, nb = (a - 1, b) if al == 0 else (a, b - 1)
        return (theta_sign((a_th,), S) * (-1) ** len(S), (na, nb, T)) if admissible(na, nb, T) else None
    return f


# psu(1,2|3) generators used in the selection-rule tests; on SU(3)_R hw vectors the members commuting with E1, E2
# (S_2, T_{+,0}) give hw vectors.  {Q_0, J} = 0 for the odd ones (the (-1)^|S| convention; without it the test fails).
OPS = {'D': D_multi, 'Kp': Kp_multi, 'S2': _odd_derivation(_S_letter(2)), 'Tp0': _odd_derivation(_T_letter(0, 0)),
       'S0': _odd_derivation(_S_letter(0))}    # S0: hw vectors to hw vectors only where S1 t = 0 (no theta_1 letters)


def F1_multi(multi):
    """SU(3)_R lowering F1 (theta index 0 -> 1, coefficient +1, the transpose of hw.E_multi(., 1)); [E1, F1] = H1."""
    out = defaultdict(int)
    for i, w in enumerate(multi):
        for pos, (a, b, S) in enumerate(w):
            if 0 not in S or 1 in S:
                continue
            s, cw = canon_word(w[:pos] + ((a, b, tuple(sorted(1 if x == 0 else x for x in S))),) + w[pos + 1:])
            if not s:
                continue
            sm, cm = canon_multi(multi[:i] + (cw,) + multi[i + 1:])
            if sm:
                out[cm] += s * sm
    return {m: c for m, c in out.items() if c}


def _Tp21(h=3):
    """hw vector of the [2,1] part of T_{+,a} (x) t for an SU(3)_R hw t of Dynkin label h along E1:
    w = T_{+,1} t - (1/h) T_{+,0} F1 t, using [E1, T_{+,1}] = T_{+,0} (measured) and [E1, F1] = H1.  Returned as a
    linear combination [(scalar, op)] of operators with small integer coefficients (see apply_lin)."""
    Tp1, Tp0 = _odd_derivation(_T_letter(0, 1)), _odd_derivation(_T_letter(0, 0))

    def Tp0F1(multi):
        out = defaultdict(int)
        for m1, x1 in F1_multi(multi).items():
            for m, x in Tp0(m1).items():
                out[m] += x1 * x
        return {m: x for m, x in out.items() if x}
    return [(1, Tp1), ((P - pow(h, P - 2, P)) % P, Tp0F1)]


OPS['Tp21'] = _Tp21(3)


def F2_multi(multi):
    """SU(3)_R lowering F2 (theta index 1 -> 2, coefficient +1, the transpose of hw.E_multi(., 2)); [E2, F2] = H2."""
    out = defaultdict(int)
    for i, w in enumerate(multi):
        for pos, (a, b, S) in enumerate(w):
            if 1 not in S or 2 in S:
                continue
            s, cw = canon_word(w[:pos] + ((a, b, tuple(sorted(2 if x == 1 else x for x in S))),) + w[pos + 1:])
            if not s:
                continue
            sm, cm = canon_multi(multi[:i] + (cw,) + multi[i + 1:])
            if sm:
                out[cm] += s * sm
    return {m: c for m, c in out.items() if c}


def _compose(*ops):
    """Composite of small-coefficient operators (applied right to left), as a word -> {word: coeff} map."""
    def op(multi):
        cur = {multi: 1}
        for f in reversed(ops):
            nxt = defaultdict(int)
            for m, c in cur.items():
                for m2, c2 in f(m).items():
                    nxt[m2] += c * c2
            cur = {m: c for m, c in nxt.items() if c}
        return cur
    return op


def _S20(h=3):
    """hw vector of the [2,0] part of S_a (x) t for an SU(3)_R hw t of Dynkin labels (h, 0):
    w = S_0 t - (c1/h) S_1 F1 t + (c1 c2/h) S_2 F2 F1 t with [E1, S_0] = c1 S_1, [E2, S_1] = c2 S_2 (measured: c1 = c2
    = -1), [E1, F1] = H1, F2 t = 0."""
    S0, S1, S2 = (_odd_derivation(_S_letter(a)) for a in (0, 1, 2))
    inv = pow(h, P - 2, P)
    return [(1, S0), (inv, _compose(S1, F1_multi)), (inv, _compose(S2, F2_multi, F1_multi))]


OPS['S20'] = _S20(3)


def apply_op(op, words, C):
    """op (word -> {word: coeff}) applied to the K vectors sum_j C[k, j] words[j]: (image words, K x n_img int64)."""
    idx, img, rs, cs_, vs = {}, [], [], [], []
    for j, w in enumerate(words):
        for m, c in op(w).items():
            i = idx.get(m)
            if i is None:
                i = idx[m] = len(img)
                img.append(m)
            rs.append(i); cs_.append(j); vs.append(c)
    out = np.zeros((C.shape[0], len(img)), dtype=np.int64)
    if not img:
        return img, out
    # int64 sparse product with the small signed coefficients of op (|c| < 2^10) and values < 2^26: no overflow
    M = sp.csr_matrix((np.asarray(vs, dtype=np.int64), (np.asarray(rs), np.asarray(cs_))), shape=(len(img), len(words)))
    assert int(abs(M).max()) < 1 << 10
    for k in range(C.shape[0]):
        out[k] = (M @ C[k]) % P
    keep = np.flatnonzero(out.any(axis=0))
    return [img[i] for i in keep], np.ascontiguousarray(out[:, keep])


_G = {}


def apply_lin(opdef, words, C):
    """apply_op for an operator or a linear combination [(scalar mod p, op)] of small-coefficient operators."""
    if callable(opdef):
        return apply_op(opdef, words, C)
    acc = {}
    for lam, op in opdef:
        img, V = apply_op(op, words, C)
        for i, m in enumerate(img):
            acc[m] = (acc.get(m, 0) + lam * V[:, i]) % P
    img = [m for m, v in acc.items() if np.any(v)]
    out = np.zeros((C.shape[0], len(img)), dtype=np.int64)
    for i, m in enumerate(img):
        out[:, i] = acc[m]
    return img, out


def _ev_task(args):
    i0, i1, N, npts, k, d, seed, cartan = args
    words, C = _G['words'], _G['C']
    fr = Frame(N, npts, k, [d], _G['letters'], seed=seed, cartan=cartan)
    acc = None
    for j0 in range(i0, i1, 20000):
        j1 = min(i1, j0 + 20000)
        X = eval_csec(fr, words[j0:j1], dtype=np.int64)
        Z = matmul_mod(C[:, j0:j1], X)
        acc = Z if acc is None else (acc + Z) % P
    return acc


def eval_combo(words, C, N, npts, seed, cartan, workers):
    """Values of the K combinations sum_j C[k, j] words[j] (C-odd projection) at fresh frames, one per fermion degree:
    {d: K x cols int64}."""
    out = {}
    if not words:
        return out
    degs = np.array([nferm(w) for w in words])
    letters = sorted({L for w in words for n in w for L in n})
    for d in sorted(set(degs.tolist())):
        sel = np.flatnonzero(degs == d)
        k = frame_k(N, d)
        if d and math.comb(k, d) == 0:
            continue
        _G.update(words=[words[j] for j in sel], C=np.ascontiguousarray(C[:, sel]), letters=letters)
        n = sel.size
        ng = max(1, min(workers, n // 2000 + 1))
        tasks = [(g * n // ng, (g + 1) * n // ng, N, npts, k, d, seed + 31 * d, cartan) for g in range(ng)]
        with mp.get_context('fork').Pool(ng) as pool:
            parts = pool.map(_ev_task, tasks)
        acc = parts[0]
        for p_ in parts[1:]:
            acc = (acc + p_) % P
        out[d] = acc
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prep', default='prep/r007_o2_out_codd')
    ap.add_argument('--cache', default='cache/fullpar_hw_r007_o2_out_codd_cartan.pkl')
    ap.add_argument('--in-cache', default='cache/fullpar_hw_r007_o2_in_codd.pkl')
    ap.add_argument('--out', default='classes/r007_o2cand')
    ap.add_argument('--dcase', default='classes/rank_3_3-7-14-8-8_9')
    ap.add_argument('--nvec', type=int, default=2)
    ap.add_argument('--allow-coulomb', action='store_true', help='x from a prep without Cartan columns: t is a random '
                    'element of ker Q_0 (all classes), so the Cartan check is reported but not required')
    ap.add_argument('--workers', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', '8')))
    ap.add_argument('--op', choices=sorted(OPS), default='D', help='operator applied to t for the exactness test')
    ap.add_argument('--stage', choices=['make', 'check', 'all'], default='all', help="make: t, D t, controls (what the "
                    "exactness prep needs); check: the fresh-point checks of a made t (runs in parallel with the prep)")
    a = ap.parse_args()
    fullpar.CSEC = -1
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.dcase, exist_ok=True)
    if a.stage in ('make', 'all'):
        make(a)
    if a.stage in ('check', 'all'):
        check(a)


def make(a):
    T0 = time.time()
    rep = dict(prep=a.prep, cache=a.cache, nvec=a.nvec)
    X = np.load(os.path.join(a.prep, 'xker.npy'))[:, :a.nvec] % P
    nB = json.load(open(os.path.join(a.prep, 'meta.json')))['nB']
    assert X.shape[0] == nB, (X.shape, nB)
    log(f"loading {a.cache}")
    ws, groups, ms, stats = pickle.load(open(a.cache, 'rb'))
    assert sum(g['k'] for g in groups) == nB
    log(f"hw cache: {len(ws)} words, {len(groups)} stripped contents, {nB} compressed rows ({time.time()-T0:.0f}s)")
    # t_k = sum_i x_ik h_i, h_i = rows of R_s H_s (per stripped content s, rows at offset g0)
    coef = np.zeros((a.nvec, len(ws)), dtype=np.int64)
    g0 = 0
    for g in groups:
        Xg = np.ascontiguousarray(X[g0:g0 + g['k']].T)                 # K x k
        for widx, H, R in g['parts']:
            W = matmul_mod(matmul_mod(Xg, R % P), H.astype(np.int64) % P)  # K x n_b
            coef[:, widx] = (coef[:, widx] + W) % P
        g0 += g['k']
    del groups
    nzw = np.flatnonzero(coef.any(axis=0))
    words = [ws[int(i)] for i in nzw]
    C = np.ascontiguousarray(coef[:, nzw])
    del coef, ws
    rep['t_words'] = len(words)
    rep['t_degrees'] = {str(d): int(c) for d, c in zip(*np.unique([nferm(w) for w in words] or [0], return_counts=True))}
    log(f"t: {len(words)} words with nonzero coefficient, degrees {rep['t_degrees']} ({time.time()-T0:.0f}s)")
    np.savez(os.path.join(a.out, 'tvec.npz'), word_index=nzw, coef=C)
    pickle.dump(words, open(os.path.join(a.out, 'twords.pkl'), 'wb'), protocol=4)
    dw, DC = apply_lin(OPS[a.op], words, C)
    rep['op'], rep['dt_words'] = a.op, len(dw)
    log(f"{a.op} t: {len(dw)} words ({time.time()-T0:.0f}s)")
    vecs = [{dw[i]: int(DC[k, i]) for i in np.flatnonzero(DC[k])} for k in range(a.nvec)]
    pickle.dump(vecs, open(os.path.join(a.dcase, 'q1.pkl'), 'wb'), protocol=4)
    del vecs, dw, DC
    # controls D Q_0 u, u = one compressed hw row of each of the three smallest stripped contents of (5,5;14,8,8) Y=8
    log(f"loading {a.in_cache}")
    wi, gi, _, _ = pickle.load(open(a.in_cache, 'rb'))
    ctrl = []
    for g in sorted(gi, key=lambda g: g['rows'])[:3]:
        u = defaultdict(int)
        for widx, H, R in g['parts']:
            row = matmul_mod(R[:1] % P, H.astype(np.int64) % P)[0]
            for j in np.flatnonzero(row):
                u[wi[int(widx[j])]] = int(row[j])
        uw = list(u)
        img, val = apply_op(Q_multi, uw, np.array([[u[w] for w in uw]], dtype=np.int64))
        dimg, dval = apply_lin(OPS[a.op], img, val)
        ctrl.append({dimg[i]: int(dval[0, i]) for i in np.flatnonzero(dval[0])})
    del wi, gi
    rep['ctrl_words'] = [len(c) for c in ctrl]
    pickle.dump(ctrl, open(os.path.join(a.dcase, 'ctrl.pkl'), 'wb'), protocol=4)
    log(f"controls {a.op} Q_0 u: sizes {rep['ctrl_words']}")
    rep['time'] = round(time.time() - T0)
    json.dump(rep, open(os.path.join(a.out, 'build_make.json'), 'w'), indent=1)
    log(json.dumps(rep))


def check(a):
    """t != 0, t = 0 on the Cartan, Q_0 t = 0 and Q_0 D t = 0, at fresh random points (C-odd projection throughout)."""
    T0 = time.time()
    N = 3
    rep = json.load(open(os.path.join(a.out, 'build_make.json')))
    words = pickle.load(open(os.path.join(a.out, 'twords.pkl'), 'rb'))
    C = np.load(os.path.join(a.out, 'tvec.npz'))['coef']
    K = C.shape[0]
    v = eval_combo(words, C, N, 1, 9001, False, a.workers)
    rep['t_nonzero_values'] = [int(sum(int(np.count_nonzero(x[k])) for x in v.values())) for k in range(K)]
    log(f"t at fresh T_2 points: nonzero values per vector {rep['t_nonzero_values']} (must be > 0)")
    v = eval_combo(words, C, N, 4, 9101, True, a.workers)
    rep['t_cartan_nonzero'] = int(sum(int(np.count_nonzero(x)) for x in v.values()))
    rep['t_cartan_values'] = int(sum(x.size for x in v.values()))
    log(f"t at fresh Cartan points: {rep['t_cartan_nonzero']} nonzero of {rep['t_cartan_values']} (must be 0)")
    qw, QC = apply_op(Q_multi, words, C)
    log(f"Q_0 t: {len(qw)} image words ({time.time()-T0:.0f}s)")
    v = eval_combo(qw, QC, N, 2, 9201, False, a.workers)
    rep['q0t_nonzero'] = int(sum(int(np.count_nonzero(x)) for x in v.values()))
    rep['q0t_values'] = int(sum(x.size for x in v.values()))
    log(f"Q_0 t at fresh points of the image sector: {rep['q0t_nonzero']} nonzero of {rep['q0t_values']} (must be 0)")
    del qw, QC
    dw, DC = apply_lin(OPS[a.op], words, C)
    v = eval_combo(dw, DC, N, 1, 9301, False, a.workers)
    rep['dt_nonzero_values'] = [int(sum(int(np.count_nonzero(x[k])) for x in v.values())) for k in range(K)]
    log(f"D t at fresh points: nonzero values per vector {rep['dt_nonzero_values']}")
    qw, QC = apply_op(Q_multi, dw, DC)
    v = eval_combo(qw, QC, N, 2, 9401, False, a.workers)
    rep['q0dt_nonzero'] = int(sum(int(np.count_nonzero(x)) for x in v.values()))
    rep['q0dt_values'] = int(sum(x.size for x in v.values()))
    log(f"Q_0 D t at fresh points: {rep['q0dt_nonzero']} nonzero of {rep['q0dt_values']} (must be 0)")
    ok = (all(x > 0 for x in rep['t_nonzero_values']) and (a.allow_coulomb or rep['t_cartan_nonzero'] == 0)
          and rep['q0t_nonzero'] == 0 and rep['q0dt_nonzero'] == 0)
    rep.update(checks_ok=bool(ok), check_time=round(time.time() - T0))
    json.dump(rep, open(os.path.join(a.out, 'build_cand.json'), 'w'), indent=1)
    log(json.dumps({k: v for k, v in rep.items() if not isinstance(v, list) or len(v) < 8}))
    assert ok, "candidate checks failed"


if __name__ == '__main__':
    main()
