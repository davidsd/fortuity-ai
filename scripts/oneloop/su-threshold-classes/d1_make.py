"""R-007 order 2, option 1: the one-loop differential d_1 on the Coulomb lowest-weight classes of T_2 and on the
non-Coulomb candidate (summary/fortuity-summary.pdf).

kernel_vec_aug.py gives (x, z) with x^T A + z^T F = 0 on the T_2 outbound prep A = [B_out | Cartan], F = rows [0 | e_f]
of the 90 abelian LW polynomials f (lw90_T2.pkl).  So t = sum_i x_i h_i (h_i the compressed hw rows) is a Q_0-cocycle of
T_2 = (4,4;14,8,8), Y = 9 (C-odd [3,0] hw vectors) with Cartan image eps(t) = -sum_j z_j f_j.  Random kernel vectors
are random elements of (the <= 8 Coulomb LW classes) + (the non-Coulomb candidate) + (exact cocycles).

d_1[t] = [Q_1 t] in H(U_2), U_2 = (3,3;14,8,8), Y = 9.  Q_1 t is tested for Q_0-exactness against the stored order-3
outbound prep r007_o3_out_codd (Q_0 from (4,4;14,8,8), Y = 8 into U_2), with veval.py (V rows at its functionals) and
cpu_rank.py; the rank increments of the K rows = rank of d_1 on the classes they represent.  Controls Q_1 Q_0 u (u hw rows
of (5,5;14,8,8), Y = 8, as in build_cand.py; Q_1 Q_0 u = -Q_0 Q_1 u) must not raise the rank.

This script: t_k (as build_cand.make), the check eps(t_k) = -z_k . f at fresh Cartan points, Q_1 t_k and Q_1 Q_0 u
(q1apply.py), written as OUT/vecs_Q1.pkl = dict(words, coef = [controls; Q_1 t_k], n_ctrl, z) for veval.py, plus
OUT/twords.pkl, OUT/tvec.npz (t itself, for d1_check.py) and OUT/d1_make.json."""
import argparse, json, os, pickle, time
from collections import defaultdict
import numpy as np
import fullpar
from fullpar import frame_k, Frame, c_frame, nferm
from fortuity.bmn.modp import P, matmul_mod
import build_cand as bc
import membership
import q1apply
try:
    from fortuity.full.words import Q_multi
except ImportError:
    from fortuity.full.words import Q_multi

log = lambda *a: print(time.strftime('%H:%M:%S'), *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prep', default='prep/r007_o2_out_codd')
    ap.add_argument('--x', default='xker_aug.npy')
    ap.add_argument('--z', default='zker_aug.npy')
    ap.add_argument('--cache', default='cache/fullpar_hw_r007_o2_out_codd_cartan.pkl')
    ap.add_argument('--in-cache', default='cache/fullpar_hw_r007_o2_in_codd.pkl')
    ap.add_argument('--lw', default='lw90_T2.pkl')
    ap.add_argument('--out', default='classes/r007_d1')
    ap.add_argument('--nvec', type=int, default=12)
    ap.add_argument('--cand', default='classes/r007_o2cand', help="build_cand output with the candidate's t' (tvec.npz, "
                    "twords.pkl) to append as extra rows; '' to skip")
    ap.add_argument('--cand-only', action='store_true', help="test only the candidate's t' (no kernel-aug input)")
    ap.add_argument('--npts', type=int, default=48, help='Cartan points per degree for the eps(t) = -z.f check')
    ap.add_argument('--workers', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', '8')))
    a = ap.parse_args()
    fullpar.CSEC = -1
    os.makedirs(a.out, exist_ok=True)
    T0 = time.time()
    rep = dict(prep=a.prep, x=a.x, z=a.z, cache=a.cache, nvec=a.nvec, cand_only=bool(a.cand_only))
    if a.cand_only:
        # only the candidate's t' (no kernel-aug vectors): words and coefficients straight from its build_cand output
        cz = np.load(os.path.join(a.cand, 'tvec.npz'))
        nzw, C = cz['word_index'], np.ascontiguousarray(cz['coef'] % P)
        words = pickle.load(open(os.path.join(a.cand, 'twords.pkl'), 'rb'))
        assert len(words) == C.shape[1]
        nLW = len(pickle.load(open(a.lw, 'rb'))['space'])
        K = C.shape[0]
        Z = np.zeros((K, nLW), dtype=np.int64)
        rep['candidate_rows'] = list(range(K))
        log(f"candidate only: {K} vectors on {len(words)} words from {a.cand}")
    else:
        words, C, Z, K, nzw = build_t(a, rep)
    rep['t_words'] = len(words)
    np.savez(os.path.join(a.out, 'tvec.npz'), word_index=nzw, coef=C)
    pickle.dump(words, open(os.path.join(a.out, 'twords.pkl'), 'wb'), protocol=4)
    run_rest(a, rep, words, C, Z, K, T0)


def build_t(a, rep):
    T0 = time.time()
    X = np.load(os.path.join(a.prep, a.x))[:, :a.nvec] % P
    Z = np.load(os.path.join(a.prep, a.z))[:a.nvec] % P
    K = X.shape[1]
    nB = json.load(open(os.path.join(a.prep, 'meta.json')))['nB']
    assert X.shape[0] == nB, (X.shape, nB)
    rep['z_nonzero_per_vector'] = [int(np.count_nonzero(Z[k])) for k in range(K)]
    # t_k = sum_i x_ik h_i over the hw cache (as build_cand.make)
    log(f"loading {a.cache}")
    ws, groups, ms, stats = pickle.load(open(a.cache, 'rb'))
    coef = np.zeros((K, len(ws)), dtype=np.int64)
    g0 = 0
    for g in groups:
        Xg = np.ascontiguousarray(X[g0:g0 + g['k']].T)
        for widx, H, R in g['parts']:
            coef[:, widx] = (coef[:, widx] + matmul_mod(matmul_mod(Xg, R % P), H.astype(np.int64) % P)) % P
        g0 += g['k']
    assert g0 == nB
    del groups
    if a.cand:
        # the non-Coulomb candidate vectors t' (build_cand make, same hw cache): appended as explicit rows with z = 0,
        # so every test (d_1 and the LW tests) also reports the candidate on its own
        cz = np.load(os.path.join(a.cand, 'tvec.npz'))
        cidx, cC = cz['word_index'], cz['coef'] % P
        cw = pickle.load(open(os.path.join(a.cand, 'twords.pkl'), 'rb'))
        assert len(cw) == len(cidx) and all(cw[i] == ws[int(cidx[i])] for i in range(0, len(cidx), max(1, len(cidx) // 1000)))
        del cw
        extra = np.zeros((cC.shape[0], coef.shape[1]), dtype=np.int64)
        extra[:, cidx] = cC
        coef = np.vstack([coef, extra])
        Z = np.vstack([Z, np.zeros((cC.shape[0], Z.shape[1]), dtype=np.int64)])
        K = coef.shape[0]
        rep['candidate_rows'] = [K - cC.shape[0] + i for i in range(cC.shape[0])]
        log(f"appended {cC.shape[0]} candidate rows from {a.cand}: K = {K}")
    nzw = np.flatnonzero(coef.any(axis=0))
    words = [ws[int(i)] for i in nzw]
    C = np.ascontiguousarray(coef[:, nzw])
    del coef, ws
    log(f"t: {K} vectors on {len(words)} words ({time.time()-T0:.0f}s)")
    return words, C, Z, K, nzw


def run_rest(a, rep, words, C, Z, K, T0):
    # check eps(t_k) = -sum_j z_kj f_j at fresh Cartan points (C-odd projection on both sides)
    LW = pickle.load(open(a.lw, 'rb'))
    space = LW['space']
    assert Z.shape[1] == len(space)
    seed = 9001
    tc = bc.eval_combo(words, C, 3, a.npts, seed, True, a.workers)
    letters = sorted({L for w in words for n in w for L in n})
    zero_z = not Z.any()             # candidate rows only: the check is that t vanishes on the Cartan
    assert zero_z or {L[1:] for f in space for m in f for L in m} <= set(letters), "LW letters missing from t's words"
    mism, tot = 0, 0
    for d, Tv in tc.items():
        if zero_z:
            R_ = Tv % P
        else:
            k = frame_k(3, d)
            fr = Frame(3, a.npts, k, [d], letters, seed=seed + 31 * d, cartan=True)
            frc = c_frame(fr)
            Fv = np.array([(membership.abelian_eval(fr, f, d) - membership.abelian_eval(frc, f, d)) % P for f in space])
            R_ = (Tv + matmul_mod(Z, Fv)) % P
        mism += int(np.count_nonzero(R_))
        tot += R_.size
        log(f"  eps check degree {d}: t nonzero values {int(np.count_nonzero(Tv))} of {Tv.size}; "
            f"eps(t) + z.f nonzero {int(np.count_nonzero(R_))}")
    rep['eps_check_nonzero'], rep['eps_check_values'] = mism, tot
    assert mism == 0, "eps(t) != -z.f"
    # Q_1 t
    img, V = q1apply.apply(words, C, N=3, workers=a.workers, chunk=2000, log=log)
    rep['q1t_words'] = len(img)
    rep['q1t_nonzero_per_vector'] = [int(np.count_nonzero(V[k])) for k in range(K)]
    log(f"Q_1 t: {len(img)} words ({time.time()-T0:.0f}s)")
    del words, C
    # controls Q_1 Q_0 u
    log(f"loading {a.in_cache}")
    wi, gi, _, _ = pickle.load(open(a.in_cache, 'rb'))
    cw, cC = [], []
    for g in sorted(gi, key=lambda g: g['rows'])[:3]:
        u = defaultdict(int)
        for widx, H, R in g['parts']:
            row = matmul_mod(R[:1] % P, H.astype(np.int64) % P)[0]
            for j in np.flatnonzero(row):
                u[wi[int(widx[j])]] = int(row[j])
        uw = list(u)
        qimg, qval = bc.apply_op(Q_multi, uw, np.array([[u[w] for w in uw]], dtype=np.int64))
        cimg, cval = q1apply.apply(qimg, qval, N=3, workers=a.workers, chunk=500)
        cw.append(cimg)
        cC.append(cval[0])
    del wi, gi
    rep['ctrl_words'] = [len(x) for x in cw]
    log(f"controls Q_1 Q_0 u: sizes {rep['ctrl_words']}")
    # one word list for controls and vectors
    idx = {w: i for i, w in enumerate(img)}
    allw = list(img)
    for x in cw:
        for w in x:
            if w not in idx:
                idx[w] = len(allw)
                allw.append(w)
    coefs = np.zeros((3 + K, len(allw)), dtype=np.int64)
    coefs[3:, :len(img)] = V
    for r, (x, cv) in enumerate(zip(cw, cC)):
        coefs[r, [idx[w] for w in x]] = cv
    pickle.dump(dict(words=allw, coef=coefs, n_ctrl=3, z=Z), open(os.path.join(a.out, 'vecs_Q1.pkl'), 'wb'), protocol=4)
    rep['vec_words'] = len(allw)
    rep['time'] = round(time.time() - T0)
    json.dump(rep, open(os.path.join(a.out, 'd1_make.json'), 'w'), indent=1)
    log(json.dumps(rep))


if __name__ == '__main__':
    main()
