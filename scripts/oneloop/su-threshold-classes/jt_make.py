"""R-007 order 2, option 2: lowest-weight tests of the Coulomb LW classes of T_2 (summary/fortuity-summary.pdf).

t_k (d1_make.py: OUT/twords.pkl, tvec.npz) are cocycles whose Cartan images are LW, so each J t_k (J a lowering
generator) has eps(J t_k) = J eps(t_k) = 0: it is exact or a non-Coulomb class of the J-target sector.  [t] is LW in E_1
iff every J t is exact.  For each J in {D, Kp, Tp0, Tp21, S2, S20} (build_cand.OPS) this writes
OUTJ/vecs_<J>.pkl = dict(words, coef = [J Q_0 u (3 controls, exact since J Q_0 = +-Q_0 J); J t_k], n_ctrl = 3) for
veval.py against the stored J-target prep (Q_0 from Y = 8 into the target at Y = 9).
usage: python3 jt_make.py [--src classes/r007_d1] [--ops D,Kp,Tp0,Tp21,S2,S20] [--in-cache ...]"""
import argparse, json, os, pickle, time
from collections import defaultdict
import numpy as np
import fullpar
from fortuity.bmn.modp import P, matmul_mod
import build_cand as bc
try:
    from fortuity.full.words import Q_multi
except ImportError:
    from fortuity.full.words import Q_multi

log = lambda *a: print(time.strftime('%H:%M:%S'), *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='classes/r007_d1')
    ap.add_argument('--ops', default='D,Kp,Tp0,Tp21,S2,S20')
    ap.add_argument('--in-cache', default='cache/fullpar_hw_r007_o2_in_codd.pkl')
    a = ap.parse_args()
    fullpar.CSEC = -1
    T0 = time.time()
    words = pickle.load(open(os.path.join(a.src, 'twords.pkl'), 'rb'))
    C = np.load(os.path.join(a.src, 'tvec.npz'))['coef'] % P
    log(f"t: {C.shape[0]} vectors on {len(words)} words")
    # controls: Q_0 u for one compressed hw row of each of the three smallest stripped contents of (5,5;14,8,8), Y = 8
    wi, gi, _, _ = pickle.load(open(a.in_cache, 'rb'))
    q0u = []
    for g in sorted(gi, key=lambda g: g['rows'])[:3]:
        u = defaultdict(int)
        for widx, H, R in g['parts']:
            row = matmul_mod(R[:1] % P, H.astype(np.int64) % P)[0]
            for j in np.flatnonzero(row):
                u[wi[int(widx[j])]] = int(row[j])
        uw = list(u)
        q0u.append(bc.apply_op(Q_multi, uw, np.array([[u[w] for w in uw]], dtype=np.int64)))
    del wi, gi
    rep = {}
    for op in a.ops.split(','):
        img, V = bc.apply_lin(bc.OPS[op], words, C)
        ctrl = [bc.apply_lin(bc.OPS[op], qw, qv) for qw, qv in q0u]
        idx = {w: i for i, w in enumerate(img)}
        allw = list(img)
        for cw, _ in ctrl:
            for w in cw:
                if w not in idx:
                    idx[w] = len(allw)
                    allw.append(w)
        coef = np.zeros((3 + C.shape[0], len(allw)), dtype=np.int64)
        coef[3:, :len(img)] = V % P
        for r, (cw, cv) in enumerate(ctrl):
            coef[r, [idx[w] for w in cw]] = cv[0] % P
        out = os.path.join(a.src, f'vecs_{op}.pkl')
        pickle.dump(dict(words=allw, coef=coef, n_ctrl=3), open(out, 'wb'), protocol=4)
        rep[op] = dict(words=len(allw), nonzero_per_row=[int(np.count_nonzero(coef[r])) for r in range(coef.shape[0])])
        log(f"{op}: J t on {len(img)} words, controls {[len(cw) for cw, _ in ctrl]} -> {out} ({time.time()-T0:.0f}s)")
    rep['time'] = round(time.time() - T0)
    json.dump(rep, open(os.path.join(a.src, 'jt_make.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
