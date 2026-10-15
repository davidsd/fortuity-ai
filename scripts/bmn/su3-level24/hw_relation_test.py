"""Q-exactness of the 'fundamental trace relations' of arXiv:2312.16443 eq. (A.4) at N=3, tested in the SU(3)_R
highest-weight complex of fortuity.hwcoh.  For a hw operator T in sector (n=2J; lam):
exact  <=>  T in span of Q(hw vectors of sector n+1), at functionals certified faithful (rank = exact hw multiplicity).
usage: python3 hw_relation_test.py <case> [seed]"""
import sys, json, time, pickle, os, numpy as np
from fractions import Fraction
from build_O import eval_term, E, addto
from build_O6 import to_int
from gens_util import canon_expr, load_O
from fortuity.hwcoh.dims import partition_dims_nf
from fortuity.hwcoh.sector import Sector
from fortuity.hwcoh.gravhw import GravitonAlgebra
from fortuity.hwcoh import ffla
from fortuity.hwcoh.hw import E_multi
from fortuity.hwcoh.evalg import nferm
from fortuity.bmn.words import expr_sector, Q_expr
from fortuity.bmn.modp import matmul_mod, P

def build(terms):
    tot = E()
    for c, f, s in terms: addto(tot, eval_term(Fraction(c), f, s))
    return canon_expr(to_int(tot)[0])
F0 = lambda l: ('fix', [l, 0])
CASES = {
 'R18': ((3, 3, 3), 2, lambda: build([(1, [('eps', ['a1', 'a2', 'a3']), ('u3', ['a1', 'b', 'c']), ('v2', ['a2', 'b']), ('v2', ['a3', 'c'])], [])])),
 'R20': ((4, 3, 3), 3, lambda: build([(2, [F0('i'), ('v2', ['a', 'c']), ('v2', ['b', 'a']), ('v3', ['i', 'c', 'b'])], []),
                                       (-3, [F0('i'), ('v2', ['i', 'a']), ('v2', ['c', 'b']), ('v3', ['a', 'b', 'c'])], [])])),
 'R22': ((5, 3, 3), 3, lambda: build([(1, [F0('i'), F0('j'), ('u2', ['i', 'j']), ('v2', ['a', 'b']), ('v2', ['b', 'c']), ('v2', ['c', 'a'])], []),
                                       (-3, [F0('i'), F0('j'), ('u2', ['a', 'i']), ('v2', ['j', 'b']), ('v2', ['b', 'c']), ('v2', ['c', 'a'])], []),
                                       (3, [F0('i'), F0('j'), ('u2', ['a', 'b']), ('v2', ['i', 'a']), ('v2', ['j', 'c']), ('v2', ['c', 'b'])], [])])),
 'R24': ((4, 4, 4), 3, lambda: build([(1, [('eps', ['a1', 'a2', 'a3']), ('u2', ['a1', 'b']), ('v2', ['a2', 'b']), ('v3', ['a3', 'c', 'd']), ('v2', ['d', 'c'])], [])])),
 'O48': ((4, 4, 4), 3, lambda: canon_expr(load_O('O_cckll.json')[0])),
 'O1':  ((4, 4, 4), 3, lambda: canon_expr(load_O('O1_control.json')[0])),
 'R26': ((5, 4, 4), 4, lambda: build([(1, [F0('i'), ('v2', ['i', 'a']), ('v2', ['a', 'b']), ('v2', ['d', 'c']), ('v3', ['b', 'c', 'd'])], [])])),
 'QR24_1': ((5, 4, 4), 4, lambda: canon_expr(__import__('fortuity.bmn.words', fromlist=['Qplus_expr']).Qplus_expr(CASES['R24'][2](), 0))),
 'QR24_12': ((5, 5, 4), 5, lambda: canon_expr((lambda Qp: Qp(Qp(CASES['R24'][2](), 0), 1))(__import__('fortuity.bmn.words', fromlist=['Qplus_expr']).Qplus_expr))),
 'QR24_123': ((5, 5, 5), 6, lambda: canon_expr((lambda Qp: Qp(Qp(Qp(CASES['R24'][2](), 0), 1), 2))(__import__('fortuity.bmn.words', fromlist=['Qplus_expr']).Qplus_expr))),
 'R30s': ((5, 5, 5), 5, lambda: build([(1, [('v2', ['a', 'b']), ('v2', ['b', 'c']), ('v2', ['c', 'd']), ('v2', ['d', 'e']), ('v2', ['e', 'a'])], [])])),
 'R30t': ((7, 4, 4), 5, lambda: build([(1, [F0('i'), F0('j'), F0('k'), ('eps', ['a1', 'a2', 'i']), ('v2', ['j', 'a1']), ('v2', ['k', 'a2']), ('v2', ['b', 'c']), ('v2', ['c', 'd']), ('v2', ['d', 'b'])], [])])),
}
def run(cases, seed=1, maxlen=None):
    t0 = time.time(); log = lambda *a: print('[%6.0fs]' % (time.time() - t0), *a, flush=True)
    lam, n = CASES[cases[0]][0], CASES[cases[0]][1]
    assert all(CASES[c][0] == lam and CASES[c][1] == n for c in cases)
    j = 2 * sum(lam)
    cache = 'runs/dims_nf_%d.pkl' % j
    if os.path.exists(cache): dims = pickle.load(open(cache, 'rb'))
    else:
        dims = partition_dims_nf(3, j); pickle.dump(dims, open(cache, 'wb'))
    log('sector j=%d lam=%s 2J=%d ; dims ready' % (j, lam, n))
    targets = {}
    for c in cases:
        T = CASES[c][2]()
        assert expr_sector(T) == (n,) + lam, (c, expr_sector(T))
        hw = True
        for i in (1, 2):
            img = {}
            for m, v in T.items():
                for m2, c2 in E_multi(m, i).items(): img[m2] = img.get(m2, 0) + v * c2
            hw &= not any(img.values())
        targets[c] = T
        log('%s: %d canonical terms | formally Q-closed: %s | highest weight (E1, E2 annihilate): %s' % (c, len(T), not canon_expr(Q_expr(T)), hw))
    S = Sector(3, lam, n, dims, seed=seed, log=lambda s: log(s), graviton_algebra=GravitonAlgebra(3), maxlen=maxlen)
    S.prepare()
    A, B, G = S.evaluate()
    ok = True; col = 0; rk = {}
    for d in sorted(S.m_d):
        w = sum(fr.ncols for dd, fr in S.colblocks if dd == d)
        rows = np.flatnonzero(A[:, col:col + w].any(axis=1))
        rk[d] = ffla.rank(A[rows, col:col + w]) if rows.size else 0
        ok &= (rk[d] == S.m_d[d]); col += w
    log('faithfulness check: rank of hw vectors per fermion degree %s vs exact multiplicities %s -> %s' % (rk, S.m_d, 'OK' if ok else 'FAIL'))
    r_up = ffla.rank(B) if B.shape[0] else 0
    rg = ffla.rank(np.concatenate([B, G], axis=0)) if G.shape[0] else r_up
    log('rank of Q-exact hw space = %d ; graviton classes = %d' % (r_up, rg - r_up))
    out = {'lam': lam, 'n': n, 'check': bool(ok), 'r_up': int(r_up), 'grav': int(rg - r_up), 'targets': {}}
    for c, T in targets.items():
        words = list(T); v = np.zeros(B.shape[1], dtype=np.int64); col = 0
        for d, fr in S.colblocks:
            idx = [i for i, w in enumerate(words) if nferm(w) == d]
            if idx:
                X = fr.eval_words([words[i] for i in idx])
                cf = np.array([[T[words[i]] % P for i in idx]], dtype=np.int64)
                v[col:col + fr.ncols] = matmul_mod(cf, X)[0]
            col += fr.ncols
        nonzero = bool(v.any())
        r1 = ffla.rank(np.concatenate([B, v[None, :]], axis=0))
        r2 = ffla.rank(np.concatenate([B, G, v[None, :]], axis=0)) if G.shape[0] else r1
        res = dict(nonzero=nonzero, exact=bool(r1 == r_up), in_graviton_plus_exact=bool(r2 == rg))
        out['targets'][c] = res
        log('%-5s nonzero at N=3: %s | Q-EXACT: %s | in gravitons + exact: %s' % (c, nonzero, res['exact'], res['in_graviton_plus_exact']))
    return out
if __name__ == '__main__':
    cases = sys.argv[1].split(','); seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    ml = int(os.environ['MAXLEN']) if os.environ.get('MAXLEN') else None
    res = run(cases, seed, ml)
    json.dump(res, open('runs/hwrel_%s_seed%d.json' % ('_'.join(cases), seed), 'w'), indent=1)
