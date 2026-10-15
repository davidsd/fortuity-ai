"""T-029 (exploratory): lower bounds on P-odd SU(3) black-hole CORES from indices + SU(2) data.

Block restriction SU(3) > SU(2)xU(1) (R-016): for every monomial x^alpha (|alpha| = n) the coefficient map rho_alpha is
a chain map, SU(3)_R-equivariant when x transforms in [1,0], preserving 2J and the graviton parity P (x is C-odd and
carries one letter).  Gravitons of SU(2) are P-even, so a P-odd SU(3) class restricts, for every alpha, to P-odd (odd-Y)
SU(2) classes, i.e. SU(2) black holes, or to zero.  Call a P-odd class a CORE if all its restrictions vanish.  As SU(3)_R
representations,

    H^{P-}_{SU(3)}(level s) / cores  embeds into  sum_{n >= 0} Sym^n[1,0] (x) H^{P-}_{SU(2)}(level s - n)

(the combined map sum_alpha rho_alpha is injective on the complement of the cores).  Hence, per irrep lam,

    #cores(lam) >= m^{P-}_{SU(3)}(lam) - C(lam),   C = multiplicity of lam in the right-hand side ("fused capacity"),

with m^{P-}_{SU(3)} >= -T_SU(3) (R-017 kernel parity index) and m^{P-}_{SU(2)} the number of odd-Y SU(2) classes.
m^{P-}_{SU(2)} is exact for j <= 28 (R-001) and in the (6,6,6) column; elsewhere it is -T_SU(2) under the hypothesis
"all SU(2) black holes have odd Y and there are no SU(2) non-Coulomb gravitons", which T-027 tests.

usage: python3 core_bounds.py [SU3_KERNEL_JSON]   (default runs/su3_kernel_parity_j40.json; SU(2) from runs/su2_predictions_j44.json)
"""
import os, sys, json
from itertools import combinations_with_replacement
HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, '..', 'runs')


def schur_weights(lam):
    """weight multiplicities of the GL(3) irrep with highest weight lam = (l1 >= l2 >= l3 >= 0): SSYT of shape lam."""
    from collections import Counter
    l1, l2, l3 = lam
    out = Counter()
    # rows: row1 length l1, row2 l2, row3 l3; entries 1..3 weakly increasing in rows, strictly in columns
    def rows_of(length, lo_bounds):
        # all weakly increasing rows of given length with entry[i] > lo_bounds[i] (column strictness)
        res = []
        def rec(i, prev, cur):
            if i == length:
                res.append(tuple(cur)); return
            lo = max(prev, lo_bounds[i] + 1 if i < len(lo_bounds) else 1)
            for v in range(lo, 4):
                cur.append(v); rec(i + 1, v, cur); cur.pop()
        rec(0, 1, [])
        return res
    for r1 in rows_of(l1, [0] * l1):
        for r2 in rows_of(l2, list(r1[:l2])):
            for r3 in rows_of(l3, list(r2[:l3])):
                c = [0, 0, 0]
                for v in r1 + r2 + r3:
                    c[v - 1] += 1
                out[tuple(c)] += 1
    return out


def hw_decompose(wts):
    """irrep multiplicities (highest weights) of a weight-multiplicity dict (GL(3) weights, fixed level)."""
    import numpy as np
    a12 = np.array([1, -1, 0]); a23 = np.array([0, 1, -1])
    terms = [(0, 0, 1), (1, 0, -1), (0, 1, -1), (2, 1, 1), (1, 2, 1), (2, 2, -1)]
    out = {}
    for lam in wts:
        if lam[0] >= lam[1] >= lam[2]:
            c = sum(sg * wts.get(tuple(np.array(lam) + i * a12 + j * a23), 0) for i, j, sg in terms)
            if c:
                out[lam] = c
    return out


if __name__ == '__main__':
    k3 = json.load(open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(RUNS, 'su3_kernel_parity_j40.json')))
    su2 = json.load(open(os.path.join(RUNS, 'su2_predictions_j44.json')))['rows']
    m2 = {tuple(r['lam']): -r['T'] for r in su2 if r['T'] < 0}          # odd-Y SU(2) classes per irrep (hypothesis)
    T3 = {tuple(r['lam']): r['T'] for r in k3['rows']}
    smax = max(sum(l) for l in T3)
    from collections import Counter
    cap = {}
    for s in range(1, smax + 1):
        w = Counter()
        for mu, mult in m2.items():
            n = s - sum(mu)
            if n < 0:
                continue
            chi = schur_weights(mu)
            for alpha in combinations_with_replacement(range(3), n):     # Sym^n [1,0]: monomials x^alpha
                a = [0, 0, 0]
                for i in alpha:
                    a[i] += 1
                for wt, c in chi.items():
                    w[(wt[0] + a[0], wt[1] + a[1], wt[2] + a[2])] += mult * c
        cap.update(hw_decompose(w))
    print('SU(3) columns with forced P-odd classes (-T > 0): -T vs fused capacity C from SU(2) black holes')
    rows = []
    for lam in sorted(T3, key=lambda l: (sum(l), l)):
        if T3[lam] >= 0:
            continue
        need, C = -T3[lam], cap.get(lam, 0)
        rows.append(dict(lam=lam, j=2 * sum(lam), forced=need, capacity=C, core_lower_bound=max(0, need - C)))
        print(f"  j={2 * sum(lam):2} {lam}: forced {need:2}  capacity {C:3}  -> cores >= {max(0, need - C)}")
    json.dump(dict(description='lower bounds on P-odd SU(3) cores: forced (-T, R-017) minus the fused capacity from SU(2) '
                               'odd-Y black holes (Sym^n[1,0] x H^{P-}_{SU(2)}); SU(2) counts = -T_SU(2) (exact for j <= 28 and '
                               '(6,6,6); elsewhere assumes all SU(2) black holes odd-Y and no non-Coulomb gravitons)', rows=rows),
              open(os.path.join(RUNS, 'su3_core_bounds.json'), 'w'), indent=1, default=str)
