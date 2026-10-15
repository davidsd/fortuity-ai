"""SU(N) > SU(N-1) x U(1) block restriction for full-letter multitrace words (generalises
fortuity.bmn.blockrestrict.restrict_u1, R-016): X = diag(A, 0) + x D, D = diag(1,..,1,-(N-1)), for the scalar letters
phi_m = (0,0,(m,)) only (the abelian components of every other letter are set to zero).  The abelian variables are
central and Q_0 acts by graded commutators, so the coefficient of a fixed monomial prod x_m^{k_m} is a chain map
C(SU(N)) -> C(SU(N-1)) (sends exact to exact).  tr W -> tr_{N-1}(prod (A_i + x_i)) + prod_i(-(N-1) x_i) [all-phi]:
replacing r scalars in a factor deletes them; a factor with one letter left vanishes; a fully replaced factor of
length L gives (N-1) + (-(N-1))^L.  Deleting bosonic letters never reorders fermions; signs come only from
re-canonicalisation."""
from collections import defaultdict
from itertools import combinations
from .words import canon_word, canon_multi

PHI = {(0, 0, (0,)): 0, (0, 0, (1,)): 1, (0, 0, (2,)): 2}


def _factor_options(w, N, target):
    opts = []
    pos = [i for i, L in enumerate(w) if L in PHI]
    tmax = sum(target)
    for r in range(0, min(len(pos), tmax) + 1):
        for S in combinations(pos, r):
            cnt = [0, 0, 0]
            for i in S:
                cnt[PHI[w[i]]] += 1
            if any(cnt[m] > target[m] for m in range(3)):
                continue
            rem = len(w) - r
            if r == 0:
                opts.append(((0, 0, 0), w, 1))
            elif rem == 0:
                L = len(w)
                opts.append((tuple(cnt), None, (N - 1) + (-(N - 1)) ** L))
            elif rem == 1:
                continue
            else:
                w2 = tuple(L for i, L in enumerate(w) if i not in S)
                s, cw = canon_word(w2)
                if s:
                    opts.append((tuple(cnt), cw, s))
    return opts


def restrict_monomial(multi, N, target):
    """Coefficient of prod_m x_m^{target[m]} in the restriction of the canonical multitrace 'multi':
    dict {SU(N-1) multitrace: integer coefficient}."""
    out = defaultdict(int)
    facs = [_factor_options(w, N, target) for w in multi]

    def rec(i, cnt, chosen, coeff):
        if i == len(facs):
            if tuple(cnt) != tuple(target):
                return
            s, cm = canon_multi(chosen) if chosen else (1, ())
            if s:
                out[cm] += s * coeff
            return
        for c, w2, val in facs[i]:
            nc = [cnt[m] + c[m] for m in range(3)]
            if any(nc[m] > target[m] for m in range(3)):
                continue
            rec(i + 1, nc, chosen + ([w2] if w2 is not None else []), coeff * val)

    rec(0, [0, 0, 0], [], 1)
    return {m: c for m, c in out.items() if c}
