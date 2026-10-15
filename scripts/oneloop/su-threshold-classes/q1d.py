"""One-loop supercharge Q_1 on letters WITH derivatives (fortuity.full.q1 covers derivative-free pairs only).

Source: Budzik-Kulp arXiv:2512.07771, App. B (also KimKimLeePark arXiv:2606.27955, eqs. (2.4)-(2.10)).  For component
fields X (underived, at the origin) and Y with m, n derivatives,
    Q_1(X d1^m d2^n Y) = [underived rule with  (1/2) d_a X' d^a Y'  replaced by
         (1/(m+n+2)) sum_{k<=m, l<=n} (1/(k+l+1)) C(m,k) C(n,l)  d1^k d2^l d_a X'  d1^(m-k) d2^(n-l) d^a Y' ],
the same weights for every component pair (so the superfield form (theta - theta')^3 d_a C^D d^a C^E carries over).
Derivatives on the first field follow from Q_1(d f g) = d Q_1(f g) - Q_1(f d g) (Q_1 commutes with derivatives):
    Q_1(d^p X  d^q Y) = sum_j C(p, j) (-1)^j d^(p-j) [ Q_1(X d^(q+j) Y) ],   per direction,
with the outer d^(p-j) distributed over the two output fields by Leibniz.

Letters (a, b, S) are Taylor coefficients: letter = d1^a d2^b C_S / (a! b!).  fortuity.full.q1.pair_rule gives the
theta / epsilon structure of the underived pair with the 1/2 of D^tri_{0,0} absorbed, so the derived weights are
2 C(m,k) C(n,l) / ((m+n+2)(k+l+1)) relative to it; _fierz gives the gauge (trace splitting/joining) structure, which does
not see derivatives.  Coefficients are returned mod p (the weights are fractions).
"""
from functools import lru_cache
from math import comb, factorial
from collections import defaultdict
try:
    from fortuity.full.q1 import pair_rule, _fierz
    from fortuity.full.letters import is_ferm
except ImportError:            # older clones: the original fullsector modules
    from fortuity.full.q1 import pair_rule, _fierz
    from fortuity.full.letters import is_ferm
from fortuity.bmn.modp import P

_inv = lambda x: pow(x % P, P - 2, P)


@lru_cache(maxsize=None)
def pair_terms(L1, L2):
    """Q_1 on the ordered pair of letters (L1 in the first slot, L2 in the second): tuple of (coeff mod p, LD, LE),
    LD replacing L1 and LE replacing L2 (before the gauge reconnection)."""
    a1, b1, S1 = L1
    a2, b2, S2 = L2
    rule = pair_rule(S1, S2)
    if not rule:
        return ()
    norm_in = _inv(factorial(a1) * factorial(b1) * factorial(a2) * factorial(b2))
    out = defaultdict(int)
    for j1 in range(a1 + 1):
        for j2 in range(b1 + 1):
            sj = (-1) ** (j1 + j2) * comb(a1, j1) * comb(b1, j2)
            m, n = a2 + j1, b2 + j2                   # derivatives on Y in the inner Q_1(X d^(m,n) Y)
            p_, q_ = a1 - j1, b1 - j2                 # outer derivatives
            for c, LD, LE in rule:
                for k in range(m + 1):
                    for l in range(n + 1):
                        w = 2 * comb(m, k) * comb(n, l) * _inv((m + n + 2) * (k + l + 1)) % P
                        Da, Db = k + LD[0], l + LD[1]
                        Ea, Eb = m - k + LE[0], n - l + LE[1]
                        for u in range(p_ + 1):
                            for v in range(q_ + 1):
                                xa, xb, ya, yb = Da + u, Db + v, Ea + p_ - u, Eb + q_ - v
                                cf = (sj * c * comb(p_, u) * comb(q_, v) * factorial(xa) * factorial(xb)
                                      * factorial(ya) * factorial(yb)) % P
                                key = ((xa, xb, LD[2]), (ya, yb, LE[2]))
                                out[key] = (out[key] + cf * w % P * norm_in) % P
    return tuple((c, LD, LE) for (LD, LE), c in out.items() if c)


def Q1d_multi(multi, N, group='SU'):
    """Q_1 of a canonical multitrace of arbitrary letters -> {multitrace: coeff mod p}."""
    letters = [L for w in multi for L in w]
    loc = [(a, u) for a, w in enumerate(multi) for u in range(len(w))]
    n = len(letters)
    par = [1 if is_ferm(L) else 0 for L in letters]
    out = defaultdict(int)
    for i in range(n):
        for j in range(i + 1, n):
            rule = pair_terms(letters[i], letters[j])
            if not rule:
                continue
            eps = par[i] * sum(par[k] for k in range(i)) + par[j] * sum(par[k] for k in range(j) if k != i)
            s_front = -1 if eps % 2 else 1
            rest = [k for k in range(n) if k != i and k != j]
            for c, LD, LE in rule:
                for term_multi, term_sign in (_fierz(multi, loc, i, j, LD, LE, rest, N, par, group) if group != 'SU'
                                              else _fierz(multi, loc, i, j, LD, LE, rest, N, par)):
                    out[term_multi] = (out[term_multi] + c * s_front * term_sign) % P
    return {m: c for m, c in out.items() if c}
