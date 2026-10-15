"""Restriction of BMN-sector operators to block subgroups (T-028, result R-016).

Two block embeddings, both with the non-abelian block H1 in the upper-left corner:

  SU(N)    >  SU(N-1) x U(1)     letters  X = [[A, 0], [0, 0]] + x D,   D = diag(1,...,1, -(N-1))
  SO(2N+1) >  SO(2N+1-2b) x SO(2)^b      X = diag(A, x^(1) eps, ..., x^(b) eps),   eps = [[0, 1], [-1, 0]]

Only the scalars phi_m get components along the abelian factors (numbers x_m, resp. x^(c)_m for block c); the
abelian components of psi_m and f are set to zero.  Q is built from commutators, the abelian components are
central, so Q acts only on the H1 part; hence the coefficient of any fixed monomial in the x's is a chain map
    C(G)  ->  C(H1)
(from G-invariant operators to H1-invariant ones), which sends G-exact to H1-exact and G-graviton polynomials to
H1-graviton polynomials (single traces restrict to polynomials in H1 single traces and x's).  If the image of a
G-class has a non-graviton component in H1, the G-class is not monotone.

Both maps are implemented formally, on multi-trace words (canonical necklaces as in fortuity.bmn.words):

  * U(1):  tr W  ->  tr_{N-1}( prod_i (A_i + x_i 1) )  +  prod_i ( -(N-1) x_i )  [second term only if W is all phi].
           Expanding, a phi letter can be replaced by x_m (deleted from the word); a trace left with one letter
           vanishes (traceless); a trace with all letters replaced contributes (N-1) + (-(N-1))^L.
  * SO(2): tr W  ->  tr_{2N+1-2b}(W)  +  sum_c [W all phi, L even] 2 (-1)^(L/2) prod_i x^(c)_{m_i}.
           A whole (pure-phi, even-length) trace factor can be moved into one SO(2) block.

Deleting bosonic letters does not reorder fermions; the only signs come from re-canonicalising rotations
(canon_word) and the order of odd trace factors (canon_multi).  Everything is mod p (coefficients as in the
repo's F_p pipeline).

Supported: SU(N) (U(1) block, restrict_u1) and SO(2N+1) (SO(2) blocks, restrict_so2).  The SO(2) map does not depend
on the rank.  Validation: code/tests/test_blockrestrict.py (formal map = direct evaluation on block-diagonal matrices,
chain-map property, composition of single-block steps).
"""
from collections import defaultdict
from itertools import combinations
import numpy as np

from fortuity.bmn.words import canon_word, canon_multi, PHI
from fortuity.bmn.modp import P


# --------------------------------------------------------------------------------------------------------------
# formal restriction maps
# --------------------------------------------------------------------------------------------------------------

def _factor_options_u1(w, Nsmall, target, p):
    """Options for one trace factor under SU(N) > SU(N-1) x U(1): list of (counts, word or None, value)."""
    opts = []
    phi_pos = [i for i, l in enumerate(w) if l in PHI]
    tmax = sum(target)
    for r in range(0, min(len(phi_pos), tmax) + 1):
        for S in combinations(phi_pos, r):
            cnt = [0, 0, 0]
            for i in S:
                cnt[w[i]] += 1
            if any(cnt[m] > target[m] for m in range(3)):
                continue
            rem = len(w) - r
            if r == 0:
                opts.append(((0, 0, 0), w, 1))
            elif rem == 0:                       # whole trace replaced: upper block tr(1) + lower block
                L = len(w)
                opts.append((tuple(cnt), None, (Nsmall + pow(-Nsmall, L, p)) % p))
            elif rem == 1:                       # tr of a single su(N-1) letter vanishes
                continue
            else:
                Sset = set(S)
                w2 = tuple(l for i, l in enumerate(w) if i not in Sset)
                s, cw = canon_word(w2)
                if s == 0:
                    continue
                opts.append((tuple(cnt), cw, s % p))
    return opts


def restrict_u1(expr, Nsmall, target, p=P):
    """Coefficient of x_1^t1 x_2^t2 x_3^t3 (target = (t1, t2, t3)) in the SU(N) > SU(Nsmall) x U(1) restriction of
    expr ({multitrace: coeff}); returns an expression in SU(Nsmall) words (coefficients mod p)."""
    target = tuple(target)
    out = defaultdict(int)
    cache = {}
    for multi, c in expr.items():
        per = []
        for w in multi:
            o = cache.get(w)
            if o is None:
                o = cache[w] = _factor_options_u1(w, Nsmall, target, p)
            per.append(o)
        # combine factors: total counts must equal target
        stack = [(0, (0, 0, 0), [], c % p)]
        while stack:
            k, acc, words, coef = stack.pop()
            if k == len(per):
                if acc == target:
                    s, m = canon_multi(words)
                    if s:
                        out[m] = (out[m] + s * coef) % p
                continue
            for cnt, cw, val in per[k]:
                new = (acc[0] + cnt[0], acc[1] + cnt[1], acc[2] + cnt[2])
                if new[0] > target[0] or new[1] > target[1] or new[2] > target[2]:
                    continue
                stack.append((k + 1, new, words if cw is None else words + [cw], coef * val % p))
    return {m: v for m, v in out.items() if v % p}


def _so2_value(w):
    """(counts, value) if the trace factor can sit in one SO(2) block (all phi, even length), else None."""
    if len(w) % 2 or any(l not in PHI for l in w):
        return None
    cnt = [0, 0, 0]
    for l in w:
        cnt[l] += 1
    return tuple(cnt), (2 if (len(w) // 2) % 2 == 0 else -2)


def restrict_so2(expr, targets, p=P):
    """Coefficient of prod_c (x^(c))^targets[c] in the SO(2N+1) > SO(2N+1-2b) x SO(2)^b restriction (b = len(targets)).
    Returns an expression in SO(2N+1-2b) words (coefficients mod p)."""
    targets = [tuple(t) for t in targets]
    b = len(targets)
    out = defaultdict(int)
    for multi, c in expr.items():
        vals = [_so2_value(w) for w in multi]
        # assign each factor to 'stay' (-1) or to one block c, block counts must equal targets
        stack = [(0, tuple((0, 0, 0) for _ in range(b)), [], c % p)]
        while stack:
            k, acc, words, coef = stack.pop()
            if k == len(multi):
                if all(acc[i] == targets[i] for i in range(b)):
                    s, m = canon_multi(words)
                    if s:
                        out[m] = (out[m] + s * coef) % p
                continue
            w = multi[k]
            stack.append((k + 1, acc, words + [w], coef))
            if vals[k] is not None:
                cnt, val = vals[k]
                for i in range(b):
                    new = (acc[i][0] + cnt[0], acc[i][1] + cnt[1], acc[i][2] + cnt[2])
                    if any(new[j] > targets[i][j] for j in range(3)):
                        continue
                    acc2 = acc[:i] + (new,) + acc[i + 1:]
                    stack.append((k + 1, acc2, words, coef * val % p))
    return {m: v for m, v in out.items() if v % p}


# --------------------------------------------------------------------------------------------------------------
# numerical validation: evaluate G-words on block-diagonal points built from H1 points
# --------------------------------------------------------------------------------------------------------------

def embed_points(pts_big, pts_small, kind, xvals, p=P):
    """Overwrite the letters of pts_big (a fortuity.bmn.evaluate.Points of the big group, same npts, k, degrees)
    with block-diagonal embeddings of pts_small's letters plus abelian scalar parts.
    kind='U1': xvals = (x1, x2, x3) numbers (U(1) parts of phi_1..3);  kind='SO2': xvals = list over blocks of triples."""
    ns = pts_small.letter[0][0].shape[-1]
    nb = pts_big.letter[0][0].shape[-1]
    for l, comps in pts_small.letter.items():
        new = {}
        for d, arr in comps.items():
            big = np.zeros(arr.shape[:-2] + (nb, nb), dtype=np.int64)
            big[..., :ns, :ns] = arr
            if d == 0 and l in PHI:
                if kind == 'U1':
                    Dg = np.ones(nb, dtype=np.int64)
                    Dg[-1] = (-(nb - 1)) % p
                    idx = np.arange(nb)
                    big[..., idx, idx] = (big[..., idx, idx] + xvals[l] * Dg) % p
                else:
                    for c, xv in enumerate(xvals):
                        o = ns + 2 * c
                        big[..., o, o + 1] = xv[l] % p
                        big[..., o + 1, o] = (-xv[l]) % p
            new[d] = big % p
        pts_big.letter[l] = new
    pts_big._neck_cache.clear()
    pts_big._cache_bytes = 0
    return pts_big


def interpolate_coefficient(values, ts, deg_target, p=P):
    """values[i] = vector evaluated at scale t = ts[i] (a polynomial in t of degree < len(ts));
    return the coefficient vector of t^deg_target (Vandermonde solve mod p)."""
    n = len(ts)
    V = np.array([[pow(int(t), e, p) for e in range(n)] for t in ts], dtype=object)
    # invert V mod p (small n): Gauss-Jordan over python ints
    M = [list(V[i]) + [1 if i == j else 0 for j in range(n)] for i in range(n)]
    for col in range(n):
        piv = next(r for r in range(col, n) if M[r][col] % p)
        M[col], M[piv] = M[piv], M[col]
        inv = pow(int(M[col][col]), p - 2, p)
        M[col] = [(a * inv) % p for a in M[col]]
        for r in range(n):
            if r != col and M[r][col] % p:
                f = M[r][col]
                M[r] = [(a - f * b) % p for a, b in zip(M[r], M[col])]
    Vinv = [row[n:] for row in M]
    Y = np.array(values, dtype=object)
    coef = sum(int(Vinv[deg_target][i]) * Y[i] for i in range(n))
    return np.array([int(c) % p for c in coef], dtype=np.int64)


# --------------------------------------------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------------------------------------------

def load_expr_json(path):
    """Expressions stored as {'terms': [[coeff, [[letters], ...]], ...]} (e.g. the T-021 representative)."""
    import gzip, json
    op = gzip.open if str(path).endswith('.gz') else open
    with op(path, 'rt') as f:
        d = json.load(f)
    return {tuple(tuple(w) for w in m): int(c) for c, m in d['terms']}


def save_expr_json(expr, path, **meta):
    import gzip, json
    op = gzip.open if str(path).endswith('.gz') else open
    with op(path, 'wt') as f:
        json.dump(dict(meta, prime=P, terms=[[int(c), [list(w) for w in m]] for m, c in sorted(expr.items())]), f)
