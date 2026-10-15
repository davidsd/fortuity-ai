"""Single-trace graviton states of SU(3) N=4 SYM on full letters, and multi-graviton products in a sector.

Single-trace graviton classes are the psu(1,2|3) descendants of Tr(phi^(I phi^J)) and Tr(phi^(I phi^J phi^K)) (higher
symmetrized traces are multi-traces at N=3).  The raising generators act letter by letter (projective vector fields of
C^{2|3}): P_al = d_{z_al} (even), Q^a = d/dtheta_a and Qt_{a,al} = theta_a d_{z_al} (odd), with the (-1)^|S| convention
of the odd derivations (fixed by {Q_0, J} = 0 on random multitraces, 0/150 failures; 130-148/150 without it).
Descendants keep the number of letters, so a multi-graviton with Y letters is a product of 2- and 3-letter traces.
Products of Q_0-closed traces are Q_0-closed, so every product is a cocycle; its Cartan image is the product of the
factors' images."""
from collections import defaultdict
from itertools import combinations, combinations_with_replacement
import numpy as np
try:
    from fortuity.full.words import canon_word, canon_multi
    from fortuity.full.letters import admissible, theta_sign, charges2
except ImportError:
    from fortuity.full.words import canon_word, canon_multi
    from fortuity.full.letters import admissible, theta_sign, charges2
from fortuity.bmn.modp import P
import build_cand as bc

THETA = (0, 1, 2)


def _P(al):
    def f(L):
        a, b, S = L
        return ((a + 1, (a + 1, b, S)) if al == 0 else (b + 1, (a, b + 1, S)))
    def op(multi):
        out = defaultdict(int)
        for i, w in enumerate(multi):
            for pos, L in enumerate(w):
                c, nl = f(L)
                s, cw = canon_word(w[:pos] + (nl,) + w[pos + 1:])
                if s:
                    sm, cm = canon_multi(multi[:i] + (cw,) + multi[i + 1:])
                    if sm:
                        out[cm] += c * s * sm
        return {m: c for m, c in out.items() if c}
    return op


def _Qtheta(a_th):
    def f(L):
        a, b, S = L
        if a_th in S:
            return None
        T = tuple(sorted(S + (a_th,)))
        return (theta_sign((a_th,), S) * (-1) ** len(S), (a, b, T)) if admissible(a, b, T) else None
    return bc._odd_derivation(f)


def _Qtilde(a_th, al):
    def f(L):
        a, b, S = L
        if a_th not in S:
            return None
        T = tuple(x for x in S if x != a_th)
        na, nb = (a + 1, b) if al == 0 else (a, b + 1)
        c = (na if al == 0 else nb) * theta_sign((a_th,), T) * (-1) ** len(S)
        return (c, (na, nb, T)) if admissible(na, nb, T) else None
    return bc._odd_derivation(f)


RAISING = ([('P%d' % al, _P(al)) for al in (0, 1)] + [('Q%d' % a, _Qtheta(a)) for a in THETA]
           + [('Qt%d%d' % (a, al), _Qtilde(a, al)) for a in THETA for al in (0, 1)])


def content_counts(multi):
    """(sum a, sum b, n_0, n_1, n_2, Y) of a multitrace."""
    A = B = 0
    n = [0, 0, 0]
    Y = 0
    for w in multi:
        for (a, b, S) in w:
            A += a; B += b; Y += 1
            for j in S:
                n[j] += 1
    return (A, B, n[0], n[1], n[2], Y)


def fits(cc, bound):
    """Can a descendant of a state with content counts cc still end inside `bound`?  The derivative counts A, B never
    decrease under the raising generators; a theta count n_a can still drop by one per remaining derivative (theta_a d_z),
    so it may exceed its bound by the remaining derivative budget."""
    A, B = cc[0], cc[1]
    if A > bound[0] or B > bound[1]:
        return False
    slack = (bound[0] - A) + (bound[1] - B)
    return all(cc[2 + j] <= bound[2 + j] + slack for j in range(3))


def inside(cc, bound):
    return all(x <= y for x, y in zip(cc[:5], bound[:5]))


def primaries():
    phi = [(0, 0, (a,)) for a in THETA]
    out = []
    for a, b in combinations_with_replacement(range(3), 2):
        s, w = canon_word((phi[a], phi[b]))
        if s:
            out.append({(w,): s})
    for a, b, c in combinations_with_replacement(range(3), 3):
        v = defaultdict(int)
        for order in ((a, b, c), (a, c, b)):
            s, w = canon_word(tuple(phi[x] for x in order))
            if s:
                v[(w,)] += s
        v = {m: x % P for m, x in v.items() if x % P}
        if v:
            out.append(v)
    return out


class SectorBasis:
    """Row-reduced basis of word vectors in one sector (dict word -> coeff mod p), for independence tests."""
    def __init__(self):
        self.rows, self.piv = [], []       # rows in reduced form keyed by pivot word

    def reduce(self, v):
        v = {m: c % P for m, c in v.items() if c % P}
        for r, pw in zip(self.rows, self.piv):
            c = v.get(pw, 0)
            if c:
                for m, x in r.items():
                    v[m] = (v.get(m, 0) - c * x) % P
                v = {m: x for m, x in v.items() if x}
        return v

    def add(self, v):
        v = self.reduce(v)
        if not v:
            return False
        pw = min(v)
        inv = pow(v[pw], P - 2, P)
        v = {m: x * inv % P for m, x in v.items()}
        self.rows.append(v)
        self.piv.append(pw)
        return True


def single_trace_states(bound, verbose=False):
    """{(A, B, n0, n1, n2, Y): [word vectors]} spanning the graviton descendants that fit in `bound`."""
    sectors = defaultdict(SectorBasis)
    states = defaultdict(list)
    queue = []
    for v in primaries():
        cc = content_counts(next(iter(v)))
        if fits(cc, bound) and sectors[cc].add(v):
            states[cc].append(v)
            queue.append(v)
    while queue:
        v = queue.pop()
        for name, op in RAISING:
            out = defaultdict(int)
            for m, c in v.items():
                for m2, c2 in op(m).items():
                    out[m2] = (out[m2] + c * c2) % P
            out = {m: c for m, c in out.items() if c}
            if not out:
                continue
            cc = content_counts(next(iter(out)))
            if not fits(cc, bound):
                continue
            if sectors[cc].add(out):
                states[cc].append(out)
                queue.append(out)
    if verbose:
        tot = sum(len(x) for x in states.values())
        print(f"single-trace graviton states: {tot} in {len(states)} sectors", flush=True)
    return states


if __name__ == '__main__':
    import sys, time
    t0 = time.time()
    for name, bound in (('T_4', (4, 4, 3, 0, 0)), ('T_2', (2, 2, 5, 2, 2))):
        st = single_trace_states(bound)
        by_y = defaultdict(int)
        for cc, vs in st.items():
            by_y[cc[5]] += len(vs)
        print(name, 'single-trace states by length:', dict(by_y), f'({time.time() - t0:.0f}s)', flush=True)


def products(target, shapes, verbose=False):
    """Multi-graviton products with content counts `target` (A, B, n0, n1, n2, Y): list of (factor vectors, product
    word vector).  Each multiset of basis states is used once (graded commutativity)."""
    from math import comb
    from collections import Counter
    st = single_trace_states(target[:5])
    by_len = defaultdict(list)
    for cc, vs in st.items():
        if not inside(cc, target):
            continue
        for v in vs:
            by_len[cc[5]].append((cc, v))
    out = []
    for shape in shapes:
        lens = Counter(shape)
        pools = {L: by_len[L] for L in lens}
        def rec(choice, L_list, start):
            if not L_list:
                cc = [sum(x[0][k] for x in choice) for k in range(5)]
                if cc == list(target[:5]):
                    out.append([x[1] for x in choice])
                return
            L = L_list[0]
            st0 = start if (len(L_list) > 1 and L_list[1] == L) or True else 0
            for i in range(start, len(pools[L])):
                x = pools[L][i]
                cc = [sum(y[0][k] for y in choice) + x[0][k] for k in range(5)]
                if any(a > b for a, b in zip(cc, target[:5])):
                    continue
                nxt = L_list[1:]
                rec(choice + [x], nxt, i if nxt and nxt[0] == L else 0)
        rec([], sorted(shape), 0)
    prods = []
    for fac in out:
        v = {(): 1}
        for f in fac:
            nv = defaultdict(int)
            for m1, c1 in v.items():
                for m2, c2 in f.items():
                    s, m = canon_multi(m1 + m2)
                    if s:
                        nv[m] = (nv[m] + s * c1 * c2) % P
            v = {m: c for m, c in nv.items() if c}
        if v:
            prods.append((fac, v))
    if verbose:
        print(f"{len(prods)} nonzero products (of {len(out)} factor combinations)", flush=True)
    return prods


def eps_word(multi):
    """Cartan image of a multitrace (species 1, 2 = diagonal entries 0, 1; entry 2 = -(0) - (1))."""
    import abelian_lw as A
    out = {(): 1}
    for w in multi:
        tr = defaultdict(int)
        for i in (0, 1, 2):
            terms = [((), 1)]
            for L in w:
                opts = [((i + 1,) + L, 1)] if i < 2 else [((1,) + L, -1), ((2,) + L, -1)]
                terms = [(t + (x,), c * cx) for t, c in terms for x, cx in opts]
            for t, c in terms:
                tr[t] += c
        new = defaultdict(int)
        for m1, c1 in out.items():
            for m2, c2 in tr.items():
                s, m = A.canon(m1 + m2)
                if s:
                    new[m] = (new[m] + s * c1 * c2) % P
        out = {m: c for m, c in new.items() if c}
    return out


def eps_vec(v):
    out = defaultdict(int)
    for m, c in v.items():
        for m2, c2 in eps_word(m).items():
            out[m2] = (out[m2] + c * c2) % P
    return {m: c for m, c in out.items() if c}
