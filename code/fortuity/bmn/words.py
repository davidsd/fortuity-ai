"""Multi-trace words in the BMN letters (phi_1,phi_2,phi_3, psi_1,psi_2,psi_3, f) and the Q action.

Letters: 0,1,2 = phi_m ; 3,4,5 = psi_m ; 6 = f.
Q phi_m = 0,  Q psi_m = eps_{mnp} phi_n phi_p,  Q f = sum_m [phi_m, psi_m].
(overall normalisation of Q irrelevant for cohomology.)

An "expression" is a dict {multitrace: coeff}, multitrace = tuple of canonical necklaces (tuples of letters).
Charges: q_I = n_phi_I + (n_psi - n_psi_I) + n_f,  2J = n_psi + 2 n_f,  j = 2 sum q_I.
"""
from itertools import permutations, product
from collections import defaultdict
from functools import lru_cache

PHI = (0, 1, 2)
PSI = (3, 4, 5)
F = 6
NLET = 7
NAMES = ['X', 'Y', 'Z', 'p1', 'p2', 'p3', 'f']


def is_ferm(l):
    return 3 <= l <= 5


def word_parity(w):
    return sum(1 for l in w if is_ferm(l)) % 2


def nferm(w):
    return sum(1 for l in w if is_ferm(l))


# ---------- canonical forms ----------

def canon_word(w):
    """Canonical rotation of a cyclic word with the fermionic sign.  Returns (sign, word) or (0, None)."""
    L = len(w)
    if L == 1:
        return 0, None  # traceless
    best = None
    best_sign = 0
    cur = tuple(w)
    sign = 1
    seen = {}
    for r in range(L):
        if cur in seen:
            if seen[cur] != sign:
                return 0, None
        else:
            seen[cur] = sign
        if best is None or cur < best:
            best, best_sign = cur, sign
        # rotate: move first letter to the end; sign from passing it through the remaining fermions
        first = cur[0]
        rest = cur[1:]
        if is_ferm(first) and (nferm(rest) % 2 == 1):
            sign = -sign
        cur = rest + (first,)
    # check consistency: the rotation that returns to the start must give sign +1 (handled by 'seen')
    return best_sign, best


def canon_multi(words):
    """Sort factors; sign from permuting odd factors; zero if a repeated odd factor."""
    ws = list(words)
    odd_idx = [i for i, w in enumerate(ws) if word_parity(w)]
    odd_words = [ws[i] for i in odd_idx]
    if len(set(odd_words)) < len(odd_words):
        return 0, None
    # sign = parity of the permutation sorting the odd words
    order = sorted(range(len(odd_words)), key=lambda i: odd_words[i])
    sign = 1
    visited = [False] * len(order)
    for i in range(len(order)):
        if visited[i]:
            continue
        j, cyc = i, 0
        while not visited[j]:
            visited[j] = True
            j = order[j]
            cyc += 1
        if cyc % 2 == 0:
            sign = -sign
    return sign, tuple(sorted(ws))


def add_term(expr, multi, coeff):
    if coeff == 0:
        return
    s, m = canon_multi(multi)
    if s == 0:
        return
    expr[m] += s * coeff
    if expr[m] == 0:
        del expr[m]


# ---------- Q action ----------

QLET = {
    3: [(1, (1, 2)), (-1, (2, 1))],
    4: [(1, (2, 0)), (-1, (0, 2))],
    5: [(1, (0, 1)), (-1, (1, 0))],
    6: [(1, (0, 3)), (-1, (3, 0)), (1, (1, 4)), (-1, (4, 1)), (1, (2, 5)), (-1, (5, 2))],
}


def Q_word(w):
    """Q acting on a single trace word -> list of (coeff, canonical word)."""
    out = defaultdict(int)
    nf_before = 0
    for pos, l in enumerate(w):
        if l in QLET:
            sign = -1 if nf_before % 2 else 1
            for c, rep in QLET[l]:
                nw = w[:pos] + rep + w[pos + 1:]
                s, cw = canon_word(nw)
                if s:
                    out[cw] += sign * c * s
        if is_ferm(l):
            nf_before += 1
    return [(c, cw) for cw, c in out.items() if c]


_Qword_cache = {}


def Q_multi(multi):
    """Q on a multitrace -> expression dict."""
    out = defaultdict(int)
    par_before = 0
    for i, w in enumerate(multi):
        if w not in _Qword_cache:
            _Qword_cache[w] = Q_word(w)
        sign = -1 if par_before % 2 else 1
        for c, cw in _Qword_cache[w]:
            add_term(out, multi[:i] + (cw,) + multi[i + 1:], sign * c)
        par_before += word_parity(w)
    return out


def Q_expr(expr):
    out = defaultdict(int)
    for multi, c in expr.items():
        for m2, c2 in Q_multi(multi).items():
            out[m2] += c * c2
            if out[m2] == 0:
                del out[m2]
    return out


def mul_expr(A, B):
    out = defaultdict(int)
    for ma, ca in A.items():
        for mb, cb in B.items():
            add_term(out, ma + mb, ca * cb)
    return out


# ---------- contents and sectors ----------

def content(multi):
    c = [0] * NLET
    for w in multi:
        for l in w:
            c[l] += 1
    return tuple(c)


def charges_of_content(c):
    npsi = c[3] + c[4] + c[5]
    nf = c[6]
    q = tuple(c[I] + (npsi - c[3 + I]) + nf for I in range(3))
    return (npsi + 2 * nf,) + q  # (2J, q1, q2, q3)


def sector_j(sec):
    return 2 * (sec[1] + sec[2] + sec[3])


def contents_of_sector(sec):
    """All letter contents (n_phi1..n_f) with the given (2J, q1, q2, q3)."""
    twoJ, q = sec[0], sec[1:]
    out = []
    for nf in range(twoJ // 2 + 1):
        npsi = twoJ - 2 * nf
        for a in range(npsi + 1):
            for b in range(npsi - a + 1):
                npsi_m = (a, b, npsi - a - b)
                nphi = [q[I] - (npsi - npsi_m[I]) - nf for I in range(3)]
                if min(nphi) < 0:
                    continue
                out.append(tuple(nphi) + npsi_m + (nf,))
    return out


@lru_cache(maxsize=None)
def necklaces_with_content(c):
    """All nonzero canonical necklaces with exactly the letter content c (tuple of 7 counts)."""
    letters = []
    for l, n in enumerate(c):
        letters += [l] * n
    L = len(letters)
    if L < 2:
        return ()
    seen = set()
    out = []
    # generate distinct arrangements
    for perm in set(permutations(letters)):
        s, cw = canon_word(perm)
        if s and cw not in seen:
            seen.add(cw)
            out.append(cw)
    return tuple(sorted(out))


def sub_contents(c, minlen, maxlen):
    ranges = [range(n + 1) for n in c]
    for sub in product(*ranges):
        L = sum(sub)
        if minlen <= L <= maxlen:
            yield sub


def multitraces_with_content(c, maxlen):
    """All multi-trace words (products of necklaces of length 2..maxlen) with total content c."""
    necks = []
    for sub in sub_contents(c, 2, maxlen):
        for w in necklaces_with_content(sub):
            necks.append(w)
    necks.sort()
    ncont = [content((w,)) for w in necks]
    npar = [word_parity(w) for w in necks]
    out = []

    def rec(start, remaining, chosen):
        if sum(remaining) == 0:
            out.append(tuple(chosen))
            return
        for i in range(start, len(necks)):
            cc = ncont[i]
            if all(cc[l] <= remaining[l] for l in range(NLET)):
                rem = tuple(remaining[l] - cc[l] for l in range(NLET))
                nxt = i + 1 if npar[i] else i  # odd necklaces at most once
                rec(nxt, rem, chosen + [necks[i]])

    rec(0, tuple(c), [])
    return out


def words_of_sector(sec, maxlen):
    out = []
    for c in contents_of_sector(sec):
        out += multitraces_with_content(c, maxlen)
    return out


# ---------- single-trace graviton generators (BMN sector) ----------

def _mperms(seq):
    """Distinct permutations of seq with multiplicity: sum over permutations(seq) of g(perm) equals
    sum over (perm, mult) in _mperms(seq) of mult * g(perm).  Avoids the n! blow-up for repeated entries."""
    from math import factorial
    counts = {}
    for x in seq:
        counts[x] = counts.get(x, 0) + 1
    mult = 1
    for v in counts.values():
        mult *= factorial(v)
    keys = sorted(counts, key=repr)
    n = len(seq)
    cur = []

    def rec():
        if len(cur) == n:
            yield tuple(cur)
            return
        for k in keys:
            if counts[k]:
                counts[k] -= 1
                cur.append(k)
                yield from rec()
                cur.pop()
                counts[k] += 1

    for perm in rec():
        yield perm, mult


def sym_trace_expr(letters_list):
    """Symmetrised trace expression: average over orderings? We use the plain sum over distinct
    arrangements (overall normalisation irrelevant).  letters_list: list of letters."""
    out = defaultdict(int)
    for perm, mult in _mperms(letters_list):
        add_term(out, (perm,), mult)
    return out


def u_gen(idx):
    """u^{i1..in} = tr(phi_(i1 ... phi_in)), idx: tuple of flavour indices 0..2."""
    return sym_trace_expr([PHI[i] for i in idx])


def T_v(idx, j):
    """tr(phi_(i1..i_{n-1}) psi_j) (symmetrised over phi's only)."""
    out = defaultdict(int)
    for perm, mult in _mperms([PHI[i] for i in idx]):
        add_term(out, (perm + (PSI[j],),), mult)
    return out


def v_gen(idx, j):
    """v^{i1..i_{n-1}}_j = T - 1/(n+1) sum_p delta^{i_p}_j T^{(idx minus i_p) a}_a.
    We multiply by (n+1) to keep integer coefficients."""
    n = len(idx) + 1
    out = defaultdict(int)
    for m, c in T_v(idx, j).items():
        out[m] += (n + 1) * c
    for pth, ip in enumerate(idx):
        if ip != j:
            continue
        rest = idx[:pth] + idx[pth + 1:]
        for a in range(3):
            for m, c in T_v(rest + (a,), a).items():
                out[m] -= c
    return {m: c for m, c in out.items() if c}


def w_gen(idx, psipsi_coeff):
    """w^{i1..i_{n-1}} = tr(f phi_(i1..i_{n-1})) + (coeff/2) eps^{jk(i_p} sum_p tr(phi.. psi_j(at p) .. phi psi_k).
    Both terms summed over all permutations of the upper indices (factor 2 absorbs the 1/2)."""
    out = defaultdict(int)
    phis = [PHI[i] for i in idx]
    for perm, mult in _mperms(phis):
        add_term(out, ((F,) + perm,), 2 * mult)
    n1 = len(idx)
    for perm, mult in _mperms(idx):
        for pos in range(n1):
            ip = perm[pos]
            others = [PHI[i] for i in perm[:pos] + perm[pos + 1:]]
            for jj in range(3):
                for kk in range(3):
                    eps = _eps(jj, kk, ip)
                    if eps == 0:
                        continue
                    w = tuple(others[:pos]) + (PSI[jj],) + tuple(others[pos:]) + (PSI[kk],)
                    add_term(out, (w,), psipsi_coeff * eps * mult)
    return {m: c for m, c in out.items() if c}


def _eps(a, b, c):
    if len({a, b, c}) < 3:
        return 0
    return 1 if (a, b, c) in ((0, 1, 2), (1, 2, 0), (2, 0, 1)) else -1


def sym_indices(n):
    """Multisets of n flavour indices (as sorted tuples)."""
    from itertools import combinations_with_replacement
    return list(combinations_with_replacement(range(3), n))


def graviton_generators(nmax, psipsi_coeff, qmax=None):
    """List of (name, expr) for u_n, v_n, w_n components, n = 2..nmax.  Only a spanning set of each rep
    (we include all index combinations; linear dependence is harmless).
    qmax: optional sector bound (2J, q1, q2, q3); generators whose sector exceeds it in any component are
    skipped (they cannot appear in products landing in sectors <= qmax)."""
    def fits(tj, q):
        return qmax is None or (tj <= qmax[0] and all(a <= b for a, b in zip(q, qmax[1:])))

    def cnt(idx):
        return [sum(1 for i in idx if i == a) for a in range(3)]

    gens = []
    for n in range(2, nmax + 1):
        for idx in sym_indices(n):
            if fits(0, cnt(idx)):
                gens.append((('u', n, idx), u_gen(idx)))
        for idx in sym_indices(n - 1):
            for j in range(3):
                q = [c + (0 if a == j else 1) for a, c in enumerate(cnt(idx))]
                if not fits(1, q):
                    continue
                e = v_gen(idx, j)
                if e:
                    gens.append((('v', n, idx, j), e))
        for idx in sym_indices(n - 1):
            if not fits(2, [c + 1 for c in cnt(idx)]):
                continue
            e = w_gen(idx, psipsi_coeff)
            if e:
                gens.append((('w', n, idx), e))
    return gens


def expr_sector(expr):
    secs = {charges_of_content(content(m)) for m in expr}
    assert len(secs) == 1, secs
    return secs.pop()


# ---------- SU(1|3) supercharges Q^m_+ (linear on letters) ----------
# Q^m_+ phi_n = eps^{mnp} psi_p,  Q^m_+ psi_n = delta^m_n f,  Q^m_+ f = 0   (fermionic operator)

def Qplus_word(w, m):
    out = defaultdict(int)
    nf_before = 0
    for pos, l in enumerate(w):
        sign = -1 if nf_before % 2 else 1
        reps = []
        if l in PHI:
            n = l
            for pp in range(3):
                e = _eps(m, n, pp)
                if e:
                    reps.append((e, PSI[pp]))
        elif l in PSI and l - 3 == m:
            reps.append((1, F))
        for c, rep in reps:
            nw = w[:pos] + (rep,) + w[pos + 1:]
            s, cw = canon_word(nw)
            if s:
                out[cw] += sign * c * s
        if is_ferm(l):
            nf_before += 1
    return [(c, cw) for cw, c in out.items() if c]


def Qplus_expr(expr, m):
    out = defaultdict(int)
    for multi, c in expr.items():
        par_before = 0
        for i, w in enumerate(multi):
            sign = -1 if par_before % 2 else 1
            for c2, cw in Qplus_word(w, m):
                add_term(out, multi[:i] + (cw,) + multi[i + 1:], c * sign * c2)
            par_before += word_parity(w)
    return out
