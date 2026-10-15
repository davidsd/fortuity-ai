"""Multi-trace words over the full 1/16-BPS letters (a, b, S) and the Q action; sector enumeration.

Sectors are labelled by (2J1, 2J2, 2q1, 2q2, 2q3) (all integers).  Q maps (2J1, 2J2, q) -> (2J1-1, 2J2-1, q).
Level j = 2 sum q_j = sum over letters of (3 - |S|).  Y = number of letters.
"""
from itertools import permutations, combinations_with_replacement, combinations
from collections import defaultdict
from functools import lru_cache
from .letters import is_ferm, charges2, Q_letter, admissible, THETA


def word_parity(w):
    return sum(1 for L in w if is_ferm(L)) % 2


def canon_word(w):
    L = len(w)
    if L == 1:
        return 0, None
    best, best_sign = None, 0
    cur, sign, seen = tuple(w), 1, {}
    for _ in range(L):
        if cur in seen:
            if seen[cur] != sign:
                return 0, None
        else:
            seen[cur] = sign
        if best is None or cur < best:
            best, best_sign = cur, sign
        first, rest = cur[0], cur[1:]
        if is_ferm(first) and (sum(1 for x in rest if is_ferm(x)) % 2 == 1):
            sign = -sign
        cur = rest + (first,)
    return best_sign, best


def canon_multi(words):
    ws = list(words)
    odd = [w for w in ws if word_parity(w)]
    if len(set(odd)) < len(odd):
        return 0, None
    order = sorted(range(len(odd)), key=lambda i: odd[i])
    sign, visited = 1, [False] * len(order)
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
    if s:
        expr[m] += s * coeff
        if expr[m] == 0:
            del expr[m]


@lru_cache(maxsize=None)
def Q_word(w):
    out = defaultdict(int)
    nf_before = 0
    for pos, L in enumerate(w):
        sign = -1 if nf_before % 2 else 1
        for c, L1, L2 in Q_letter(L):
            nw = w[:pos] + (L1, L2) + w[pos + 1:]
            s, cw = canon_word(nw)
            if s:
                out[cw] += sign * c * s
        if is_ferm(L):
            nf_before += 1
    return tuple((c, cw) for cw, c in out.items() if c)


def Q_multi(multi):
    out = defaultdict(int)
    par_before = 0
    for i, w in enumerate(multi):
        sign = -1 if par_before % 2 else 1
        for c, cw in Q_word(w):
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


# ---------- sectors ----------

def sector_of(multi):
    tot = [0] * 5
    for w in multi:
        for L in w:
            for i, x in enumerate(charges2(L)):
                tot[i] += x
    return tuple(tot)


def level(sec):
    return sec[2] + sec[3] + sec[4]     # 2 sum q = j


_LEVEL_CONTENTS = {}


def _letter_types(j):
    """Admissible letters (a, b, S) with level 3(a + b) + 2|S| <= j, with their doubled charges and level."""
    out = []
    for a in range(j // 3 + 1):
        for b in range(j // 3 + 1 - a):
            for k in range(4):
                for S in combinations(THETA, k):
                    if admissible(a, b, S):
                        c = charges2((a, b, S))
                        lvl = sum(c[2:])
                        if 0 < lvl <= j:
                            out.append(((a, b, S), c, lvl))
    return out


def level_contents(j):
    """{sector (2J1, 2J2, 2q1, 2q2, 2q3): sorted list of letter contents} for every sector at level j = sum of 2q, by a
    knapsack over letter types (the full-letter SO/USp development code, 2026-10-08; promoted in T-048). Same contents, in the
    same (sorted) order, as `contents_of_sector_direct`, at a small fraction of the cost (tests/test_full_groups.py).
    Cached per level."""
    if j in _LEVEL_CONTENTS:
        return _LEVEL_CONTENTS[j]
    T = sorted(_letter_types(j), key=lambda t: -t[2])
    out = {}

    def rec(i, rem, J1, J2, q, cur):
        if rem == 0:
            out.setdefault((J1, J2) + q, []).append(tuple(sorted(cur)))
            return
        for k in range(i, len(T)):
            L, c, lvl = T[k]
            if lvl > rem:
                continue
            cur.append(L)
            rec(k, rem - lvl, J1 + c[0], J2 + c[1], (q[0] + c[2], q[1] + c[3], q[2] + c[4]), cur)
            cur.pop()
    rec(0, j, 0, 0, (0, 0, 0), [])
    out = {sec: sorted(set(cs)) for sec, cs in out.items()}
    _LEVEL_CONTENTS[j] = out
    return out


def contents_of_sector(sec, Ymax=None):
    """Letter contents (multisets of letters, as sorted tuples) with total charges sec = (2J1, 2J2, 2q1, 2q2, 2q3), at
    most Ymax letters. Sorted list, identical to `contents_of_sector_direct` (fast path through `level_contents`)."""
    sec = tuple(sec)
    j = sum(sec[2:])
    if j <= 0 or min(sec[2:]) < 0:
        return contents_of_sector_direct(sec, Ymax)
    return [c for c in level_contents(j).get(sec, []) if Ymax is None or len(c) <= Ymax]


def contents_of_sector_direct(sec, Ymax=None):
    """Letter contents (multisets of letters, as sorted tuples) with total charges sec = (2J1, 2J2, 2q1, 2q2, 2q3).
    Per letter: 2J1 = -1 + 2a + |S|, 2J2 = -1 + 2b + |S|, 2q_j = a + b + 2[j in S]."""
    J1, J2, tq = sec[0], sec[1], sec[2:]
    j = sum(tq)
    out = []
    types = [()] + [(i,) for i in THETA] + [(0, 1), (0, 2), (1, 2)] + [(0, 1, 2)]
    Ymax = Ymax if Ymax is not None else j
    for Y in range(1, Ymax + 1):
        for counts in _compositions(Y, len(types)):
            sumS = sum(c * len(t) for c, t in zip(counts, types))
            nj = [sum(c for c, t in zip(counts, types) if jj in t) for jj in THETA]
            D = [tq[jj] - 2 * nj[jj] for jj in THETA]          # total derivatives, must agree for all j
            if D[0] != D[1] or D[0] != D[2] or D[0] < 0:
                continue
            two_a = J1 + Y - sumS
            two_b = J2 + Y - sumS
            if two_a < 0 or two_b < 0 or two_a % 2 or two_b % 2 or (two_a + two_b) // 2 != D[0]:
                continue
            A, B = two_a // 2, two_b // 2
            base = []
            for c, t in zip(counts, types):
                base += [t] * c
            for da in _distribute(A, Y):
                for db in _distribute(B, Y):
                    letters = tuple(sorted((da[i], db[i], base[i]) for i in range(Y)))
                    if all(admissible(*L) for L in letters):
                        out.append(letters)
    return sorted(set(out))


def sectors_at_level(j):
    """All sectors (2J1, 2J2, 2q1, 2q2, 2q3) with 2 sum q = j that admit at least one letter content."""
    secs = set()
    types = [()] + [(i,) for i in THETA] + [(0, 1), (0, 2), (1, 2)] + [(0, 1, 2)]
    for Y in range(1, j + 1):
        for counts in _compositions(Y, len(types)):
            sumS = sum(c * len(t) for c, t in zip(counts, types))
            if 2 * sumS > j or (j - 2 * sumS) % 3:
                continue
            D = (j - 2 * sumS) // 3
            nlam = counts[0]
            if D < nlam:
                continue
            nj = [sum(c for c, t in zip(counts, types) if jj in t) for jj in THETA]
            tq = tuple(D + 2 * nj[jj] for jj in THETA)
            for A in range(D + 1):
                B = D - A
                secs.add((-Y + 2 * A + sumS, -Y + 2 * B + sumS) + tq)
    return sorted(secs)


def _compositions(n, k):
    if k == 1:
        yield (n,)
        return
    for i in range(n + 1):
        for rest in _compositions(n - i, k - 1):
            yield (i,) + rest


@lru_cache(maxsize=None)
def _distribute(total, slots):
    return tuple(_compositions(total, slots)) if slots > 0 else ((),)


@lru_cache(maxsize=None)
def necklaces_with_letters(letters):
    """Nonzero canonical necklaces using exactly the multiset 'letters' (sorted tuple)."""
    if len(letters) < 2:
        return ()
    seen, out = set(), []
    for perm in set(permutations(letters)):
        s, cw = canon_word(perm)
        if s and cw not in seen:
            seen.add(cw)
            out.append(cw)
    return tuple(sorted(out))


def multitraces_with_letters(letters, maxlen):
    """All multi-trace words (products of necklaces of length 2..maxlen) using exactly the multiset 'letters'."""
    from collections import Counter
    cnt = Counter(letters)
    keys = sorted(cnt)
    # all sub-multisets of size 2..maxlen
    necks = []
    def subs(i, remaining, chosen):
        if i == len(keys):
            if 2 <= len(chosen) <= maxlen:
                yield tuple(chosen)
            return
        for c in range(remaining[keys[i]] + 1):
            yield from subs(i + 1, remaining, chosen + [keys[i]] * c)
    for sub in subs(0, cnt, []):
        necks += list(necklaces_with_letters(sub))
    necks = sorted(set(necks))
    ncnt = [Counter(w) for w in necks]
    npar = [word_parity(w) for w in necks]
    out = []
    def rec(start, remaining, chosen):
        if sum(remaining.values()) == 0:
            out.append(tuple(chosen))
            return
        for i in range(start, len(necks)):
            if all(ncnt[i][k] <= remaining[k] for k in ncnt[i]):
                rem = remaining.copy()
                for k, v in ncnt[i].items():
                    rem[k] -= v
                rec(i + 1 if npar[i] else i, rem, chosen + [necks[i]])
    rec(0, cnt, [])
    return out


def words_of_sector(sec, maxlen, Ymax=None):
    out = []
    for c in contents_of_sector(sec, Ymax):
        out += multitraces_with_letters(c, maxlen)
    return out
