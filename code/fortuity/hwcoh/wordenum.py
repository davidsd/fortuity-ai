"""Fast enumeration of BMN multitrace words by sector, grouped by shape.

Replaces fortuity.bmn.words.necklaces_with_content (which iterates over all permutations) by a fixed-content
necklace generator (FKM recursion restricted to a multiset).  Conventions, canonical forms and signs are those of
fortuity.bmn.words: a canonical necklace is the lexicographically smallest rotation; it vanishes iff its rotation
symmetry acts with a fermionic sign -1; a multitrace is a sorted tuple of canonical necklaces with odd necklaces
appearing at most once.
"""
from functools import lru_cache
from itertools import product
from fortuity.bmn.words import NLET, is_ferm, contents_of_sector, content, word_parity

FERM = tuple(is_ferm(l) for l in range(NLET))


def necklaces_fixed_content(c):
    """All nonvanishing canonical necklaces (tuples) with letter content c (length >= 2)."""
    n = sum(c)
    if n < 2:
        return []
    cnt = list(c)
    first = next(l for l in range(NLET) if cnt[l])
    a = [0] * (n + 1)
    a[1] = first
    cnt[first] -= 1
    out = []
    ferm = FERM

    def gen(t, p):
        if t > n:
            if n % p == 0:
                w = tuple(a[1:])
                # rotation by p maps w to itself with sign (-1)^{f_p (f - f_p)}
                fp = sum(1 for l in w[:p] if ferm[l])
                ft = fp * (n // p)
                if (fp * (ft - fp)) % 2 == 0:
                    out.append(w)
            return
        lo = a[t - p]
        for j in range(lo, NLET):
            if cnt[j]:
                a[t] = j
                cnt[j] -= 1
                gen(t + 1, p if j == lo else t)
                cnt[j] += 1

    gen(2, 1)
    return out


@lru_cache(maxsize=None)
def necklaces_with_content(c):
    return tuple(necklaces_fixed_content(c))


def multitraces_with_content(c, maxlen=None):
    """All multitrace words with total content c (necklaces of length >= 2, odd necklaces at most once).
    Enumerates vector partitions of c into necklace contents, then multisets of necklaces per part."""
    import numpy as np
    from itertools import combinations, combinations_with_replacement
    subs = [sub for sub in product(*[range(x + 1) for x in c])
            if 2 <= sum(sub) <= (maxlen or 10**9) and necklaces_with_content(sub)]
    subs.sort(reverse=True)
    S = np.array(subs, dtype=np.int64).reshape(-1, NLET)
    out = []

    def rec(rem, maxi, parts):
        if not rem.any():
            words_from_parts(parts, out)
            return
        ok = np.flatnonzero((S[:maxi + 1] <= rem).all(axis=1))
        for i in ok:
            parts.append(i)
            rec(rem - S[i], i, parts)
            parts.pop()

    def words_from_parts(parts, out):
        from collections import Counter
        choices = []
        for i, r in Counter(parts).items():
            necks = necklaces_with_content(subs[i])
            odd = word_parity(necks[0])       # all necklaces of one content have the same parity
            gen = combinations(necks, r) if odd else combinations_with_replacement(necks, r)
            choices.append(list(gen))
        for combo in product(*choices):
            out.append(tuple(sorted(n for grp in combo for n in grp)))

    rec(np.array(c, dtype=np.int64), len(subs) - 1, [])
    return out


def words_of_sector(sec, maxlen=None):
    """Multitraces of the sector; with maxlen, only traces of length <= maxlen (an SU(3)_R-invariant subset:
    callers must verify that it still spans, e.g. by the rank = m check)."""
    out = []
    for c in contents_of_sector(sec):
        out += multitraces_with_content(c, maxlen)
    return out


STRIP = (0, 0, 0, 1, 1, 1, 2)   # phi -> 0, psi -> 1, f -> 2


def _min_rot(t):
    return min(t[i:] + t[:i] for i in range(len(t)))


def shape(multi):
    """Flavour-stripped shape of a multitrace (invariant under SU(3)_R)."""
    return tuple(sorted(_min_rot(tuple(STRIP[l] for l in w)) for w in multi))
