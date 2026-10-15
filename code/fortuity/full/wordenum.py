"""Fast enumeration of full-letter multitrace words (FKM necklaces restricted to a multiset), same canonical
forms and signs as words_full (canonical necklace = lexicographically smallest rotation; vanishes iff a rotation
symmetry acts with sign -1; multitrace = sorted tuple, odd necklaces at most once)."""
from functools import lru_cache
from itertools import product, combinations, combinations_with_replacement
from collections import Counter
import numpy as np
from .letters import is_ferm
from .words import word_parity, contents_of_sector


def necklaces_fixed_content(letters):
    """letters: sorted tuple (multiset).  Returns all nonvanishing canonical necklaces with exactly this content."""
    n = len(letters)
    if n < 2:
        return []
    keys = sorted(set(letters))
    cnt = [letters.count(k) for k in keys]
    ferm = [is_ferm(k) for k in keys]
    K = len(keys)
    a = [0] * (n + 1)
    a[1] = 0
    cnt[0] -= 1
    out = []

    def gen(t, p):
        if t > n:
            if n % p == 0:
                w = a[1:]
                fp = sum(1 for l in w[:p] if ferm[l])
                ft = fp * (n // p)
                if (fp * (ft - fp)) % 2 == 0:
                    out.append(tuple(keys[l] for l in w))
            return
        lo = a[t - p]
        for j in range(lo, K):
            if cnt[j]:
                a[t] = j
                cnt[j] -= 1
                gen(t + 1, p if j == lo else t)
                cnt[j] += 1

    gen(2, 1)
    return out


@lru_cache(maxsize=None)
def necklaces_with_content(letters):
    return tuple(necklaces_fixed_content(letters))


def multitraces_with_content(letters, maxlen=None):
    """All multitrace words with exactly the letter multiset 'letters' (sorted tuple)."""
    cnt = Counter(letters)
    keys = sorted(cnt)
    c = tuple(cnt[k] for k in keys)
    K = len(keys)
    subs = []
    for sub in product(*[range(x + 1) for x in c]):
        if 2 <= sum(sub) <= (maxlen or 10**9):
            lets = tuple(k for k, r in zip(keys, sub) for _ in range(r))
            if necklaces_with_content(lets):
                subs.append(sub)
    subs.sort(reverse=True)
    if not subs:
        return []
    S = np.array(subs, dtype=np.int64).reshape(-1, K)
    out = []

    def words_from_parts(parts):
        choices = []
        for i, r in Counter(parts).items():
            lets = tuple(k for k, rr in zip(keys, subs[i]) for _ in range(rr))
            necks = necklaces_with_content(lets)
            odd = word_parity(necks[0])
            gen = combinations(necks, r) if odd else combinations_with_replacement(necks, r)
            choices.append(list(gen))
        for combo in product(*choices):
            out.append(tuple(sorted(n for grp in combo for n in grp)))

    def rec(rem, maxi, parts):
        if not rem.any():
            words_from_parts(parts)
            return
        ok = np.flatnonzero((S[:maxi + 1] <= rem).all(axis=1))
        for i in ok:
            parts.append(i)
            rec(rem - S[i], i, parts)
            parts.pop()

    rec(np.array(c, dtype=np.int64), len(subs) - 1, [])
    return out


def words_of_sector(sec, maxlen=None, Y=None):
    out = []
    for c in contents_of_sector(sec):
        if Y is None or len(c) == Y:
            out += multitraces_with_content(c, maxlen)
    return out


def _min_rot(t):
    return min(t[i:] + t[:i] for i in range(len(t)))


def shape(multi):
    """Flavour-stripped shape (a, b, |S|) per letter, invariant under SU(3)_R."""
    return tuple(sorted(_min_rot(tuple((L[0], L[1], len(L[2])) for L in w)) for w in multi))
