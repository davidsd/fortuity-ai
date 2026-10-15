"""Letter contents of a full-letter sector at FIXED letter number Y (same output as
words_full.contents_of_sector(sec) filtered to len == Y, but without enumerating all Y <= j).
prime_cache(sec, Y) writes cache/contents_<sec>_Y<Y>.pkl in the current directory, which is where
sector_full.contents_cached looks, so the fullsector code then runs on this workspace's cache."""
import os, pickle
from fortuity.full.words import _compositions, _distribute
from fortuity.full.letters import admissible, THETA
from fortuity.full.dims import weyl_orbit_signed

TYPES = [()] + [(i,) for i in THETA] + [(0, 1), (0, 2), (1, 2)] + [(0, 1, 2)]


def contents_at(sec, Y):
    J1, J2, tq = sec[0], sec[1], sec[2:]
    out = []
    for counts in _compositions(Y, len(TYPES)):
        sumS = sum(c * len(t) for c, t in zip(counts, TYPES))
        nj = [sum(c for c, t in zip(counts, TYPES) if jj in t) for jj in THETA]
        D = [tq[jj] - 2 * nj[jj] for jj in THETA]
        if D[0] != D[1] or D[0] != D[2] or D[0] < 0:
            continue
        two_a, two_b = J1 + Y - sumS, J2 + Y - sumS
        if two_a < 0 or two_b < 0 or two_a % 2 or two_b % 2 or (two_a + two_b) // 2 != D[0]:
            continue
        A, B = two_a // 2, two_b // 2
        base = []
        for c, t in zip(counts, TYPES):
            base += [t] * c
        for da in _distribute(A, Y):
            for db in _distribute(B, Y):
                letters = tuple(sorted((da[i], db[i], base[i]) for i in range(Y)))
                if all(admissible(*L) for L in letters):
                    out.append(letters)
    return sorted(set(out))


def prime_cache(sec, Y, cache='cache'):
    """Write the contents cache for sec and for every dominant weight in its Weyl-alternation orbit."""
    os.makedirs(cache, exist_ok=True)
    secs = [tuple(sec)] + [tuple(sec[:2]) + tuple(w) for _, w in weyl_orbit_signed(sec[2:]) if min(w) >= 0]
    for s in secs:
        fn = os.path.join(cache, f'contents_{"_".join(map(str, s))}_Y{Y}.pkl')
        if not os.path.exists(fn):
            pickle.dump(contents_at(s, Y), open(fn, 'wb'))
