"""SU(3) Weyl group (permutations of the theta labels) acting on full-letter words, as an automorphism of the Q complex,
so that single-trace generators at non-dominant weights are Weyl images of generators at dominant weights.
A letter (a, b, S) maps to sign * (a, b, sigma(S)) with sign the parity of sorting the relabelled thetas; on a word the
letter signs multiply (each letter is an independent Taylor coefficient).  Checked: Q(sigma w) = sigma(Q w) on random words
(code/tests/test_full_groups.py); no extra global sign is needed for odd permutations. full-letter SO/USp development code,
workspace 2026-10-09; promoted in T-048.
"""
from itertools import permutations
from collections import defaultdict
from .words import canon_word, canon_multi


PERMS = list(permutations(range(3)))


def _perm_sign(seq):
    inv = sum(1 for i in range(len(seq)) for j in range(i + 1, len(seq)) if seq[i] > seq[j])
    return -1 if inv % 2 else 1


def weyl_letter(L, perm):
    a, b, S = L
    img = [perm[x] for x in S]           # thetas relabelled, in the original order
    return _perm_sign(img), (a, b, tuple(sorted(img)))


def weyl_word(w, perm):
    sign = 1
    nw = []
    for L in w:
        s, L2 = weyl_letter(L, perm)
        sign *= s
        nw.append(L2)
    s, cw = canon_word(tuple(nw))
    return sign * s, cw


def weyl_multi(multi, perm):
    sign = 1
    ws = []
    for w in multi:
        s, cw = weyl_word(w, perm)
        if s == 0:
            return 0, None
        sign *= s
        ws.append(cw)
    s, cm = canon_multi(ws)
    return sign * s, cm


def weyl_expr(expr, perm):
    out = defaultdict(int)
    for m, c in expr.items():
        s, cm = weyl_multi(m, perm)
        if s:
            out[cm] += s * c
    return {m: c for m, c in out.items() if c}


def weyl_sector(sec, perm):
    q = sec[2:]
    nq = [0, 0, 0]
    for i in range(3):
        nq[perm[i]] = q[i]
    return (sec[0], sec[1]) + tuple(nq)
