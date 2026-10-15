"""T-019 (SO/Sp part): {Q_0, Q_1} = 0 for so(2N+1), checked numerically on random derivative-free words.
Q_1 is Budzik-Kulp's (q1_full.py, with the so(n) trace-rejoining term added); Q_0 is group independent.
Run: cd scripts/oneloop/so-odd-classes && PYTHONPATH=../../../code python test_q1_so.py"""
import random
import numpy as np
from collections import defaultdict
from fortuity.full.letters import is_ferm
from fortuity.full.words import canon_multi, canon_word, Q_expr, add_term
from fortuity.full.q1 import Q1_expr
from fortuity.full.evalg import Frame, nferm

PHI = [(0, 0, (i,)) for i in range(3)]
PSI = [(0, 0, tuple(x for x in range(3) if x != k)) for k in range(3)]
F = (0, 0, (0, 1, 2))


def random_multi(rng, nletters):
    pool = PHI * 3 + PSI + [F]
    lets = [rng.choice(pool) for _ in range(nletters)]
    cut = sorted(rng.sample(range(1, nletters), rng.randint(0, max(0, nletters // 3 - 1)))) if nletters > 3 else []
    words, prev = [], 0
    for c in cut + [nletters]:
        words.append(tuple(lets[prev:c]))
        prev = c
    if any(len(w) < 2 for w in words):
        return None
    cws, sign = [], 1
    for w in words:
        s, cw = canon_word(w)
        if not s:
            return None
        sign *= s
        cws.append(cw)
    s, m = canon_multi(cws)
    return (m, sign * s) if s else None


def values(expr, N, group, seed):
    if not expr:
        return None
    words = list(expr)
    letters = sorted({L for m in words for w in m for L in w})
    degs = sorted({nferm(m) for m in words})
    fr = Frame(N, 6, max(degs) + 2, degs, letters, seed=seed, group=group)
    M = fr.eval_words(words, dtype=np.int64)
    c = np.array([expr[m] % fr.p for m in words], dtype=np.int64)
    return (c @ (M % fr.p)) % fr.p


def anticomm_zero(m, N, group, n_matrix):
    e = {m: 1}
    t1 = Q_expr(Q1_expr(e, n_matrix, group))
    t2 = Q1_expr(Q_expr(e), n_matrix, group)
    tot = defaultdict(int)
    for x in (t1, t2):
        for k, v in x.items():
            tot[k] += v
    tot = {k: v for k, v in tot.items() if v}
    if not tot:
        return True, 0
    v = values(tot, N, group, seed=5)
    return (v is None or not v.any()), len(tot)


if __name__ == "__main__":
    rng = random.Random(1)
    for group, N, n in (('SO_odd', 3, 7), ('SO_odd', 2, 5), ('SU', 3, 3)):
        ok, tried = True, 0
        while tried < 40:
            r = random_multi(rng, rng.randint(3, 6))
            if r is None:
                continue
            m, _ = r
            z, nt = anticomm_zero(m, N, group, n)
            ok &= z
            tried += 1
            if not z:
                print("FAIL", group, N, m)
        print(f"{group} N={N}: {{Q0,Q1}} w = 0 on {tried} random words: {ok}")
