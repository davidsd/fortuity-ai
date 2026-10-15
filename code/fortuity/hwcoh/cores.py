"""Genuine SU(N) black-hole cores in a complete SU(3)_R highest-weight column (T-032).

Block restriction (R-016, fortuity.bmn.blockrestrict): SU(N) > SU(N-1) x U(1) sends a class of the sector (2J; lam)
to sum_alpha x^alpha (x) (SU(N-1) operators in (2J; lam - alpha)), where x = (x_1, x_2, x_3) are the U(1) components of
the scalars (SU(3)_R irrep [1,0], 2J = 0).  The map is SU(3)_R-equivariant and sends exact to exact and gravitons to
gravitons.  A non-graviton class is a CORE if all its images are graviton or exact; otherwise it is fused.  The images
of the non-graviton classes of (2J; lam) lie in the lam-isotypic part of
    sum_n  Sym^n[1,0] (x) (SU(N-1) non-graviton classes at level |lam| - n, same 2J),
whose multiplicity is the fused capacity C(2J).  Hence, per 2J,  #cores >= BH - C.

Column version, independent of the graviton counts: the C-twisted index (R-017, fortuity.bmn.twistedindex) fixes the
number of P-odd classes, (H_total - (-1)^|lam| I_C) / 2, and every P-odd class is non-graviton (gravitons are P-even),
so  #cores >= #P-odd - sum_2J C(2J).

Inputs are fortuity.hwcoh.run_sectors records (one dict per sector with m, r_up, grav, check), of the column itself
and of complete SU(N-1) columns (nongraviton_content).  Over F_p these records are exact, since every sector's
self-check certifies its functionals; the SU(N-1) content can only be overcounted by a bad prime, which only raises
the capacity (the safe direction).
"""
from collections import Counter
from itertools import combinations_with_replacement

import numpy as np

from fortuity.bmn.words import contents_of_sector

_A12, _A23 = np.array([1, -1, 0]), np.array([0, 1, -1])
_WEYL = [(0, 0, 1), (1, 0, -1), (0, 1, -1), (2, 1, 1), (1, 2, 1), (2, 2, -1)]


def gl3_weights(lam):
    """Weight multiplicities of the GL(3) irrep with highest weight lam (semistandard tableaux)."""
    out = Counter()

    def rows(length, above):
        res = []

        def rec(i, prev, cur):
            if i == length:
                res.append(tuple(cur))
                return
            for v in range(max(prev, above[i] + 1 if i < len(above) else 1), 4):
                cur.append(v)
                rec(i + 1, v, cur)
                cur.pop()
        rec(0, 1, [])
        return res
    l1, l2, l3 = lam
    for r1 in rows(l1, [0] * l1):
        for r2 in rows(l2, list(r1[:l2])):
            for r3 in rows(l3, list(r2[:l3])):
                c = [0, 0, 0]
                for v in r1 + r2 + r3:
                    c[v - 1] += 1
                out[tuple(c)] += 1
    return out


def hw_multiplicity(wts, lam):
    """Multiplicity of the irrep lam in a weight system (Weyl alternation over the six terms)."""
    lam = np.array(lam)
    return int(sum(sg * wts.get(tuple(lam + i * _A12 + j * _A23), 0) for i, j, sg in _WEYL))


def capacity(lam, content):
    """Fused capacity per 2J: multiplicity of lam in sum_n Sym^n[1,0] (x) (content at level |lam| - n, same 2J).
    content: iterable of (level, 2J, highest weight) of SU(N-1) non-graviton classes, with repetition.  Returns
    (Counter {2J: C}, list of (level, 2J, hw, n, contribution) for the contributing entries)."""
    cap, src = Counter(), []
    for level, twoJ, mu in content:
        n = sum(lam) - level
        if n < 0:
            continue
        w = Counter()
        mw = gl3_weights(mu)
        for alpha in combinations_with_replacement(range(3), n):
            a = [0, 0, 0]
            for i in alpha:
                a[i] += 1
            for wt, c in mw.items():
                w[(wt[0] + a[0], wt[1] + a[1], wt[2] + a[2])] += c
        k = hw_multiplicity(w, lam)
        if k:
            cap[twoJ] += k
            src.append((level, twoJ, tuple(mu), n, k))
    return cap, src


def column_table(records, lam):
    """{2J: (m, h, grav, BH)} for a complete column from run_sectors records ({2J: record}); raises if a sector is
    missing or failed its self-check."""
    lam = tuple(lam)
    allowed = [n for n in range(0, sum(lam) + 1) if contents_of_sector((n,) + lam)]
    missing = [n for n in allowed if n not in records]
    bad = [n for n in allowed if n in records and not records[n].get('check')]
    if missing or bad:
        raise ValueError(f'column {lam}: missing 2J {missing}, failed self-check {bad}')
    out = {}
    for n in allowed:
        r = records[n]
        h = r['m'] - r['r_up'] - (records[n - 1]['r_up'] if n > 0 else 0)
        out[n] = (r['m'], h, r['grav'], h - r['grav'])
    return out


def nongraviton_content(records_by_column):
    """(level, 2J, lam) of the non-graviton classes, with multiplicity, from complete columns {lam: {2J: record}}
    (incomplete columns are skipped)."""
    out = []
    for lam, recs in records_by_column.items():
        try:
            tab = column_table(recs, lam)
        except ValueError:
            continue
        for n, (_, _, _, bh) in tab.items():
            out += [(sum(lam), n, tuple(lam))] * bh
    return out


_TWIST = {}


def twisted_and_plain(N, lam):
    """((-1)^|lam| I_C(lam), I(lam)): SU(3)_R multiplicities of lam in the C-twisted and in the plain BMN index of SU(N)."""
    from fortuity.bmn.twistedindex import SUN, integrate, to_int, hw_multiplicities
    Q = sum(lam)
    if (N, Q) not in _TWIST:
        tw, e1 = to_int(integrate(SUN(N), Q, True)[0])
        pl, e2 = to_int(integrate(SUN(N), Q, False)[0])
        assert max(e1, e2) < 1e-6, (e1, e2)
        _TWIST[(N, Q)] = (hw_multiplicities(tw, Q), hw_multiplicities(pl, Q))
    tw, pl = _TWIST[(N, Q)]
    return (-1) ** Q * tw.get(tuple(lam), 0), pl.get(tuple(lam), 0)


def core_test(N, lam, records, content):
    """Per-2J table, Euler check, P-odd count and core lower bounds for a complete column (N = rank of SU(N))."""
    lam = tuple(lam)
    tab = column_table(records, lam)
    cap, src = capacity(lam, content)
    tw, ind = twisted_and_plain(N, lam)
    H = sum(h for _, h, _, _ in tab.values())
    euler = sum((-1) ** n * h for n, (_, h, _, _) in tab.items())
    assert (H - tw) % 2 == 0
    p_odd = (H - tw) // 2
    bh = sum(b for _, _, _, b in tab.values())
    per2J = {n: max(0, b - cap.get(n, 0)) for n, (_, _, _, b) in tab.items()}
    return dict(lam=list(lam), sectors={n: dict(m=m, h=h, grav=g, BH=b) for n, (m, h, g, b) in tab.items()},
                capacity=dict(cap), capacity_sources=src, H_total=H, euler=euler, index=ind, euler_ok=euler == ind,
                twisted=tw, p_odd=p_odd, BH=bh, cores_per_2J=per2J, cores_lower_bound=sum(per2J.values()),
                cores_lower_bound_column=max(0, p_odd - sum(cap.values())))
