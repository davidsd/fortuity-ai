"""Corrected 'Table 1' of arXiv:2312.16443 from the kernel parity data of R-017 (data/su3/kernel_parity_j40.json).

Per hw column lam (GL(3) partition, |lam| = j/2):  T = #P+ - #P-,  O = sum (-1)^Y  over the classes invisible on the
Coulomb branch.  Both series have the same multiplet structure: SU(1|3) descendants prod_I (1 + x_I) (Q_+^m keeps Y
and C) and the conjectured w_2 hairs 1 / prod_I (1 - x_I x_1 x_2 x_3).  Core = series * prod(1 - x_I e_3) / prod(1 + x_I),
decomposed into Schur functions.  The paper's signed core is (-1)^{j/2} * core_O: it reproduces every entry of
Table 1 of the paper through j = 40.  T adds the parity: T < 0 entries are P-odd, hence not gravitons (R-017; certified
by direct cohomology only for j <= 30); T > 0 entries are P-even and invisible on the Coulomb branch (at j = 24 and 30
they are the gravitons R_24^{(0,3)} and R_30^{(0,5)} of R-009/R-015).
usage (from the repo root): python3 scripts/bmn/su3-level24/table1_core.py"""
import json, itertools
from collections import defaultdict

def padd(a, b, c=1):
    out = defaultdict(int, a)
    for k, v in b.items():
        out[k] += c * v
    return {k: v for k, v in out.items() if v}

def pmul(a, b, dmax):
    out = defaultdict(int)
    for (i, j, k), v in a.items():
        for (l, m, n), w in b.items():
            if i + j + k + l + m + n <= dmax:
                out[(i + l, j + m, k + n)] += v * w
    return {k: v for k, v in out.items() if v}

def schur(lam):
    """s_lam(x1,x2,x3) as a monomial dict, by the bialternant formula (division done by expanding monomials)."""
    l1, l2, l3 = lam
    # sum over semistandard tableaux = weights; use Gelfand-Tsetlin patterns
    out = defaultdict(int)
    for m1 in range(l2, l1 + 1):
        for m2 in range(l3, l2 + 1):
            for n in range(m2, m1 + 1):
                w1 = n
                w2 = m1 + m2 - n
                w3 = l1 + l2 + l3 - m1 - m2
                out[(w1, w2, w3)] += 1
    return dict(out)

def to_schur(f):
    """coefficients of f (symmetric polynomial) in the Schur basis: coefficient of x^(lam+delta) in f * a_delta"""
    adelta = {(2, 1, 0): 1, (1, 2, 0): -1, (2, 0, 1): -1, (0, 2, 1): 1, (1, 0, 2): 1, (0, 1, 2): -1}
    g = pmul(f, adelta, 10 ** 6)
    out = {}
    for (a, b, c), v in g.items():
        if a > b > c >= 0:
            out[(a - 2, b - 1, c)] = v
    return out

DMAX = 20
rows = json.load(open('data/su3/kernel_parity_j40.json'))['rows']
GT, GO = {}, {}
for r in rows:
    s = schur(tuple(r['lam']))
    GT = padd(GT, s, r['T'])
    GO = padd(GO, s, r['O'])
# factors
one = {(0, 0, 0): 1}
x = [{(1, 0, 0): 1}, {(0, 1, 0): 1}, {(0, 0, 1): 1}]
e3 = {(1, 1, 1): 1}
hair_inv = one
for xi in x:
    hair_inv = pmul(hair_inv, padd(one, pmul(xi, e3, DMAX), -1), DMAX)
inv_desc = one                                  # 1 / prod (1 + x_I) = prod sum_n (-x_I)^n
for i in range(3):
    ser = {tuple(n if k == i else 0 for k in range(3)): (-1) ** n for n in range(DMAX + 1)}
    inv_desc = pmul(inv_desc, ser, DMAX)
fac = pmul(hair_inv, inv_desc, DMAX)
coreT = to_schur(pmul(GT, fac, DMAX))
coreO = to_schur(pmul(GO, fac, DMAX))
# also the series with descendants only removed (no hair assumption)
primT = to_schur(pmul(GT, inv_desc, DMAX))
primO = to_schur(pmul(GO, inv_desc, DMAX))
def dyn(l):
    return f"[{l[0]-l[1]},{l[1]-l[2]}]"
res = {}
for name, (cT, cO) in (('core (descendants and w_2 hairs removed)', (coreT, coreO)), ('primaries (descendants removed only)', (primT, primO))):
    print('==', name)
    tab = defaultdict(list)
    for lam in sorted(set(cT) | set(cO)):
        j = 2 * sum(lam)
        t, o = cT.get(lam, 0), cO.get(lam, 0)
        if t or o:
            tab[j].append((dyn(lam), lam, t, o))
    for j in sorted(tab):
        if j > 40:
            continue
        print(f" j={j}: " + '; '.join(f"{d} T={t:+d} O={o:+d} index sign {(-1)**(j//2)*o:+d}" for d, lam, t, o in tab[j]))
    res[name] = {str(j): [dict(dynkin=d, lam=list(lam), T=t, O=o, signed_index=(-1) ** (j // 2) * o) for d, lam, t, o in v] for j, v in sorted(tab.items()) if j <= 40}
# The descendant factor.  The paper prints prod_{I<J} (1 - e^{-Delta_I - Delta_J}) = prod (1 - t^4 x_I x_J); the supercharges
# Q_+^m (phi^n -> eps^{mnp} psi_p, psi_m -> f) carry e^{-Delta_m} = t^2 x_m.  Signed core (the paper's convention,
# (-1)^{j/2} O per column) with each of the two factors: only prod_I (1 - t^2 x_I) reproduces Table 1.
GS = {}
for r in rows:
    GS = padd(GS, schur(tuple(r['lam'])), (-1) ** sum(r['lam']) * r['O'])
def geom(mono):                                  # 1 / (1 - mono)
    out, cur = dict(one), dict(one)
    for _ in range(DMAX):
        cur = pmul(cur, {mono: 1}, DMAX)
        if not cur:
            break
        out = padd(out, cur)
    return out
for name, monos in (('signed core with prod_I (1 - t^2 x_I)', [(1, 0, 0), (0, 1, 0), (0, 0, 1)]),
                    ('signed core with the printed prod_{I<J} (1 - t^4 x_I x_J)', [(0, 1, 1), (1, 0, 1), (1, 1, 0)])):
    f = pmul(GS, hair_inv, DMAX)
    for mono in monos:
        f = pmul(f, geom(mono), DMAX)
    core = to_schur(f)
    tab = defaultdict(list)
    for lam, v in sorted(core.items()):
        if v and 2 * sum(lam) <= 40:
            tab[2 * sum(lam)].append((dyn(lam), v))
    print('==', name)
    for j in sorted(tab):
        print(f" j={j}: " + '; '.join(f"{d} {v:+d}" for d, v in tab[j]))
    res[name] = {str(j): [dict(dynkin=d, signed_index=v) for d, v in tab[j]] for j in sorted(tab)}
json.dump(res, open('scripts/bmn/su3-level24/runs/table1_core_j40.json', 'w'), indent=1)
