"""Checks behind R-010 (G2 against SO(7) at the BMN sector (1;3,3,3)), promoted from
scripts/bmn/so-usp-g2-scans/{restrict_so7_class,completeness}.py (review C14).

- SO7_CLASS_TERMS / so7_class(): the SO(7) fortuitous class of R-005, eq. (4.14) of arXiv:2506.13887 (Gadde-Lee-Raj-Tomar),
  transcribed with X, Y, Z = phi_1, phi_2, phi_3 and without its "-(grav)" projection.
- classify(O, group, N, sec): is O zero, Q-exact, or in span(Q-exact + multi-graviton) in sector sec of the group?
- trace_word_completeness(group, N, jmax): in every BMN sector with j <= jmax (q sorted), is the rank of the multi-trace
  words at random group-valued matrices equal to the exact sector dimension (Haar integral, which counts all invariants,
  e.g. for G2 also those built with the 3-form)?
CLI (from code/):  python -m fortuity.bmn.g2_checks restriction      (R-010 (ii), (iii))
                   python -m fortuity.bmn.g2_checks completeness 18  (R-010 (i); ~10 min on a laptop)
"""
import copy
from collections import defaultdict
from .words import add_term, Q_expr, PHI, PSI
from .evaluate import Points
from .symdirect import SymDirect

X, Y, Z = PHI
p1, p2, p3 = PSI

# (coefficient, list of traces); each trace a tuple of letters in order. Transcribed from arXiv:2506.13887 eq. (4.14).
SO7_CLASS_TERMS = [
    (1, [(Y, Y), (X, p1), (X, Z), (X, Z)]),
    (-4, [(Y, Y), (X, Z), (Z, X, Z, p3)]),
    (-1, [(X, Z), (X, Z), (Z, Y, p3, Y)]),
    (-4, [(X, Z), (X, Z), (Z, Y, Y, p3)]),
    (8, [(X, Z), (Z, X, Y, Y, Z, p3)]),
    (4, [(X, Z), (Z, X, Z, Y, p3, Y)]),
    (16, [(X, Z), (Z, X, Z, Y, Y, p3)]),
    (-4, [(Z, p3), (Z, X), (Z, X, Y, Y)]),
    (8, [(Z, X, Z, p3), (X, Y, Z, Y)]),
    (-2, [(Y, Y), (Z, p3), (Z, X, Z, X)]),
    (8, [(Z, X, Z, X), (Y, Y, Z, p3)]),
    (2, [(Z, X, Z, X), (Y, Z, Y, p3)]),
    (16, [(Y, Z, p3), (Z, X, Z, X, Y)]),
    (8, [(Z, p3), (Z, X, Z, X, Y, Y)]),
    (8, [(Y, Y), (Z, X, Z, X, Z, p3)]),
    (16, [(Z, X, Z, X, Y, Z, Y, p3)]),
    (-8, [(Z, X, Z, X, Z, Y, p3, Y)]),
    (-32, [(Z, X, Z, X, Z, Y, Y, p3)]),
    (-16, [(Z, X, Z, Y, X, Y, Z, p3)]),
    (-16, [(Z, X, Z, Y, X, Z, Y, p3)]),
    (-16, [(Z, X, Z, Y, Y, X, Z, p3)]),
]
SO7_CLASS_SECTOR = (1, 3, 3, 3)


def so7_class():
    O = defaultdict(int)
    for c, traces in SO7_CLASS_TERMS:
        add_term(O, tuple(traces), c)
    return O


def vanishes(expr, group, N, deg, seed=7):
    """True if expr vanishes identically on group-valued matrices (random evaluation over F_p)."""
    if not expr:
        return True
    return not Points(N, 8, deg + 3, [deg], seed=seed, group=group).expr_vector(expr).any()


def classify(O, group, N, sec=SO7_CLASS_SECTOR, gen_nmax=8, seed=777):
    """(is_zero, in_exact, in_exact_plus_graviton) for the expression O in sector sec = (2J, q1, q2, q3)."""
    sd = SymDirect(N, gen_nmax=gen_nmax, seed=seed, verbose=False, symmetrize=False, group=group)
    up = (sec[0] + 1,) + tuple(sec[1:])
    sd.symwords(sec)
    sd.rankQ(up)
    basis, pts = sd._Qrows[up]
    v = pts.expr_vector(O)[None, :]
    if not v.any():
        return True, True, True
    b1 = copy.deepcopy(basis)
    in_exact = not bool(b1.add_batched(v).sum())
    b2 = copy.deepcopy(basis)
    G, _ = sd.graviton_rows(sec, pts)
    if G.size:
        b2.add_batched(G)
    in_eg = not bool(b2.add_batched(v).sum())
    return False, in_exact, in_eg


def trace_word_completeness(group, N, jmax, log=None):
    """Returns (number of sectors checked, list of (sector, exact dim, trace-word rank) mismatches)."""
    from fortuity.hwcoh.dims import partition_dims_nf, collapse
    from .words import words_of_sector
    dims = collapse(partition_dims_nf(N, jmax, group=group))
    sd = SymDirect(N, gen_nmax=2, verbose=False, symmetrize=False, group=group, margin=16)
    bad, n = [], 0
    for sec, d in sorted(dims.items()):
        tj, q1, q2, q3 = sec
        if not (q1 >= q2 >= q3):
            continue
        sw = [{w: 1} for w in words_of_sector(sec, 100)]
        if not sw:
            if d:
                bad.append((sec, d, 0))
            continue
        b, _ = sd._basis_doubling(sec, len(sw), lambda p: (p.expr_vector(e) for e in sw), 'E')
        n += 1
        if b.rank != d:
            bad.append((sec, d, b.rank))
    return n, bad


if __name__ == "__main__":
    import sys
    what = sys.argv[1] if len(sys.argv) > 1 else 'restriction'
    if what == 'completeness':
        jmax = int(sys.argv[2]) if len(sys.argv) > 2 else 18
        n, bad = trace_word_completeness('G2', 0, jmax)
        print(f"G2: sectors checked (q sorted, j<={jmax}): {n}; mismatches (sector, exact dim, trace-word rank): {bad}")
    else:
        O = so7_class()
        QO = Q_expr(O)
        print(f"O: {len(O)} canonical terms; Q O: {len(QO)} terms (formal)")
        print("(a) Q O = 0 on so(7):", vanishes(QO, 'SO_odd', 3, 0), "| Q O = 0 on so(9):", vanishes(QO, 'SO_odd', 4, 0))
        z, e, eg = classify(O, 'SO_odd', 3)
        print(f"(b) SO(7): zero={z}, exact={e}, in exact+graviton={eg}  -> non-graviton class: {not eg}")
        print("(c) restriction to g2: Q O = 0:", vanishes(QO, 'G2', 0, 0))
        z, e, eg = classify(O, 'G2', 0)
        print(f"    G2: zero={z}, exact={e}, in exact+graviton={eg}")
