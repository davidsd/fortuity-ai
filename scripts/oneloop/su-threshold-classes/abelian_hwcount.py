"""Dimension of the Weyl-invariant, SU(3)_R-hw part of the abelian (Coulomb-branch) module in a sector, all SU(2)_R
spins: the space that eps(H(Q_0)) of SU(3) lands in.  Compare with the Coulomb rank of the sector (eps surjective iff
equal).  Species-U(2) hw vectors (kernel of E1, E2, e12) are counted per species weight (n1, n2); the Weyl group S_3
acts on the species plane by its reflection representation, so the GL(2) irrep (n1, n2) = Sym^(n1-n2) (x) det^n2
contributes #S_3-invariants (det = sign on S_3).
usage: python3 abelian_hwcount.py 2J1 2J2 2q1 2q2 2q3 Y"""
import sys, time
from collections import defaultdict
import numpy as np
from fortuity.bmn.modp import P
from fortuity.hwcoh import ffla
import abelian_lw as A
import abelian_lw2 as A2

log = lambda *a: print(time.strftime('%H:%M:%S'), *a, flush=True)
sec, Y = tuple(int(x) for x in sys.argv[1:6]), int(sys.argv[6])
t0 = time.time()
basis = A.monomials(sec, Y)
log(f"sector {sec} Y={Y}: {len(basis)} abelian monomials ({time.time()-t0:.0f}s)")
ops = [op for name, op in A.OPS if name in ('E1', 'E2')] + [A2.e12]
groups = defaultdict(list)
for m in basis:
    groups[A2.coarse(m)].append(m)
dims = defaultdict(int)
for g, ms in groups.items():
    # split each block by species weight (all three operators map weight (n1, n2) to fixed shifts; the kernel splits)
    byw = defaultdict(list)
    for m in ms:
        byw[(sum(1 for L in m if L[0] == 1), sum(1 for L in m if L[0] == 2))].append(m)
    for w, mw in byw.items():
        rows, ent = {}, []
        for j, m in enumerate(mw):
            for k, op in enumerate(ops):
                for m2, c in op(m).items():
                    if c % P:
                        ent.append((rows.setdefault((k, m2), len(rows)), j, c % P))
        if not rows:
            dims[w] += len(mw)
            continue
        M = np.zeros((len(rows), len(mw)), dtype=np.int64)
        for r, j, c in ent:
            M[r, j] = (M[r, j] + c) % P
        dims[w] += len(mw) - int(ffla.rank(M))
inv = lambda m: sum(1 for j in range(m // 3 + 1) if (m - 3 * j) % 2 == 0) if m >= 0 else 0
sgn = lambda m: inv(m - 3) if m >= 3 else 0
tot = 0
for (n1, n2), d in sorted(dims.items()):
    if d and n1 >= n2:
        k = inv(n1 - n2) if n2 % 2 == 0 else sgn(n1 - n2)
        tot += d * k
        log(f"  species weight ({n1},{n2}): {d} species-hw SU(3)-hw vectors x {k} Weyl invariants")
log(f"Weyl-invariant SU(3)_R-hw dimension of the abelian module: {tot} ({time.time()-t0:.0f}s)")
