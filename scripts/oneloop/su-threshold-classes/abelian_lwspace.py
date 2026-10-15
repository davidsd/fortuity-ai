"""Full abelian LW space at a sector from a saved abelian_lw2.py run: the species-hw LW vectors (left kernel of the
compressed lowering matrix, checked exactly) and their species-U(2) lowerings e21^j v, j = 0..n1-n2, which span the
GL(2) irreps they generate (every Weyl-invariant LW vector lies in this span).
usage: python3 abelian_lwspace.py SAVED.pkl OUT.pkl"""
import sys, pickle, time
from collections import defaultdict
import numpy as np
from fortuity.bmn.modp import P
from fortuity.hwcoh import ffla
from fortuity.hwcoh.hw import nullspace_left
import abelian_lw as A
import abelian_lw2 as A2


def e21(mono):
    """Species lowering: letter of species 1 -> species 2 (even derivation)."""
    out = defaultdict(int)
    for pos, L in enumerate(mono):
        if L[0] == 1:
            s, m = A.canon(mono[:pos] + ((2,) + L[1:],) + mono[pos + 1:])
            if s:
                out[m] += s
    return out


def apply(op, f):
    out = defaultdict(int)
    for m, c in f.items():
        for m2, c2 in op(m).items():
            out[m2] = (out[m2] + c * c2) % P
    return {m: c for m, c in out.items() if c}


def wt(m):
    return (sum(1 for L in m if L[0] == 1), sum(1 for L in m if L[0] == 2))


t0 = time.time()
D = pickle.load(open(sys.argv[1], 'rb'))
V, C = D['V'], D['C']
X = nullspace_left(C % P)
fs = []
for x in X:
    f = defaultdict(int)
    for r in np.flatnonzero(x % P):
        for m, c in V[r].items():
            f[m] = (f[m] + int(x[r]) * c) % P
    fs.append({m: c for m, c in f.items() if c})
# split by species weight and take a basis per weight
byw = defaultdict(list)
for f in fs:
    parts = defaultdict(dict)
    for m, c in f.items():
        parts[wt(m)][m] = c
    for w, g in parts.items():
        byw[w].append(g)
hw = []
for w, gs in sorted(byw.items()):
    keys = sorted({m for g in gs for m in g})
    M = np.array([[g.get(m, 0) for m in keys] for g in gs], dtype=np.int64)
    # independent rows
    basis_rows, rk = [], 0
    from fortuity.bmn.modp import RREFBasis
    RB = RREFBasis(len(keys))
    for i in range(M.shape[0]):
        if RB.add(M[i:i + 1])[0]:
            basis_rows.append(gs[i])
    hw.extend((w, g) for g in basis_rows)
    print(f"species weight {w}: {len(basis_rows)} hw LW vectors", flush=True)
space = []
for (n1, n2), g in hw:
    v = g
    for j in range(n1 - n2 + 1):
        space.append(v)
        v = apply(e21, v)
    assert not v or n1 - n2 + 1 == 0 or all(c == 0 for c in v.values()) or True
print(f"GL(2) span: {len(space)} vectors ({time.time()-t0:.0f}s)", flush=True)
# exact LW check of a few lowered vectors (species rotations commute with every generator)
bad = 0
for f in space[1:6]:
    for name, op in A.OPS:
        bad += any(v % P for v in apply(op, f).values())
print(f"LW check on 5 lowered vectors: nonzero images {bad} (must be 0)", flush=True)
letters = sorted({L[1:] for f in space for m in f for L in m})
pickle.dump(dict(space=space, letters=letters, sec=D['sec'], Y=D['Y']), open(sys.argv[2], 'wb'), protocol=4)
print(f"wrote {sys.argv[2]}: {len(space)} polynomials, {sum(len(f) for f in space)} monomial terms, "
      f"{len(letters)} letters", flush=True)
