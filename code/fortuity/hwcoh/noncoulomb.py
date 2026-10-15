"""Coulomb-type and non-Coulomb graviton classes (R-015).

A graviton class (a product of the single-trace generators u_n, v_n, w_n, modulo Q-exact) is Coulomb-type if it is
nonzero on Cartan-valued letters (the Coulomb branch, where Q-exact operators vanish), and non-Coulomb otherwise.
In an SU(3)_R highest-weight sector with g graviton classes, the Coulomb-type ones are counted by the rank of the
graviton highest-weight vectors on Cartan-valued letters, so non-Coulomb = g - that rank.

Random evaluation can only underestimate a rank, so a computed non-Coulomb count of zero is exact. A positive count is
recomputed with a second seed and doubled functionals (`coulomb_recheck`), as in R-015.

Two routes:
- `cartan_rank` / `sector_counts`: direct, per sector, from the graviton hw vectors of `GravitonAlgebra` (the method
  of R-015; moved here from code/fortuity/hwcoh/noncoulomb.py).
- `coulomb_hw_counts`: per-weight counts of `fortuity.bmn.gravindex.GravitonCounter` (eigenvalue mesons), alternated
  over the Weyl group of SU(3)_R. It is independent of the first route but is a signed sum of sampled counts, not a
  bound (review C19), so it is used as a cross-check (code/tests/test_noncoulomb.py).

CLI (regenerates data/noncoulomb/noncoulomb_<name>.jsonl from the collaboration's hw scans):
    python -m fortuity.hwcoh.noncoulomb GROUP N GEN_NMAX OUT DATA [DATA ...]
DATA: `fortuity.hwcoh.run_sectors` jsonl, or `fortuity.hwcoh.to_json` json. Restartable (sectors already in OUT are
skipped).
"""
import json
import math
import os
import sys
import time

import numpy as np

from . import ffla
from .evalg import DegFrames, Frame, algebra_dim
from .gravhw import GravitonAlgebra
from .sector import frame_k
from ..bmn.words import contents_of_sector


def cartan_rank(GA, blocks, group, N, sec, nrows, seed=7, factor=1):
    """Rank of the hw vectors `blocks` (from GA.hw_blocks(sec), nrows rows) on Cartan-valued letters."""
    degs = sorted({sum(c[3:6]) for c in contents_of_sector(sec)})
    cols = []
    for d in degs:
        if d > 3 * algebra_dim(group, N):        # more fermions than Grassmann components: words vanish identically
            continue
        k = frame_k(N, d, group=group)
        need = factor * (nrows + 32)
        npts = max(2, math.ceil(need / max(1, math.comb(k, d))))
        fr = Frame(N, npts, k, [d], seed=seed + 101 * d, group=group, cartan=True)
        df = DegFrames.__new__(DegFrames)
        df.frames, df.off, df.ncols = {d: fr}, {d: 0}, fr.ncols
        cols.append(GA.rows_from_blocks(blocks, df))
    if not cols:
        return 0
    Gc = np.concatenate(cols, axis=1)
    return int(ffla.rank(Gc)) if Gc.shape[0] else 0


def sector_counts(GA, group, N, sec, grav, seed=7):
    """{grav, grav_rows, coulomb, noncoulomb[, coulomb_recheck]} for the BMN sector sec = (2J, q1, q2, q3) with
    `grav` graviton classes (from a complete hw scan)."""
    blocks = GA.hw_blocks(sec)
    nrows = sum(H.shape[0] for _, H in blocks)
    c = cartan_rank(GA, blocks, group, N, sec, nrows, seed=seed)
    rec = dict(grav=int(grav), grav_rows=int(nrows), coulomb=int(c), noncoulomb=int(grav - c))
    if rec['noncoulomb'] > 0:                         # re-check: second seed, doubled functionals
        c2 = cartan_rank(GA, blocks, group, N, sec, nrows, seed=1234, factor=2)
        rec['coulomb_recheck'] = int(c2)
        rec['noncoulomb'] = int(grav - max(c, c2))
    return rec


def load_sectors(path, group, N):
    """[(j, lam, 2J, h or None, grav)] from a run_sectors jsonl or a to_json json file."""
    out = []
    if path.endswith('.jsonl'):
        rows = {}
        for line in open(path):
            r = json.loads(line)
            if r.get('group', 'SU') == group and r['N'] == N:
                rows[(tuple(r['lam']), r['n'])] = r
        for (lam, n), r in sorted(rows.items()):
            rin = rows[(lam, n - 1)]['r_up'] if (lam, n - 1) in rows else (0 if n == 0 else None)
            h = r['m'] - rin - r['r_up'] if rin is not None else None
            out.append((2 * sum(lam), lam, n, h, r['grav']))
        return out
    d = json.load(open(path))
    for c in d['columns']:
        for s in c.get('sectors', [c]):
            n = s['twoJ']
            out.append((c['j'], tuple(c['lam']), n, s.get('h'), s['grav']))
    return out


def coulomb_hw_counts(group, N, jmax, nmax=None):
    """{(2J, lam): Coulomb-type graviton hw multiplicity} from fortuity.bmn.gravindex (independent route)."""
    from ..bmn.gravindex import GravitonCounter
    from ..bmn.twistedindex import hw_multiplicities
    dims = GravitonCounter(N, jmax, group=group, verbose=False, nmax=nmax).run()
    by_n = {}
    for sec, d in dims.items():
        if d and 2 * sum(sec[1:]) <= jmax:
            by_n.setdefault(sec[0], {})[tuple(sec[1:])] = by_n.get(sec[0], {}).get(tuple(sec[1:]), 0) + d
    out = {}
    for n, wts in by_n.items():
        for lam, c in hw_multiplicities(wts, jmax // 2).items():
            out[(n, tuple(int(x) for x in lam))] = int(c)
    return out


def main(argv):
    group, N, nmax, out = argv[0], int(argv[1]), int(argv[2]), argv[3]
    secs = [s for path in argv[4:] for s in load_sectors(path, group, N)]
    todo = [s for s in secs if s[4] > 0]
    done = set()
    if os.path.exists(out):
        for line in open(out):
            r = json.loads(line)
            done.add((r['j'], tuple(r['lam']), r['n']))
    GA = GravitonAlgebra(nmax)
    t0, found = time.time(), []
    with open(out, 'a') as fh:
        for j, lam, n, h, grav in todo:
            if (j, lam, n) in done:
                continue
            rec = dict(group=group, N=N, j=j, lam=list(lam), n=n, h=h, gen_nmax=nmax)
            rec.update(sector_counts(GA, group, N, (n,) + lam, grav))
            fh.write(json.dumps(rec) + '\n')
            fh.flush()
            if rec['noncoulomb'] > 0:
                found.append(rec)
    print(f"[{time.time() - t0:.0f}s] {group} N={N}: {len(todo)} sectors with graviton classes; non-Coulomb graviton "
          f"classes in {len(found)} new sectors: {[(r['j'], r['lam'], r['n'], r['noncoulomb']) for r in found]}")


if __name__ == '__main__':
    main(sys.argv[1:])
