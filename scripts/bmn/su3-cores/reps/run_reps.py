"""Fused or core?  Explicit non-graviton representatives of one SU(3) hw sector and their SU(3) > SU(2) x U(1) channels.

usage (repo root, PYTHONPATH=code):
  python3 run_reps.py LAM N2J CHANNELS RECORDS DIMS_PKL OUTDIR
    LAM       q1,q2,q3                 e.g. 8,5,4
    N2J       2J of the sector          e.g. 6
    CHANNELS  alpha;alpha;...           e.g. "3,1,0;4,0,0"  (x^alpha channels whose SU(2) target holds black holes,
                                        from analysis/core_test.py: the fused capacity)
    RECORDS   run_sectors jsonl of the column (the engine records of sectors 2J-1 and 2J give r_down and the checks)
    DIMS_PKL  nf-refined SU(3) dims (cluster/dims_nf_su3_36.pkl)
env: SEED (1; must be the engine records' seed only for the expected ranks, which do not depend on it), MAXLEN,
     MAXLEN_UP (6), GEN_NMAX (3), TARGET_SEED (17), NCHECK (1: closedness of every representative at N=3 and of the
     cores at N=4 with fresh points), HWPAR_WORKERS, TMPDIR (memmaps).
Output: OUTDIR/reps_<lam>_<2J>.json (summary) and the representatives, restricted images and cores as .json.gz.
"""
import json
import os
import pickle
import sys
import time
from collections import Counter

import numpy as np

from fortuity.bmn.modp import P
from fortuity.bmn.words import Q_expr, nferm
from fortuity.bmn.blockrestrict import save_expr_json, restrict_u1
from fortuity.bmn.fullindex import partition_dims
from fortuity.hwcoh import classreps_par as cp
from fortuity.hwcoh.gravhw import GravitonAlgebra
from fortuity.hwcoh.evalg import Frame
from fortuity.hwcoh.sector import frame_k

t0 = time.time()


def log(*a):
    print(f'[{time.time() - t0:8.1f}s]', *a, flush=True)


lam = tuple(int(x) for x in sys.argv[1].split(','))
n = int(sys.argv[2])
channels = [tuple(int(x) for x in c.split(',')) for c in sys.argv[3].split(';') if c]
recs = {}
for line in open(sys.argv[4]):
    r = json.loads(line)
    if r.get('N', 3) == 3 and tuple(r['lam']) == lam:
        recs[r['n']] = r
dims = pickle.load(open(sys.argv[5], 'rb'))
outdir = sys.argv[6]
os.makedirs(outdir, exist_ok=True)
seed = int(os.environ.get('SEED', '1'))
maxlen = int(os.environ['MAXLEN']) if os.environ.get('MAXLEN') else None
maxlen_up = int(os.environ['MAXLEN_UP']) if os.environ.get('MAXLEN_UP') else maxlen
gen_nmax = int(os.environ.get('GEN_NMAX', '3'))
tseed = int(os.environ.get('TARGET_SEED', '17'))
ncheck = os.environ.get('NCHECK', '1') == '1'
tag = f"{'_'.join(map(str, lam))}_{n}"

# expected numbers from the engine records: h(2J) = m - r_up(2J) - r_up(2J-1)
rn, rd = recs[n], recs.get(n - 1)
assert rn.get('check') and (rd is None or rd.get('check')), 'engine self-checks must have passed'
r_down = rd['r_up'] if rd else 0
h = rn['m'] - rn['r_up'] - r_down
expect = dict(m=rn['m'], r_up=rn['r_up'], grav=rn['grav'], h=h, bh=h - rn['grav'])
log(f'sector SU(3) (2J={n}; {lam}): expected {expect}, r_down {r_down}; channels {channels}')

GA = GravitonAlgebra(gen_nmax)
reps, info = cp.nongraviton_reps(3, lam, n, dims, GA, r_down, expect=expect, seed=seed, maxlen=maxlen,
                                 maxlen_up=maxlen_up, log=log)
log(f'representatives: {info}')
for i, rep in enumerate(reps):
    save_expr_json(rep, os.path.join(outdir, f'rep_{tag}_{i}.json.gz'),
                   description=f'non-graviton representative {i} of SU(3) (2J={n}; {lam}), mod p={P}, seed {seed}')


def content(expr):
    """term counts by (number of psi, number of f, number of traces)"""
    c = Counter()
    for m in expr:
        c[(sum(nferm(w) for w in m), sum(w.count(6) for w in m), len(m))] += 1
    return {str(k): v for k, v in sorted(c.items())}


def eval_vec(N, expr, seed, npts=6):
    """values of expr at random SU(N) points (per fermion degree; fresh frames)"""
    by = {}
    for m, c in expr.items():
        by.setdefault(sum(nferm(w) for w in m), []).append((m, c))
    out = []
    for d, items in sorted(by.items()):
        fr = Frame(N, npts, frame_k(N, d), [d], seed=seed + 31 * d)
        acc = np.zeros(fr.ncols, dtype=np.int64)
        for a in range(0, len(items), 20000):
            chunk = items[a:a + 20000]
            X = fr.eval_words([m for m, _ in chunk]).astype(np.int64) % P
            cs = np.array([c % P for _, c in chunk], dtype=np.int64)
            acc = (acc + (cs[:, None] * X % P).sum(axis=0)) % P
        out.append(acc)
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


# restriction channels
dims2 = partition_dims(2, 2 * (sum(lam) - min(sum(a) for a in channels)) + 2, chunk=4) if channels else {}
chan = cp.channel_residues(reps, 3, lam, n, channels, dims2, gen_nmax=3, tseed=tseed, log=log)
rank, K = cp.cores_from_residues(chan)
log(f'restriction map on the {len(reps)} non-graviton classes: rank {rank}; cores {len(reps) - rank}')
cores = []
for j, row in enumerate(K if K is not None else []):
    core = {}
    for c, rep in zip(row, reps):
        if c % P:
            for m, v in rep.items():
                core[m] = (core.get(m, 0) + int(c) * v) % P
    core = {m: v for m, v in core.items() if v}
    cores.append(core)
    save_expr_json(core, os.path.join(outdir, f'core_{tag}_{j}.json.gz'),
                   description=f'core {j} of SU(3) (2J={n}; {lam}): combination of the representatives with zero '
                               f'residue in the channels {channels}')
# images of the cores must have zero residue (recomputed from the combined expression)
core_check = []
if cores:
    chk = cp.channel_residues(cores, 3, lam, n, channels, dims2, gen_nmax=3, tseed=tseed, log=log)
    core_check = [[bool(np.any(r % P)) for r in c['residues']] for c in chk]

checks = {}
if ncheck:
    for i, rep in enumerate(reps):
        q = Q_expr(rep)
        checks[f'rep{i}_closed_N3'] = not eval_vec(3, q, seed=9001 + i).any()
    for j, core in enumerate(cores):
        q = Q_expr(core)
        checks[f'core{j}_closed_N3'] = not eval_vec(3, q, seed=9101 + j).any()
        checks[f'core{j}_closed_N4'] = not eval_vec(4, q, seed=9201 + j, npts=3).any()
    log(f'closedness checks (fresh points): {checks}')

summary = dict(sector=dict(N=3, lam=list(lam), twoJ=n), expect=expect, info=info,
               channels=[dict({k: v for k, v in c.items() if k != 'residues'},
                              residue_nonzero=[bool(np.any(r % P)) for r in c['residues']]) for c in chan],
               restriction_rank=int(rank), cores=len(cores),
               core_coefficients=[[int(x) for x in row] for row in (K if K is not None else [])],
               core_terms=[len(c) for c in cores], core_residues_nonzero=core_check,
               rep_content=[content(r) for r in reps], core_content=[content(c) for c in cores],
               checks=checks, prime=P, seed=seed, target_seed=tseed, time=round(time.time() - t0, 1))
json.dump(summary, open(os.path.join(outdir, f'reps_{tag}.json'), 'w'), indent=1, default=str)
log('done')
