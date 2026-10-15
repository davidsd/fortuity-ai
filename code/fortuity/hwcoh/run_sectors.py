"""Run hw-complex sectors and append one JSON line per sector.

usage: python -m fortuity.hwcoh.run_sectors N DIMS_PKL OUT_JSONL TASK [TASK ...]
  TASK = q1,q2,q3        (all 2J of that column)   or   q1,q2,q3:n1,n2,...
DIMS_PKL: nf-refined dims (hwcoh.dims.partition_dims_nf) or 'auto:JMAX' to compute them here. For several runs on the same
  group, compute the dims once to a pickle and pass its path (SO(9) at JMAX = 22: ~5 min with 8 workers).
Env FORTUITY_WORKERS: processes for the 'auto:' dims (default SLURM_CPUS_PER_TASK, else all cores; the SO/Sp/G2 dims are
  otherwise single-threaded: ~40 min for SO(9) at JMAX = 22).
Env GROUP: 'SU' (default), 'SO_odd' (SO(2N+1)), 'Sp' (USp(2N)) or 'G2' (N ignored, pass 0).
Env ENGINE=par: parallel, memmap-based engine (hwcoh.sectorpar); workers from SLURM_CPUS_PER_TASK.
Env MAXLEN_UP: cap for the 2J+1 words (Q-images); valid if that sector's own check passed with a cap <= MAXLEN_UP.
Env MAXLEN: restrict to traces of length <= MAXLEN (verified per sector by the rank = m check).
Sectors already present in OUT_JSONL (same N, lam, n, check true) are skipped (restartable).
"""
import sys, os, json, time, pickle
if os.environ.get('ENGINE') == 'par':
    # no FLINT thread pool: forked hw workers use FLINT (single-threaded); big null spaces use the BLIS kernel
    os.environ['FLINT_THREADS'] = '1'
from fortuity.bmn.words import contents_of_sector
from fortuity.hwcoh.sector import Sector
from fortuity.hwcoh.sectorpar import SectorPar
from fortuity.hwcoh.gravhw import GravitonAlgebra
from fortuity.hwcoh.dims import partition_dims_nf

maxlen = int(os.environ['MAXLEN']) if os.environ.get('MAXLEN') else None
maxlen_up = int(os.environ['MAXLEN_UP']) if os.environ.get('MAXLEN_UP') else None
seed = int(os.environ.get('SEED', '1'))
group = os.environ.get('GROUP', 'SU')
N = int(sys.argv[1]); dsrc = sys.argv[2]; out = sys.argv[3]
if dsrc.startswith('auto:'):
    dworkers = int(os.environ.get('FORTUITY_WORKERS') or os.environ.get('SLURM_CPUS_PER_TASK') or os.cpu_count() or 1)
    print(f"computing exact dims: group {group}, N {N}, JMAX {dsrc[5:]}, {dworkers} workers", flush=True)
    dims = partition_dims_nf(N, int(dsrc[5:]), group=group, workers=dworkers, progress=True)
    print("dims done", flush=True)
else:
    dims = pickle.load(open(dsrc, 'rb'))
done = set()
if os.path.exists(out):
    for line in open(out):
        r = json.loads(line)
        if r.get('check'):
            done.add((r.get('group', 'SU'), r['N'], tuple(r['lam']), r['n']))
from fortuity.bmn.gravindex import default_nmax
# default: the top Casimir degree of the group (N for SU(N), 2N for SO(2N+1)/USp(2N), 6 for G2); the old default N was
# wrong for SO/USp/G2 (review B.3)
gen_nmax = int(os.environ.get('GEN_NMAX', default_nmax(group, N)))
GA = GravitonAlgebra(gen_nmax)
for task in sys.argv[4:]:
    lamstr, _, nstr = task.partition(':')
    lam = tuple(int(x) for x in lamstr.split(','))
    ns = [int(x) for x in nstr.split(',')] if nstr else [n for n in range(0, sum(lam) + 1) if contents_of_sector((n,) + lam)]
    pre = None
    for n in ns:
        if (group, N, lam, n) in done:
            print(f"skip N={N} lam={lam} n={n} (done)", flush=True)
            pre = None
            continue
        print(f"=== N={N} lam={lam} 2J={n}", flush=True)
        Eng = SectorPar if os.environ.get('ENGINE') == 'par' else Sector
        S = Eng(N, lam, n, dims, log=lambda s: print(s, flush=True), graviton_algebra=GA, maxlen=maxlen, pre=pre, seed=seed, group=group, maxlen_up=maxlen_up)
        res = S.run()
        res['seed'] = seed
        res['group'] = group
        res['gen_nmax'] = gen_nmax
        pre = S.handoff() if (maxlen_up is None or maxlen_up == maxlen) else None   # handed words must match the cap
        del S
        print("RESULT", json.dumps(res), flush=True)
        with open(out, 'a') as f:
            f.write(json.dumps(res) + "\n")
