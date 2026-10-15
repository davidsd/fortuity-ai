"""Environment check for the T-032 runs (run on a compute node after setup_env.sh; ~1-2 min).
1) which finite-field kernels load; 2) kernel ranks agree with the pure-numpy fallback on random matrices;
3) a tiny known SU(3) sector pair, (7,4,4) 2J=7,8 of R-007's column, reproduces m, rank and graviton counts."""
import os, sys, json, subprocess, tempfile, time
import numpy as np
from fortuity.bmn.modp import P, RREFBasis, matmul_mod
from fortuity.hwcoh import ffla, ffblas

print(f"FLINT kernel loaded: {ffla.HAVE_FLINT}   BLAS kernel loaded: {ffblas.HAVE_FFBLAS}", flush=True)
rng = np.random.default_rng(0)
A = matmul_mod(rng.integers(0, P, (300, 120)), rng.integers(0, P, (120, 400)))        # rank 120
b = RREFBasis(A.shape[1]); b.add_batched(A % P, 128); r_np = b.rank
r_fl = ffla.rank(A)
r1, r2 = ffblas.rank_pair32(A.astype(np.int32), None)
print(f"random rank test (expect 120): numpy {r_np}, ffla {r_fl}, ffblas {r1}", flush=True)
ok = (r_np == r_fl == r1 == 120)

here = os.path.dirname(os.path.abspath(__file__))
dims = os.path.join(here, 'dims_nf_su3_36.pkl')
if not os.path.exists(dims):
    print('MISSING dims_nf_su3_36.pkl (rsync it from the laptop, or run dims.sbatch)'); sys.exit(1)
out = os.path.join(tempfile.mkdtemp(), 'smoke.jsonl')
env = dict(os.environ, GROUP='SU', MAXLEN='6', MAXLEN_UP='6', GEN_NMAX='3', SEED='1', ENGINE='par')
t = time.time()
subprocess.run([sys.executable, '-m', 'fortuity.hwcoh.run_sectors', '3', dims, out, '7,4,4:7,8'], env=env, check=True,
               stdout=subprocess.DEVNULL)
recs = {json.loads(l)['n']: json.loads(l) for l in open(out)}
got = [(n, recs[n]['m'], recs[n]['r_up'], recs[n]['grav'], recs[n]['check']) for n in (7, 8)]
want = [(7, 747, 72, 1, True), (8, 72, 0, 0, True)]            # from data/su3/bmn_hw_cohomology_j30.json (R-007)
print(f"smoke test (n, m, r_up, grav, check): got {got}, expected {want}  ({time.time() - t:.0f}s)")
ok &= (got == want)
print('ENVIRONMENT OK' if ok else 'ENVIRONMENT PROBLEM: do not start the production runs')
sys.exit(0 if ok else 1)
