"""Robustness of the graviton-mod-exact count in a sector: different random seed, and generators up to n=4."""
import sys
from fortuity.bmn.symdirect import SymDirect
sec = tuple(int(x) for x in sys.argv[1].split(','))
for seed, nmax in [(4242, 3), (777, 4)]:
    D = SymDirect(N=3, gen_nmax=nmax, seed=seed); D.guard.log = False
    r = D.analyse(sec)
    print(f'seed={seed} gen_nmax={nmax}: basis={r["basis"]} closed={r["closed"]} exact={r["exact"]} coh={r["coh"]} grav={r["grav"]} BH={r["BH"]} nmono={r["nmono"]}', flush=True)
