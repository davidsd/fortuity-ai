"""Instrumented re-run of GravitonCounter (read-only use of the repo code).
usage: rerun.py N JMAX SEED PRIME OUT.pkl
Records per sector: dim, number of evaluation functionals actually used (final nps*cpp), rank."""
import sys, time, pickle, math
import fortuity.bmn.gravindex as gi
import fortuity.bmn.modp as modp
N, jmax, seed, prime, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
assert prime < 2**26
# patch the prime everywhere it is bound as a default
gi.P = prime
modp.RREFBasis.__init__.__defaults__ = (prime, 256)
d = list(gi.MesonSystem.__init__.__defaults__); d[1] = prime; gi.MesonSystem.__init__.__defaults__ = tuple(d)
last = {}
_orig_init = modp.RREFBasis.__init__
def _init(self, ncols, p=prime, capacity=256):
    _orig_init(self, ncols, p, capacity)
    last['ncols'] = ncols; last['obj'] = self
modp.RREFBasis.__init__ = _init
gi.RREFBasis = modp.RREFBasis
info = {}
_orig_cs = gi.GravitonCounter.count_sector
def _cs(self, sec):
    if sec in self.dims:
        return self.dims[sec]
    last.clear()
    r = _orig_cs(self, sec)
    info[sec] = (r, last.get('ncols'), last['obj'].rank if 'obj' in last else None)
    last.clear()
    return r
gi.GravitonCounter.count_sector = _cs
t0 = time.time()
G = gi.GravitonCounter(N, jmax, verbose=True, npts=4096, seed=seed)
assert G.p == prime and G.ms.p == prime
print('npts', G.npts, 'k', G.k, 'mesons', len(G.mes), 'p', prime, 'seed', seed, flush=True)
dims = G.run()
pickle.dump(dict(N=N, jmax=jmax, seed=seed, p=prime, dims=dims, info=info), open(out, 'wb'))
print(f'done {len(dims)} sectors {time.time()-t0:.0f}s', flush=True)
