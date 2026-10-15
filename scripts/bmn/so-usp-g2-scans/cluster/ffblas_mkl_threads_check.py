import importlib.util, time, numpy as np, sys
spec = importlib.util.spec_from_file_location("ffblas_new", sys.argv[1]); ff = importlib.util.module_from_spec(spec); spec.loader.exec_module(ff)
from fortuity.bmn.modp import P
print("HAVE_FFBLAS", ff.HAVE_FFBLAS, "MKL hook", ff._mkl_set is not None)
rng = np.random.default_rng(1)
A = rng.integers(0, P, (3000, 2500)).astype(np.int32)
for t in (1, 8):
    ff.set_threads(t); t0 = time.time(); r = ff.rank_pair32(A)[0]; print(f"threads={t}: rank {r} in {time.time()-t0:.2f}s")
