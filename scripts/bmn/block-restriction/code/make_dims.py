"""Exact n_f-refined sector dimensions (hw self-check input) for the groups used in T-028; cached as pickles.
usage: python3 make_dims.py GROUP N JMAX TWOJ_MAX WORKERS"""
import sys, time, pickle, os
from fortuity.hwcoh.dims import partition_dims_nf
group, N, jmax, tj, w = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'runs', 'cache', f'dims_{group}{N}_j{jmax}_2J{tj}.pkl')
t = time.time()
d = partition_dims_nf(N, jmax, group=group, workers=w, twoJ_max=(tj if group != 'SU' else None), progress=False)
pickle.dump(d, open(out, 'wb'))
print(f"{group} N={N} j<={jmax} 2J<={tj}: {len(d)} entries, {time.time()-t:.0f}s -> {out}", flush=True)
