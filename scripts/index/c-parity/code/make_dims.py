"""Exact n_f-refined Haar dims (all 2J, or 2J <= TWOJ_MAX) for the hw self-check of full columns (T-029).
usage: python3 make_dims.py GROUP N JMAX TWOJ_MAX(or 'all') WORKERS   -> runs/cache/dims_{GROUP}{N}_j{JMAX}_2J{TWOJ_MAX}.pkl"""
import sys, time, pickle, os
from fortuity.hwcoh.dims import partition_dims_nf
group, N, jmax, tj, w = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], int(sys.argv[5])
tjv = None if tj == 'all' else int(tj)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'runs', 'cache', f'dims_{group}{N}_j{jmax}_2J{tj}.pkl')
t = time.time()
d = partition_dims_nf(N, jmax, group=group, workers=w, twoJ_max=(tjv if group != 'SU' else None), progress=True)
pickle.dump(d, open(out + '.tmp', 'wb'))
os.replace(out + '.tmp', out)
print(f"{group} N={N} j<={jmax} 2J<={tj}: {len(d)} entries, {time.time()-t:.0f}s -> {out}", flush=True)
