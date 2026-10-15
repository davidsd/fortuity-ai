import pickle, time
from fortuity.hwcoh.dims import partition_dims_nf
t = time.time()
pickle.dump(partition_dims_nf(0, 22, group='G2', workers=8), open('dims_g2_22.pkl', 'wb')); print('g2', time.time() - t, flush=True)
pickle.dump(partition_dims_nf(3, 22, group='SO_odd', workers=8), open('dims_so7_22.pkl', 'wb')); print('so7', time.time() - t, flush=True)
