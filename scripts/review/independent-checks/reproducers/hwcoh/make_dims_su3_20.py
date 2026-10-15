"""Exact SU(3) dimension table to j=20 used by t_classreps.py, t_dimsrange.py (writes dims_su3_20.pkl, ~65 KB)."""
import pickle
from fortuity.hwcoh.dims import partition_dims_nf
pickle.dump(partition_dims_nf(3, 20), open('dims_su3_20.pkl', 'wb'))
