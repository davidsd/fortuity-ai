import time, pickle
from weylcount import enumerate_N3
t = time.time(); E = enumerate_N3(22); U = pickle.load(open('my_dup_q22.pkl','rb'))['N3']
print('explicit enumeration N3 |q|<=22 == closed form (== agent Molien):', E == U, f'{time.time()-t:.0f}s')
