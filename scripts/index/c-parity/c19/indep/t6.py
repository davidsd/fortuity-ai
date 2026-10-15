import time, pickle
from itertools import permutations
from fortuity.bmn.twistedindex import SUN, integrate, to_int
t = time.time()
tw, err = to_int(integrate(SUN(3), 22, True)[0])
print('twisted SU(3) q<=22 rounding', err, f'{time.time()-t:.0f}s')
rho = (2, 1, 0)
psign = lambda p: 1 if sum(1 for i in range(3) for j in range(i+1,3) if p[i] > p[j]) % 2 == 0 else -1
SH = [(tuple(rho[i] - rho[w[i]] for i in range(3)), psign(w)) for w in permutations(range(3))]
for lam in [(7,4,4),(8,4,4),(9,6,6),(10,6,6),(9,7,6),(8,5,5)]:
    c = sum(sg * tw.get(tuple(lam[i] + sh[i] for i in range(3)), 0) for sh, sg in SH)
    print(lam, 's =', (-1) ** sum(lam) * c)
pickle.dump(tw, open('su3_twisted_q22_lib.pkl', 'wb'))
