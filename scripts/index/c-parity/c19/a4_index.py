import time, pickle, sys
from fortuity.bmn.twistedindex import SUN, integrate, to_int
Q=int(sys.argv[1])
t0=time.time()
pl, e1 = to_int(integrate(SUN(3), Q, False)[0])
tw, e2 = to_int(integrate(SUN(3), Q, True)[0])
print('done', time.time()-t0, 'rounding', e1, e2, flush=True)
pickle.dump(dict(pl=pl, tw=tw), open(f'su3_index_q{Q}.pkl','wb'))
