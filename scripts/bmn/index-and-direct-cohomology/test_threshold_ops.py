import numpy as np
from collections import defaultdict
from fortuity.bmn.words import *
from fortuity.bmn.direct import Direct
from fortuity.bmn.modp import rank_mod, RREFBasis

X,Y,Z,p1,p2,p3 = 0,1,2,3,4,5
def E(*terms):
    out = defaultdict(int)
    for c, multi in terms:
        add_term(out, multi, c)
    return out
def tr(*ls): return tuple(ls)
# O = (tr(p1 X) - tr(p2 Y)) tr(p3 X) tr(p2 p1 p1) + cyclic
def term(X,Y,Z,p1,p2,p3):
    return [(1, (tr(p1,X), tr(p3,X), tr(p2,p1,p1))), (-1, (tr(p2,Y), tr(p3,X), tr(p2,p1,p1)))]
O = E(*(term(X,Y,Z,p1,p2,p3) + term(Y,Z,X,p2,p3,p1) + term(Z,X,Y,p3,p1,p2)))
print("O terms:", len(O), " sector:", expr_sector(O))
QO = Q_expr(O)
D = Direct(N=2, maxlen=3, gen_nmax=2, verbose=False)
sec = expr_sector(O)
pts = D.points(sec)
# Q-closed?  evaluate QO at points of sector (4;4,4,4)
low = (sec[0]-1,)+sec[1:]
if QO:
    v = D.points(low).expr_vector(QO); print("Q O nonzero components:", int(np.count_nonzero(v)))
else: print("Q O = 0 identically as expression")
# non-exact?  compare rank of Q-images of up-sector words with and without O
up = (sec[0]+1,)+sec[1:]
rup = D.rankQ(up); M = D._Qimg_rows[up]
vO = pts.expr_vector(O)
print("O nonzero:", bool(np.any(vO)), " rank(exact)=", rup, " rank(exact+O)=", rank_mod(np.vstack([M, vO[None,:]])))
# descendants Q^m_+ O
for m in range(3):
    d = Qplus_expr(O, m)
    if not d:
        print(f"Q^{m+1}_+ O = 0 as an expression"); continue
    dsec = expr_sector(d); print(f"Q^{m+1}_+ O: sector {dsec}, terms {len(d)}")
    # is it Q-closed?
    dl = (dsec[0]-1,)+dsec[1:]
    qd = Q_expr(d)
    print("   Q(Q^m_+ O) zero?", (not qd) or not np.any(D.points(dl).expr_vector(qd)))
    P2 = D.points(dsec)
    vd = P2.expr_vector(d)
    dup = (dsec[0]+1,)+dsec[1:]
    r_ex = D.rankQ(dup); Mex = D._Qimg_rows[dup]
    r1 = rank_mod(np.vstack([Mex, vd[None,:]]))
    print(f"   nonzero: {bool(np.any(vd))}; exact-rank {r_ex}, with descendant {r1}  -> exact? {r1==r_ex}")
    G, nm = D.graviton_matrix(dsec)
    r2 = rank_mod(np.vstack([Mex, G])); r3 = rank_mod(np.vstack([Mex, G, vd[None,:]]))
    print(f"   graviton+exact rank {r2}, with descendant {r3} -> graviton mod exact? {r3==r2}")
