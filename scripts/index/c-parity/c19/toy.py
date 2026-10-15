# Toy: per-weight underestimate (Weyl-symmetric, nonneg) can OVERestimate a hw multiplicity -> T_comp < T_true.
from fortuity.bmn.twistedindex import hw_multiplicities
import itertools
Q = 2
# true Coulomb weights = character of Sym^2(3) (hw (2,0,0)) : weights (2,0,0)-orbit and (1,1,0)-orbit, each mult 1
true = {}
for w in set(itertools.permutations((2,0,0))) | set(itertools.permutations((1,1,0))):
    true[w] = 1
comp = dict(true)
for w in set(itertools.permutations((2,0,0))):   # random evaluation misses the (2,0,0) orbit
    comp[w] = 0
Ht, Hc = hw_multiplicities(true, Q), hw_multiplicities(comp, Q)
print('D_true hw:', Ht, ' D_comp hw:', Hc)
# with s*I_C(lam) held fixed, T = s*I_C - D
for lam in [(2,0,0), (1,1,0)]:
    print(lam, 'D_comp - D_true =', Hc.get(lam,0) - Ht.get(lam,0), '=> T_true - T_comp =', Hc.get(lam,0) - Ht.get(lam,0))
