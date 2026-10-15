"""Explicit multi-graviton class in SU(3) sector (3;4,4,4) that vanishes on diagonal matrices but is not Q-exact.
Works with UNsymmetrized graviton monomials (plain products of generator components), so the result is a
plain polynomial in u_n, v_n, w_n."""
import sys, copy, numpy as np
from fortuity.bmn.symdirect import SymDirect
from fortuity.bmn.evaluate import Points
from fortuity.bmn.modp import RREFBasis, matmul_mod, inv_mod, P
sec = (3, 4, 4, 4)
D = SymDirect(N=3, gen_nmax=3, symmetrize=False); D.guard.log = False
k, degs, cpp = D._sector_k(sec)
# generic points and Q-exact basis of the (unsymmetrized) up sector
up = (sec[0] + 1,) + sec[1:]
rQup = D.rankQ(up); qb, qpts = D._Qrows[up]
print('rank Q(up) =', rQup, 'cols', qpts.ncols, flush=True)
Gg, nm = D.graviton_rows(sec, qpts)            # (nm, cols) generic
print('graviton monomials', nm, flush=True)
# monomial names
names = []
target = np.array(sec); gs = [np.array(s) for s in D.gen_sec]
def rec(start, remaining, chosen):
    if not remaining.any(): names.append(tuple(chosen)); return
    for i in range(start, len(D.gens)):
        if np.all(gs[i] <= remaining):
            rec(i + 1 if D.gen_par[i] else i, remaining - gs[i], chosen + [i])
rec(0, target, [])
assert len(names) == nm
# diagonal points (adaptive)
n = 16; prev = None
while True:
    pd = Points(3, n, k, degs, seed=1234 + n, diagonal=True)
    Gd, _ = D.graviton_rows(sec, pd)
    rd = int(RREFBasis(pd.ncols).add_batched(Gd).sum())
    if prev == rd and rd < pd.ncols - 16: break
    prev = rd; n *= 2
print('rank on diagonal =', rd, flush=True)
# pick a monomial basis of the generic graviton row space
gb = RREFBasis(qpts.ncols); indep = gb.add_batched(Gg); basis_idx = np.nonzero(indep)[0]
print('generic rank of graviton rows =', len(basis_idx), flush=True)
Bg = Gg[basis_idx]; Bd = Gd[basis_idx]
# null space of Bd^T c = 0 (combinations of basis monomials vanishing on the diagonal): solve via RREF of Bd (rows = monomials)
# we want c (len r) with c @ Bd = 0. Compute RREF of Bd^T ... simpler: rank-nullity via random combinations:
r = len(basis_idx)
# RREF of Bd (r x cols): rows dependent -> the elimination gives relations; use the transposed system:
# find c with c@Bd=0 <=> Bd^T c = 0. Build augmented matrix [Bd | I] and row-reduce: rows that become zero on the left give c on the right.
aug = np.concatenate([Bd % P, np.eye(r, dtype=np.int64)], axis=1)
M = aug.copy(); rows, ncols_d = M.shape[0], Bd.shape[1]; piv_row = 0
for c in range(ncols_d):
    pr = None
    for i in range(piv_row, rows):
        if M[i, c]: pr = i; break
    if pr is None: continue
    M[[piv_row, pr]] = M[[pr, piv_row]]
    M[piv_row] = (M[piv_row] * inv_mod(M[piv_row, c])) % P
    for i in range(rows):
        if i != piv_row and M[i, c]:
            M[i] = (M[i] - M[i, c] * M[piv_row]) % P
    piv_row += 1
    if piv_row == rows: break
null = M[piv_row:, ncols_d:]           # each row: coefficients c over basis monomials with c@Bd = 0
print('null space dim (vanish on diagonal) =', null.shape[0], flush=True)
# test each null vector for non-exactness; collect the non-exact ones
vecs = matmul_mod(null % P, Bg % P)     # generic evaluation of the combinations
tb = copy.deepcopy(qb); ne = tb.add_batched(vecs)
print('non-exact combinations among them:', int(ne.sum()), flush=True)
idx = np.nonzero(ne)[0][0]
c = null[idx]
# lift coefficients to small integers if possible (they are mod P): show as symmetric residues
def sym(x): return int(x) if x <= P // 2 else int(x) - P
terms = [(sym(c[i]), names[basis_idx[i]]) for i in range(r) if c[i]]
print('representative: %d terms' % len(terms))
for coef, mono in terms[:60]:
    print(coef, [D.gens[g][0] for g in mono])
np.save('results/su3_graviton_class_coeffs.npy', c)
import pickle; pickle.dump(dict(terms=terms, names=[D.gens[g][0] for g in range(len(D.gens))]), open('results/su3_graviton_class.pkl', 'wb'))
