"""Decompose the CCKLL SU(3) threshold representative O (sector (3;4,4,4), Weyl-symmetric subspace) as
O = G + Q(lambda) with G a polynomial in the graviton generators u_n, v_n, w_n (n<=3), and compare with the
explicit polynomial of result R-002 (summary/fortuity-summary.pdf).  Linear algebra over F_p by random evaluation on traceless 3x3
matrices with fermions in an exterior-algebra frame (the machinery of code/fortuity/bmn).
usage: python3 decompose.py <prime> <seed> <npts> <out.json> [check_basis]
"""
import sys, json, time, copy, numpy as np
from collections import defaultdict
from fractions import Fraction
from fortuity.bmn.symdirect import SymDirect
from fortuity.bmn.words import Q_multi, contents_of_sector
from fortuity.bmn.evaluate import Points
from fortuity.bmn.modp import RREFBasis
from gens_util import load_O, poly_from_terms, ratrec

p, seed, npts, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
check_basis = len(sys.argv) > 5
sec, up = (3, 4, 4, 4), (4, 4, 4, 4)
t0 = time.time()
log = lambda *a: print('[%6.0fs]' % (time.time() - t0), *a, flush=True)
D = SymDirect(N=3, gen_nmax=3, seed=seed, verbose=False); D.guard.log = False
k, degs, cpp = D._sector_k(sec)
pts = Points(3, npts, k, degs, seed=seed, p=p)
log('prime', p, 'seed', seed, 'points', npts, 'frame k', k, 'degrees', degs, 'functionals', pts.ncols)
O, meta = load_O('O_cckll.json')
vO = pts.expr_vector(O)
log('O evaluated; nonzero:', bool(np.any(vO)))
dN = json.load(open('../../../data/su3/graviton_class_vanishing_on_diagonal_j24.json'))
from fortuity.bmn.symdirect import symmetrize_expr
GN = symmetrize_expr(poly_from_terms([(t['coeff'], t['monomial']) for t in dN['operator']]))  # Weyl symmetrisation (the object verified in R-002)
vGN = pts.expr_vector(GN)
log('R-002 polynomial (Weyl-symmetrised) evaluated; nonzero:', bool(np.any(vGN)))

# --- Q-exact space of the symmetric subspace: Q of the symmetrised words of the up sector ---
sw_up = D.symwords(up)
log('symmetrised up-words:', len(sw_up))
qexprs = []
for e in sw_up:
    q = defaultdict(int)
    for m, c in e.items():
        for m2, c2 in Q_multi(m).items():
            q[m2] += c * c2
    qexprs.append({m: c for m, c in q.items() if c})
Qrows = np.array([pts.expr_vector(q) if q else np.zeros(pts.ncols, dtype=np.int64) for q in qexprs], dtype=np.int64)
log('Q-image rows evaluated', Qrows.shape)
exact = RREFBasis(pts.ncols, p)
indep = exact.add_batched(Qrows)
sel = np.nonzero(indep)[0]
log('rank of Q-exact space (symmetric subspace) =', exact.rank, '(Table 2 of 2312.16443: 924)')
if check_basis:
    sw = D.symwords(sec)
    eb = RREFBasis(pts.ncols, p)
    for s0 in range(0, len(sw), 256):
        eb.add(np.array([pts.expr_vector(e) for e in sw[s0:s0 + 256]], dtype=np.int64))
    log('rank of all symmetrised words of the sector =', eb.rank, '(Table 2: 1898); functionals', pts.ncols)
    del eb

# --- O is closed (checked in build_O.py) and not exact ---
t = copy.deepcopy(exact); print('O independent of exact space (non-exact):', bool(t.add(vO[None, :])[0]), flush=True); del t
t = copy.deepcopy(exact); print('Sym(R-002 polynomial) non-exact:', bool(t.add(vGN[None, :])[0]), flush=True); del t

# --- graviton monomials (symmetrised) ---
target = np.array(sec); gs = [np.array(s) for s in D.gen_sec]
monos = []
def rec(start, remaining, chosen):
    if not remaining.any():
        monos.append(tuple(chosen)); return
    for i in range(start, len(D.gens)):
        if np.all(gs[i] <= remaining):
            rec(i + 1 if D.gen_par[i] else i, remaining - gs[i], chosen + [i])
rec(0, target, [])
G, nm = D.graviton_rows(sec, pts)
assert nm == len(monos)
log('symmetrised graviton monomials:', nm)
gb = copy.deepcopy(exact)
gind = gb.add_batched(G)
piv = np.nonzero(gind)[0]
log('graviton classes modulo exact =', len(piv), '(R-002: 29)')
red = gb.reduce(vO[None, :])[0]
in_span = not np.any(red)
log('O in span(gravitons) + exact:', in_span)
# f-free gravitons only (no w generators)
uv = [i for i, m in enumerate(monos) if all(D.gens[g][0][0] != 'w' for g in m)]
gb2 = copy.deepcopy(exact); gb2.add_batched(G[uv])
log('f-free graviton classes modulo exact =', gb2.rank - exact.rank, '; O in span(f-free gravitons)+exact:', not np.any(gb2.reduce(vO[None, :])[0]))

# --- explicit coefficients: O = sum_{piv} x_m G_m + sum_{sel} y_w Q(w) ---
M = np.concatenate([G[piv], Qrows[sel]], axis=0) % p
nrow = M.shape[0]
A = np.concatenate([M, np.eye(nrow, dtype=np.int64)], axis=1)
full = RREFBasis(A.shape[1], p); full.add_batched(A)
r = full.reduce(np.concatenate([vO % p, np.zeros(nrow, dtype=np.int64)])[None, :])[0]
assert not np.any(r[:pts.ncols]), 'O not in the span?!'
z = (-r[pts.ncols:]) % p
x, y = z[:len(piv)], z[len(piv):]
log('solved: graviton coefficients on', len(piv), 'pivot monomials; lambda on', len(sel), 'up-words')
# verify the identity at fresh points (same prime)
pts2 = Points(3, 24, k, degs, seed=seed + 1000, p=p)
vO2 = pts2.expr_vector(O)
G2, _ = D.graviton_rows(sec, pts2)
Q2 = np.array([pts2.expr_vector(qexprs[i]) for i in sel], dtype=np.int64)
lhs = (x @ (G2[piv] % p) + y @ (Q2 % p)) % p
log('identity O = G + Q(lambda) holds at 24 fresh points:', bool(np.array_equal(lhs % p, vO2 % p)))
# --- relation to the R-002 polynomial: O - c * G_N exact? ---
rO = exact.reduce(vO[None, :])[0]; rG = exact.reduce(vGN[None, :])[0]
i0 = int(np.nonzero(rG)[0][0]); c = (int(rO[i0]) * pow(int(rG[i0]), p - 2, p)) % p
same_class = bool(np.array_equal((c * rG) % p, rO % p))
log('O == c * Sym(G_N) modulo exact with c =', c, 'mod p; consistent on all functionals:', same_class, '; c rational =', ratrec(c, p))
# --- rational reconstruction of the graviton coefficients ---
xr = [ratrec(int(v), p) for v in x]
log('rational reconstruction of x succeeded for', sum(v is not None for v in xr), 'of', len(xr))
def mono_name(m):
    Lx = 'XYZ'
    out = []
    for g in m:
        nmn = D.gens[g][0]
        if nmn[0] == 'u': out.append('u^{%s}' % ''.join(Lx[i] for i in nmn[2]))
        elif nmn[0] == 'v': out.append('v^{%s}_{%s}' % (''.join(Lx[i] for i in nmn[2]), Lx[nmn[3]]))
        else: out.append('w^{%s}' % ''.join(Lx[i] for i in nmn[2]))
    return out
res = dict(prime=p, seed=seed, npts=npts, ncols=int(pts.ncols), rank_exact=int(exact.rank), n_monomials=int(nm),
           graviton_classes=int(len(piv)), O_in_span=bool(in_span), O_nonexact=True,
           c_mod_p=int(c), c_rational=str(ratrec(c, p)), same_class_as_R002=same_class,
           pivot_monomials=[mono_name(monos[i]) for i in piv],
           x_mod_p=[int(v) for v in x], x_rational=[str(v) if v is not None else None for v in xr],
           lambda_upword_indices=[int(i) for i in sel], lambda_mod_p=[int(v) for v in y],
           note='G = sum_m x_m Sym(m) with Sym = sum over the 6 Weyl images (code convention, generators in code normalisation: u_n = sum over orderings of tr(phi..phi), v_n = (n+1) tr(phi..psi) - trace, w_n = 2 tr(f phi..) + eps psi psi terms); lambda = sum_w y_w Sym(w) over symmetrised words of sector (4;4,4,4) enumerated by SymDirect.symwords')
json.dump(res, open(out, 'w'), indent=1)
log('saved', out)
