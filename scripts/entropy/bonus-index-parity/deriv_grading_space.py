"""T-038 derivation: dimension of the space of per-letter additive gradings h(L) preserved by Q_0,
i.e. h(L) = h(L1) + h(L2) for every term (L1, L2) of Q_letter(L) (fortuity.full.letters, read-only use).
Expected: 5, spanned by sum a, sum b, n_1, n_2, n_3 (the five KKLP exponents of arXiv:2606.27955 eq. 4.1)."""
import sys, time
import sympy as sp
import os  # repo root: $REPO, or three levels up from scripts/entropy/bonus-index-parity/
sys.path.insert(0, os.path.join(os.environ.get('REPO') or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')), 'code'))
from fortuity.full.letters import letters_up_to, Q_letter
t0 = time.time()
for AM in (3, 4):
    Ls = letters_up_to(AM, AM)
    idx = {L: i for i, L in enumerate(Ls)}
    rows = []
    for L in Ls:
        for c, L1, L2 in Q_letter(L):
            r = [0] * len(Ls)
            r[idx[L]] += 1; r[idx[L1]] -= 1; r[idx[L2]] -= 1
            rows.append(r)
    M = sp.Matrix(rows)
    ns = M.nullspace()
    # check that the five KKLP gradings are in the null space and span it
    kk = [[L[0] for L in Ls], [L[1] for L in Ls]] + [[1 if j in L[2] else 0 for L in Ls] for j in range(3)]
    K = sp.Matrix(kk).T
    span_rank = sp.Matrix.hstack(K, *ns).rank()
    print(f'a,b <= {AM}: {len(Ls)} letters, {len(rows)} constraints, null space dim = {len(ns)}, '
          f'KKLP gradings in null space: {all(v == 0 for v in (M * K))}, rank[KKLP | null] = {span_rank}')
print('time %.1fs' % (time.time() - t0))
