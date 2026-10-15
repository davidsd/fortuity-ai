"""Does the two-loop vertex U (q2bmn.py) respect the trace relations of SU(N)?

U is defined on formal multitrace words; test_q2.py checks nilpotency as a formal identity (N a parameter).  At rank N
words are not independent, and an operator on gauge-invariant operators must map every relation to a relation.
For a small derivative-free sector at rank N:
  1. all multitraces of the sector (traces of any length), evaluated at random su(N) points with Grassmann frames;
  2. relations = left null space of the value matrix, per fermion degree, confirmed at fresh points;
  3. U, Q_1 and Q_0 applied to random combinations of relations must vanish at fresh points;
     control: the same operators on random combinations of words (not relations) must not vanish.
usage: python3 test_q2_tracerel.py [N 2J1 2J2 2q1 2q2 2q3 Y]"""
import math, random, sys
args = [int(x) for x in sys.argv[1:]]
sys.argv = ['x']
import numpy as np
import fullpar
from fullpar import frame_k, Frame, nferm
import q1d, q2bmn
from fortuity.full.sector import words_of
from fortuity.full.words import Q_multi
from fortuity.bmn.modp import P

fullpar.CSEC = 0


def nullspace_left(M):
    M = np.asarray(M, dtype=np.int64) % P
    r, c = M.shape
    A = np.concatenate([M, np.eye(r, dtype=np.int64)], axis=1)
    row = 0
    for col in range(c):
        if row == r:
            break
        nz = np.flatnonzero(A[row:, col])
        if nz.size == 0:
            continue
        piv = row + int(nz[0])
        if piv != row:
            A[[row, piv]] = A[[piv, row]]
        A[row] = A[row] * pow(int(A[row, col]), P - 2, P) % P
        rows = np.flatnonzero(A[:, col])
        rows = rows[rows != row]
        if rows.size:
            A[rows] = (A[rows] - A[rows, col:col + 1] * A[row:row + 1]) % P
        row += 1
    return A[row:, c:]


def values(N, vec, npts, seed):
    """values of the word vector {word: coeff} at fresh points, all fermion degrees: list of arrays"""
    out = []
    words = list(vec)
    degs = [nferm(w) for w in words]
    letters = sorted({L for w in words for n in w for L in n})
    for d in sorted(set(degs)):
        k = frame_k(N, d)
        if d and math.comb(k, d) == 0:
            continue
        sel = [w for w, dd in zip(words, degs) if dd == d]
        fr = Frame(N, npts, k, [d], letters, seed=seed + 31 * d)
        X = fr.eval_words(sel, dtype=np.int64) % P
        c = np.array([vec[w] % P for w in sel], dtype=object)
        out.append(np.array([int(sum(int(ci) * int(x) for ci, x in zip(c, X[:, j])) % P) for j in range(X.shape[1])]))
    return out


def app(op, v):
    out = {}
    for w, c in v.items():
        for m, c2 in op(w).items():
            out[m] = (out.get(m, 0) + c * c2) % P
    return {m: c for m, c in out.items() if c}


def run(N, sec, Y):
    words = words_of(tuple(sec), Y, Y)
    assert all(L[0] == 0 and L[1] == 0 for w in words for n in w for L in n), "sector with derivatives"
    degs = np.array([nferm(w) for w in words])
    letters = sorted({L for w in words for n in w for L in n})
    rng = random.Random(1)
    ops = (('U', lambda m: q2bmn.U_multi(m, N)), ('Q_1', lambda m: q1d.Q1d_multi(m, N)), ('Q_0', lambda m: dict(Q_multi(m))))
    print(f"N = {N}, sector {tuple(sec)}, Y = {Y}: {len(words)} multitraces (traces of any length)", flush=True)
    tot_rel = 0
    for d in sorted(set(degs.tolist())):
        sel = [words[i] for i in np.flatnonzero(degs == d)]
        k = frame_k(N, d)
        if d and math.comb(k, d) == 0:
            continue
        fr = Frame(N, max(4, 2 * len(sel) // max(1, math.comb(k, d)) + 2), k, [d], letters, seed=100 + d)
        X = fr.eval_words(sel, dtype=np.int64) % P
        R = nullspace_left(X)
        rank = len(sel) - R.shape[0]
        tot_rel += R.shape[0]
        if not R.shape[0]:
            print(f"  degree {d}: {len(sel)} words, rank {rank}: no relations", flush=True)
            continue
        ntest = min(12, R.shape[0])
        res = {name: [0, 0] for name, _ in ops}        # [nonzero on relations, nonzero on controls]
        nontriv = {name: 0 for name, _ in ops}
        fresh_bad = 0
        for t in range(ntest):
            coef = [rng.randrange(P) for _ in range(R.shape[0])]
            r = {}
            for ci, row in zip(coef, R):
                for j in np.flatnonzero(row):
                    r[sel[j]] = (r.get(sel[j], 0) + ci * int(row[j])) % P
            r = {w: c for w, c in r.items() if c}
            fresh_bad += any(v.any() for v in values(N, r, 6, 500 + t))
            ctrl = {w: rng.randrange(1, P) for w in rng.sample(sel, min(len(sel), max(3, len(r))))}
            for name, op in ops:
                img = app(op, r)
                nontriv[name] += bool(img)
                res[name][0] += any(v.any() for v in values(N, img, 6, 700 + t)) if img else 0
                cimg = app(op, ctrl)
                res[name][1] += any(v.any() for v in values(N, cimg, 6, 900 + t)) if cimg else 0
        print(f"  degree {d}: {len(sel)} words, rank {rank}, {R.shape[0]} relations; {ntest} random relations: nonzero at "
              f"fresh points {fresh_bad}; " + '; '.join(f"{name}(relation) nonzero {res[name][0]} (image nontrivial as "
              f"a word vector {nontriv[name]}, control nonzero {res[name][1]})" for name, _ in ops), flush=True)
    return tot_rel


if __name__ == '__main__':
    cases = [tuple(args)] if args else [(2, 4, 4, 6, 6, 6, 5), (2, 5, 5, 8, 6, 6, 5), (3, 5, 5, 8, 8, 6, 6)]
    for c in cases:
        run(c[0], c[1:6], c[6])
