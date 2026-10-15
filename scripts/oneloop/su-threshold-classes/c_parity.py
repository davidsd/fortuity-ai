"""Charge conjugation C: X -> -X^T on full-letter multitraces, and checks of the parity selection rules:
  (1) C^2 = 1, [C, Q_0] = 0, [C, Q_1] = 0 formally (random words; Q_1 on derivative-free words, N = 2, 3);
  (2) on Cartan-valued letters C acts as (-1)^Y  (so P = C(-1)^Y = -1 forces vanishing on the Coulomb branch);
  (3) C on the cohomology of the R-007 sector (5 classes) and the R_24 sector (1 class): predicted eigenvalue
      (-1)^Y on graviton classes (P = +1) and P = -1 on the fortuitous R-007 class."""
import sys, random, pickle, math
from collections import defaultdict
import numpy as np
from fastcontents import prime_cache
from fortuity.bmn.modp import P, RREFBasis, matmul_mod
from fortuity.full.words import canon_word, canon_multi, Q_multi
from fortuity.full.q1 import Q1_multi
from fortuity.full.letters import is_ferm
from fortuity.full.sector import words_of, hw_mult_by_degree, frame_k
from fortuity.full.evalg import Frame, nferm
from fortuity.full.hw import hw_vectors


def C_multi(multi):
    sign, new = 1, []
    for w in multi:
        n, f = len(w), sum(1 for L in w if is_ferm(L))
        s2, cw = canon_word(tuple(reversed(w)))
        if not s2:
            return 0, None
        sign *= (-1) ** n * (-1) ** (f * (f - 1) // 2) * s2
        new.append(cw)
    sm, cm = canon_multi(new)
    return (sign * sm, cm) if sm else (0, None)


def C_dict(v):
    out = defaultdict(int)
    for w, c in v.items():
        s, cw = C_multi(w)
        if s:
            out[cw] = (out[cw] + s * c) % P
    return {w: c for w, c in out.items() if c}


def apply(op, v):
    out = defaultdict(int)
    for w, c in v.items():
        for m, c2 in op(w).items():
            out[m] = (out[m] + c * c2) % P
    return {m: c for m, c in out.items() if c}


def modp(d):
    return {k: v % P for k, v in d.items() if v % P}


def main():
    random.seed(7)
    # (1) formal commutation on random words from a few sectors
    secs = [((3, 3, 8, 8, 8), 8), ((5, 5, 14, 8, 8), 9), ((4, 4, 14, 8, 8), 10), ((3, 3, 8, 8, 8), 9)]
    bad = {'C2': 0, 'Q0': 0, 'Q1_N2': 0, 'Q1_N3': 0}
    ntest = 0
    for sec, Y in secs:
        prime_cache(sec, Y)
        ws = words_of(sec, Y, 6)
        for w in random.sample(ws, 60):
            ntest += 1
            s, cw = C_multi(w)
            s2, cw2 = C_multi(cw)
            if not (cw2 == w and s * s2 == 1):
                bad['C2'] += 1
            v = {w: 1}
            if modp(C_dict(apply(lambda x: Q_multi(x), v))) != modp(apply(lambda x: Q_multi(x), C_dict(v))):
                bad['Q0'] += 1
            if all(L[0] == 0 and L[1] == 0 for n in w for L in n):
                for N in (2, 3):
                    if modp(C_dict(apply(lambda x: Q1_multi(x, N), v))) != modp(apply(lambda x: Q1_multi(x, N), C_dict(v))):
                        bad[f'Q1_N{N}'] += 1
    print(f"(1) {ntest} random words: failures {bad}", flush=True)

    # (2) Cartan action: C(w) = (-1)^Y w on diagonal letters
    prime_cache((5, 5, 14, 8, 8), 9)
    ws = random.sample(words_of((5, 5, 14, 8, 8), 9, 6), 300)
    letters = sorted({L for w in ws for n in w for L in n} | {L for w in ws for n in C_multi(w)[1] for L in n})
    okc = 0
    for d in sorted({nferm(w) for w in ws}):
        sub = [w for w in ws if nferm(w) == d]
        fr = Frame(3, 6, max(d + 4, 6) if d else 0, [d], letters, seed=3 + d, cartan=True)
        X = fr.eval_words(sub, dtype=np.int64)
        XC = fr.eval_words([C_multi(w)[1] for w in sub], dtype=np.int64)
        for i, w in enumerate(sub):
            s = C_multi(w)[0]
            Yw = sum(len(n) for n in w)
            okc += int(np.array_equal((s * XC[i]) % P, ((-1) ** Yw * X[i]) % P))
    print(f"(2) C = (-1)^Y on Cartan-valued letters: {okc}/{len(ws)} words", flush=True)


    # (3) C on cohomology classes
    def c_matrix(case, N, SEC, Y, ML):
        R = pickle.load(open(f'classes/{case}/reps.pkl', 'rb'))
        vecs = R['vecs']
        UP = (SEC[0] + 1, SEC[1] + 1) + SEC[2:]
        prime_cache(UP, Y - 1)
        wu = words_of(UP, Y - 1, ML)
        ex = []
        for widx, H in hw_vectors(wu):
            for r in range(H.shape[0]):
                ex.append(apply(Q_multi, {wu[int(widx[c])]: int(H[r, c]) for c in range(H.shape[1]) if H[r, c]}))
        cvecs = [C_dict(v) for v in vecs]
        m_d = hw_mult_by_degree(N, SEC, Y)
        allw = sorted({w for d in ex + vecs + cvecs for w in d})
        lets = sorted({L for w in allw for n in w for L in n})
        frames = []
        for d, md in sorted(m_d.items()):
            k = frame_k(N, d)
            frames.append((d, Frame(N, max(8, math.ceil(1.5 * (md + 64) / math.comb(k, d))), k, [d], lets, seed=91 + d)))
        ncols = sum(fr.ncols for _, fr in frames)
        idx = {w: i for i, w in enumerate(allw)}
        X = np.zeros((len(allw), ncols), dtype=np.int64)
        deg = np.array([nferm(w) for w in allw])
        col = 0
        for d, fr in frames:
            ii = np.flatnonzero(deg == d)
            if ii.size:
                X[ii, col:col + fr.ncols] = fr.eval_words([allw[i] for i in ii], dtype=np.int64)
            col += fr.ncols

        def ev(ds):
            M = np.zeros((len(ds), ncols), dtype=np.int64)
            for r, d in enumerate(ds):
                if d:
                    rr = np.array([idx[w] for w in d]); cc = np.array([c % P for c in d.values()], dtype=np.int64)
                    M[r] = matmul_mod(cc[None, :], X[rr])[0]
            return M
        E = RREFBasis(ncols); E.add_batched(ev(ex))
        Rv = E.reduce(ev(vecs)); Rc = E.reduce(ev(cvecs))
        # solve Rc = M Rv (M: h x h) using h independent columns of Rv
        h = len(vecs)
        b = RREFBasis(h)
        cols = []
        RvT = Rv.T % P
        for j in range(ncols):
            if b.add(RvT[j:j + 1])[0]:
                cols.append(j)
            if len(cols) == h:
                break
        A = Rv[:, cols] % P
        Bm = Rc[:, cols] % P
        import sympy
        Ms = (sympy.Matrix(Bm.tolist()) * sympy.Matrix(A.tolist()).inv_mod(P)).applyfunc(lambda x: x % P)
        Mi = np.array([[int(x) if int(x) < P // 2 else int(x) - P for x in row] for row in Ms.tolist()])
        ok = np.array_equal((Mi @ (Rv % P)) % P, Rc % P)
        ev_ = sympy.Matrix(Mi.tolist()).eigenvals()
        print(f"(3) {case}: C matrix on the {h} classes (rows: C v_k in the basis v_j, mod exact) =\n{Mi}\n"
              f"    consistent on all functionals: {ok}; eigenvalues {ev_}", flush=True)


    c_matrix('su3_j30', 3, (5, 5, 14, 8, 8), 10, 6)
    c_matrix('su3_j24', 3, (3, 3, 8, 8, 8), 9, 6)


if __name__ == '__main__':
    main()
