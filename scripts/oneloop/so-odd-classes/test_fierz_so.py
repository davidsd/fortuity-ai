"""T-019 (SO/Sp part): checks of the gauge structure of Q_1 that the {Q_0,Q_1} test cannot see.
(1) Direct Fierz check: for bosonic letters (real matrices, no Grassmann signs), the value of
    sum_C [T_C, X_i] (x) [T_C, Y_j] inside a multitrace, computed with an explicit basis T_C of the algebra, must equal
    the value of q1_full._fierz's rejoined traces (up to one overall constant per group).
(2) Choi-Lee check (arXiv:2511.09519): for so(n), Q_1 tr(phi_1 psi_1) is proportional to (n-2) tr(lam+ lam-).
Run: cd scripts/oneloop/so-odd-classes && PYTHONPATH=../../../code python test_fierz_so.py"""
import itertools
import numpy as np
from fortuity.full.q1 import _fierz, Q1_expr
from fortuity.full.evalg import Frame

rng = np.random.default_rng(3)


def so_basis(n):
    B = []
    for a in range(n):
        for b in range(a + 1, n):
            T = np.zeros((n, n)); T[a, b], T[b, a] = 1, -1
            B.append(T)
    return B


def su_basis(n):                                    # a basis orthonormal for tr(XY) up to the traceless projection
    B = []
    for a in range(n):
        for b in range(n):
            E = np.zeros((n, n)); E[a, b] = 1
            B.append(E - np.eye(n) * (a == b) / n)
    return B                                        # overcomplete by one direction (the trace part cancels in commutators)


def trace_val(necks, mats):
    v = 1.0
    for w in necks:
        M = np.eye(mats[w[0]].shape[0])
        for k in w:
            M = M @ mats[k]
        v *= np.trace(M)
    return v


def direct(multi_ids, i, j, mats, basis, dual):
    """sum_C over the basis of the multitrace value with slot i -> [T_C, X_i], slot j -> [T^C, X_j] (dual basis)."""
    tot = 0.0
    for T, Td in zip(basis, dual):
        m = dict(mats)
        m[i] = T @ mats[i] - mats[i] @ T
        m[j] = Td @ mats[j] - mats[j] @ Td
        tot += trace_val(multi_ids, m)
    return tot


def via_fierz(multi, i, j, mats, N):
    """Evaluate _fierz's output with bosonic stand-in letters: we pass letter ids and read back the id layout."""
    # build a multitrace of dummy bosonic letters (0,0,(k,)) labelled by instance id through a lookup
    loc = [(a, u) for a, w in enumerate(multi) for u in range(len(w))]
    n = len(loc)
    par = [0] * n
    rest = [k for k in range(n) if k not in (i, j)]
    return loc, rest


if __name__ == "__main__":
    # (1) Fierz: compare the two completeness relations directly on index contractions
    for group, n in (('SO_odd', 5), ('SO_odd', 7), ('SU', 4)):
        if group == 'SO_odd':
            basis = so_basis(n); dual = [-T for T in basis]          # sum_C T_ab (-T)_cd = delta_ad delta_bc - delta_ac delta_bd
        else:
            basis = su_basis(n); dual = [T.T for T in basis]
        S = sum(np.einsum('ab,cd->abcd', T, Td) for T, Td in zip(basis, dual))
        d = np.eye(n)
        target = np.einsum('ad,bc->abcd', d, d) - (np.einsum('ac,bd->abcd', d, d) if group == 'SO_odd'
                                                    else np.einsum('ab,cd->abcd', d, d) / n)
        print(f"{group} n={n}: completeness relation as used in q1_full: {np.allclose(S, target)}")
    # (2) Choi-Lee: Q_1 tr(phi_1 psi_1) vs tr(lam+ lam-) at random so(n) points
    phi1, psi1 = (0, 0, (0,)), (0, 0, (1, 2))
    lp, lm = (1, 0, ()), (0, 1, ())
    for N in (2, 3, 4):
        n = 2 * N + 1
        q = Q1_expr({((phi1, psi1),): 1}, n, 'SO_odd')
        words = list(q)
        fr = Frame(N, 8, 4, [2], sorted({L for m in words for w in m for L in w} | {lp, lm}), seed=9, group='SO_odd')
        vq = np.zeros(fr.ncols, dtype=np.int64)
        M = fr.eval_words(words, dtype=np.int64)
        for c, row in zip([q[m] for m in words], M):
            vq = (vq + c * row) % fr.p
        vl = fr.eval_words([((lp, lm),)], dtype=np.int64)[0] % fr.p
        # ratio vq / vl (mod p) must be a constant; print it as a small integer
        k = next(t for t in range(len(vl)) if vl[t])
        r = (vq[k] * pow(int(vl[k]), fr.p - 2, fr.p)) % fr.p
        const = np.all((vq - r * vl) % fr.p == 0)
        rs = r if r < fr.p // 2 else r - fr.p
        print(f"SO({n}): Q_1 tr(phi1 psi1) = {rs} x tr(lam+ lam-)  (proportional: {const}); words in Q_1: {words}")


def fierz_direct_check(group, n, trials=60, seed=1):
    """(3) q1_full._fierz vs an explicit basis sum, for bosonic stand-in letters (distinct labels (a,0,(0,)), so
    canonicalisation only rotates and there are no Grassmann signs). Random 1- and 2-trace words, random slots i<j."""
    r = np.random.default_rng(seed)
    if group == 'SO_odd':
        basis = so_basis(n); dual = [-T for T in basis]
        rand = lambda: (lambda M: M - M.T)(r.normal(size=(n, n)))
    else:
        basis = su_basis(n); dual = [T.T for T in basis]
        rand = lambda: (lambda M: M - np.trace(M) / n * np.eye(n))(r.normal(size=(n, n)))
    ok = True
    for t in range(trials):
        ntr = r.integers(1, 3)
        lens = [int(r.integers(2, 5)) for _ in range(ntr)]
        lab, multi = 0, []
        for L in lens:
            multi.append(tuple((lab + u, 0, (0,)) for u in range(L)))
            lab += L
        nlet = lab
        i, j = sorted(r.choice(nlet, 2, replace=False))
        LD, LE = (100, 0, (0,)), (101, 0, (0,))
        mats = {(a, 0, (0,)): rand() for a in range(nlet)}
        mats[LD], mats[LE] = rand(), rand()
        loc = [(a, u) for a, w in enumerate(multi) for u in range(len(w))]
        rest = [k for k in range(nlet) if k not in (i, j)]
        res = _fierz(tuple(multi), loc, i, j, LD, LE, rest, n, [0] * nlet, group)
        v1 = sum(s * np.prod([np.trace(np.linalg.multi_dot([mats[L] for L in w]) if len(w) > 1 else mats[w[0]])
                              for w in m]) for m, s in res)
        # direct: instance ids -> matrices; slot i holds [T, X_LD], slot j holds [T^dual, X_LE]
        ids, k = [], 0
        for w in multi:
            ids.append(list(range(k, k + len(w)))); k += len(w)
        base = {a: mats[(a, 0, (0,))] for a in range(nlet)}
        base[i], base[j] = mats[LD], mats[LE]
        v2 = direct(ids, i, j, base, basis, dual)
        if not np.isclose(v1, v2, rtol=1e-8, atol=1e-8 * (1 + abs(v2))):
            ok = False
            print("MISMATCH", group, multi, i, j, v1, v2)
            break
    return ok


if __name__ == "__main__":
    for group, n in (('SO_odd', 5), ('SO_odd', 7), ('SU', 4), ('SU', 3)):
        print(f"{group} n={n}: _fierz equals the explicit basis sum on 60 random 1-/2-trace words: {fierz_direct_check(group, n)}")


def fierz_grassmann_check(group, N, trials=40, seed=2):
    """(4) As (3) but with fermionic letters, evaluated in a Grassmann frame (evalg_full.Frame) over F_p: checks the
    reversal signs of the so(n) term. The direct side replaces slot i by [T_C, X_LD] and slot j by [T^C, X_LE] and is
    reordered to _fierz's reference Grassmann order (LD, LE, rest)."""
    from fortuity.full.q1 import _reorder_sign
    r = np.random.default_rng(seed)
    from fortuity.hwcoh.evalg import matrix_size
    n = matrix_size(group, N)
    if group == 'SO_odd':
        basis = []
        for a in range(n):
            for b in range(a + 1, n):
                T = np.zeros((n, n), dtype=np.int64); T[a, b], T[b, a] = 1, -1
                basis.append((T, -T))
    else:
        basis = []
        for a in range(n):
            for b in range(n):
                E = np.zeros((n, n), dtype=np.int64); E[a, b] = 1
                basis.append((E, E.T.copy()))           # trace parts cancel in the commutators
    ok = True
    for t in range(trials):
        ntr = int(r.integers(1, 3))
        lens = [int(r.integers(2, 5)) for _ in range(ntr)]
        lab, multi, par = 0, [], []
        for L in lens:
            w = []
            for u in range(L):
                ferm = bool(r.integers(0, 2))
                w.append((lab, 0, (0, 1)) if ferm else (lab, 0, (0,)))
                par.append(1 if ferm else 0)
                lab += 1
            multi.append(tuple(w))
        nlet = lab
        i, j = sorted(int(x) for x in r.choice(nlet, 2, replace=False))
        fD, fE = int(r.integers(0, 2)), int(r.integers(0, 2))
        LD = (100, 0, (0, 1)) if fD else (100, 0, (0,))
        LE = (101, 0, (0, 1)) if fE else (101, 0, (0,))
        loc = [(a, u) for a, w in enumerate(multi) for u in range(len(w))]
        rest = [k for k in range(nlet) if k not in (i, j)]
        res = _fierz(tuple(multi), loc, i, j, LD, LE, rest, n, par, group)
        letters_all = [L for w in multi for L in w]
        nf = sum(par) - par[i] - par[j] + fD + fE
        extra = [(200 + c, 0, LD[2]) for c in range(len(basis))] + [(400 + c, 0, LE[2]) for c in range(len(basis))]
        fr = Frame(N, 3, nf + 2, [nf], sorted(set(letters_all) | {LD, LE}), seed=int(r.integers(1 << 30)), group=group)
        p = fr.p
        for c, (T, Td) in enumerate(basis):
            dl, X = fr.L[LD]
            fr.L[(200 + c, 0, LD[2])] = (dl, (np.matmul(T, X) - np.matmul(X, T)) % p)
            dl, Y = fr.L[LE]
            fr.L[(400 + c, 0, LE[2])] = (dl, (np.matmul(Td % p, Y) - np.matmul(Y, Td % p)) % p)
        v1 = np.zeros(fr.ncols, dtype=np.int64)
        if res:
            M = fr.eval_words([m for m, s in res], dtype=np.int64)
            for (m, s), row in zip(res, M):
                v1 = (v1 + s * row) % p
        # direct, written order; then reorder to (i, j, rest)
        written = list(range(nlet))
        newpar = {i: fD, j: fE}
        eps = _reorder_sign(written, [i, j] + rest, lambda k: newpar.get(k, par[k]))
        words = []
        for c in range(len(basis)):
            ww = []
            for a, w in enumerate(multi):
                ww.append(tuple((200 + c, 0, LD[2]) if (a, u) == loc[i] else (400 + c, 0, LE[2]) if (a, u) == loc[j]
                                else L for u, L in enumerate(w)))
            words.append(tuple(ww))
        M2 = fr.eval_words(words, dtype=np.int64)
        v2 = (eps * (M2.sum(axis=0) % p)) % p
        if not np.array_equal(v1 % p, v2 % p):
            ok = False
            print("MISMATCH", group, multi, i, j, LD, LE)
            break
    return ok


if __name__ == "__main__":
    for group, N in (('SO_odd', 2), ('SO_odd', 3), ('SU', 3)):
        print(f"{group} N={N}: _fierz equals the basis sum with fermions (Grassmann frame, 40 words): "
              f"{fierz_grassmann_check(group, N)}")
