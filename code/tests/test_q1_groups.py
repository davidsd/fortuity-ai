"""Gauge-structure tests of the one-loop supercharge fortuity.full.q1 for SU(N) and SO(n) (T-039(a)).
{Q_0, Q_1} = 0 is NOT sensitive to the gauge structure (it also holds with the so(n) term flipped or dropped); these are:
(1) the completeness relations used; (2) Choi-Lee (arXiv:2511.09519): Q_1 tr(phi_1 psi_1) = c (n-2) tr(lam lam) for SO(n),
and the SU(3) anchor (review B.5); (3) the rejoined traces equal an explicit basis sum sum_C [T_C,X] (x) [T^C,Y], with
bosonic letters and with fermions in a Grassmann frame; (4) the SU rejoining does not reproduce the so(n) basis sum."""
import numpy as np
from fortuity.full.q1 import _fierz, Q1_expr, _reorder_sign
from fortuity.full.evalg import Frame
from fortuity.hwcoh.evalg import matrix_size

PHI1, PSI1, LP, LM = (0, 0, (0,)), (0, 0, (1, 2)), (1, 0, ()), (0, 1, ())


def _basis(group, n):
    out = []
    if group == 'SO_odd':
        for a in range(n):
            for b in range(a + 1, n):
                T = np.zeros((n, n), dtype=np.int64); T[a, b], T[b, a] = 1, -1
                out.append((T, -T))
    else:
        for a in range(n):
            for b in range(n):
                E = np.zeros((n, n), dtype=np.int64); E[a, b] = 1
                out.append((E, E.T.copy()))
    return out


def test_completeness_relations():
    for group, n in (('SO_odd', 5), ('SU', 4)):
        S = sum(np.einsum('ab,cd->abcd', T, Td) for T, Td in _basis(group, n))
        d = np.eye(n)
        tgt = np.einsum('ad,bc->abcd', d, d) - (np.einsum('ac,bd->abcd', d, d) if group == 'SO_odd' else 0)
        if group == 'SU':
            assert np.allclose(S, np.einsum('ad,bc->abcd', d, d))          # U(n) basis; the 1/n part cancels in commutators
        else:
            assert np.allclose(S, tgt)


def _ratio(group, N):
    n = matrix_size(group, N)
    q = Q1_expr({((PHI1, PSI1),): 1}, n, group)
    words = list(q)
    fr = Frame(N, 8, 4, [2], sorted({L for m in words for w in m for L in w} | {LP, LM}), seed=9, group=group)
    p = fr.p
    M = fr.eval_words(words, dtype=np.int64)
    vq = sum((q[m] % p) * row for m, row in zip(words, M)) % p
    vl = fr.eval_words([((LP, LM),)], dtype=np.int64)[0] % p
    k = next(t for t in range(len(vl)) if vl[t])
    r = (vq[k] * pow(int(vl[k]), p - 2, p)) % p
    assert not ((vq - r * vl) % p).any()                                   # proportional
    return r if r < p // 2 else r - p


def test_choi_lee_factor_and_su3_anchor():
    assert [_ratio('SO_odd', N) for N in (2, 3, 4)] == [12, 20, 28]       # 4 (n - 2), n = 5, 7, 9
    assert abs(_ratio('SU', 3)) == 12                                      # SU(3): 4N (the anchor, up to the lam order)


def _grassmann_check(group, N, trials, seed, rejoin_group=None):
    r = np.random.default_rng(seed)
    n = matrix_size(group, N)
    basis = _basis(group, n)
    for _ in range(trials):
        lens = [int(r.integers(2, 5)) for _ in range(int(r.integers(1, 3)))]
        lab, multi, par = 0, [], []
        for L in lens:
            w = []
            for u in range(L):
                ferm = bool(r.integers(0, 2))
                w.append((lab, 0, (0, 1)) if ferm else (lab, 0, (0,)))
                par.append(int(ferm)); lab += 1
            multi.append(tuple(w))
        i, j = sorted(int(x) for x in r.choice(lab, 2, replace=False))
        fD, fE = int(r.integers(0, 2)), int(r.integers(0, 2))
        LD = (100, 0, (0, 1)) if fD else (100, 0, (0,))
        LE = (101, 0, (0, 1)) if fE else (101, 0, (0,))
        loc = [(a, u) for a, w in enumerate(multi) for u in range(len(w))]
        rest = [k for k in range(lab) if k not in (i, j)]
        res = _fierz(tuple(multi), loc, i, j, LD, LE, rest, n, par, rejoin_group or group)
        nf = sum(par) - par[i] - par[j] + fD + fE
        fr = Frame(N, 3, nf + 2, [nf], sorted({L for w in multi for L in w} | {LD, LE}), seed=int(r.integers(1 << 30)),
                   group=group)
        p = fr.p
        for c, (T, Td) in enumerate(basis):
            dl, X = fr.L[LD]; fr.L[(200 + c, 0, LD[2])] = (dl, (np.matmul(T, X) - np.matmul(X, T)) % p)
            dl, Y = fr.L[LE]; fr.L[(400 + c, 0, LE[2])] = (dl, (np.matmul(Td % p, Y) - np.matmul(Y, Td % p)) % p)
        v1 = np.zeros(fr.ncols, dtype=np.int64)
        if res:
            for (m, sg), row in zip(res, fr.eval_words([m for m, _ in res], dtype=np.int64)):
                v1 = (v1 + sg * row) % p
        newpar = {i: fD, j: fE}
        eps = _reorder_sign(list(range(lab)), [i, j] + rest, lambda k: newpar.get(k, par[k]))
        words = [tuple(tuple((200 + c, 0, LD[2]) if (a, u) == loc[i] else (400 + c, 0, LE[2]) if (a, u) == loc[j] else L
                             for u, L in enumerate(w)) for a, w in enumerate(multi)) for c in range(len(basis))]
        v2 = (eps * (fr.eval_words(words, dtype=np.int64).sum(axis=0) % p)) % p
        if not np.array_equal(v1 % p, v2 % p):
            return False
    return True


def test_rejoining_equals_basis_sum_with_fermions():
    assert _grassmann_check('SO_odd', 2, 25, 2)
    assert _grassmann_check('SO_odd', 3, 15, 3)
    assert _grassmann_check('SU', 3, 25, 4)


def test_su_rejoining_does_not_reproduce_so_basis_sum():
    assert not _grassmann_check('SO_odd', 2, 25, 2, rejoin_group='SU')
