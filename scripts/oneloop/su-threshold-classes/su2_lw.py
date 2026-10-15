"""SU(2) threshold class O_0 = (5,5;8,8,8), Y = 7: the superconformal selection rule on its even-order targets.

T_n = (4,4;8,8,8) at Y = 8 - n.  E_1(T_2) has 27 classes and E_1(T_4) has 4, all Coulomb-type (log 2026-09-29,
runs/run_su2p_Y6.txt, run_su2p_Y4.txt), so eps (Cartan evaluation) is injective on them.  A target of O_0 must be a
lowest-weight (LW) vector with O_0's weight: SU(3)_R singlet, SU(2)_R singlet, annihilated by K, S, T (summary/fortuity-summary.pdf
sec. 7, given the rule).  eps commutes with the generators, so eps of a LW class is a LW vector of the abelian module.

SU(2) has one Cartan direction: letters diag(x, -x), so tr(L_1 ... L_k) -> (1 + (-1)^k) x_1 ... x_k and the abelian
module is the graded-symmetric algebra on ONE species of letters (abelian_lw.py with species 1 only).

This script: (a) the LW dimension of the one-species abelian module at T_n's weight; (b) the Cartan images of the stored
class representatives (classes/su2p_Y{6,4}/reps.pkl), their rank, and the dimension of the subspace annihilated by the
14 conditions (E_1, E_2, D, K_+-, S_a, T_{al,a}).  usage: python3 su2_lw.py"""
import json, pickle, sys, time
from collections import defaultdict
import numpy as np
import abelian_lw as A
from fortuity.full.letters import is_ferm
from fortuity.bmn.modp import P
from fortuity.hwcoh import ffla

SEC = (4, 4, 8, 8, 8)


def monomials1(sec, Y):
    """Graded-symmetric monomials of one species (i = 1) with charges sec and Y letters."""
    Ls = sorted(L for L in A.letters_for(sec, Y) if L[0] == 1)
    chs = [np.array(A.lch(L)) for L in Ls]
    target = np.array(sec)
    out = []

    def rec(start, k, acc, cur):
        if k == 0:
            if (acc == target).all():
                out.append(tuple(cur))
            return
        for j in range(start, len(Ls)):
            a2 = acc + chs[j]
            if a2[0] - (k - 1) > target[0] or a2[1] - (k - 1) > target[1] or (a2[2:] > target[2:]).any():
                continue
            cur.append(Ls[j])
            rec(j if not is_ferm(Ls[j][1:]) else j + 1, k - 1, a2, cur)
            cur.pop()
    rec(0, Y, np.zeros(5, dtype=np.int64), [])
    return out


def eps_su2(multi):
    """Cartan image of a multitrace at diag(x, -x): prod over traces of (1 + (-1)^len), letters in written order."""
    c = 1
    for w in multi:
        if len(w) % 2:
            return 0, None
        c *= 2
    s, m = A.canon(tuple((1,) + L for w in multi for L in w))
    return (c * s, m) if s else (0, None)


def eps_vec(v):
    out = defaultdict(int)
    for w, c in v.items():
        s, m = eps_su2(w)
        if s:
            out[m] = (out[m] + s * c) % P
    return {m: c for m, c in out.items() if c}


def stacked(vecs, ops):
    """matrix with one column per vector and rows (op, image monomial): its null space = combinations killed by all ops"""
    rows, ent = {}, defaultdict(int)
    for j, v in enumerate(vecs):
        for name, op in ops:
            for m, c in v.items():
                for m2, c2 in op(m).items():
                    r = rows.setdefault((name, m2), len(rows))
                    ent[(r, j)] = (ent[(r, j)] + c * c2) % P
    M = np.zeros((max(1, len(rows)), len(vecs)), dtype=np.int64)
    for (r, j), c in ent.items():
        M[r, j] = c
    return M


if __name__ == '__main__':
    res = {}
    for n, Y in ((2, 6), (4, 4)):
        t0 = time.time()
        basis = monomials1(SEC, Y)
        idx = {m: i for i, m in enumerate(basis)}
        Mab = stacked([{m: 1} for m in basis], A.OPS)
        lw_ab = len(basis) - (int(ffla.rank(Mab)) if basis else 0)
        reps = pickle.load(open(f'classes/su2p_Y{Y}/reps.pkl', 'rb'))
        vecs = reps['vecs']
        imgs = [eps_vec(v) for v in vecs]
        assert all(m in idx for f in imgs for m in f), "Cartan image outside the one-species monomial basis"
        E = np.zeros((len(imgs), len(basis)), dtype=np.int64)
        for r, f in enumerate(imgs):
            for m, c in f.items():
                E[r, idx[m]] = c
        rk = int(ffla.rank(E)) if len(imgs) else 0
        Mcl = stacked(imgs, A.OPS)
        lw_cl = len(imgs) - int(ffla.rank(Mcl))
        per = {name: len(imgs) - int(ffla.rank(stacked(imgs, [(name, op)]))) for name, op in A.OPS}
        res[f'order_{n}'] = dict(Y=Y, abelian_monomials=len(basis), abelian_LW=lw_ab, classes=len(vecs), h=reps['h'],
                                 cartan_rank=rk, LW_combinations_of_classes=lw_cl, kernel_dim_per_condition=per)
        print(f"order {n} (Y = {Y}): one-species abelian space {len(basis)}, abelian LW vectors {lw_ab}; "
              f"{len(vecs)} classes, Cartan rank {rk}; combinations annihilated by all 14 conditions: {lw_cl} "
              f"({time.time()-t0:.0f}s)", flush=True)
        print("   kernel dimension per condition:", per, flush=True)
    json.dump(res, open('runs/su2_O0_lowest_weight.json', 'w'), indent=1)
