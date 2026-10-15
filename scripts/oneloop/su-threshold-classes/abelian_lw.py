"""Lowest-weight vectors of psu(1,2|3) in the free abelian (Coulomb-branch) theory of SU(3): the selection rule for
Coulomb-type targets of R-007 (summary/fortuity-summary.pdf).

The Cartan evaluation eps (diagonal letters) kills Q_0-exact operators and commutes with every superconformal generator
(they act letter by letter).  If t in E_1(T_n) is a lowest-weight (LW) vector, J t is Q_0-exact for every lowering J,
so eps(t) is annihilated by every lowering J: eps(t) is a LW vector of the abelian module.  Hence, if the abelian module
has no LW vector at R-007's weight in the sector T_n, every LW vector of E_1(T_n) is non-Coulomb.

Abelian module: graded-symmetric polynomials in the Taylor coefficients (i, a, b, S) of two abelian superfields
(i = 1, 2: the Cartan of SU(3); the Weyl group is not imposed, which can only enlarge the LW space, so the bound stays
valid).  On diagonal letters C: X -> -X^T is X -> -X, so C = (-1)^Y (all of T_2, T_4 is C-odd: Y odd).
LW conditions at an SU(3)_R hw [3,0], SU(2)_R singlet weight: E1, E2 (SU(3)_R raising), D (SU(2)_R lowering at
J_R = 0), K_+, K_- (z E), S_a (theta_a E), T_{al,a} (z_al d/dtheta_a), all as derivations with the letter maps of
build_cand.py (checked there to (super)commute with Q_0).
usage: python3 abelian_lw.py 2J1 2J2 2q1 2q2 2q3 Y
"""
import sys, time
from itertools import combinations, combinations_with_replacement
from collections import defaultdict
import numpy as np
from fortuity.bmn.modp import P
from fortuity.full.letters import admissible, is_ferm, charges2, theta_sign

THETA = (0, 1, 2)


def letters_for(sec, Y):
    """Abelian letters (i, a, b, S) whose charges fit in the sector (crude bound: each charge of a letter <= total)."""
    Ss = [()] + [tuple(c) for r in (1, 2, 3) for c in combinations(THETA, r)]
    out = []
    for a in range(0, 8):
        for b in range(0, 8):
            for S in Ss:
                if not admissible(a, b, S):
                    continue
                ch = charges2((a, b, S))
                # remaining Y-1 letters each have 2J >= -1 and q >= 0
                if ch[0] - (Y - 1) > sec[0] or ch[1] - (Y - 1) > sec[1] or any(c > s for c, s in zip(ch[2:], sec[2:])):
                    continue
                for i in (1, 2):
                    out.append((i, a, b, S))
    return out


def lch(L):
    return charges2(L[1:])


def monomials(sec, Y):
    """All graded-symmetric monomials (sorted letter tuples; no repeated fermionic letter) with charges sec, Y letters."""
    Ls = sorted(letters_for(sec, Y))
    chs = [np.array(lch(L)) for L in Ls]
    target = np.array(sec)
    out = []

    def rec(start, k, acc, cur):
        if k == 0:
            if (acc == target).all():
                out.append(tuple(cur))
            return
        for j in range(start, len(Ls)):
            L = Ls[j]
            nxt = j if not is_ferm(L[1:]) else j + 1          # fermionic letters at most once
            a2 = acc + chs[j]
            # prune: remaining k-1 letters contribute 2J >= -1 each, q >= 0
            if a2[0] - (k - 1) > target[0] or a2[1] - (k - 1) > target[1] or (a2[2:] > target[2:]).any():
                continue
            cur.append(L)
            rec(nxt, k - 1, a2, cur)
            cur.pop()
    rec(0, Y, np.zeros(5, dtype=np.int64), [])
    return out


def canon(mono):
    """Sort letters; sign = parity of the permutation restricted to fermionic letters; 0 if a fermion repeats."""
    lst = list(mono)
    sign = 1
    for i in range(1, len(lst)):                     # insertion sort counting fermion-fermion swaps
        j = i
        while j > 0 and lst[j - 1] > lst[j]:
            if is_ferm(lst[j - 1][1:]) and is_ferm(lst[j][1:]):
                sign = -sign
            lst[j - 1], lst[j] = lst[j], lst[j - 1]
            j -= 1
    for x, y in zip(lst, lst[1:]):
        if x == y and is_ferm(x[1:]):
            return 0, None
    return sign, tuple(lst)


def derivation(letter_map, odd):
    """letter_map: (a, b, S) -> (coeff, (a', b', S')) or None; species unchanged.  Odd derivations pick up
    (-1)^(fermionic letters before)."""
    def op(mono):
        out = defaultdict(int)
        nf = 0
        for pos, L in enumerate(mono):
            r = letter_map(L[1:])
            if r is not None:
                c, nl = r
                s, m = canon(mono[:pos] + ((L[0],) + nl,) + mono[pos + 1:])
                if s:
                    out[m] += (-1 if (odd and nf % 2) else 1) * c * s
            if is_ferm(L[1:]):
                nf += 1
        return out
    return op


def E_letter(i):
    src, dst = (1, 0) if i == 1 else (2, 1)

    def f(L):
        a, b, S = L
        if src in S and dst not in S:
            return 1, (a, b, tuple(sorted(dst if x == src else x for x in S)))
        return None
    return f


def D_letter(L):
    a, b, S = L
    return ((b + 1), (a - 1, b + 1, S)) if a >= 1 else None


def K_letter(al):
    def f(L):
        a, b, S = L
        if (a, b)[al] == 0:
            return None
        c = a + b - 1 + len(S)
        nl = (a - 1, b, S) if al == 0 else (a, b - 1, S)
        return (c, nl) if c and admissible(*nl) else None
    return f


def S_letter(a_th):
    def f(L):
        a, b, S = L
        if a_th not in S:
            return None
        T = tuple(x for x in S if x != a_th)
        c = (a + b + len(S) - 1) * theta_sign((a_th,), T) * (-1) ** len(S)
        return (c, (a, b, T)) if c and admissible(a, b, T) else None
    return f


def T_letter(al, a_th):
    def f(L):
        a, b, S = L
        if a_th in S or (a, b)[al] == 0:
            return None
        T = tuple(sorted(S + (a_th,)))
        na, nb = (a - 1, b) if al == 0 else (a, b - 1)
        return (theta_sign((a_th,), S) * (-1) ** len(S), (na, nb, T)) if admissible(na, nb, T) else None
    return f


OPS = ([('E1', derivation(E_letter(1), False)), ('E2', derivation(E_letter(2), False)),
        ('D', derivation(D_letter, False)), ('K+', derivation(K_letter(0), False)),
        ('K-', derivation(K_letter(1), False))]
       + [(f'S{a}', derivation(S_letter(a), True)) for a in THETA]
       + [(f'T{al}{a}', derivation(T_letter(al, a), True)) for al in (0, 1) for a in THETA])


def lw_dimension(sec, Y, ops=OPS, verbose=True):
    from fortuity.hwcoh import ffla
    t0 = time.time()
    basis = monomials(sec, Y)
    n = len(basis)
    if verbose:
        print(f"sector {sec} Y={Y}: {n} abelian monomials ({time.time()-t0:.0f}s)", flush=True)
    if n == 0:
        return 0, 0, 0
    # stacked map: rows = (op, target monomial), columns = source monomials
    rows, idx = {}, {}
    entries = defaultdict(int)
    for j, m in enumerate(basis):
        for name, op in ops:
            for m2, c in op(m).items():
                if c % P:
                    r = rows.setdefault((name, m2), len(rows))
                    entries[(r, j)] = (entries[(r, j)] + c) % P
    M = np.zeros((len(rows), n), dtype=np.int64)
    for (r, j), c in entries.items():
        M[r, j] = c % P
    rk = int(ffla.rank(M)) if M.size else 0
    if verbose:
        print(f"lowering + raising map: {len(rows)} x {n}, rank {rk}; LW dimension {n - rk} ({time.time()-t0:.0f}s)",
              flush=True)
    return n, rk, n - rk


if __name__ == '__main__':
    sec = tuple(int(x) for x in sys.argv[1:6])
    Y = int(sys.argv[6])
    lw_dimension(sec, Y)
