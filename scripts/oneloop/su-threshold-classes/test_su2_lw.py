"""Checks behind su2_lw.py (SU(2) threshold class, selection rule on the even-order targets).

1. The one-species LW counter is not trivially zero: phi_0^Y is the lowest state of Sym^Y, so the sector Y * charges(phi_0)
   must give exactly 1; a weight with no SU(3)_R hw vector in one species ((0,0;4,2,0), Y = 3) must give 0.
2. eps intertwines the generators: eps(J w) = J eps(w) for the 14 conditions on random SU(2)-type multitraces (all
   traces of even length, otherwise eps = 0), with the word-level operators of build_cand.py / fortuity.full.hw and the
   abelian derivations of abelian_lw.py.  A sign-convention mismatch for the odd generators would show up here.
3. Negative control for 2: the same test with the (-1)^(fermions before) factor of the abelian odd derivations removed."""
import random, sys
from collections import defaultdict
from itertools import combinations
sys.argv = ['x']
import numpy as np
import abelian_lw as A
import build_cand as bc
import su2_lw as S
from test_q1_symm import Km_multi
from fortuity.full.words import canon_word, canon_multi
from fortuity.full.letters import admissible, is_ferm, charges2
from fortuity.full.hw import E_multi
from fortuity.bmn.modp import P
from fortuity.hwcoh import ffla


def lw_count(sec, Y):
    basis = S.monomials1(sec, Y)
    if not basis:
        return 0, 0
    return len(basis), len(basis) - int(ffla.rank(S.stacked([{m: 1} for m in basis], A.OPS)))


def app_ab(op, f):
    out = defaultdict(int)
    for m, c in f.items():
        for m2, c2 in op(m).items():
            out[m2] = (out[m2] + c * c2) % P
    return {m: c for m, c in out.items() if c}


def eps_of(v):
    return S.eps_vec({m: c % P for m, c in v.items()})


WORD_OPS = {'E1': lambda m: E_multi(m, 1), 'E2': lambda m: E_multi(m, 2), 'D': bc.D_multi, 'K+': bc.Kp_multi, 'K-': Km_multi}
WORD_OPS.update({f'S{a}': bc._odd_derivation(bc._S_letter(a)) for a in range(3)})
WORD_OPS.update({f'T{al}{a}': bc._odd_derivation(bc._T_letter(al, a)) for al in (0, 1) for a in range(3)})


def plain_odd(letter_map):
    """abelian odd derivation WITHOUT the (-1)^(fermions before) factor (negative control)"""
    return A.derivation(letter_map, False)


if __name__ == '__main__':
    ph0 = charges2((0, 0, (0,)))
    print("1. counter controls (one species):")
    for Y in (2, 4, 6):
        sec = tuple(Y * c for c in ph0)
        print(f"   phi_0^{Y}: sector {sec}: monomials, LW = {lw_count(sec, Y)} (expect LW 1)")
    print(f"   (0,0;4,2,0), Y = 3: {lw_count((0, 0, 4, 2, 0), 3)} (expect LW 0: no hw vector in one species)")
    rng = random.Random(5)
    Ss = [()] + [tuple(c) for r in (1, 2, 3) for c in combinations((0, 1, 2), r)]
    letters = [(a, b, s) for a in range(3) for b in range(3) for s in Ss if admissible(a, b, s)]
    words = []
    while len(words) < 300:
        ws = [tuple(rng.choice(letters) for _ in range(rng.choice((2, 4)))) for _ in range(rng.choice((1, 2, 3)))]
        if not all(canon_word(w)[0] for w in ws):
            continue
        s, cm = canon_multi([canon_word(w)[1] for w in ws])
        if s and S.eps_su2(cm)[0]:
            words.append(cm)
    print(f"2. eps(J w) = J eps(w) on {len(words)} random multitraces with nonzero Cartan image:")
    ab = dict(A.OPS)
    neg = {f'S{a}': plain_odd(A.S_letter(a)) for a in range(3)}
    neg.update({f'T{al}{a}': plain_odd(A.T_letter(al, a)) for al in (0, 1) for a in range(3)})
    tot = 0
    for name, wop in WORD_OPS.items():
        bad = badneg = nz = 0
        for w in words:
            lhs = eps_of(wop(w))
            rhs = app_ab(ab[name], eps_of({w: 1}))
            nz += bool(lhs or rhs)
            bad += lhs != rhs
            if name in neg:
                badneg += lhs != app_ab(neg[name], eps_of({w: 1}))
        tot += bad
        print(f"   {name:4s}: mismatches {bad:3d} (nonzero on {nz:3d} words)" + (f"; without the fermion sign: {badneg}" if name in neg else ''))
    print("TOTAL MISMATCHES", tot)
