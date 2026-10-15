"""Does Q_1 graded-commute with the tree-level psu(1,2|3) generators?  (summary/fortuity-summary.pdf: the selection rule needs
d_n to commute with the classical generators on every page; the notes check only [Q_0, J} = 0.)

For every generator J_0 (projective vector fields of C^{2|3}, acting letter by letter) and random canonical multitraces
with derivatives, at N = 2, 3:
  control : [Q_0, J_0} = 0 must hold on every word, and the wrong-sign combination must fail;
  test    : [Q_1, J_0} on the same words (Q_1 = q1d.Q1d_multi).
If [Q_1, J_0} = 0 identically, J needs no one-loop correction and d_1 commutes with J_0 outright.  If not, the rule
needs the weaker statement that [Q_1, J_0} is Q_0-exact on cocycles (J_1 exists), which this script does not test."""
import random, sys
from collections import defaultdict
from itertools import combinations
sys.argv = ['x']
import q1d
import build_cand as bc
import gravitons_full as G
from test_q1d import app, deriv, random_words
from fortuity.full.words import canon_word, canon_multi, Q_multi
from fortuity.full.letters import admissible
from fortuity.full.hw import E_multi
from fortuity.bmn.modp import P


def Km_multi(multi):
    """K_- = z_- E: (a, b, S) -> (a + b - 1 + |S|) (a, b - 1, S), the partner of bc.Kp_multi."""
    out = defaultdict(int)
    for i, w in enumerate(multi):
        for pos, (a, b, S) in enumerate(w):
            c = a + b - 1 + len(S)
            if b == 0 or c == 0:
                continue
            s, cw = canon_word(w[:pos] + ((a, b - 1, S),) + w[pos + 1:])
            if s:
                sm, cm = canon_multi(multi[:i] + (cw,) + multi[i + 1:])
                if sm:
                    out[cm] += c * s * sm
    return {m: c for m, c in out.items() if c}


EVEN = [('P_+', deriv(0)), ('P_-', deriv(1)), ('D (SU(2)_R)', bc.D_multi), ('E_1', lambda m: E_multi(m, 1)),
        ('E_2', lambda m: E_multi(m, 2)), ('F_1', bc.F1_multi), ('F_2', bc.F2_multi),
        ('K_+', bc.Kp_multi), ('K_-', Km_multi)]
ODD = ([(f'd_theta{a}', G._Qtheta(a)) for a in range(3)]
       + [(f'theta{a} d_z{"+-"[al]}', G._Qtilde(a, al)) for a in range(3) for al in (0, 1)]
       + [(f'S_{a}', bc._odd_derivation(bc._S_letter(a))) for a in range(3)]
       + [(f'T_{"+-"[al]},{a}', bc._odd_derivation(bc._T_letter(al, a))) for al in (0, 1) for a in range(3)])


def fails(A, B, words, sign):
    """number of words on which A B w + sign * B A w != 0"""
    bad = 0
    for w in words:
        l, r = app(A, app(B, {w: 1})), app(B, app(A, {w: 1}))
        bad += any((l.get(k, 0) + sign * r.get(k, 0)) % P for k in set(l) | set(r))
    return bad


if __name__ == '__main__':
    nw = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 60
    rng = random.Random(11)
    Ss = [()] + [tuple(c) for r in (1, 2, 3) for c in combinations((0, 1, 2), r)]
    full = [(a, b, S) for a in range(3) for b in range(3) for S in Ss if admissible(a, b, S)]
    words = random_words(rng, full, nw)
    print(f"{nw} random multitraces (letters with up to 2+2 derivatives, 1-2 traces of 2-4 letters)", flush=True)
    print('%-16s %8s %10s | %12s | %12s' % ('generator', '[Q_0,J}', 'wrong sign', '[Q_1,J} N=2', '[Q_1,J} N=3'), flush=True)
    for kind, gens in (('even', EVEN), ('odd', ODD)):
        s = -1 if kind == 'even' else +1          # commutator for even J, anticommutator for odd J
        for name, J in gens:
            c0 = fails(Q_multi, J, words, s)
            w0 = fails(Q_multi, J, words, -s)
            row = [fails(lambda m, N=N: q1d.Q1d_multi(m, N), J, words, s) for N in (2, 3)]
            print(f"{name:16s} {c0:8d} {w0:10d} | " + ' | '.join(f"{r:12d}" for r in row), flush=True)
