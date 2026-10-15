"""Full 1/16-BPS letter set of N=4 SYM via the Chang-Lin BPS superfield (arXiv:2209.06728 sec. 2.1).

A letter is a Taylor coefficient of Psi(z+, z-, theta1, theta2, theta3): a triple (a, b, S) with a, b >= 0 the
powers of z+, z- and S a subset of {0,1,2} (theta indices).  |S| = 0: gaugino component (needs a+b >= 1, since
Psi(Z=0) = 0), |S| = 1: scalar phi_i (i in S), |S| = 2: fermion psi_k (k not in S), |S| = 3: field strength f.
Statistics: the letter is fermionic iff |S| is even.  Charges (Chang-Lin Table 1, with J1 = J_L + J_R, J2 = J_L - J_R):
    J1 = -1/2 + a + |S|/2,   J2 = -1/2 + b + |S|/2,   R_j = 1/2 - |S|/2 + [j in S],   q_j = R_j + J_L = (a+b)/2 + [j in S],   Y = 1;
    level j = 2 sum q_j = 3(a+b) + 2|S|.
The Q action follows from {Q, Psi(Z)} = Psi(Z)^2 with Psi(Z) = sum_m Z^m Psi_m:
    [Q, Psi_m} = (-1)^{|S|} sum_{m1+m2=m} (-1)^{|Psi_{m1}| |Z^{m2}|} eps(S1,S2) Psi_{m1} Psi_{m2},
with |Psi_{m1}| = (|S1|+1) mod 2, |Z^{m2}| = |S2| mod 2, eps(S1,S2) the sign of sorting theta^{S1} theta^{S2}
(zero unless S1, S2 disjoint), and both factors admissible letters.  The (2J, q) labels of the BMN package are
recovered for a = b = 0, |S| >= 1 (2J = J1 + J2).  Normalisations of letters differ from D^a D^b (fields) by a!b!
type factors, which is irrelevant for cohomology.
"""
from itertools import product, combinations
from functools import lru_cache

THETA = (0, 1, 2)


def admissible(a, b, S):
    return (len(S) >= 1) or (a + b >= 1)


def is_ferm(L):
    return len(L[2]) % 2 == 0




def charges2(L):
    """(2J1, 2J2, 2q1, 2q2, 2q3), q_j = R_j + J_L; all doubled to stay integral.
    R_j = 1/2 - |S|/2 + [j in S], J_L = -1/2 + (a+b)/2 + |S|/2, J_R = (a-b)/2  (Chang-Lin Table 1)."""
    a, b, S = L
    s = len(S)
    return (-1 + 2 * a + s, -1 + 2 * b + s) + tuple(a + b + 2 * (1 if j in S else 0) for j in THETA)


def name(L):
    a, b, S = L
    s = len(S)
    base = {0: 'lam', 1: 'phi%d' % (tuple(S)[0] + 1 if s == 1 else 0), 2: 'psi%d' % ([k for k in THETA if k not in S][0] + 1 if s == 2 else 0), 3: 'f'}[s]
    if s == 1:
        base = 'phi%d' % (tuple(S)[0] + 1)
    if s == 2:
        base = 'psi%d' % ([k for k in THETA if k not in S][0] + 1)
    return base + ('' if a == 0 else '+' * a) + ('' if b == 0 else '-' * b)


@lru_cache(maxsize=None)
def theta_sign(S1, S2):
    """Sign of theta^{S1} theta^{S2} -> theta^{S1 u S2} (sorted); 0 if not disjoint."""
    if set(S1) & set(S2):
        return 0
    inv = sum(1 for x in S1 for y in S2 if x > y)
    return -1 if inv % 2 else 1


@lru_cache(maxsize=None)
def Q_letter(L):
    """[Q, Psi_L} as a list of (coeff, L1, L2) with the product Psi_{L1} Psi_{L2} (matrix product)."""
    a, b, S = L
    out = []
    s = len(S)
    for a1 in range(a + 1):
        for b1 in range(b + 1):
            a2, b2 = a - a1, b - b1
            for r in range(s + 1):
                for S1 in combinations(S, r):
                    S2 = tuple(x for x in S if x not in S1)
                    L1, L2 = (a1, b1, S1), (a2, b2, S2)
                    if not (admissible(*L1) and admissible(*L2)):
                        continue
                    eps = theta_sign(S1, S2)
                    if eps == 0:
                        continue
                    sgn = (-1) ** s * (-1) ** (((len(S1) + 1) % 2) * (len(S2) % 2)) * eps
                    out.append((sgn, L1, L2))
    return tuple(out)


def letters_up_to(amax, bmax):
    """All admissible letters with a <= amax, b <= bmax."""
    out = []
    for a in range(amax + 1):
        for b in range(bmax + 1):
            for r in range(4):
                for S in combinations(THETA, r):
                    if admissible(a, b, S):
                        out.append((a, b, S))
    return out
