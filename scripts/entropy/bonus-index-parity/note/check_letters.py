"""Letter-level identities behind T-038 (pure Python, no repo import).
Charges from code/fortuity/full/letters.py:7 (Chang-Lin Table 1). Runtime < 1 s."""
import itertools, random
from fractions import Fraction as Fr

def letters(nmax):
    for a in range(nmax + 1):
        for b in range(nmax + 1 - a):
            for k in range(4):
                for S in itertools.combinations(range(3), k):
                    if k == 0 and a + b == 0:      # Psi(Z=0) = 0: no underived gaugino
                        continue
                    yield a, b, frozenset(S)

def charges(L):
    a, b, S = L; s = len(S)
    F = 1 if s % 2 == 0 else 0                    # fermionic iff |S| even
    J1 = Fr(-1, 2) + a + Fr(s, 2); J2 = Fr(-1, 2) + b + Fr(s, 2)
    R = [Fr(1, 2) - Fr(s, 2) + (j in S) for j in range(3)]
    JL, JR = (J1 + J2) / 2, (J1 - J2) / 2
    q = [R[j] + JL for j in range(3)]
    return dict(F=F, J1=J1, J2=J2, R=R, JR=JR, q=q, Y=1)

n = fails = 0
for L in letters(8):
    a, b, S = L; c = charges(L); s = len(S)
    T = sum(c['R']) - c['Y']                      # KKLP (4.1): T = R1+R2+R3-Y
    d = sum(c['R']) + (c['J1'] + c['J2']) / 2 - c['Y']   # KKLP (2.21)
    qB = 3 * (sum(c['R']) / 3 - T)                # KKLP bonus charge 3(R-T)
    ok = (c['J1'] + T == a and c['J2'] + T == b
          and all(c['R'][j] - T == (j in S) for j in range(3))   # exponents = Taylor degrees
          and d == Fr(a + b, 2)                                  # = repo d_BMN
          and qB == s == sum(c['q']) - 3 * d                     # bonus charge = theta-degree
          and (-1) ** (c['F'] + c['Y']) == (-1) ** s             # (-1)^(F+Y) = (-1)^#theta
          and (-1) ** c['F'] == (-1) ** int(2 * c['J1']) == (-1) ** int(2 * c['J2']))
    n += 1; fails += not ok
print(f"letters checked: {n}, failures: {fails}")

# Q_0 Psi = Psi^2: every splitting (a,b,S) -> (a1,b1,S1)(a2,b2,S2) shifts charges by those of Q
bad = 0; terms = 0
for L in letters(5):
    a, b, S = L; c = charges(L)
    for a1 in range(a + 1):
        for b1 in range(b + 1):
            for k in range(len(S) + 1):
                for S1 in itertools.combinations(sorted(S), k):
                    L1 = (a1, b1, frozenset(S1)); L2 = (a - a1, b - b1, S - frozenset(S1))
                    if (len(L1[2]) == 0 and a1 + b1 == 0) or (len(L2[2]) == 0 and L2[0] + L2[1] == 0):
                        continue
                    c1, c2 = charges(L1), charges(L2); terms += 1
                    dJ1 = c1['J1'] + c2['J1'] - c['J1']; dR = [c1['R'][j] + c2['R'][j] - c['R'][j] for j in range(3)]
                    dq = [c1['q'][j] + c2['q'][j] - c['q'][j] for j in range(3)]
                    dF = (c1['F'] + c2['F'] - c['F']) % 2
                    bad += not (dJ1 == Fr(-1, 2) and dR == [Fr(1, 2)] * 3 and dq == [0, 0, 0] and dF == 1)
print(f"Q_0 splitting terms: {terms}, charge-rule failures: {bad}")

# single-letter index = integrand of KKLP (4.1)
random.seed(1)
p, qq = 0.31, 0.23; u = [0.5, -0.7, 0.35]
lhs = 0.0
for (a, b, S) in letters(70):
    w = p ** a * qq ** b
    for j in S: w *= u[j]
    lhs += (-1) ** charges((a, b, S))['F'] * w
rhs = 1 - (1 - u[0]) * (1 - u[1]) * (1 - u[2]) / ((1 - p) * (1 - qq))
print(f"single-letter index: sum = {lhs:.12f}, 1-(1-u)(1-v)(1-w)/((1-p)(1-q)) = {rhs:.12f}")
