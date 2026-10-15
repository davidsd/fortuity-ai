"""(3) C-twisted Cardy entropy of SU(N) at the involutive points of the twisted torus.   Runs in ~1 s.

At an involution s (s^2 = 1 on su(N)):  tr_adj(s^n) = N^2-1 (n even),  c_odd (n odd), with
  so-type (fixed so(N); X -> -X^T):            c_odd = N(N-1)/2 - (N(N+1)/2 - 1) = 1 - N
  sp-type (N even, fixed usp(N); X -> -Om X^T Om^-1): c_odd = N(N+1)/2 - (N(N-1)/2 - 1) = N + 1
so F/dimG = [c_odd G(u) + (N^2-1-c_odd) G(u^2)/8]/(N^2-1):
  F_so = [(N+2) G(u^2)/8 - G(u)]/(N+1),     F_sp = [(N-2) G(u^2)/8 + G(u)]/(N-1),     F_untwisted = G(u).
Protected slice u = e^{2 pi i/3} (G(u) = -iA, G(u^2) = +iA, A = 4 pi^3/27):
  S_so/S = ((N+10)/(8(N+1)))^{1/3},   S_sp/S = (|N-10|/(8(N-1)))^{1/3}   (both -> 1/2).
u = -1 (phi = pi; G(-1) = 7 zeta(3), G(1) = 0):  S_so/S = (7 zeta(3)/(N+1))^{1/3}/S0,  S_sp/S = (7 zeta(3)/(N-1))^{1/3}/(2 S0).
Large N:  F -> G(u^2)/8, so S_C(u) = S(u^2)/2: protected S/2, maximum 1.0121344 S/2 at phi = phi_0/2 or pi - phi_0/2.
"""
import time
import numpy as np
import mpmath as mp
from cardy_lib import G, ReS, S0, ZETA3

t0 = time.time()
grid = np.linspace(0, np.pi, 40001)          # phi in [0, pi] suffices: F(-phi) = conj F(phi)
g1, g2 = G(grid), G(2 * grid)


def golden(f, a, b, it=80):
    gr = (np.sqrt(5) - 1) / 2
    for _ in range(it):
        c, d = b - gr * (b - a), a + gr * (b - a)
        if f(c) > f(d):
            b = d
        else:
            a = c
    return (a + b) / 2


def Fso(N, ph):
    return ((N + 2) * G(2 * ph) / 8 - G(ph)) / (N + 1) if N != np.inf else G(2 * ph) / 8


def Fsp(N, ph):
    return ((N - 2) * G(2 * ph) / 8 + G(ph)) / (N - 1) if N != np.inf else G(2 * ph) / 8


def Fgrid(Ffun, N):                          # same as Ffun(N, grid), from the precomputed g1 = G(grid), g2 = G(2 grid)
    if N == np.inf:
        return g2 / 8
    return ((N + 2) * g2 / 8 - g1) / (N + 1) if Ffun is Fso else ((N - 2) * g2 / 8 + g1) / (N - 1)


def maxphi(Ffun, N, refine=True):
    vals = ReS(Fgrid(Ffun, N)) / S0
    k = int(np.argmax(vals))
    if not refine:                           # grid location only (used for the crossover scan)
        return grid[k], vals[k], []
    lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
    p = golden(lambda x: float(ReS(Ffun(N, np.array([x])))[0]), lo, hi)
    p = min(max(p, 0.0), np.pi)
    v = float(ReS(Ffun(N, np.array([p])))[0]) / S0
    if v < vals[k]:
        p, v = grid[k], vals[k]
    # second-best local maximum (other branch), for the crossover
    loc = list(np.where((vals[1:-1] >= vals[:-2]) & (vals[1:-1] >= vals[2:]))[0] + 1)
    if vals[-1] >= vals[-2]:
        loc.append(len(grid) - 1)
    return p, v, sorted(((vals[i], grid[i] / np.pi) for i in loc), reverse=True)


print("protected and maximal classical C-twisted entropies, in units of the untwisted protected S")
print("%6s %4s | %-10s %-10s | %-10s | %-10s %-9s | other local max (S/S, phi/pi)" %
      ("N", "type", "prot(num)", "prot(cl.f)", "phi=pi", "max_phi", "phi*/pi"))
rows = []
for N in [3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 21, 22, 24, 30, 50, 100, 1000, np.inf]:
    types = [("so", Fso)] + ([("sp", Fsp)] if (N != np.inf and N % 2 == 0) else [])
    for typ, Ff in types:
        prot = float(ReS(Ff(N, np.array([2 * np.pi / 3])))[0]) / S0
        if N == np.inf:
            cf = 0.5
        elif typ == "so":
            cf = ((N + 10) / (8 * (N + 1))) ** (1 / 3)
        else:
            cf = (abs(N - 10) / (8 * (N - 1))) ** (1 / 3)
        atpi = float(ReS(Ff(N, np.array([np.pi])))[0]) / S0
        p, v, loc = maxphi(Ff, N)
        other = [(round(a, 5), round(b, 4)) for a, b in loc if abs(b - p / np.pi) > 1e-3][:1]
        print("%6s %4s | %.6f   %.6f   | %.6f   | %.6f   %.5f   | %s" %
              ("inf" if N == np.inf else N, typ, prot, cf, atpi, v, p / np.pi, other))
        rows.append((N, typ, prot, atpi, v, p / np.pi))

print("\nclosed forms at phi = pi: so: (7 zeta(3)/(N+1))^{1/3}/S0 ; N=3: %.6f, N=4: %.6f, N=5: %.6f"
      % tuple((7 * ZETA3 / (N + 1)) ** (1 / 3) / S0 for N in (3, 4, 5)))
print("                          sp: (7 zeta(3)/(N-1))^{1/3}/(2 S0) ; N=4: %.6f, N=6: %.6f"
      % tuple((7 * ZETA3 / (N - 1)) ** (1 / 3) / (2 * S0) for N in (4, 6)))
# crossover of the so-type maximum from phi = pi (O(N) odd-n part) to the G(u^2)/8 branch near phi_0/2
prev = None
for N in range(3, 200):
    p, v, loc = maxphi(Fso, N, refine=False)
    branch = "pi" if abs(p - np.pi) < 1e-6 else "inner"
    if prev is not None and branch != prev:
        print("so-type maximum leaves phi = pi between N = %d and N = %d (now at phi/pi = %.5f, value %.6f)" % (N - 1, N, p / np.pi, v))
    prev = branch
print("large-N reference: 1.0121344/2 = %.7f ; phi_0/2 = %.7f pi" % (1.01213443354067 / 2, 0.708158923467 / 2))
print("time %.2f s" % (time.time() - t0))
