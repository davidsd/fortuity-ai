"""C-twisted (involutive-point) values on the x' = -1 slice (phi = pi/3, pi) and x' = +1 (phi = 2pi/3).
Tr[P w(x)] = I_C(-x) (the P-graded classical index at the bonus point x is the C-twisted index at -x), so the P-graded
index at x = 1 (invariant under d_0, d_2, d_4, ...) has Cardy entropy max(S_C(pi/3), S_C(pi)), and at x = -1 (invariant
under every d_n) it is the exact C-twisted index S_C(2pi/3).  Runs in < 1 s."""
import numpy as np
from cardy_lib import G, ReS, S0
ph = np.array([np.pi / 3, 2 * np.pi / 3, np.pi])
print("%-12s %10s %10s %10s" % ("", "phi=pi/3", "phi=2pi/3", "phi=pi"))
print("%-12s %10.6f %10.6f %10.6f" % ("untwisted", *(ReS(G(ph)) / S0)))
for N in (3, 4, 5, 6, 8, 100):
    so = ((N + 2) * G(2 * ph) / 8 - G(ph)) / (N + 1)
    print("%-12s %10.6f %10.6f %10.6f" % ("SU(%d) so" % N, *(ReS(so) / S0)))
    if N % 2 == 0:
        sp = ((N - 2) * G(2 * ph) / 8 + G(ph)) / (N - 1)
        print("%-12s %10.6f %10.6f %10.6f" % ("SU(%d) sp" % N, *(ReS(sp) / S0)))
print("%-12s %10.6f %10.6f %10.6f" % ("N=inf", *(ReS(G(2 * ph) / 8) / S0)))
