"""Extra numbers for the T-038 analysis (< 1 s): (a) SU(N) so-type twisted value at KKLP's phi_0 (balance of C-sectors at the
classical maximum); (b) U(N) (KKLP's (4.1) gauge group) so-type twisted closed forms, for comparison with SU(N):
F_so^{U(N)} = [(N+1) G(u^2)/8 - G(u)]/N  (C = -1 on the trace part: c_odd = -N, c_even = N^2)."""
import numpy as np
from cardy_lib import G, ReS, S0, ZETA3
phi0 = 0.708158923467 * np.pi
for N in (3, 4, 5):
    so = lambda ph: ((N + 2) * G(2 * ph) / 8 - G(ph)) / (N + 1)
    v = ReS(so(np.array([phi0])))[0] / S0
    print("(a) SU(%d) so-type at phi_0: S_C/S = %.6f  (untwisted 1.012134; gap %.6f S)" % (N, v, 1.01213443354 - v))
for N in (2, 3, 4, 5):
    uso = lambda ph: ((N + 1) * G(2 * ph) / 8 - G(ph)) / N
    p = ReS(uso(np.array([2 * np.pi / 3])))[0] / S0
    pi_ = ReS(uso(np.array([np.pi])))[0] / S0
    print("(b) U(%d) so-type: protected %.6f (closed ((N+9)/(8N))^(1/3) = %.6f); phi=pi %.6f (closed (7 zeta3/N)^(1/3)/S0 = %.6f)"
          % (N, p, ((N + 9) / (8 * N)) ** (1 / 3), pi_, (7 * ZETA3 / N) ** (1 / 3) / S0))
