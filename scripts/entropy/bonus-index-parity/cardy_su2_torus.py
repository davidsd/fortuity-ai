"""SU(2) untwisted maximal torus x bonus phase (KKLP's own gauge group at finite N): does the identity dominate the
protected slice and the classical maximum?  t = diag(e^{i th/2}, e^{-i th/2}), roots +-th; F = [G(u) + Gz(th) + Gz(-th)]/3.
Also the -1-on-Cartan element th = pi (tr_adj = -1): F = [G(u^2)/2 - G(u)]/3 at every phi.  Runs in < 1 s."""
import numpy as np
from cardy_lib import Gz, G, ReS, S0
th = np.linspace(0, np.pi, 1441)            # th -> -th and th -> 2pi - th are Weyl/periodicity symmetries
ph = np.linspace(0, np.pi, 2881)
T, P = np.meshgrid(th, ph, indexing='ij')
V = ReS((G(P) + Gz(T, P) + Gz(-T, P)) / 3) / S0
k = np.unravel_index(np.argmax(V), V.shape)
print("classical max over torus x phi: %.6f at th/pi = %.4f, phi/pi = %.5f" % (V[k], th[k[0]] / np.pi, ph[k[1]] / np.pi))
kp = np.argmin(np.abs(ph - 2 * np.pi / 3))
print("protected slice: max %.6f at th/pi = %.4f (identity %.6f)" % (V[:, kp].max(), th[np.argmax(V[:, kp])] / np.pi, V[0, kp]))
vpi = ReS((G(ph) + Gz(np.pi + 0 * ph, ph) + Gz(-np.pi + 0 * ph, ph)) / 3) / S0
j = np.argmax(vpi)
print("element th = pi: max over phi %.6f at phi/pi = %.5f; protected %.6f; at phi = pi %.6f" % (vpi[j], ph[j] / np.pi,
      vpi[np.argmin(np.abs(ph - 2 * np.pi / 3))], vpi[-1]))
for p in (np.pi / 3, np.pi):
    i = np.argmin(np.abs(ph - p))
    print("phi/pi = %.4f: identity %.6f, max over torus %.6f at th/pi = %.4f" % (p / np.pi, V[0, i], V[:, i].max(), th[np.argmax(V[:, i])] / np.pi))
