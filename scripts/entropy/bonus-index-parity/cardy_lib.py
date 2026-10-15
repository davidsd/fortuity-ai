"""Shared helpers for the T-038 Cardy numerics (scratch, not in the repo).

Conventions (KKLP arXiv:2606.27955v2 sec. 4):
  (4.4)  log Z ~ -(N^2/(w1 w2)) G(u),  G(u) = Li3(1) - 3 Li3(u) + 3 Li3(u^2) - Li3(u^3) = sum_n (1-u^n)^3/n^3,
         KKLP's f(mu) = -G(e^{-mu});  u = e^{i phi} on the unit circle (mu = -i phi).
  (4.12)-(4.14)  entropy ~ 3 (N^2 j1 j2)^{1/3} Re[(-G)^{1/3}], cube root with maximal real part.
  For z = |z| e^{i psi}, psi in (-pi, pi], the maximal-real-part cube root is the principal one, so
  Re = |z|^{1/3} cos(psi/3)  (ties only on the negative real axis, where cos(pi/3) = cos(-pi/3)).

With a holonomy (or twisted holonomy) g, the Cardy exponent is -(1/(w1 w2)) F with
  F(g, u) = sum_n tr_adj(g^n) (1-u^n)^3/n^3,
and every tr_adj(g^n) used here is a finite sum of exponentials e^{i n alpha}; so F is a sum of
  Gz(alpha, phi) = sum_n e^{i n alpha} (1-u^n)^3/n^3 = Li3(z) - 3 Li3(z u) + 3 Li3(z u^2) - Li3(z u^3),  z = e^{i alpha},
restricted to even n  (Geven(alpha, phi) = Gz(2 alpha, 2 phi)/8)  or odd n  (Godd = Gz - Geven).
Entropy ratios are always quoted as Re[(-F/dimG)^{1/3}] / Re[(-G(e^{2 pi i/3}))^{1/3}]  ("/S").
"""
import numpy as np
import mpmath as mp

ZETA3 = float(mp.zeta(3))
# Re Li3(e^{i t}) = zeta(3) - (t^2/2)(3/2 - log|t|) + sum_{m>=2} (-1)^m zeta(3-2m) t^{2m}/(2m)!   (|t| < 2 pi)
# Im Li3(e^{i t}) = pi^2 t/6 - pi t|t|/4 + t^3/12                                                (|t| <= 2 pi)
_M = 40
_C = np.array([float((-1) ** m * mp.zeta(3 - 2 * m) / mp.factorial(2 * m)) for m in range(2, _M)])


def li3c(t):
    """Li3(e^{i t}) for real arrays t (vectorized; |error| ~ 1e-15)."""
    t = np.asarray(t, dtype=float)
    t = t - 2 * np.pi * np.round(t / (2 * np.pi))          # reduce to [-pi, pi]
    t2 = t * t
    at = np.abs(t)
    with np.errstate(divide='ignore', invalid='ignore'):
        lg = np.where(at > 0, np.log(np.where(at > 0, at, 1.0)), 0.0)
    ser = np.zeros_like(t)
    for c in _C[::-1]:                                     # Horner in t^2, starting at t^4
        ser = ser * t2 + c
    re = ZETA3 - 0.5 * t2 * (1.5 - lg) + ser * t2 * t2
    im = np.pi ** 2 * t / 6 - np.pi * t * at / 4 + t * t2 / 12
    return re + 1j * im


def Gz(alpha, phi):
    return li3c(alpha) - 3 * li3c(alpha + phi) + 3 * li3c(alpha + 2 * phi) - li3c(alpha + 3 * phi)


def Geven(alpha, phi):
    return Gz(2 * alpha, 2 * phi) / 8


def Godd(alpha, phi):
    return Gz(alpha, phi) - Geven(alpha, phi)


def G(phi):
    return Gz(0 * np.asarray(phi, float), phi)


def ReS(F):
    """Re of the maximal-real-part cube root of -F (vectorized)."""
    z = -np.asarray(F)
    return np.abs(z) ** (1 / 3) * np.cos(np.angle(z) / 3)


S0 = float((4 * mp.pi ** 3 / 27) ** (mp.mpf(1) / 3) * mp.cos(mp.pi / 6))   # = ReS(G(2 pi/3)), KKLP (4.6)-(4.7)


# ------------------------------------------------------------------ SU(N) twisted torus
def su_twisted_data(N):
    """Twisted torus T0^sigma = {diag(s_1..s_r, [1], 1/s_r..1/s_1)}, s_k = e^{i theta_k}, r = N//2, sigma(X) = -J X^T J^-1,
    J = antidiag(1,-1,1,...) (the repo's choice in code/fortuity/bmn/twistedindex.py; so-type at t=1 for N odd,
    sp-type at t=1 for N even).  Returns integer frequency matrices (rows = coefficient vectors over theta):
      even n: tr_adj((t sigma)^n) = tr_adj(t^n) = sum_rows e^{i n row.theta}            (N^2-1 rows, N-1 of them zero)
      odd  n: tr_adj((t sigma)^n) = c0 + eps * sum_rows e^{i n row.theta}                 (2r rows: +-2 e_k)
    with c0 = 1 - (N mod 2), eps = (-1)^N."""
    r = N // 2
    ang = np.zeros((N, r), dtype=int)
    for k in range(r):
        ang[k, k] = 1
        ang[N - 1 - k, k] = -1
    even = [np.zeros(r, dtype=int)] * (N - 1) + [ang[i] - ang[j] for i in range(N) for j in range(N) if i != j]
    odd = []
    for k in range(r):
        e = np.zeros(r, dtype=int); e[k] = 2
        odd += [e, -e]
    return np.array(even), np.array(odd), 1 - (N % 2), (-1) ** N


def F_su_twisted(N, theta, phi, data=None):
    """Twisted Cardy coefficient per dim G: F(t sigma, u)/(N^2-1).  theta: (M, r) array, phi: (M,) array."""
    even, odd, c0, eps = data if data is not None else su_twisted_data(N)
    theta = np.atleast_2d(np.asarray(theta, float))
    phi = np.asarray(phi, float)
    F = np.zeros(phi.shape, dtype=complex)
    for row in even:
        F += Geven(theta @ row, phi)
    if c0:
        F += c0 * Godd(0 * phi, phi)
    for row in odd:
        F += eps * Godd(theta @ row, phi)
    return F / (N * N - 1)


def su_twisted_matrix_traces(N, theta, nmax):
    """Explicit check: tr_adj((Ad_t o sigma)^n), n = 1..nmax, from matrices on sl(N) (as in twistedindex.py)."""
    r = N // 2
    s = [np.exp(1j * th) for th in theta]
    diag = list(s) + ([1.0] if N % 2 else []) + [1 / x for x in reversed(s)]
    t = np.diag(diag)
    J = np.fliplr(np.diag([(-1.0) ** a for a in range(N)]))
    Ji = np.linalg.inv(J)
    basis = []
    for i in range(N):
        for j in range(N):
            if i != j:
                E = np.zeros((N, N), complex); E[i, j] = 1; basis.append(E)
    for i in range(N - 1):
        H = np.zeros((N, N), complex); H[i, i] = 1; H[i + 1, i + 1] = -1; basis.append(H)
    Bm = np.array([b.reshape(-1) for b in basis]).T
    Bp = np.linalg.pinv(Bm)
    ti = np.linalg.inv(t)
    M = Bp @ np.array([(t @ (-J @ X.T @ Ji) @ ti).reshape(-1) for X in basis]).T
    out, P = [], np.eye(N * N - 1, dtype=complex)
    for n in range(1, nmax + 1):
        P = P @ M
        out.append(np.trace(P))
    return np.array(out)


def su_twisted_formula_traces(N, theta, nmax):
    even, odd, c0, eps = su_twisted_data(N)
    theta = np.asarray(theta, float)
    out = []
    for n in range(1, nmax + 1):
        if n % 2 == 0:
            out.append(np.exp(1j * n * (even @ theta)).sum())
        else:
            out.append(c0 + eps * np.exp(1j * n * (odd @ theta)).sum())
    return np.array(out)


# ------------------------------------------------------------------ untwisted maximal tori of other groups
def roots(group):
    """Nonzero weights of the adjoint (roots) and the rank, in an orthonormal e_i basis of the torus angles."""
    g = group.lower()
    R = []
    if g in ('so7', 'b3'):
        n = 3
        for i in range(n):
            e = np.zeros(n); e[i] = 1; R += [e, -e]
    elif g in ('usp6', 'sp6', 'c3'):
        n = 3
        for i in range(n):
            e = np.zeros(n); e[i] = 2; R += [e, -e]
    elif g in ('so8', 'd4'):
        n = 4
    elif g in ('g2',):
        # G2 inside the plane: short roots +-(e1-e2), +-(e2-e3), +-(e1-e3) and long +-(2e1-e2-e3) etc., theta in R^3 / (1,1,1)
        n = 3
        short = [np.array(v, float) for v in ([1, -1, 0], [0, 1, -1], [1, 0, -1])]
        long_ = [np.array(v, float) for v in ([2, -1, -1], [-1, 2, -1], [-1, -1, 2])]
        for v in short + long_:
            R += [v, -v]
        return np.array(R), 2, n
    else:
        raise ValueError(group)
    for i in range(n):
        for j in range(i + 1, n):
            for s1 in (1, -1):
                for s2 in (1, -1):
                    v = np.zeros(n); v[i] = s1; v[j] = s2; R.append(v)
    return np.array(R), n, n


def F_untwisted(Rts, rank, theta, phi):
    """Untwisted Cardy coefficient per dim G at torus angles theta (M, n) and bonus phase phi (M,)."""
    theta = np.atleast_2d(np.asarray(theta, float))
    phi = np.asarray(phi, float)
    F = rank * G(phi)
    for row in Rts:
        F = F + Gz(theta @ row, phi)
    return F / (rank + len(Rts))


# ------------------------------------------------------------------ small Nelder-Mead (no scipy available)
def nelder_mead(f, x0, step=0.05, tol=1e-12, maxit=4000):
    """Minimize f: R^d -> R."""
    x0 = np.asarray(x0, float)
    d = len(x0)
    pts = [x0] + [x0 + step * np.eye(d)[i] for i in range(d)]
    vals = [f(p) for p in pts]
    for _ in range(maxit):
        idx = np.argsort(vals)
        pts = [pts[i] for i in idx]; vals = [vals[i] for i in idx]
        if abs(vals[-1] - vals[0]) < tol and np.max(np.abs(np.array(pts) - pts[0])) < 1e-9:
            break
        c = np.mean(pts[:-1], axis=0)
        xr = c + (c - pts[-1]); fr = f(xr)
        if fr < vals[0]:
            xe = c + 2 * (c - pts[-1]); fe = f(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = c + 0.5 * (pts[-1] - c); fc = f(xc)
            if fc < vals[-1]:
                pts[-1], vals[-1] = xc, fc
            else:
                pts = [pts[0]] + [pts[0] + 0.5 * (p - pts[0]) for p in pts[1:]]
                vals = [vals[0]] + [f(p) for p in pts[1:]]
    i = int(np.argmin(vals))
    return pts[i], vals[i]
