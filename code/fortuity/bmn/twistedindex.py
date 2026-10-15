"""The charge-conjugation-twisted BMN index of SU(N) (T-029, result R-017),  I_C(q) = sum_2J (-1)^2J tr(C | V_(2J;q)).

C: X -> -X^T commutes with Q, so I_C(q) = sum_2J (-1)^2J tr(C | H_(2J;q)) = (-1)^(q1+q2+q3) (dim H^{P=+1} - dim H^{P=-1})
with P = C (-1)^Y (graviton parity, Y = number of letters; every multi-graviton has P = +1).  Prediction from the
cohomology data: for every SU(3)_R highest-weight column lam,
    (-1)^(sum lam) I_C(lam) = (total cohomology of the column, all 2J) - 2 #(P-odd classes).
SU(3), j <= 30 (R-007): the only non-graviton class is at (7,4,4) and is C-odd with even Y (T-028/T-029), so P-odd.
SU(4), j <= 30 (R-008, R-014): no non-graviton class, so no P-odd class.

Method: the trace over G-invariants of (g C) is a Haar integral over the coset G.sigma, where sigma is the diagram
automorphism X -> -J X^T J^-1; C and sigma differ by the inner automorphism Ad_J, trivial on invariants.  Twisted Weyl
integration (Wendt, "Weyl's character formula for non-connected Lie groups", 2001): every element of G.sigma is
conjugate into T0^sigma.sigma, T0^sigma = {diag(s_1..s_k, [1], 1/s_k..1/s_1)}, with weight det(1 - Ad(t sigma)) on
g / t^sigma, normalised so that the constant function integrates to 1.  Plethystic exponential with
tr_adj((Ad(t sigma))^k) (explicit matrix powers); fermions enter with a minus sign ((-1)^F).
Validation: the same code on the ordinary maximal torus (no twist) reproduces the exact index of the data files.

Also: kernel_parity(N, qmax, diag_dims), the P-resolved index of the kernel of the Coulomb-branch restriction,
T(lam) = (-1)^|lam| I_C(lam) - D(lam) = #P+ - #P- in ker(rho), with D the unsigned diagonal-graviton count per column
(diag_dims = GravitonCounter(N, 2 qmax).run() of fortuity.bmn.gravindex).  T < 0 forces P-odd, hence non-graviton,
classes provided D is not overcounted; for groups where C is inner (SU(2)) P = (-1)^Y.  Caveat: the diag_dims are
random-evaluation lower bounds per weight, but D is their signed Weyl alternation (hw_multiplicities), so a computed
T < 0 is rigorous only where D is certified (e.g. D = 0, or exact weight dimensions); see R-017 Checked by (4).
O(lam) = (-1)^|lam| (I - I_grav^diag)(lam) = sum (-1)^Y.
"""
import itertools
import numpy as np



class SUN:
    """sl(N) in a basis adapted to the diagram automorphism: [t^sigma | t^-sigma | root vectors]."""

    def __init__(self, N):
        self.N = N
        k = N // 2
        fixed, anti = [], []
        for a in range(k):                                   # sigma(diag x) = -diag(x reversed)
            v = np.zeros(N); v[a] = 1; v[N - 1 - a] = -1
            fixed.append(np.diag(v))
        sym = []                                             # anti-fixed traceless diagonal: x_i = x_{N+1-i}
        for a in range((N + 1) // 2):
            v = np.zeros(N); v[a] = 1; v[N - 1 - a] = 1
            sym.append(v)
        for a in range(1, len(sym)):
            anti.append(np.diag(sym[a] / sym[a].sum() - sym[0] / sym[0].sum()))
        roots = []
        for i in range(N):
            for j in range(N):
                if i != j:
                    E = np.zeros((N, N)); E[i, j] = 1
                    roots.append(E)
        self.nfix = len(fixed)
        self.basis = fixed + anti + roots
        assert len(self.basis) == N * N - 1
        Bm = np.array([b.reshape(-1) for b in self.basis]).T
        self.Bpinv = np.linalg.pinv(Bm)
        self.J = np.fliplr(np.diag([(-1.) ** a for a in range(N)]))
        self.Ji = np.linalg.inv(self.J)

    def ad_matrix(self, g, twist):
        gi = np.linalg.inv(g)
        cols = []
        for b in self.basis:
            X = -self.J @ b.T @ self.Ji if twist else b
            cols.append((g @ X @ gi).reshape(-1))
        return self.Bpinv @ np.array(cols).T


# ---------------------------------------------------------------- series in z = (z1,z2,z3), graded by d = q1+q2+q3
def zeros(d):
    return np.zeros((d + 1, d + 1), dtype=np.complex128)


def mul_h(A, da, Bm, db):
    out = zeros(da + db)
    for i in range(da + 1):
        for j in range(da + 1 - i):
            a = A[i, j]
            if a != 0:
                out[i:i + db + 1, j:j + db + 1] += a * Bm
    return out


def letter_series(k):
    """f(z^k): phi_m (+) at k e_m, psi_m (-) at k((1,1,1) - e_m), f (+) at k(1,1,1)."""
    parts = {}
    def add(q, c):
        d = sum(q)
        parts.setdefault(d, zeros(d))
        parts[d][q[0], q[1]] += c
    for m in range(3):
        e = [0, 0, 0]; e[m] = k
        add(tuple(e), 1.)
        q = [k, k, k]; q[m] -= k
        add(tuple(q), -1.)
    add((k, k, k), 1.)
    return parts


class Series:
    def __init__(self, qmax):
        self.qmax = qmax
        self.LS = {k: letter_series(k) for k in range(1, qmax + 1)}

    def pe(self, trM):
        Q = self.qmax
        F = {d: zeros(d) for d in range(1, Q + 1)}
        for k in range(1, Q + 1):
            for d, arr in self.LS[k].items():
                if d <= Q:
                    F[d] += arr * (trM[k] / k)
        E = {0: np.ones((1, 1), dtype=np.complex128)}
        for d in range(1, Q + 1):
            acc = zeros(d)
            for i in range(1, d + 1):
                acc += i * mul_h(F[i], i, E[d - i], d - i)
            E[d] = acc / d
        return E


def traces(M, kmax):
    out = {}
    P = np.eye(M.shape[0], dtype=np.complex128)
    for k in range(1, kmax + 1):
        P = P @ M
        out[k] = np.trace(P)
    return out


def integrate(G, qmax, twist, Mg=None):
    """Normalised Haar integral of the plethystic exponential over G (twist=False) or G.sigma (twist=True)."""
    S = Series(qmax)
    N = G.N
    if twist:
        dim = N // 2
        Mg = Mg or (4 * qmax + 12)
    else:
        dim = N - 1
        Mg = Mg or (4 * qmax + 9)
    tot = {d: zeros(d) for d in range(qmax + 1)}
    norm = 0.
    for idx in itertools.product(range(Mg), repeat=dim):
        s = [np.exp(2j * np.pi * a / Mg) for a in idx]
        if twist:
            diag = list(s) + ([1.] if N % 2 else []) + [1. / x for x in reversed(s)]
            M = G.ad_matrix(np.diag(diag), True)
            w = np.linalg.det(np.eye(N * N - 1 - G.nfix) - M[G.nfix:, G.nfix:])
        else:
            diag = list(s) + [1. / np.prod(s)]
            M = G.ad_matrix(np.diag(diag), False)
            w = np.prod([1 - diag[i] / diag[j] for i in range(N) for j in range(N) if i != j])
        E = S.pe(traces(M, qmax))
        norm += w
        for d in range(qmax + 1):
            tot[d] += w * E[d]
    return {d: tot[d] / norm for d in tot}, norm / Mg ** dim


def to_int(series):
    out, err = {}, 0.
    for d, arr in series.items():
        for q1 in range(d + 1):
            for q2 in range(d + 1 - q1):
                v = arr[q1, q2]
                r = round(v.real)
                err = max(err, abs(v - r))
                if r:
                    out[(q1, q2, d - q1 - q2)] = r
    return out, err


def hw_multiplicities(wts, qmax):
    a12 = np.array([1, -1, 0]); a23 = np.array([0, 1, -1])
    terms = [(0, 0, 1), (1, 0, -1), (0, 1, -1), (2, 1, 1), (1, 2, 1), (2, 2, -1)]
    doms = {(a, b, d - a - b) for d in range(qmax + 1) for a in range(d + 1) for b in range(d + 1 - a)
            if a >= b >= d - a - b}
    out = {}
    for lam in sorted(doms):          # all dominant weights (a zero weight multiplicity can still carry an irrep)
        c = sum(sg * wts.get(tuple(np.array(lam) + i * a12 + j * a23), 0) for i, j, sg in terms)
        if c:
            out[lam] = c
    return out


def kernel_parity(N, qmax, diag_dims, with_ordinary=True):
    """Per SU(3)_R hw column lam (sum lam <= qmax): T = (-1)^|lam| I_C - D and (optionally) O = (-1)^|lam| (I - I_diag).
    diag_dims: {(2J, q1, q2, q3): dim of the diagonal graviton space} covering j <= 2 qmax.  Returns {lam: dict}."""
    uns, sig = {}, {}
    for sec, d in diag_dims.items():
        q = tuple(sec[1:])
        if sum(q) <= qmax and d:
            uns[q] = uns.get(q, 0) + d
            sig[q] = sig.get(q, 0) + (-1) ** sec[0] * d
    G = SUN(N)
    tw, _ = to_int(integrate(G, qmax, True)[0])
    H_tw, H_un = hw_multiplicities(tw, qmax), hw_multiplicities(uns, qmax)
    H_pl = H_sg = None
    if with_ordinary:
        pl, _ = to_int(integrate(G, qmax, False)[0])
        H_pl, H_sg = hw_multiplicities(pl, qmax), hw_multiplicities(sig, qmax)
    out = {}
    for lam in sorted(set(H_tw) | set(H_un) | (set(H_pl) if H_pl else set()), key=lambda l: (sum(l), l)):
        if lam == (0, 0, 0):
            continue
        sgn = (-1) ** sum(lam)
        row = dict(T=sgn * H_tw.get(lam, 0) - H_un.get(lam, 0), twisted=H_tw.get(lam, 0), diag_unsigned=H_un.get(lam, 0))
        if with_ordinary:
            row.update(O=sgn * (H_pl.get(lam, 0) - H_sg.get(lam, 0)), index=H_pl.get(lam, 0))
        out[lam] = row
    return out
