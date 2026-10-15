"""Quick independent checks of KKLP arXiv:2606.27955v2 sec. 4 (runs in ~1 s).

f(mu) is defined by KKLP (4.4):  log Z ~ -(N^2/(w1 w2)) [Li3(1) - 3Li3(u) + 3Li3(u^2) - Li3(u^3)] = (N^2/(w1 w2)) f(mu),
u = e^{-mu}.  Their (4.14) curve is Re[f(-i phi)^{1/3}] with the cube root of maximal real part.
"""
import mpmath as mp
from fractions import Fraction as Fr

mp.mp.dps = 30
L3 = lambda z: mp.polylog(3, z)


def f(mu):
    u = mp.e ** (-mu)
    return -(L3(1) - 3 * L3(u) + 3 * L3(u ** 2) - L3(u ** 3))


def fprime(mu):  # d f / d mu, using d/dmu Li3(e^{-k mu}) = -k Li2(e^{-k mu})
    u = mp.e ** (-mu)
    return -3 * mp.polylog(2, u) + 6 * mp.polylog(2, u ** 2) - 3 * mp.polylog(2, u ** 3)


def maxroot(z):  # cube root of z with maximal real part (KKLP branch rule below (4.14))
    r = abs(z) ** (mp.mpf(1) / 3)
    th = mp.arg(z) / 3
    return max((r * mp.expj(th + 2 * mp.pi * k / 3) for k in range(3)), key=lambda w: mp.re(w))


curve = lambda phi: mp.re(maxroot(f(-1j * phi)))

# 1. eq. (4.7): sign check
for s in (+1, -1):  # s=+1 is the upper sign in f(-+ 2 pi i/3)
    mu = -s * 2j * mp.pi / 3
    printed = mp.mpf(1) / 2 * (s * 2j * mp.pi / 3) ** 3  # KKLP (4.7) as printed: (1/2)(+- 2 pi i/3)^3
    print(f"mu = {'-' if s > 0 else '+'}2pi i/3 (u = e^{{{'+' if s > 0 else '-'}2pi i/3}}): f = {mp.nstr(f(mu), 12)};"
          f"  (4.7) printed = {mp.nstr(printed, 12)};  mu^3/2 = {mp.nstr(mu ** 3 / 2, 12)};"
          f"  max-Re cube-root phase/pi = {mp.nstr(mp.arg(maxroot(f(mu))) / mp.pi, 8)}")

# 2. maximum of the (4.14) curve, golden section around the coarse maximum
grid = [(curve(mp.pi * k / 400), k) for k in range(1, 400)]
k0 = max(grid)[1]
a, b = mp.pi * (k0 - 1) / 400, mp.pi * (k0 + 1) / 400
g = (mp.sqrt(5) - 1) / 2
for _ in range(90):
    c, d = b - g * (b - a), a + g * (b - a)
    if curve(c) > curve(d):
        b = d
    else:
        a = c
phi0 = (a + b) / 2
ratio = curve(phi0) / curve(2 * mp.pi / 3)
print(f"phi0 = {mp.nstr(phi0, 10)} = {mp.nstr(phi0 / mp.pi, 8)} pi;  ratio (4.15) = {mp.nstr(ratio, 10)}")
print(f"symmetry: curve(2pi - phi0) - curve(phi0) = {mp.nstr(curve(2 * mp.pi - phi0) - curve(phi0), 5)}")
print(f"(4.15) as printed with f(+i phi): Re[f(2.2247 i)^(1/3)]/Re[f(2 pi i/3)^(1/3)] = "
      f"{mp.nstr(mp.re(maxroot(f(2.2247j))) / mp.re(maxroot(f(2j * mp.pi / 3))), 10)}")
print(f"curve(2pi/3) = {mp.nstr(curve(2 * mp.pi / 3), 10)} vs (4 pi^3/27)^(1/3) cos(pi/6) = "
      f"{mp.nstr((4 * mp.pi ** 3 / 27) ** (mp.mpf(1) / 3) * mp.cos(mp.pi / 6), 10)}")
print(f"curve(pi) = {mp.nstr(curve(mp.pi), 8)} -> ratio at x=-1 (phi=pi): {mp.nstr(curve(mp.pi) / curve(2 * mp.pi / 3), 8)}")

# 3. bonus charge at the maximum, q = -dF/dmu with F = 3 (N^2 j1 j2)^{1/3} f^{1/3}; in units of (N^2 j1 j2)^{1/3}
mu0 = -1j * phi0
r0 = maxroot(f(mu0))
qhat = -(r0 / f(mu0)) * fprime(mu0)  # d/dmu [3 f^{1/3}] = f^{1/3} f'/f
print(f"q(phi0)/(N^2 j1 j2)^(1/3) = {mp.nstr(qhat, 10)}  (imaginary part should vanish)")
mup = -2j * mp.pi / 3
rp = maxroot(f(mup))
print(f"at the protected point mu=-2pi i/3: -dF/dmu/(N^2 j1 j2)^(1/3) = {mp.nstr(-(rp / f(mup)) * fprime(mup), 10)}")

# 4. letter-level identities (exact rationals).  Letters: (R1,R2,R3,J1,J2,Y,F)
h = Fr(1, 2)
letters = {
    'phi1': (1, 0, 0, 0, 0, 1, 0), 'phi2': (0, 1, 0, 0, 0, 1, 0), 'phi3': (0, 0, 1, 0, 0, 1, 0),
    'psi1': (-h, h, h, h, h, 1, 1), 'psi2': (h, -h, h, h, h, 1, 1), 'psi3': (h, h, -h, h, h, 1, 1),
    'f': (0, 0, 0, 1, 1, 1, 0), 'lam+': (h, h, h, h, -h, 1, 1), 'lam-': (h, h, h, -h, h, 1, 1),
    'D1': (0, 0, 0, 1, 0, 0, 0), 'D2': (0, 0, 0, 0, 1, 0, 0),
}
ok = True
for nm, (R1, R2, R3, J1, J2, Y, F) in letters.items():
    Rs = R1 + R2 + R3
    T = Rs - Y
    kklp = (J1 + T, J2 + T, R1 - T, R2 - T, R3 - T)  # exponents of (p, q, u, v, w) in (4.1)
    d = Rs + Fr(J1 + J2, 2) - Y  # (2.21)
    jj = 2 * Rs + 3 * (J1 + J2)  # KKLP's j
    qB = 3 * (Fr(Rs, 3) - T)  # KKLP's bonus charge 3(R - T), R = Rs/3
    JR = Fr(J1 - J2, 2)
    qj = tuple(Ri + Fr(J1 + J2, 2) for Ri in (R1, R2, R3))
    # (p/q)^{J_R} u^{q1} v^{q2} w^{q3} x^{-d}, x = uvw/(pq): exponents of (p, q, u, v, w)
    alt = (JR + d, -JR + d, qj[0] - d, qj[1] - d, qj[2] - d)
    ok &= (kklp == alt) and (qB == 3 * Y - 2 * Rs) and (qB == Fr(jj, 2) - 3 * d)
    print(f"{nm:5s} KKLP exps={tuple(str(e) for e in kklp)} d_BMN={d} j={jj} q_B={qB}")
print("identities kklp-exponents == (p/q)^JR u^q x^-d, q_B = 3Y-2R_sum = j/2 - 3 d_BMN:", ok)
# Q_n: Delta(R_i) = 1/2, Delta(J_i) = -1/2, Delta Y = 1 - n
for n in range(0, 4):
    T = Fr(3, 2) - (1 - n)
    print(f"Q_{n}: Delta(J1+T, J2+T, R_i-T) = {(str(-h + T), str(-h + T), str(h - T))}, Delta q_B = {3 * (1 - n) - 3},"
          f" Delta d_BMN = {Fr(3, 2) - h - (1 - n)}")

# 5. (-1)^(F+Y) = (-1)^(q_B) letter by letter (hence on all words: both sides multiplicative, q_B additive)
par_ok = all(((F + Y) - (3 * Y - 2 * (R1 + R2 + R3))) % 2 == 0
             for (R1, R2, R3, J1, J2, Y, F) in letters.values())
print("(-1)^(F+Y) == (-1)^(q_B) on every letter:", par_ok)
# 6. the other Cardy branches of x = -1 (u^3 = -1): phi = pi/3, pi, 5pi/3
for k in (1, 3, 5):
    print(f"phi = {k}pi/3: curve/curve(2pi/3) = {mp.nstr(curve(k * mp.pi / 3) / curve(2 * mp.pi / 3), 8)}")

# 7. single-letter index from the per-letter exponents reproduces the (4.1) integrand; q <= 3 min(j1, j2) per letter
import sympy as sp
P_, Qf, U, V, W = sp.symbols('p q u v w')
fug = (P_, Qf, U, V, W)
bos = ['phi1', 'phi2', 'phi3', 'f']
ferm = ['psi1', 'psi2', 'psi3', 'lam+', 'lam-']
def mono(nm):
    R1, R2, R3, J1, J2, Y, F = letters[nm]
    T = R1 + R2 + R3 - Y
    ex = (J1 + T, J2 + T, R1 - T, R2 - T, R3 - T)
    return sp.Mul(*[b ** sp.Rational(e) for b, e in zip(fug, ex)])
single = (sum(mono(n) for n in bos) - sum(mono(n) for n in ferm) + P_ * Qf) / ((1 - P_) * (1 - Qf))  # +pq: EOM
target = 1 - (1 - U) * (1 - V) * (1 - W) / ((1 - P_) * (1 - Qf))
print("single-letter index == 1-(1-u)(1-v)(1-w)/((1-p)(1-q)):", sp.simplify(single - target) == 0)
bound_ok = True
for nm, (R1, R2, R3, J1, J2, Y, F) in letters.items():
    R = Fr(R1 + R2 + R3, 3); qB = 3 * Y - 2 * (R1 + R2 + R3)
    bound_ok &= qB <= 3 * min(J1 + R, J2 + R) and min(J1 + R, J2 + R) >= 0
print("q <= 3 min(j1, j2) and j_i >= 0 on every letter:", bound_ok)
