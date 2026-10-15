import mpmath as mp                                   # mpmath only; about 0.3 s
mp.mp.dps = 25
G = lambda u: sum(c*mp.polylog(3, u**k) for k, c in ((0, 1), (1, -3), (2, 3), (3, -1)))  # sum_n (1-u^n)^3/n^3
h = lambda F: mp.re((-F) ** (mp.mpf(1)/3))            # principal cube root = maximal real part (KKLP p. 19)
S0 = h(G(mp.exp(2j*mp.pi/3)))                         # protected point; = (4 pi^3/27)^(1/3) cos(pi/6)
phi0 = mp.findroot(lambda t: mp.diff(lambda s: h(G(mp.exp(1j*s))), t), 2.22)
print("phi0/pi, ratio:", phi0/mp.pi, h(G(mp.exp(1j*phi0)))/S0)        # 0.708158923467, 1.01213443354
for N in (3, 4, 5):                                   # SU(N), C-twist at the so-type involution
    F = lambda u: ((N + 2)*G(u**2)/8 - G(u))/(N + 1)
    print("SU(%d) twisted: protected, max:" % N, h(F(mp.exp(2j*mp.pi/3)))/S0, h(F(mp.mpf(-1)))/S0)
for name, r, dim in (("SU(2)", 1, 3), ("SO(7)", 3, 21)):  # x = -1, holonomy acting as -1 on a Cartan
    print(name, "Coulomb gas:", (7*mp.zeta(3)*r/dim) ** (mp.mpf(1)/3)/S0)
