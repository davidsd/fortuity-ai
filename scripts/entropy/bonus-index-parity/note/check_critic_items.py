# Checks of numbers raised by the two critic reports (T-038 note, revision 2). mpmath only; ~2-4 s.
# Units: X = (dim G j1 j2)^(1/3); S_cl = 3 X f^(1/3) + mu q_B with f = -F/dimG; protected S = 3 X S0.
import mpmath as mp
mp.mp.dps = 25
Li = mp.polylog
G = lambda u: Li(3, 1) - 3*Li(3, u) + 3*Li(3, u**2) - Li(3, u**3)      # sum_n (1-u^n)^3/n^3
cr = lambda z: mp.cbrt(abs(z))*mp.expj(mp.arg(z)/3)                       # principal root = max real part
S0 = mp.re(cr(-G(mp.expj(2*mp.pi/3))))
z3 = mp.zeta(3)
gas = lambda r: (7*z3*r)**(mp.mpf(1)/3)/S0                                 # unsigned abelian gas, r = rank/dimG
print("S0 =", mp.nstr(S0, 10))

# 1. U(2): so-type C-twist at phi = pi has c_odd = -N = -2 (trace multiplet + SU(2) Cartan), dim G = 4
print("U(2) so-type at phi=pi, rank 2/dim 4:", mp.nstr(gas(mp.mpf(2)/4), 7))
print("U(2) trace multiplet alone, 1/dim 4  :", mp.nstr(gas(mp.mpf(1)/4), 7))
print("U(2) so-type protected (11/16)^(1/3) :", mp.nstr(mp.cbrt(mp.mpf(11)/16), 7))
# 2. SU(N) unsigned Coulomb gas (7 zeta(3)/(N+1))^(1/3)/S0 = graviton-gas proxy; SU(2) margin below KKLP
ratio_kklp = mp.mpf('1.01213443354067')
for N in (2, 3, 4, 5):
    print("SU(%d) Coulomb gas: %s" % (N, mp.nstr(gas(mp.mpf(1)/(N+1)), 7)))
print("KKLP - SU(2) gas margin:", mp.nstr(ratio_kklp - gas(mp.mpf(1)/3), 5))
# 3. Naive 'twisted KKLP ratio' = (twisted max over bonus)/(exact twisted index growth)
for N in (3, 4, 5):
    prot = mp.cbrt(mp.mpf(N + 10)/(8*(N + 1)))
    mx = gas(mp.mpf(1)/(N + 1))                                            # max at phi = pi for N <= 12
    print("SU(%d): twisted max/protected = %s/%s = %s" % (N, mp.nstr(mx, 7), mp.nstr(prot, 7), mp.nstr(mx/prot, 6)))
print("large N: 0.5060672/0.5 =", mp.nstr(ratio_kklp/2/mp.mpf('0.5'), 7))
print("SO(7): 1.0121344/0.738577 =", mp.nstr(ratio_kklp/gas(mp.mpf(3)/21), 6))
# 4. Fixed-sector twisted growth at KKLP's real q_B* (one Newton solve from the critic's saddle, SU(3) and SU(5))
phi0 = mp.mpf('2.22474687153909')
u0 = mp.expj(phi0)
f0 = -G(u0); fp0 = -3*Li(2, u0) + 6*Li(2, u0**2) - 3*Li(2, u0**3)          # d f/d mu at mu = -i phi0
qB = mp.re(-(cr(f0)**-2)*fp0)
print("q_B*/X =", mp.nstr(qB, 8), " (imag part", mp.nstr(mp.im(-(cr(f0)**-2)*fp0), 3), ")")
print("untwisted check: Re S(q_B*)/S =", mp.nstr(mp.re(3*cr(f0) + (-1j*phi0)*qB)/(3*S0), 9))
for N, start in ((3, mp.mpc('0.0716', '-1.1267')), (5, mp.mpc('0.07', '-1.12'))):
    fC = lambda mu: (G(mp.exp(-mu)) - (N + 2)*G(mp.exp(-2*mu))/8)/(N + 1)  # -F_so/dimG
    S = lambda mu: 3*cr(fC(mu)) + mu*qB
    mu = mp.findroot(lambda m: mp.diff(S, m), start)
    print("SU(%d) so-type at q_B*: mu = %s, Re S_C/S = %s, |dS| = %s" % (N, mp.nstr(mu, 6), mp.nstr(mp.re(S(mu))/(3*S0), 6), mp.nstr(abs(mp.diff(S, mu)), 2)))
