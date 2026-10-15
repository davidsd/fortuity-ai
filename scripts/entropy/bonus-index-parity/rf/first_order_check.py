# First finite-omega correction to the KKLP ratio at equal charges (u=v=w, j1=j2=j), large N, collapsed saddle.
#   log Z = N^2 [ f(mu)/(w1 w2) + h(mu) (w1+w2)/(w1 w2) ] + O(N^2 log w),  mu = (nu + w1 + w2)/3,  x = e^{-nu}
#   => S = 3 (N^2 j^2)^{1/3} [ f0^{1/3} + (2/3) eps H / f0^{1/3} ] + O(eps^2),  eps = (N^2/j)^{1/3},
#      f0 = f(nu/3), H = f'(nu/3)/3 + h(nu/3).
# Checks: (1) on the protected surface H(-2 pi i) = -2 pi^2/9, the O(1/w) term of the exact large-N cubic
#         N^2 mu^3/(2 w^2) with 3 mu = 2 w - 2 pi i;  (2) lead + first order vs the exact cubic saddle (Newton).
# Runtime ~2 s.
import mpmath as mp
mp.mp.dps = 40
L3 = lambda z: mp.polylog(3, z); L2 = lambda z: mp.polylog(2, z)
def F0(nu):
    u = mp.e**(-nu/3); return -(L3(1) - 3*L3(u) + 3*L3(u**2) - L3(u**3))
def H(nu):
    u = mp.e**(-nu/3)
    fp = -3*L2(u) + 6*L2(u**2) - 3*L2(u**3)
    h = -(L2(1) - 3*L2(u) + 3*L2(u**2) - L2(u**3))/2
    return fp/3 + h
def root(z):
    r = abs(z)**(mp.mpf(1)/3); th = mp.arg(z)/3
    return max([r*mp.expj(th + 2*mp.pi*k/3) for k in range(3)], key=mp.re)
kap = mp.mpf(2)/3
print('H(-2 pi i) = %s ; -2 pi^2/9 = %s' % (mp.nstr(H(-2j*mp.pi), 12), mp.nstr(-2*mp.pi**2/9, 12)))
f0 = F0(-2j*mp.pi); r0 = root(f0)
for j in [mp.mpf(10)**9, mp.mpf(10)**7, mp.mpf(10)**5]:
    mu = lambda w: (2*w - 2j*mp.pi)/3
    S = lambda w: mu(w)**3/(2*w**2) + 2*w*j
    dS = lambda w: mu(w)**2/w**2 - mu(w)**3/w**3 + 2*j
    w = r0/j**(mp.mpf(1)/3)
    for _ in range(30):
        w = w - dS(w)/mp.diff(dS, w)
    eps = j**(-mp.mpf(1)/3)
    lead = 3*j**(mp.mpf(2)/3)*r0; first = 3*j**(mp.mpf(2)/3)*kap*eps*H(-2j*mp.pi)/r0
    print('j=%.0e  (exact - lead - first)/lead = %.2e   (eps^2 = %.1e)' % (float(j), float(mp.re(S(w) - lead - first)/mp.re(lead)), float(eps**2)))
P0 = 3*mp.mpf('0.70815892')*mp.pi; Pp = 2*mp.pi
lead = lambda P: mp.re(root(F0(-1j*P))); corr = lambda P: mp.re(kap*H(-1j*P)/root(F0(-1j*P)))
L0, Lp, C0, Cp = lead(P0), lead(Pp), corr(P0), corr(Pp)
print('S_cl,max/S = %.6f + %.6f eps + O(eps^2)' % (float(L0/Lp), float((C0*Lp - L0*Cp)/Lp**2)))
