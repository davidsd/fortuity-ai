"""Independent exact C-twisted SU(3) index.  For t = diag(s,1,1/s), sigma(X) = -J X^T J^-1 (J antidiag(1,-1,1)):
Ad(t sigma) on sl(3) has eigenvalues  +1 (t^sigma), -1 (other Cartan), +s,-s (E12,E23 pair), +1/s,-1/s (E21,E32 pair),
-s^2 (E13), -s^-2 (E31)  [derived by hand: sigma(E12)=E23, sigma(E23)=E12, sigma(E13)=-E13, sigma on diag(1,-2,1) = -1].
Measure on g/t^sigma: prod (1 - e) over the 7 others = 2 (1-s^4)(1-s^-4).  Normalised CT of that = 1 after /2 ... :
I_C(q) = CT_s[(1-s^4)(1-s^-4) PE] / CT_s[(1-s^4)(1-s^-4)] = (2F_0 - F_4 - F_-4)/2.
First verify the eigenvalues numerically against an explicit matrix (my own construction, not the library's)."""
import numpy as np, pickle
from itertools import permutations
from weylcount import LET
# --- numeric check of eigenvalues with an explicit basis of sl(3)
J = np.array([[0,0,1],[0,-1,0],[1,0,0]], dtype=complex)
basis = []
for i in range(3):
    for j in range(3):
        if i != j:
            E = np.zeros((3,3), complex); E[i,j] = 1; basis.append(E)
basis += [np.diag([1,-1,0]).astype(complex), np.diag([0,1,-1]).astype(complex)]
Bm = np.array([b.reshape(-1) for b in basis]).T
s0 = np.exp(0.7j)
t = np.diag([s0, 1, 1/s0])
M = np.array([np.linalg.lstsq(Bm, (t @ (-J @ b.T @ np.linalg.inv(J)) @ np.linalg.inv(t)).reshape(-1), rcond=None)[0] for b in basis]).T
ev = np.sort_complex(np.linalg.eigvals(M))
pred = np.sort_complex(np.array([1, -1, s0, -s0, 1/s0, -1/s0, -s0**2, -s0**-2]))
print('eigenvalue check max diff:', np.max(np.abs(ev - pred)))
EIG = [(1, 0), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1), (-1, 2), (-1, -2)]   # (c, a): c s^a
Q = 22; ZM = 2 * Q; NZ = 2 * ZM + 1
A = np.zeros((Q + 1, Q + 1, Q + 1, NZ), dtype=np.int64); A[0, 0, 0, ZM] = 1
def shift(B, q, a):
    out = np.zeros_like(B); s1, s2, s3 = q
    if a >= 0: out[s1:, s2:, s3:, a:] = B[:Q+1-s1, :Q+1-s2, :Q+1-s3, :NZ-a]
    else: out[s1:, s2:, s3:, :NZ+a] = B[:Q+1-s1, :Q+1-s2, :Q+1-s3, -a:]
    return out
for (L, ferm, twoJ, q) in LET:
    for c, a in EIG:
        if ferm:
            A = A - c * shift(A, q, a)
        else:
            B = A.copy()
            for _ in range(Q // sum(q) + 1):
                B = A + c * shift(B, q, a)
            A = B
tw = {}
for q1 in range(Q+1):
    for q2 in range(Q+1-q1):
        for q3 in range(Q+1-q1-q2):
            F = A[q1, q2, q3]; v = 2*F[ZM] - F[ZM+4] - F[ZM-4]
            assert v % 2 == 0
            if v: tw[(q1, q2, q3)] = v // 2
lib = pickle.load(open('su3_twisted_q22_lib.pkl', 'rb'))
print('exact twisted index == library numerical (all weights |q|<=22):', tw == lib, len(tw), len(lib))
rho = (2, 1, 0)
psign = lambda p: 1 if sum(1 for i in range(3) for j in range(i+1,3) if p[i] > p[j]) % 2 == 0 else -1
SH = [(tuple(rho[i] - rho[w[i]] for i in range(3)), psign(w)) for w in permutations(range(3))]
S = {}
for s in range(Q+1):
    for a in range(s+1):
        for b in range(s+1-a):
            lam = (a, b, s-a-b)
            if lam[0] >= lam[1] >= lam[2]:
                c = sum(sg * tw.get(tuple(lam[i]+sh[i] for i in range(3)), 0) for sh, sg in SH)
                if c: S[lam] = (-1) ** s * c
pickle.dump(S, open('my_su3_s_q22.pkl', 'wb'))
import os
if os.path.exists('t4_out.pkl'):      # written by t4.py (the T_rig re-check); optional
    o = pickle.load(open('t4_out.pkl', 'rb'))
    bad = [(r['lam'], r['s'], S.get(tuple(r['lam']), 0)) for r in o['o44'] if r['s'] != S.get(tuple(r['lam']), 0)]
    print('stored s (T+diag_unsigned) vs my exact s, all 36 SU(3) T<0 columns: mismatches', bad)
print('I_C(7,4,4) =', tw.get((7,4,4)), ' s(7,4,4)=', S[(7,4,4)], ' s(9,6,6)=', S[(9,6,6)], ' s(10,6,6)=', S[(10,6,6)])
print('all SU(3) columns |lam|<=22 with s<0:', sorted([(l, v) for l, v in S.items() if v < 0], key=lambda x: (sum(x[0]), x[0])))
