"""Explicit-matrix check of the C-twisted adjoint traces used in sec. 5 (runtime < 1 s).
sigma(X) = -J X^T J^{-1}, J = antidiag((-1)^{i+1}); t = diag(e^{i a}), a = (th_1..th_r, [0], -th_r..-th_1).
Claim: tr_su(N)((Ad_t sigma)^n) = (N-1) + sum_{i!=j} e^{i n (a_i-a_j)} (n even);
                               = [N even] + (-1)^N sum_k 2cos(2 n th_k)  (n odd)."""
import numpy as np
rng = np.random.default_rng(7)
worst = 0.0
for N in (3, 4, 5, 6):
    r = N // 2
    J = np.zeros((N, N))
    for i in range(N):
        J[i, N - 1 - i] = (-1) ** i
    Ji = np.linalg.inv(J)
    for trial in range(3):
        th = rng.uniform(0, np.pi, r)
        a = np.concatenate([th, [0.0] if N % 2 else [], -th[::-1]])
        t = np.diag(np.exp(1j * a)); ti = np.diag(np.exp(-1j * a))
        M = np.zeros((N * N, N * N), complex)          # matrix of X -> t sigma(X) t^{-1} on gl(N)
        for k in range(N * N):
            E = np.zeros((N, N)); E.flat[k] = 1
            M[:, k] = (t @ (-J @ E.T @ Ji) @ ti).ravel()
        Mn = np.eye(N * N, dtype=complex)
        for n in range(1, 8):
            Mn = Mn @ M
            tr = np.trace(Mn) - (-1) ** n               # remove the identity (acted on by -1)
            if n % 2 == 0:
                pred = (N - 1) + sum(np.exp(1j * n * (a[i] - a[j])) for i in range(N) for j in range(N) if i != j)
            else:
                pred = (1 if N % 2 == 0 else 0) + (-1) ** N * sum(2 * np.cos(2 * n * x) for x in th)
            worst = max(worst, abs(tr - pred))
    # involutions: theta = 0 and (for even N) theta = (pi/2)^r; c_odd = tr at n = 1
    for lab, th in (("theta=0", np.zeros(r)), ("theta=pi/2", np.full(r, np.pi / 2))):
        if lab == "theta=pi/2" and N % 2: continue
        c = (1 if N % 2 == 0 else 0) + (-1) ** N * sum(2 * np.cos(2 * x) for x in th)
        print(f"SU({N}) {lab}: c_odd = {c:+.0f}  (so-type 1-N = {1-N}, sp-type N+1 = {N+1})")
print("max |explicit - formula| over N=3..6, n<=7, random t:", f"{worst:.1e}")
