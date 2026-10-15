"""Finite-N multi-graviton counting in the BMN sector via diagonal (eigenvalue) mesons.

Gravitons = polynomials in the single-trace cohomologies u_n, v_n, w_n (n = 2..N) restricted to
diagonal traceless matrices; trace relations (mod Q-exact) become exact polynomial identities.
The dimension of the graviton space in each charge sector is computed as the number of
"standard monomials" (staircase), with linear independence tested by random evaluation over F_p,
fermions handled by the exterior-algebra frame trick.
"""
import math
import time
import itertools
import numpy as np
from collections import defaultdict
from .modp import P, RREFBasis
from .grassmann import gmul_scalar, gadd, subsets
from .words import sym_indices, _eps


def coulomb_slots(group, N):
    """(number of eigenvalue slots, traceless constraint?) of the Coulomb branch of the BMN letters.
    SU(N): N diagonal entries summing to zero. SO(2N+1): N rotation blocks a_k eps (tr of n Cartan letters is
    2(-1)^{n/2} sum_k prod a_k for n even, 0 for n odd). USp(2N): diag(a, -a), tr = 2 sum_k prod a_k (n even), 0 (n odd).
    G2 (inside SO(7)): three rotation blocks with a_1 + a_2 + a_3 = 0. For the non-SU groups the per-generator constant
    2(-1)^{n/2} only rescales and is dropped; generators with an odd number of letters vanish and are skipped."""
    return {'SU': (N, True), 'SO_odd': (N, False), 'Sp': (N, False), 'G2': (3, True)}[group]


def default_nmax(group, N):
    """Largest number of letters of a single-trace generator needed: N for SU(N), the top Casimir degree otherwise."""
    return {'SU': N, 'SO_odd': 2 * N, 'Sp': 2 * N, 'G2': 6}[group]


def coulomb_rank(group, N):
    return {'SU': N - 1, 'SO_odd': N, 'Sp': N, 'G2': 2}[group]


class MesonSystem:
    def __init__(self, N, k, npts, seed=7, p=P, nmax=None, group='SU'):
        self.group = group
        self.N, self.k, self.P, self.p = N, k, npts, p
        self.nmax = nmax if nmax is not None else default_nmax(group, N)
        rng = np.random.default_rng(seed)
        ns, traceless = coulomb_slots(group, N)
        N = ns                                   # from here on N is the number of eigenvalue slots
        self.N = ns

        def slots(shape):
            if not traceless:
                return rng.integers(0, p, shape + (N,), dtype=np.int64)
            X = rng.integers(0, p, shape + (N - 1,), dtype=np.int64)
            last = (-X.sum(axis=-1)) % p
            return np.concatenate([X, last[..., None]], axis=-1)  # (..., N)

        self.lam = [slots((npts,)) for _ in range(3)]          # lam[m][pt, slot]
        self.f = slots((npts,))
        self.V = [slots((npts, k)) for _ in range(3)]          # V[m][pt, s, slot]
        self.mesons = []      # list of dicts: name, sector, parity, value
        self._build()

    # Grassmann scalar helpers: value = {deg: (P, C(k,deg))}
    def _psi(self, m):
        return {1: self.V[m]}   # (P, k, N) per slot: degree-1 with component index s

    def _sum_slots(self, prod_lam, extra):
        """prod_lam: (P, N) bosonic product; extra: dict {deg: (P, C, N)} -> sum over slots."""
        out = {}
        for d, arr in extra.items():
            out[d] = np.einsum('pn,pcn->pc', prod_lam, arr) % self.p
        return out

    def _lam_prod(self, idx):
        pr = np.ones((self.P, self.N), dtype=np.int64)
        for i in idx:
            pr = (pr * self.lam[i]) % self.p
        return pr

    def _build(self):
        N, p, k = self.N, self.p, self.k
        ones = {0: np.ones((self.P, 1, N), dtype=np.int64)}
        for n in range(2, self.nmax + 1):
            if self.group != 'SU' and n % 2:
                continue                                 # odd-letter generators vanish on the Cartan of SO/Sp/G2
            # u_n
            for idx in sym_indices(n):
                val = self._sum_slots(self._lam_prod(idx), ones)
                self._add(('u', n, idx), val, (0,) + tuple(idx.count(i) for i in range(3)), 0)
            # v_n : T^{idx}_j - 1/(n+1) sum_p delta^{i_p}_j T^{idx\i_p, a}_a   (times (n+1))
            for idx in sym_indices(n - 1):
                for j in range(3):
                    val = self._T(idx, j, scale=n + 1)
                    for pth, ip in enumerate(idx):
                        if ip != j:
                            continue
                        rest = idx[:pth] + idx[pth + 1:]
                        for a in range(3):
                            val = gadd(val, self._T(rest + (a,), a, scale=1), p, coeff=-1)
                    if any(np.any(v) for v in val.values()):
                        q = [idx.count(i) for i in range(3)]
                        q = [q[I] + (1 - (1 if I == j else 0)) for I in range(3)]
                        self._add(('v', n, idx, j), val, (1,) + tuple(q), 1)
            # w_n : 2 f prod lam + sum_p eps^{jk i_p} prod' lam psi_j psi_k
            for idx in sym_indices(n - 1):
                prl = self._lam_prod(idx)
                val = {0: (2 * np.einsum('pn,pn->p', prl, self.f) % p)[:, None]}
                for pth, ip in enumerate(idx):
                    rest = idx[:pth] + idx[pth + 1:]
                    prr = self._lam_prod(rest)
                    for jj in range(3):
                        for kk in range(3):
                            e = _eps(jj, kk, ip)
                            if e == 0:
                                continue
                            # psi_j psi_k on slot n: degree-2 element with components from V
                            pp = self._psi_psi(jj, kk)  # (P, C(k,2), N)
                            term = np.einsum('pn,pcn->pc', prr, pp) % p
                            val = gadd(val, {2: term}, p, coeff=e)
                q = [idx.count(i) for i in range(3)]
                q = [q[I] + 1 for I in range(3)]
                self._add(('w', n, idx), val, (2,) + tuple(q), 0)

    def _T(self, idx, j, scale):
        prl = self._lam_prod(idx)
        arr = np.einsum('pn,psn->ps', prl, self.V[j]) % self.p
        return {1: (scale * arr) % self.p}

    def _psi_psi(self, jj, kk):
        """psi_j psi_k on each slot as a degree-2 Grassmann element: components theta_s theta_t (s<t):
        V_j[s] V_k[t] - V_j[t] V_k[s]."""
        key = (jj, kk)
        if not hasattr(self, '_pp_cache'):
            self._pp_cache = {}
        if key not in self._pp_cache:
            S2 = subsets(self.k, 2)
            out = np.zeros((self.P, len(S2), self.N), dtype=np.int64)
            for c, (s, t) in enumerate(S2):
                out[:, c, :] = (self.V[jj][:, s, :] * self.V[kk][:, t, :] - self.V[jj][:, t, :] * self.V[kk][:, s, :]) % self.p
            self._pp_cache[key] = out
        return self._pp_cache[key]

    def _add(self, name, val, sector, parity):
        val = {d: (v % self.p) for d, v in val.items() if np.any(v)}
        if not val:
            return
        self.mesons.append(dict(name=name, sector=tuple(int(x) for x in sector), parity=parity, value=val))


def sector_add(s, t):
    return tuple(a + b for a, b in zip(s, t))


def sector_sub(s, t):
    return tuple(a - b for a, b in zip(s, t))


class GravitonCounter:
    def __init__(self, N, jmax, k=None, npts=None, seed=7, verbose=True, nmax=None, group='SU'):
        """group: 'SU' (default; diagonal SU(N) letters), 'SO_odd' (SO(2N+1)), 'Sp' (USp(2N)) or 'G2' (N ignored):
        Coulomb-branch (Cartan-valued) graviton counts, see coulomb_slots."""
        self.N, self.jmax, self.verbose = N, jmax, verbose
        self.group = group
        self.seed = seed
        # at most 3 * rank independent fermionic eigenvalue variables: Lambda^d vanishes beyond that
        M = 3 * coulomb_rank(group, N)
        self.k = min(k if k is not None else max(1, jmax // 4 + 1), M)
        self.p = P
        self.nmax = nmax if nmax is not None else default_nmax(group, N)
        self._meson_bound = None
        self.npts = npts if npts is not None else self._auto_npts()
        self.ms = MesonSystem(N, self.k, self.npts, seed=seed, nmax=self.nmax, group=group)
        self.mes = self.ms.mesons
        self.msec = [m['sector'] for m in self.mes]
        self.std = {(0, 0, 0, 0): {()}}         # sector -> set of standard monomials
        self.dims = {(0, 0, 0, 0): 1}
        self.cols_cache = {}

    # --- crude upper bound on graviton dims: number of monomials (large-N count) ---
    def monomial_counts(self, jmax):
        """DP: number of monomials in meson components per sector, up to level jmax (gives large-N bound)."""
        # meson sectors without building values: replicate names
        secs = []
        for n in range(2, self.nmax + 1):
            for idx in sym_indices(n):
                secs.append(((0,) + tuple(idx.count(i) for i in range(3)), 0))
            for idx in sym_indices(n - 1):
                for j in range(3):
                    q = [idx.count(i) for i in range(3)]
                    q = [q[I] + (0 if I == j else 1) for I in range(3)]
                    # traceless projection removes some components; ignore (upper bound)
                    secs.append(((1,) + tuple(q), 1))
            for idx in sym_indices(n - 1):
                q = [idx.count(i) + 1 for i in range(3)]
                secs.append(((2,) + tuple(q), 0))
        counts = defaultdict(int)
        counts[(0, 0, 0, 0)] = 1
        for sec, par in secs:
            jm = 2 * sum(sec[1:])
            new = defaultdict(int)
            for s, c in counts.items():
                js = 2 * sum(s[1:])
                mult = 1
                while js + mult * jm <= jmax:
                    if par and mult > 1:
                        break
                    new[sector_add(s, tuple(mult * x for x in sec))] += c
                    mult += 1
            for s, c in new.items():
                counts[s] += c
        return counts

    def _auto_npts(self):
        counts = self.monomial_counts(self.jmax)
        best = 1
        for s, c in counts.items():
            twoJ = s[0]
            # degrees present: nψ = 2J - 2 nf; columns per point = sum_d C(k, d)
            degs = [d for d in range(twoJ % 2, twoJ + 1, 2) if d <= self.k]
            cpp = sum(math.comb(self.k, d) for d in degs)
            best = max(best, math.ceil((c + 8) / cpp))
        return best + 4

    def columns(self, sec):
        twoJ = sec[0]
        degs = [d for d in range(twoJ % 2, twoJ + 1, 2) if d <= self.k]
        return degs

    def flatten_n(self, val, degs, n):
        parts = [val[d] if d in val else np.zeros((n, math.comb(self.k, d)), dtype=np.int64) for d in degs]
        return np.concatenate(parts, axis=1).reshape(-1)

    def flatten(self, val, degs):
        parts = [val[d] if d in val else np.zeros((self.npts, math.comb(self.k, d)), dtype=np.int64) for d in degs]
        return np.concatenate(parts, axis=1).reshape(-1)

    def count_sector(self, sec):
        """Compute dim of graviton space in sector sec (all lower sectors must be done)."""
        if sec in self.dims:
            return self.dims[sec]
        cands = set()
        for mi, m in enumerate(self.mes):
            ms = self.msec[mi]
            if any(ms[i] > sec[i] for i in range(4)):
                continue
            low = sector_sub(sec, ms)
            if low not in self.std:
                continue
            for b in self.std[low]:
                if m['parity'] and mi in b:
                    continue
                cands.add(tuple(sorted(b + (mi,))))
        # keep only candidates all of whose divisors are standard
        good = []
        for mu in cands:
            ok = True
            for mi in set(mu):
                lst = list(mu)
                lst.remove(mi)
                low = sector_sub(sec, self.msec[mi])
                if tuple(lst) not in self.std.get(low, ()):
                    ok = False
                    break
            if ok:
                good.append(mu)
        good.sort(key=lambda mu: (len(mu), mu))
        degs = self.columns(sec)
        cpp = sum(math.comb(self.k, d) for d in degs)

        def count_with(nps):
            """Standard monomials among 'good' using the first nps points.  Values are computed as
            products along a prefix stack (candidates are sorted, so prefixes are shared)."""
            ncols = nps * cpp
            basis = RREFBasis(ncols)
            std = set()
            batch, batch_mu = [], []
            mes_sl = [{d: a[:nps] for d, a in m['value'].items()} for m in self.mes]
            stack_mu, stack_val = [], []   # stack_val[i] = value of stack_mu[:i+1]

            def flush():
                if not batch:
                    return
                indep = basis.add(np.array(batch, dtype=np.int64))
                for ok, mu in zip(indep, batch_mu):
                    if ok:
                        std.add(mu)
                batch.clear()
                batch_mu.clear()

            for mu in good:
                # longest common prefix with the stack
                c = 0
                while c < len(stack_mu) and c < len(mu) and stack_mu[c] == mu[c]:
                    c += 1
                del stack_mu[c:]
                del stack_val[c:]
                for t in range(c, len(mu)):
                    v = mes_sl[mu[t]] if t == 0 else gmul_scalar(stack_val[t - 1], mes_sl[mu[t]], self.k, self.p)
                    stack_mu.append(mu[t])
                    stack_val.append(v)
                batch.append(self.flatten_n(stack_val[-1], degs, nps))
                batch_mu.append(mu)
                if len(batch) >= 128:
                    flush()
            flush()
            return std, basis

        if getattr(self, 'force_full', False):
            nps = self.npts
            std, basis = count_with(nps)
        else:
            # start small: the rank is often far below the candidate count at high level
            nps = min(self.npts, max(2, math.ceil(min(len(good) // 3 + 12, 96) / cpp)))
            std, basis = count_with(nps)
            stable = False
            while nps < self.npts:
                nps2 = min(self.npts, 2 * nps)
                del basis                      # free the previous basis before building the larger one
                std2, basis = count_with(nps2)
                stable = (std2 == std)
                std, nps = std2, nps2
                if stable:
                    break
            if basis.rank >= nps * cpp - 8 or (nps >= self.npts and not stable):
                raise RuntimeError(f"sector {sec}: standard set not stable at npts={self.npts}; increase npts")
        ncols = nps * cpp
        if basis.rank >= ncols - 2:
            raise RuntimeError(f"sector {sec}: rank {basis.rank} saturates columns {ncols}; increase npts")
        self.std[sec] = std
        self.dims[sec] = len(std)
        return len(std)

    def run(self, jmax=None, checkpoint=None, guard=None):
        """Run level by level.  checkpoint: path to a pickle updated after every level (resumable).
        guard: MemGuard instance checked after every sector."""
        import pickle, os
        jmax = jmax or self.jmax
        t0 = time.time()
        start = 1
        if checkpoint and os.path.exists(checkpoint):
            ck = pickle.load(open(checkpoint, 'rb'))
            if ck['N'] == self.N and ck['k'] == self.k and ck['npts'] == self.npts and ck['seed'] == self.seed:
                self.std, self.dims = ck['std'], ck['dims']
                start = ck['qsum_done'] + 1
                print(f"[grav N={self.N}] resumed from checkpoint at j={2*ck['qsum_done']}", flush=True)
        for qsum in range(start, jmax // 2 + 1):
            for q1 in range(qsum + 1):
                for q2 in range(qsum - q1 + 1):
                    q3 = qsum - q1 - q2
                    for twoJ in range(0, qsum + 1):
                        sec = (twoJ, q1, q2, q3)
                        self.count_sector(sec)
                        if guard:
                            guard.check(f"sector {sec}")
            if self.verbose:
                nst = sum(len(v) for s, v in self.std.items() if 2 * sum(s[1:]) == 2 * qsum)
                print(f"[grav N={self.N}] j={2*qsum:3d} done, #std at this level={nst}, t={time.time()-t0:.1f}s", flush=True)
            if checkpoint:
                tmp = checkpoint + '.tmp'
                pickle.dump(dict(N=self.N, k=self.k, npts=self.npts, seed=self.seed, qsum_done=qsum,
                                 std=self.std, dims=self.dims), open(tmp, 'wb'))
                os.replace(tmp, checkpoint)
        return self.dims

    def index_array(self, A=None):
        """Z_grav as array [j, a, b] with a = q1 - q3, b = q3 - q2 (shifted by A)."""
        A = A if A is not None else self.jmax // 2
        Z = np.zeros((self.jmax + 1, 2 * A + 1, 2 * A + 1), dtype=np.int64)
        for sec, d in self.dims.items():
            j = 2 * sum(sec[1:])
            if j > self.jmax:
                continue
            a, b = sec[1] - sec[3], sec[3] - sec[2]
            Z[j, a + A, b + A] += (-1) ** sec[0] * d
        return Z
