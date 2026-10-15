"""T-031: exact rank mod p of the stacked rows [B; controls; V_1; ...; V_K] on a GPU (CuPy) or on the CPU (numpy).

Input (written by fullpar.py --prep-dir DIR): DIR/B.i32 (nB x ncols int32, row-major), DIR/V.i32 (nV x ncols, the
first n_ctrl rows are controls), DIR/meta.json.  Output: DIR/rank_<backend>.json with rank B, the rank after the
controls (must equal rank B), and the rank increment of each V_k (1 = Q_1 v_k is not in the Q_0-exact span).

Algorithm: the streaming blocked elimination of fortuity/hwcoh/flint/ffblas.c.  The basis is a list of blocks, each
in RREF on its own pivots and zero on the pivots of earlier blocks.  A new row block R is reduced block by block,
R <- R - R[:, piv_j] E_j (mod p), then row-reduced internally.  Products are exact in float64: E = Eh 2^13 + El with
Eh, El < 2^13, so every partial sum of a 26-bit coefficient times a 13-bit entry over at most 2^13 terms is < 2^52.
"""
import os, sys, json, time, argparse
import numpy as np

P = 67108859            # 2^26 - 5, fortuity.bmn.modp.P
SPLIT = 1 << 13
KMAX = 1 << 13          # max inner dimension per exact float64 GEMM


def backend(force_cpu=False):
    if not force_cpu:
        try:
            import cupy as cp
            cp.cuda.runtime.getDeviceCount()
            int(cp.arange(8).sum())                      # probe a device (import alone succeeds on CPU nodes)
            return cp, 'gpu'
        except Exception as e:                           # noqa: BLE001
            print('no usable GPU:', repr(e)[:200], flush=True)
    return np, 'cpu'


class ModBasis:
    def __init__(self, xp, ncols, colchunk=32768):
        self.xp, self.c, self.cc = xp, ncols, colchunk
        self.blocks = []                                 # (E int32 (r x c), piv int64 (r,))
        self.rank = 0

    def _gemm_sub(self, R, coef, E):
        """R <- (R - coef @ E) mod p, coef (m x r) int64 in [0,p), E (r x c) int32, exact via the 13-bit split."""
        xp = self.xp
        negc = (P - coef) % P
        for k0 in range(0, coef.shape[1], KMAX):
            k1 = min(coef.shape[1], k0 + KMAX)
            A = negc[:, k0:k1].astype(xp.float64)
            for c0 in range(0, self.c, self.cc):
                c1 = min(self.c, c0 + self.cc)
                Ek = E[k0:k1, c0:c1]
                Eh = (Ek >> 13).astype(xp.float64)
                El = (Ek & (SPLIT - 1)).astype(xp.float64)
                T1 = xp.fmod(A @ Eh, P)
                T2 = xp.fmod(A @ El, P)
                T = xp.fmod(T1 * SPLIT + T2, P).astype(xp.int64)
                R[:, c0:c1] = (R[:, c0:c1] + T) % P

    def reduce(self, R):
        xp = self.xp
        for E, piv in self.blocks:
            coef = R[:, piv]
            if int(xp.count_nonzero(coef)):
                self._gemm_sub(R, coef, E)
        return R

    def _rref(self, R):
        """In-place RREF of a small row block; returns (independent rows int32, pivot columns)."""
        xp = self.xp
        rows, piv = [], []
        for i in range(R.shape[0]):
            nz = xp.flatnonzero(R[i])
            if nz.size == 0:
                continue
            j = int(nz[0])
            inv = pow(int(R[i, j]), P - 2, P)
            R[i] = (R[i] * inv) % P
            col = R[:, j].copy()
            col[i] = 0
            R -= col[:, None] * R[i][None, :]           # |products| < 2^52: exact in int64; row i unchanged
            R %= P
            rows.append(i)
            piv.append(j)
        if not rows:
            return None, None
        return R[xp.asarray(rows)].astype(xp.int32), xp.asarray(piv, dtype=xp.int64)

    def add(self, R):
        """Add rows (m x c, int64 in [0,p)); returns the number of new independent rows."""
        R = self.reduce(R)
        E, piv = self._rref(R)
        if E is None:
            return 0
        self.blocks.append((E, piv))
        self.rank += E.shape[0]
        return E.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('prep_dir')
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--blk', type=int, default=2048)
    ap.add_argument('--colchunk', type=int, default=32768)
    a = ap.parse_args()
    xp, name = backend(a.cpu)
    meta = json.load(open(os.path.join(a.prep_dir, 'meta.json')))
    assert meta.get('p', P) == P, meta.get('p')
    nB, nV, c0, nc = meta['nB'], meta['nV'], meta['ncols'], meta['n_ctrl']
    # optional Cartan columns (fullpar --cartan): B is nB x (ncols + ncart).  The elimination always takes the leftmost
    # pivot, so the pivots in the first ncols columns count rank(B) and the rest count rank([B|E]) - rank(B) = the
    # Coulomb rank of the preimage sector's cohomology (rows with a Cartan pivot have zero B part).
    ncart = int(meta.get('ncart', 0) or 0)
    c = c0 + ncart
    B = np.memmap(os.path.join(a.prep_dir, 'B.i32'), dtype=np.int32, mode='r', shape=(max(nB, 1), c))
    V0 = np.memmap(os.path.join(a.prep_dir, 'V.i32'), dtype=np.int32, mode='r', shape=(max(nV, 1), c0))
    V = V0 if not ncart else np.hstack([np.asarray(V0), np.zeros((V0.shape[0], ncart), dtype=np.int32)])
    if name == 'gpu':
        import cupy as cp
        dev = cp.cuda.Device()
        print(f"GPU {dev.id}: {cp.cuda.runtime.getDeviceProperties(dev.id)['name'].decode()}, "
              f"{dev.mem_info[1] / 1e9:.0f} GB", flush=True)
    t0 = time.time()
    basis = ModBasis(xp, c, a.colchunk)
    for r0 in range(0, nB, a.blk):
        r1 = min(nB, r0 + a.blk)
        R = xp.asarray(np.asarray(B[r0:r1], dtype=np.int64)) % P
        basis.add(R)
        if (r0 // a.blk) % 10 == 0:
            print(f"  rows {r1}/{nB}: rank {basis.rank} ({time.time() - t0:.0f}s)", flush=True)
    rB = basis.rank
    rBE = None
    if ncart:
        rBE = rB
        rB = int(sum(int((piv < c0).sum()) for _, piv in basis.blocks))
        print(f"rank [B | Cartan] = {rBE}, rank B = {rB}, Coulomb rank = {rBE - rB}", flush=True)
    print(f"rank B = {rB} ({time.time() - t0:.0f}s)", flush=True)
    if nc:
        basis.add(xp.asarray(np.asarray(V[:nc], dtype=np.int64)) % P)
    rC = basis.rank
    inc = []
    for k in range(nc, nV):
        before = basis.rank
        basis.add(xp.asarray(np.asarray(V[k:k + 1], dtype=np.int64)) % P)
        inc.append(int(basis.rank - before))
    res = dict(backend=name, nB=nB, nV=nV, ncols=c0, ncart=ncart, n_ctrl=nc, rank_B=int(rB),
               rank_after_controls=int(rC) if not ncart else None, controls_exact=bool(rC == rB) if not ncart else None,
               lifted_increments=inc, n_lifted=int(sum(inc)), time=round(time.time() - t0), case=meta.get('case'))
    if ncart:
        res.update(rank_B_with_cartan=int(rBE), coulomb_rank=int(rBE - rB))
    print(json.dumps(res), flush=True)
    json.dump(res, open(os.path.join(a.prep_dir, f'rank_{name}.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
