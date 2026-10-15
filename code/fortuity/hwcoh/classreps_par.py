"""Non-graviton class representatives of large SU(3)_R highest-weight sectors (parallel engine), and the test of each
class against the SU(N) > SU(N-1) x U(1) block restriction: fused or core (T-032 follow-up; R-016).

fortuity.hwcoh.classreps does the same linear algebra in RAM; for sectors of ~10^6 words its lower Q-images alone
need > 100 GB.  Here everything big lives in the memmaps of fortuity.hwcoh.sectorpar:

1. Lower Q-images.  A SectorPar of sector n-1 that skips its own hw vectors (pre = (words, [])) evaluates Q of the hw
   vectors of sector n at the functionals of sector n-1 (memmap L).  The evaluation is faithful on the image iff
   rank L = r_down, the rank of Q from n to n-1 found by the engine run of sector n-1 (r_up there).  That rank is exact
   over F_p, because that run certified its functionals (rank of the hw vectors = exact multiplicity), so equality
   certifies L.  Otherwise the functionals are doubled.
2. Sector n.  SectorPar with pre = the hw vectors handed over by step 1 (the same blocks in the same order), memmaps
   A, B, G and the per-degree self-check rank A = m_d of SectorPar.run.
3. Linear algebra.  The hw rows of each fermion degree d (overcomplete, because of trace relations) are compressed to
   m_d + extra random combinations, regenerated from (seed, d, row) so that representatives expand back to words.
   Checks: rank of the compressed rows = m_d per degree (they span the hw space); rank of their L rows = r_down.
   Closed combinations: left kernel of the compressed L.  Their classes modulo exact + graviton (row space of [B; G],
   rank r_up + grav) are read off through random combinations of the right null space of [B; G]; representatives are
   closed combinations independent there, h - grav of them.

channel_residues: the coefficient of x^alpha in the restriction of each representative
(fortuity.bmn.blockrestrict.restrict_u1), reduced modulo exact + graviton in the SU(N-1) target sector
(fortuity.bmn.classify.TargetSector.residue).  Restriction maps exact to exact and gravitons to gravitons, so the
residues of exact or graviton classes vanish, and only channels whose target holds SU(N-1) non-graviton classes can
see anything.  A class is fused if some residue is nonzero; combinations with zero residue in every such channel are
cores (cores_from_residues).
"""
import gc
import os
import time
import numpy as np

from fortuity.bmn.modp import P, RREFBasis
from . import ffblas, ffla
from . import wordenum as enum
from .sectorpar import SectorPar, _mmap
from .hw import left_kernel


def _mm(A, B):
    return ffblas.matmul(A, B)          # BLIS kernel if built, else numpy (fortuity.bmn.modp.matmul_mod)


def _rank(M):
    """rank mod p of an int array (BLIS kernel on int32 if built)."""
    if M.shape[0] == 0 or M.shape[1] == 0:
        return 0
    if ffblas.HAVE_FFBLAS:
        return ffblas.rank_pair32(np.ascontiguousarray(np.asarray(M) % P, dtype=np.int32))[0]
    return ffla.rank(np.asarray(M, dtype=np.int64))


def _R(seed, d, a, rows, cols):
    """compression block for the hw rows [a, a + cols) of fermion degree d (deterministic in (seed, d, a))"""
    return np.random.default_rng([seed, d, a]).integers(0, P, (rows, cols), dtype=np.int64)


def lower_images(N, lam, n, dims, r_down, seed=1, maxlen=None, maxlen_up=None, group='SU', log=print,
                 workdir=None, max_attempts=4):
    """Step 1.  Returns dict(ws, hw, fL, nL, ncols, rowL): words and hw blocks of sector n (from the n-1 engine, so
    exactly the blocks behind the rows of L), the memmap file of L, and rowL[i] = first L row of hw block i (-1 if
    the block has no Q-image)."""
    lam = tuple(lam)
    ws_lo = enum.words_of_sector((n - 1,) + lam, maxlen)
    S = SectorPar(N, lam, n - 1, dims, log=log, graviton_algebra=None, maxlen=maxlen, pre=(ws_lo, []), seed=seed,
                  group=group, maxlen_up=maxlen_up, workdir=workdir)
    S.prepare()
    factor = 1
    for attempt in range(max_attempts):
        S.evaluate(factor, attempt)
        L = _mmap(S.fB, S.nB, S.ncols, 'r')
        rL = ffblas.rank_pair32(L)[0] if S.nB else 0
        del L
        log(f"  lower Q-images: {S.nB} rows x {S.ncols} functionals of sector {n - 1}, rank {rL} vs r_down {r_down}")
        if rL == r_down:
            break
        if rL > r_down:
            raise RuntimeError(f"rank {rL} of the lower Q-images exceeds r_down = {r_down}: inconsistent inputs")
        S._cleanup()
        factor *= 2
    else:
        raise RuntimeError('lower Q-images: functionals not faithful')
    row_of = {}
    for lst in S.q_d.values():
        for row0, H, _, _, _ in lst:
            row_of[id(H)] = row0
    rowL = np.array([row_of.get(id(H), -1) for _, H in S.hw_up], dtype=np.int64)
    out = dict(ws=S.ws_up, hw=S.hw_up, fL=S.fB, nL=S.nB, ncols=S.ncols, rowL=rowL, rank=rL,
               timing={k: round(v, 1) for k, v in S.timing.items()})
    for f in (S.fA, S.fG):
        try:
            os.remove(f)
        except OSError:
            pass
    del S
    gc.collect()
    return out


def upper_rows(N, lam, n, dims, GA, low, seed=1, maxlen=None, maxlen_up=None, group='SU', log=print, workdir=None,
               max_attempts=4):
    """Step 2.  SectorPar of sector n on the handed-over hw blocks; returns (S, r_up, grav) with A, B, G memmaps kept."""
    S = SectorPar(N, tuple(lam), n, dims, log=log, graviton_algebra=GA, maxlen=maxlen, pre=(low['ws'], low['hw']),
                  seed=seed, group=group, maxlen_up=maxlen_up, workdir=workdir)
    S.prepare()
    assert S.hw is low['hw']
    factor, prev = 1, None
    for attempt in range(max_attempts):
        S.evaluate(factor, attempt)
        A = _mmap(S.fA, S.nA, S.ncols, 'r')
        ok, rk = True, {}
        for d in sorted(S.m_d):
            r0, r1 = S.rows_d.get(d, (0, 0))
            c0, c1 = S.cols_d[d]
            rk[d] = ffblas.rank_pair32(A[r0:r1, c0:c1])[0] if r1 > r0 else 0
            ok &= rk[d] == S.m_d[d]
        del A
        log(f"  sector {n}: rank per degree {rk} vs m_d {S.m_d} -> {'OK' if ok else 'FAIL'}")
        if ok:
            break
        S._cleanup()
        if prev == rk:
            raise RuntimeError('the word set does not span (maxlen too small?)')
        prev, factor = rk, factor * 2
    else:
        raise RuntimeError('sector self-check failed')
    B = _mmap(S.fB, S.nB, S.ncols, 'r')
    G = _mmap(S.fG, S.nG, S.ncols, 'r')
    rB, rBG = ffblas.rank_pair32(B, G)
    del B, G
    return S, rB, rBG - rB


def _rowL_of_A(S, rowL):
    """L row of every A row of the upper engine (-1: no Q-image)."""
    pos = {id(H): i for i, (_, H) in enumerate(S.hw)}
    out = np.full(S.nA, -1, dtype=np.int64)
    for lst in S.hw_d.values():
        for row0, _, H in lst:
            r = rowL[pos[id(H)]]
            if r >= 0:
                out[row0:row0 + H.shape[0]] = r + np.arange(H.shape[0])
    return out


def compress(S, low, seed, extra=16, chunk=4096, log=print):
    """Step 3a.  Per degree d: CA[d] = R_d A_d (degree-d columns), CL[d] = R_d L_d (all columns of L)."""
    A = _mmap(S.fA, S.nA, S.ncols, 'r')
    L = _mmap(low['fL'], low['nL'], low['ncols'], 'r')
    rowA = _rowL_of_A(S, low['rowL'])
    CA, CL, Md = {}, {}, {}
    for d in sorted(S.m_d):
        r0, r1 = S.rows_d.get(d, (0, 0))
        c0, c1 = S.cols_d[d]
        if r1 == r0:
            continue
        M = Md[d] = S.m_d[d] + extra
        ca = np.zeros((M, c1 - c0), dtype=np.int64)
        cl = np.zeros((M, low['ncols']), dtype=np.int64)
        for a in range(r0, r1, chunk):
            a1 = min(r1, a + chunk)
            R = _R(seed, d, a, M, a1 - a)
            ca = (ca + _mm(R, np.asarray(A[a:a1, c0:c1], dtype=np.int64))) % P
            idx = rowA[a:a1]
            m = idx >= 0
            if m.any():
                Ls = np.zeros((a1 - a, low['ncols']), dtype=np.int64)
                Ls[m] = L[idx[m]]
                cl = (cl + _mm(R, Ls)) % P
        CA[d], CL[d] = ca, cl
        log(f"  compressed degree {d}: {r1 - r0} hw rows -> {M}")
    del A, L
    return CA, CL, Md


def nongraviton_reps(N, lam, n, dims, GA, r_down, expect=None, seed=1, maxlen=None, maxlen_up=None, group='SU',
                     log=print, workdir=None, extra=16, nproj=32):
    """Representatives (formal expressions mod p, over the words of sector n) of a basis of the non-graviton classes
    of sector (n; lam), i.e. of hw(H) / (exact + graviton).  expect: optional dict(m=, r_up=, grav=, bh=) from the
    engine run of the sector, checked.  Returns (reps, info)."""
    t0 = time.time()
    lam = tuple(lam)
    info = dict(N=N, group=group, lam=list(lam), n=n, seed=seed, maxlen=maxlen, maxlen_up=maxlen_up, r_down=r_down)
    low = lower_images(N, lam, n, dims, r_down, seed, maxlen, maxlen_up, group, log, workdir)
    info.update(lower_rows=int(low['nL']), lower_ncols=int(low['ncols']), lower_timing=low['timing'])
    S, r_up, grav = upper_rows(N, lam, n, dims, GA, low, seed, maxlen, maxlen_up, group, log, workdir)
    info.update(m=int(S.m), m_d={str(k): int(v) for k, v in S.m_d.items()}, words=len(low['ws']), hw_rows=int(S.nA),
                ncols=int(S.ncols), r_up=int(r_up), grav=int(grav))
    try:
        CA, CL, Md = compress(S, low, seed, extra, log=log)
        degs = sorted(Md)
        for d in degs:
            rd = _rank(CA[d])
            assert rd == S.m_d[d], f"compressed rows of degree {d} have rank {rd} != m_d {S.m_d[d]}"
        Lc = np.concatenate([CL[d] for d in degs])
        del CL
        K = left_kernel(Lc)
        rL = Lc.shape[0] - K.shape[0]
        log(f"  compressed L: {Lc.shape}, rank {rL} vs r_down {r_down}; closed combinations {K.shape[0]}")
        assert rL == r_down, f"compressed L has rank {rL} != r_down {r_down}"
        del Lc
        # evaluations of the closed combinations at the functionals of sector n
        CV = np.zeros((K.shape[0], S.ncols), dtype=np.int64)
        off = 0
        for d in degs:
            c0, c1 = S.cols_d[d]
            CV[:, c0:c1] = _mm(K[:, off:off + Md[d]], CA[d])
            off += Md[d]
        dimZ = _rank(CV)
        h = dimZ - r_up
        log(f"  closed: rank {dimZ} = m - r_down = {S.m - r_down}; h = {h}, grav = {grav}, non-graviton = {h - grav}")
        assert dimZ == S.m - r_down
        # classes modulo exact + graviton: random combinations of the annihilator of [B; G]
        B = _mmap(S.fB, S.nB, S.ncols, 'r')
        G = _mmap(S.fG, S.nG, S.ncols, 'r')
        if ffblas.HAVE_FFBLAS:
            sb = ffblas.StreamBasis(S.ncols)
            for a in range(0, S.nB, 8192):
                sb.add(np.asarray(B[a:a + 8192], dtype=np.int64))
            if S.nG:
                sb.add(np.asarray(G, dtype=np.int64))
            assert sb.rank == r_up + grav
            Wp = sb.null_space()
            del sb
        else:
            Wp = ffla.nullspace_right(np.concatenate([np.asarray(B, dtype=np.int64), np.asarray(G, dtype=np.int64)]))
        del B, G
        Y = _mm(np.ascontiguousarray(Wp.T), np.random.default_rng([seed, 77]).integers(0, P, (Wp.shape[0], nproj)))
        del Wp
        PZ = _mm(CV, Y)
        del CV
        bas = RREFBasis(nproj)
        idx = np.flatnonzero(bas.add_batched(PZ % P))
        log(f"  non-graviton classes found: {len(idx)}")
        assert len(idx) == h - grav
        # expand the representatives to words
        reps = []
        for i in idx:
            u = K[i]
            coef = np.zeros(S.nA, dtype=np.int64)
            off = 0
            for d in degs:
                r0, r1 = S.rows_d[d]
                ud = u[off:off + Md[d]][None, :]
                for a in range(r0, r1, 4096):
                    a1 = min(r1, a + 4096)
                    coef[a:a1] = _mm(ud, _R(seed, d, a, Md[d], a1 - a))[0]
                off += Md[d]
            rep = {}
            for lst in S.hw_d.values():
                for row0, widx, H in lst:
                    cb = coef[row0:row0 + H.shape[0]]
                    if not cb.any():
                        continue
                    wc = _mm(cb[None, :], H % P)[0]
                    for wi, c in zip(widx, wc):
                        if c:
                            w = low['ws'][int(wi)]
                            rep[w] = (rep.get(w, 0) + int(c)) % P
            reps.append({w: c for w, c in rep.items() if c})
        info.update(h=int(h), nongrav=len(reps), rep_terms=[len(r) for r in reps], extra=extra,
                    timing={k: round(v, 1) for k, v in S.timing.items()}, time=round(time.time() - t0, 1))
        if expect:
            for k, v in expect.items():
                got = dict(m=info['m'], r_up=r_up, grav=grav, bh=len(reps), h=h).get(k)
                assert got == v, f"{k}: got {got}, expected {v}"
        return reps, info
    finally:
        S._cleanup()
        try:
            os.remove(low['fL'])
        except OSError:
            pass


def channel_residues(reps, N, lam, n, channels, dims_small, gen_nmax=3, tseed=17, maxlen_small=3, log=print):
    """For each alpha in channels: classify the coefficient of x^alpha of the SU(N) > SU(N-1) x U(1) restriction of every
    representative in the SU(N-1) sector (n; lam - alpha).  dims_small: exact weight-sector dimensions of SU(N-1)
    (fortuity.bmn.fullindex.partition_dims), which certify the target functionals.  Returns one dict per channel with
    the TargetSector data, the classify() verdicts and the residues (vectors modulo exact + graviton)."""
    from fortuity.bmn.blockrestrict import restrict_u1
    from fortuity.bmn.classify import TargetSector
    out = []
    for alpha in channels:
        alpha = tuple(alpha)
        sec = (n,) + tuple(l - a for l, a in zip(lam, alpha))
        T = TargetSector(N - 1, 'SU', sec, gen_nmax=gen_nmax, seed=tseed, log=log, maxlen=maxlen_small,
                         exact_dim=dims_small.get(sec, 0))
        assert T.dimV == dims_small.get(sec, 0), (T.dimV, dims_small.get(sec))
        cls, res, terms = [], [], []
        for rep in reps:
            img = restrict_u1(rep, N - 1, alpha)
            terms.append(len(img))
            if img:
                cls.append(T.classify(img))
                res.append(T.residue(img) % P)
            else:
                cls.append(dict(nonzero=False, closed=True, not_exact=False, non_graviton_component=False))
                res.append(np.zeros(T.pts.ncols, dtype=np.int64))
        log(f"  channel x^{alpha} -> SU({N - 1}) {sec}: dim {T.dimV}, exact {T.r_exact}, graviton {T.grav}; "
            f"non-graviton components {[c['non_graviton_component'] for c in cls]}")
        out.append(dict(alpha=list(alpha), sector=list(sec), dim=int(T.dimV), r_exact=int(T.r_exact),
                        grav=int(T.grav), terms=terms, classify=cls, residues=res))
    return out


def cores_from_residues(chan):
    """Rank of the restriction map on the non-graviton classes, and the combinations of representatives with zero
    residue in every channel (a basis of the cores, as rows of coefficients over the representatives)."""
    if not chan:
        return 0, None
    M = np.concatenate([np.stack(c['residues']) for c in chan], axis=1) % P      # (reps x all residue columns)
    from .hw import nullspace_left
    rank = _rank(M) if M.size else 0
    K = nullspace_left(M) if M.shape[0] else np.zeros((0, 0), dtype=np.int64)
    assert K.shape[0] == M.shape[0] - rank
    return rank, K
