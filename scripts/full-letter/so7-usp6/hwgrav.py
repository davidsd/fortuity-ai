"""Multi-graviton count in an SU(3)_R highest-weight sector of the full-letter complex (for sectors whose weight space is
too large for fullgrav.graviton_count).  Products of single-trace generators landing in the weight sector are projected
to highest weight (left null space of the raising operators restricted to the product span) and evaluated with the frames
of sector_group.SectorG; the count is rank([Q-images hw rows; graviton hw rows]) - rank(Q-images hw rows).
usage: python hwgrav.py GROUP N 2J1,2J2,2q1,2q2,2q3 Y GENS.pkl [SEED]
"""
import sys, time, pickle
import numpy as np
from collections import defaultdict
from fortuity.bmn.modp import P, matmul_mod, RREFBasis
from fortuity.hwcoh import ffla
from fortuity.full.hw import E_multi, nullspace_left
from sector_group import SectorG
from fortuity.full.gravitons import products_in_sector


def hw_graviton_count(group, N, sec, Y, gens, maxlen=None, seed=1, log=print):
    sec = tuple(sec)
    maxlen = maxlen or Y
    S = SectorG(group, N, sec, Y, maxlen, seed=seed, log=log)
    t0 = time.time()
    S.prepare()
    if S.m == 0:
        return dict(group=group, sec=list(sec), Y=Y, m=0, h=0, graviton=0, nongraviton=0)
    idx = S.idx
    prods = products_in_sector(gens, sec, Y, log=log)
    # product vectors in the word basis (words may be extended by Q-images already; products are words of the sector)
    rows = []
    for e in prods:
        v = {}
        for m, c in e.items():
            j = idx.get(m)
            if j is None:
                j = idx[m] = len(S.ws); S.ws.append(m)
            v[j] = (v.get(j, 0) + c) % P
        if v:
            rows.append(v)
    nP = len(rows)
    log(f"  products: {nP}; projecting to highest weight")
    # raising operators on the products, EVALUATED at random points of the target sectors (the highest-weight
    # condition holds modulo trace relations, so a formal kernel over words would miss combinations): M_i rows =
    # E_i(product) evaluated in the sector of weight lambda + alpha_i; the hw combinations are the left null space.
    from fortuity.full.evalg import Frame
    from fortuity.full.letters import is_ferm
    from fortuity.full.sector import frame_k
    from fortuity.hwcoh.evalg import algebra_dim
    import math
    def nferm(m): return sum(1 for w in m for L in w if is_ferm(L))
    # rank certificate for the raising-operator matrix: double the evaluation points until the rank is stable
    def build_blocks(factor):
        blocks = []
        for i in (1, 2):
            blocks.append(_eblock(i, factor))
        return [b for b in blocks if b is not None]
    E_exprs = {}
    EXACT_T = {}
    def _eblock(i, factor):
        exprs = E_exprs.get(i)
        if exprs is None:
            exprs = []
            for v in rows:
                d = defaultdict(int)
                for j, c in v.items():
                    for m2, c2 in E_multi(S.ws[j], i).items():
                        d[m2] = (d[m2] + c * c2) % P
                exprs.append({k: c for k, c in d.items() if c})
            E_exprs[i] = exprs
        # exact operators of the target sector (Q-images of the words of the sector above it, Y-1 letters): the hw
        # condition on cohomology classes is E_i(v) in exact, not E_i(v) = 0
        from sector_group import fast_contents_of_sector
        from fortuity.full.wordenum import multitraces_with_content
        from fortuity.full.words import Q_multi
        alpha = (2, -2, 0) if i == 1 else (0, 2, -2)
        tsec = (sec[0], sec[1], sec[2] + alpha[0], sec[3] + alpha[1], sec[4] + alpha[2])
        usec = (tsec[0] + 1, tsec[1] + 1) + tsec[2:]
        if i not in EXACT_T:
            uw = [w for c in fast_contents_of_sector(usec) if len(c) == Y - 1 for w in multitraces_with_content(c)]
            EXACT_T[i] = [dict(Q_multi(w)) for w in uw]
            EXACT_T[i] = [e for e in EXACT_T[i] if e]
        exact_t = EXACT_T[i]
        ewords = sorted({m for e in exprs for m in e} | {m for e in exact_t for m in e})
        if not ewords:
            return None
        eidx = {m: t for t, m in enumerate(ewords)}
        edeg = np.array([nferm(m) for m in ewords]); eletters = sorted({L for m in ewords for w in m for L in w})
        kmax = 3 * algebra_dim(group, N)
        frames = []
        for d in sorted(set(edeg.tolist())):
            k = 0 if d == 0 else min(kmax, max(d + 2, next(kk for kk in range(d, kmax + 1) if math.comb(kk, d) >= 12 or kk == kmax)))
            cpp = math.comb(k, d) if d else 1
            frames.append((d, Frame(N, max(1, math.ceil(factor * (nP + 48) / cpp)), k, [d], eletters, seed=seed + 2718 * i + 7919 * d + 100003 * factor, group=group)))
        ncolE = sum(fr.ncols for _, fr in frames)
        EX = np.zeros((len(ewords), ncolE), dtype=np.int64); c0 = 0
        for d, fr in frames:
            wd = np.flatnonzero(edeg == d)
            if wd.size:
                EX[wd, c0:c0 + fr.ncols] = fr.eval_words([ewords[t] for t in wd])
            c0 += fr.ncols
        Mi = np.zeros((nP, ncolE), dtype=np.int64)
        for r, e in enumerate(exprs):
            for m, c in e.items():
                Mi[r] = (Mi[r] + c * EX[eidx[m]]) % P
        # reduce modulo the exact subspace of the target sector
        if exact_t:
            Xe = np.zeros((len(exact_t), ncolE), dtype=np.int64)
            for r, e in enumerate(exact_t):
                for m, c in e.items():
                    Xe[r] = (Xe[r] + (c % P) * EX[eidx[m]]) % P
            basis = RREFBasis(ncolE)
            basis.add_batched(Xe)
            Mi = basis.reduce(Mi) % P
            log(f"  E_{i}: target {tsec}: {len(exact_t)} exact rows (rank {basis.rank}); E-images reduced modulo them")
        return Mi
    blocks = build_blocks(1)
    if blocks:
        M = np.hstack(blocks); rM = ffla.rank(M)
        for factor in (2, 4, 8):
            M2 = np.hstack(build_blocks(factor)); r2 = ffla.rank(M2)
            log(f"  raising-operator matrix rank: {rM} (x1) vs {r2} (x{factor})")
            if r2 == rM:
                M = M2; break
            rM, M = r2, M2
        Nmat = nullspace_left(M)
    else:
        Nmat = np.eye(nP, dtype=np.int64)
    if False:
      for i in (1, 2):
        exprs = []
        for v in rows:
            d = defaultdict(int)
            for j, c in v.items():
                for m2, c2 in E_multi(S.ws[j], i).items():
                    d[m2] = (d[m2] + c * c2) % P
            exprs.append({k: c for k, c in d.items() if c})
        ewords = sorted({m for e in exprs for m in e})
        if not ewords:
            continue
        eidx = {m: t for t, m in enumerate(ewords)}
        edeg = np.array([nferm(m) for m in ewords]); eletters = sorted({L for m in ewords for w in m for L in w})
        kmax = 3 * algebra_dim(group, N)
        frames = []
        for d in sorted(set(edeg.tolist())):
            k = 0 if d == 0 else min(kmax, max(d + 2, next(kk for kk in range(d, kmax + 1) if math.comb(kk, d) >= 12 or kk == kmax)))
            cpp = math.comb(k, d) if d else 1
            frames.append((d, Frame(N, max(1, math.ceil(factor * (nP + 48) / cpp)), k, [d], eletters, seed=seed + 2718 * i + 7919 * d + 100003 * factor, group=group)))
        ncolE = sum(fr.ncols for _, fr in frames)
        EX = np.zeros((len(ewords), ncolE), dtype=np.int64); c0 = 0
        for d, fr in frames:
            wd = np.flatnonzero(edeg == d)
            if wd.size:
                EX[wd, c0:c0 + fr.ncols] = fr.eval_words([ewords[t] for t in wd])
            c0 += fr.ncols
        Mi = np.zeros((nP, ncolE), dtype=np.int64)
        for r, e in enumerate(exprs):
            for m, c in e.items():
                Mi[r] = (Mi[r] + c * EX[eidx[m]]) % P
        blocks.append(Mi)
    if blocks:
        M = np.hstack(blocks)
        Nmat = nullspace_left(M)            # rows: combinations of products that are highest weight (as functions)
    else:
        Nmat = np.eye(nP, dtype=np.int64)
    log(f"  hw graviton combinations: {Nmat.shape[0]} of {nP} products ({time.time()-t0:.0f}s)")
    # evaluate: build the sector's A/B (hw rows, Q-image rows) in frames, then the graviton hw rows in the same frames
    A, B = S.evaluate(factor=1, attempt=0)
    # product word matrix in frame columns: reuse S.colblocks frames
    ncol = A.shape[1]
    Xp = np.zeros((nP, ncol), dtype=np.int64)
    col = 0
    deg = np.array([sum(1 for w in m for L in w if __import__('fortuity.full.letters', fromlist=['x']).is_ferm(L)) for m in S.ws], dtype=np.int64)
    for d, fr in S.colblocks:
        need = sorted({j for v in rows for j in v if deg[j] == d})
        if need:
            X = fr.eval_words([S.ws[j] for j in need]).astype(np.int64)   # int32 from the frame: cast before multiplying mod p
            pos = {j: t for t, j in enumerate(need)}
            for r, v in enumerate(rows):
                for j, c in v.items():
                    if deg[j] == d:
                        Xp[r, col:col + fr.ncols] = (Xp[r, col:col + fr.ncols] + c * X[pos[j]]) % P
        col += fr.ncols
    G = matmul_mod(Nmat % P, Xp)
    rB = ffla.rank(B) if B.shape[0] else 0
    rBG = ffla.rank(np.vstack([B, G])) if G.shape[0] else rB
    # cohomology of the sector: m - rank in - rank out needs the sector below; here report m, exact (=rB) and graviton
    res = dict(group=group, sec=list(sec), Y=Y, m=int(S.m), exact=int(rB), graviton=int(rBG - rB), n_products=nP,
               hw_products=int(Nmat.shape[0]), time=round(time.time() - t0, 1))
    log(f"  {group} {sec} Y={Y}: m {S.m}, exact {rB}, hw graviton classes {rBG - rB} ({res['time']}s)")
    return res


if __name__ == '__main__':
    group, N = sys.argv[1], int(sys.argv[2])
    sec = tuple(int(x) for x in sys.argv[3].split(',')); Y = int(sys.argv[4])
    gens = pickle.load(open(sys.argv[5], 'rb'))
    seed = int(sys.argv[6]) if len(sys.argv) > 6 else 1
    print(hw_graviton_count(group, N, sec, Y, gens, seed=seed))
