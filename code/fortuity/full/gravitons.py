"""Single-trace Q-cohomology of the full 1/16-BPS letters at large N (the graviton generators with derivatives and
gauginos), and the multi-graviton span test in a finite-N weight sector.

Large N: single-trace words are the nonvanishing canonical necklaces (fortuity.full.wordenum), with no trace relations;
Q maps necklaces to necklaces (fortuity.full.words.Q_word), lowering (2J1, 2J2) by one each and raising Y by one.
The cohomology of the single-trace complex at charges (2J1, 2J2; 2q) and letter number Y is computed mod p = 2^26-5
with explicit representatives (vectors in the necklace basis).  These are the single-graviton generators (D-005).

Finite N: in a weight sector sec at letter number Y, the multi-graviton classes are the span of all products of
generators whose charges add up to (sec, Y), modulo Q-exact operators; the count is
    rank([exact rows; product rows]) - rank(exact rows)
with rows evaluated at random group-valued points (fortuity.full.evalg.Frame, group = SU / SO_odd / Sp).
2026-10-08 (full-letter SO/USp development code); promoted to code/ in T-048 (2026-10-13).
Validated on both columns of ChangLin2025SDuality Table 1 for SO(7) and USp(6), including the monotone split.
Caveat: generators_up_to_level(jmax, Ymax) caps the letter number; single-graviton totals at level j need Ymax >= j/2
(at level 18, Ymax = 8 misses the 55 Y = 9 classes of [9,0]: 996 instead of 1051).
"""
import math, time
from collections import defaultdict
import numpy as np

from ..bmn.modp import P, RREFBasis
from ..hwcoh import ffla
from ..hwcoh.ffla import nullspace_right
from ..hwcoh.evalg import algebra_dim
from .words import Q_word, Q_multi, add_term, contents_of_sector, level_contents
from .wordenum import necklaces_with_content, multitraces_with_content
from .letters import is_ferm
from .evalg import Frame
from . import gdims
from .weyl import PERMS, weyl_expr, weyl_sector


def necklaces_of_sector(sec, Y):
    out = []
    for c in contents_of_sector(sec):
        if len(c) == Y:
            out.extend(necklaces_with_content(c))
    return sorted(set(out))


def _qmatrix(src_words, dst_index):
    """Rows = Q(src word) in the dst basis (mod p)."""
    M = np.zeros((len(src_words), len(dst_index)), dtype=np.int64)
    for i, w in enumerate(src_words):
        for c, cw in Q_word(w):
            j = dst_index.get(cw)
            if j is None:
                raise KeyError(f"Q image {cw} outside the destination basis")
            M[i, j] = (M[i, j] + c) % P
    return M


_GEN_CACHE = {}


def single_trace_cohomology(sec, Y):
    """(words, reps): necklace basis of the sector and representatives (rows mod p) of its large-N single-trace
    Q-cohomology."""
    key = (tuple(sec), Y)
    if key in _GEN_CACHE:
        return _GEN_CACHE[key]
    sec = tuple(sec)
    words = necklaces_of_sector(sec, Y)
    if not words:
        _GEN_CACHE[key] = (words, np.zeros((0, 0), dtype=np.int64))
        return _GEN_CACHE[key]
    index = {w: i for i, w in enumerate(words)}
    # Q out: into (2J1-1, 2J2-1; 2q) at Y+1
    down = (sec[0] - 1, sec[1] - 1) + sec[2:]
    dwords = necklaces_of_sector(down, Y + 1)
    dindex = {w: i for i, w in enumerate(dwords)}
    Qout = _qmatrix(words, dindex) if dwords else np.zeros((len(words), 0), dtype=np.int64)
    # kernel of Q out (row vectors v with v Qout = 0)
    if Qout.shape[1]:
        ker = nullspace_right(np.ascontiguousarray(Qout.T))     # rows: vectors in the word basis
    else:
        ker = np.eye(len(words), dtype=np.int64)
    # Q in: from (2J1+1, 2J2+1; 2q) at Y-1
    up = (sec[0] + 1, sec[1] + 1) + sec[2:]
    uwords = necklaces_of_sector(up, Y - 1) if Y > 1 else []
    im = _qmatrix(uwords, index) if uwords else np.zeros((0, len(words)), dtype=np.int64)
    # representatives: kernel vectors independent modulo the image (incremental RREF basis)
    reps = []
    basis = RREFBasis(len(words))
    for row in im:
        basis.add(row[None, :] % P)
    for v in ker:
        ok = basis.add((v % P)[None, :])
        if (ok[0] if isinstance(ok, tuple) else ok):
            reps.append(v % P)
    reps = np.array(reps, dtype=np.int64) if reps else np.zeros((0, len(words)), dtype=np.int64)
    _GEN_CACHE[key] = (words, reps)
    return _GEN_CACHE[key]


def _dominant_task(args):
    sec, Y = args
    words, reps = single_trace_cohomology(sec, Y)
    return sec, Y, [{words[i]: int(row[i]) for i in np.flatnonzero(row)} for row in reps]


def generators_up_to_level(jmax, Ymax, workers=None, log=None):
    """All single-trace generators with level <= jmax and letters <= Ymax, as (sec, Y, expr) with expr = {necklace: coeff}.
    Computed at dominant weights (2q1 >= 2q2 >= 2q3) and transported to the other weights by the SU(3) Weyl group
    (weyl.py: permutations of the theta labels commute with Q)."""
    import multiprocessing as mp, os
    tasks = []
    for j in range(2, jmax + 1):
        for sec, cs in level_contents(j).items():
            if not (sec[2] >= sec[3] >= sec[4]):
                continue
            for Y in sorted({len(c) for c in cs}):
                if 2 <= Y <= Ymax:
                    tasks.append((tuple(sec), Y))
    tasks.sort(key=lambda t: -(t[0][2] + t[0][3] + t[0][4]))
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    t0 = time.time()
    if workers > 1:
        with mp.get_context('fork').Pool(workers) as pool:
            results = pool.map(_dominant_task, tasks, chunksize=4)
    else:
        results = [_dominant_task(t) for t in tasks]
    if log:
        log(f"  dominant-weight single-trace cohomology: {len(tasks)} sectors, {sum(len(r[2]) for r in results)} classes ({time.time()-t0:.0f}s)")
    gens = []
    for sec, Y, exprs in results:
        images = set()
        for perm in PERMS:
            wsec = weyl_sector(sec, perm)
            if wsec in images:
                continue
            images.add(wsec)
            for e in exprs:
                we = weyl_expr({(w,): c for w, c in e.items()}, perm)
                if we:
                    gens.append((wsec, Y, {m[0]: c for m, c in we.items()}))
    return gens


def rep_expr(words, row):
    """Representative as {single-trace word: coeff} (legacy signature)."""
    return {words[i]: int(row[i]) for i in np.flatnonzero(row)}


def multiply(exprA, exprB):
    """Product of two multitrace expressions (dict multi -> coeff), canonicalised with Grassmann signs."""
    out = defaultdict(int)
    for ma, ca in exprA.items():
        for mb, cb in exprB.items():
            add_term(out, tuple(ma) + tuple(mb), (ca * cb) % P)
    return {m: c % P for m, c in out.items() if c % P}


def products_in_sector(gens, sec, Y, log=None):
    """All products (multisets) of generators with total charges sec and letter number Y, as multitrace expressions."""
    sec = tuple(sec)
    tgt = sec + (Y,)
    G = [(g[0] + (g[1],), g) for g in gens]
    # prune: level and Y are positive and additive
    G = [(ch, g) for ch, g in G if ch[2] + ch[3] + ch[4] <= sec[2] + sec[3] + sec[4] and ch[5] <= Y]
    G.sort(key=lambda t: (-(t[0][2] + t[0][3] + t[0][4]), t[0]))
    out = []

    def rec(i, rem, chosen):
        if all(x == 0 for x in rem):
            out.append(list(chosen)); return
        if rem[5] < 2 or rem[2] + rem[3] + rem[4] < 4:
            return
        for k in range(i, len(G)):
            ch, g = G[k]
            if ch[5] > rem[5] or ch[2] + ch[3] + ch[4] > rem[2] + rem[3] + rem[4]:
                continue
            if any(ch[t] > rem[t] for t in (2, 3, 4)):
                continue
            chosen.append(k)
            rec(k, tuple(r - c for r, c in zip(rem, ch)), chosen)
            chosen.pop()
    rec(0, tgt, [])
    exprs = []
    for combo in out:
        e = {(): 1}
        for k in combo:
            ch, g = G[k]
            e = multiply(e, {(w,): c for w, c in g[2].items()})
            if not e:
                break
        if e:
            exprs.append(e)
    if log:
        log(f"  products landing in {sec} Y={Y}: {len(out)} combinations, {len(exprs)} nonzero")
    return exprs


def graviton_count(group, N, sec, Y, gens, seed=1, margin=48, log=print, maxlen=None, cartan=False):
    """Weight-space counts in sector (sec, Y) at finite N: dim, exact, rank out, h, multi-graviton, non-graviton."""
    sec = tuple(sec)
    t0 = time.time()
    words = []
    for c in contents_of_sector(sec):
        if len(c) == Y:
            words.extend(multitraces_with_content(c, maxlen))
    words = sorted(set(words))
    idx = {w: i for i, w in enumerate(words)}
    dim = gdims.sector_dim(group, N, sec, Y)
    up = (sec[0] + 1, sec[1] + 1) + sec[2:]
    uwords = []
    if Y > 1:
        for c in contents_of_sector(up):
            if len(c) == Y - 1:
                uwords.extend(multitraces_with_content(c, maxlen))
    # exact rows: Q of the words one step up, in the word basis (Q may lengthen traces: extend the basis)
    def to_vec(expr):
        v = {}
        for m, c in expr.items():
            j = idx.get(m)
            if j is None:
                j = idx[m] = len(words); words.append(m)
            v[j] = (v.get(j, 0) + c) % P
        return v
    exact = [to_vec(Q_multi(w)) for w in uwords]
    exact = [v for v in exact if v]
    prods = [to_vec(e) for e in products_in_sector(gens, sec, Y, log=log)]
    prods = [v for v in prods if v]
    # evaluation frames by fermion degree
    def nferm(m): return sum(1 for w in m for L in w if is_ferm(L))
    deg = np.array([nferm(m) for m in words])
    letters = sorted({L for m in words for w in m for L in w})
    degs = sorted(set(deg.tolist()))
    kmax = 3 * algebra_dim(group, N)
    ncols_needed = dim + margin
    cols = []
    for d in degs:
        k = 0 if d == 0 else min(kmax, max(d + 2, next(kk for kk in range(d, kmax + 1) if math.comb(kk, d) >= 12 or kk == kmax)))
        cpp = math.comb(k, d) if d else 1
        npts = max(1, math.ceil(ncols_needed / cpp))
        cols.append((d, Frame(N, npts, k, [d], letters, seed=seed + 7919 * d, group=group)))
    ncol = sum(fr.ncols for _, fr in cols)
    X = np.zeros((len(words), ncol), dtype=np.int64)
    c0 = 0
    for d, fr in cols:
        wd = np.flatnonzero(deg == d)
        if wd.size:
            X[wd, c0:c0 + fr.ncols] = fr.eval_words([words[i] for i in wd])
        c0 += fr.ncols
    def rows_of(vecs):
        M = np.zeros((len(vecs), ncol), dtype=np.int64)
        for i, v in enumerate(vecs):
            for j, c in v.items():
                M[i] = (M[i] + c * X[j]) % P
        return M
    E = rows_of(exact); G = rows_of(prods)
    coulomb = None
    if cartan:
        ccols = []
        for d in degs:
            k = 0 if d == 0 else min(kmax, max(d + 2, next(kk for kk in range(d, kmax + 1) if math.comb(kk, d) >= 12 or kk == kmax)))
            cpp = math.comb(k, d) if d else 1
            ccols.append((d, Frame(N, max(1, math.ceil((len(prods) + margin) / cpp)), k, [d], letters, seed=seed + 31337 + 7919 * d, group=group, cartan=True)))
        cncol = sum(fr.ncols for _, fr in ccols)
        CX = np.zeros((len(words), cncol), dtype=np.int64); c0 = 0
        for d, fr in ccols:
            wd = np.flatnonzero(deg == d)
            if wd.size:
                CX[wd, c0:c0 + fr.ncols] = fr.eval_words([words[i] for i in wd])
            c0 += fr.ncols
        CG = np.zeros((len(prods), cncol), dtype=np.int64)
        for i, v in enumerate(prods):
            for j, c in v.items():
                CG[i] = (CG[i] + c * CX[j]) % P
        CE = np.zeros((len(exact), cncol), dtype=np.int64)
        for i, v in enumerate(exact):
            for j, c in v.items():
                CE[i] = (CE[i] + c * CX[j]) % P
        coulomb = int(ffla.rank(CG)) if CG.shape[0] else 0
        exact_on_cartan = int(ffla.rank(CE)) if CE.shape[0] else 0
    rank_all_words = ffla.rank(X)
    r_exact = ffla.rank(E) if E.shape[0] else 0
    r_eg = ffla.rank(np.vstack([E, G])) if G.shape[0] else r_exact
    # rank out: Q of this sector's words into the sector below, evaluated there
    down = (sec[0] - 1, sec[1] - 1) + sec[2:]
    dwords = []
    for c in contents_of_sector(down):
        if len(c) == Y + 1:
            dwords.extend(multitraces_with_content(c, maxlen))
    dwords = sorted(set(dwords)); didx = {w: i for i, w in enumerate(dwords)}
    r_out = 0
    if dwords:
        ddeg = np.array([nferm(m) for m in dwords]); dletters = sorted({L for m in dwords for w in m for L in w})
        ddim = gdims.sector_dim(group, N, down, Y + 1)
        dcols = []
        for d in sorted(set(ddeg.tolist())):
            k = 0 if d == 0 else min(kmax, max(d + 2, next(kk for kk in range(d, kmax + 1) if math.comb(kk, d) >= 12 or kk == kmax)))
            cpp = math.comb(k, d) if d else 1
            dcols.append((d, Frame(N, max(1, math.ceil((ddim + margin) / cpp)), k, [d], dletters, seed=seed + 104729 + 7919 * d, group=group)))
        dncol = sum(fr.ncols for _, fr in dcols)
        DX = np.zeros((len(dwords), dncol), dtype=np.int64); c0 = 0
        for d, fr in dcols:
            wd = np.flatnonzero(ddeg == d)
            if wd.size:
                DX[wd, c0:c0 + fr.ncols] = fr.eval_words([dwords[i] for i in wd])
            c0 += fr.ncols
        Qo = np.zeros((len(words), dncol), dtype=np.int64)
        for i, m in enumerate(words):
            for m2, c in Q_multi(m).items():
                j = didx.get(m2)
                if j is None:
                    raise RuntimeError("Q image outside the enumerated words of the sector below (increase maxlen)")
                Qo[i] = (Qo[i] + c * DX[j]) % P
        r_out = ffla.rank(Qo)
    h = dim - r_exact - r_out
    grav = r_eg - r_exact
    res = dict(group=group, sec=list(sec), Y=Y, dim=dim, rank_words=int(rank_all_words), exact=int(r_exact), rank_out=int(r_out),
               h=int(h), graviton=int(grav), nongraviton=int(h - grav), n_products=len(prods), n_generators=len(gens),
               time=round(time.time() - t0, 1))
    if cartan:
        res['coulomb_graviton'] = coulomb; res['noncoulomb_graviton'] = int(grav - coulomb); res['exact_on_cartan'] = exact_on_cartan
    if log:
        log(f"  {group} {sec} Y={Y}: dim {dim} (words rank {rank_all_words}), exact {r_exact}, out {r_out}, h {h}, graviton {grav}, non-graviton {h - grav}" + (f", Coulomb-type gravitons {coulomb} (exact rows on the Cartan: rank {exact_on_cartan})" if cartan else "") + f" ({res['time']}s)")
    return res
