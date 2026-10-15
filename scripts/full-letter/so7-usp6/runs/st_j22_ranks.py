"""T-048 follow-up: large-N single-trace Q-cohomology dimension at (sec, Y) by ranks only,
dim H = #necklaces - rank(Q out) - rank(Q in), with fortuity.full.gravitons' necklace bases and Q matrices and the FLINT
rank kernel (FORTUITY_FFLA). Validated on sectors with known counts before the three level-22 targets of R-025."""
import sys, time
from fortuity.full.gravitons import necklaces_of_sector, _qmatrix, single_trace_cohomology
from fortuity.hwcoh import ffla
assert ffla.HAVE_FLINT


def st_dim(sec, Y):
    sec = tuple(sec)
    W = necklaces_of_sector(sec, Y)
    if not W:
        return 0, {}
    idx = {w: i for i, w in enumerate(W)}
    down = (sec[0] - 1, sec[1] - 1) + sec[2:]
    D = necklaces_of_sector(down, Y + 1)
    up = (sec[0] + 1, sec[1] + 1) + sec[2:]
    U = necklaces_of_sector(up, Y - 1) if Y > 1 else []
    r_out = ffla.rank(_qmatrix(W, {w: i for i, w in enumerate(D)})) if D else 0
    r_in = ffla.rank(_qmatrix(U, idx)) if U else 0
    return len(W) - r_out - r_in, dict(necklaces=len(W), down=len(D), up=len(U), rank_out=r_out, rank_in=r_in)


if __name__ == '__main__':
    t0 = time.time()
    checks = [((0, 0, 4, 4, 4), 6), ((0, 0, 6, 6, 6), 8), ((1, 1, 2, 2, 2), 2), ((0, 0, 8, 8, 6), 8), ((-2, 2, 8, 8, 6), 8)]
    for sec, Y in checks:
        d, info = st_dim(sec, Y)
        ref = single_trace_cohomology(sec, Y)[1].shape[0]
        print(f"check {sec} Y={Y}: ranks {d}, representatives {ref}, {info} ({time.time()-t0:.0f}s)", flush=True)
        assert d == ref
    for sec in [(1, 1, 10, 6, 6), (-1, 3, 8, 8, 6), (3, -1, 8, 8, 6), (1, 1, 8, 8, 6)]:
        for seedless in [0]:
            d, info = st_dim(sec, 8)
            print(f"level-22 target {sec} Y=8: single-trace cohomology {d}, {info} ({time.time()-t0:.0f}s)", flush=True)
