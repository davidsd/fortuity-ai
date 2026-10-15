#!/usr/bin/env python3
"""compare_segments.py -- T-041: independent check of the segments with one negative charge (added 2026-10-02).

Compares indep_delta_r1.py at the points of points_segments.txt (merged by merge_grid.py --points) with the values of
outputs/extended/ray1.tsv ... ray12.tsv (physics/scan_extended.py on physics/classical_max.py, job 568139), which print
7 significant digits (mp.nstr(v, 7)).

Usage:  python compare_segments.py INDEP.tsv EXTDIR

EXTDIR must hold ray1.tsv ... ray12.tsv with 245 rows in all (ray0, theta = 0, lies in the triangle and is skipped).
Matching: rows are matched on (theta_deg, r) to 1e-6; every EXT point must have exactly one independent row and vice
versa. Rows of INDEP.tsv without a numeric theta_deg and r (e.g. a traceback in an unmerged chunk) are skipped and
counted.
Agreement, per point, for Q1/sQ, Q2/sQ, Q3/sQ, Sprot/K, Scl/K, Phi/pi, delta0 and r1:
  |indep - EXT| <= 0.6 ulp + 5e-14, where ulp = 10^(floor(log10|EXT|) - 6) is one unit in the 7th significant digit of
  the EXT value (half a unit from its rounding, plus 20% margin), and 5e-14 allows for EXT's working precision
  (mp.dps = 16 in physics/classical_max.py; it matters only for the smallest delta0, ~6e-7 at the end of ray 1).
  r1 is compared in two columns: EXT prints the envelope formula (2/3) Re[H/g] at the leading saddles; the independent code
  prints the same formula at its own saddles (column r1_env), and the slope of full finite-eps Legendre transforms
  of the same truncated model (column r1; run with a small --h near the arc). By the envelope theorem that slope
  equals r1_env, so it checks the implementation, not the formula; both must agree with EXT's r1.
  Also the independent code's own diagnostics: ncross = 0 (no polylog cut crossed on the continuation path from
  KKLP's saddle) and d2ReS/dbeta2 < 0 (the bonus extremum is a local maximum).
Printed: one line per point (differences in units of the EXT ulp), then the largest differences per column, the
ranges of the independent code's other checks (dB-dA, BH0, cubic, cut_cl, r1-r1_env), and the final line
"all N points agree with EXTDIR to its printed precision" or "N points: M disagree, ...". Exit status 0 only if all
245 agree.
"""
import math
import os
import sys

KEYS = ("Q1/sQ", "Q2/sQ", "Q3/sQ", "Sprot/K", "Scl/K", "Phi/pi", "delta0")
FLOOR = 5e-14            # EXT's working precision (classical_max.py: mp.dps = 16)
NEXT = 245               # points of outputs/extended/ray1..ray12.tsv


def read(path):
    rows, cols = [], None
    with open(path) as fh:
        for line in fh:
            if line.startswith('#') or not line.strip():
                continue
            tok = line.rstrip('\n').split('\t')
            if tok[0] == 'theta_deg':
                cols = tok
                continue
            rows.append(dict(zip(cols, tok)))
    return rows


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float('nan')


def ulp7(v):
    """One unit in the 7th significant digit of v (nan if v is not a finite non-zero number)."""
    if not math.isfinite(v) or v == 0:
        return float('nan')
    return 10.0 ** (math.floor(math.log10(abs(v))) - 6)


def main(indep_path, extdir):
    ext = []
    for k in range(1, 13):                # ray 0 (theta = 0) lies inside the triangle; not a segment
        path = os.path.join(extdir, "ray%d.tsv" % k)
        if not os.path.exists(path):
            print("compare_segments: %s is missing" % path, file=sys.stderr)
            return 1
        ext += read(path)
    if len(ext) != NEXT:
        print("compare_segments: %s holds %d rows in ray1..ray12, expected %d" % (extdir, len(ext), NEXT),
              file=sys.stderr)
        return 1
    indep = read(indep_path)
    key = lambda r: (round(num(r.get('theta_deg')), 6), round(num(r.get('r')), 6))
    idx, nskip = {}, 0
    for r in indep:
        k = key(r)
        if not (math.isfinite(k[0]) and math.isfinite(k[1])):
            nskip += 1
            continue
        idx.setdefault(k, []).append(r)
    cols = KEYS + ("r1", "r1_env")
    print("theta\tr\t" + "\t".join("d(%s)/ulp" % c for c in cols) + "\tncross\td2ReS/dbeta2\tstatus")
    nbad, nmiss, worst, nfloor, absd = 0, 0, {c: (0.0, None) for c in cols}, 0, {}
    diag = {c: [] for c in ("dB-dA", "BH0", "cubic", "cut_cl", "r1-r1_env")}
    for e in ext:
        k = key(e)
        got = idx.get(k, [])
        if len(got) != 1:
            nmiss += 1
            print("%s\t%s\t%d independent rows for this point" % (k[0], k[1], len(got)))
            continue
        i = got[0]
        ds, tol_ok = [], []
        for c in cols:
            ev = num(e.get('r1' if c == 'r1_env' else c))
            diff = num(i.get(c)) - ev
            u = ulp7(ev)
            ds.append(diff / u)
            tol_ok.append(math.isfinite(diff) and abs(diff) <= 0.6 * u + FLOOR)
            if math.isfinite(diff) and 0.6 * u < abs(diff) <= 0.6 * u + FLOOR:
                nfloor += 1
        ncross, d2 = num(i.get('ncross')), num(i.get('d2ReS/dbeta2'))
        bad = not all(tol_ok) or ncross != 0 or not d2 < 0
        nbad += bad
        for c, d in zip(cols, ds):
            if math.isfinite(d) and abs(d) > abs(worst[c][0]) + 1e-6:
                worst[c] = (d, k)
            if math.isfinite(d):
                absd.setdefault(c, []).append(abs(d))
        for c in diag:
            diag[c].append(num(i.get(c)))
        print("%s\t%s\t" % k + "\t".join("%+.2f" % d for d in ds) + "\t%s\t%s\t%s"
              % (i.get('ncross'), i.get('d2ReS/dbeta2'), "MISMATCH" if bad else "ok"))
    matched = {key(e) for e in ext}
    extra = sum(len(v) for k, v in idx.items() if k not in matched)
    print("# largest |difference| per column, in units of one EXT ulp (7th significant digit); 'ties' counts the points")
    print("# within 1e-6 ulp of it (the Q fractions, printed with 8 digits by the independent code, can reach exactly 0.5):")
    for c in cols:
        m = abs(worst[c][0])
        ties = sum(1 for x in absd.get(c, []) if abs(x - m) <= 1e-6)
        print("#   %-12s %+.4f at %s (ties: %d)" % (c, worst[c][0], worst[c][1], ties))
    print("#   differences within 0.6 ulp only thanks to the %.0e floor: %d" % (FLOOR, nfloor))
    for c, v in diag.items():
        fin = [x for x in v if math.isfinite(x)]
        print("#   independent %-10s range [%.2g, %.2g]%s" % (c, min(fin) if fin else float('nan'),
                                                            max(fin) if fin else float('nan'),
                                                            "" if len(fin) == len(v) else
                                                            " (%d non-finite: +inf means no cut reachable)"
                                                            % (len(v) - len(fin))))
    n = len(ext)
    if nbad == 0 and nmiss == 0 and extra == 0 and nskip == 0:
        print("all %d points agree with %s to its printed precision" % (n, extdir))
        return 0
    print("%d points: %d disagree, %d without exactly one independent row, %d independent rows unmatched, "
          "%d independent rows without numeric theta and r" % (n, nbad, nmiss, extra, nskip))
    return 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
