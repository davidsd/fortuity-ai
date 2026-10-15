"""Merge the chunks of one run_indep_grid.sbatch job into indep_grid12.tsv, after checking that they belong together.

Usage:  python merge_grid.py OUTDIR K [--grid 12] [--rmax0 1.0] [--out PATH]
        python merge_grid.py OUTDIR K --points FILE --prefix NAME [--out PATH]

K is the chunk count printed in the job's Slurm log ("chunks K=48"); --grid and --rmax0 must be those of the run
(the job script uses 12 and RMAX0, default 1.0, also printed in the log).
The second form merges a run of indep_delta_r1.py --file FILE --chunk k/K whose chunks are OUTDIR/NAME<k>.tsv, e.g.
run_indep_segments.sbatch: --points points_segments.txt --prefix seg_chunk (added 2026-10-02). The expected points
are then those of FILE, read as indep_delta_r1.py reads them, and the default output is OUTDIR/indep_<FILE stem>.tsv.

Checks, before anything is written:
  - grid12_chunk0.tsv ... grid12_chunk{K-1}.tsv (or NAME0.tsv ...) are all present, and OUTDIR holds no other chunk
    of that name (a leftover chunk from an earlier run with a larger K would otherwise enter the merge);
  - each chunk has the column header of indep_delta_r1.py and no '# fail' line;
  - every data row has all the header's columns, each a finite number, except that cut_cl and cut_pr may be +inf
    (no polylog argument can reach a cut). A '-' (written for r1 and its diagnostics by --no-r1), nan or a
    truncated row is rejected;
  - chunk k holds exactly the points k, k+K, k+2K, ... of grid_points(grid, rmax0) (or of FILE), in that order (theta
    and r compared to 1e-9), so a chunk written by a run with a different K, grid or point list is rejected;
  - together the chunks hold every point once.
Then it writes the column header and all rows sorted by (theta, r), the same file as the shell merge in the header of
run_indep_grid.sbatch, to PATH (default OUTDIR/indep_grid12.tsv). On any failed check it exits with status 1 and
writes nothing. Runtime: under a second. Tested by test_merge_grid.py (output: local_test_merge_grid.out).

Added 2026-10-02 after review: the job script's own merge (a wildcard over grid12_chunk*.tsv) is only safe in a
fresh OUTDIR, which is why the rerun instructions give every run its own OUT.
"""
import glob
import math
import os
import re
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from indep_delta_r1 import COLS, grid_points, parse_line, set_precision, to_mp, xi_to_polar  # noqa: E402


DIAG_INF = ("cut_cl", "cut_pr")      # distance to a polylog cut: +inf when no argument can reach one


def fail(msg):
    print("merge_grid: " + msg, file=sys.stderr)
    sys.exit(1)


def bad_field(line):
    """None if the data row has every column as a finite number (+inf allowed in DIAG_INF), else the reason."""
    tok = line.split("\t")
    if len(tok) != len(COLS):
        return "%d fields, the header has %d" % (len(tok), len(COLS))
    for c, t in zip(COLS, tok):
        try:
            v = float(t)
        except ValueError:
            return "column %s is %r, not a number" % (c, t)
        if math.isnan(v) or (math.isinf(v) and not (c in DIAG_INF and v > 0)):
            return "column %s is %r" % (c, t)
    return None


def main(argv):
    args, opts, i = [], dict(grid=12, rmax0='1.0', out=None, points=None, prefix=None), 0
    while i < len(argv):
        if argv[i] in ('--grid', '--rmax0', '--out', '--points', '--prefix'):
            opts[argv[i][2:]] = argv[i + 1]
            i += 2
        else:
            args.append(argv[i])
            i += 1
    if len(args) != 2:
        print(__doc__)
        return 2
    outdir, K = args[0], int(args[1])
    header = "\t".join(COLS)
    set_precision(25)                    # the run's default dps; sets the constants grid_points uses
    if opts['points']:
        if not opts['prefix']:
            fail("--points needs --prefix (the chunk file name before <k>.tsv)")
        prefix = opts['prefix']
        pts = []
        with open(opts['points']) as fh:          # as indep_delta_r1.py --file reads it
            for line in fh:
                tok = line.split('#')[0].split()
                if tok:
                    pts.append(parse_line(tok))
        stem = os.path.splitext(os.path.basename(opts['points']))[0]
        default_out = os.path.join(outdir, "indep_%s.tsv" % stem)
    else:
        prefix = opts['prefix'] or "grid12_chunk"
        pts = [xi for label, xi in grid_points(int(opts['grid']), to_mp(opts['rmax0']))]
        default_out = os.path.join(outdir, "indep_grid12.tsv")
    expected = []
    for xi in pts:
        th, r = xi_to_polar(xi)
        expected.append((float(th), float(r)))

    present = {}
    for path in glob.glob(os.path.join(outdir, prefix + "*.tsv")):
        m = re.fullmatch(re.escape(prefix) + r"(\d+)\.tsv", os.path.basename(path))
        if m:
            present[int(m.group(1))] = path
    missing = [k for k in range(K) if k not in present]
    extra = sorted(k for k in present if k >= K)
    if missing:
        fail("missing chunks %s of K=%d in %s" % (missing, K, outdir))
    if extra:
        fail("chunks %s are not part of a K=%d run (left over from another run?); use a fresh OUT" % (extra, K))

    rows = []
    for k in range(K):
        with open(present[k]) as fh:
            lines = fh.read().splitlines()
        if header not in lines:
            fail("%s has no indep_delta_r1.py column header" % present[k])
        if any(l.startswith("# fail") for l in lines):
            fail("%s reports failed points" % present[k])
        data = [l for l in lines if l and not l.startswith("#") and l != header]
        for i, l in enumerate(data):
            why = bad_field(l)
            if why:
                fail("%s, data row %d: %s" % (present[k], i + 1, why))
        want = expected[k::K]
        got = [(float(l.split("\t")[0]), float(l.split("\t")[1])) for l in data]
        if len(got) != len(want) or any(abs(a[0] - b[0]) > 1e-9 or abs(a[1] - b[1]) > 1e-9 for a, b in zip(got, want)):
            fail("%s does not hold points %d::%d of the point list (%d rows, %d expected)" % (present[k], k, K,
                                                                                             len(got), len(want)))
        rows += data
    keys = [(float(l.split("\t")[0]), float(l.split("\t")[1])) for l in rows]
    if len(set(keys)) != len(expected) or len(rows) != len(expected):
        fail("the chunks hold %d rows (%d distinct points), the point list has %d" % (len(rows), len(set(keys)),
                                                                                     len(expected)))
    rows = [l for _, l in sorted(zip(keys, rows))]
    out = opts['out'] or default_out
    with open(out, "w") as fh:
        fh.write(header + "\n" + "\n".join(rows) + "\n")
    print("merge_grid: %d chunks, %d points -> %s" % (K, len(rows), out))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
