#!/usr/bin/env python3
"""compare_scan.py -- T-041: compare indep_delta_r1.py output with the production scan of
t038/rf/scan_kklp_charges.py scan 12 (both tab-separated, the scan with header line
theta_deg r xi1 xi2 xi3 Q1/sQ Q2/sQ Q3/sQ Sprot/K Scl/K Phi/pi delta0 r1, all printed with %.5f).

Usage:  python compare_scan.py SCAN.tsv INDEP.tsv
Rows are matched on (theta_deg, r) to 1e-4 (the scan prints 5 decimals).  For each matched point the
differences in Q_I/sQ, Sprot/K, Scl/K, Phi/pi, delta0 and r1 are printed.  Since the scan prints 5 decimals, an
agreement is |diff| <= 6e-6 (half a unit of the last printed digit plus margin); larger differences are flagged
'MISMATCH', and so is a missing, non-numeric or non-finite value on either side (e.g. the '-' that
indep_delta_r1.py --no-r1 writes for r1), which earlier versions let pass as agreement (fixed 2026-10-02 after
review).  Off-grid points of INDEP.tsv are listed as unmatched.  Expected output when both codes agree: every
matched row 'ok', and the final line 'all N matched points agree to the scan's printed precision'.
"""
import math
import sys

KEYS = ("Q1/sQ", "Q2/sQ", "Q3/sQ", "Sprot/K", "Scl/K", "Phi/pi", "delta0", "r1")


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
    except ValueError:
        return float('nan')


def main(scan_path, indep_path):
    scan, indep = read(scan_path), read(indep_path)
    index = {(round(num(r['theta_deg']), 3), round(num(r['r']), 4)): r for r in scan}
    nmatch, nbad = 0, 0
    print("theta\tr\t" + "\t".join("d(%s)" % k for k in KEYS) + "\tstatus")
    for r in indep:
        key = (round(num(r['theta_deg']), 3), round(num(r['r']), 4))
        s = index.get(key)
        if s is None:
            print("%s\t%s\tunmatched (not on the scan grid, or not reached by the scan): delta0=%s r1=%s"
                  % (r['theta_deg'], r['r'], r['delta0'], r['r1']))
            continue
        nmatch += 1
        d = [num(r.get(k, '')) - num(s.get(k, '')) for k in KEYS]
        bad = any(not math.isfinite(x) or abs(x) > 6e-6 for x in d)
        nbad += bad
        print("%s\t%s\t" % key + "\t".join("%.1e" % x for x in d) + "\t" + ("MISMATCH" if bad else "ok"))
    if nmatch and not nbad:
        print("all %d matched points agree to the scan's printed precision" % nmatch)
    else:
        print("%d matched, %d mismatched" % (nmatch, nbad))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    main(sys.argv[1], sys.argv[2])
