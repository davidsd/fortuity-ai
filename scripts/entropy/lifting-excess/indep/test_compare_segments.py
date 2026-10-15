"""Test compare_segments.py on synthetic input (pure logic, no physics solver; added 2026-10-02).

A fake "independent" file is built from the 245 rows of outputs/extended/ray1..ray12.tsv: the same values, with
r1_env = r1 = EXT's r1, ncross = 0, d2ReS/dbeta2 = -1 and the other diagnostic columns 0. compare_segments.py must
report all 245 points in agreement (exit 0), and must flag each damaged copy (exit 1): a delta0 moved by 1 unit in
the 7th significant digit at a typical point and by 1.2 units at the smallest delta0 (where the 5e-14 floor widens the
tolerance to 1.1 units, so a 1-unit move passes there), a negative Q3/sQ moved by 0.61 units, r1 or r1_env alone moved
by 1 unit, a dropped row, a duplicated row, '-' in r1, ncross = 1, d2ReS/dbeta2 > 0, a traceback line in the input,
and a missing ray file. Moves by 0.5 units must pass. Usage: python test_compare_segments.py.  Runtime: about a
second.
"""
import math
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.join(HERE, '..', 'outputs', 'extended')
COLS = ["theta_deg", "r", "xi1", "xi2", "xi3", "Q1/sQ", "Q2/sQ", "Q3/sQ", "Sprot/K", "Scl/K", "Phi/pi", "delta0",
        "r1", "r1_env", "r1-r1_env", "r1_os-r1_env", "r1cub-r1", "dB-dA", "c2_trunc", "beta*", "d2ReS/dbeta2",
        "dcbar/deps", "BH0", "cubic", "BHfin", "cut_cl", "cut_pr", "|Phi-2pi k|", "argf/pi", "ncross", "time_s"]


def ext_rows():
    rows = []
    for k in range(1, 13):
        cols = None
        for line in open(os.path.join(EXT, "ray%d.tsv" % k)):
            if line.startswith('#') or not line.strip():
                continue
            tok = line.rstrip('\n').split('\t')
            if tok[0] == 'theta_deg':
                cols = tok
                continue
            rows.append(dict(zip(cols, tok)))
    return rows


def fake(rows):
    out = []
    for e in rows:
        d = {c: "0" for c in COLS}
        for c in ("theta_deg", "r", "Q1/sQ", "Q2/sQ", "Q3/sQ", "Sprot/K", "Scl/K", "Phi/pi", "delta0", "r1"):
            d[c] = e[c]
        d["r1_env"] = e["r1"]
        d["d2ReS/dbeta2"] = "-1"
        out.append(d)
    return out


def write(path, rows, extra_lines=()):
    with open(path, "w") as fh:
        fh.write("\t".join(COLS) + "\n")
        for d in rows:
            fh.write("\t".join(d[c] for c in COLS) + "\n")
        for l in extra_lines:
            fh.write(l + "\n")


def run(path, extdir=EXT):
    p = subprocess.run([sys.executable, '-B', os.path.join(HERE, 'compare_segments.py'), path, extdir],
                       capture_output=True, text=True)
    lines = (p.stdout + p.stderr).strip().splitlines()
    return p.returncode, lines[-1] if lines else ""


def shifted(v, units):
    x = float(v)
    u = 10.0 ** (math.floor(math.log10(abs(x))) - 6)
    return repr(x + units * u)


def main():
    ok = True
    base = fake(ext_rows())
    i_small = min(range(len(base)), key=lambda i: float(base[i]["delta0"]))
    i_neg = min(range(len(base)), key=lambda i: float(base[i]["Q3/sQ"]))
    with tempfile.TemporaryDirectory() as tmp:
        cases = [("unchanged", lambda R: R, [], 0)]

        def mod(i, c, v):
            def f(R):
                R = [dict(d) for d in R]
                R[i][c] = v
                return R
            return f
        cases += [
            ("delta0 +0.5 unit at the smallest delta0", mod(i_small, "delta0", shifted(base[i_small]["delta0"], 0.5)), [], 0),
            ("delta0 +1 unit at the smallest delta0 (floor: passes)", mod(i_small, "delta0", shifted(base[i_small]["delta0"], 1.0)), [], 0),
            ("delta0 +1.2 units at the smallest delta0", mod(i_small, "delta0", shifted(base[i_small]["delta0"], 1.2)), [], 1),
            ("delta0 +1 unit at a typical point (row 0)", mod(0, "delta0", shifted(base[0]["delta0"], 1.0)), [], 1),
            ("Q3/sQ +0.61 unit at the most negative Q3", mod(i_neg, "Q3/sQ", shifted(base[i_neg]["Q3/sQ"], 0.61)), [], 1),
            ("Q3/sQ -0.5 unit at the most negative Q3", mod(i_neg, "Q3/sQ", shifted(base[i_neg]["Q3/sQ"], -0.5)), [], 0),
            ("finite-eps r1 alone +1 unit (r1_env unchanged)", mod(7, "r1", shifted(base[7]["r1"], 1.0)), [], 1),
            ("r1_env alone +1 unit (r1 unchanged)", mod(7, "r1_env", shifted(base[7]["r1_env"], 1.0)), [], 1),
            ("'-' in r1", mod(3, "r1", "-"), [], 1),
            ("ncross = 1", mod(5, "ncross", "1"), [], 1),
            ("d2ReS/dbeta2 = +0.1", mod(9, "d2ReS/dbeta2", "0.1"), [], 1),
            ("a dropped row", lambda R: R[:100] + R[101:], [], 1),
            ("a duplicated row", lambda R: R + [dict(R[40])], [], 1),
            ("a traceback line in the input", lambda R: R, ["Traceback (most recent call last):"], 1),
        ]
        for label, f, extra, want in cases:
            path = os.path.join(tmp, "indep.tsv")
            write(path, f(base), extra)
            code, last = run(path)
            good = code == want
            ok &= good
            print("%-50s exit %d (want %d) %s\n   %s" % (label, code, want, "ok" if good else "WRONG", last))
        # a missing ray file
        d = os.path.join(tmp, "ext")
        shutil.copytree(EXT, d)
        os.remove(os.path.join(d, "ray7.tsv"))
        write(os.path.join(tmp, "indep.tsv"), base)
        code, last = run(os.path.join(tmp, "indep.tsv"), d)
        good = code == 1
        ok &= good
        print("%-50s exit %d (want 1) %s\n   %s" % ("ray7.tsv missing", code, "ok" if good else "WRONG",
                                                   last.replace(tmp, "<tmp>")))
    print("test_compare_segments %s" % ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
