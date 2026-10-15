"""Driver of R-001: SU(2) BMN-sector direct Q-cohomology per charge sector through level JMAX, with the
multi-graviton subspace (generators u_2, v_2, w_2).  This is the production command of
data/su2/bmn_direct_cohomology_j28.json (promoted from scripts/bmn/index-and-direct-cohomology/run_su2_direct.py;
review C14).

usage: python -m fortuity.bmn.run_su2_direct JMAX [OUT.json] [--check DATA.json]
  --check: compare every computed sector with the stored data file and exit nonzero on any difference.

SU(2) only.  The `Direct` engine fixes the number of evaluation points per sector without a rank-saturation guard;
that is exact for N = 2 (all 683 dimensions of the stored file agree with the exact Haar-integral dimensions) but can
undercount for N >= 3 (review B.2), where fortuity.bmn.symdirect / fortuity.hwcoh must be used instead.
"""
import io
import json
import sys
from collections import defaultdict
from contextlib import redirect_stdout
from itertools import permutations

from fortuity.bmn.direct import Direct, sectors_up_to

FIELDS = ('nwords', 'dimV', 'rQ', 'rQup', 'coh', 'grav', 'nongrav')


def run(jmax, quiet=True):
    """List of per-sector records (dicts as returned by Direct.analyse) for all sectors with j <= jmax."""
    D = Direct(N=2, maxlen=3, gen_nmax=2)
    out = []
    for sec in sectors_up_to(jmax):
        if quiet:
            with redirect_stdout(io.StringIO()):
                r = D.analyse(sec)
        else:
            r = D.analyse(sec)
        r = dict(r)
        r['sector'] = list(r['sector'])
        out.append(r)
    return out


def summary(res):
    """{j: (classes, gravitons, non-gravitons, signed non-graviton count)} summed over SU(3) flavour images."""
    byj = defaultdict(lambda: [0, 0, 0, 0])
    for r in res:
        sec = r['sector']
        mult = len(set(permutations(sec[1:])))
        sign = -1 if sec[0] % 2 else 1
        byj[r['j']][0] += mult * r['coh']
        byj[r['j']][1] += mult * r['grav']
        byj[r['j']][2] += mult * r['nongrav']
        byj[r['j']][3] += mult * sign * r['nongrav']
    return {j: tuple(v) for j, v in sorted(byj.items())}


def compare(res, datafile):
    """Differences between computed records and the stored data file (same sectors, fields FIELDS)."""
    stored = {tuple(s['sector']): s for s in json.load(open(datafile))['sectors']}
    diffs = []
    for r in res:
        s = stored.get(tuple(r['sector']))
        if s is None:
            diffs.append((r['sector'], 'missing in data'))
            continue
        for f in FIELDS:
            if r[f] != s[f]:
                diffs.append((r['sector'], f, r[f], s[f]))
    return diffs


def main(argv):
    jmax = int(argv[1])
    args = argv[2:]
    check = None
    if '--check' in args:
        i = args.index('--check')
        check = args[i + 1]
        args = args[:i] + args[i + 2:]
    res = run(jmax, quiet=False)
    if args:
        json.dump(res, open(args[0], 'w'), indent=1)
    print(" j   #coh  #grav  #BH  index(BH)")
    for j, v in summary(res).items():
        print(f"{j:3d} {v[0]:6d} {v[1]:6d} {v[2]:4d} {v[3]:5d}")
    if check:
        diffs = compare(res, check)
        print(f"compared {len(res)} sectors with {check}: {len(diffs)} differences")
        for d in diffs[:20]:
            print("  DIFF", d)
        return 1 if diffs else 0
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
