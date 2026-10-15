"""Complete full-letter levels for any group: scan every dominant SU(3)_R column at given (2J1, 2J2) and every letter
number Y, then report the cohomology, compare two groups, and check weight-space numbers (Kostka-weighted sums).
From the full-letter SO/USp development code (scan_level.py, level_report.py, census_numbers.py, table1_check.py, 2026-10-08);
promoted in T-048.

h(sec, Y) = m(sec, Y) - r_up(sec, Y) - r_up(sec_below, Y + 1), sec_below = (2J1 - 1, 2J2 - 1, 2q): Q raises Y by one
and lowers both 2J by one; r_up(s) is the rank of Q into s (the exact part).

CLI (from code/):
  python -m fortuity.full.levels scan GROUP N J OUT.jsonl J1,J2 [J1,J2 ...]   (or a pairs .json); env MAXLEN (default 8)
  python -m fortuity.full.levels census J OUT.json GROUP:SCAN.jsonl[,SCAN2...] GROUP2:...
  python -m fortuity.full.levels weight SCAN.jsonl J1,J2 mu1,mu2,mu3 Y          (doubled weight mu)
"""
import collections
import json
import os
import sys
import time

from .words import contents_of_sector


def kostka(lam, mu):
    """Multiplicity of the weight mu in the GL(3) irrep with highest weight lam (3 parts): Gelfand-Tsetlin patterns."""
    a, b, c = lam
    cnt = 0
    for x in range(b, a + 1):
        for y in range(c, b + 1):
            for z in range(y, x + 1):
                if (z, x + y - z, a + b + c - x - y) == tuple(mu):
                    cnt += 1
    return cnt


def load(paths):
    """{(sec, Y): row} from scan jsonl files (rows with check true)."""
    R = {}
    for path in ([paths] if isinstance(paths, str) else paths):
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                if r.get('check'):
                    R[(tuple(r['sec']), r['Y'])] = r
    return R


def cohomology(R):
    """{(sec, Y): h} for the sectors with h > 0."""
    H = {}
    for (sec, Y), r in R.items():
        below = R.get(((sec[0] - 1, sec[1] - 1) + sec[2:], Y + 1))
        h = r['m'] - r['r_up'] - (below['r_up'] if below else 0)
        assert h >= 0, (sec, Y, h)
        if h:
            H[(sec, Y)] = h
    return H


def irrep(lam2):
    return [(lam2[0] - lam2[1]) // 2, (lam2[1] - lam2[2]) // 2]


def weight_space(R, J1, J2, mu2, Y):
    """(states, exact, cohomology) at weight mu2 (doubled) in the sectors (J1, J2; *) at letter number Y."""
    H = cohomology(R)
    st = ex = co = 0
    for (sec, y), r in R.items():
        if (sec[0], sec[1]) != (J1, J2) or y != Y or any(v % 2 for v in sec[2:]) or any(v % 2 for v in mu2):
            continue
        k = kostka(tuple(v // 2 for v in sec[2:]), tuple(v // 2 for v in mu2))
        if k:
            st += k * r['m']
            ex += k * r['r_up']
            co += k * H.get((sec, y), 0)
    return st, ex, co


def census(j, groups):
    """All level-j numbers for one or two groups ({group: R}), as a JSON-serialisable dict (multiplets by Y, by
    (2J1, 2J2), by derivative count D = j - 2Y - (2J1 + 2J2); the (2J1, 2J2) <-> (2J2, 2J1) symmetry check; the sectors
    where two groups differ)."""
    res = {'level': j, 'groups': {}}
    H = {}
    for g, R in groups.items():
        H[g] = cohomology(R)
        byY, byJ, byD = collections.Counter(), collections.Counter(), collections.Counter()
        for (sec, Y), h in H[g].items():
            byY[Y] += h
            byJ[(sec[0], sec[1])] += h
            byD[j - 2 * Y - (sec[0] + sec[1])] += h
        res['groups'][g] = {
            'sectors_scanned': len(R),
            'sectors_nonzero_m': sum(1 for r in R.values() if r['m'] > 0),
            'multiplets_total': sum(H[g].values()),
            'multiplets_by_Y': {str(k): v for k, v in sorted(byY.items())},
            'multiplets_by_2J1_2J2': {f"{k[0]},{k[1]}": v for k, v in sorted(byJ.items())},
            'multiplets_by_derivative_count_D': {str(k): v for k, v in sorted(byD.items())},
            'symmetry_violations': sum(1 for (sec, Y), h in H[g].items()
                                       if H[g].get(((sec[1], sec[0]) + sec[2:], Y), 0) != h),
            'largest_m': max((r['m'] for r in R.values()), default=0),
        }
    if len(groups) == 2:
        ga, gb = list(groups)
        diffs = []
        for k in sorted(set(H[ga]) | set(H[gb])):
            a, b = H[ga].get(k, 0), H[gb].get(k, 0)
            if a != b:
                sec, Y = k
                diffs.append({'sec': list(sec), 'Y': Y, 'irrep': irrep(sec[2:]), 'D': j - 2 * Y - (sec[0] + sec[1]),
                              ga: a, gb: b})
        res['differences'] = diffs
        res['difference_total'] = sum(d[ga] - d[gb] for d in diffs)
    return res


def scan(group, N, j, out, pairs, maxlen=8, log=print):
    """Run every dominant column at level j for the given (2J1, 2J2) pairs and every Y; restartable jsonl."""
    from .sector import Sector
    done = set()
    if os.path.exists(out):
        for line in open(out):
            r = json.loads(line)
            if r.get('check'):
                done.add((tuple(r['sec']), r['Y']))
    cols = [(a, b, j - a - b) for a in range(j + 1) for b in range(a + 1) if 0 <= j - a - b <= b]
    for J1, J2 in pairs:
        for lam2 in cols:
            sec = (J1, J2) + lam2
            for Y in sorted({len(c) for c in contents_of_sector(sec)}):
                if (sec, Y) in done:
                    continue
                t0 = time.time()
                res = Sector(N, sec, Y, maxlen, log=lambda s: None, group=group).run()
                with open(out, 'a') as fh:
                    fh.write(json.dumps(res) + '\n')
                log(f"{group} N={N} sec={sec} Y={Y}: m={res['m']} r_up={res['r_up']} check={res['check']} "
                    f"({time.time() - t0:.0f}s)")


def main(argv):
    cmd = argv[0]
    if cmd == 'scan':
        group, N, j, out = argv[1], int(argv[2]), int(argv[3]), argv[4]
        pairs = ([tuple(p) for p in json.load(open(argv[5]))['pairs']] if argv[5].endswith('.json')
                 else [tuple(int(x) for x in a.split(',')) for a in argv[5:]])
        scan(group, N, j, out, pairs, maxlen=int(os.environ.get('MAXLEN', '8')))
    elif cmd == 'census':
        j, out = int(argv[1]), argv[2]
        groups = {}
        for a in argv[3:]:
            g, paths = a.split(':', 1)
            groups[g] = load(paths.split(','))
        res = census(j, groups)
        json.dump(res, open(out, 'w'), indent=1)
        print(json.dumps({k: v for k, v in res.items() if k != 'differences'}, indent=1))
        for d in res.get('differences', []):
            print(d)
    elif cmd == 'weight':
        R = load(argv[1])
        J1, J2 = (int(x) for x in argv[2].split(','))
        mu2 = tuple(int(x) for x in argv[3].split(','))
        print('states, exact, cohomology:', weight_space(R, J1, J2, mu2, int(argv[4])))
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main(sys.argv[1:])
