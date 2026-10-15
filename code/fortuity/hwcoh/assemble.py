"""Assemble per-sector JSONL results into columns: h_n = m_n - r_n - r_{n+1}, BH_n = h_n - grav_n.
usage: python -m fortuity.hwcoh.assemble RESULTS_JSONL[,MORE,...] [N] [--index] [--nonzero]"""
import sys, json, glob
from collections import defaultdict
files = [f for pat in sys.argv[1].split(',') for f in sorted(glob.glob(pat))]
rows = [json.loads(l) for f in files for l in open(f)]
args = [a for a in sys.argv[2:] if not a.startswith('--')]
if args:
    rows = [r for r in rows if r['N'] == int(args[0])]
col = defaultdict(dict)
for r in rows:
    col[(r['N'], tuple(r['lam']))][r['n']] = r
for (N, lam), secs in sorted(col.items()):
    j = 2 * sum(lam)
    for n in sorted(secs):
        r = secs[n]
        rn = secs[n - 1]['r_up'] if (n - 1) in secs else (0 if n == 0 else None)
        h = r['m'] - r['r_up'] - rn if rn is not None else None
        bh = h - r['grav'] if h is not None else None
        E = j / 2 - n / 2
        flag = '' if r['check'] else ' CHECK-FAILED'
        # r_up of sector n used the 2J+1 words with cap maxlen_up; valid only if sector n+1 was checked with a cap
        # <= maxlen_up (None = no cap)
        cap_up = r.get('maxlen_up', r.get('maxlen'))
        if (n + 1) in secs and cap_up is not None:
            cap_next = secs[n + 1].get('maxlen')
            if cap_next is None or cap_next > cap_up:
                flag += ' CAP-INCONSISTENT'
        if '--nonzero' in sys.argv and not (h or not r['check'] or h is None):
            continue
        print(f"N={N} j={j} lam={lam} [{lam[0]-lam[1]},{lam[1]-lam[2]}] 2J={n} E={E}: m={r['m']} r_up={r['r_up']} "
              f"h={h} grav={r['grav']} BH={bh}{flag}")

# Euler-characteristic check against the exact Haar-integral index (fortuity.bmn.fullindex), per SU(3)_R irrep
if '--index' in sys.argv:
    from fortuity.bmn.fullindex import full_index, su3_multiplicities
    byN = defaultdict(list)
    for (N, lam), secs in col.items():
        byN[N].append((lam, secs))
    for N, cl in byN.items():
        jmax = max(2 * sum(l) for l, _ in cl)
        Z, A = full_index(N, jmax, verbose=False)
        for lam, secs in sorted(cl):
            j = 2 * sum(lam)
            mult = su3_multiplicities(Z[j], A)
            rep = (lam[0] - lam[1], lam[1] - lam[2])
            ns = sorted(secs)
            hs = []
            for n in ns:
                r = secs[n]
                rn = secs[n - 1]['r_up'] if (n - 1) in secs else (0 if n == 0 else None)
                hs.append(None if rn is None else r['m'] - r['r_up'] - rn)
            if None in hs:
                print(f"INDEX N={N} lam={lam}: incomplete column"); continue
            chi = sum((-1) ** n * h for n, h in zip(ns, hs))
            ref = mult.get(rep, 0)
            print(f"INDEX N={N} j={j} lam={lam} {list(rep)}: sum (-1)^2J h = {chi}, index = {ref} -> {'OK' if chi == ref else 'MISMATCH'}")
