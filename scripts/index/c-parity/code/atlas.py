"""T-029: atlas of BMN-sector cohomology per fixed-charge column, from the collaboration's data files (read-only).

A column is a fixed SU(3)_R highest weight lam = (q1,q2,q3) (hw data) or a fixed weight q (SU(2) direct data); inside a
column the complex is graded by 2J (Q lowers 2J by one; equivalently it raises the letter number Y = q1+q2+q3-2J).
For every column we record the profile over 2J of
    m     dimension of the complex (hw multiplicity, or weight-space dimension for direct data)
    h     cohomology,  grav = multi-graviton part (D-005),  BH = h - grav (non-graviton = fortuitous candidates)
and ask whether the BH part is concentrated (a single 2J, hence no boson/fermion cancellation in the index), versus
the graviton part.

usage (repo root): python3 scripts/index/c-parity/code/atlas.py  -> runs/atlas_columns.json, printed summary
"""
import json, os, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
RUNS = os.path.join(HERE, '..', 'runs')
D = lambda *p: os.path.join(REPO, 'data', *p)


def col_key(group, N, lam):
    return (group, N, tuple(lam))


def load_columns_json(path, group, N):
    """SU(3)/SU(4) format: {'columns': [{j, lam, rep, sectors: [{twoJ, m, h, grav, BH, check}], index}]}"""
    out = {}
    for c in json.load(open(path))['columns']:
        prof = {s['twoJ']: dict(m=s['m'], h=s['h'], grav=s['grav'], BH=s['BH'], check=s.get('check', True))
                for s in c['sectors']}
        out[col_key(group, N, c['lam'])] = dict(j=c['j'], lam=tuple(c['lam']), prof=prof, index=c.get('index'),
                                               complete=True, kind='hw', source=os.path.relpath(path, REPO))
    return out


def load_so5(path):
    d = json.load(open(path))
    cols = defaultdict(dict)
    js = {}
    for s in d['columns']:
        k = col_key('SO_odd', 2, s['lam'])
        cols[k][s['twoJ']] = dict(m=s['m'], h=s['h'], grav=s['grav'], BH=s['nongrav'], check=s.get('check', True))
        js[k] = s['j']
    return {k: dict(j=js[k], lam=k[2], prof=p, index=None, complete=True, kind='hw', source=os.path.relpath(path, REPO))
            for k, p in cols.items()}


def load_run_sectors(paths, complete_hint=True):
    """run_sectors jsonl: per (group, N, lam, n): m, r_up (rank of Q from n+1 into n), grav.  h_n = m_n - r_up(n-1) - r_up(n)."""
    recs = {}
    src = {}
    for path in paths:
        for line in open(path):
            r = json.loads(line)
            if not r.get('check', False):
                continue
            k = col_key(r.get('group', 'SU'), r['N'], r['lam'])
            recs.setdefault(k, {})[r['n']] = r
            src[k] = os.path.relpath(path, REPO)
    out = {}
    for k, rs in recs.items():
        prof = {}
        ns = sorted(rs)
        for n in ns:
            r = rs[n]
            if n - 1 >= 0 and (n - 1) not in rs:
                # rank of Q out of sector n unknown unless the sector below is empty (m=0 there is not recorded)
                below_known = False
            else:
                below_known = True
            r_in = rs[n - 1]['r_up'] if (n - 1) in rs else 0
            h = r['m'] - r_in - r['r_up'] if below_known else None
            prof[n] = dict(m=r['m'], h=h, grav=r['grav'], BH=(h - r['grav']) if h is not None else None, check=True)
        lam = k[2]
        from fortuity.bmn.words import contents_of_sector
        allowed = [n for n in range(0, sum(lam) + 1) if contents_of_sector((n,) + tuple(lam))]
        complete = all(n in rs for n in allowed)
        out[k] = dict(j=2 * sum(lam), lam=lam, prof=prof, index=None, complete=complete, kind='hw', source=src[k])
    return out


def load_su2_direct(path):
    """Weight sectors (2J; q sorted ascending) with dimV, coh, grav, nongrav."""
    d = json.load(open(path))
    cols = defaultdict(dict)
    for s in d['sectors']:
        n, *q = s['sector']
        k = col_key('SU', 2, tuple(sorted(q, reverse=True)))
        cols[k][n] = dict(m=s['dimV'], h=s['coh'], grav=s['grav'], BH=s['nongrav'], check=True)
    return {k: dict(j=2 * sum(k[2]), lam=k[2], prof=p, index=None, complete=True, kind='weight',
                    source=os.path.relpath(path, REPO)) for k, p in cols.items()}


def load_all():
    cols = {}
    cols.update(load_su2_direct(D('su2', 'bmn_direct_cohomology_j28.json')))
    cols.update(load_columns_json(D('su3', 'bmn_hw_cohomology_j30.json'), 'SU', 3))
    cols.update(load_columns_json(D('su4', 'bmn_hw_cohomology_j28.json'), 'SU', 4))
    cols.update(load_columns_json(D('su4', 'bmn_hw_cohomology_j30.json'), 'SU', 4))
    cols.update(load_so5(D('so5', 'bmn_hw_cohomology_j28.json')))
    cols.update(load_run_sectors([D('g2', f) for f in ('so7_hw_cohomology_j20.jsonl', 'so7_hw_cohomology_j22_j24.jsonl',
                                                        'so7_hw_cohomology_j26.jsonl', 'g2_hw_cohomology_j20.jsonl',
                                                        'g2_hw_cohomology_j22_j24.jsonl', 'g2_hw_cohomology_j26.jsonl',
                                                        'usp6_hw_cohomology_j20_j24.jsonl')]))
    cols.update(load_run_sectors([D('so_sp', 'so11_hw_cohomology_j26.jsonl'), D('so_sp', 'usp10_hw_cohomology_j26.jsonl')]))
    # new complete columns computed in this workspace (run_sectors jsonl in runs/), plus any files given on the command line
    import glob
    extra = sorted(glob.glob(os.path.join(RUNS, '*.jsonl'))) + sys.argv[1:]
    for f in extra:
        cols.update(load_run_sectors([f]))
    return cols


def summarize(c):
    prof = c['prof']
    ns = sorted(prof)
    known = [n for n in ns if prof[n]['h'] is not None]
    bh = {n: prof[n]['BH'] for n in known if prof[n]['BH']}
    gr = {n: prof[n]['grav'] for n in known if prof[n]['grav']}
    ms = {n: prof[n]['m'] for n in ns}
    peak = max(ms, key=lambda n: ms[n]) if ms and max(ms.values()) > 0 else None
    sgn = lambda d: sum((-1) ** n * v for n, v in d.items())
    return dict(bh=bh, grav=gr, m=ms, m_peak=peak,
                bh_total=sum(bh.values()), bh_index=sgn(bh), grav_total=sum(gr.values()), grav_index=sgn(gr),
                bh_support=sorted(bh), grav_support=sorted(gr))


if __name__ == '__main__':
    cols = load_all()
    groups = defaultdict(list)
    for k, c in cols.items():
        c.update(summarize(c))
        groups[(k[0], k[1])].append((k, c))
    name = {('SU', 2): 'SU(2)', ('SU', 3): 'SU(3)', ('SU', 4): 'SU(4)', ('SO_odd', 2): 'SO(5)', ('SO_odd', 3): 'SO(7)',
            ('G2', 0): 'G2', ('G2', 2): 'G2', ('G2', 3): 'G2', ('Sp', 3): 'USp(6)', ('SO_odd', 5): 'SO(11)',
            ('Sp', 5): 'USp(10)', ('SO_odd', 4): 'SO(9)'}
    print('group    columns  complete  with-BH  BH-classes  BH-multi-2J  BH-cancel | grav-cols  grav-multi-2J  grav-cancel')
    table = []
    for g in sorted(groups, key=lambda g: (g[0], g[1])):
        L = groups[g]
        comp = [c for _, c in L if c['complete'] and all(p['h'] is not None for p in c['prof'].values())]
        wbh = [c for c in comp if c['bh_total']]
        multi = [c for c in wbh if len(c['bh_support']) > 1]
        canc = [c for c in wbh if c['bh_total'] != abs(c['bh_index'])]
        wgr = [c for c in comp if c['grav_total']]
        gmulti = [c for c in wgr if len(c['grav_support']) > 1]
        gcanc = [c for c in wgr if c['grav_total'] != abs(c['grav_index'])]
        print(f"{name.get(g, g)!s:8} {len(L):7}  {len(comp):8}  {len(wbh):7}  {sum(c['bh_total'] for c in wbh):10}  "
              f"{len(multi):11}  {len(canc):9} | {len(wgr):9}  {len(gmulti):13}  {len(gcanc):11}")
        table.append(dict(group=name.get(g, str(g)), columns=len(L), complete=len(comp), with_BH=len(wbh),
                          BH_classes=sum(c['bh_total'] for c in wbh), BH_multi_2J=len(multi), BH_cancel=len(canc),
                          grav_columns=len(wgr), grav_multi_2J=len(gmulti), grav_cancel=len(gcanc)))
    print()
    print('Columns with non-graviton (BH) classes:')
    bhrows = []
    for g in sorted(groups):
        for k, c in sorted(groups[g], key=lambda kc: (kc[1]['j'], kc[0][2])):
            if c['bh_total'] or any(p['BH'] for p in c['prof'].values() if p['BH']):
                prof = ' '.join(f"{n}:{c['prof'][n]['m']}/{c['prof'][n]['h']}/{c['prof'][n]['grav']}" for n in sorted(c['prof']))
                print(f"  {name.get(g, g)!s:7} j={c['j']:2} lam={c['lam']}  BH at 2J={c['bh_support']} (x{c['bh_total']}), "
                      f"grav at 2J={c['grav_support']}, m-peak 2J={c['m_peak']}, complete={c['complete']}")
                print(f"      2J:m/h/grav  {prof}")
                bhrows.append(dict(group=name.get(g, str(g)), j=c['j'], lam=c['lam'], bh=c['bh'], grav=c['grav'],
                                   m=c['m'], m_peak=c['m_peak'], complete=c['complete'], source=c['source']))
    # letter-number parity Y = q1+q2+q3-2J of every class (graviton parity lemma: for groups whose Weyl group
    # contains -1, i.e. all here except SU(3), SU(4), every multi-graviton has even Y)
    print()
    print('Classes by letter-number parity (Y even / Y odd):   gravitons | non-graviton')
    parity = []
    for g in sorted(groups):
        cnt = {('grav', 0): 0, ('grav', 1): 0, ('BH', 0): 0, ('BH', 1): 0}
        for k, c in groups[g]:
            for n, p in c['prof'].items():
                if p['h'] is None:
                    continue
                Y = sum(c['lam']) - n
                cnt[('grav', Y % 2)] += p['grav']
                cnt[('BH', Y % 2)] += p['BH']
        print(f"  {name.get(g, g)!s:8} {cnt[('grav', 0)]:6} / {cnt[('grav', 1)]:<6} | {cnt[('BH', 0)]:3} / {cnt[('BH', 1)]}")
        parity.append(dict(group=name.get(g, str(g)), grav_Y_even=cnt[('grav', 0)], grav_Y_odd=cnt[('grav', 1)],
                           BH_Y_even=cnt[('BH', 0)], BH_Y_odd=cnt[('BH', 1)]))
    os.makedirs(RUNS, exist_ok=True)
    json.dump(dict(description='T-029 atlas: per-group column statistics and all columns with non-graviton classes '
                               '(read from data/ of the collaboration)', summary=table, bh_columns=bhrows, parity=parity),
              open(os.path.join(RUNS, 'atlas_columns.json'), 'w'), indent=1, default=str)
