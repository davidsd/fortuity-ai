"""Test merge_grid.py on the saved chunks of job 568141 (outputs/indep/grid12_chunk*.tsv, K = 48), in a temporary
directory outside the repo: the merge must reproduce outputs/indep/indep_grid12.tsv byte for byte, and each damaged
copy must be rejected with nothing written. Then compare_scan.py must flag a missing value (the '-' of --no-r1) as a
mismatch, and still find every point of the undamaged merge in agreement with outputs/main/scan12_patched.tsv.
The --points mode (added for run_indep_segments.sbatch) is tested on the same chunks, renamed, with the grid written
out as a "polar TH R" point list: it must give the same file, and a point list that lacks one point must be refused.
Usage: python test_merge_grid.py.  Runtime: about a second.
"""
import filecmp
import glob
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SAVED = os.path.join(HERE, '..', 'outputs', 'indep')
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)


def damage(path, row, col, value=None, drop_last=False):
    """Replace column col of data row `row` (0-based) of a chunk or merged file, or drop that row's last field."""
    with open(path) as fh:
        lines = fh.read().splitlines()
    cols = [l for l in lines if l.startswith('theta_deg')][0].split('\t')
    data = [i for i, l in enumerate(lines) if l and not l.startswith('#') and not l.startswith('theta_deg')]
    tok = lines[data[row]].split('\t')
    if drop_last:
        tok = tok[:-1]
    else:
        tok[cols.index(col)] = value
    lines[data[row]] = '\t'.join(tok)
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')


def merge(d, *args):
    out = os.path.join(d, 'merged.tsv')
    p = subprocess.run([sys.executable, '-B', os.path.join(HERE, 'merge_grid.py'), d] + list(args) + ['--out', out],
                       capture_output=True, text=True)
    return p.returncode, os.path.exists(out), (p.stdout + p.stderr).strip().replace(d, '<dir>')


def main():
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        def copy(name):
            d = os.path.join(tmp, name)
            os.makedirs(d)
            for f in glob.glob(os.path.join(SAVED, 'grid12_chunk*.tsv')):
                shutil.copy(f, d)
            return d
        d = copy('saved')
        code, wrote, msg = merge(d, '48')
        same = wrote and filecmp.cmp(os.path.join(d, 'merged.tsv'), os.path.join(SAVED, 'indep_grid12.tsv'),
                                     shallow=False)
        print("saved run, K=48: exit %d, identical to outputs/indep/indep_grid12.tsv: %s\n   %s" % (code, same, msg))
        ok &= code == 0 and same
        cases = []
        d = copy('missing'); os.remove(os.path.join(d, 'grid12_chunk17.tsv')); cases.append(('missing chunk 17', d, '48'))
        d = copy('leftover'); shutil.copy(os.path.join(d, 'grid12_chunk0.tsv'), os.path.join(d, 'grid12_chunk48.tsv'))
        cases.append(('leftover chunk 48', d, '48'))
        cases.append(('K=24 merge of the 48-chunk run (a smaller rerun into the same OUT)', copy('k24'), '24'))
        d = copy('other_k'); shutil.copy(os.path.join(d, 'grid12_chunk6.tsv'), os.path.join(d, 'grid12_chunk5.tsv'))
        cases.append(('chunk 5 replaced by other points', d, '48'))
        d = copy('failed')
        with open(os.path.join(d, 'grid12_chunk3.tsv'), 'a') as fh:
            fh.write('# fail [polar 0 1.0]: test\n')
        cases.append(("a '# fail' line in chunk 3", d, '48'))
        cases.append(('wrong --rmax0 0.9', copy('rmax'), '48', '--rmax0', '0.9'))
        d = copy('nor1'); damage(os.path.join(d, 'grid12_chunk9.tsv'), 0, 'r1', '-')
        cases.append(("r1 = '-' in chunk 9 (as written by --no-r1)", d, '48'))
        d = copy('nan'); damage(os.path.join(d, 'grid12_chunk2.tsv'), 1, 'delta0', 'nan')
        cases.append(("delta0 = nan in chunk 2", d, '48'))
        d = copy('short'); damage(os.path.join(d, 'grid12_chunk4.tsv'), 0, None, drop_last=True)
        cases.append(("a truncated row in chunk 4", d, '48'))
        d = copy('neginf'); damage(os.path.join(d, 'grid12_chunk7.tsv'), 0, 'cut_cl', '-inf')
        cases.append(("cut_cl = -inf in chunk 7 (only +inf is allowed there)", d, '48'))
        for label, d, *args in cases:
            code, wrote, msg = merge(d, *args)
            print("%s: exit %d, output written: %s\n   %s" % (label, code, wrote, msg))
            ok &= code == 1 and not wrote
        # --points mode: the grid as a point list, the chunks renamed to pt_chunk<k>.tsv
        from indep_delta_r1 import grid_points, set_precision, to_mp
        set_precision(25)
        labels = [lab for lab, xi in grid_points(12, to_mp('1.0'))]
        d = os.path.join(tmp, 'pts')
        os.makedirs(d)
        for f in glob.glob(os.path.join(SAVED, 'grid12_chunk*.tsv')):
            shutil.copy(f, os.path.join(d, os.path.basename(f).replace('grid12_chunk', 'pt_chunk')))
        for label, lab_list, want_ok in (("--points with the grid as a point list", labels, True),
                                         ("--points with a point list lacking one point", labels[:-1], False)):
            pfile = os.path.join(tmp, 'gridpoints.txt')
            with open(pfile, 'w') as fh:
                fh.write("# the grid of scan 12 (rmax0 1.0) as a point list\n" + "\n".join(lab_list) + "\n")
            out = os.path.join(d, 'merged.tsv')
            if os.path.exists(out):
                os.remove(out)
            p = subprocess.run([sys.executable, '-B', os.path.join(HERE, 'merge_grid.py'), d, '48', '--points', pfile,
                                '--prefix', 'pt_chunk', '--out', out], capture_output=True, text=True)
            same = os.path.exists(out) and filecmp.cmp(out, os.path.join(SAVED, 'indep_grid12.tsv'), shallow=False)
            print("%s (%d points): exit %d, identical to outputs/indep/indep_grid12.tsv: %s\n   %s"
                  % (label, len(lab_list), p.returncode, same, (p.stdout + p.stderr).strip().replace(tmp, '<tmp>')))
            ok &= (p.returncode == 0 and same) if want_ok else (p.returncode == 1 and not os.path.exists(out))
        # compare_scan.py: the undamaged merge agrees everywhere; a '-' in r1 at a matched point is a mismatch
        scan = os.path.join(HERE, '..', 'outputs', 'main', 'scan12_patched.tsv')
        cmp_py = os.path.join(HERE, 'compare_scan.py')
        good = os.path.join(SAVED, 'indep_grid12.tsv')
        bad = os.path.join(tmp, 'grid_r1_dash.tsv')
        shutil.copy(good, bad)
        damage(bad, 5, 'r1', '-')
        for label, path, want_ok in (("undamaged merge", good, True), ("merge with r1 = '-' in data row 6", bad, False)):
            p = subprocess.run([sys.executable, '-B', cmp_py, scan, path], capture_output=True, text=True)
            lines = p.stdout.strip().splitlines()
            last = lines[-1] if lines else p.stderr.strip()
            flagged = [l.split('\t')[:2] for l in lines if l.endswith('MISMATCH')]
            print("compare_scan.py, %s: '%s'%s" % (label, last, ("; flagged %s" % flagged) if flagged else ""))
            ok &= (last.startswith('all ') and 'agree' in last) == want_ok
    print("test_merge_grid %s" % ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
