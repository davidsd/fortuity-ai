"""T-034: kernel parity index for groups where charge conjugation acts trivially (SO(2N+1), USp(2N), G2).
There P = C (-1)^n = (-1)^n (n = number of letters), every multi-graviton has n even (R-017), and every class visible on the
Coulomb branch has P = +1. Per SU(3)_R highest-weight column lam:
    T(lam) = (-1)^{|lam|} I(lam) - D(lam),
I = exact BMN index multiplicity of the irrep (fortuity.bmn.groupindex), D = Coulomb-branch graviton count of the irrep
(fortuity.bmn.gravindex with group=...). If D were exact, T = #P+ - #P- among the classes that vanish on the Coulomb
branch, and T < 0 would force at least -T classes with an odd number of letters (non-graviton: black holes). CAVEAT (
review item C19, which applies here as to R-017): D is NOT a lower bound. The per-weight Coulomb dimensions are random-
evaluation lower bounds, but the irrep multiplicity is a signed (+1/-1) combination of them (twistedindex.hw_multiplicities),
so an underestimated dimension with a negative coefficient overestimates D. Hence neither T < 0 nor the exact value of a
T > 0 is rigorous without certified per-weight ranks or a direct highest-weight Coulomb count (hwcoh evalg cartan=True,
e.g. scripts/bmn/so-usp-g2-scans/extra_checks.py coulomb). Black holes with an even number of letters (e.g. the SO(7)
class of R-005, 8 letters) are invisible in any case.
usage (from code/): python ../scripts/index/kernel-parity-so-usp-g2/kernel_index.py GROUP N JMAX [out.json]"""
import sys, json, time
from fortuity.bmn.groupindex import bmn_index
from fortuity.bmn.fullindex import su3_multiplicities
from fortuity.bmn.gravindex import GravitonCounter
from fortuity.bmn.twistedindex import hw_multiplicities


def kernel_index(group, N, jmax, verbose=True):
    t0 = time.time()
    gc = GravitonCounter(N, jmax, group=group, verbose=verbose)
    dims = gc.run()
    if verbose:
        print(f"[{group} N={N}] Coulomb graviton counts to j={jmax}: {time.time() - t0:.0f} s", flush=True)
    uns = {}
    for sec, d in dims.items():
        if d and 2 * sum(sec[1:]) <= jmax:
            q = tuple(sec[1:])
            uns[q] = uns.get(q, 0) + d
    Dhw = hw_multiplicities(uns, jmax // 2)
    Z, A = bmn_index(group, N, jmax)
    if verbose:
        print(f"[{group} N={N}] exact index to j={jmax}: {time.time() - t0:.0f} s", flush=True)
    out = {}
    for j in range(4, jmax + 1, 2):
        Ij = su3_multiplicities(Z[j], A)                      # {(a, b) Dynkin: multiplicity}
        lams = {lam for lam in Dhw if 2 * sum(lam) == j}
        lams |= {((j // 2 + 2 * a + b) // 3, (j // 2 - a + b) // 3, (j // 2 - a - 2 * b) // 3)
                 for (a, b) in Ij if (j // 2 + 2 * a + b) % 3 == 0}
        for lam in lams:
            ab = (lam[0] - lam[1], lam[1] - lam[2])
            I = int(Ij.get(ab, 0))
            D = int(Dhw.get(lam, 0))
            T = (-1) ** sum(lam) * I - D
            out[lam] = dict(j=j, dynkin=list(ab), index=I, coulomb=D, T=T)
    return out


if __name__ == "__main__":
    group, N, jmax = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    res = kernel_index(group, N, jmax)
    nz = {lam: r for lam, r in sorted(res.items(), key=lambda x: (x[1]['j'], x[0])) if r['T']}
    for lam, r in nz.items():
        print(f"  j={r['j']:3d} lam={lam} {r['dynkin']}: index {r['index']:6d}, Coulomb {r['coulomb']:6d}, T = {r['T']:+d}")
    print(f"{group} N={N} j<={jmax}: {len(res)} columns, T nonzero in {len(nz)}, T<0 in {sum(1 for r in nz.values() if r['T'] < 0)}")
    if len(sys.argv) > 4:
        json.dump({str([int(x) for x in l]): {k: (int(v) if not isinstance(v, list) else [int(x) for x in v])
                                              for k, v in r.items()} for l, r in res.items()},
                  open(sys.argv[4], 'w'), indent=0)
