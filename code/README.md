# code: the `fortuity` library

Python library for the Q-cohomology of weakly coupled N=4 SYM, with tests in `tests/`. Install with
`python3 -m pip install -e '.[dev]'` (numpy, sympy, pytest) or put this directory on `PYTHONPATH`; run the tests
with `python3 -m pytest -q` (a few minutes; `FORTUITY_SLOW=1` adds the slow ones). Physics conventions: section 2 of
`summary/fortuity-summary.pdf`. All ranks are exact linear algebra over F_p (p = 2^26 - 5 unless stated) on random
evaluations; nothing is decided with floating point.

## `fortuity.bmn`: BMN sector (letters phi_m, psi_m, f)

| module | contents | used for |
|---|---|---|
| `words.py` | multi-trace words, sectors (2J; q), the action of Q | everything |
| `evaluate.py`, `grassmann.py`, `modp.py`, `guard.py` | evaluation at random points over F_p with Grassmann frames; mod-p linear algebra; memory guard | everything |
| `direct.py`, `run_su2_direct.py` | direct cohomology sector by sector (SU(2); no rank-saturation guard, use for N = 2 only) | R-001 |
| `symdirect.py` | direct cohomology on Weyl-symmetric subspaces and weight spaces, any group | R-002, R-005, R-006 |
| `fullindex.py`, `groupindex.py`, `fugacityindex.py` | exact BMN indices by Haar integration: SU(N); SO, USp, G2; fixed fugacities | R-001, R-003, R-004, R-021 |
| `gravindex.py` | diagonal (eigenvalue) graviton counting, all groups | R-003, R-017, T-034 |
| `twistedindex.py`, `cparity.py` | charge conjugation on words, C-twisted index, kernel parity index | R-017 |
| `blockrestrict.py`, `classify.py` | block restrictions SU(N) > SU(N-1) x U(1), SO(2N+1) > SO(2N+1-2b) x SO(2)^b; classification of an operator in a sector | R-016, R-024 |
| `dmk.py` | the operator (O^3_30)^{ijk} of arXiv:2412.08695 rebuilt from its printed formulas | R-016 |
| `g2_checks.py` | G2 trace-word completeness and the restriction of the SO(7) class (`python -m fortuity.bmn.g2_checks restriction`, `completeness 18`) | R-010 |
| `r002_check.py` | the graviton class that vanishes on diagonal matrices (`python -m fortuity.bmn.r002_check [--exact]`) | R-002 |

## `fortuity.hwcoh`: SU(3)_R highest-weight complex (production engine)

| module | contents |
|---|---|
| `dims.py` | exact highest-weight multiplicities refined by fermion number (Haar integral + Weyl formula) |
| `wordenum.py`, `hw.py`, `gravhw.py` | words by shape; highest-weight vectors shape by shape; highest-weight vectors of the graviton algebra |
| `evalg.py`, `ffla.py`, `ffblas.py`, `flint/` | evaluation frames (also on Cartan-valued letters); mod-p linear algebra with optional FLINT 3 / BLAS kernels (`flint/build*.sh`) |
| `sector.py`, `sectorpar.py`, `column.py` | one sector (serial and parallel engines, with the exact-dimension self-check); one column |
| `run_sectors.py` | driver: `python -m fortuity.hwcoh.run_sectors N DIMS_PKL\|auto:JMAX OUT_JSONL q1,q2,q3[:n1,n2,...] ...` (one JSON line per sector, restartable), with `GROUP` (SU, SO_odd, Sp, G2), `ENGINE=par`, `MAXLEN`, `MAXLEN_UP`, `GEN_NMAX`, `SEED`, `HW_CACHE`, `FORTUITY_WORKERS` from the environment (see its docstring) |
| `make_tasks.py`, `assemble.py`, `to_json.py` | task lists; per-column assembly; data files with the Euler check against the exact index |
| `noncoulomb.py` | Coulomb-type vs non-Coulomb graviton classes (`python -m fortuity.hwcoh.noncoulomb GROUP N GEN_NMAX OUT DATA...`) |
| `restrict.py` | restriction of classes from G to a subgroup (SO(7) > G2) |
| `classreps.py`, `classreps_par.py`, `cores.py` | explicit non-graviton representatives; their block restrictions; fused capacity and core test |

## `fortuity.full`: all 1/16-BPS letters (gaugini, derivatives) and loop corrections

| module | contents |
|---|---|
| `letters.py`, `words.py`, `wordenum.py` | letters as Taylor coefficients (a, b, S) of the Chang-Lin superfield; words; sectors (2J1, 2J2; 2q) at letter number Y |
| `dims.py`, `gdims.py` | exact dimensions (SU(N); SO(2N+1), USp(2N), G2) |
| `evalg.py`, `evalpoints.py`, `hw.py`, `sector.py`, `cohom.py` | evaluation, highest-weight vectors, one sector of the highest-weight complex, direct cohomology |
| `q1.py` | the one-loop supercharge Q_1 of arXiv:2512.07771 on derivative-free letters, SU(N) and so(n) |
| `lifting.py` | one-loop lifting tests of the SO(2N+1) classes (`python -m fortuity.full.lifting so7|so9|so11`) |
| `restrict.py`, `weyl.py` | block restriction; SU(3) Weyl action |
| `levels.py`, `gravitons.py` | complete full-letter levels for any group; large-N single-trace cohomology and the multi-graviton span |

## Tests

`tests/` covers the identities behind every result: Q^2 = 0 and [E_i, Q] = 0 on random words, exact dimensions
against brute force, the published SU(2), SO(7)/USp(6) counts, the chain-map property of block restrictions, the
twisted index against brute-force C counts, the R-009 preimage identity, the dMK operator, the one-loop supercharge
(nilpotency {Q_0, Q_1} = 0, the so(n) Konishi coefficient 4(n-2)), the SO(7) lifting with rank 741, the core test on
the stored SU(3) data, and the certification guards of the highest-weight engine.

The one-loop engine for large sectors (parallel, GPU and BLAS ranks) and the two-loop supercharge on derivative-free
words are in `scripts/oneloop/su-threshold-classes/`, not in the library.
