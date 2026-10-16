# scripts

Drivers, cluster job files and run outputs behind the results, one directory per project, grouped by area. The
library they call is `code/fortuity`; the data they produce or read are in `data/`. Result labels (R-0xx, T-0xx)
refer to `summary/fortuity-summary.pdf`, whose appendix A gives the same map from the side of the results.

Conventions for running them:
- put `code/` on `PYTHONPATH` (or install it with `pip install -e 'code[dev]'`); scripts that read other files find
  them relative to their own location, or through `REPO` (the repository root) where noted;
- `*.sbatch` and `env.sh` files are Slurm job files kept as they were run, with machine-specific values replaced by
  placeholders (`${REPO}`, `${SCRATCH}`, `${STORE}`, `${VENV}`, `${ACCOUNT}`, `${USER}`, `${EMAIL}`, `${HOME}`). On
  the cluster the library sat under `${REPO}/code` and jobs wrote per-sector records into `results/`; those records
  are now in `data/`;
- `runs/` and `outputs/` subdirectories hold the outputs quoted in the summary.

## bmn: BMN-sector cohomology

| directory | results | contents |
|---|---|---|
| `bmn/index-and-direct-cohomology` | R-001, R-002, R-003 | exact indices and diagonal graviton counts (`run_index.py`, `run_index_levels.py`, `extract_core.py`, `check_su2_index.py`); direct cohomology on Weyl-symmetric sectors (`run_sym.py`, `run_level.py`, `run_su4_energy.py`); the failure of the diagonal criterion (`test_diag_criterion.py 3,4,4,4`, `test_diag_criterion_N.py`, `test_grav_sym.py`, `test_grav_robust.py`) and the explicit counterexample (`explicit_graviton_class.py`, `verify_explicit_class.py`; the library check is `python -m fortuity.bmn.r002_check`) |
| `bmn/su3-level24` | R-009 | the operator (4.8) of arXiv:2312.16443 (`build_O.py`, `build_O6.py`; its corrected form in version 3 of that paper: `cckll_v3.py`, classified by `decompose_multi.py`, outputs `runs/multi_v3_p*.json`), its preimage (`singlet_preimage*.py`, `preimage_search.py`), exactness tests (`hw_relation_test.py`, `triple_check.py`, `decompose*.py`, `full_complex_check.py`, `f_restricted_exactness.py`, `R24_check.py`, `lambda_O1.py`), the descendant relation (`descendant_check.py 1`), the diagonal mechanism (`diag_vanishing.py`), Table 1 (`table1_core.py`), the unfinished singlet column at j = 36 (`singlet_column_j36.py`) |
| `bmn/hw-scans-cluster` | R-007, R-008, R-012 (j = 28), R-014 | Slurm launchers and task lists of the SU(3) j <= 30, SU(4) j <= 30 and SO(5) j = 28 scans; `scripts/so5_json.py` assembles the SO(5) data |
| `bmn/su3-level30-representative` | T-021 (R-007) | `representative.py`: explicit highest-weight representative of the j = 30 class; `runs/` holds it (201,564 words, mod p) |
| `bmn/su3-level32-scan` | T-042 | job files and task lists of the complete SU(3) j = 32 scan (records in `data/su3/hw_cohomology_j32_records`) |
| `bmn/su3-cores` | R-024 | the (8,5,4) column at j = 34 (`cluster/`), its analysis (`analysis/make_core_data.py`), seed-1 and seed-2 records (`results*/`), explicit representatives and their restrictions (`reps/run_reps.py`, `reps/make_reps_data.py`) |
| `bmn/su2-level44-scan` | T-027 | job files of the complete SU(2) scan to j = 44 (records in `data/su2/hw_cohomology_j44_records`) |
| `bmn/so-odd-thresholds` | R-018 | task lists and job files of the SO(7), SO(9), SO(11) scans; audit: `python3 analyze.py 5 24 ../../../data/so_sp/so11_hw_cohomology_j24.jsonl` (N = 3, 4, 5 for SO(7), SO(9), SO(11)) |
| `bmn/so-usp-g2-scans` | R-011, R-013, R-019, R-021, R-022, T-043 | SO(7), USp(6), G2 scans (`make_sector_tasks.py`, `restrict_scan.py`, `euler_check.py`, `extra_checks.py`, `cluster/`); `extensions/`: the SO(9) closure test, the N = 6 fugacity check, the G2 j <= 18 run |
| `bmn/so5-scan` | R-012 | the SO(5) driver (`run_so5.sh`) and its check against the stored records (`check_so5_driver.py`) |
| `bmn/block-restriction` | R-016 | `code/run_su3_to_su2.py`, `code/run_so_tower.py`, `code/make_r016_data.py`; outputs and representatives in `runs/` |

## index: parity and twisted indices

| directory | results | contents |
|---|---|---|
| `index/c-parity` | R-017 | `code/`: twisted index, kernel parity, C on classes, SU(2) predictions, SU(4) twisted scan to j = 44, atlas; `runs/`: outputs; `c19/`: the analysis of the rigour of the kernel parity index requested by the independent review |
| `index/kernel-parity-so-usp-g2` | T-034 | `kernel_index.py` and its outputs for SO(7), USp(6), G2 |

## oneloop: loop corrections

| directory | results | contents |
|---|---|---|
| `oneloop/su-threshold-classes` | T-030, T-031, T-040, all orders, two loops | one-loop tests (`oneloop_test.py`; large sectors with `fullpar.py` + `gpu_rank.py` / `cpu_rank.py`), C-parity (`c_parity.py`, `twisted_index.py`), higher orders and the selection rule (`size_r007_orders.py`, `membership.py`, `hwcheck.py`, `abelian_*.py`, `su2_lw.py`, `test_q1_symm.py`), the index tower (`su2_index_tower.py`), the two-loop supercharge (`q2bmn.py`, `test_q2.py`, `test_q2_tracerel.py`, `d2_make.py`, `d2_report.py`), job files (`cluster/`) and outputs (`runs/`) |
| `oneloop/so-odd-classes` | R-020 | development scripts of the SO(2N+1) lifting tests (the library version is `fortuity.full.lifting`), the hairy-class Coulomb check, outputs in `runs/` |
| `oneloop/protection` | T-033 | intertwining of block restriction and Q_1 (`code/so9_intertwining.py`, `code/so11_intertwining.py`), fused channels of the SU(3) class (`code/su2_channels.py`), outputs in `runs/` |
| `oneloop/su2-fate-predictions` | T-037 | the a-priori SU(2) fate table (`scripts/run_all.sh`, reads `REPO`), outputs in `outputs/` |

## full-letter: all 1/16-BPS letters

| directory | results | contents |
|---|---|---|
| `full-letter/so7-usp6` | R-025 | level enumeration (`enum_level.py`), group sectors (`sector_group.py`), graviton classification (`hwgrav.py`), the runs behind the census (`runs/*.py`; outputs in `data/fullso`) |
| `full-letter/su-checks` | validation of `fortuity.full` | Chang-Lin SU(2) counts (`validate_su2.py 6`, `validate_hw_su2.py`), BMN cross-check (`run_bmn_check.py`), class counts (`classes_full.py`), Q_1 tests (`test_q1.py`) |

## entropy

| directory | results | contents |
|---|---|---|
| `entropy/bonus-index-parity` | T-038 | Cardy analysis of the bonus-charge index with parity (`cardy_lib.py` and drivers, validation `cardy_validate.py`, outputs `*.out`) |
| `entropy/lifting-excess` | T-041 | charge map of the lifting excess (`code/`, `physics/`), an independent implementation (`indep/`), job files (`cluster/`), outputs and figures |

## review

| directory | contents |
|---|---|
| `review/independent-checks` | reproducers written by the independent review of R-001 to R-018 (`reproducers/`: certification guards of the engines, undercount cases, planted Q_1 defects) and its follow-up checks (`followup/`: the kernel parity index, the dMK operator; `run_all.sh`) |
