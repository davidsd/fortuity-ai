#!/bin/bash
# R-007 order 2 (summary/fortuity-summary.pdf): the d_1 chain and the option-2 (LW-test) chain behind a kernel-aug job.
# usage: bash submit_o2chain.sh KERNEL_AUG_JOBID [S20_PREP_JOBID] [T21_MERGE_JOBID]
set -eo pipefail
cd ${REPO}/scripts/oneloop/su-threshold-classes/cluster
KA=$1; S20=${2:-19185272}; T21M=${3:-19185276}
L=${REPO}/log; C=cache/fullpar_hw_rank_3; V=classes/r007_d1
dep() { local d=""; for j in "$@"; do [ -n "$j" ] && d="$d:$j"; done; echo "afterok$d"; }
M=$(sbatch --parsable --dependency=$(dep $KA) d1make.sbatch)
VE=$(sbatch --parsable --dependency=$(dep $M) d1veval.sbatch)
RK=$(sbatch --parsable --dependency=$(dep $VE) --cpus-per-task=32 --mem=200G r007_orders_cpurank_ff.sbatch r007_d1_U2)
CK=$(sbatch --parsable --dependency=$(dep $M) d1check.sbatch)
J=$(sbatch --parsable --dependency=$(dep $M) jt_make.sbatch)
vK=$(sbatch --parsable --dependency=$(dep $J) veval_generic.sbatch prep/o2lw_Kp ${C}_3-5-13-7-7_9_codd.pkl $V/vecs_Kp.pkl prep/o2lw_Kp_V --log $L/gprep_19185269.out)
vT0=$(sbatch --parsable --dependency=$(dep $J) veval_generic.sbatch prep/o2lw_T0 ${C}_4-6-15-7-7_9_codd.pkl $V/vecs_Tp0.pkl prep/o2lw_T0_V --log $L/gprep_19185270.out)
vS2=$(sbatch --parsable --dependency=$(dep $J) veval_generic.sbatch prep/o2lw_S2 ${C}_4-4-14-8-6_9_codd.pkl $V/vecs_S2.pkl prep/o2lw_S2_V --log $L/gprep_19185271.out)
vS20=$(sbatch --parsable --dependency=$(dep $J $S20) veval_generic.sbatch prep/o2lw_S20 ${C}_4-4-12-8-8_9_codd.pkl $V/vecs_S20.pkl prep/o2lw_S20_V --log $L/gprep_$S20.out)
vT21=$(sbatch --parsable --dependency=$(dep $J $T21M) veval_generic.sbatch prep/o2lw_T21 ${C}_4-6-13-9-7_9_codd.pkl $V/vecs_Tp21.pkl prep/o2lw_T21_V --log $L/sprep_19185274_0.out,$L/sprep_19185274_1.out,$L/sprep_19185274_2.out,$L/sprep_19185274_3.out)
vD=$(sbatch --parsable --dependency=$(dep $J) veval_generic.sbatch prep/r007_o2cand_D ${C}_3-7-14-8-8_9_codd.pkl $V/vecs_D.pkl prep/o2lw_D_V --log $(echo $L/sprep_19177964_{0,1,2,3,4,5}.out | tr ' ' ,) --orig-v classes/rank_3_3-7-14-8-8_9/ctrl.pkl,classes/rank_3_3-7-14-8-8_9/q1.pkl)
rK=$(sbatch --parsable --dependency=$(dep $vK) resid_ff.sbatch o2lw_Kp_V)
rT0=$(sbatch --parsable --dependency=$(dep $vT0) resid_ff.sbatch o2lw_T0_V)
rS2=$(sbatch --parsable --dependency=$(dep $vS2) resid_ff.sbatch o2lw_S2_V)
rS20=$(sbatch --parsable --dependency=$(dep $vS20) resid_ff.sbatch o2lw_S20_V)
rT21=$(sbatch --parsable --dependency=$(dep $vT21) --mem=480G resid_ff.sbatch o2lw_T21_V)
rD=$(sbatch --parsable --dependency=$(dep $vD) --mem=480G resid_ff.sbatch o2lw_D_V)
rQ=$(sbatch --parsable --dependency=$(dep $VE) resid_ff.sbatch r007_d1_U2)
P6="prep/o2lw_Kp_V prep/o2lw_T0_V prep/o2lw_S2_V prep/o2lw_S20_V prep/o2lw_T21_V prep/o2lw_D_V"
j6=$(sbatch --parsable --dependency=$(dep $rK $rT0 $rS2 $rS20 $rT21 $rD) resid_joint.sbatch $V/joint_lw.json $P6)
j7=$(sbatch --parsable --dependency=$(dep $rK $rT0 $rS2 $rS20 $rT21 $rD $rQ) resid_joint.sbatch $V/joint_lw_d1.json $P6 prep/r007_d1_U2)
echo "d1: make $M veval $VE rank $RK check $CK | jt $J | veval Kp $vK T0 $vT0 S2 $vS2 S20 $vS20 T21 $vT21 D $vD | resid Kp $rK T0 $rT0 S2 $rS2 S20 $rS20 T21 $rT21 D $rD d1 $rQ | joint LW $j6 LW+d1 $j7"
