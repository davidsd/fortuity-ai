#!/bin/bash
# R-007 order 2: the non-Coulomb candidate t' alone (classes/r007_o2cand, 2 vectors) through d_1 and the six LW tests.
# With one class in play, per-sector rank increments decide: [0,0] = J t' (or Q_1 t') exact, [1,0] = not exact.
set -eo pipefail
cd ${REPO}/scripts/oneloop/su-threshold-classes/cluster
PARTS=${PARTS:-cs}
L=${REPO}/log; C=cache/fullpar_hw_rank_3; V=classes/r007_d1c
sb() { sbatch --parsable -p $PARTS "$@"; }
M=$(sb --cpus-per-task=24 --mem=120G --job-name=d1c-make --output=$L/d1cmake_%j.out --wrap "source ${REPO}/scripts/oneloop/su-threshold-classes/cluster/r007_env.sh; cd ${REPO}/scripts/oneloop/su-threshold-classes_orders && python3 -u d1_make.py --cand-only --cand classes/r007_o2cand --out $V --workers \$SLURM_CPUS_PER_TASK" --time=03:00:00 --account=${ACCOUNT})
VE=$(sb --dependency=afterok:$M --cpus-per-task=32 --mem=160G --time=08:00:00 veval_generic.sbatch prep/r007_o3_out_codd cache/fullpar_hw_r007_o3_out_codd.pkl $V/vecs_Q1.pkl prep/r007_d1c_U2 --log $L/oprep_18905656.out)
RQ=$(sb --dependency=afterok:$VE --cpus-per-task=32 --mem=120G --time=04:00:00 r007_orders_cpurank_ff.sbatch r007_d1c_U2)
CK=$(sb --dependency=afterok:$M --cpus-per-task=16 --mem=100G --job-name=d1c-check --output=$L/d1ccheck_%j.out --time=03:00:00 --account=${ACCOUNT} --wrap "source ${REPO}/scripts/oneloop/su-threshold-classes/cluster/r007_env.sh; cd ${REPO}/scripts/oneloop/su-threshold-classes_orders && python3 -u d1_check.py $V --workers \$SLURM_CPUS_PER_TASK")
J=$(sb --dependency=afterok:$M --cpus-per-task=16 --mem=120G --time=03:00:00 jt_make.sbatch --src $V)
declare -A PREP=( [Kp]=o2lw_Kp [Tp0]=o2lw_T0 [S2]=o2lw_S2 [S20]=o2lw_S20 [Tp21]=o2lw_T21 [D]=r007_o2cand_D )
declare -A CACHE=( [Kp]=3-5-13-7-7 [Tp0]=4-6-15-7-7 [S2]=4-4-14-8-6 [S20]=4-4-12-8-8 [Tp21]=4-6-13-9-7 [D]=3-7-14-8-8 )
declare -A LOGS=( [Kp]=$L/gprep_19185269.out [Tp0]=$L/gprep_19185270.out [S2]=$L/gprep_19185271.out [S20]=$L/gprep_19185272.out
                  [Tp21]=$L/sprep_19185274_0.out,$L/sprep_19185274_1.out,$L/sprep_19185274_2.out,$L/sprep_19185274_3.out
                  [D]=$(echo $L/sprep_19177964_{0,1,2,3,4,5}.out | tr ' ' ,) )
declare -A MEM=( [Kp]=80G [Tp0]=100G [S2]=140G [S20]=140G [Tp21]=220G [D]=200G )
declare -A VMEM=( [Kp]=120G [Tp0]=120G [S2]=140G [S20]=140G [Tp21]=160G [D]=200G )
OUT="d1c: make $M veval $VE rank $RQ check $CK | jt $J |"
for op in Kp Tp0 S2 S20 Tp21 D; do
  extra=""; [ $op = D ] && extra="--orig-v classes/rank_3_3-7-14-8-8_9/ctrl.pkl,classes/rank_3_3-7-14-8-8_9/q1.pkl"
  ve=$(sb --dependency=afterok:$J --cpus-per-task=32 --mem=${VMEM[$op]} --time=08:00:00 veval_generic.sbatch prep/${PREP[$op]} ${C}_${CACHE[$op]}_9_codd.pkl $V/vecs_$op.pkl prep/${PREP[$op]}_Vc --log ${LOGS[$op]} $extra)
  rk=$(sb --dependency=afterok:$ve --cpus-per-task=32 --mem=${MEM[$op]} --time=05:00:00 r007_orders_cpurank_ff.sbatch ${PREP[$op]}_Vc)
  OUT="$OUT $op: veval $ve rank $rk |"
done
echo "$OUT"
