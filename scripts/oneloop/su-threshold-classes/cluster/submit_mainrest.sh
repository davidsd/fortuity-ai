#!/bin/bash
# R-007 order 2, main chain (12 kernel-aug vectors + 2 candidate rows, classes/r007_d1): d_1 and the six LW tests with
# the veval settings that fit (24 workers, 300 GB) and per-sector cpu_rank increments.  Idempotent (marker file).
set -o pipefail
cd ${REPO}/scripts/oneloop/su-threshold-classes/cluster
L=${REPO}/log; C=cache/fullpar_hw_rank_3; V=classes/r007_d1; JT=${JT:-19191214}
MARK=$L/mainrest_submitted.txt; touch $MARK
got() { grep -m1 "^$1 [0-9]" $MARK | cut -d' ' -f2; }
sub() { local name=$1; shift; local id; id=$(got $name); if [ -z "$id" ]; then id=$(sbatch --parsable "$@") || return 1;
        [ -n "$id" ] || return 1; echo "$name $id" >> $MARK; fi; echo $id; }
VQ=$(sub vevalQ --cpus-per-task=24 --mem=300G --time=10:00:00 veval_generic.sbatch prep/r007_o3_out_codd cache/fullpar_hw_r007_o3_out_codd.pkl $V/vecs_Q1.pkl prep/r007_d1_U2 --log $L/oprep_18905656.out) || exit 1
RQ=$(sub rankQ --dependency=afterok:$VQ --cpus-per-task=32 --mem=120G --time=05:00:00 r007_orders_cpurank_ff.sbatch r007_d1_U2) || exit 1
declare -A PREP=( [Kp]=o2lw_Kp [Tp0]=o2lw_T0 [S2]=o2lw_S2 [S20]=o2lw_S20 [Tp21]=o2lw_T21 [D]=r007_o2cand_D )
declare -A CACHE=( [Kp]=3-5-13-7-7 [Tp0]=4-6-15-7-7 [S2]=4-4-14-8-6 [S20]=4-4-12-8-8 [Tp21]=4-6-13-9-7 [D]=3-7-14-8-8 )
declare -A LOGS=( [Kp]=$L/gprep_19185269.out [Tp0]=$L/gprep_19185270.out [S2]=$L/gprep_19185271.out [S20]=$L/gprep_19185272.out
                  [Tp21]=$L/sprep_19185274_0.out,$L/sprep_19185274_1.out,$L/sprep_19185274_2.out,$L/sprep_19185274_3.out
                  [D]=$(echo $L/sprep_19177964_{0,1,2,3,4,5}.out | tr ' ' ,) )
declare -A MEM=( [Kp]=80G [Tp0]=100G [S2]=140G [S20]=140G [Tp21]=220G [D]=200G )
OUT="d1: veval $VQ rank $RQ |"
for op in Kp Tp0 S2 S20 Tp21 D; do
  extra=""; [ $op = D ] && extra="--orig-v classes/rank_3_3-7-14-8-8_9/ctrl.pkl,classes/rank_3_3-7-14-8-8_9/q1.pkl"
  ve=$(sub veval$op --dependency=afterok:$JT --cpus-per-task=24 --mem=300G --time=10:00:00 veval_generic.sbatch prep/${PREP[$op]} ${C}_${CACHE[$op]}_9_codd.pkl $V/vecs_$op.pkl prep/${PREP[$op]}_V --log ${LOGS[$op]} $extra) || exit 1
  rk=$(sub rank$op --dependency=afterok:$ve --cpus-per-task=32 --mem=${MEM[$op]} --time=05:00:00 r007_orders_cpurank_ff.sbatch ${PREP[$op]}_V) || exit 1
  OUT="$OUT $op: veval $ve rank $rk |"
done
echo "ALL SUBMITTED: $OUT"
