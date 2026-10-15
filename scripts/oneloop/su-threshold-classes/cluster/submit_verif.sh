#!/bin/bash
# Idempotent submission of the candidate's verification jobs (one marker line per job in $MARK); safe to rerun while the
# Slurm controller is flaky: already-submitted jobs are skipped, and a failed sbatch stops the script.
set -eo pipefail
cd ${REPO}/scripts/oneloop/su-threshold-classes/cluster
L=${REPO}/log; V=classes/r007_d1c
MARK=${REPO}/log/verif_submitted.txt; touch $MARK
ENV="source ${REPO}/scripts/oneloop/su-threshold-classes/cluster/r007_env.sh; cd ${REPO}/scripts/oneloop/su-threshold-classes_orders"
got() { grep -m1 "^$1 " $MARK | cut -d' ' -f2; }
sub() { local name=$1; shift; local id; id=$(got $name); if [ -z "$id" ]; then id=$(sbatch --parsable "$@") || return 1; [ -n "$id" ] || return 1; echo "$name $id" >> $MARK; fi; echo $id; }
H=$(sub hwcheck -p cs --account=${ACCOUNT} --cpus-per-task=32 --mem=200G --time=04:00:00 --job-name=hwcheck --output=$L/hwcheck_%j.out --wrap "$ENV && python3 -u hwcheck.py $V/vecs_Tp21.pkl $V/vecs_Tp0.pkl $V/vecs_S2.pkl $V/vecs_Kp.pkl $V/vecs_D.pkl $V/vecs_S20.pkl --t $V --workers \$SLURM_CPUS_PER_TASK") || exit 1
SP=$(CASE="rank:3:4,6,13,9,7:9" K=4 sub spanT21 --export=ALL --array=0-3 r007_orders_span_slice.sbatch) || exit 1
VQ=$(sub vevalQ --cpus-per-task=24 --mem=300G --time=10:00:00 veval_generic.sbatch prep/r007_o3_out_codd cache/fullpar_hw_r007_o3_out_codd.pkl $V/vecs_Q1.pkl prep/r007_d1c_U2 --log $L/oprep_18905656.out) || exit 1
RQ=$(sub rankQ --dependency=afterok:$VQ --cpus-per-task=32 --mem=120G --time=05:00:00 r007_orders_cpurank_ff.sbatch r007_d1c_U2) || exit 1
VS=$(sub vevalS20 --cpus-per-task=24 --mem=300G --time=10:00:00 veval_generic.sbatch prep/o2lw_S20 cache/fullpar_hw_rank_3_4-4-12-8-8_9_codd.pkl $V/vecs_S20.pkl prep/o2lw_S20_Vc --log $L/gprep_19185272.out) || exit 1
RS=$(sub rankS20 --dependency=afterok:$VS --cpus-per-task=32 --mem=140G --time=05:00:00 r007_orders_cpurank_ff.sbatch o2lw_S20_Vc) || exit 1
echo "ALL SUBMITTED: hwcheck $H | span T21 $SP | d1c veval $VQ rank $RQ | S20c veval $VS rank $RS"
