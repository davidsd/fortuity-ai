#!/bin/bash
# R-007 order 2, the second-order differential d_2 (summary/fortuity-summary.pdf): transpose -> kernel (c B = -Q_1 x) -> make
# (y, z, checks, test rows) -> veval at the T_2 inbound prep -> rank increments on CPU (MKL) and on one H200.
# Idempotent (marker file): a rerun submits only what is missing.
set -o pipefail
cd ${REPO}/scripts/oneloop/su-threshold-classes/cluster
L=${REPO}/log
MARK=$L/d2_submitted.txt; touch $MARK
got() { grep -m1 "^$1 [0-9]" $MARK | cut -d' ' -f2; }
sub() { local name=$1; shift; local id; id=$(got $name); if [ -z "$id" ]; then id=$(sbatch --parsable "$@") || return 1;
        [ -n "$id" ] || return 1; echo "$name $id" >> $MARK; fi; echo $id; }
TR=$(sub transpose d2_transpose.sbatch) || exit 1
KE=$(sub kernel --dependency=afterok:$TR d2_kernel.sbatch) || exit 1
MK=$(sub make --dependency=afterok:$KE d2_make.sbatch) || exit 1
VE=$(sub veval --dependency=afterok:$MK --cpus-per-task=24 --mem=300G --time=10:00:00 veval_generic.sbatch prep/r007_o2_in_codd cache/fullpar_hw_r007_o2_in_codd.pkl classes/r007_d2/vecs_z.pkl prep/r007_d2_T2 --log $L/oprep_18905662.out) || exit 1
RC=$(sub rankcpu --dependency=afterok:$VE --cpus-per-task=64 --mem=450G --time=12:00:00 r007_orders_cpurank_ff.sbatch r007_d2_T2) || exit 1
RG=$(sub rankgpu --dependency=afterok:$VE --mem=200G r007_orders_gpu.sbatch r007_d2_T2) || exit 1
echo "ALL SUBMITTED: transpose $TR kernel $KE make $MK veval $VE rank cpu $RC gpu $RG"
