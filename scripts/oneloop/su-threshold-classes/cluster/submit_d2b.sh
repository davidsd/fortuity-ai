#!/bin/bash
# R-007 d_2, second independent chain (different kernel vectors, hence different y and y'): kernel B -> make B ->
# veval B -> rank on one H200.  Added because the first kernel job (19290480) landed on a slow node.
set -o pipefail
cd ${REPO}/scripts/oneloop/su-threshold-classes/cluster
L=${REPO}/log
MARK=$L/d2_submitted.txt; touch $MARK
got() { grep -m1 "^$1 [0-9]" $MARK | cut -d' ' -f2; }
sub() { local name=$1; shift; local id; id=$(got $name); if [ -z "$id" ]; then id=$(sbatch --parsable "$@") || return 1;
        [ -n "$id" ] || return 1; echo "$name $id" >> $MARK; fi; echo $id; }
KE=$(sub kernelB d2_kernel_b.sbatch) || exit 1
MK=$(sub makeB --dependency=afterok:$KE d2_make_b.sbatch) || exit 1
VE=$(sub vevalB --dependency=afterok:$MK --cpus-per-task=24 --mem=300G --time=10:00:00 veval_generic.sbatch prep/r007_o2_in_codd cache/fullpar_hw_r007_o2_in_codd.pkl classes/r007_d2b/vecs_z.pkl prep/r007_d2b_T2 --log $L/oprep_18905662.out) || exit 1
RG=$(sub rankgpuB --dependency=afterok:$VE --mem=200G r007_orders_gpu.sbatch r007_d2b_T2) || exit 1
echo "CHAIN B SUBMITTED: kernel $KE make $MK veval $VE rank gpu $RG"
