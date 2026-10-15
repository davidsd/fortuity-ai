#!/bin/bash
# R-007 d_2, third chain with the kernel step on a GPU (the CPU kernel job 19290480 sits on a node whose cores are
# shared with another job): augmented prep -> kernel_vec.py on one H200 -> make C -> veval C -> rank on one H200.
set -o pipefail
cd ${REPO}/scripts/oneloop/su-threshold-classes/cluster
L=${REPO}/log
MARK=$L/d2_submitted.txt; touch $MARK
got() { grep -m1 "^$1 [0-9]" $MARK | cut -d' ' -f2; }
sub() { local name=$1; shift; local id; id=$(got $name); if [ -z "$id" ]; then id=$(sbatch --parsable "$@") || return 1;
        [ -n "$id" ] || return 1; echo "$name $id" >> $MARK; fi; echo $id; }
AU=$(sub augC d2_aug.sbatch) || exit 1
KE=$(sub kernelC --dependency=afterok:$AU d2_kernel_gpu.sbatch) || exit 1
MK=$(sub makeC --dependency=afterok:$KE d2_make_c.sbatch) || exit 1
VE=$(sub vevalC --dependency=afterok:$MK --cpus-per-task=24 --mem=300G --time=10:00:00 veval_generic.sbatch prep/r007_o2_in_codd cache/fullpar_hw_r007_o2_in_codd.pkl classes/r007_d2c/vecs_z.pkl prep/r007_d2c_T2 --log $L/oprep_18905662.out) || exit 1
RG=$(sub rankgpuC --dependency=afterok:$VE --mem=200G r007_orders_gpu.sbatch r007_d2c_T2) || exit 1
echo "CHAIN C SUBMITTED: aug $AU kernel(gpu) $KE make $MK veval $VE rank gpu $RG"
