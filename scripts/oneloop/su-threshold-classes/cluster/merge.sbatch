#!/bin/bash
#SBATCH --job-name=merge
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=${REPO}/log/merge_%j.out
# usage: sbatch merge.sbatch NAME K   (reassembles prep/NAME/slice_*ofK into prep/NAME/B.i32, V.i32, meta.json)
set -eo pipefail
source ${REPO}/scripts/oneloop/su-threshold-classes/cluster/r007_env.sh
cd ${REPO}/scripts/oneloop/su-threshold-classes_orders
python3 -u merge_slices.py prep/$1 $2
test -s prep/$1/meta.json
rm -rf prep/$1/slice_*of$2
echo "MERGE DONE $(date)"
