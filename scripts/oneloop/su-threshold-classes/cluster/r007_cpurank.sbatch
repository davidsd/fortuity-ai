#!/bin/bash
#SBATCH --job-name=r007-cpurank
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --output=${REPO}/log/cpurank_%j.out
# usage: sbatch --cpus-per-task=C --mem=M --dependency=afterok:<prep job> r007_cpurank.sbatch CASE [CASE ...]
set -eo pipefail
source ${REPO}/scripts/oneloop/su-threshold-classes/cluster/r007_env.sh
for CASE in "$@"; do
  echo "=== $CASE $(date)"
  python3 cpu_rank.py prep/$CASE
done
echo "CPU RANK DONE $(date)"
