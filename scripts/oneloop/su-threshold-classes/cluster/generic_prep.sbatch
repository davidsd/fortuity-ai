#!/bin/bash
#SBATCH --job-name=gen-prep
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --output=${REPO}/log/gprep_%j.out
# usage: sbatch --cpus-per-task=C --mem=M generic_prep.sbatch PREPNAME CASE [fullpar flags]   -> oneloop_orders/prep/PREPNAME
set -eo pipefail
source ${REPO}/scripts/oneloop/su-threshold-classes/cluster/r007_env.sh
cd ${REPO}/scripts/oneloop/su-threshold-classes_orders
NAME=$1; shift
rm -rf prep/$NAME
python3 -u fullpar.py "$@" --workers $SLURM_CPUS_PER_TASK --xcap 12e9 --workdir $TMPDIR --prep-dir prep/$NAME
test -s prep/$NAME/meta.json
echo "PREP DONE $(date)"
