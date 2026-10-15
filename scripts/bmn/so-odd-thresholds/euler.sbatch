#!/bin/bash
#SBATCH --job-name=t025-euler
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=${REPO}/log/euler_%j.out
# per-column Euler characteristic of the SO(11) scan vs the exact refined BMN index (euler_check.py)
set -eo pipefail
source ${REPO}/env.sh
cd ${REPO}/code
FORTUITY_WORKERS=$SLURM_CPUS_PER_TASK python3 -u ../scripts/bmn/so-usp-g2-scans/euler_check.py SO_odd 5 24 ../data/so_sp/so11_hw_cohomology_j24.jsonl
echo "EULER DONE $(date)"
