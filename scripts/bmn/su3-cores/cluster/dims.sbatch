#!/bin/bash
#SBATCH --job-name=su3dims
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
# Only needed if dims_nf_su3_36.pkl was not copied from the laptop:  sbatch -p yin dims.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source env.sh
python3 -u -c "
import pickle, time
from fortuity.hwcoh.dims import partition_dims_nf
t = time.time(); d = partition_dims_nf(3, 36)
pickle.dump(d, open('dims_nf_su3_36.pkl', 'wb')); print('dims', len(d), f'{time.time()-t:.0f}s')"
