#!/bin/bash
#SBATCH --job-name=t025-dims
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=${REPO}/log/dims_%j.out
# usage: sbatch dims.sbatch N JMAX  -> exact nf-refined hw multiplicities of SO(2N+1), all 2J, j <= JMAX
set -eo pipefail
source ${REPO}/env.sh
N=$1; J=$2; F=dims_so$((2*N+1))_j${J}.pkl
[ -s $F ] && { echo "$F exists"; exit 0; }
python3 -u -c "
import pickle, time
from fortuity.hwcoh.dims import partition_dims_nf
t = time.time(); d = partition_dims_nf($N, $J, group='SO_odd', workers=$SLURM_CPUS_PER_TASK, progress=True)
pickle.dump(d, open('$F.tmp', 'wb')); print('dims SO($((2*N+1))) j<=$J:', len(d), 'entries', f'{time.time()-t:.0f}s')"
mv $F.tmp $F
echo "DIMS DONE $(date)"
