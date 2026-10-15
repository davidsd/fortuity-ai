#!/bin/bash
#SBATCH --job-name=t042-dims
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=${SCRATCH}/t042/log/dims_%j.out
# exact nf-refined SU(3)_R hw multiplicities of SU(3) BMN sectors, j <= 32 (serial SU path of partition_dims_nf)
set -eo pipefail
source ${SCRATCH}/t042/env.sh
F=dims_su3_j32.pkl
[ -s $F ] && { echo "$F exists"; exit 0; }
python3 -u -c "
import pickle, time
from fortuity.hwcoh.dims import partition_dims_nf
t = time.time(); d = partition_dims_nf(3, 32, progress=True)
pickle.dump(d, open('$F.tmp', 'wb')); print('dims SU(3) j<=32:', len(d), 'entries', f'{time.time()-t:.0f}s')"
mv $F.tmp $F
echo "DIMS DONE $(date)"
