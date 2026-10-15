#!/bin/bash
#SBATCH --job-name=t040-d2
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=${REPO}/log/t040d2_%A_%a.out
# T-040, two loops: d_2 on the one-loop survivors of one SU(2) BMN sector (line SLURM_ARRAY_TASK_ID of
# t040_d2_cases.txt: "<case> <representatives>"): one-loop prep, kernel of d_1 on the representatives, y, z = K_2 U x +
# Q_1 y with the closedness checks (d2_make_v2.py = d2_make.py of the repo), exactness of z in T_2 (fullpar), with six
# random closed y' for the rank of d_1 into T_2.
set -eo pipefail
source ${REPO}/scripts/oneloop/su-threshold-classes/cluster/r007_env.sh
cd ${REPO}/scripts/oneloop/su-threshold-classes_orders
read CASE REPS <<< "$(sed -n "${SLURM_ARRAY_TASK_ID}p" t040_d2_cases.txt)"
D=$(echo $CASE | sed 's/:/_/g; s/,/-/g')
T2=$(python3 -c "
import sys
g, twoJ, q = sys.argv[1].split(':'); q = [int(x) for x in q.split(',')]; twoJ = int(twoJ)
print('rank:%s:%d,%d,%d,%d,%d:%d' % (g[2:], twoJ, twoJ, 2*q[0], 2*q[1], 2*q[2], sum(q) - twoJ - 1))" $CASE)
T2D=$(echo $T2 | sed 's/:/_/g; s/,/-/g')
echo "node $(hostname) case $CASE reps $REPS dir $D T2 $T2"
cp classes/$D/fullpar_seed41.json $TMPDIR/keep_seed41.json           # --prep-dir overwrites the one-loop result file
python3 -u fullpar.py $CASE --workers $SLURM_CPUS_PER_TASK --seed 41 --workdir $TMPDIR --prep-dir prep/d2_$D
cp $TMPDIR/keep_seed41.json classes/$D/fullpar_seed41.json
python3 -u fullpar.py $T2 --workers $SLURM_CPUS_PER_TASK --workdir $TMPDIR --hw-only
python3 -u d2_make_v2.py --case $D --N 2 --prep prep/d2_$D --reps $REPS --nclosed 6 --libnull --fullpar $T2 \
    --in-cache cache/fullpar_hw_$T2D.pkl --out classes/d2_$D --workers $SLURM_CPUS_PER_TASK
python3 -u fullpar.py $T2 --workers $SLURM_CPUS_PER_TASK --workdir $TMPDIR --span
echo "RESULT $CASE $(python3 -c "
import json, sys
d = json.load(open('classes/$T2D/fullpar_seed41.json')); m = json.load(open('classes/d2_$D/d2_make.json'))
print('rows', m['test_rows'], 'increments', d['lifted_increments'], 'controls_exact', d['controls_exact'], 'rank_B', d['rank_B'], 'span', d.get('span', {}).get('spans'))")"
echo "T040 D2 DONE $(date)"
