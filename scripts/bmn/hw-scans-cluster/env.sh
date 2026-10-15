# source me: module stack for the fortuity code on Expanse (verified 2026-09-26)
module purge
module load cpu/0.21.2a aocc/4.2.0/unqsjyx python/3.11.9/xeukfw7 py-numpy/1.26.4/n3rmtzt
export PYTHONPATH=${REPO}/code:$PYTHONPATH
# storage: node-local NVMe $TMPDIR (=/scratch/$USER/job_$SLURM_JOB_ID, ~900 GB, wiped at job end) for work;
# Lustre scratch for anything kept. Never write outputs to $HOME.
export FORT_STORE=${STORE}
export FORT_WORK=${TMPDIR:-$FORT_STORE/tmp}
