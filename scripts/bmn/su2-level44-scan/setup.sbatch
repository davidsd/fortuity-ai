#!/bin/bash
#SBATCH --job-name=fort-setup
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=log/setup_%j.out
# Build $SCRATCH/fortuity-venv ON A COMPUTE NODE (node images differ from the login node).
set -e
cd $SCRATCH/fortuity-torch
mkdir -p log results
PY=""
for c in python3.12 python3.11 python3.10 python3; do
  if command -v $c >/dev/null && $c -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)'; then PY=$(command -v $c); break; fi
done
if [ -z "$PY" ]; then module load python 2>/dev/null || true; PY=$(command -v python3); fi
$PY -c 'import sys; assert sys.version_info >= (3,10), sys.version' || { echo "NO PYTHON >= 3.10 ON THIS NODE -- aborting"; exit 1; }
echo "using $PY ($($PY --version))"
rm -rf $SCRATCH/fortuity-venv
$PY -m venv $SCRATCH/fortuity-venv
source $SCRATCH/fortuity-venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -e ./code
python3 -c "import fortuity.hwcoh.sector, numpy, sympy, sys; print('ok: python', sys.version.split()[0], 'numpy', numpy.__version__)"
