#!/bin/bash
#SBATCH --job-name=eegdash-eoec
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
# Submit from the repository root; supply your site's account and partition.
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export EEGDASH_CACHE_DIR="${EEGDASH_CACHE_DIR:-$PWD/.eegdash_cache}"
export NUM_SUBJECTS="${NUM_SUBJECTS:-3}"
export NUM_TEST_SUBJECTS="${NUM_TEST_SUBJECTS:-1}"
export EPOCHS="${EPOCHS:-6}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export SEED="${SEED:-42}"
# Activate a project environment before submission, or set EEGDASH_PYTHON.
"${EEGDASH_PYTHON:-python}" examples/hpc/tutorial_hpc_cache_and_slurm.py
