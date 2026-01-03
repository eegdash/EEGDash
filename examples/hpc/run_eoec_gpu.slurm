#!/bin/bash
#SBATCH -J eegdash_eoec_gpu
#SBATCH -A csd403
#SBATCH -p gpu-shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err

set -euo pipefail

module purge
module load singularitypro

WORKDIR=/home/kkokate/eegdash/tutorial
CONTAINER=${WORKDIR}/eegdash-tutorial.sif
SCRIPT=${WORKDIR}/tutorial_eoec.py
DATA_ROOT=/expanse/projects/nemar/openneuro

cd ${WORKDIR}

# Avoid CPU oversubscription
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Experiment config
export EPOCHS=20
export NUM_SUBJECTS=10
export NUM_TEST_SUBJECTS=2
export BATCH_SIZE=64
export SEED=42

# Run in container with GPU
singularity exec --nv \
  --bind ${DATA_ROOT}:${DATA_ROOT} \
  --bind ${WORKDIR}:${WORKDIR} \
  ${CONTAINER} \
  python ${SCRIPT}
