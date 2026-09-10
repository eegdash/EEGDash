# Run the real HBN workload with Slurm

From the repository root, create an environment with the project's dependencies
and activate it before submitting a job. The default workload loads three actual
HBN participants, trains for six epochs, and evaluates one held-out participant.
The default IDs are NDARAE710YWG, NDARAH239PGG and NDARAL897CYV.
Set `SUBJECTS` to a comma-separated explicit cohort to change it.
Inspect catalogue file sizes before the first download. A failed acquisition
stops the job; the script does not replace or skip selected participants.

```bash
export EEGDASH_PYTHON="$PWD/.venv/bin/python"
export EEGDASH_CACHE_DIR=/path/to/persistent/eegdash-cache
sbatch --account=YOUR_ACCOUNT --partition=YOUR_CPU_PARTITION examples/hpc/run_eoec_cpu.slurm
```

For a GPU environment with CUDA-enabled PyTorch:

```bash
sbatch --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION examples/hpc/run_eoec_gpu.slurm
```

Both templates use `SLURM_SUBMIT_DIR`, so submit from the repository root.
Override `NUM_SUBJECTS`, `NUM_TEST_SUBJECTS`, `EPOCHS`, `BATCH_SIZE` or `SEED`
in the submitting environment. Keep at least one training and one test subject.
For node-local storage, pre-stage the existing cache to `SLURM_TMPDIR` in the
batch script and point `EEGDASH_CACHE_DIR` there. Copy outputs back before the
allocation ends. The script writes `sample_epoch.png` in the working directory;
use a separate checkout or output directory for concurrent jobs.

Inspect `slurm-*.out` for the exact participant IDs, recording metadata,
window counts, label balance, and final held-out accuracy. Low accuracy is a
valid measured result. The test cohort is evaluated only after training.

## Optional container

The Dockerfile supplies a Python environment; site Slurm templates above use
the activated environment directly. Build from this directory:

```bash
docker build -t eegdash-hpc examples/hpc
```

On a host with Apptainer and access to the Docker daemon, convert it:

```bash
apptainer build eegdash-hpc.sif docker-daemon://eegdash-hpc:latest
```

Replace the final Python invocation in the batch template with your site's
container command, binding both checkout and cache. For example:

```bash
apptainer exec --bind "$PWD:$PWD" --bind "$EEGDASH_CACHE_DIR:$EEGDASH_CACHE_DIR" \
  eegdash-hpc.sif python examples/hpc/tutorial_hpc_cache_and_slurm.py
```

Add `--nv` for NVIDIA GPU access. Container construction and scheduler execution
must be tested on your cluster; local source checks do not validate them.
