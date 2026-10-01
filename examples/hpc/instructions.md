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
export EEGDASH_OUTPUT_DIR=/path/to/persistent/eegdash-results
# After staging this exact cohort (see the linked staging recipe):
export EEGDASH_OFFLINE=1
mkdir -p logs "$EEGDASH_OUTPUT_DIR"
sbatch --account=YOUR_ACCOUNT --partition=YOUR_CPU_PARTITION \
  --output='logs/%x-%j.out' examples/hpc/run_eoec_cpu.slurm
```

For a GPU environment with CUDA-enabled PyTorch:

```bash
sbatch --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION \
  --output='logs/%x-%j.out' examples/hpc/run_eoec_gpu.slurm
```

Both templates use `SLURM_SUBMIT_DIR`, so submit from the repository root.
Override `NUM_SUBJECTS`, `NUM_TEST_SUBJECTS`, `EPOCHS`, `BATCH_SIZE` or `SEED`
in the submitting environment. Keep at least one training and one test subject.
For node-local storage, pre-stage the existing cache to `SLURM_TMPDIR` in the
batch script and point `EEGDASH_CACHE_DIR` there. Copy outputs back before the
allocation ends. The script creates a unique `eoec-*` directory beneath
`EEGDASH_OUTPUT_DIR` (default: cache/hpc-runs), preserving older results and
partial failures. It saves a labelled voltage figure, `training.csv` and `metrics.json`;
only completed runs receive `_SUCCESS`. No reusable
model checkpoint is saved. Keep the output root on persistent storage.

Inspect `logs/*.out` for the exact participant IDs, recording metadata,
window counts, label balance, and final held-out accuracy. Low accuracy is a
valid measured result. The test cohort is evaluated only after training.

Stage the selected recordings first using the [staging recipe](../how_to/how_to_run_preprocessing_on_slurm.md).
`EEGDASH_OFFLINE=1` prevents dataset acquisition on the compute node; leave it
at `0` only where acquisition is intended and permitted. The CPU template
requests eight CPUs, 32 GB and one hour; GPU adds one GPU. These are starting
allocations, not measured minima. The script respects the CPU allocation for
Torch threads; measure memory and wall time before scaling the cohort.

## Optional container

The Dockerfile supplies a Python environment; site Slurm templates above use
the activated environment directly. Build from the repository root:

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
  --bind "$EEGDASH_OUTPUT_DIR:$EEGDASH_OUTPUT_DIR" \
  eegdash-hpc.sif python examples/hpc/tutorial_hpc_cache_and_slurm.py
```

Add `--nv` for NVIDIA GPU access. Container construction and scheduler execution
must be tested on your cluster; local source checks do not validate them.
