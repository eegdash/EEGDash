# Run the existing EEGDash workload on Slurm

**Scope:** submit the repository's real EO/EC preprocessing-and-training script.
This is **not** a generic preprocessing CLI or a tested subject-array pipeline.
There is no `eegdash.preprocess_main` entry point. For reusable prepared windows,
see tutorial 13; adapting that workflow to your cohort requires your own tested
analysis script and output contract.

The checked-in entry points are:

- [`../hpc/tutorial_hpc_cache_and_slurm.py`](../hpc/tutorial_hpc_cache_and_slurm.py):
  bounded HBN acquisition, EEGPrep components, instructed-state windows and a
  fixed-budget participant-held-out model.
- [`../hpc/run_eoec_cpu.slurm`](../hpc/run_eoec_cpu.slurm): one task, eight CPUs,
  32 GB RAM, one hour; invokes that Python file from `SLURM_SUBMIT_DIR`.
- [`../hpc/run_eoec_gpu.slurm`](../hpc/run_eoec_gpu.slurm): the same allocation
  plus one GPU. These are starting requests, **not measured minimum resources**.

## Stage first, submit second

Install the project and compatible EEGPrep/PyTorch dependencies on your site.
Use a network-enabled host to stage the exact default selection before submitting
an offline job (281.2 MB of signals plus sidecars). From the **repository root**:

```bash
export EEGDASH_PYTHON="$PWD/.venv/bin/python"
export EEGDASH_CACHE_DIR=/path/to/persistent/eegdash-cache
export EEGDASH_OUTPUT_DIR=/path/to/persistent/eegdash-results
"$EEGDASH_PYTHON" - <<'PY'
from eegdash import EEGDashDataset
from eegdash.paths import get_default_cache_dir

for subject in ("NDARAE710YWG", "NDARAH239PGG", "NDARAL897CYV"):
    dataset = EEGDashDataset(
        cache_dir=get_default_cache_dir(), dataset="ds005514",
        subject=subject, task="RestingState", n_jobs=1,
    )
    if len(dataset.datasets) != 1:
        raise ValueError(f"Expected one RestingState recording for {subject}.")
    print(dataset.description)
    dataset.download_all(n_jobs=1)
PY

# Slurm opens log files before the script starts: create this directory now.
mkdir -p logs "$EEGDASH_OUTPUT_DIR"
export EEGDASH_OFFLINE=1
sbatch --account=YOUR_ACCOUNT --partition=YOUR_CPU_PARTITION \
  --output='logs/%x-%j.out' examples/hpc/run_eoec_cpu.slurm
```

Change the staging cohort and `SUBJECTS` together if needed; `NUM_SUBJECTS`
selects a prefix. Do not submit a one-person array task to a workload that needs
both training and held-out participants. Each complete workload is one job.
For CUDA use the GPU template after checking the site's PyTorch/CUDA environment.

## Output ownership and failure behavior

Each invocation exclusively creates `eoec-*` beneath `EEGDASH_OUTPUT_DIR`.
It never deletes or replaces an older run. `configuration.json` records the
requested experiment; successful runs add `training.csv`, `metrics.json`, two
labelled figures and finally `_SUCCESS`. A failed run can leave diagnostic
partial output **without** `_SUCCESS`; downstream jobs must not treat that
folder as complete. Scheduler logs/`sacct -j JOB_ID --format=JobID,State,ExitCode`
remain the failure record, including kills before Python can report an error.
There is no combined cross-job status manifest or automatic retry claim.

Prefer persistent output with scratch used only for staged inputs. If your site
requires output on scratch, copy the entire completed run to a **new unique**
persistent directory, verify the copy, then expose its completion marker last.
Do not delete a prior successful directory before copying, and do not call a
cross-filesystem move atomic. For atomic visibility, stage under the destination
filesystem and rename to a collision-free final name only after validation.
Design locking/collision handling before introducing shared subject-level names.

## Resource and offline checks

- The templates cap BLAS/OpenMP threads at the allocation; Python also caps Torch
  threads and uses one acquisition worker. Avoid multiplying workers by a full
  allocation of threads per worker.
- Keep the full BIDS tree when copying to node-local storage; set
  `EEGDASH_CACHE_DIR` to its root. Verify free space and offline opening first.
- Request a writable HOME/config location according to site policy; the example
  does not change private MNE settings or silently disable JIT.
- Right-size memory/time using your cluster's accounting after a representative
  job. An out-of-memory failure is not evidence that merely adding CPUs helps.

**Verification limit:** the paths, invocation and resource directives above were
checked against repository source. No scheduler submission, container build or
cluster execution was performed in this revision. Test account, partition,
filesystem, environment propagation and GPU access on the target cluster before
scaling up. See [the HPC instructions](../hpc/instructions.md).
