# How to run an EEGDash preprocessing job on SLURM

**Goal.** Copy a single `.slurm` template, point it at your project's environment
and cache directory, and run an EEGDash preprocessing entry-point reproducibly
on a SLURM cluster — sharded one-subject-per-task with per-task logs, atomic
output renames so a re-queued job picks up cleanly, and a final manifest
summarizing shard status.

This how-to is a template + commentary, not a Python script. It assumes you
already downloaded the data offline (see `how_to_download_a_dataset`) and know
where your shared cache lives (see `how_to_use_hpc_cache`).

## Prerequisites

- A SLURM-managed cluster with `sbatch`, `srun`, and one of `module` /
  `conda` / `uv` / `venv` available on compute nodes.
- A working EEGDash environment (`pip install eegdash` or a project venv).
- A scratch or project filesystem readable from every compute node, exposed via
  a cluster env var (commonly `$SCRATCH`, `$WORK`, or `$PSCRATCH`).
- A subject-list file `subjects.txt` (one subject ID per line) that defines
  the array shards.
- A preprocessing entry-point exposed as `python -m eegdash.preprocess_main`
  (or your own `scripts/preprocess.py`) that accepts `--subject` and
  `--out-dir`.

## SLURM template

Save the script below as `examples/how_to/preprocess.slurm`, edit the four
clearly marked variables at the top, then submit with
`sbatch --array=0-$(( $(wc -l < subjects.txt) - 1 ))%32 preprocess.slurm`.

```bash
#!/bin/bash
#SBATCH --job-name=eegdash_preproc
#SBATCH --partition=cpu                  # GPU variant: see "Variations" below
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%A_%a.out       # %A = jobid, %a = array task id
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --array=0-9%4                    # 10 shards, 4 concurrent (placeholder)

set -euo pipefail

# --- 1. EDIT THESE FOUR LINES -----------------------------------------------
PROJECT_ROOT="${HOME}/projects/eegdash"
SUBJECT_LIST="${PROJECT_ROOT}/subjects.txt"
DATASET="ds002718"
OUT_ROOT="${SCRATCH:-${HOME}/scratch}/eegdash_preproc/${DATASET}"
# ----------------------------------------------------------------------------

# --- 2. Cache: from cluster env var, NEVER hard-coded ---
export EEGDASH_CACHE_DIR="${SCRATCH:-${HOME}/scratch}/eegdash_cache"
mkdir -p "${EEGDASH_CACHE_DIR}" "${OUT_ROOT}" logs

# --- 3. Activate environment (pick ONE branch) ---
module purge
module load python/3.11           # cluster-specific; comment out if absent

# Conda branch:
# source "${HOME}/miniconda3/etc/profile.d/conda.sh"
# conda activate eegdash
# uv branch:
# source "${PROJECT_ROOT}/.venv/bin/activate"
# Plain venv branch:
source "${PROJECT_ROOT}/.venv/bin/activate"

# --- 4. Avoid CPU oversubscription inside numpy/MNE/sklearn ---
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# --- 5. Resolve this shard's subject ---
SUBJECT="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${SUBJECT_LIST}")"
SHARD_TMP="${OUT_ROOT}/_tmp/${SUBJECT}.${SLURM_JOB_ID}"
SHARD_FINAL="${OUT_ROOT}/${SUBJECT}"

# Idempotency: skip if already complete
if [[ -f "${SHARD_FINAL}/_SUCCESS" ]]; then
    echo "[skip] ${SUBJECT} already complete"; exit 0
fi
mkdir -p "${SHARD_TMP}"

# --- 6. Run preprocessing ---
cd "${PROJECT_ROOT}"
python -m eegdash.preprocess_main \
    --dataset "${DATASET}" \
    --subject "${SUBJECT}" \
    --cache-dir "${EEGDASH_CACHE_DIR}" \
    --out-dir "${SHARD_TMP}"

# --- 7. Atomic copy-back: rename only on success ---
touch "${SHARD_TMP}/_SUCCESS"
rm -rf "${SHARD_FINAL}"
mv "${SHARD_TMP}" "${SHARD_FINAL}"

# --- 8. Append to manifest (file-locked) ---
{ flock -x 200
  printf "%s\t%s\tOK\t%s\n" \
      "${SLURM_JOB_ID}" "${SUBJECT}" "$(date -Iseconds)" \
      >> "${OUT_ROOT}/manifest.tsv"
} 200>"${OUT_ROOT}/manifest.lock"
```

## Variations

**GPU jobs.** Add `#SBATCH --gres=gpu:1` (or `--gpus=1`), switch
`--partition=gpu`, and request fewer CPUs (typically 4–8). Most preprocessing
is CPU-bound (filtering, resampling); reach for GPU only when your
entry-point includes a deep model in the loop.

**Array jobs for multi-subject parallelism.** The template already shards
one-subject-per-task. Tune the array spec: `--array=0-499%50` runs 500
subjects with at most 50 concurrent tasks, which keeps the cache filesystem
from thrashing. Use `sacct -j <jobid> --format=JobID,State,ExitCode` to find
failed shards, then re-submit with `--array=12,47,103` to retry only those.

**Single-node multi-subject.** If subjects are tiny, drop `--array` and
spawn workers in-script with GNU parallel or
`python -m eegdash.preprocess_main --n-jobs ${SLURM_CPUS_PER_TASK}` — see
`how_to_parallelize_feature_extraction`.

## Pitfalls

- **Cache location.** Never hard-code paths under `$HOME` for the cache;
  home filesystems are typically slow and quota-bound. Use `$SCRATCH` (or
  whatever your site exposes) and verify with `df -h "${EEGDASH_CACHE_DIR}"`.
- **Env activation.** A login-shell `~/.bashrc` is *not* sourced inside SLURM
  jobs by default. Always activate the env explicitly inside the script —
  every branch shown above is a real activation, not a comment.
- **Log capture.** Use `%A_%a` (not `%j`) in `--output`/`--error` so each
  array task gets its own file; otherwise tasks clobber each other's logs.
- **OOM on long recordings.** If the kernel kills the job (`exit 137`),
  raise `--mem` first, then verify your preprocessing streams data from
  disk rather than loading the full recording. Watch `seff <jobid>` for
  memory headroom after a successful run and right-size on the next batch.
- **Partial outputs on requeue.** Always write to `_tmp/` and `mv` to the
  final path only after `_SUCCESS` exists — otherwise a node crash leaves
  half-written files that look complete to downstream code.

## See also

- `how_to_use_hpc_cache` — choosing and warming `EEGDASH_CACHE_DIR`.
- `how_to_parallelize_feature_extraction` — `n_jobs`, batch sizes, and
  joblib persistence inside one node.
- `how_to_work_offline` — preflight check that all required files are local.

## References

- Cisotto, G. & Chicco, D. (2024). *Ten quick tips for clinical electroencephalographic (EEG) data acquisition and signal processing.* PeerJ Computer Science 10:e2256. doi:[10.7717/peerj-cs.2256](https://doi.org/10.7717/peerj-cs.2256) — general scientific-computing best practices for reproducible EEG pipelines.
- Your cluster's SLURM user guide (e.g., NERSC Perlmutter, SDSC Expanse, Jean Zay) — partition names, account flags, and `module` stack are site-specific; treat the template above as a skeleton.
