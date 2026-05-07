HPC tutorials
=============

Reference setup for running EEGDash on shared HPC clusters: SLURM
submission scripts (CPU and GPU), a Dockerfile, and a tutorial showing
how to combine the on-disk cache with batch scheduling for an eyes-open
/ eyes-closed run. Difficulty 2; assumes the core workflow track and
that you have admin access on a SLURM cluster.

Single-machine examples cover the API; the HPC track covers the
operational layer that lets you run those examples at the scale a real
study requires. The cache layout, environment setup, and SLURM array
patterns here are the same ones used in production for the EEGDash
benchmark runs. Sourced from
``docs/tutorial_restructure_plan.md`` Category J (HPC operations).
Cross-link with the *How-To* guide for parallel feature extraction.

What you will learn:

- How to package an EEGDash environment in a Docker image so cluster
  jobs see the same dependencies you tested locally.
- How to write a SLURM submission script that pins one EEGDash job per
  GPU (``run_eoec_gpu.slurm``) or per CPU node (``run_eoec_cpu.slurm``).
- How to lay out the on-disk EEGDash cache so concurrent jobs share
  preprocessed windows without race conditions.
- How to run an eyes-open / eyes-closed end-to-end study from a single
  ``sbatch`` invocation, with logs and outputs landing in the right
  cache location for downstream analysis.

Each script in this directory is a runnable example; ``instructions.md``
walks through the cluster-side setup once before you launch.
