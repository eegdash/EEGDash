# EEGDash on Expanse: end-to-end guide

This page documents a minimal, reproducible workflow for running EEGDash experiments on SDSC Expanse. It covers local development on macOS, CPU and GPU Slurm jobs, and the Docker plus Singularity (Apptainer) toolchain that links them.

---

## Docker Creation

### 1. Create a Dockerfile

Create a Dockerfile for your project containing all necessary requirements. Refer to the Dockerfile in this folder for an example.

### 2. Build and Push Your Docker Image

```bash
docker build -t eegdash-tutorial:latest .
docker tag eegdash-tutorial:latest <dockerhub_user>/eegdash-tutorial:latest
docker push <dockerhub_user>/eegdash-tutorial:latest
```

Replace `<dockerhub_user>` with your Docker Hub username.

### 3. Convert to Singularity on Expanse

Once on Expanse, start a compute node and convert your Docker image to Singularity format:

```bash
module purge
module load singularitypro
singularity pull eegdash-tutorial.sif docker://<dockerhub_user>/eegdash-tutorial:latest
```

This will automatically convert the Docker image to a `.sif` (Singularity Image Format) file.

### 4. Create and Submit Slurm Batch Jobs

Create your Slurm batch job files and submit them to run your experiments. Refer to the Slurm job files in this folder for examples.

**Key difference:** Instead of running `python code.py` directly, you run it inside the Singularity container:

```bash
singularity exec eegdash-tutorial.sif python code.py
```

---

## Alternative Workflow for Linux

For Linux distributions, you can convert the Docker image to Singularity/Apptainer format **locally** and then transfer it to Expanse:

```bash
# Install apptainer on your Linux machine (if not already installed)
# Convert Docker image to .sif locally
apptainer build eegdash-tutorial.sif docker-daemon://eegdash-tutorial:latest

# Transfer the .sif file to Expanse using scp
scp eegdash-tutorial.sif <username>@login.expanse.sdsc.edu:/path/to/destination/
```

This is often faster than pulling from Docker Hub on Expanse, especially for large images or slow network connections.

---

## Tips

- Make sure the Dockerfile pulls in every Python dependency and system library EEGDash needs.
- Test the image locally before pushing to Docker Hub.
- In your Slurm scripts, request realistic CPU/GPU, memory, and wall-time limits.
- Track running jobs with `squeue -u $USER` on Expanse.
