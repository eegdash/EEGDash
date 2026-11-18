:og:description: Pretraining REVE on EEGDash Corpus v0 in three steps: download the converted store, install, and pretrain with REVE's own trainer, using two scripts.

.. meta::
   :description: Pretraining REVE on EEGDash Corpus v0 in three steps: download the converted store, install, and pretrain with REVE's own trainer, using two scripts.

.. include:: /_static/corpus_generated/numbers.inc

.. _corpus-pretraining-reve:

################
Pretraining REVE
################

This page pretrains the REVE foundation model from scratch on EEGDash Corpus v0, in three
steps: download the corpus, install, pretrain. The two scripts it uses, ``download.sh``
and ``pretrain.sh``, come from the code repository `eegdash/eegdash-corpus
<https://github.com/eegdash/eegdash-corpus>`__ and are shown in full below. They run
REVE's own training code with the settings of the paper, whose runs used 16 GPUs on 4
machines; the same commands work on one machine with fewer GPUs, only slower. The paper
(`arXiv:2611.06356 <https://arxiv.org/abs/2611.06356>`__) and the repository's README
hold the details this page leaves out.

**Before you start.** A Linux machine with NVIDIA GPUs and Python 3.11, about 4 TB of
free disk (the store is |v0_size|), and a Hugging Face account that accepted the terms
of the store (:ref:`corpus-download`). Where there is no ``python3.11`` command, any Python
3.11 does in its place in the two ``venv`` lines below, a conda environment's ``python``
for instance.

1. Download the corpus
======================

The code repository first, then the newest Hugging Face client in a small Python
environment of its own, then the download. Everything on this page goes side by side
into the folder you start in, so pick one outside a small home directory:

.. code-block:: bash

   git clone https://github.com/eegdash/eegdash-corpus.git && cd eegdash-corpus
   python3.11 -m venv ../hf-env && source ../hf-env/bin/activate
   pip install -U huggingface_hub hf_xet
   hf auth login                         # paste a token of your Hugging Face account
   bash download.sh /data
   deactivate

This fetches the store at tag |v0_tag| into ``/data/eegdash-corpus-v0`` and prints the
path the last step takes. One bar counts the finished files. The download has its own
environment because the training environment of the next step pins an older Hugging Face
client, which spends one request per file, and a free account gets 1,000 requests per
five minutes: with that client the script still finishes, pausing now and then.

The download runs for hours, so start it where it survives a logout (``tmux``,
``screen`` or a job); run again after an interruption, it goes on where it stopped.
``VERIFY=1 JOBS=16 bash download.sh /data`` also checks every file against the published
SHA-256 list and ends with ``checksums OK``. On an HPC cluster the download took
about 5.5 hours (near 200 MB/s) and the check about 3.5 hours.

``download.sh``:

.. code-block:: bash

   #!/bin/bash
   # Download eegdash-corpus-v0, the converted Zarr store (3.6 TB, 44,801 files), from Hugging Face.
   #
   #   bash download.sh /data                      # -> /data/eegdash-corpus-v0/eegdash-corpus-v0.zarr
   #   VERIFY=1 JOBS=16 bash download.sh /data     # also check every file against the published SHA-256 list
   #
   # Needs a Hugging Face account that accepted the dataset's terms, a token on this machine (`hf auth login`) and the
   # newest Hugging Face client, in an environment of its own (README, Quick start, step 1). The training environment
   # pins an older client that spends a request per file, and a free account gets 1,000 per five minutes: with it the
   # script still finishes, pausing whenever the Hub refuses. After an interruption, run the same line again.
   set -euo pipefail
   OUT=${1:?usage: bash download.sh <a directory with 4 TB free>}
   DEST=$OUT/eegdash-corpus-v0

   hf auth whoami > /dev/null || { echo "not logged in: run 'hf auth login' and accept the terms on the dataset's page" >&2; exit 1; }
   python - "$DEST" << 'EOF'
   import os, sys, time, traceback
   from pathlib import Path
   from huggingface_hub import snapshot_download
   from huggingface_hub.utils import disable_progress_bars

   dest = Path(sys.argv[1])
   listed = dest / "eegdash-corpus-v0-SHA256SUMS"                 # names every file of the store
   for per_file in ("huggingface_hub.xet_get", "huggingface_hub.http_get"):
       disable_progress_bars(per_file)                            # leaves one bar that counts the finished files
   while True:
       try:
           snapshot_download("EEGDash/eegdash-corpus-v0", repo_type="dataset", revision="v1.0", local_dir=dest)
           if listed.exists() and all((dest / line.split()[1]).exists() for line in listed.open()):
               break                                              # all there (a refused request can also return early)
       except Exception as error:
           if "429 Client Error" not in "".join(traceback.format_exception(error)):
               raise                                              # anything but the rate limit: stop
       print("paused by the Hub's rate limit: finished files are kept, going on in a minute", flush=True)
       time.sleep(float(os.environ.get("WAIT", 60)))
   EOF

   if [ "${VERIFY:-0}" = 1 ]; then   # hashes every file, JOBS at a time: hours on the full store
     parts=$(mktemp -d)
     split -l 500 "$DEST/eegdash-corpus-v0-SHA256SUMS" "$parts/part."
     (cd "$DEST" && ls "$parts"/part.* | xargs -P "${JOBS:-8}" -n 1 sha256sum --quiet -c)
     rm -r "$parts"
     echo "checksums OK"
   fi
   echo "DATA=$DEST/eegdash-corpus-v0.zarr"

2. Install
==========

A second environment with the packages training needs, and REVE's training code with the
repository's patches, which let it read the Zarr store:

.. code-block:: bash

   python3.11 -m venv ../reve-env && source ../reve-env/bin/activate
   pip install -r requirements-pretraining.txt
   git clone https://github.com/elouayas/reve_eeg.git ../reve_eeg && git -C ../reve_eeg checkout 06a7059
   bash patches/apply_patches.sh ../reve_eeg

3. Pretrain REVE
================

.. code-block:: bash

   DATA=/data/eegdash-corpus-v0/eegdash-corpus-v0.zarr CKPT=/ckpt/reve-base-v0 bash pretrain.sh

On one machine this uses every GPU it has. The paper's runs used 16 GPUs: 4 machines
with 4 GPUs each. On several machines, run the same line once on each machine, with three
more variables:

.. code-block:: bash

   # on the first of 4 machines; the same line on the others with NODE_RANK=1, 2 and 3
   NNODES=4 NODE_RANK=0 MAIN_HOST=node01 \
       DATA=/data/eegdash-corpus-v0/eegdash-corpus-v0.zarr CKPT=/ckpt/reve-base-v0 bash pretrain.sh

- ``NNODES``: the number of machines, 4 here.
- ``NODE_RANK``: the index of the machine the line runs on, 0 on the first, then 1, 2 and 3.
- ``MAIN_HOST``: the name of the first machine, the one with ``NODE_RANK=0``; the others
  connect to it.

The GPUs of each machine are counted as on one machine, so 4 machines with 4 GPUs give
16 training processes. Every machine needs the same number of GPUs and must see the same
``DATA`` and ``CKPT`` folders; training starts once the line runs on all of them.

``pretrain.sh``:

.. code-block:: bash

   #!/bin/bash
   # Pretrain REVE-Base from scratch on eegdash-corpus-v0 with the paper's settings (17 epochs, peak learning rate
   # 2.4e-4, global batch 300), with REVE's own trainer (reve_eeg) patched to read the Zarr store.
   #
   #   DATA=/data/eegdash-corpus-v0/eegdash-corpus-v0.zarr CKPT=/ckpt/reve-base-v0 bash pretrain.sh
   #
   # One machine: every GPU it has is used. Several machines (the paper: 16 GPUs on 4 machines): run the same line once
   # on each machine, with the number of machines, this machine's index and the first machine's name:
   #
   #   NNODES=4 NODE_RANK=0 MAIN_HOST=node01 DATA=... CKPT=... bash pretrain.sh     # on node01; NODE_RANK=1, 2, 3 on the others
   #
   # REVE's patched code is expected beside this repository (../reve_eeg, as the Quick start clones it); REVE_EEG names
   # another place. Run again with the same CKPT, it continues after the last saved epoch. It writes $CKPT/run/encoder.pth (the
   # pretrained encoder) and $CKPT/run/loss.csv. Every other setting (SEED, EPOCHS, ENCODER=small, WORKERS,
   # MIXED_PRECISION, ...) is train.sh's, which this script calls: its header lists them.
   set -euo pipefail
   HERE=$(cd "$(dirname "$0")" && pwd)
   : "${DATA:?set DATA to the eegdash-corpus-v0.zarr store (download.sh prints it)}"
   : "${CKPT:?set CKPT to a directory for the checkpoints}"
   export DATA CKPT SEED=${SEED:-2} REVE_EEG=${REVE_EEG:-$(dirname "$HERE")/reve_eeg}   # REVE's code: beside this repository unless REVE_EEG says otherwise
   export RUN=${RUN:-reve-base-v0-seed$SEED}

   grep -q "PATCH 8" "$REVE_EEG/src/utils/data_loading.py" 2> /dev/null || {
     echo "no patched reve_eeg at $REVE_EEG: clone it at 06a7059 and run patches/apply_patches.sh (README, Quick start)" >&2; exit 1; }

   GPUS=${GPUS:-$(nvidia-smi -L | wc -l)}                    # the GPUs of this machine
   export NNODES=${NNODES:-1}
   export NPROC=$((NNODES * GPUS))                           # processes over all machines, one per GPU
   if [ "$NNODES" -gt 1 ]; then                              # where the machines find each other
     : "${MAIN_HOST:?several machines: set MAIN_HOST to the name of the first machine}"
     : "${NODE_RANK:?several machines: set NODE_RANK to the index of this machine, 0 on the first}"
     export ACCELERATE_ARGS="--machine_rank $NODE_RANK --main_process_ip $MAIN_HOST --main_process_port 29500"
   fi
   if [ -d "$CKPT/states/$RUN/last" ]; then export RESUME=True; fi
   bash "$HERE/train.sh"

4. What you get
===============

The pretrained encoder, ``$CKPT/run/encoder.pth``, and the loss of every epoch,
``$CKPT/run/loss.csv``. The paper trained REVE-Base twice this way (seeds 2 and 3), 17
epochs each on 16 GPUs across 4 machines: 23.5 hours on A40 GPUs and 29.7 hours on V100
GPUs.

The two encoders, scored on seven public OpenEEGBench tasks, against the released
REVE-Base (chance-adjusted balanced accuracy, 0 = chance; mean of the two runs):

.. list-table::
   :header-rows: 1
   :widths: 30 10 10 10 10 10 10 10 10

   * - **Linear probe**
     - PhysioNet-MI
     - BCIC-IV-2a
     - BCIC-2020-3
     - FACED
     - Mumtaz
     - Zyma
     - CHB-MIT
     - mean
   * - released REVE-Base
     - 0.377
     - 0.191
     - 0.088
     - 0.167
     - 0.681
     - 0.295
     - 0.529
     - **0.333**
   * - pretrained on the corpus
     - 0.373
     - 0.230
     - 0.095
     - 0.159
     - 0.694
     - 0.340
     - 0.423
     - **0.331**

.. list-table::
   :header-rows: 1
   :widths: 30 10 10 10 10 10 10 10 10

   * - **Full fine-tuning**
     - PhysioNet-MI
     - BCIC-IV-2a
     - BCIC-2020-3
     - FACED
     - Mumtaz
     - Zyma
     - CHB-MIT
     - mean
   * - released REVE-Base
     - 0.436
     - 0.352
     - 0.041
     - 0.183
     - 0.675
     - 0.323
     - 0.264
     - **0.325**
   * - pretrained on the corpus
     - 0.379
     - 0.296
     - 0.041
     - 0.160
     - 0.667
     - 0.226
     - 0.144
     - **0.273**

The released model's scores are the OpenEEGBench leaderboard's; CHB-MIT, which the
leaderboard does not score, is the released weights run through the same scoring.

To use the encoder, convert it to braindecode's format with ``scripts/export_encoder.py``
from the code repository, then fine-tune it as in
:ref:`the REVE fine-tuning tutorial <sphx_glr_generated_auto_examples_tutorials_70_transfer_foundation_plot_75_finetune_reve.py>`.

5. Settings of the scripts
==========================

Every setting is an environment variable written before the command, as ``DATA`` and
``CKPT`` are above; ``download.sh`` also takes one argument, the target directory.

5.1 ``download.sh``
-------------------

.. code-block:: bash

   bash download.sh <directory>

.. list-table::
   :header-rows: 1
   :widths: 22 22 56

   * - setting
     - default
     - meaning
   * - ``<directory>``
     - required
     - The one argument: a directory with 4 TB free. The store lands in
       ``<directory>/eegdash-corpus-v0/eegdash-corpus-v0.zarr``.
   * - ``VERIFY``
     - ``0``
     - ``1`` checks every file against the published SHA-256 list after the download
       (hours on the full store) and prints ``checksums OK``.
   * - ``JOBS``
     - ``8``
     - With ``VERIFY=1``, the number of files hashed at a time.
   * - ``WAIT``
     - ``60``
     - Seconds to wait before going on when the Hub's rate limit pauses the download.

5.2 ``pretrain.sh``
-------------------

.. code-block:: bash

   DATA=<store> CKPT=<directory> bash pretrain.sh

.. list-table::
   :header-rows: 1
   :widths: 22 22 56

   * - setting
     - default
     - meaning
   * - ``DATA``
     - required
     - The store, ``…/eegdash-corpus-v0.zarr``; ``download.sh`` prints the path.
   * - ``CKPT``
     - required
     - The directory the run writes to. Run again with the same one, the run continues
       after its last saved epoch.
   * - ``SEED``
     - ``2``
     - Seeds the weights and the batch order. The paper's two runs used 2 and 3.
   * - ``RUN``
     - ``reve-base-v0-seed$SEED``
     - The run's name: it names the saved states under ``$CKPT/states/``.
   * - ``REVE_EEG``
     - ``../reve_eeg``
     - The patched checkout of REVE's code, by default beside this repository.
   * - ``GPUS``
     - all of the machine's
     - The number of GPUs to use on each machine.
   * - ``NNODES``
     - ``1``
     - The number of machines.
   * - ``NODE_RANK``
     - required when ``NNODES`` > 1
     - The index of the machine the line runs on, 0 on the first.
   * - ``MAIN_HOST``
     - required when ``NNODES`` > 1
     - The name of the first machine; the others connect to it.

5.3 ``train.sh``
----------------

.. code-block:: bash

   DATA=<store> RUN=<name> CKPT=<directory> REVE_EEG=<checkout> SEED=<n> bash train.sh

``pretrain.sh`` calls ``train.sh``, the script that launches REVE's trainer, and passes
these settings on; they can be set on the ``pretrain.sh`` line too. Called by itself,
``train.sh`` needs ``DATA``, ``RUN``, ``CKPT``, ``REVE_EEG`` and ``SEED``, and takes
``NNODES`` (default 1) and ``NPROC``, the number of processes over all machines
(default 1).

.. list-table::
   :header-rows: 1
   :widths: 22 22 56

   * - setting
     - default
     - meaning
   * - ``EPOCHS``
     - ``17``
     - The number of epochs; 17 is the paper's.
   * - ``ENCODER``
     - ``base``
     - The model size, one of the encoder configurations of REVE's code: ``base`` (the
       paper's), ``tiny``, ``small`` or ``large``.
   * - ``LR``
     - ``2.4e-4``
     - The peak learning rate.
   * - ``BS``
     - ``300``
     - REVE's batch-size setting: a batch holds 16 × ``BS`` ÷ channels samples on each
       GPU.
   * - ``WORKERS``
     - ``8``
     - Data-loading workers per GPU.
   * - ``MIXED_PRECISION``
     - ``fp16``
     - ``fp16`` runs on every recent NVIDIA GPU; ``bf16`` on those that support it
       (A100, H100).
   * - ``EXCLUDE_DATASETS``
     - none
     - Dataset ids left out of training, separated by commas
       (``EXCLUDE_DATASETS=ds004395,nm000338``).
   * - ``REVE_ENCODER_EVERY``
     - ``5``
     - Saves a snapshot of the encoder, ``encoder_epoch_<k>.pth``, every so many
       epochs; ``0`` turns it off.
   * - ``RESUME``
     - ``False``
     - ``True`` continues from the last saved state in ``$CKPT``. ``pretrain.sh`` sets
       it when that state exists.
   * - ``ACCELERATE_ARGS``
     - none
     - Extra flags for ``accelerate launch``. ``pretrain.sh`` sets them on several
       machines.

A run writes under ``$CKPT``: ``run/encoder.pth`` (the pretrained encoder, at the end),
``run/encoder_epoch_<k>.pth`` (the snapshots), ``run/loss.csv`` (one row per epoch) and
``states/$RUN/`` (a state per epoch, to continue from).
