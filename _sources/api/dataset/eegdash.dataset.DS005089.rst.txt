..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS005089
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS005089
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS005089``
- **Summary:** Modality: Visual | Type: Attention | Subjects: Healthy
- **Number of Subjects:** 36
- **Number of Recordings:** 36
- **Number of Tasks:** 1
- **Number of Channels:** 63
- **Sampling Frequencies:** 1000
- **Total Duration (hours):** 68.82
- **Dataset Size:** 68.01 GB
- **OpenNeuro:** `ds005089 <https://openneuro.org/datasets/ds005089>`__
- **NeMAR:** `ds005089 <https://nemar.org/dataexplorer/detail?dataset_id=ds005089>`__

=========  =======  =======  ==========  ==========  =============  ========
dataset      #Subj    #Chan    #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  ========
ds005089        36       63           1        1000          68.82  68.01 GB
=========  =======  =======  ==========  ==========  =============  ========


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS005089

   dataset = DS005089(cache_dir="./data")

   print(f"Number of recordings: {len(dataset)}")

   if len(dataset):
       recording = dataset[0]
       raw = recording.load()
       print(f"Sampling rate: {raw.info['sfreq']} Hz")
       print(f"Channels: {len(raw.ch_names)}")


See Also
--------

* :class:`eegdash.dataset.EEGDashDataset`
* :mod:`eegdash.dataset`
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds005089>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds005089>`__

