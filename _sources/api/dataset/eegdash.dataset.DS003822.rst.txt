..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS003822
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS003822
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS003822``
- **Summary:** Modality: Visual | Type: Affect | Subjects: Healthy
- **Number of Subjects:** 25
- **Number of Recordings:** 25
- **Number of Tasks:** 1
- **Number of Channels:** 64
- **Sampling Frequencies:** 500
- **Total Duration (hours):** 12.877
- **Dataset Size:** 5.82 GB
- **OpenNeuro:** `ds003822 <https://openneuro.org/datasets/ds003822>`__
- **NeMAR:** `ds003822 <https://nemar.org/dataexplorer/detail?dataset_id=ds003822>`__

=========  =======  =======  ==========  ==========  =============  =======
dataset      #Subj    #Chan    #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  =======
ds003822        25       64           1         500         12.877  5.82 GB
=========  =======  =======  ==========  ==========  =============  =======


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS003822

   dataset = DS003822(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds003822>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds003822>`__

