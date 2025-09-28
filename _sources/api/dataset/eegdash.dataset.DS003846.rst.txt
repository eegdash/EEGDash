..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS003846
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS003846
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS003846``
- **Summary:** Modality: Multisensory | Type: Decision-making | Subjects: Healthy
- **Number of Subjects:** 19
- **Number of Recordings:** 60
- **Number of Tasks:** 1
- **Number of Channels:** 64
- **Sampling Frequencies:** 500
- **Total Duration (hours):** 24.574
- **Dataset Size:** 11.36 GB
- **OpenNeuro:** `ds003846 <https://openneuro.org/datasets/ds003846>`__
- **NeMAR:** `ds003846 <https://nemar.org/dataexplorer/detail?dataset_id=ds003846>`__

=========  =======  =======  ==========  ==========  =============  ========
dataset      #Subj    #Chan    #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  ========
ds003846        19       64           1         500         24.574  11.36 GB
=========  =======  =======  ==========  ==========  =============  ========


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS003846

   dataset = DS003846(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds003846>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds003846>`__

