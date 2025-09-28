..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS001849
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS001849
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS001849``
- **Summary:** Modality: Multisensory | Type: Clinical/Intervention | Subjects: Healthy
- **Number of Subjects:** 20
- **Number of Recordings:** 120
- **Number of Tasks:** 1
- **Number of Channels:** 30
- **Sampling Frequencies:** 5000
- **Total Duration (hours):** 0.0
- **Dataset Size:** 44.51 GB
- **OpenNeuro:** `ds001849 <https://openneuro.org/datasets/ds001849>`__
- **NeMAR:** `ds001849 <https://nemar.org/dataexplorer/detail?dataset_id=ds001849>`__

=========  =======  =======  ==========  ==========  =============  ========
dataset      #Subj    #Chan    #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  ========
ds001849        20       30           1        5000              0  44.51 GB
=========  =======  =======  ==========  ==========  =============  ========


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS001849

   dataset = DS001849(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds001849>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds001849>`__

