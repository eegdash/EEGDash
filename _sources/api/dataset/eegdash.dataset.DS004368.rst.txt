..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS004368
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS004368
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS004368``
- **Summary:** Modality: Visual | Type: Perception | Subjects: Schizophrenia/Psychosis
- **Number of Subjects:** 39
- **Number of Recordings:** 40
- **Number of Tasks:** 1
- **Number of Channels:** 63
- **Sampling Frequencies:** 128
- **Total Duration (hours):** 0.033
- **Dataset Size:** 997.14 MB
- **OpenNeuro:** `ds004368 <https://openneuro.org/datasets/ds004368>`__
- **NeMAR:** `ds004368 <https://nemar.org/dataexplorer/detail?dataset_id=ds004368>`__

=========  =======  =======  ==========  ==========  =============  =========
dataset      #Subj    #Chan    #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  =========
ds004368        39       63           1         128          0.033  997.14 MB
=========  =======  =======  ==========  ==========  =============  =========


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS004368

   dataset = DS004368(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds004368>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds004368>`__

