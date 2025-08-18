.. _overview:

========
Overview
========

eegdash is an interface designed to streamline the access and use of EEG data for machine learning applications. It is composed of three main components that work together to provide a seamless experience for researchers and developers.

The architecture of eegdash can be visualized as follows:

.. code-block:: text

      +-----------------+
      |     MongoDB     |
      |    (Metadata)   |
      +-----------------+
            |
            |
      +-----------v-----------+      +-----------------+
      |       eegdash         |<---->|   S3 Filesystem |
      |     Interface         |      |    (Raw Data)   |
      +-----------------------+      +-----------------+
            |
            |
      +-----------v-----------+
      |      BIDS Parser      |
      +-----------------------+



The components are:

* **MongoDB**: This is a NoSQL database that centralizes all the metadata related to the EEG datasets. It stores information about subjects, sessions, tasks, and other relevant details, allowing for fast and efficient querying.

* **S3 Filesystem**: The raw EEG data is stored in an S3-compatible object storage. This allows for scalable and reliable storage of large datasets. eegdash interacts with the S3 filesystem to download the data when it is needed.

* **BIDS Parser**: The BIDS (Brain Imaging Data Structure) parser is responsible for interpreting the structure of the datasets. It ensures that the data is organized in a standardized way, making it easier to work with and understand.
