# EEG-Dash
To leverage recent and ongoing advancements in large-scale computational methods and to ensure the preservation of scientific data generated from publicly funded research, the EEG-DaSh data archive will create a data-sharing resource for MEEG (EEG, MEG) data contributed by collaborators for machine learning (ML) and deep learning (DL) applications. 

## Data source
The data in EEG-DaSh originates from a collaboration involving 25 laboratories, encompassing 27,053 participants. This extensive collection includes MEEG data, which is a combination of EEG and MEG signals. The data is sourced from various studies conducted by these labs, involving both healthy subjects and clinical populations with conditions such as ADHD, depression, schizophrenia, dementia, autism, and psychosis. Additionally, data spans different mental states like sleep, meditation, and cognitive tasks. In addition, EEG-DaSh will also incorporate data converted from NEMAR, which includes a subset of the 330 MEEG BIDS-formatted datasets available on OpenNeuro, further expanding the archive with well-curated, standardized neuroelectromagnetic data.

## Data formatting
The data in EEG-DaSh is formatted to facilitate machine learning (ML) and deep learning (DL) applications by using a simplified structure commonly adopted by these communities. This will involve converting raw MEEG data into a matrix format, where samples (e.g., individual EEG or MEG recordings) are represented by rows, and values (such as time or channel data) are represented by columns. The data is also divided into training and testing sets, with 80% of the data allocated for training and 20% for testing, ensuring a balanced representation of relevant labels across sets. Hierarchical Event Descriptor (HED) tags will be used to annotate labels, which will be stored in a text table, and detailed metadata, including dataset origins and methods. This formatting process will ensure that data is ready for ML/DL models, allowing for efficient training and testing of algorithms while preserving data integrity and reusability.

![Screenshot 2024-10-03 at 09 07 28](https://github.com/user-attachments/assets/b30a79bb-0d94-410a-843c-44c3fcea01fc)

## Data access
The data in EEG-DaSh is formatted to facilitate machine learning (ML) and deep learning (DL) applications by using a simplified structure commonly adopted by these communities. This will involve converting raw MEEG data into a matrix format, where samples (e.g., individual EEG or MEG recordings) are represented by rows, and values (such as time or channel data) are represented by columns. The data is also divided into training and testing sets, with 80% of the data allocated for training and 20% for testing, ensuring a balanced representation of relevant labels across sets. Hierarchical Event Descriptor (HED) tags will be used to annotate labels, which will be stored in a text table, and detailed metadata, including dataset origins and methods. This formatting process will ensure that data is ready for ML/DL models, allowing for efficient training and testing of algorithms while preserving data integrity and reusability.

The data in EEG-DaSh is accessed through Python and MATLAB libraries specifically designed for this platform. These libraries will use objects compatible with deep learning data storage formats in each language, such as <i>Torchvision.dataset</i> in Python and <i>DataStore</i> in MATLAB. Users can dynamically fetch data from the EEG-DaSh server which is then cached locally. 

### Python data access

To create a local object for accessing the database, use the following code:

```python
EEGDashInstance = EEGDash()
```

Once the object is instantiated, it can be utilized to search datasets. Providing an empty parameter will search the entire database and return all available datasets.

```python
EEGDashInstance.find()
```

Additionally, users can search for a specific dataset by specifying criteria.

```python
EEGDashInstance.find({'task': 'Simon'})
```

After locating the desired dataset, users can download it locally by executing the following command:

```python
EEGDashInstance.get({'task': 'Simon'})
```

## Education

We organize workshops and educational events to foster cross-cultural education and student training, offering both online and in-person opportunities in collaboration with US and Israeli partners. There is no event planned for 2024. Events for 2025 will be advertised on the EEGLABNEWS mailing list so make sure to [subscribe](https://sccn.ucsd.edu/mailman/listinfo/eeglabnews).

## About EEG-DaSh

EEG-DaSh is a collaborative initiative between the United States and Israel, supported by the National Science Foundation (NSF). The partnership brings together experts from the Swartz Center for Computational Neuroscience (SCCN) at the University of California San Diego (UCSD) and Ben-Gurion University (BGU) in Israel. 

![Screenshot 2024-10-03 at 09 14 06](https://github.com/user-attachments/assets/327639d3-c3b4-46b1-9335-37803209b0d3)

Coming soon...

## Education

We organize workshops and educational events to foster cross-cultural education and student training, offering both online and in-person opportunities in collaboration with US and Israeli partners. There is no event planned for 2024. Events for 2025 will be advertised on the EEGLABNEWS mailing list so make sure to [subscribe](https://sccn.ucsd.edu/mailman/listinfo/eeglabnews).

## About EEG-DaSh

EEG-DaSh is a collaborative initiative between the United States and Israel, supported by the National Science Foundation (NSF). The partnership brings together experts from the Swartz Center for Computational Neuroscience (SCCN) at the University of California San Diego (UCSD) and Ben-Gurion University (BGU) in Israel. 

![Screenshot 2024-10-03 at 09 14 06](https://github.com/user-attachments/assets/327639d3-c3b4-46b1-9335-37803209b0d3)




