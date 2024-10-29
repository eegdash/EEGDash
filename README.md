# EEG-Dash
To leverage recent and ongoing advancements in large-scale computational methods and to ensure the preservation of scientific data generated from publicly funded research, the EEG-DaSh data archive will create a data-sharing resource for MEEG (EEG, MEG) data contributed by collaborators for machine learning (ML) and deep learning (DL) applications. 

## Data source
The data in EEG-DaSh originates from a collaboration involving 25 laboratories, encompassing 27,053 participants. This extensive collection includes MEEG data, which is a combination of EEG and MEG signals. The data is sourced from various studies conducted by these labs, involving both healthy subjects and clinical populations with conditions such as ADHD, depression, schizophrenia, dementia, autism, and psychosis. Additionally, data spans different mental states like sleep, meditation, and cognitive tasks. In addition, EEG-DaSh will also incorporate data converted from NEMAR, which includes a subset of the 330 MEEG BIDS-formatted datasets available on OpenNeuro, further expanding the archive with well-curated, standardized neuroelectromagnetic data.

## Dataset available



## Data formatting
The data in EEG-DaSh is formatted to facilitate machine learning (ML) and deep learning (DL) applications by using a simplified structure commonly adopted by these communities. This will involve converting raw MEEG data into a matrix format, where samples (e.g., individual EEG or MEG recordings) are represented by rows, and values (such as time or channel data) are represented by columns. The data is also divided into training and testing sets, with 80% of the data allocated for training and 20% for testing, ensuring a balanced representation of relevant labels across sets. Hierarchical Event Descriptor (HED) tags will be used to annotate labels, which will be stored in a text table, and detailed metadata, including dataset origins and methods. This formatting process will ensure that data is ready for ML/DL models, allowing for efficient training and testing of algorithms while preserving data integrity and reusability.

![Screenshot 2024-10-03 at 09 07 28](https://github.com/user-attachments/assets/b30a79bb-0d94-410a-843c-44c3fcea01fc)

## Data access
The data in EEG-DaSh is formatted to facilitate machine learning (ML) and deep learning (DL) applications by using a simplified structure commonly adopted by these communities. This will involve converting raw MEEG data into a matrix format, where samples (e.g., individual EEG or MEG recordings) are represented by rows, and values (such as time or channel data) are represented by columns. The data is also divided into training and testing sets, with 80% of the data allocated for training and 20% for testing, ensuring a balanced representation of relevant labels across sets. Hierarchical Event Descriptor (HED) tags will be used to annotate labels, which will be stored in a text table, and detailed metadata, including dataset origins and methods. This formatting process will ensure that data is ready for ML/DL models, allowing for efficient training and testing of algorithms while preserving data integrity and reusability.

The data in EEG-DaSh is accessed through Python and MATLAB libraries specifically designed for this platform. These libraries will use objects compatible with deep learning data storage formats in each language, such as <i>Torchvision.dataset</i> in Python and <i>DataStore</i> in MATLAB. Users can dynamically fetch data from the EEG-DaSh server which is then cached locally. 

### Install
Use your preferred Python environment manager with Python > 3.9 to install the package. Here we show example using Conda environment with Python 3.11.5:
* Create a new environment Python 3.11.5 -> `conda create --name eegdash python=3.11.5`
* Switch to the right environment -> `conda activate eegdash`
* Install dependencies (this is a temporary link that will be updated soon) -> `pip install -r https://raw.githubusercontent.com/sccn/EEG-Dash-Data/refs/heads/develop/requirements.txt`
* Install _eegdash_ package (this is a temporary link that will be updated soon) -> `pip install -i https://test.pypi.org/simple/ eegdash`
* Check installation. Start a Python session and type `from eegdash import EEGDash`

### Python data access

To create a local object for accessing the database, use the following code:

```python
from eegdash import EEGDash
EEGDashInstance = EEGDash()
```

Once the object is instantiated, it can be utilized to search datasets. Providing an empty parameter will search the entire database and return all available datasets.

```python
EEGDashInstance.find({})
```
A list of dataset is returned.

```python
[{'schema_ref': 'eeg_signal',
  'data_name': 'ds004745_sub-001_task-unnamed_eeg.set',
  'dataset': 'ds004745',
  'subject': '001',
  'task': 'unnamed',
  'session': '',
  'run': '',
  'modality': 'EEG',
  'sampling_frequency': 1000,
  'version_timestamp': 0,
  'has_file': True,
  'time_of_save': datetime.datetime(2024, 10, 25, 14, 11, 48, 843593, tzinfo=datetime.timezone.utc),
  'time_of_removal': None}, ...

```



Additionally, users can search for a specific dataset by specifying criteria.

```python
EEGDashInstance.find({'task': 'FaceRecognition'})
```

After locating the desired dataset or data record, users can download it locally by executing the following command:

```python
EEGDashInstance.get({'task': 'FaceRecognition', 'subject': '019'})
```

Optionally, this is how you may access the raw data for the first record.

```
EEGDashInstance.get({'task': 'FaceRecognition', 'subject': '019'})[0].values
```


## Education

We organize workshops and educational events to foster cross-cultural education and student training, offering both online and in-person opportunities in collaboration with US and Israeli partners. There is no event planned for 2024. Events for 2025 will be advertised on the EEGLABNEWS mailing list so make sure to [subscribe](https://sccn.ucsd.edu/mailman/listinfo/eeglabnews).

## About EEG-DaSh

EEG-DaSh is a collaborative initiative between the United States and Israel, supported by the National Science Foundation (NSF). The partnership brings together experts from the Swartz Center for Computational Neuroscience (SCCN) at the University of California San Diego (UCSD) and Ben-Gurion University (BGU) in Israel. 

![Screenshot 2024-10-03 at 09 14 06](https://github.com/user-attachments/assets/327639d3-c3b4-46b1-9335-37803209b0d3)

## Education - Coming soon...

We organize workshops and educational events to foster cross-cultural education and student training, offering both online and in-person opportunities in collaboration with US and Israeli partners. There is no event planned for 2024. Events for 2025 will be advertised on the EEGLABNEWS mailing list so make sure to [subscribe](https://sccn.ucsd.edu/mailman/listinfo/eeglabnews).

## About EEG-DaSh

EEG-DaSh is a collaborative initiative between the United States and Israel, supported by the National Science Foundation (NSF). The partnership brings together experts from the Swartz Center for Computational Neuroscience (SCCN) at the University of California San Diego (UCSD) and Ben-Gurion University (BGU) in Israel. 

![Screenshot 2024-10-03 at 09 14 06](https://github.com/user-attachments/assets/327639d3-c3b4-46b1-9335-37803209b0d3)




