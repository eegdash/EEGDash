# EEG-Dash-Data
To leverage recent and ongoing advancements in large-scale computational methods and to ensure the preservation of scientific data generated from publicly funded research, the EEG-DaSh data archive will create a data-sharing resource for MEEG (EEG, MEG) data contributed by collaborators for machine learning (ML) and deep learning (DL) applications. 

## Data source
The data in EEG-DaSh originates from a collaboration involving 25 laboratories, encompassing 27,053 participants. This extensive collection includes MEEG data, which is a combination of EEG and MEG signals. The data is sourced from various studies conducted by these labs, involving both healthy subjects and clinical populations with conditions such as ADHD, depression, schizophrenia, dementia, autism, and psychosis. Additionally, data spans different mental states like sleep, meditation, and cognitive tasks. In addition, EEG-DaSh will also incorporate data converted from NEMAR, which includes a subset of the 330 MEEG BIDS-formatted datasets available on OpenNeuro, further expanding the archive with well-curated, standardized neuroelectromagnetic data.

## Data formatting
The data in EEG-DaSh is formatted to facilitate machine learning (ML) and deep learning (DL) applications by using a simplified structure commonly adopted by these communities. This will involve converting raw MEEG data into a matrix format, where samples (e.g., individual EEG or MEG recordings) are represented by rows, and values (such as time or channel data) are represented by columns. The data is also divided into training and testing sets, with 80% of the data allocated for training and 20% for testing, ensuring a balanced representation of relevant labels across sets. Hierarchical Event Descriptor (HED) tags will be used to annotate labels, which will be stored in a text table, and detailed metadata, including dataset origins and methods. This formatting process will ensure that data is ready for ML/DL models, allowing for efficient training and testing of algorithms while preserving data integrity and reusability.

## Data access
The data in EEG-DaSh is formatted to facilitate machine learning (ML) and deep learning (DL) applications by using a simplified structure commonly adopted by these communities. This will involve converting raw MEEG data into a matrix format, where samples (e.g., individual EEG or MEG recordings) are represented by rows, and values (such as time or channel data) are represented by columns. The data is also divided into training and testing sets, with 80% of the data allocated for training and 20% for testing, ensuring a balanced representation of relevant labels across sets. Hierarchical Event Descriptor (HED) tags will be used to annotate labels, which will be stored in a text table, and detailed metadata, including dataset origins and methods. This formatting process will ensure that data is ready for ML/DL models, allowing for efficient training and testing of algorithms while preserving data integrity and reusability.

## Data content
The data in EEG-DaSh is accessed through Python and MATLAB libraries specifically designed for this platform. These libraries will use objects compatible with deep learning data storage formats in each language, such as ''Torchvision.dataset'' in Python and ''DataStore'' in MATLAB. Users can dynamically fetch data from the EEG-DaSh server which is then cached locally. 

## Education

Finally, we will organize workshops, educational events, and a yearly Hackathon to foster cross-cultural education and student training, offering both online and in-person opportunities in collaboration with US and Israeli partners.

## Data accessing
### AWS S3

### EEG-Dash API


