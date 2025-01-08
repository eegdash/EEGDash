## Understanding SignalStore
- UnitOfWorkProvider in `unit_of_work_provider.py` creates a DataRepository (in `repositories.py`) that do the high-level io orchestration.
- DataRepository contains `self._records` which is the MongoDB table containing metadata for each recording, and `self._data` which is the FileSystemDAO in the `data_access_objects.py` module which is object managing the S3 endpoint.
- FileSystemDao contains the `get()` method to retrieve the file from initialized filesystem (i.e. S3) given the schema_ref, data_name, data_adapter, etc.
- FileSystemDao needs to be derived to implement a solution for openneuro when the files already exist

## Steps to use Openneuro S3
- Create a FileSystemDao subclass that implements the logic of retrieving files from S3 given needed metadata to identify a recording. In our case, it's probably the whole path to the filename.
- Or, create an OpenneuroDataRepository that knows how to provide the correct path to the Openneuro S3 FileSystemDao. This would probably be the best way to go.