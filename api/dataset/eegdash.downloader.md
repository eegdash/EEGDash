# eegdash.downloader module

File downloading utilities for EEG data from cloud storage.

This module provides functions for downloading EEG data files and BIDS dependencies from
AWS S3 storage, with support for caching and progress tracking. It handles the communication
between the EEGDash metadata database and the actual EEG data stored in the cloud.

<!-- !! processed by numpydoc !! -->

### eegdash.downloader.download_files(files: [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]] | [Iterable](https://docs.python.org/3/library/typing.html#typing.Iterable)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]], , filesystem: S3FileSystem | [None](https://docs.python.org/3/library/constants.html#None) = None, skip_existing: [bool](https://docs.python.org/3/library/functions.html#bool) = True, skip_missing: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]

Download multiple S3 URIs to local destinations.

* **Parameters:**
  * **files** (*iterable* *of*  *(*[*str*](https://docs.python.org/3/library/stdtypes.html#str) *,* *Path* *)*) – Pairs of (S3 URI, local destination path).
  * **filesystem** (*s3fs.S3FileSystem* *|* *None*) – Optional pre-created filesystem to reuse across multiple downloads.
  * **skip_existing** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, do not download files that already exist locally.
  * **skip_missing** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, skip files that do not exist on S3 instead of raising.

<!-- !! processed by numpydoc !! -->

### eegdash.downloader.download_s3_file(s3_path: [str](https://docs.python.org/3/library/stdtypes.html#str), local_path: [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path), , filesystem: S3FileSystem | [None](https://docs.python.org/3/library/constants.html#None) = None) → [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path)

Download a single file from S3 to a local path.

Handles the download of a raw EEG data file from an S3 bucket, caching it
at the specified local path. Creates parent directories if they do not exist.

* **Parameters:**
  * **s3_path** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)) – The full S3 URI of the file to download.
  * **local_path** ([*pathlib.Path*](https://docs.python.org/3/library/pathlib.html#pathlib.Path)) – The local file path where the downloaded file will be saved.
  * **filesystem** (*s3fs.S3FileSystem* *|* *None*) – Optional pre-created filesystem to reuse across multiple downloads.
* **Returns:**
  The local path to the downloaded file.
* **Return type:**
  [pathlib.Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path)

<!-- !! processed by numpydoc !! -->

### eegdash.downloader.get_s3_filesystem(, max_concurrency: [int](https://docs.python.org/3/library/functions.html#int) = 20, region: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'us-east-2') → S3FileSystem

Get an anonymous S3 filesystem object.

Initializes and returns an `s3fs.S3FileSystem` for anonymous access
to public S3 buckets.

* **Parameters:**
  * **max_concurrency** ([*int*](https://docs.python.org/3/library/functions.html#int)) – Maximum number of parallel transfer connections (default 20).
  * **region** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)) – AWS region for the S3 endpoint (default `"us-east-2"`).
* **Returns:**
  An S3 filesystem object.
* **Return type:**
  s3fs.S3FileSystem

<!-- !! processed by numpydoc !! -->

### eegdash.downloader.get_s3path(s3_bucket: [str](https://docs.python.org/3/library/stdtypes.html#str), filepath: [str](https://docs.python.org/3/library/stdtypes.html#str)) → [str](https://docs.python.org/3/library/stdtypes.html#str)

Construct an S3 URI from a bucket and file path.

* **Parameters:**
  * **s3_bucket** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)) – The S3 bucket name (e.g., “s3://my-bucket”).
  * **filepath** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)) – The path to the file within the bucket.
* **Returns:**
  The full S3 URI (e.g., “s3://my-bucket/path/to/file”).
* **Return type:**
  [str](https://docs.python.org/3/library/stdtypes.html#str)

<!-- !! processed by numpydoc !! -->
