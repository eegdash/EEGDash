# eegdash.downloader

File downloading utilities for EEG data from cloud storage.

This module provides functions for downloading EEG data files and BIDS dependencies from
AWS S3 storage, with support for caching and progress tracking. It handles the communication
between the EEGDash metadata database and the actual EEG data stored in the cloud.

<!-- !! processed by numpydoc !! -->

### Functions

| `download_s3_file`(s3_path, local_path, \*[, ...])   | Download a single file from S3 to a local path.   |
|------------------------------------------------------|---------------------------------------------------|
| `download_files`(files, \*[, filesystem, ...])       | Download multiple S3 URIs to local destinations.  |
| `get_s3path`(s3_bucket, filepath)                    | Construct an S3 URI from a bucket and file path.  |
| `get_s3_filesystem`()                                | Get an anonymous S3 filesystem object.            |

### eegdash.downloader.download_s3_file(s3_path: str, local_path: Path, , filesystem: S3FileSystem | None = None) → Path

Download a single file from S3 to a local path.

Handles the download of a raw EEG data file from an S3 bucket, caching it
at the specified local path. Creates parent directories if they do not exist.

* **Parameters:**
  * **s3_path** (*str*) – The full S3 URI of the file to download.
  * **local_path** (*pathlib.Path*) – The local file path where the downloaded file will be saved.
  * **filesystem** (*s3fs.S3FileSystem* *|* *None*) – Optional pre-created filesystem to reuse across multiple downloads.
* **Returns:**
  The local path to the downloaded file.
* **Return type:**
  pathlib.Path

<!-- !! processed by numpydoc !! -->

### eegdash.downloader.download_files(files: Sequence[tuple[str, Path]] | Iterable[tuple[str, Path]], , filesystem: S3FileSystem | None = None, skip_existing: bool = True, skip_missing: bool = False) → list[Path]

Download multiple S3 URIs to local destinations.

* **Parameters:**
  * **files** (*iterable* *of*  *(**str* *,* *Path* *)*) – Pairs of (S3 URI, local destination path).
  * **filesystem** (*s3fs.S3FileSystem* *|* *None*) – Optional pre-created filesystem to reuse across multiple downloads.
  * **skip_existing** (*bool*) – If True, do not download files that already exist locally.
  * **skip_missing** (*bool*) – If True, skip files that do not exist on S3 instead of raising.

<!-- !! processed by numpydoc !! -->

### eegdash.downloader.get_s3path(s3_bucket: str, filepath: str) → str

Construct an S3 URI from a bucket and file path.

* **Parameters:**
  * **s3_bucket** (*str*) – The S3 bucket name (e.g., “s3://my-bucket”).
  * **filepath** (*str*) – The path to the file within the bucket.
* **Returns:**
  The full S3 URI (e.g., “s3://my-bucket/path/to/file”).
* **Return type:**
  str

<!-- !! processed by numpydoc !! -->

### eegdash.downloader.get_s3_filesystem() → S3FileSystem

Get an anonymous S3 filesystem object.

Initializes and returns an `s3fs.S3FileSystem` for anonymous access
to public S3 buckets, configured for the ‘us-east-2’ region.

* **Returns:**
  An S3 filesystem object.
* **Return type:**
  s3fs.S3FileSystem

<!-- !! processed by numpydoc !! -->
