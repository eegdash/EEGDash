# Download integrity

For files with a git-annex key, EEGDash uses that key's byte count as the
independent expected size. This applies to NEMAR content-addressed objects,
BIDS-named OpenNeuro twins, and explicitly pinned historical S3 versions.
A URI ending in an annex key governs over a separately supplied key.
HEAD and GET ContentLength describe the stored object; neither proves that
an already-short deposit matches its record. A mismatch is refused, not
accepted as a stale record. Inspect the deposit and re-digest the catalogue
record when the authoritative version has changed.

Transfers validate a unique staging file before atomic replacement, preserving
an existing shared cache on failure. Version pins do not fall back to latest.
`download_all()` scans the cache without network calls, attempts each target
record, and raises one `DataIntegrityError` with `.failures` (record/error pairs)
afterward if any recording failed. Per-record errors include the destination
and pin (when present), rather than a temporary staging filename.
A record whose key size disagrees with the latest nemar-py manifest is refused
before transfer; verified latest bytes are not silently substituted and fetched
again on each load. A missing manifest may still use the exact-key S3 route.
Known-bad remote objects are not negatively cached: a later call may retry them.

## Limits

- Normal dataset loads check byte counts, not content hashes. The low-level
  `download_s3_file(..., verify_hash=True)` option checks supported annex hashes
  on both cached and acquired files and rereads the entire file on every call.
  EEGDashDataset does not expose this option; there is no hash-verification cache.
- Cached directories are not recursively revalidated. Newly downloaded NEMAR
  directory entries use their individual object keys; OpenNeuro directory entries
  and auto-discovered companions/split FIF files without a record key only have
  transport-size checks, not record-key verification.
- Records without a usable key retain transport-size/existence behavior.
- Keyed channels.tsv, participants.tsv and VHDR files are not rewritten by the
  remaining read-repair helpers; malformed deposited metadata may need repair
  upstream rather than mutation of the shared cache.
- S3Client provides EEGDash's small synchronous S3 interface, not a drop-in s3fs
  replacement when its annex_key/verify_hash extensions are used.
