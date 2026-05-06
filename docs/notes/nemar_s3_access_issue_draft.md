# Draft issue: NEMAR public S3 access doesn't match the documented layout

> Filing target: `nemarDatasets/.github` discussions
> (https://github.com/nemarDatasets/.github/discussions),
> or the NEMAR contact form on https://nemar.org.

## Summary

The NEMAR org README at `github.com/nemarDatasets/.github/profile/README.md`
documents three ways to download a dataset, two of which appear to assume the
S3 bucket exposes BIDS paths anonymously:

```bash
aws s3 ls s3://nemar/<dataset-id>/ --recursive --no-sign-request
aws s3 cp s3://nemar/<dataset-id>/path/to/file.edf . --no-sign-request
```

In practice, anonymous access only works against the git-annex content layer
at `s3://nemar/<dataset-id>/objects/<annex-key>`. Anonymous calls against
the BIDS-path layout return `AccessDenied`, including `ListObjectsV2` and
direct `GetObject` on `dataset_description.json` / `participants.tsv`.

## Reproducible probes

All run against `https://nemar.s3.us-east-2.amazonaws.com/...` (region from
the `git-annex` branch's `remote.log`):

| Path                                                                                     | Status |
|------------------------------------------------------------------------------------------|--------|
| `nm000103/objects/MD5E-s53486328--6c1719b19eae88a46e0aa577b85826f5.set`                  | **200** |
| `nm000103/dataset_description.json`                                                      | 403    |
| `nm000103/participants.tsv`                                                              | 403    |
| `nm000103/sub-01/eeg/sub-01_task-rest_eeg.set`                                           | 403    |
| `aws s3 ls s3://nemar/nm000103/ --no-sign-request --region us-east-2`                    | AccessDenied |

Tested across `nm000103`, `nm000156`, `nm000237`, `nm000339`, `nm000351` —
identical behavior.

## Why this matters

Downstream archives (e.g., EEGDash) build per-record `storage.base` URIs by
following the README. With the current bucket policy, every record we ingest
points at a 403 unless we fall back to git-annex key resolution per file —
which negates the appeal of "Direct S3 Access" advertised in the README.

The org also documents `npm install -g @nemarorg/nemar-cli`, but
`https://registry.npmjs.org/@nemarorg/nemar-cli/latest` returns `404 Not
Found`, so users hitting the README cannot fall back to that path either.

## Possible resolutions (any one works for downstream consumers)

1. **Match the README**: relax the bucket policy to allow anonymous
   `GetObject` on the BIDS-path tree (and `ListObjectsV2` if `aws s3 ls` is
   meant to work). Mirror what `s3://openneuro.org` does today.
2. **Match reality**: update the README to drop the `aws s3 cp` /
   `aws s3 ls` snippets and direct users to DataLad / `git-annex get`,
   which transparently resolve the `objects/<key>` path.
3. **Publish the CLI**: ship `@nemarorg/nemar-cli` to npm (or rename the
   README to whatever the supported install path is now).

We're happy with any of these — option 1 is the cheapest for clients;
option 2 is the cheapest for NEMAR.

## Context

- Affected EEGDash issue: <https://github.com/eegdash/EEGDash/pull/329>
  (we patched 35k records to point at `s3://nemar/<id>/<bids-path>`
  expecting the documented layout to work).
- Probe script can be shared on request.
