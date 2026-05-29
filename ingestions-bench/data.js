window.BENCHMARK_DATA = {
  "lastUpdate": 1780068923579,
  "repoUrl": "https://github.com/eegdash/EEGDash",
  "entries": {
    "Benchmark": [
      {
        "commit": {
          "author": {
            "email": "b.aristimunha@gmail.com",
            "name": "Bru",
            "username": "bruAristimunha"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": false,
          "id": "309d6fe141a1fea123e9771d438df12922a3da27",
          "message": "release: prepare v0.8.1 (#366)",
          "timestamp": "2026-05-29T17:11:45+02:00",
          "tree_id": "cec92b037f0de9528ae1575740456b443045159b",
          "url": "https://github.com/eegdash/EEGDash/commit/309d6fe141a1fea123e9771d438df12922a3da27"
        },
        "date": 1780068901114,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/perf/test_manifest_walk.py::test_list_git_files_benchmark_synthetic_tree",
            "value": 7.097678851674669,
            "unit": "iter/sec",
            "range": "stddev: 0.00039153978073960095",
            "extra": "mean: 140.89113087499783 msec\nrounds: 8"
          },
          {
            "name": "tests/perf/test_perf.py::test_parse_vhdr_median_under_5ms",
            "value": 6871.729967386788,
            "unit": "iter/sec",
            "range": "stddev: 0.000009472426464330355",
            "extra": "mean: 145.52376253810863 usec\nrounds: 6540"
          },
          {
            "name": "tests/perf/test_perf.py::test_digest_dataset_e2e_under_10s_on_snapshot",
            "value": 5.1504861312085595,
            "unit": "iter/sec",
            "range": "stddev: 0.2447059406072252",
            "extra": "mean: 194.1564300000067 msec\nrounds: 2"
          },
          {
            "name": "tests/perf/test_perf.py::test_fingerprint_throughput_1000_files",
            "value": 1394.5878273189344,
            "unit": "iter/sec",
            "range": "stddev: 0.000013748937921688716",
            "extra": "mean: 717.0577430913612 usec\nrounds: 1339"
          }
        ]
      }
    ]
  }
}