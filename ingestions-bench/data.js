window.BENCHMARK_DATA = {
  "lastUpdate": 1780143637659,
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
          "distinct": true,
          "id": "a4faf2cfd1b7a66cd13d09ba9a99b7c3fa42955a",
          "message": "release: v0.8.2 (#370)\n\nCo-authored-by: avivdotan <avivd220@gmail.com>",
          "timestamp": "2026-05-30T14:17:55+02:00",
          "tree_id": "c802b4431ecc886efe7527c9b37c2bb2200cdcf8",
          "url": "https://github.com/eegdash/EEGDash/commit/a4faf2cfd1b7a66cd13d09ba9a99b7c3fa42955a"
        },
        "date": 1780143610681,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/perf/test_manifest_walk.py::test_list_git_files_benchmark_synthetic_tree",
            "value": 8.53000338176726,
            "unit": "iter/sec",
            "range": "stddev: 0.0002922110132733422",
            "extra": "mean: 117.23324777777735 msec\nrounds: 9"
          },
          {
            "name": "tests/perf/test_perf.py::test_parse_vhdr_median_under_5ms",
            "value": 8497.769440665368,
            "unit": "iter/sec",
            "range": "stddev: 0.000005583684145026605",
            "extra": "mean: 117.67793972081465 usec\nrounds: 7598"
          },
          {
            "name": "tests/perf/test_perf.py::test_digest_dataset_e2e_under_10s_on_snapshot",
            "value": 5.905846485345578,
            "unit": "iter/sec",
            "range": "stddev: 0.21266347247819462",
            "extra": "mean: 169.32373750000806 msec\nrounds: 2"
          },
          {
            "name": "tests/perf/test_perf.py::test_fingerprint_throughput_1000_files",
            "value": 1613.503234516818,
            "unit": "iter/sec",
            "range": "stddev: 0.000012079158506188603",
            "extra": "mean: 619.7694424203999 usec\nrounds: 1537"
          }
        ]
      }
    ]
  }
}