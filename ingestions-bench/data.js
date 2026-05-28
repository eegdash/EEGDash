window.BENCHMARK_DATA = {
  "lastUpdate": 1780006866215,
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
          "id": "6b418a7a1756f3cf60f2404ae8d2fbfd71e0c222",
          "message": "release: prepare v0.8.0 (#363)\n\nBump __version__ 0.7.2 -> 0.8.0 + CHANGELOG [0.8.0] - 2026-05-28.",
          "timestamp": "2026-05-29T00:07:04+02:00",
          "tree_id": "4681de884e3acb52317c26151898e46216a3b83f",
          "url": "https://github.com/eegdash/EEGDash/commit/6b418a7a1756f3cf60f2404ae8d2fbfd71e0c222"
        },
        "date": 1780006844977,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/perf/test_manifest_walk.py::test_list_git_files_benchmark_synthetic_tree",
            "value": 8.486051034282825,
            "unit": "iter/sec",
            "range": "stddev: 0.00030722027905315117",
            "extra": "mean: 117.84044144444768 msec\nrounds: 9"
          },
          {
            "name": "tests/perf/test_perf.py::test_parse_vhdr_median_under_5ms",
            "value": 8388.26926965401,
            "unit": "iter/sec",
            "range": "stddev: 0.000005748093534911761",
            "extra": "mean: 119.21410339289773 usec\nrounds: 7457"
          },
          {
            "name": "tests/perf/test_perf.py::test_digest_dataset_e2e_under_10s_on_snapshot",
            "value": 5.784203212617353,
            "unit": "iter/sec",
            "range": "stddev: 0.2183368963608742",
            "extra": "mean: 172.8846589999904 msec\nrounds: 2"
          },
          {
            "name": "tests/perf/test_perf.py::test_fingerprint_throughput_1000_files",
            "value": 1588.2245076314014,
            "unit": "iter/sec",
            "range": "stddev: 0.00004492515038778111",
            "extra": "mean: 629.633905782848 usec\nrounds: 1539"
          }
        ]
      }
    ]
  }
}