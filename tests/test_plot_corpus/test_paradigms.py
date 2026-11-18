"""The full paradigm table uses the same synced inputs as the charts and paper."""

from __future__ import annotations

import collections
import html
import re

import pytest
from plot_corpus.data import v0_rows
from plot_corpus.paradigms_table import (
    paradigm_summary,
    paradigms_table_html,
    summary_cells,
)

from .conftest import DOCS
from .test_snapshot import committed


def test_summary_uses_bucket_precision_and_recording_tokens(rows, buckets, recordings):
    v0 = v0_rows(rows)
    v0[0]["hours"] = "9999"  # atlas rounding cannot override measured bucket hours
    summary = paradigm_summary(v0, buckets, recordings)
    assert summary[0]["hours"] == 120.5
    assert summary[0]["tokens"] == 6_000_000_000
    assert summary[-1]["datasets"] == 2
    assert summary[-1]["subjects"] == 50
    assert summary[-1]["hours"] == 150.5
    assert summary[-1]["tokens"] == 8_000_000_000
    assert summary[0]["pct_tokens"] == 75
    changed = [dict(r) for r in recordings]
    changed[0]["tokens"] = "2000000000"
    assert paradigm_summary(v0, buckets, changed)[0]["pct_tokens"] == 50


def test_summary_handles_optional_data_and_empty_corpus(rows, buckets, recordings):
    for recs in ([], recordings[:1]):
        summary = paradigm_summary(v0_rows(rows), buckets, recs)
        assert all(r["pct_tokens"] is None for r in summary)
        body, cap = paradigms_table_html(v0_rows(rows), buckets, recs)
        assert "—" in body and "not available" in cap
    assert paradigm_summary([], [], []) == []
    assert "No paradigm data" in paradigms_table_html([], [], [])[0]
    with pytest.raises(ValueError, match="duplicate dataset"):
        paradigm_summary(v0_rows(rows), buckets, recordings + recordings)


def test_committed_paradigm_totals_and_rendered_cells():
    rows, buckets, _, _, recordings = committed()
    v0 = v0_rows(rows)
    summary = paradigm_summary(v0, buckets, recordings)
    assert len(summary) == 11
    counts = collections.Counter(r["paradigm"] for r in v0)
    for row in summary[:-1]:
        members = [r for r in v0 if r["paradigm"] == row["paradigm"]]
        ids = {r["dataset_id"] for r in members}
        assert row["datasets"] == counts[row["paradigm"]]
        assert row["subjects"] == sum(int(r["n_subjects"]) for r in members)
        assert row["hours"] == pytest.approx(
            sum(float(b["hours"]) for b in buckets if b["dataset"] in ids)
        )
        assert row["tokens"] == sum(
            int(r["tokens"]) for r in recordings if r["dataset_id"] in ids
        )
    assert summary_cells(summary[-1]) == [
        "Total",
        "77",
        "7,272",
        "14,075",
        "100.0",
        "6.21",
        "100.0",
    ]
    assert summary_cells(summary[0]) == [
        "memory & cognition",
        "5",
        "510",
        "9,659",
        "68.6",
        "4.79",
        "77.2",
    ]
    assert sum(r["pct_hours"] for r in summary[:-1]) == pytest.approx(100)
    assert sum(r["pct_tokens"] for r in summary[:-1]) == pytest.approx(100)
    body, _ = paradigms_table_html(v0, buckets, recordings)
    rendered = [
        [
            html.unescape(re.sub("<[^>]+>", "", c))
            for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row)
        ]
        for row in re.findall(r"<tr>(.*?)</tr>", body)[1:]
    ]
    assert rendered == [summary_cells(r) for r in summary]
    assert "<tfoot>" in body and 'scope="row"' in body


def test_docs_use_named_python_api_but_do_not_claim_named_cli_support():
    for page in ("introduction", "v0"):
        text = (DOCS / f"source/corpus/{page}.rst").read_text()
        assert "from eegdash import EEGDashCorpus" in text
        assert 'EEGDashCorpus("eegdash-corpus-v0",' in text
        assert not re.search(
            r"python -m eegdash.corpus \w+ eegdash-corpus-v0(?:\s|$)", text
        )
    text = (DOCS / "source/corpus/v0.rst").read_text()
    assert "without_nc_gpl = " in text and "permissive" not in text
    assert "CC BY-SA retains its share-alike terms" in text
