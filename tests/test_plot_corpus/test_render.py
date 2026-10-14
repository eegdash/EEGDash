"""The renderer: fragments are offline and scoped, numbers come from the rows, bad input is refused."""

from __future__ import annotations

import json
import re

import pytest
from plot_corpus import charts
from plot_corpus.data import STATUSES, check_statuses, kpis, read_rows
from plot_corpus.electrodes import aggregate_positions, check_coverage
from plot_corpus.fragments import build, build_fragments, is_stale

from .conftest import write_csv


@pytest.fixture
def built(data_dir, tmp_path):
    out = tmp_path / "out"
    return build(data_dir, out), out


def test_fragments_are_offline_and_scoped(built):
    manifest, out = built
    for key, meta in manifest["fragments"].items():
        text = (out / meta["file"]).read_text()
        no_links = re.sub(r'<a href="[^"]*"', "<a", text)
        assert "http" not in no_links, key
        assert (
            "<script src" not in text and "<link" not in text and "@import" not in text
        ), key
        assert "prefers-color-scheme" not in text, key
        assert ' style="fill' not in text and "#2f9e44" not in text, (
            key
        )  # colours come from CSS classes
        for cls in re.findall(r'class="([^"]+)"', text):
            assert all(c.startswith("corpus-") for c in cls.split()), (key, cls)
        assert not [
            i
            for i in re.findall(r'\bid="([^"]+)"', text)
            if not i.startswith("corpus-")
        ], key
        assert meta["caption"] and "<figcaption>" in text, key


def test_kpis_flow_and_numbers_come_from_rows(rows, built):
    manifest, out = built
    k = kpis(rows)
    assert (k["datasets"], k[STATUSES[0]], k["v1_datasets"], k["pending"]) == (
        8,
        2,
        4,
        2,
    )  # v1 counts to ingest
    assert k["v0_hours"] == pytest.approx(150.5)
    flow = (out / "flow.html").read_text()
    assert (
        'data-step="v0" data-count="2"' in flow
        and 'data-step="v1" data-count="4"' in flow
    )
    assert (
        "2 of the release card's 2 entries are converted (150 h)"
        in manifest["fragments"]["flow"]["caption"]
    )
    inc = (out / "numbers.inc").read_text()
    assert (
        ".. |v0_datasets| replace:: 2" in inc and ".. |v1_datasets| replace:: 4" in inc
    )
    assert (
        ".. |v0_hours| replace:: 150" in inc
        and ".. |synced| replace:: 2026-01-02" in inc
    )
    assert manifest["rows"] == {
        "atlas": 8,
        "v0": 2,
        "v1": 4,
        "buckets": 4,
        "added": 1,
        "electrodes": 6,
    }


def test_unknown_status_refused(rows, tmp_path):
    rows[0]["status"] = "Converted"
    with pytest.raises(ValueError, match="not one of"):
        check_statuses(rows)
    write_csv(tmp_path / "a.csv", rows, list(rows[0]))
    with pytest.raises(ValueError, match="not one of"):
        read_rows(tmp_path / "a.csv")


def test_unknown_channel_refused(buckets, bank, electrodes):
    buckets[0]["channels"] += "|XYZ"
    with pytest.raises(ValueError, match="XYZ"):
        aggregate_positions(buckets, bank)
    with pytest.raises(ValueError, match="no electrode position"):
        check_coverage(electrodes, buckets)


def test_electrodes_merge_case_fold_spellings(electrodes):
    cz = next(e for e in electrodes if "Cz" in e["names"])
    assert cz["names"] == ["Cz", "CZ"]  # most common spelling first
    assert (cz["hours"], cz["datasets"], cz["buckets"]) == (150.5, 2, 4)
    assert (
        electrodes[0] is cz and len(electrodes) == 6
    )  # five of v0 and Pz, only in the added channel set
    assert (
        cz["added_hours"],
        cz["added_datasets"],
        electrodes[-1]["names"],
        electrodes[-1]["hours"],
    ) == (12.0, 1, ["Pz"], 0.0)


def test_electrode_views(rows, buckets, electrodes, added):
    frags = build_fragments(rows, buckets, electrodes, "2026-01-02", added)
    two_d = frags["electrodes"][0]
    assert (
        "eegdash-corpus-v0" not in two_d
        and "in 2 of 2 datasets · in 4 of 4 channel sets" in two_d
    )
    labels_2d = set(re.findall(r'class="corpus-lab"[^>]*>([^<]+)</text>', two_d))
    assert (
        "Cz" in labels_2d and two_d.count('Z"/>') >= 1
    )  # the nose is a closed triangle
    assert (
        two_d.count('class="corpus-el ') == 5
        and "channel set (bucket)" in two_d.lower()
    )
    payload = re.search(
        r'<script type="application/json" class="corpus-3d-data">(.*?)</script>',
        frags["electrodes-3d"][0],
    ).group(1)
    pts = json.loads(payload)
    assert len(pts) == 5 and all(len(p["p"]) == 3 and len(p["q"]) == 2 for p in pts)
    assert {
        p["l"] for p in pts if "l" in p
    } == labels_2d  # the 3-D view labels the same landmarks
    three_d = frags["electrodes-3d"][0]
    assert (
        three_d.count('class="corpus-lg"') == 2 and 'class="corpus-note"' in three_d
    )  # legend and note as in 2-D
    # v1: v0 plus the added channel set; Pz appears only there, Cz carries both shares
    v1 = frags["electrodes-v1"][0]
    assert (
        v1.count('class="corpus-el ') == 6
        and "Cz / CZ · 162 h (150 h in v0, 12 h added) · in 3 of 3 datasets (2 in v0, 1 added) · in 5 of 5 channel sets"
        in v1
    )
    assert (
        "6 electrode positions, 1 of them only in the datasets v1 adds"
        in frags["electrodes-v1"][1]
    )


def test_scrolling_bar_lists_pin_their_axis(rows):
    v0 = [r for r in rows if r["status"] == STATUSES[0]]
    body, cap = charts.hours_chart(v0)
    assert (
        'class="corpus-scroll-axis"' in body and body.count('class="corpus-row"') == 2
    )
    assert 'data-k-hours="120.5"' in body and 'data-k-name="ds000001"' in body
    assert "The largest dataset (DS000001) holds 80 %" in cap


def test_table_rows_carry_status_and_sort_values(built):
    _, out = built
    t = (out / "table-all.html").read_text()
    assert t.count("<tr data-status=") == 8
    assert (
        'data-v="120.5"' in t and '<td class="corpus-n" data-v="">' in t
    )  # blanks sort last
    assert (
        '<option value="" selected>All statuses (8)' in t
        and '<option value="not-converted">' in t
    )  # opens on all
    assert "{static}corpus_generated/atlas.csv" in t
    assert "<option" not in (out / "table-v0.html").read_text()
    v1 = (
        out / "table-v1.html"
    ).read_text()  # v1: all, then one option per status present; no "not yet converted"
    assert v1.count("<option") == 4 and "not-converted" not in v1


def test_stale_check(data_dir, built):
    _, out = built
    assert not is_stale(data_dir, out)
    (out / "manifest.json").unlink()
    assert is_stale(data_dir, out)


def test_nothing_left_to_ingest_drops_the_deposit_step_and_the_zero(
    rows, buckets, electrodes
):
    """Once every v1 dataset is on NEMAR (Siena found there, 2026-10-13) the pages must not say "0 still to be
    ingested" nor list a deposit step with nobody to deposit.
    """
    from plot_corpus.data import numbers, roadmap_items

    nums = numbers(rows, buckets, electrodes, "x")
    assert (
        nums["v1_added_prose"].endswith(", 1 still to be ingested")
        and nums["v1_added_short"] == "1 to convert, 1 to ingest"
    )
    assert any("Deposit" in item for item in roadmap_items(rows))
    rest = [r for r in rows if r["status"] != STATUSES[2]]
    nums = numbers(rest, buckets, electrodes, "x")
    assert (
        nums["v1_added_prose"] == "1 ingested by EEGDash and waiting for conversion"
        and nums["v1_added_short"] == "1 to convert"
    )
    assert not any("Deposit" in item for item in roadmap_items(rest))
