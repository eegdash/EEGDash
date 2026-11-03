"""The renderer: fragments are offline and scoped, numbers come from the rows, bad input is refused."""

from __future__ import annotations

import json
import re

import pytest
from plot_corpus import charts
from plot_corpus.data import (
    BLOCKED,
    CONVERTED,
    EXCLUDED,
    GATED,
    SLUGS,
    STATUSES,
    check_statuses,
    kpis,
    numbers,
    read_rows,
    reason,
)
from plot_corpus.electrodes import (
    aggregate_positions,
    check_coverage,
    unplaced_names,
    unplaced_note,
)
from plot_corpus.fragments import build, build_fragments, is_stale

from .conftest import DOCS, write_csv


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
        assert ' style="fill' not in text and "#82c991" not in text, (
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
    assert (k["datasets"], k[CONVERTED], k["v0_datasets"], k["out"]) == (6, 2, 2, 3)
    assert k["v0_hours"] == pytest.approx(150.5)
    flow = (out / "flow.html").read_text()
    assert (
        'data-step="v0" data-count="2"' in flow
        and 'data-step="tracked" data-count="6"' in flow
    )
    assert manifest["fragments"]["flow"]["caption"].startswith(
        "2 of the 6 datasets tracked are converted (150 h): 2 of the release card's 2 entries"
    )
    inc = (out / "numbers.inc").read_text()
    assert (
        ".. |v0_datasets| replace:: 2" in inc and ".. |v0_hours| replace:: 150" in inc
    )
    assert (
        ".. |synced| replace:: 2026-01-02" in inc
        and ".. |v0_tokens| replace:: 8.00 B" in inc
    )
    assert manifest["rows"] == {"atlas": 6, "v0": 2, "buckets": 4, "electrodes": 5, "unplaced": 0,
                                "recordings": 2, "demographics": 2, "table": 2}  # fmt: skip
    assert sorted(p.name for p in out.glob("*.csv")) == ["atlas-v0.csv", "atlas.csv", "eegdash-corpus-v0.csv"]


def test_corpus_table_is_copied_as_synced_and_must_list_v0(data_dir, built, tmp_path):
    """The downloadable corpus table is the synced file itself, and a table that is not v0's datasets stops the build."""
    _, out = built
    assert (out / "eegdash-corpus-v0.csv").read_bytes() == (data_dir / "eegdash-corpus-v0.csv").read_bytes()
    table = data_dir / "eegdash-corpus-v0.csv"
    table.write_text(table.read_text().replace("ds000002", "ds000009"))
    with pytest.raises(ValueError, match=r"only in the table \['ds000009'\]"):
        build(data_dir, tmp_path / "again")
    table.write_text(table.read_text().replace("ds000009", "ds000001"))
    with pytest.raises(ValueError, match="its own dataset_id"):
        build(data_dir, tmp_path / "again")


def test_headline_numbers_name_the_reasons(rows, buckets, electrodes):
    """12 cannot be redistributed in the corpus, split by why; the excluded are named with their reason."""
    nums = numbers(rows, buckets, electrodes, "x")
    assert (nums["not_redistributable"], nums["blocked"], nums["blocked_nd"], nums["blocked_none"]) == (
        "3", "2", "1", "1")  # fmt: skip
    assert (nums["gated"], nums["gated_names"], nums["excluded"]) == (
        "1",
        "X-GATED",
        "1",
    )
    assert (
        nums["excluded_prose"]
        == "Slow2016: recorded at 100 Hz, below REVE's 200 Hz input"
    )
    assert nums["v0_licences"] == "CC-BY-4.0 (1 dataset) and CC0-1.0 (1: DS000002)"
    assert [reason(r) for r in rows] == [
        "", "", "No Derivatives licence (CC-BY-NC-ND-4.0)", "no licence stated",
        "gated (registration): redistribution forbidden", "recorded at 100 Hz, below REVE's 200 Hz input",
    ]  # fmt: skip


def test_four_statuses_only(rows, tmp_path):
    """The status vocabulary is the four the data carries; a status of an unreleased store is refused."""
    assert STATUSES == [
        "Ingested and Converted (v0)",
        "Blocked (ND Licence)",
        "Gated (Distribution Forbidden)",
        "Excluded",
    ]
    for old in (
        "Converted",
        "Ingested and Converted (v1)",
        "To Be Ingested (v1)",
        "Paper Only",
    ):
        rows[0]["status"] = old
        with pytest.raises(ValueError, match="not one of"):
            check_statuses(rows)
    write_csv(tmp_path / "a.csv", rows, list(rows[0]))
    with pytest.raises(ValueError, match="not one of"):
        read_rows(tmp_path / "a.csv")


def test_earlier_gated_spelling_is_read_as_the_current_one(rows, tmp_path):
    """A data copy synced before the rename spells the gated status "Gated (To Be Ingested)": it still builds."""
    rows[4]["status"] = "Gated (To Be Ingested)"
    write_csv(tmp_path / "a.csv", rows, list(rows[0]))
    read = read_rows(tmp_path / "a.csv")
    assert read[4]["status"] == GATED == "Gated (Distribution Forbidden)"


def test_status_colours_in_both_themes():
    """Converted green, blocked red, gated purple, excluded grey, in the light and the dark theme; no other status."""
    css = (DOCS / "source/_static/css/corpus.css").read_text()
    assert list(SLUGS.values()) == ["converted", "blocked", "gated", "excluded"]
    light, dark = css.split('html[data-theme="dark"] .corpus {')[:2]
    dark = dark.split("}")[0]
    for slug in SLUGS.values():
        assert f"--corpus-st-{slug}:" in light and f"--corpus-st-{slug}:" in dark, slug
        for kind in ("fill", "stroke", "bg", "bd"):
            assert f".corpus-{kind}-{slug} " in css, (kind, slug)
    assert re.findall(r"--corpus-st-([\w-]+):", light) == list(SLUGS.values())
    for gone in ("converted-v1", "to-convert", "to-ingest", "paper-only"):
        assert gone not in css, gone


def test_a_name_the_bank_lacks_is_listed_unplaced_not_drawn(buckets, bank, electrodes):
    """A channel the converter placed from its dataset's own coordinates (a name the bank lacks, as ds002680's
    ``CB1``) is left off the map and listed with its hours and datasets; a name neither drawn nor listed is refused.
    """
    buckets[0]["channels"] += "|XYZ"
    assert aggregate_positions(buckets, bank) == electrodes  # skipped, not refused
    assert unplaced_names(buckets, bank) == [
        {"name": "XYZ", "hours": 100.0, "datasets": ["ds000001"]}
    ]
    with pytest.raises(ValueError, match="no electrode position"):
        check_coverage(electrodes, buckets)
    check_coverage(electrodes, buckets, unplaced_names(buckets, bank))


def test_unplaced_electrodes_are_counted_in_the_captions(
    rows, buckets, bank, electrodes
):
    buckets[0]["channels"] += "|XYZ"
    buckets[2]["channels"] += "|XYZ|QRS"
    frags = build_fragments(
        rows, buckets, electrodes, "2026-01-02", unplaced=unplaced_names(buckets, bank)
    )
    note = "2 electrodes of two datasets were placed from their own coordinates and are not drawn."
    assert frags["electrodes"][1].endswith(note) and frags["electrodes-3d"][1].endswith(
        note
    )
    one = build_fragments(rows, buckets[:1], electrodes, "2026-01-02",
                          unplaced=unplaced_names(buckets[:1], bank))  # fmt: skip
    assert one["electrodes"][1].endswith(
        "1 electrode of one dataset was placed from its own coordinates and is not drawn."
    )
    assert (
        unplaced_note(None) == unplaced_note([]) == ""
    )  # none placed that way: no sentence


def test_electrodes_merge_case_fold_spellings(electrodes):
    cz = next(e for e in electrodes if "Cz" in e["names"])
    assert cz["names"] == ["Cz", "CZ"]  # most common spelling first
    assert (cz["hours"], cz["datasets"], cz["buckets"]) == (150.5, 2, 4)
    assert electrodes[0] is cz and len(electrodes) == 5
    assert set(cz) == {"names", "xyz", "hours", "datasets", "buckets"}


def test_electrode_views(rows, buckets, electrodes):
    frags = build_fragments(rows, buckets, electrodes, "2026-01-02")
    two_d = frags["electrodes"][0]
    assert "in 2 of 2 datasets · in 4 of 4 channel sets" in two_d
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
    assert three_d.count('class="corpus-lg"') == 2 and 'class="corpus-note"' in three_d
    assert frags["electrodes"][1].startswith(
        "5 electrode positions; Cz carries the most data (150 h"
    )


def test_scrolling_bar_lists_pin_their_axis(rows):
    v0 = [r for r in rows if r["status"] == CONVERTED]
    body, cap = charts.hours_chart(v0)
    assert (
        'class="corpus-scroll-axis"' in body and body.count('class="corpus-row"') == 2
    )
    assert 'data-k-hours="120.5"' in body and 'data-k-name="ds000001"' in body
    assert "The largest dataset (DS000001) holds 80 %" in cap


def test_table_rows_carry_status_reason_and_sort_values(built):
    _, out = built
    t = (out / "table-all.html").read_text()
    assert t.count("<tr data-status=") == 6
    assert (
        'data-v="120.5"' in t and '<td class="corpus-n" data-v="">' in t
    )  # blanks sort last
    assert (
        '<option value="" selected>All statuses (6)' in t and t.count("<option") == 5
    )  # all, then the four
    assert (
        "{static}corpus_generated/atlas.csv" in t
        and '<th scope="col" data-num="0" tabindex="0">Reason' in t
    )
    reasons = re.findall(r'<td class="corpus-reason" data-v="([^"]*)">', t)
    assert len(reasons) == 6 and reasons[:2] == ["", ""] and all(reasons[2:])
    assert "v1" not in t
    v0 = (out / "table-v0.html").read_text()
    assert (
        "<option" not in v0
        and "corpus-reason" not in v0
        and v0.count("<tr data-status=") == 2
    )


def test_stale_check(data_dir, built):
    _, out = built
    assert not is_stale(data_dir, out)
    (out / "manifest.json").unlink()
    assert is_stale(data_dir, out)


def test_a_removed_view_is_not_left_behind(data_dir, built):
    """A fragment or CSV of an earlier build that no page uses any more is deleted by the next build."""
    _, out = built
    (out / "old-view.html").write_text("x")
    (out / "atlas-old.csv").write_text("x")
    build(data_dir, out)
    assert not (out / "old-view.html").exists() and not (out / "atlas-old.csv").exists()


def test_status_figure_kpis_and_summary(rows, buckets, electrodes):
    """The status bar and its four boxes, the universe's KPI boxes and the one-sentence summary."""
    frags = build_fragments(rows, buckets, electrodes, "x")
    summary = ("2 of the 6 datasets tracked are converted and released in v0; 3 cannot be redistributed "
               "(2 blocked by the licence, 1 gated by an access agreement) and 1 is excluded.")  # fmt: skip
    assert frags["kpis"][1] == frags["status"][1] == summary
    status = frags["status"][0]
    assert re.findall(r'<g data-status="([^"]+)" data-count="(\d+)"', status) == [
        (CONVERTED, "2"), (BLOCKED, "2"), (GATED, "1"), (EXCLUDED, "1")]  # fmt: skip
    boxes = re.findall(r'<li class="corpus-bd-([\w-]+)"[^>]*><b>(\d+)</b><span>([^<]*)</span><small>([^<]*)</small>',
                       status)  # fmt: skip
    assert boxes == [
        ("converted", "2", "converted and released (v0)", "150 h, 50 subjects"),
        (
            "blocked",
            "2",
            "blocked by the licence",
            "1 No Derivatives and 1 with no licence",
        ),
        ("gated", "1", "gated by an access agreement", "X-GATED"),
        ("excluded", "1", "excluded", "for other reasons (the table says why)"),
    ]
    kpi = re.findall(r'<div class="corpus-kpi( corpus-bd-[\w-]+)?"><b>([^<]*)</b><span>([^<]*)</span>',
                     frags["kpis"][0])  # fmt: skip
    assert [(b, t, c.replace(" corpus-bd-", "")) for c, b, t in kpi] == [
        ("2", "datasets converted and released (v0)", "converted"),
        ("150", "hours (v0)", "converted"),
        ("50", "subjects (v0)", "converted"),
        ("2", "blocked (No Derivatives or no licence)", "blocked"),
        ("1", "gated (distribution forbidden)", "gated"),
        ("1", "excluded", "excluded"),
        ("6", "datasets tracked (REVE&#x27;s paper and release card)", ""),
    ]
    assert frags["table-all"][1].startswith(
        "All 6 datasets tracked (REVE's paper and its release card); the 4 not in the corpus say why"
    )
    legend = re.search(
        r'<div class="corpus-legend">(.*?)</div>', frags["status"][0]
    ).group(1)
    assert re.findall(r"</i>([^<]+)</span>", legend) == [
        "converted (v0)", "blocked (ND licence)", "gated (distribution forbidden)", "excluded"]  # fmt: skip
