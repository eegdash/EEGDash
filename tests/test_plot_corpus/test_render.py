"""The renderer: fragments are offline and scoped, numbers come from the rows, bad input is refused."""

from __future__ import annotations

import json
import re

import pytest
from plot_corpus import charts
from plot_corpus.data import (
    CONVERTED_V1,
    GATED,
    SLUGS,
    STATUSES,
    TO_CONVERT,
    TO_INGEST,
    check_statuses,
    kpis,
    read_rows,
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
        assert (
            ' style="fill' not in text
            and "#82c991" not in text
            and "#2b8a3e" not in text
        ), key  # colours come from CSS classes
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
    assert (k["datasets"], k[STATUSES[0]], k["v0_datasets"], k["v1_datasets"]) == (
        8,
        2,
        2,
        4,
    )  # v1 counts to ingest
    assert "pending" not in k
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
        "unplaced": 0,
        "recordings": 2,
        "recordings_v1": 1,
        "demographics": 2,
    }


def test_unknown_status_refused(rows, tmp_path):
    rows[0]["status"] = "Converted"
    with pytest.raises(ValueError, match="not one of"):
        check_statuses(rows)
    write_csv(tmp_path / "a.csv", rows, list(rows[0]))
    with pytest.raises(ValueError, match="not one of"):
        read_rows(tmp_path / "a.csv")


def test_earlier_gated_spelling_is_read_as_the_current_one(rows, tmp_path):
    """The data copy synced before 2026-10-13 spells the gated status "Gated (To Be Ingested)": it still builds."""
    rows[5]["status"] = "Gated (To Be Ingested)"
    write_csv(tmp_path / "a.csv", rows, list(rows[0]))
    read = read_rows(tmp_path / "a.csv")
    assert read[5]["status"] == GATED == "Gated (Distribution Forbidden)"


def test_a_name_the_bank_lacks_is_listed_unplaced_not_drawn(
    buckets, bank, added, electrodes
):
    """A channel the converter placed from its dataset's own coordinates (a name the bank lacks, as ds002680's
    ``CB1``) is left off the map and listed with its hours and datasets; a name neither drawn nor listed is refused.
    """
    buckets[0]["channels"] += "|XYZ"
    assert (
        aggregate_positions(buckets, bank, added) == electrodes
    )  # skipped, not refused
    assert unplaced_names(buckets, bank, added) == [
        {
            "name": "XYZ",
            "hours": 100.0,
            "datasets": ["ds000001"],
            "added_hours": 0.0,
            "added_datasets": [],
        }
    ]
    with pytest.raises(ValueError, match="no electrode position"):
        check_coverage(electrodes, buckets)
    check_coverage(electrodes, buckets, unplaced_names(buckets, bank))


def test_unplaced_electrodes_are_counted_in_the_captions(
    rows, buckets, bank, added, electrodes
):
    buckets[0]["channels"] += "|XYZ"
    buckets[2]["channels"] += "|XYZ|QRS"
    unplaced = unplaced_names(buckets, bank, added)
    frags = build_fragments(
        rows, buckets, electrodes, "2026-01-02", added, unplaced=unplaced
    )
    note = "2 electrodes of two datasets were placed from their own coordinates and are not drawn."
    assert frags["electrodes"][1].endswith(note) and frags["electrodes-3d"][1].endswith(
        note
    )
    assert frags["electrodes-v1"][1].endswith(note)
    one = build_fragments(rows, buckets[:1], electrodes, "2026-01-02", added,
                          unplaced=unplaced_names(buckets[:1], bank, added))  # fmt: skip
    assert one["electrodes"][1].endswith(
        "1 electrode of one dataset was placed from its own coordinates and is not drawn."
    )
    assert (
        unplaced_note(None) == unplaced_note([]) == ""
    )  # none placed that way: no sentence


def test_converted_v1_status_has_a_slug_and_a_colour_in_both_themes():
    css = (DOCS / "source/_static/css/corpus.css").read_text()
    assert SLUGS[CONVERTED_V1] == "converted-v1" and STATUSES[1] == CONVERTED_V1
    light, dark = css.split('html[data-theme="dark"] .corpus {')[:2]
    assert (
        "--corpus-st-converted-v1:" in light
        and "--corpus-st-converted-v1:" in dark.split("}")[0]
    )
    for kind in ("fill", "stroke", "bg", "bd"):
        assert f".corpus-{kind}-converted-v1 " in css, kind


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
        and t.count("<option") == 8
        and "not-converted" not in t
    )  # opens on all; then one option per status present, no "not yet converted" shortcut
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
        nums["v1_added_prose"]
        == "1 converted into the v1 store, 1 still to be ingested"
        and nums["v1_added_short"] == "1 converted, 1 to ingest"
    )
    assert any("Deposit" in item for item in roadmap_items(rows))
    rest = [r for r in rows if r["status"] != TO_INGEST]
    nums = numbers(rest, buckets, electrodes, "x")
    assert (
        nums["v1_added_prose"] == "1 converted into the v1 store"
        and nums["v1_added_short"] == "1 converted"
    )
    assert not any("Deposit" in item for item in roadmap_items(rest))


def test_v1_converted_steps_and_nothing_left_to_convert(
    rows, buckets, electrodes, added, to_convert
):
    """Once v1 is converted (2026-10-18) the progress strip, KPIs and roadmap carry it, and no "0 to convert" step
    or conversion pass is listed; a dataset waiting for conversion brings both back.
    """
    from plot_corpus.data import progress, roadmap_items

    rest = [r for r in rows if r["status"] != TO_INGEST]
    steps = {st["status"]: st for st in progress(rest)}
    assert (
        steps[CONVERTED_V1]["count"] == 1
        and TO_CONVERT not in steps
        and TO_INGEST not in steps
    )
    items = roadmap_items(rest)
    assert any("eegdash-corpus-v1 (3 datasets, 162 h)" in i for i in items)
    assert not any("conversion pass" in i for i in items)
    frags = build_fragments(rest, buckets, electrodes, "x", added)
    boxes = re.findall(
        r'<div class="corpus-kpi( corpus-bd-[\w-]+)?"><b>([^<]*)</b><span>([^<]*)</span>',
        frags["kpis"][0],
    )
    assert [(b, t, c.replace(" corpus-bd-", "")) for c, b, t in boxes] == [
        ("2", "datasets converted (v0)", "converted"),
        ("3", "datasets converted (v1)", "converted-v1"),
        ("150", "hours (v0)", "converted"),
        ("162", "hours (v1)", "converted-v1"),
        ("50", "subjects (v0)", "converted"),
        ("55", "subjects (v1)", "converted-v1"),
        ("1", "gated (distribution forbidden)", "gated"),
        ("1", "blocked (No Derivative or no licence)", "blocked"),
        ("7", "datasets tracked (REVE&#x27;s paper and release card)", ""),
    ]  # v0 and v1 side by side, then what will not be converted
    summary = (
        "3 of the 7 datasets tracked are converted (2 in v0, 1 added in v1); 1 is named by the paper only; "
        "the remaining 3 will not be converted: 2 blocked (1 by licence, 1 gated) and 1 excluded."
    )
    assert frags["kpis"][1] == frags["progress"][1] == frags["status"][1] == summary
    assert frags["table-all"][1].startswith(
        "All 7 datasets tracked (REVE's paper and its release card); 3 of them will not be converted. Filter"
    )
    v0_spans = re.findall(r"<span>([^<]*)</span>", frags["v0-kpis"][0])
    assert (
        re.findall(r"<span>([^<]*)</span>", frags["v1-kpis"][0]) == v0_spans
    )  # the same boxes on both pages
    assert ">5</b><span>channel sets (buckets)" in frags["v1-kpis"][0]
    assert (
        ">6</b><span>electrode positions" in frags["v1-kpis"][0]
    )  # Pz only in the added set
    assert ">5</b><span>electrode positions" in frags["v0-kpis"][0]
    assert frags["v1-adds-table"][1] == (
        "The 1 datasets v1 adds: 1 converted into the v1 store; 12.0 h in 1 channel sets, measured in the v1 store."
    )
    for key in (
        "kpis",
        "progress",
        "status",
        "v1-kpis",
        "v1-adds-table",
        "tokens-v1",
        "electrodes-v1",
    ):
        assert not re.search(
            r"to convert|not converted|until converted|EEGDash's records", frags[key][1]
        ), key
    more = rest + [to_convert]
    assert any("conversion pass" in i for i in roadmap_items(more))
    steps = {st["status"]: st for st in progress(more)}
    assert steps[TO_CONVERT]["count"] == 1
    frags = build_fragments(more, buckets, electrodes, "x", added)
    assert (
        "1 more dataset is in this version but in no store yet" in frags["v1-kpis"][1]
    )
    assert "1 more is in v1 but in no store yet; " in frags["status"][1]
    assert "1 more in v1 but in no store yet" in frags["table-all"][1]
