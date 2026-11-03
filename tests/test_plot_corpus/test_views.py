"""The universe (every row counts) and the v0 views: paradigms, recordings, people, licence flow, budget and montages;
numbers from the tables, missing data handled, the treemap tiles its frame, the Sankey's flows balance.
"""

from __future__ import annotations

import re

import pytest
from plot_corpus import charts
from plot_corpus.data import (
    CONVERTED,
    CSV_COLUMNS,
    EXCLUDED,
    kpis,
    label_of,
    numbers,
    read_rows,
    v0_rows,
)
from plot_corpus.fragments import build, build_fragments
from plot_corpus.layout import sankey_flows, squarify

from .conftest import make_row, write_csv

NEW_KEYS = ["lengths", "per-subject", "people", "flow-licence", "tokens", "montages"]


def frags_of(rows, buckets, electrodes, recordings=None, demographics=None):
    return build_fragments(
        rows, buckets, electrodes, "2026-01-02", recordings, demographics
    )


# -------------------------------------------------------------------------------------------------- universe


@pytest.fixture
def card_only():
    """A release-card entry the paper does not name that names no dataset (excluded)."""
    return make_row("ds009999", EXCLUDED, in_reve_paper="no", in_release_card="yes", hours="1000", n_subjects="500",
                    archive="OpenNeuro", licence="CC-BY-4.0", short_name="", name="ds009999",
                    reason="a release-card entry that names no dataset")  # fmt: skip


@pytest.fixture
def card_converted():
    """A release-card entry the paper does not name that was converted with the rest (ds004262, in v0)."""
    return make_row("ds004262", CONVERTED, in_reve_paper="no", in_release_card="yes", hours="8", n_subjects="3",
                    short_name="Feedback")  # fmt: skip


def test_card_only_entries_count_everywhere(
    rows, card_converted, card_only, buckets, electrodes, recordings
):
    """The universe is the union of the paper and the card: a card-only row counts in every number, chart and table,
    with no badge.
    """
    every = rows + [card_converted, card_only]
    frags = frags_of(every, buckets, electrodes, recordings)
    k = kpis(every)
    assert (k["datasets"], k[CONVERTED], k["v0_datasets"], k[EXCLUDED]) == (8, 3, 3, 2)
    assert (k["paper"], k["card"], k["card_only"]) == (6, 4, 2)
    assert (
        "<b>8</b><span>datasets tracked (REVE&#x27;s paper and release card)</span>"
        in frags["kpis"][0]
    )
    assert frags["status"][1].startswith(
        "3 of the 8 datasets tracked are converted and released in v0"
    )
    assert (
        'data-count="3"' in frags["status"][0]
        and "Feedback (ds004262)" in frags["hours"][0]
    )
    assert "entries (2 not in the paper)" in frags["flow"][0]
    table = frags["table-all"][0]
    assert (
        table.count("<tr data-status=") == 8
        and "ds009999" in table
        and "card only" not in table
    )
    assert "a release-card entry that names no dataset" in table
    nums = numbers(every, buckets, electrodes, "x")
    assert (
        nums["universe"],
        nums["paper_named"],
        nums["card_entries"],
        nums["card_only"],
    ) == ("8", "6", "4", "2")
    assert nums["card_only_names"] == "ds004262 and ds009999"
    assert nums["excluded_prose"] == ("Slow2016: recorded at 100 Hz, below REVE's 200 Hz input; "
                                      "ds009999: a release-card entry that names no dataset")  # fmt: skip


def test_tracked_column_is_ignored(rows, buckets, electrodes, tmp_path):
    """A ``tracked`` column left by an earlier sync, whatever it says, changes nothing; a copy without it reads."""
    rows[1].update(in_reve_paper="no", in_release_card="yes")
    write_csv(tmp_path / "plain.csv", rows, CSV_COLUMNS)
    marked = [dict(r, tracked="no" if j == 1 else "maybe") for j, r in enumerate(rows)]
    write_csv(tmp_path / "marked.csv", marked, [*CSV_COLUMNS, "tracked"])
    plain, old = read_rows(tmp_path / "plain.csv"), read_rows(tmp_path / "marked.csv")
    assert old == plain and all("tracked" not in r for r in old)
    assert frags_of(old, buckets, electrodes) == frags_of(plain, buckets, electrodes)
    assert kpis(old)["datasets"] == len(rows)


def test_axis_labels_name_the_id_and_cut_the_name_only():
    long = make_row(
        "icare", CONVERTED, short_name="International Cardiac Arrest REsearch"
    )
    assert label_of(long) == "International Cardiac… (icare)"
    assert (
        label_of(make_row("ds005597", CONVERTED, short_name="", name="ds005597"))
        == "ds005597"
    )
    assert (
        label_of({"short_name": "BCI (MI & P300)"}) == "BCI (MI & P300)"
    )  # a paradigm: no id


def test_every_row_ships_in_the_full_csv(data_dir, rows, card_only, tmp_path):
    write_csv(data_dir / "atlas.csv", rows + [card_only], CSV_COLUMNS)
    m = build(data_dir, tmp_path / "out")
    assert m["rows"]["atlas"] == 7 and "tracked" not in m["rows"]
    full = (tmp_path / "out/atlas.csv").read_text()
    assert (
        "ds009999" in full
        and "tracked" not in full.splitlines()[0]
        and "in_v1" not in full.splitlines()[0]
    )
    assert "ds009999" not in (tmp_path / "out/atlas-v0.csv").read_text()
    assert ".. |universe| replace:: 7" in (tmp_path / "out/numbers.inc").read_text()


# ------------------------------------------------------------------------------------------------- paradigms


def test_paradigms_come_from_the_data(rows):
    """Every paradigm the rows carry is a bar, most hours first, whatever its name: no fixed list to go stale."""
    more = rows + [
        make_row(
            "ds000003", CONVERTED, hours="2", paradigm="a paradigm never seen before"
        )
    ]
    body, cap = charts.paradigms_chart(v0_rows(more))
    labels = re.findall(
        r'<text x="[\d.]+" y="[\d.]+" text-anchor="end">([^<]+)</text>', body
    )
    assert labels == [
        "BCI (MI &amp; P300)",
        "resting &amp; consciousness",
        "a paradigm never seen before",
    ]
    assert cap.startswith(
        "3 paradigms; BCI (MI & P300) holds 79 % of the hours with 1 dataset"
    )


# ------------------------------------------------------------------------------------------------- recordings


def test_lengths_rows_and_caption(rows, recordings):
    body, cap = charts.lengths_chart(v0_rows(rows), recordings)
    assert (
        body.count('class="corpus-row"') == 2 and 'class="corpus-scroll-axis"' in body
    )
    labels = re.findall(r'text-anchor="end">([^<]+)</text>', body)
    assert labels == [
        "DS000001 (ds000001)",
        "DS000002 (ds000002)",
    ]  # longest median first, name (id)
    assert cap.startswith(
        "The median recording lasts 1.5 h over 90 recordings of 2 datasets"
    )
    assert "from 12 min (DS000002) to 1.5 h (DS000001)" in cap and "up to 10.0 h" in cap


def test_per_subject_caption(rows, recordings):
    body, cap = charts.per_subject_chart(v0_rows(rows), recordings)
    assert (
        body.count('class="corpus-row"') == 2 and body.count("<circle") == 4
    )  # two panels
    assert "recordings per subject" in body and "hours per subject" in body
    assert cap.startswith(
        "The median dataset records 2 recordings and 3.0 h per subject"
    )
    assert "DS000001 has the most recordings for one subject (4)" in cap


def test_missing_tables_keep_the_page_building(rows, buckets, electrodes):
    frags = frags_of(rows, buckets, electrodes)
    for key in ["lengths", "per-subject", "people", "tokens"]:
        text, cap = frags[key]
        assert "corpus-note" in text and cap.startswith("No "), key


# ---------------------------------------------------------------------------------------------------- people


def test_people_coverage_and_rows_without_ages(rows, demographics):
    body, cap = charts.people_chart(v0_rows(rows), demographics)
    assert cap.startswith(
        "Ages known for 1 of 2 datasets, sex for 2; median age 25 (DS000001)"
    )
    assert "42 % of the 50 subjects with a recorded sex are female" in cap
    assert (
        body.count('class="corpus-row"') == 2
    )  # ds000002 has sex only: a row with no age marks
    no_age = re.search(
        r'data-tip="DS000002 \(ds000002\) · ages unknown[^"]*">(.*?)</g>', body
    ).group(1)
    assert 'fill="currentColor"' not in no_age and "6 F · 4 M" in no_age


# ---------------------------------------------------------------------------------------------------- budget


def test_squarify_tiles_the_frame():
    vals = [50, 30, 10, 5, 3, 1, 1]
    rects = squarify(vals, 0, 0, 960, 480)
    assert sum(w * h for _, _, w, h in rects) == pytest.approx(960 * 480)
    for v, (_, _, w, h) in zip(vals, rects):
        assert w * h == pytest.approx(v / sum(vals) * 960 * 480)
    assert all(
        0 <= x and x + w <= 960 + 1e-6 and 0 <= y and y + h <= 480 + 1e-6
        for x, y, w, h in rects
    )


def test_tokens_treemap_sums_to_the_frame_and_lists_every_paradigm(rows, recordings):
    body, cap = charts.tokens_chart(v0_rows(rows), recordings)
    leaves = re.findall(
        r'<rect x="([\d.]+)" y="([\d.]+)" width="([\d.]+)" height="([\d.]+)" class="corpus-q(\d)" data-tokens',
        body,
    )
    assert len(leaves) == 2 and [s for *_, s in leaves] == [
        "7",
        "7",
    ]  # 75 % and 25 %: both in the darkest shade
    assert sum(float(w) * float(h) for _, _, w, h, _ in leaves) == pytest.approx(
        960 * 480, rel=1e-3
    )
    assert (
        body.count('height="12" class="corpus-q') == 8 and "≥ 10 %" in body
    )  # the ramp legend under the chart
    groups = re.findall(r'data-group="([^"]+)" data-tokens="(\d+)"', body)
    assert sorted(groups) == [
        ("BCI (MI &amp; P300)", "6000000000"),
        ("resting &amp; consciousness", "2000000000"),
    ]
    key = re.search(r'<ul class="corpus-key"[^>]*>(.*?)</ul>', body).group(1)
    assert re.findall(r"<b>([^<]+)</b> (\d+ %)", key) == [("BCI (MI &amp; P300)", "75 %"),
                                                          ("resting &amp; consciousness", "25 %")]  # fmt: skip
    assert cap.startswith(
        "8.00 B tokens per epoch over 2 datasets: DS000001 alone holds 75 %"
    )
    assert (
        "colour is each dataset's share" in cap
        and "the 2 paradigms listed below" in cap
    )


# ------------------------------------------------------------------------------------------------ licence flow


def test_sankey_flows_balance(rows):
    data = [
        dict(r, archive=a)
        for r, a in zip(rows, ["OpenNeuro", "NEMAR", "NEMAR", "", "OpenNeuro", "NEMAR"])
    ]
    for measure in ("hours", "datasets"):
        text, nodes, links = charts.sankey_svg(data, measure)
        assert len(nodes) == 2  # archive -> licence
        node_v = {
            m.group(1): float(m.group(2))
            for m in re.finditer(r'data-node="([^"]+)" data-v="([^"]+)"', text)
        }
        flows = [(f, t, float(v)) for f, t, v in re.findall(r'data-from="([^"]+)" data-to="([^"]+)" data-v="([^"]+)"',
                                                           text)]  # fmt: skip
        for n, v in node_v.items():
            col = int(n.split(":")[0])
            if col == 0:
                assert sum(x for f, _, x in flows if f == n) == pytest.approx(v), (
                    measure,
                    n,
                )
            else:
                assert sum(x for _, t, x in flows if t == n) == pytest.approx(v), (
                    measure,
                    n,
                )
    _, links = sankey_flows(
        data, [lambda r: r["archive"], lambda r: r["status"]], lambda r: 1
    )
    assert sum(links.values()) == len(data)


def test_flow_licence_toggle_and_caption(rows):
    v0 = v0_rows(rows) + [make_row("ds000003", CONVERTED)]  # a dataset without hours
    body, cap = charts.flow_licence_chart(v0)
    assert 'data-corpus-group="flow" data-corpus-state="hours"' in body
    assert body.count("<svg") == 2 and 'data-corpus-show="datasets"' in body
    assert cap.startswith("150 h over 2 of the 3 datasets")
    assert (
        "1 dataset without hours shows in the datasets view only" in cap
        and "status" not in body
    )


# ---------------------------------------------------------------------------------------------------- montages


def test_montage_cells_and_caption(rows, buckets, electrodes):
    """Electrodes are the rows (front to back), datasets the columns (most electrodes first), each with its k of n."""
    body, cap = charts.montage_chart(v0_rows(rows), buckets, electrodes, min_datasets=2)
    assert cap.startswith("Columns are the 2 datasets, most electrodes first, in channel-count bands; rows are the 2 "
                          "10-20 / 10-10 names that at least 2 datasets use, front to back.")  # fmt: skip
    assert (
        "Fz is the most shared (2 datasets)" in cap
    )  # a tie goes to the first row, front to back
    assert (
        "the median dataset carries 2 of the 2, and 0 datasets use their own naming"
        in cap
        and "k of n" in cap
    )
    assert re.findall(r'data-dataset="([^"]+)" data-on="(\d+)" data-n="(\d+)"', body) == [
        ("ds000001", "2", "4"), ("ds000002", "2", "3")]  # fmt: skip
    pin = re.search(r'<div class="corpus-hscroll-pin">(.*?)</div>', body).group(1)
    assert re.findall(r">(\w+)</text>", pin) == [
        "Fz",
        "Cz",
    ]  # electrode names pinned on the left
    assert (
        'data-tip="DS000001 (ds000001) · 2 of its 4 electrodes among the 2 rows · 1–16 channels"'
        in body
    )
    assert (
        "rotate(-90)" in body and ">DS000001 (ds000001)</text>" in body
    )  # the column label, with its tooltip
    assert re.findall(r'data-band="([^"]+)" data-count="(\d+)"', body) == [
        ("1–16", "2")
    ]
    assert "1–16 channels · 2 datasets" in body
    assert charts.montage_chart(v0_rows(rows), buckets, electrodes)[1].startswith(
        "No electrode names"
    )


def test_montage_columns_by_channel_band(rows, buckets, bank):
    """Columns sorted by electrode count into bands, a bracket per band; an EGI net (E1…E20 and Cz) counts as its
    own naming, its column showing the one name that coincides.
    """
    from plot_corpus.electrodes import aggregate_positions

    egi_names = [f"E{j}" for j in range(1, 21)]
    egi = make_row("x-egi", CONVERTED, short_name="EGI net")
    more = [{"bucket": "4", "dataset": "x-egi", "n_chans": "21", "n_sessions": "1", "hours": "1",
             "channels": "|".join(egi_names + ["Cz"])}]  # fmt: skip
    pos = dict(
        bank, **{n: [0.01 + 0.001 * j, 0.02, 0.09] for j, n in enumerate(egi_names)}
    )
    els = aggregate_positions(buckets + more, pos)
    body, cap = charts.montage_chart(
        v0_rows(rows) + [egi], buckets + more, els, min_datasets=2
    )
    cols = re.findall(r'data-dataset="([^"]+)" data-on="(\d+)" data-n="(\d+)"', body)
    assert cols == [
        ("x-egi", "1", "21"),
        ("ds000001", "2", "4"),
        ("ds000002", "2", "3"),
    ]
    assert re.findall(r'data-band="([^"]+)" data-count="(\d+)"', body) == [
        ("17–32", "1"),
        ("1–16", "2"),
    ]
    assert "17–32 channels · 1 dataset" in body and ">1–16 · 2<" in body
    assert (
        "and 1 dataset uses its own naming (fewer than half of its electrodes among the rows)"
        in cap
    )
    assert "data-family" not in body  # no row grouping by naming family


def test_channel_bands_measured_per_bucket(buckets):
    body, cap = charts.channel_bands_buckets(buckets)
    assert "corpus-fill-converted" in body and "v1" not in body
    assert cap.startswith(
        "100 % of the 150 converted hours come from 1–16-channel recordings (2 datasets)"
    )


def test_new_fragments_build_scoped_and_small(data_dir, tmp_path):
    manifest = build(data_dir, tmp_path / "out")
    for key in NEW_KEYS:
        assert key in manifest["fragments"], key
        assert manifest["fragments"][key]["bytes"] < 200_000
