"""The corpus by paradigm, from the synced atlas, bucket and recording tables."""

from __future__ import annotations

import html

from .charts import missing
from .data import by_id


def paradigm_summary(rows, buckets, recordings):
    """Aggregate datasets using bucket hours and the recording table's tokens.

    Subjects are summed per dataset, not deduplicated across releases. Bucket
    hours retain more precision than atlas hours; a dataset without buckets uses
    its atlas hours. Incomplete recording tables leave tokens unknown, not zero.
    """
    records = by_id(recordings)
    if len(records) != len(recordings):
        raise ValueError("duplicate dataset ids in the recording table")
    have_tokens = bool(rows) and all(
        r["dataset_id"] in records and records[r["dataset_id"]]["tokens"] for r in rows
    )
    hours = {}
    for bucket in buckets:
        dsid = bucket["dataset"]
        hours[dsid] = hours.get(dsid, 0.0) + float(bucket["hours"])
    groups = {}
    for row in rows:
        dsid = row["dataset_id"]
        group = groups.setdefault(
            row["paradigm"] or "unspecified",
            dict(datasets=0, subjects=0, hours=0.0, tokens=0),
        )
        group["datasets"] += 1
        group["subjects"] += int(row["n_subjects"])
        group["hours"] += hours.get(dsid, float(row["hours"]))
        if have_tokens:
            group["tokens"] += int(records[dsid]["tokens"])
    result = [dict(paradigm=key, **value) for key, value in groups.items()]
    result.sort(key=lambda r: (-r["hours"], r["paradigm"]))
    if not result:
        return []
    total = {
        key: sum(r[key] for r in result)
        for key in ("datasets", "subjects", "hours", "tokens")
    }
    result.append(dict(paradigm="Total", **total))
    for row in result:
        row["pct_hours"] = 100 * row["hours"] / total["hours"] if total["hours"] else 0
        row["pct_tokens"] = (
            100 * row["tokens"] / total["tokens"]
            if have_tokens and total["tokens"]
            else None
        )
    return result


def summary_cells(row):
    """The paper table's rounding: integer hours, billion tokens, 0.1% shares."""
    return [
        row["paradigm"],
        f"{row['datasets']:,}",
        f"{row['subjects']:,}",
        f"{row['hours']:,.0f}",
        f"{row['pct_hours']:.1f}",
        f"{row['tokens'] / 1e9:.2f}" if row["pct_tokens"] is not None else "—",
        f"{row['pct_tokens']:.1f}" if row["pct_tokens"] is not None else "—",
    ]


def paradigms_table_html(rows, buckets, recordings):
    """Render every paradigm and the total with the existing corpus table styles."""
    summary = paradigm_summary(rows, buckets, recordings)
    if not summary:
        return missing("paradigm data")
    headers = (
        "Paradigm",
        "Datasets",
        "Subjects",
        "Hours",
        "Hours (%)",
        "Tokens (B)",
        "Tokens (%)",
    )
    head = "".join(f'<th scope="col">{h}</th>' for h in headers)
    body = []
    for row in summary:
        values = summary_cells(row)
        cells = [f'<th scope="row">{html.escape(values[0])}</th>']
        cells.extend(f'<td class="corpus-n">{html.escape(v)}</td>' for v in values[1:])
        body.append(f"<tr>{''.join(cells)}</tr>")
    caption = (
        "The corpus by paradigm, ordered by hours. Hours and token shares use unrounded totals; "
        "subjects are summed across datasets. Tokens per epoch: 11 patches per channel in each "
        "10-second window (1-second patches, 0.9-second stride). B means billion."
    )
    if summary[-1]["pct_tokens"] is None:
        caption += " Tokens are not available for every dataset in this data copy."
    return (
        '<div class="corpus-table-wrap" tabindex="0" role="region" aria-label="Corpus by paradigm">'
        f'<table class="corpus-table corpus-paradigm-table"><thead><tr>{head}</tr></thead>'
        f"<tbody>{''.join(body[:-1])}</tbody><tfoot>{body[-1]}</tfoot></table></div>",
        caption,
    )
