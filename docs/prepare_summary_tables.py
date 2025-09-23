import glob
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px


def _to_numeric_median_list(val) -> float | None:
    """Return a numeric value from possible list-like strings.

    Examples
    --------
    - "64" -> 64
    - "6,129" -> median -> 67.5 -> 68
    - "128, 512" -> 320
    - 500.0 -> 500

    """
    if pd.isna(val):
        return None
    try:
        # already numeric
        return float(val)
    except Exception:
        pass
    s = str(val).strip().strip("[]")
    if not s:
        return None
    try:
        nums = [float(x) for x in s.split(",") if str(x).strip()]
        if not nums:
            return None
        return float(np.median(nums))
    except Exception:
        return None


def _safe_int(x, default=None):
    try:
        if x is None or pd.isna(x):
            return default
        return int(round(float(x)))
    except Exception:
        return default


def gen_datasets_bubble(
    df: pd.DataFrame,
    out_html: str = "_static/dataset/dataset_bubble.html",
    x_var: str = "records",  # one of: 'records', 'duration_h', 'size_gb', 'tasks'
):
    """Generate an interactive bubble chart for datasets.

    - x: total duration (hours)
    - y: number of subjects
    - size: on-disk size (GB)
    - color: sampling frequency band
    """
    d = df.copy()
    d = d[d["dataset"].str.lower() != "test"]

    # numeric columns
    d["duration_h"] = pd.to_numeric(d.get("duration_hours_total"), errors="coerce")
    d["subjects"] = pd.to_numeric(d.get("n_subjects"), errors="coerce")
    d["records"] = pd.to_numeric(d.get("n_records"), errors="coerce")
    d["tasks"] = pd.to_numeric(d.get("n_tasks"), errors="coerce")
    d["size_bytes"] = pd.to_numeric(d.get("size_bytes"), errors="coerce")

    # parse sampling and channels into representative numeric values
    d["sfreq"] = d["sampling_freqs"].map(_to_numeric_median_list)
    d["nchans"] = d["nchans_set"].map(_to_numeric_median_list)

    # disk size in GB for sizing
    GB = 1024**3
    d["size_gb"] = d["size_bytes"] / GB

    # color bands by sampling frequency
    def _sf_band(x):
        if pd.isna(x):
            return "Unknown"
        x = float(x)
        if x <= 250:
            return "≤ 250 Hz"
        if x <= 500:
            return "251–500 Hz"
        if x <= 1000:
            return "501–1000 Hz"
        return "> 1000 Hz"

    d["sf_band"] = d["sfreq"].apply(_sf_band)

    # hover content
    def _fmt_size(bytes_):
        return human_readable_size(_safe_int(bytes_, 0))

    # choose x axis field and labels
    x_field = (
        x_var if x_var in {"records", "duration_h", "size_gb", "tasks"} else "records"
    )
    x_label = {
        "records": "#Records",
        "duration_h": "Duration (hours)",
        "size_gb": "Size (GB)",
        "tasks": "#Tasks",
    }[x_field]

    # hover text adapts to x
    if x_field == "duration_h":
        x_hover = "Duration: %{x:.2f} h"
    elif x_field == "size_gb":
        x_hover = "Size: %{x:.2f} GB"
    elif x_field == "tasks":
        x_hover = "Tasks: %{x:,}"
    else:
        x_hover = "Records (x): %{x:,}"

    hover = (
        "<b>%{customdata[0]}</b>"  # dataset id
        "<br>Subjects: %{y:,}"
        f"<br>{x_hover}"
        "<br>Records: %{customdata[1]:,}"
        "<br>Tasks: %{customdata[2]:,}"
        "<br>Channels: %{customdata[3]}"
        "<br>Sampling: %{customdata[4]} Hz"
        "<br>Size: %{customdata[5]}"
        "<extra></extra>"
    )

    d = d.dropna(subset=["duration_h", "subjects", "size_gb"])  # need these

    # Marker sizing: scale into a good visual range
    max_size = max(d["size_gb"].max(), 1)
    sizeref = (2.0 * max_size) / (40.0**2)  # target ~40px max marker

    # Prepare prettified strings for hover
    def _fmt_int(v):
        if v is None or pd.isna(v):
            return ""
        try:
            return str(int(round(float(v))))
        except Exception:
            return str(v)

    sfreq_str = d["sfreq"].map(_fmt_int)
    nchans_str = d["nchans"].map(_fmt_int)

    fig = px.scatter(
        d,
        x=x_field,
        y="subjects",
        size="size_gb",
        color="sf_band",
        hover_name="dataset",
        custom_data=[
            d["dataset"],
            d["records"],
            d["tasks"],
            nchans_str,
            sfreq_str,
            d["size_bytes"].map(_fmt_size),
        ],
        size_max=40,
        labels={"subjects": "#Subjects", "sf_band": "Sampling Freq.", x_field: x_label},
        title="Dataset Landscape",
    )

    # tune marker sizing explicitly for better control
    for tr in fig.data:
        tr.marker.update(
            sizemin=6,
            sizemode="area",
            sizeref=sizeref,
            line=dict(width=0.6, color="rgba(0,0,0,0.3)"),
        )
        tr.hovertemplate = hover

    fig.update_layout(
        height=560,
        margin=dict(l=40, r=20, t=80, b=40),
        template="plotly_white",
        legend=dict(
            title="Sampling Freq.",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=0.99,
        ),
        font=dict(
            family="Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif",
            size=14,
        ),
        title=dict(
            text="Dataset Landscape",
            x=0.01,
            xanchor="left",
            y=0.98,
            yanchor="top",
            pad=dict(t=10, b=8),
        ),
    )

    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.12)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.12)", zeroline=False)

    out_path = Path(out_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        str(out_path),
        full_html=False,
        include_plotlyjs="cdn",
        div_id="dataset-bubble",
        config={
            "responsive": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        },
    )
    return str(out_path)


def human_readable_size(num_bytes: int) -> str:
    """Format bytes using the closest unit among MB, GB, TB (fallback to KB/B).

    Chooses the largest unit such that the value is >= 1. Uses base 1024.
    """
    if num_bytes is None:
        return "0 B"
    size = float(num_bytes)
    units = [
        (1024**4, "TB"),
        (1024**3, "GB"),
        (1024**2, "MB"),
        (1024**1, "KB"),
        (1, "B"),
    ]
    for factor, unit in units:
        if size >= factor:
            value = size / factor
            # Use no decimals for B/KB; two decimals otherwise
            if unit in ("B", "KB"):
                return f"{int(round(value))} {unit}"
            return f"{value:.2f} {unit}"
    return "0 B"


def wrap_tags(cell: str):
    if pd.isna(cell):
        return ""
    tags_cell = [f'<span class="tag">{p.strip()}</span>' for p in cell.split(", ")]
    return " ".join(tags_cell)


def wrap_dataset_name(name: str):
    # Remove any surrounding whitespace
    name = name.strip()
    # Construct the URL based on the dataset name
    url = f"api/eegdash.dataset.html#eegdash.dataset.{name.upper()}.html"
    return f'<a href="{url}">{name.upper()}</a>'


def prepare_table(df: pd.DataFrame):
    # drop test dataset and create a copy to avoid SettingWithCopyWarning
    df = df[df["dataset"] != "test"].copy()

    df["dataset"] = df["dataset"].apply(wrap_dataset_name)
    # changing the column order
    df = df[
        [
            "dataset",
            "n_records",
            "n_subjects",
            "n_tasks",
            "nchans_set",
            "sampling_freqs",
            "duration_hours_total",
            "size",
            "size_bytes",
            "Type Subject",
            "modality of exp",
            "type of exp",
        ]
    ]

    # renaming time for something small
    df = df.rename(
        columns={
            "duration_hours_total": "duration (h)",
            "modality of exp": "modality",
            "type of exp": "type",
            "Type Subject": "pathology",
        }
    )
    # number of subject are always int
    df["n_subjects"] = df["n_subjects"].astype(int)
    # number of tasks are always int
    df["n_tasks"] = df["n_tasks"].astype(int)
    # number of records are always int
    df["n_records"] = df["n_records"].astype(int)

    # from the sample frequency list, I will apply str
    df["sampling_freqs"] = df["sampling_freqs"].apply(parse_freqs)
    # from the channels set, I will follow the same logic of freq
    df["nchans_set"] = df["nchans_set"].apply(parse_freqs)
    # rename the nchans to channels
    # Creating the total line
    df["duration (h)"] = df["duration (h)"].round(2)

    df.loc["Total"] = df.sum(numeric_only=True)
    df.loc["Total", "dataset"] = f"Total {len(df) - 1} datasets"
    df.loc["Total", "nchans_set"] = ""
    df.loc["Total", "sampling_freqs"] = ""
    df.loc["Total", "duration (h)"] = None
    df.loc["Total", "size"] = human_readable_size(df.loc["Total", "size_bytes"])
    df = df.drop(columns=["size_bytes"])
    # arrounding the hours

    df.index = df.index.astype(str)

    return df


def main(source_dir: str, target_dir: str):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    files = glob.glob(str(Path(source_dir) / "dataset" / "*.csv"))
    for f in files:
        target_file = target_dir / Path(f).name
        print(f"Processing {f} -> {target_file}")
        df_raw = pd.read_csv(
            f, index_col=False, header=0, skipinitialspace=True
        )  # , sep=";")
        # Generate bubble chart from the raw data to have access to size_bytes
        # Use x-axis as number of records for better spread
        gen_datasets_bubble(
            df_raw, str(target_dir / "dataset_bubble.html"), x_var="records"
        )

        df = prepare_table(df_raw)
        # preserve int values
        df["n_subjects"] = df["n_subjects"].astype(int)
        df["n_tasks"] = df["n_tasks"].astype(int)
        df["n_records"] = df["n_records"].astype(int)
        int_cols = ["n_subjects", "n_tasks", "n_records"]

        # Coerce to numeric, allow NAs, and keep integer display
        df[int_cols] = (
            df[int_cols].apply(pd.to_numeric, errors="coerce").astype("Int64")
        )
        df = df.rename(
            columns={
                "dataset": "Dataset",
                "nchans_set": "# of channels",
                "sampling_freqs": "sampling (Hz)",
                "size": "size",
                "n_records": "# of records",
                "n_subjects": "# of subjects",
                "n_tasks": "# of tasks",
            }
        )
        # (If you add a 'Total' row after this, cast again or build it as Int64.)
        html_table = df.to_html(
            classes=["sd-table", "sortable"], index=False, escape=False
        )
        with open(
            f"{target_dir}/dataset_summary_table.html", "+w", encoding="utf-8"
        ) as f:
            f.write(html_table)


def parse_freqs(value) -> str:
    if isinstance(value, str):
        freq = [int(float(f)) for f in value.strip("[]").split(",")]
        if len(freq) == 1:
            return f"{int(freq[0])}"
        else:
            return f"{int(np.median(freq))}*"

    elif isinstance(value, (int, float)) and not pd.isna(value):
        return f"{int(value)}"
    return ""  # for other types like nan


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("source_dir", type=str)
    parser.add_argument("target_dir", type=str)
    args = parser.parse_args()
    main(args.source_dir, args.target_dir)
    print(args.target_dir)
