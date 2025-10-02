from pathlib import Path

import pandas as pd
import pytest

import docs.prepare_summary_tables as prepare_module


def test_prepare_summary_tables_generates_plotly_html(tmp_path, monkeypatch, caplog):
    pytest.importorskip("plotly")

    static_dir = tmp_path / "static"
    monkeypatch.setattr(prepare_module, "STATIC_DATASET_DIR", static_dir, raising=False)

    def _stubbed_html_export(_df, output_path: Path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("<html></html>", encoding="utf-8")
        return output_path

    monkeypatch.setattr(
        prepare_module,
        "generate_dataset_sankey",
        _stubbed_html_export,
        raising=False,
    )
    monkeypatch.setattr(
        prepare_module,
        "generate_modality_ridgeline",
        _stubbed_html_export,
        raising=False,
    )

    source_dir = tmp_path / "source"
    datasets_dir = source_dir / "dataset"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    dataset_rows = [
        {
            "dataset": "DemoDataset",
            "n_records": 10,
            "n_subjects": 4,
            "n_tasks": 2,
            "nchans_set": "[32]",
            "sampling_freqs": "[256]",
            "size": "1.0 GB",
            "size_bytes": 1_073_741_824,
            "Type Subject": "Healthy",
            "modality of exp": "Visual",
            "type of exp": "Perception",
            "duration_hours_total": 1.5,
        }
    ]
    csv_path = datasets_dir / "demo.csv"
    pd.DataFrame(dataset_rows).to_csv(csv_path, index=False)

    target_dir = tmp_path / "build"
    with caplog.at_level("INFO"):
        prepare_module.main(
            str(source_dir),
            str(target_dir),
        )

    html_path = target_dir / "dataset_bubble.html"
    assert html_path.exists()

    html_markup = html_path.read_text(encoding="utf-8")
    assert "plotly-graph-div" in html_markup
    assert "updatemenus" not in html_markup

    static_html_path = static_dir / "dataset_bubble.html"
    assert static_html_path.exists()
    assert html_markup == static_html_path.read_text(encoding="utf-8")
