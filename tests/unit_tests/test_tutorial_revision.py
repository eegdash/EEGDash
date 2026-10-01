"""Bounded tutorial contracts: execute source cells without acquisition or training.

Numerical fixtures here exercise indexing and validation, never gallery results.
"""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parents[2]
FOUNDATION = ROOT / "examples/tutorials/70_transfer_foundation"


def _tree(path):
    return ast.parse(path.read_text(), filename=str(path))


def _execute(nodes, namespace):
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), "<tutorial-cell>", "exec"),
        namespace,
    )


def _assignment(tree, name):
    return next(
        n
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)
    )


@pytest.mark.parametrize(
    "filename",
    [
        "plot_73_finetune_pretrained_model.py",
        "plot_73_finetune_pretrained_model_simple.py",
    ],
)
def test_cbramod_epochs_respect_cue_bounds_bad_spans_and_sample_origin(filename):
    """The last open cue needs 19, not 29 s; BAD-overlapping epochs are excluded."""
    from types import SimpleNamespace

    import mne

    tree = _tree(FOUNDATION / filename)
    ns = {
        "np": np,
        "pd": pd,
        "mne": mne,
        "SFREQ": 100,
        "CUE_WINDOW": {
            "instructed_toOpenEyes": (5, 19),
            "instructed_toCloseEyes": (15, 29),
        },
    }
    raw = mne.io.RawArray(
        np.arange(6000, dtype=float)[None],
        mne.create_info(["E70"], 100, "eeg"),
        first_samp=500,
        verbose=False,
    )
    raw.set_annotations(
        mne.Annotations(
            [0, 40, 46, 55],
            [0, 0, 1, 0],
            [
                "instructed_toCloseEyes",
                "instructed_toOpenEyes",
                "BAD_test",
                "instructed_toCloseEyes",
            ],
        )
    )
    ns["dataset"] = SimpleNamespace(
        datasets=[SimpleNamespace(raw=raw, description={"subject": "fixture"})]
    )
    # Execute the actual epoch construction, not a duplicated implementation.
    first = next(
        i
        for i, n in enumerate(tree.body)
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Tuple)
            and any(isinstance(x, ast.Name) and x.id == "epoch_arrays" for x in t.elts)
            for t in n.targets
        )
    )
    _execute(tree.body[first : first + 2], ns)
    epochs = ns["epochs"]
    assert len(epochs) == 13  # 14 requested minus one overlapping BAD_test
    np.testing.assert_array_equal(
        epochs.get_data()[:, 0, 0], epochs.events[:, 0] - raw.first_samp
    )
    assert epochs.get_data().shape[-1] == 200
    assert set(ns["metadata_tables"][0].target) == {0, 1}
    assert any("BAD_test" in reasons for reasons in epochs.drop_log)


@pytest.mark.parametrize(
    "filename",
    [
        "plot_73_finetune_pretrained_model.py",
        "plot_73_finetune_pretrained_model_simple.py",
    ],
)
def test_cbramod_normalization_centers_and_scales_channels(filename):
    ns = {"np": np}
    _execute(
        [
            n
            for n in _tree(FOUNDATION / filename).body
            if isinstance(n, ast.FunctionDef) and n.name == "standardize_recording"
        ],
        ns,
    )
    normalize = ns["standardize_recording"]
    result = normalize(np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]]))
    np.testing.assert_allclose(result.mean(axis=1), 0, atol=1e-12)
    np.testing.assert_allclose(result.std(axis=1), 1)


def _feature_handoff():
    producer = _tree(ROOT / "examples/tutorials/40_features/plot_40_first_features.py")
    # Evaluate the actual schema dictionary used by json.dumps in tutorial 40.
    schema_node = next(
        n.args[0]
        for n in ast.walk(producer)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "dumps"
    )
    column = "spectral_power_alpha_E70"
    ns = {
        "subjects": ["1", "2", "3"],
        "mapping": {"10": 0, "12": 1},
        "feature_columns": [column],
        "sfreq": 256,
        "window_size": 1024,
        "bands": {"alpha": (8, 12)},
        "channel_names": ["E70"],
        "version": lambda name: "test",
    }
    schema = eval(compile(ast.Expression(schema_node), "<schema>", "eval"), ns)
    table = pd.DataFrame(
        {
            "subject": ["1", "2", "3"],
            "session": ["0"] * 3,
            "run": ["0"] * 3,
            "i_start_in_trial": [0, 0, 0],
            "target": [0, 1, 0],
            "frequency_hz": [10.0, 12.0, 10.0],
            column: [1e-12, 0.0, 3e-12],
        }
    )
    consumer = _tree(
        ROOT / "examples/tutorials/40_features/plot_42_features_to_sklearn.py"
    )
    start = consumer.body.index(_assignment(consumer, "columns"))
    stop = consumer.body.index(_assignment(consumer, "groups")) + 1
    return schema, table, consumer.body[start:stop]


def test_feature_schema_producer_consumer_contract():
    schema, table, nodes = _feature_handoff()
    ns = {"np": np, "pd": pd, "schema": schema, "table": table}
    _execute(nodes, ns)
    assert ns["X"].shape == (3, 1)
    assert np.isfinite(ns["X"]).all()
    assert ns["X"][1, 0] == -30  # zero power uses the documented log floor
    np.testing.assert_array_equal(ns["y"], [0, 1, 0])
    np.testing.assert_array_equal(ns["groups"], ["1", "2", "3"])


@pytest.mark.parametrize("batch_size", [1, 7, 32])
def test_intro_loader_retains_labels_for_changed_and_partial_batches(batch_size):
    import torch
    from torch.utils.data import DataLoader

    path = ROOT / "examples/tutorials/00_start_here/plot_02_dataset_to_dataloader.py"
    tree = _tree(path)
    labels = np.arange(19) % 3
    windows = [
        (torch.full((2, 8), float(i)), int(label), (i, 0, 8))
        for i, label in enumerate(labels)
    ]
    ns = {"DataLoader": DataLoader, "windows": windows, "batch_size": batch_size}
    _execute([_assignment(tree, "loader")], ns)
    batches = list(ns["loader"])
    assert len(batches[-1][1]) == (19 % batch_size or batch_size)
    np.testing.assert_array_equal(
        np.concatenate([batch[1] for batch in batches]), labels
    )
    for signals, targets, _ in batches:
        np.testing.assert_array_equal(signals[:, 0, 0].numpy().astype(int) % 3, targets)


@pytest.mark.parametrize(
    "filename,annotation_name",
    [
        (
            "tutorials/10_core_workflow/plot_10_preprocess_and_window.py",
            "annotations_before",
        ),
        (
            "tutorials/20_event_related/plot_20_visual_p300_oddball.py",
            "source_annotations",
        ),
        (
            "tutorials/20_event_related/plot_21_auditory_oddball.py",
            "source_annotations",
        ),
        ("applied/project_p300_transfer.py", "source_annotations"),
    ],
)
@pytest.mark.parametrize("absolute_origin", [False, True])
def test_annotation_restore_does_not_add_first_sample_twice(
    filename, annotation_name, absolute_origin
):
    from datetime import datetime, timezone

    import mne

    raw = mne.io.RawArray(
        np.zeros((1, 1000)),
        mne.create_info(["Pz"], 100, "eeg"),
        first_samp=500,
        verbose=False,
    )
    if absolute_origin:
        raw.set_meas_date(datetime(2020, 1, 1, tzinfo=timezone.utc))
    raw.set_annotations(mne.Annotations([1, 2], [0, 0.5], ["cue", "BAD_test"]))
    expected = raw.annotations.onset.copy()
    tree = _tree(ROOT / "examples" / filename)
    # Execute only the origin conversion and set_annotations call from the source.
    guard = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.If)
        and any(
            isinstance(x, ast.Attribute) and x.attr == "orig_time"
            for x in ast.walk(n.test)
        )
    )
    restore = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Expr)
        and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Attribute)
        and n.value.func.attr == "set_annotations"
        and isinstance(n.value.args[0], ast.Name)
        and n.value.args[0].id == annotation_name
    )
    _execute([guard, restore], {"raw": raw, annotation_name: raw.annotations.copy()})
    np.testing.assert_array_equal(raw.annotations.onset, expected)
    np.testing.assert_array_equal(raw.annotations.description, ["cue", "BAD_test"])


@pytest.mark.parametrize("stage", ["training", "held-out"])
@pytest.mark.parametrize("nonfinite", [False, True])
def test_hpc_rejects_nonfinite_model_outputs_before_success(stage, nonfinite):
    """Exercise actual loop bodies with a tiny model, without EEG or training."""
    import torch
    from torch.nn import functional as F

    tree = _tree(ROOT / "examples/hpc/tutorial_hpc_cache_and_slurm.py")
    loops = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and isinstance(node.iter, ast.Name)
        and node.iter.id == ("train_loader" if stage == "training" else "test_loader")
    ]
    body = loops[0].body
    if stage == "training":
        # Stop before backward/step; only forward and finite-loss gate are needed.
        stop = next(i for i, node in enumerate(body) if isinstance(node, ast.AugAssign))
        body = body[:stop]
    scores = torch.tensor([[float("nan") if nonfinite else 1.0, 0.0]])
    ns = {
        "torch": torch,
        "F": F,
        "model": lambda x: scores,
        "x": torch.ones(1, 2),
        "y": torch.tensor([0]),
        "device": torch.device("cpu"),
        "correct_test": 0,
    }
    if nonfinite:
        with pytest.raises(ValueError, match="Nonfinite"):
            _execute(body, ns)
    else:
        _execute(body, ns)
        if stage == "training":
            assert torch.isfinite(ns["loss"])
        else:
            assert ns["correct_test"] == 1


def test_pipeline_comparison_plots_paired_subject_scores(monkeypatch):
    """Each participant contributes the two measured pipeline scores."""
    import matplotlib.pyplot as plt

    tree = _tree(
        ROOT / "examples/tutorials/50_evaluation/plot_54_compare_two_pipelines.py"
    )
    results = pd.DataFrame(
        {"Logistic": [0.4, 0.6, 0.7], "LDA": [0.5, 0.55, 0.8]},
        index=["1", "2", "3"],
    )
    monkeypatch.setattr(plt, "show", lambda: None)
    existing = set(plt.get_fignums())
    try:
        _execute(
            tree.body[tree.body.index(_assignment(tree, "difference")) :],
            {"plt": plt, "results": results, "mapping": {"10": 0, "12": 1}},
        )
        created = sorted(set(plt.get_fignums()) - existing)
        assert len(created) == 1
        scores = plt.figure(created[0]).axes[0]
        assert scores.get_ylabel() == "LOSO balanced accuracy"
        for line, values in zip(scores.lines[:3], results.to_numpy(), strict=True):
            np.testing.assert_array_equal(line.get_ydata(), values)
    finally:
        for number in set(plt.get_fignums()) - existing:
            plt.close(number)


def test_dataset_plot_inherits_released_braindecode_viewer(tmp_path):
    """The public plotting method embeds file bytes without a local server."""
    import mne
    from IPython.display import HTML

    from braindecode.datasets import BaseConcatDataset, RawDataset
    from eegdash import EEGDashDataset

    path = tmp_path / "sample_raw.fif"
    raw = mne.io.RawArray(
        np.zeros((1, 100)), mne.create_info(["Cz"], 100, "eeg"), verbose=False
    )
    raw.save(path, overwrite=True, verbose=False)
    dataset = BaseConcatDataset([RawDataset(mne.io.read_raw_fif(path, verbose=False))])
    assert EEGDashDataset.plot is BaseConcatDataset.plot
    result = dataset.plot(index=0)
    assert isinstance(result, HTML)
    assert "iframe" in result.data
    assert "https://eegdash.github.io/eegdash-viewer" in result.data
