"""Guard the NM000134 EEG-to-image start-kit tutorial contract."""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
TUTORIAL = ROOT / "examples/eeg2026/tutorial_eeg_to_image_start_kit.py"
README = ROOT / "examples/eeg2026/README.rst"
SPHINX_CONF = ROOT / "docs/source/conf.py"
AUTO_EXAMPLES_INDEX = ROOT / "docs/source/_extensions/auto_examples_index.py"
NM000134_STIMULUS_GIT_REF = "61b04adf7bca47f220b85f3744a610b44046c62f"


def test_eeg_to_image_start_kit_uses_the_real_nm000134_test_record():
    """Keep the lesson executable, real-data-only, and viewer-native."""
    assert TUTORIAL.is_file()
    source = TUTORIAL.read_text(encoding="utf-8")
    prose = source.replace("\n# ", " ")
    compile(source, str(TUTORIAL), "exec")

    assert 'DATASET = "nm000134"' in source
    assert 'SUBJECT = "09"' in source
    assert 'SESSION = "01"' in source
    assert 'RUN = "02"' in source
    assert 'TASK = "images"' in source
    assert "sub-09/ses-01/eeg/sub-09_ses-01_task-images_run-02_eeg.edf" in source
    assert "EXPECTED_STIM_TEST_EVENTS = 936" in source
    assert "EXPECTED_UNIQUE_IMAGE_IDS = 200" in source
    assert "EXPECTED_EEG_CHANNELS = 32" in source
    assert "SAMPLING_FREQUENCY = 256.0" in source
    assert "assert len(dataset.records) == 1" in source
    assert "len(dataset)" not in source
    assert "stim_test" in source
    assert "event_intervals_s" in source
    assert "raw.get_data" in source
    assert "trace_display_microvolts" in source
    assert "np.nanmedian(" in source
    assert "axis=1, keepdims=True" in source
    assert "median-centered for display" in source
    assert "dataset.plot(index=0, height=520)" in source
    assert "200 referenced image assets" in prose
    assert "NM000134 BIDS v1.0.1 Git release" in prose
    assert NM000134_STIMULUS_GIT_REF in prose
    assert "immutable commit" in prose
    assert "mutable HEAD" in prose
    assert "local viewer payload" in prose
    assert "does not commit or substitute JPEGs" in prose
    assert "initial download" in prose
    assert "catalog and hosted-viewer network access" in prose
    assert "Fresh materialization is pinned" in prose
    assert "Pre-existing local BIDS stimulus files are intentionally reused" in prose
    assert "NeuralBench owns" in prose
    assert "not an official train/test split" in prose
    assert "EEGDASH_RUN_2026_TUTORIALS" not in source
    assert "os.environ" not in source
    assert "np.random" not in source
    assert "def main()" not in source
    assert ".. figure::" not in source
    assert source.count("plt.show()") >= 2
    assert "# Step 1." in source
    assert "# Step 2." in source
    assert "# Step 3." in source
    assert "# Step 4." in source
    assert "# Step 5." in source


def test_eeg_to_image_gallery_header_describes_only_the_nm000134_start_kit():
    """A clean Sphinx-Gallery checkout needs the committed RST header."""
    assert README.is_file()
    header = README.read_text(encoding="utf-8")

    assert "NM000134" in header
    assert "tutorial_eeg_to_image_start_kit.py" in header
    assert "NeuralBench" in header
    assert "tutorial_bci_start_kit.py" not in header


def test_eeg2026_is_a_sphinx_gallery_leaf_directory():
    """Keep the new tutorial directory visible to the generated documentation."""
    tree = ast.parse(SPHINX_CONF.read_text(encoding="utf-8"))
    leaf_dirs = next(
        ast.literal_eval(statement.value)
        for statement in tree.body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "LEAF_DIRS"
            for target in statement.targets
        )
    )

    assert "eeg2026" in leaf_dirs


def _load_auto_examples_index():
    """Load the docs-only gallery-index extension without changing sys.path."""
    pytest.importorskip("sphinx", reason="gallery-index extension requires sphinx")
    spec = importlib.util.spec_from_file_location(
        "eegdash_auto_examples_index", AUTO_EXAMPLES_INDEX
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_eeg2026_gallery_is_linked_from_generated_root_index(tmp_path):
    """Exercise the generator rather than only checking its source roster."""
    extension = _load_auto_examples_index()
    extension._write_auto_examples_root_index(SimpleNamespace(srcdir=tmp_path))

    generated_index = (
        tmp_path / "generated" / "auto_examples" / "index.rst"
    ).read_text(encoding="utf-8")
    assert "generated/auto_examples/eeg2026/index" in generated_index
