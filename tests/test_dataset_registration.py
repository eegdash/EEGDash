import importlib.util
from pathlib import Path


class DummyBase:
    pass


def test_register_openneuro_datasets(tmp_path: Path):
    module_path = Path(__file__).resolve().parents[1] / "eegdash" / "registry.py"
    spec = importlib.util.spec_from_file_location("registry", module_path)
    registry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(registry)

    summary = tmp_path / "dataset_summary.csv"
    summary.write_text(
        "\n".join(
            [
                "dataset_id,num_subjects,num_sessions,num_runs,num_channels,sampling_rate,duration",
                "ds002718,18,18,1,74,250,14.844",
                "ds000001,1,1,1,1,1,1",
            ]
        )
    )
    namespace = {}
    registered = registry.register_openneuro_datasets(
        summary, namespace=namespace, base_class=DummyBase
    )

    assert set(registered) == {"DS002718", "DS000001"}
    ds_class = registered["DS002718"]
    assert ds_class is namespace["DS002718"]
    assert issubclass(ds_class, DummyBase)
    assert ds_class._dataset == "ds002718"
