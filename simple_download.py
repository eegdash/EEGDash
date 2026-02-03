import shutil
import sys
from pathlib import Path
from eegdash import EEGDashDataset

def verify(ds_id, subject):
    cache_dir = Path(".verify_cache")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)

    print(f"\n[Testing {ds_id}]")
    sys.stdout.flush()

    # Phase 1: Online (query only, no download)
    try:
        print(f"1. Loading {ds_id} with download=True (subject={subject})...")
        sys.stdout.flush()
        ds_on = EEGDashDataset(dataset=ds_id, cache_dir=cache_dir, download=True, subject=subject, n_jobs=1)
        n_on = len(ds_on.datasets)
        print(f"   Online found {n_on} recording(s).")

        # Actually download the files (lazy by default)
        print(f"   Downloading files...")
        sys.stdout.flush()
        ds_on.download_all(n_jobs=1)
        print(f"   Download complete.")
    except Exception as e:
        print(f"   Online failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Phase 2: Offline
    try:
        print(f"2. Loading {ds_id} with download=False...")
        sys.stdout.flush()
        ds_off = EEGDashDataset(dataset=ds_id, cache_dir=cache_dir, download=False, subject=subject)
        n_off = len(ds_off.datasets)
        print(f"   Discovered {n_off} recording(s) offline.")
    except Exception as e:
        print(f"   Offline failed: {e}")
        import traceback
        traceback.print_exc()
        return

    if n_on == n_off and n_off > 0:
        print(f"SUCCESS: {ds_id} verified.")
    else:
        print(f"FAILURE: Count mismatch (On: {n_on}, Off: {n_off})")

if __name__ == "__main__":
    verify("ds003104", "01")
    verify("ds005929", "6016")
