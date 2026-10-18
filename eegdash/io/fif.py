"""Read the complete part of a FIF file whose last data buffer runs past the end of the file.

MNE opens such a file (``read_raw_fif`` scans the tags when the directory at the
end is missing, with the warning "FIF tag directory missing at the end of the
file ... possibly corrupted file") and counts every buffer it found, including
the cut one. Reading that buffer then fails (``cannot reshape array of size ...``
or ``buffer size must be a multiple of element size``), so the recording was
refused. Seen in deposits where the stored object itself is short (sizes equal
OpenNeuro's and the NEMAR annex key's): on004012 sub-19 stim04 and sub-22 stim02,
on004346 sub-01 run-01, each losing its last 1 s.

Only MNE's public API is used: ``get_data`` to find the last sample that reads,
and ``Raw.crop`` to drop the rest. Nothing is padded or made up; the recording
comes back shorter than its header says, with a warning that gives the loss.
"""

from mne.utils import warn


def _reads(raw, i):
    try:
        raw.get_data(start=i, stop=i + 1)
    except Exception:  # noqa: BLE001 - whatever MNE raises on a cut buffer
        return False
    return True


def crop_unreadable_tail(raw):
    """Crop a FIF ``raw`` (``preload=False``) to the samples it can read.

    Returns the cropped raw, or ``None`` when ``raw`` is not a FIF file, when its
    first sample does not read either, or when its last sample reads (nothing to
    do here: the failure is elsewhere).
    """
    fnames = [str(f) for f in (getattr(raw, "filenames", None) or []) if f]
    if not fnames or not all(f.lower().endswith((".fif", ".fif.gz")) for f in fnames):
        return None
    n = raw.n_times
    if n < 2 or not _reads(raw, 0) or _reads(raw, n - 1):
        return None
    lo, hi = 0, n - 1  # sample lo reads, sample hi does not; a cut tail is one block
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if _reads(raw, mid):
            lo = mid
        else:
            hi = mid
    keep = lo + 1
    sfreq = raw.info["sfreq"]
    warn(
        f"{fnames[-1]}: the last data buffer runs past the end of the file "
        f"(truncated file). Reading the first {keep} of {n} samples; the last "
        f"{(n - keep) / sfreq:.3f} s are not in the file."
    )
    return raw.crop(tmax=raw.times[keep - 1])
