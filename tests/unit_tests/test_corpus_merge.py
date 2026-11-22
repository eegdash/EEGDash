"""merge_parts and EEGDashCorpus.merge: parts into a store, and a finished store extended.

The parts are real conversions of tiny local datasets (the trees of
``test_corpus_convert``), made once per module and copied for each test.
Nothing touches the network. Every test needs ``zarr``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil

import numpy as np
import pytest
from test_corpus_convert import NAMES, _recording

import eegdash.corpus as corpus_mod
import eegdash.corpus_merge as merge_mod
from eegdash.corpus import EEGDashCorpus
from eegdash.corpus_convert import Bank
from eegdash.corpus_merge import merge_parts

zarr = pytest.importorskip("zarr")

ALPHA = ["alpha-1.0_000", "alpha-1.0_001"]  # two parts of one dataset
BETA = ["beta-1.0_000"]


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    """A cache of two local datasets and their parts, converted once."""
    import mne  # noqa: PLC0415 (heavy import, once per module)

    work = tmp_path_factory.mktemp("merge")
    cache = work / "cache"
    _recording(cache / "alpha-1.0", "01", seconds=30)  # 6 channels, 2 samples
    _recording(cache / "alpha-1.0", "02", names=NAMES[:4], seconds=30, sfreq=500)
    _recording(cache / "beta-1.0", "01", seconds=45, run=2)  # 4 samples
    pos = mne.channels.make_standard_montage("standard_1020").get_positions()["ch_pos"]
    bank = Bank(list(pos), np.array(list(pos.values())))
    table = work / "corpus.csv"
    table.write_text("dataset_id,licence\nalpha-1.0,CC0-1.0\nbeta-1.0,\n")
    report = EEGDashCorpus(table, cache_dir=cache).convert(
        work / "out", positions=bank, max_bytes=1
    )
    assert report.ok, report.problems
    return work


@pytest.fixture
def work(converted, tmp_path):
    """A copy of the converted work folder, this test's to change."""
    shutil.copytree(converted, tmp_path / "w")
    return tmp_path / "w"


def _parts(work, names):
    return [work / "out" / "parts" / n for n in names]


def _nothing_left(out, name):
    return not (out / f"{name}.zarr").exists() and not list(out.glob("*.partial"))


def _snapshot(root):
    """Every file and folder under *root*: path, size, SHA-256, mtime."""
    out = {}
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root).as_posix()
        if p.is_dir():
            out[rel] = "dir"
        else:
            st = p.stat()
            out[rel] = (
                st.st_size,
                hashlib.sha256(p.read_bytes()).hexdigest(),
                st.st_mtime_ns,
            )
    return out


def _chunks(group_dir):
    """The files of a bucket other than its zarr.json."""
    return sorted(
        p for p in group_dir.rglob("*") if p.is_file() and p.name != "zarr.json"
    )


def _edit_meta(path, change):
    meta = json.loads(path.read_text())
    change(meta.setdefault("attributes", {}))
    path.write_text(json.dumps(meta, indent=2))


def _store_one(work, **kwargs):
    """Alpha's parts merged into ``stores/one.zarr``."""
    return merge_parts(_parts(work, ALPHA), work / "stores", "one", **kwargs)


def _published(work):
    """A copy of store one as the published store is: no recipe at its root, licences."""
    _store_one(work)
    pub = work / "published" / "pub.zarr"
    shutil.copytree(work / "stores" / "one.zarr", pub)
    shutil.copy(work / "stores" / "one-manifest.csv", pub.parent / "pub-manifest.csv")

    def published(a):
        del a["recipe"], a["local_datasets"]
        a.update(
            licence="other",
            licences={"alpha-1.0": "CC0-1.0"},
            licence_file="LICENSES.csv",
        )

    _edit_meta(pub / "zarr.json", published)
    return pub


# ------------------------------------------------------------- parts alone


def test_parts_are_merged_into_one_store(work):
    report = merge_parts(_parts(work, ALPHA + BETA), work / "stores", "all")
    store = zarr.open_group(work / "stores" / "all.zarr", mode="r")
    assert sorted(k for k, _ in store.groups()) == ["0", "1", "2"]
    a = store.attrs
    assert (a["format"], a["version"], a["sfreq"], a["window"]) == (
        "reve-samples",
        1,
        200,
        2000,
    )
    assert a["units"] == "uV" and a["band_pass_hz"] == [0.5, 99.5]
    assert a["recipe"] == json.loads(
        (_parts(work, BETA)[0] / "recipe.json").read_text()
    )
    assert a["datasets"] == ["alpha-1.0", "beta-1.0"]
    assert a["local_datasets"] == ["alpha-1.0", "beta-1.0"]  # not for redistribution
    assert (a["n_buckets"], a["n_samples"]) == (3, 8) and "base" not in a
    assert a["buckets"] == [
        dict(
            bucket=0,
            dataset="alpha-1.0",
            shard=ALPHA[0],
            n_samples=2,
            n_sessions=1,
            n_chans=6,
        ),
        dict(
            bucket=1,
            dataset="alpha-1.0",
            shard=ALPHA[1],
            n_samples=2,
            n_sessions=1,
            n_chans=4,
        ),
        dict(
            bucket=2,
            dataset="beta-1.0",
            shard=BETA[0],
            n_samples=4,
            n_sessions=1,
            n_chans=6,
        ),
    ]
    for k, part in enumerate(_parts(work, ALPHA + BETA)):
        g, src = store[str(k)], zarr.open_group(part / "samples.zarr", mode="r")["0"]
        assert g.attrs["bucket"] == k and g.attrs["source_shard"] == part.name
        assert "licence" not in g.attrs
        for name in ("data", "session", "stats", "positions"):
            np.testing.assert_array_equal(g[name][:], src[name][:])
    buckets = (work / "stores" / "all-buckets.csv").read_text().splitlines()
    assert buckets[0] == "bucket,dataset,shard,n_samples,n_sessions,n_chans"
    assert buckets[3] == "2,beta-1.0,beta-1.0_000,4,1,6"
    manifest = (work / "stores" / "all-manifest.csv").read_text().splitlines()
    assert len(manifest) == 4 and manifest[0].startswith(
        "dataset_id,file,big_recording_index"
    )
    assert [line.split(",")[2] for line in manifest[1:]] == ["0", "1", "2"]
    assert (report.n_buckets, report.n_samples, report.n_base_buckets) == (3, 8, 0)
    assert report.new_datasets == report.datasets == ["alpha-1.0", "beta-1.0"]
    assert str(report).splitlines()[-1].startswith("merge: all: 3 buckets, 8 samples")
    assert report.to_dict()["store"] == str(work / "stores" / "all.zarr")


def test_licences_are_set_on_the_new_buckets(work):
    stale = work / "s" / "l.zarr.partial" / "0"  # left by an interrupted merge
    stale.mkdir(parents=True)
    merge_parts(
        _parts(work, ALPHA + BETA), work / "s", "l", licences={"alpha-1.0": "CC0-1.0"}
    )
    store = zarr.open_group(work / "s" / "l.zarr", mode="r")
    assert [store[k].attrs.get("licence") for k in "012"] == [
        "CC0-1.0",
        "CC0-1.0",
        None,
    ]
    assert store.attrs["licences"] == {"alpha-1.0": "CC0-1.0"}
    assert not stale.parent.exists()


# ------------------------------------------------------------------ extend


def test_a_store_is_extended_and_never_written_to(work):
    _store_one(work)
    one = work / "stores" / "one.zarr"
    before = _snapshot(one)
    report = merge_parts(_parts(work, BETA), work / "stores", "two", base=one)
    assert _snapshot(one) == before
    two = zarr.open_group(work / "stores" / "two.zarr", mode="r")
    first = zarr.open_group(one, mode="r")
    assert sorted(k for k, _ in two.groups()) == ["0", "1", "2"]
    for k in "01":  # the base's buckets: the same numbers, attributes and values
        assert dict(two[k].attrs) == dict(first[k].attrs)
        np.testing.assert_array_equal(two[k]["data"][:], first[k]["data"][:])
    beta = zarr.open_group(_parts(work, BETA)[0] / "samples.zarr", mode="r")["0"]
    np.testing.assert_array_equal(two["2"]["data"][:], beta["data"][:])
    assert two["2"].attrs["bucket"] == 2 and two["2"].attrs["source_shard"] == BETA[0]
    a = two.attrs
    assert a["base"] == {"name": "one", "n_buckets": 2, "n_samples": 4}
    assert a["buckets"][:2] == first.attrs["buckets"] and a["n_samples"] == 8
    assert a["datasets"] == ["alpha-1.0", "beta-1.0"]
    assert report.n_base_buckets == 2 and report.new_datasets == ["beta-1.0"]
    manifest = (work / "stores" / "two-manifest.csv").read_text().splitlines()
    assert (
        manifest[1:3]
        == (work / "stores" / "one-manifest.csv").read_text().splitlines()[1:]
    )
    assert manifest[3].split(",")[2] == "2"
    # hard links: one inode for a chunk of store two, of its part and of the base
    store_two, beta_group = (
        work / "stores" / "two.zarr",
        _parts(work, BETA)[0] / "samples.zarr" / "0",
    )
    new = _chunks(store_two / "2")[0]
    assert (
        os.stat(new).st_ino
        == os.stat(beta_group / new.relative_to(store_two / "2")).st_ino
    )
    kept = _chunks(store_two / "0")[0]
    assert (
        os.stat(kept).st_ino
        == os.stat(one / "0" / kept.relative_to(store_two / "0")).st_ino
    )
    assert (
        os.stat(store_two / "2" / "zarr.json").st_ino
        != os.stat(beta_group / "zarr.json").st_ino
    )


@pytest.mark.parametrize("link", ["symlink", "copy"])
def test_symbolic_links_and_copies_give_the_same_store(work, link):
    _store_one(work)
    one = work / "stores" / "one.zarr"
    before = _snapshot(one)
    merge_parts(_parts(work, BETA), work / "stores", "two", base=one, link=link)
    assert _snapshot(one) == before
    two = work / "stores" / "two.zarr"
    chunk = _chunks(two / "2")[0]
    if link == "symlink":
        assert chunk.is_symlink() and os.readlink(chunk) == str(
            (
                _parts(work, BETA)[0]
                / "samples.zarr"
                / "0"
                / chunk.relative_to(two / "2")
            ).resolve()
        )
    else:
        assert not chunk.is_symlink() and chunk.stat().st_nlink == 1
    assert not (two / "2" / "zarr.json").is_symlink()
    store, first = zarr.open_group(two, mode="r"), zarr.open_group(one, mode="r")
    beta = zarr.open_group(_parts(work, BETA)[0] / "samples.zarr", mode="r")["0"]
    np.testing.assert_array_equal(store["0"]["data"][:], first["0"]["data"][:])
    np.testing.assert_array_equal(store["2"]["data"][:], beta["data"][:])


def test_a_published_store_without_a_recipe_is_extended(work):
    pub = _published(work)
    before = _snapshot(pub.parent)
    merge_parts(
        _parts(work, BETA),
        work / "stores",
        "v0-plus",
        base=pub,
        licences={"beta-1.0": "CC-BY-4.0"},
    )
    assert _snapshot(pub.parent) == before
    a = zarr.open_group(work / "stores" / "v0-plus.zarr", mode="r").attrs
    assert a["base"]["name"] == "pub" and a["recipe"]["name"] == "reve"
    assert a["licence"] == "other" and a["licence_file"] == "LICENSES.csv"
    assert a["licences"] == {"alpha-1.0": "CC0-1.0", "beta-1.0": "CC-BY-4.0"}
    manifest = (work / "stores" / "v0-plus-manifest.csv").read_text().splitlines()
    assert len(manifest) == 4  # the published manifest's two rows first


@pytest.mark.parametrize(
    "field, value", [("band_pass_hz", [1.0, 99.5]), ("sfreq", 250)]
)
def test_a_published_store_of_another_recipe_is_refused(work, field, value):
    pub = _published(work)
    _edit_meta(pub / "zarr.json", lambda a: a.update({field: value}))
    with pytest.raises(ValueError, match=field):
        merge_parts(_parts(work, BETA), work / "stores", "x", base=pub)
    assert _nothing_left(work / "stores", "x")


def test_a_base_with_a_recipe_must_have_the_parts_recipe(work):
    _store_one(work)
    one = work / "stores" / "one.zarr"
    _edit_meta(one / "zarr.json", lambda a: a["recipe"].update(h_freq=45.0))
    with pytest.raises(ValueError, match="h_freq"):
        merge_parts(_parts(work, BETA), work / "stores", "x", base=one)
    assert _nothing_left(work / "stores", "x")


def test_a_base_bucket_missing_on_disk_is_refused(work):
    _store_one(work)
    one = work / "stores" / "one.zarr"
    shutil.rmtree(one / "1")
    with pytest.raises(ValueError, match="not on disk, first: 1"):
        merge_parts(_parts(work, BETA), work / "stores", "x", base=one)
    assert _nothing_left(work / "stores", "x")


# ---------------------------------------------------------------- refusals


def test_parts_of_different_recipes_are_refused(work):
    recipe = _parts(work, BETA)[0] / "recipe.json"
    recipe.write_text(json.dumps({**json.loads(recipe.read_text()), "h_freq": 45.0}))
    with pytest.raises(ValueError, match=r"part beta-1.0_000 .* h_freq differ"):
        merge_parts(_parts(work, ALPHA + BETA), work / "stores", "x")
    assert _nothing_left(work / "stores", "x")


def test_an_unfinished_part_is_refused(work):
    (_parts(work, ALPHA)[1] / "_DONE").unlink()
    with pytest.raises(ValueError, match=r"alpha-1.0_001: no _DONE"):
        merge_parts(_parts(work, ALPHA), work / "stores", "x")
    with pytest.raises(ValueError, match="at least one part"):
        merge_parts([], work / "stores", "x")
    assert _nothing_left(work / "stores", "x")


def test_a_dataset_already_in_the_base_is_refused(work):
    _store_one(work)
    with pytest.raises(ValueError, match=r"already in the base: \['alpha-1.0'\]"):
        merge_parts(
            _parts(work, ALPHA[1:] + BETA),
            work / "stores",
            "x",
            base=work / "stores" / "one.zarr",
        )
    assert _nothing_left(work / "stores", "x")


def test_a_benchmark_source_is_refused_in_the_parts_and_in_the_base(work):
    part = _parts(work, BETA)[0]
    pub = _published(work)
    _edit_meta(pub / "zarr.json", lambda a: a["datasets"].append("nm000139"))
    with pytest.raises(ValueError, match="benchmark source.*nm000139"):
        merge_parts([part], work / "stores", "x", base=pub)
    assert _nothing_left(work / "stores", "x")
    for meta in (
        part / "samples.zarr" / "zarr.json",
        part / "samples.zarr" / "0" / "zarr.json",
    ):
        _edit_meta(meta, lambda a: a.update(dataset="on004362"))  # PhysioNet-MI's twin
    with pytest.raises(ValueError, match="benchmark source.*on004362"):
        merge_parts([part], work / "stores", "x")
    assert _nothing_left(work / "stores", "x")


def test_an_existing_store_is_replaced_only_when_told(work):
    _store_one(work)
    one = work / "stores" / "one.zarr"
    before = _snapshot(one)
    with pytest.raises(ValueError, match="exists"):
        _store_one(work)
    assert _snapshot(one) == before and not list((work / "stores").glob("*.partial"))
    report = merge_parts(
        _parts(work, ALPHA + BETA), work / "stores", "one", overwrite=True
    )
    assert report.n_buckets == 3
    assert zarr.open_group(one, mode="r").attrs["n_buckets"] == 3


def test_a_store_inside_its_base_is_refused(work):
    _store_one(work)
    one = work / "stores" / "one.zarr"
    before = _snapshot(one)
    for out, name in ((one, "x"), (work / "stores", "one"), (one / "0", "x")):
        with pytest.raises(ValueError, match="outside each other"):
            merge_parts(_parts(work, BETA), out, name, base=one)
    inner = work / "outer.zarr" / "one.zarr"  # the base inside the new store
    shutil.copytree(one, inner)
    with pytest.raises(ValueError, match="outside each other"):
        merge_parts(_parts(work, BETA), work, "outer", base=inner, overwrite=True)
    assert _snapshot(one) == before and _snapshot(inner) == before


def test_an_unknown_link_mode_is_refused(work):
    with pytest.raises(ValueError, match="link must be one of"):
        merge_parts(_parts(work, BETA), work / "stores", "x", link="reflink")
    assert not (work / "stores").exists()


def test_a_damaged_part_is_refused(work):
    group = _parts(work, ALPHA)[0] / "samples.zarr" / "0" / "zarr.json"
    _edit_meta(group, lambda a: a["sessions"][0].update(n_samples=1))
    with pytest.raises(
        ValueError, match=r"alpha-1.0_000: bucket 0: 2 samples, 2 session ids, 1 per"
    ):
        merge_parts(_parts(work, ALPHA), work / "stores", "x")
    assert _nothing_left(work / "stores", "x")


def test_a_failed_hard_link_leaves_nothing_and_keeps_the_old_store(work, monkeypatch):
    _store_one(work)
    one = work / "stores" / "one.zarr"
    before = _snapshot(one)

    def cross_device(src, dst, **kwargs):
        raise OSError(18, "Cross-device link")

    monkeypatch.setattr(merge_mod.os, "link", cross_device)
    with pytest.raises(ValueError, match="link='symlink' or link='copy'"):
        merge_parts(_parts(work, BETA), work / "stores", "x")
    assert _nothing_left(work / "stores", "x")
    with pytest.raises(ValueError, match="hard links are not possible"):
        merge_parts(_parts(work, BETA), work / "stores", "one", overwrite=True)
    assert _snapshot(one) == before and not list((work / "stores").glob("*.partial"))


# -------------------------------------------------------- corpus and command


def test_corpus_merge_takes_its_parts_and_licences(work):
    corpus = EEGDashCorpus(work / "corpus.csv", cache_dir=work / "cache")
    report = corpus.merge(work / "out", "corpus", max_bytes=1)
    assert report.store == work / "out" / "corpus.zarr" and report.n_buckets == 3
    store = zarr.open_group(report.store, mode="r")
    assert [store[k].attrs["source_shard"] for k in "012"] == ALPHA + BETA
    assert [store[k].attrs.get("licence") for k in "012"] == [
        "CC0-1.0",
        "CC0-1.0",
        None,
    ]
    (work / "out" / "parts" / BETA[0] / "_DONE").unlink()
    with pytest.raises(ValueError, match="1 part.s. without _DONE.*beta-1.0_000"):
        corpus.merge(work / "out", "again", max_bytes=1)


def test_corpus_merge_writes_the_store_to_dest(work):
    """The store and its two side files go to dest; nothing new lands beside the parts."""
    corpus = EEGDashCorpus(work / "corpus.csv", cache_dir=work / "cache")
    before = sorted(p.name for p in (work / "out").iterdir())
    report = corpus.merge(work / "out", "apart", max_bytes=1, dest=work / "zarr")
    assert report.store == work / "zarr" / "apart.zarr" and report.n_buckets == 3
    assert sorted(p.name for p in (work / "zarr").iterdir()) == [
        "apart-buckets.csv",
        "apart-manifest.csv",
        "apart.zarr",
    ]
    assert sorted(p.name for p in (work / "out").iterdir()) == before


def test_cli_dest_takes_the_store(work, capsys):
    args = ["merge", str(work / "corpus.csv"), "--cache-dir", str(work / "cache")]
    args += ["--out", str(work / "out"), "--name", "cli-dest", "--max-gb", "1e-9"]
    assert corpus_mod.main([*args, "--dest", str(work / "cli-zarr")]) == 0
    assert "merge: cli-dest: 3 buckets" in capsys.readouterr().out
    assert (work / "cli-zarr" / "cli-dest.zarr").is_dir()
    assert not (work / "out" / "cli-dest.zarr").exists()


def test_cli_merges_and_reports_a_refusal(work, capsys):
    args = ["merge", str(work / "corpus.csv"), "--cache-dir", str(work / "cache")]
    args += ["--out", str(work / "out"), "--name", "cli", "--max-gb", "1e-9"]
    assert corpus_mod.main(args) == 0
    assert "merge: cli: 3 buckets, 8 samples" in capsys.readouterr().out
    assert corpus_mod.main(args) == 1
    assert capsys.readouterr().out.startswith("MERGE REFUSED: ")
    assert corpus_mod.main([*args, "--overwrite", "--link", "copy"]) == 0
    one = work / "out" / "cli.zarr"
    base = ["--base", str(one), "--name", "cli-two"]
    assert corpus_mod.main([*args[:-4], *base, "--max-gb", "1e-9"]) == 1
    assert "already in the base" in capsys.readouterr().out
