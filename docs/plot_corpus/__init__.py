"""EEGDash Corpus pages: offline renderer of the corpus tables into HTML fragments.

The data in ``plot_corpus/data/`` is a generated copy (see ``sync_data.py`` and
``data/README.md``); ``docs/prepare_corpus.py`` turns it into the fragments,
``numbers.inc`` and ``manifest.json`` that ``docs/source/corpus/*.rst`` embed
through the ``corpus-figure`` directive. Nothing here touches the network.
"""
