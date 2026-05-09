# Minimal makefile for Sphinx documentation
#

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = _build
PKG ?= eegdash
APIDIR := $(SOURCEDIR)/api
# Shared cache so tutorials find data regardless of sphinx CWD (mirrors CI's EEGDASH_CACHE_DIR).
EEGDASH_CACHE_DIR ?= $(CURDIR)/../.eegdash_cache

.PHONY: help clean apidoc llms-txt html html-noplot html-fast markdown electrode-layouts traces-recordings

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

apidoc:
	# Generate full API docs, then prune duplicates covered by autosummary
	@rm -f "$(APIDIR)"/dataset/eegdash.features*
	@SPHINX_APIDOC_OPTIONS=members,undoc-members,show-inheritance,noindex \
		python -m sphinx.ext.apidoc -f -e -T -o "$(APIDIR)/dataset" "../$(PKG)" "../$(PKG)/features"

# Regenerate $(SOURCEDIR)/_extra/llms.txt from the current source tree.
# Runs after `apidoc` so the per-dataset pages that drive the sitemap
# coverage metric are already materialised.
llms-txt: apidoc
	@python generate_llms_txt.py --source "$(SOURCEDIR)" --output "$(SOURCEDIR)/_extra/llms.txt"

# Refresh source/_static/dataset_generated/electrode-layouts.json from the
# live montage registry. The script fails soft (zero layouts → warning
# but exit 0) so a transient API outage never blocks the docs build;
# the --keep-existing-on-empty flag preserves the last good manifest.
electrode-layouts:
	@python build_electrode_layouts.py --keep-existing-on-empty || true

# Refresh source/_static/dataset_generated/traces-viewer-recordings.json —
# maps dataset_id → representative recording URL params for the
# eegdash-viewer iframe embedded on each dataset page. Hand-curated
# manifest today; future work: query the eegdash recording index.
traces-recordings:
	@python build_traces_recordings.py

# Standard build runs examples
html: apidoc llms-txt electrode-layouts traces-recordings
	EEGDASH_CACHE_DIR=$(EEGDASH_CACHE_DIR) python prepare_summary_tables.py --target $(BUILDDIR)
	EEGDASH_DATASET_REGISTRY_FROM_API=1 EEGDASH_CACHE_DIR=$(EEGDASH_CACHE_DIR) $(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@python fingerprint_assets.py --html-dir "$(BUILDDIR)/html"

# Fast build: do NOT execute examples (sphinx-gallery)
html-noplot: apidoc llms-txt electrode-layouts traces-recordings
	@EEGDASH_CACHE_DIR=$(EEGDASH_CACHE_DIR) python prepare_summary_tables.py --target $(BUILDDIR)
	@EEGDASH_DATASET_REGISTRY_FROM_API=1 EEGDASH_CACHE_DIR=$(EEGDASH_CACHE_DIR) $(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" \
		$(SPHINXOPTS) -D sphinx_gallery_conf.plot_gallery=0 $(O)
	@python fingerprint_assets.py --html-dir "$(BUILDDIR)/html"

# Very fast build: limit datasets to 5 (skip the electrode registry leg;
# manifest stays at whatever was last generated)
html-fast: apidoc llms-txt
	@EEGDASH_DOC_LIMIT=5 EEGDASH_CACHE_DIR=$(EEGDASH_CACHE_DIR) python prepare_summary_tables.py --target $(BUILDDIR)
	@EEGDASH_DOC_LIMIT=5 EEGDASH_DATASET_REGISTRY_FROM_API=1 EEGDASH_CACHE_DIR=$(EEGDASH_CACHE_DIR) $(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" \
		$(SPHINXOPTS) -D sphinx_gallery_conf.plot_gallery=0 $(O)
	@python fingerprint_assets.py --html-dir "$(BUILDDIR)/html"

# Markdown sibling build. Run *after* `make html` — we reuse the already
# generated apidoc/summary tables and only build the markdown tree, then
# mirror .md files into the html output and emit `llms-full.txt`. Gallery
# execution is disabled here: the markdown corpus is for LLM/agent
# consumption, and re-running the examples would double the build time
# with no textual gain.
markdown:
	@EEGDASH_DATASET_REGISTRY_FROM_API=1 EEGDASH_CACHE_DIR=$(EEGDASH_CACHE_DIR) $(SPHINXBUILD) -M markdown "$(SOURCEDIR)" "$(BUILDDIR)" \
		$(SPHINXOPTS) -D sphinx_gallery_conf.plot_gallery=0 $(O) || true
	@python build_markdown_assets.py --source "$(BUILDDIR)/markdown" --target "$(BUILDDIR)/html"

# Custom clean target to remove generated API docs and build files
clean:
	@echo "Removing generated API documentation..."
	@rm -rf "$(APIDIR)/dataset"
	@rm -rf "$(APIDIR)/generated"
	@echo "Removing generated dataset pages..."
	@rm -rf "$(APIDIR)/datasets"
	@rm -f "$(APIDIR)/api_dataset.rst"
	@echo "Removing other generated directories..."
	@rm -rf "$(SOURCEDIR)/generated"
	@rm -rf "$(SOURCEDIR)/gen_modules"
	@echo "Removing build directory (including CSV cache)..."
	@rm -rf "$(BUILDDIR)" build
	@echo "Clean completed."

# Catch-all target for other sphinx-build formats
# Prevent Makefile from matching the catch-all rule
Makefile: ;

%: Makefile
	EEGDASH_CACHE_DIR=$(EEGDASH_CACHE_DIR) python prepare_summary_tables.py --target $(BUILDDIR)
	EEGDASH_DATASET_REGISTRY_FROM_API=1 EEGDASH_CACHE_DIR=$(EEGDASH_CACHE_DIR) $(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
