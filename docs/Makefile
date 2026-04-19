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

.PHONY: help clean apidoc html html-noplot html-fast markdown

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

apidoc:
	# Generate full API docs, then prune duplicates covered by autosummary
	@rm -f "$(APIDIR)"/dataset/eegdash.features*
	@SPHINX_APIDOC_OPTIONS=members,undoc-members,show-inheritance,noindex \
		python -m sphinx.ext.apidoc -f -e -T -o "$(APIDIR)/dataset" "../$(PKG)" "../$(PKG)/features"


# Standard build runs examples
html: apidoc
	EEGDASH_CACHE_DIR=$(EEGDASH_CACHE_DIR) python prepare_summary_tables.py --target $(BUILDDIR)
	EEGDASH_DATASET_REGISTRY_FROM_API=1 EEGDASH_CACHE_DIR=$(EEGDASH_CACHE_DIR) $(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Fast build: do NOT execute examples (sphinx-gallery)
html-noplot: apidoc
	@EEGDASH_CACHE_DIR=$(EEGDASH_CACHE_DIR) python prepare_summary_tables.py --target $(BUILDDIR)
	@EEGDASH_DATASET_REGISTRY_FROM_API=1 EEGDASH_CACHE_DIR=$(EEGDASH_CACHE_DIR) $(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" \
		$(SPHINXOPTS) -D sphinx_gallery_conf.plot_gallery=0 $(O)

# Very fast build: limit datasets to 5
html-fast: apidoc
	@EEGDASH_DOC_LIMIT=5 EEGDASH_CACHE_DIR=$(EEGDASH_CACHE_DIR) python prepare_summary_tables.py --target $(BUILDDIR)
	@EEGDASH_DOC_LIMIT=5 EEGDASH_DATASET_REGISTRY_FROM_API=1 EEGDASH_CACHE_DIR=$(EEGDASH_CACHE_DIR) $(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" \
		$(SPHINXOPTS) -D sphinx_gallery_conf.plot_gallery=0 $(O)

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
