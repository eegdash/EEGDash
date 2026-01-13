# Minimal makefile for Sphinx documentation
#

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = _build
PKG ?= eegdash
APIDIR := $(SOURCEDIR)/api

.PHONY: help clean apidoc html html-noplot html-fast

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

apidoc:
	# Generate full API docs, then prune duplicates covered by autosummary
	@rm -f "$(APIDIR)"/dataset/eegdash.features*
	@SPHINX_APIDOC_OPTIONS=members,undoc-members,show-inheritance,noindex \
		python -m sphinx.ext.apidoc -f -e -T -o "$(APIDIR)/dataset" "../$(PKG)" "../$(PKG)/features"


# Standard build runs examples
html: apidoc

# Fast build: do NOT execute examples (sphinx-gallery)
html-noplot: apidoc
	@python prepare_summary_tables.py --target $(BUILDDIR)
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" \
		$(SPHINXOPTS) -D sphinx_gallery_conf.plot_gallery=0 $(O)

# Very fast build: limit datasets to 5
html-fast: apidoc
	@EEGDASH_DOC_LIMIT=5 python prepare_summary_tables.py --target $(BUILDDIR)
	@EEGDASH_DOC_LIMIT=5 $(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" \
		$(SPHINXOPTS) -D sphinx_gallery_conf.plot_gallery=0 $(O)

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
	@echo "Removing build directory..."
	@rm -rf "$(BUILDDIR)" build
	@echo "Clean completed."

# Catch-all target for other sphinx-build formats
%:
	@python prepare_summary_tables.py --target $(BUILDDIR)
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
