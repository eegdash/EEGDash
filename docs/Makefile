# Minimal makefile for Sphinx documentation
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build
PKG ?= eegdash
APIDIR := $(SOURCEDIR)/api

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: apidoc
apidoc:
	# Generate full API docs, then prune duplicates covered by autosummary
	@SPHINX_APIDOC_OPTIONS=members,undoc-members,show-inheritance,noindex \
		python -m sphinx.ext.apidoc -f -e -M -o "$(APIDIR)/dataset" "../$(PKG)"
	# Remove top-level package page and modules covered elsewhere


# Standard build runs examples
html: apidoc 

# Fast build: do NOT execute examples (sphinx-gallery)
.PHONY: html-noplot
html-noplot: apidoc 
	@python prepare_summary_tables.py ../eegdash/ $(BUILDDIR)
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" \
		$(SPHINXOPTS) -D sphinx_gallery_conf.plot_gallery=0 $(O)

# Custom clean target to remove generated API docs and build files
.PHONY: clean
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
	@rm -rf "$(BUILDDIR)"
	@echo "Clean completed."

.PHONY: help apidoc
Makefile: ;          

%: Makefile
	@python prepare_summary_tables.py ../eegdash/ $(BUILDDIR)
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
