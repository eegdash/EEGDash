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
	# Generate API docs using the top-level package so modules are importable
	# as eegdash.* instead of an unimportable bare 'dataset.*'
	@python -m sphinx.ext.apidoc -o "$(APIDIR)/dataset" "../$(PKG)" -f -e -M

# Standard build runs examples
html: apidoc

# Fast build: do NOT execute examples (sphinx-gallery)
.PHONY: html-noplot
html-noplot: apidoc
	@python prepare_summary_tables.py ../eegdash/ $(BUILDDIR)
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" \
		$(SPHINXOPTS) -D sphinx_gallery_conf.plot_gallery=0 $(O)

.PHONY: help apidoc
Makefile: ;          

%: Makefile
	@python prepare_summary_tables.py ../eegdash/ $(BUILDDIR)
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
