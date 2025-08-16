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
	@sphinx-apidoc -o "$(APIDIR)" "../$(PKG)" -f -e -M

# Standard build runs examples
html: apidoc

# Fast build: do NOT execute examples (sphinx-gallery)
.PHONY: html-noplot
html-noplot: apidoc
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" \
		$(SPHINXOPTS) -D sphinx_gallery_conf.plot_gallery=0 $(O)

.PHONY: help apidoc
Makefile: ;          

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)