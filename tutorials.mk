# tutorials.mk -- Make targets for the EEGDash tutorial audit pipeline.
#
# Usage (no top-level Makefile exists yet, so always pass -f explicitly):
#
#   make -f tutorials.mk tutorial-help
#   make -f tutorials.mk tutorial-audit TUTORIAL=plot_11_leakage_safe_split
#   make -f tutorials.mk tutorial-claim TUTORIAL=plot_11_leakage_safe_split BY=author-A
#   make -f tutorials.mk tutorial-baseline
#   make -f tutorials.mk tutorial-release TUTORIAL=plot_11_leakage_safe_split
#   make -f tutorials.mk tutorial-dossier TUTORIAL=plot_11_leakage_safe_split
#   make -f tutorials.mk tutorial-phase-report PHASE=2
#
# All targets shell out to scripts/tutorial_audit/*. State changes to the
# spec YAML files go through a Python one-liner that uses `pyyaml`, never
# through `sed`/`awk`, because YAML's whitespace contract is too easy to
# break with line-oriented tools.

PYTHON ?= python
SPEC_DIR := docs/tutorials/_spec
EVIDENCE_DIR := docs/evidence/tutorials
TUTORIAL_GLOB := examples/tutorials/**/plot_*.py
DATE := $(shell date +%Y-%m-%d)

.PHONY: tutorial-help tutorial-audit tutorial-baseline tutorial-claim \
        tutorial-release tutorial-dossier tutorial-phase-report

tutorial-help:
	@echo "EEGDash tutorial audit -- Make targets"
	@echo ""
	@echo "  tutorial-help          Show this listing."
	@echo "  tutorial-audit         Run static stage on TUTORIAL=<id> or all if blank."
	@echo "  tutorial-baseline      Run static + runtime audit on every plot_*.py and"
	@echo "                         write docs/evidence/tutorials/_baseline_<date>/."
	@echo "  tutorial-claim         Set assignee + state=drafted on TUTORIAL=<id> by BY=<who>."
	@echo "  tutorial-release       Advance TUTORIAL=<id> from state=reviewed to state=merged."
	@echo "  tutorial-dossier       Render report.md from evidence.json for TUTORIAL=<id>."
	@echo "  tutorial-phase-report  Aggregate diff vs _baseline_*/ for PHASE=<n>."
	@echo ""
	@echo "Variables:"
	@echo "  TUTORIAL    Tutorial id matching docs/tutorials/_spec/<id>.yaml"
	@echo "  BY          Author handle written to spec.assignee"
	@echo "  PHASE       Migration phase number from tutorial_restructure_plan.md"

# tutorial-audit: static stage on TUTORIAL=<id> or every plot_*.py if blank.
tutorial-audit:
	@if [ -z "$(TUTORIAL)" ]; then \
		echo "Running static audit across $(TUTORIAL_GLOB)"; \
		$(PYTHON) -m scripts.tutorial_audit.pipeline --stage static --pattern "$(TUTORIAL_GLOB)"; \
	else \
		echo "Running static audit for $(TUTORIAL)"; \
		$(PYTHON) -m scripts.tutorial_audit.pipeline --stage static --tutorial "$(TUTORIAL)"; \
	fi

# tutorial-baseline: full static + runtime sweep, snapshot to dated dir.
tutorial-baseline:
	@mkdir -p "$(EVIDENCE_DIR)/_baseline_$(DATE)"
	$(PYTHON) -m scripts.tutorial_audit.pipeline \
		--stage static \
		--pattern "$(TUTORIAL_GLOB)" \
		--out "$(EVIDENCE_DIR)/_baseline_$(DATE)"
	$(PYTHON) -m scripts.tutorial_audit.pipeline \
		--stage runtime \
		--pattern "$(TUTORIAL_GLOB)" \
		--out "$(EVIDENCE_DIR)/_baseline_$(DATE)"
	@echo "Baseline written to $(EVIDENCE_DIR)/_baseline_$(DATE)"

# tutorial-claim: edit spec YAML in place via pyyaml. Requires TUTORIAL and BY.
tutorial-claim:
	@if [ -z "$(TUTORIAL)" ] || [ -z "$(BY)" ]; then \
		echo "tutorial-claim requires TUTORIAL=<id> and BY=<author-handle>"; exit 2; \
	fi
	$(PYTHON) -c "import sys, yaml; \
p = '$(SPEC_DIR)/$(TUTORIAL).yaml'; \
data = yaml.safe_load(open(p)); \
state = data.get('state', 'proposed'); \
assert state in ('proposed', 'drafted'), f'cannot claim from state={state}'; \
data['assignee'] = '$(BY)'; \
data['state'] = 'drafted'; \
yaml.safe_dump(data, open(p, 'w'), sort_keys=False); \
print(f'claimed {p} for $(BY); state -> drafted')"

# tutorial-release: only allowed when state==reviewed; sets state=merged.
tutorial-release:
	@if [ -z "$(TUTORIAL)" ]; then \
		echo "tutorial-release requires TUTORIAL=<id>"; exit 2; \
	fi
	$(PYTHON) -c "import sys, yaml; \
p = '$(SPEC_DIR)/$(TUTORIAL).yaml'; \
data = yaml.safe_load(open(p)); \
state = data.get('state'); \
assert state == 'reviewed', f'tutorial-release requires state=reviewed, got {state!r}'; \
data['state'] = 'merged'; \
yaml.safe_dump(data, open(p, 'w'), sort_keys=False); \
print(f'released {p}; state -> merged')"

# tutorial-dossier: render report.md from evidence.json.
tutorial-dossier:
	@if [ -z "$(TUTORIAL)" ]; then \
		echo "tutorial-dossier requires TUTORIAL=<id>"; exit 2; \
	fi
	$(PYTHON) -m scripts.tutorial_audit.report \
		--tutorial "$(TUTORIAL)" \
		--render-md \
		--out "$(EVIDENCE_DIR)/$(TUTORIAL)/report.md"

# tutorial-phase-report: aggregate diff vs _baseline_*/ for PHASE=<n>.
tutorial-phase-report:
	@if [ -z "$(PHASE)" ]; then \
		echo "tutorial-phase-report requires PHASE=<n>"; exit 2; \
	fi
	$(PYTHON) -m scripts.tutorial_audit.report \
		--aggregate \
		--phase "$(PHASE)" \
		--baseline-glob "$(EVIDENCE_DIR)/_baseline_*"
