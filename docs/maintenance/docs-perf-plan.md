# EEGDash Docs Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> Plan lives in `docs/maintenance/` (not `docs/superpowers/plans/`) because `.gitignore` blocks `**/superpowers/`.

**Goal:** Cut the docs site's worst load costs — the 825-row summary table (CLS 0.91 / TBT 1070ms), eager 1.45MB Plotly, 1.49MB fontawesome.js on every page, and two ~7.4MB API pages — using the existing server API and native browser features, verified by Lighthouse on the PR preview.

**Architecture:** Four independent phases, each shippable alone. Phase 1 fixes the summary page (virtual-scroll table + lazy charts). Phase 2 drops fontawesome.js site-wide. Phase 3 shrinks the two orphan API dumps. Phase 4 (optional) moves the table's row data off the critical path to the server API. No new runtime dependencies; reuse the `lazy-embed.js` pattern already in the repo.

**Tech Stack:** Sphinx + pydata-sphinx-theme 0.16.1, sphinx-design tabs, DataTables 1.13.4 (+ Scroller 2.2.0), Plotly 3.1, pandas (`df.to_html`), the public `data.eegdash.org` HTTP API.

## Global Constraints

- **pydata-sphinx-theme is 0.16.1** (pinned only in `uv.lock:3057`, not `pyproject.toml`). Any `layout.html` override must be re-audited on theme upgrade.
- **Verify metrics on the PR surge preview**, not locally: a full `make html` is ~30–40 min and needs network + data; `make html-fast` (`EEGDASH_DOC_LIMIT=5`) is the local smoke build but only emits 5 table rows so it cannot reproduce the 825-row CLS. Real CLS/TBT numbers come from Lighthouse on the preview.
- **Lighthouse command** (desktop, used for every metric check):
  ```bash
  CHROME_PATH="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  npx --yes lighthouse "<url>" --quiet --output=json --output-path=lh.json \
    --only-categories=performance --preset=desktop \
    --chrome-flags="--headless=new --no-sandbox --disable-gpu"
  ```
- **Baselines (live, desktop):** summary Perf 39 / CLS 0.91 / TBT 1070ms / TTI 4.9s; `api/dataset/eegdash.dataset.dataset.html` = 7.43MB; `eegdash.dataset.html` = 7.07MB; `fontawesome.js` = 1.49MB raw on every page; home CLS 0.43.
- **Python edits:** ruff, line length 88, NumPy docstrings. **JS:** match existing `_static/js/*.js` style (IIFE, `'use strict'`).
- **Do not break URLs.** `prepare_summary_tables.py:497` links to `/api/dataset/eegdash.dataset.<ID>.html` (the per-dataset *Brief* pages, which must keep working); nothing links into the giant automodule dumps.

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `docs/source/_static/custom.css` | modify | Remove the inert content-visibility block; skeleton height already elsewhere |
| `docs/prepare_summary_tables.py` | modify | Add Scroller CDN + match skeleton height in `DATA_TABLE_TEMPLATE` |
| `docs/source/_static/js/dataset_table.js` | modify | DataTables init: `paging:false` → Scroller virtual scroll |
| `docs/source/dataset_summary.rst` | modify | Replace eager Plotly `<script>` with a capture-stub |
| `docs/source/_static/js/lazy-charts.js` | **create** | Load Plotly + replay chart calls on first chart-tab activation |
| `docs/source/conf.py` | modify | Register `lazy-charts.js` (deferred) |
| `docs/source/_templates/layout.html` | modify | Override theme `css` block to drop `fontawesome.js` |
| `docs/prune_apidoc.py` | **create** | Restrict the two giant `automodule` dumps to hand-written classes |
| `docs/Makefile` | modify | Run `prune_apidoc.py` after `sphinx-apidoc` |

---

## Phase 1 — Summary page (`dataset_summary.html`): Perf 39 → target ≥ 85

### Task 1.1: Remove the inert content-visibility CSS

**Why:** Measured on the live deploy — it produced no change (CLS 0.909, TBT 1110ms). Dead code; delete it.

**Files:**
- Modify: `docs/source/_static/custom.css` (the block added just after `#datasets-table { font-size: 0.95rem; }`)

- [ ] **Step 1: Delete the block**

Remove exactly:
```css

/* Perf: the table ships all ~800 rows (paging:false), which is the main
   driver of the summary page's main-thread work / TBT / CLS. content-visibility
   lets the browser skip layout + paint for rows outside the viewport;
   contain-intrinsic-size reserves ~one row's height so the scrollbar stays
   put. ponytail: native browser feature, no JS/deps.
   VERIFY on the PR preview: DataTables column alignment + CLS. If columns
   misalign (DataTables measures row widths), delete this block. */
#datasets-table tbody tr {
  content-visibility: auto;
  contain-intrinsic-size: auto 44px;
}
```

- [ ] **Step 2: Verify it's gone**

Run: `grep -n "content-visibility" docs/source/_static/custom.css`
Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add docs/source/_static/custom.css
git commit -m "perf(docs): drop inert content-visibility rule (measured no-op)"
```

### Task 1.2: Virtual-scroll the dataset table (DataTables Scroller)

**Why:** Root cause of CLS 0.91 = a fixed 420px skeleton swapping to a multi-thousand-px `paging:false` table; root cause of TBT/TTI = building 825 rows of DOM + DataTables init over all of them. DataTables **Scroller** renders only the ~30 visible rows into a fixed-height (`70vh`) viewport, so (a) the skeleton height can match the final height → CLS ≈ 0, and (b) only visible rows hit the DOM → TBT/TTI drop. Keeps the scroll-all UX (no pagination chrome).

**Files:**
- Modify: `docs/prepare_summary_tables.py:835-901` (`DATA_TABLE_TEMPLATE`)
- Modify: `docs/source/_static/js/dataset_table.js:103-105`

**Interfaces:**
- Produces: the built `_static/dataset_generated/dataset_summary_table.html` now loads `dataTables.scroller.min.js` and the init uses `scroller:true`.

- [ ] **Step 1: Add the Scroller CDN bundle to the template**

In `docs/prepare_summary_tables.py`, inside `DATA_TABLE_TEMPLATE`, after the SearchPanes line (`...searchpanes/2.3.1/js/dataTables.searchPanes.min.js"></script>`, ~line 848) add:
```html
<!-- Scroller (virtual scroll: render only visible rows) -->
<link rel="stylesheet" href="https://cdn.datatables.net/scroller/2.2.0/css/scroller.dataTables.min.css">
<script src="https://cdn.datatables.net/scroller/2.2.0/js/dataTables.scroller.min.js"></script>
```

- [ ] **Step 2: Match the skeleton height to the scroll viewport**

In the same file's `<style>` block, change the skeleton height so the pre-init placeholder equals the post-init `scrollY:'70vh'` table height (this is what removes CLS). Replace:
```css
    .dt-loading-skeleton {
        height: 420px;
```
with:
```css
    .dt-loading-skeleton {
        height: 70vh;
```
Update the neighbouring comment to say the skeleton height must track `scrollY` in `dataset_table.js`.

- [ ] **Step 3: Switch the DataTables init to Scroller**

In `docs/source/_static/js/dataset_table.js`, replace:
```javascript
    const dataTable = $table.DataTable({
        dom: 'Blfrtip',
        paging: false,
        searching: true,
        info: false,
```
with:
```javascript
    const dataTable = $table.DataTable({
        dom: 'Blfrtip',
        // Virtual scroll: only the ~30 visible rows are rendered into a
        // fixed 70vh viewport. deferRender is required by Scroller. The 70vh
        // MUST match .dt-loading-skeleton height in prepare_summary_tables.py
        // so the skeleton->table swap causes no layout shift (was CLS 0.91).
        scrollY: '70vh',
        scrollCollapse: true,
        scroller: true,
        deferRender: true,
        searching: true,
        info: false,
```

- [ ] **Step 4: Keep column widths correct after show/hide**

Confirm the existing handler (`dataset_table.js:257-259`) still runs; append a Scroller re-measure so revealed columns don't misalign in virtual-scroll mode. Change:
```javascript
    dataTable.on('column-visibility.dt', function () {
        dataTable.columns.adjust();
    });
```
to:
```javascript
    dataTable.on('column-visibility.dt', function () {
        dataTable.columns.adjust();
        if (dataTable.scroller) dataTable.scroller.measure();
    });
```

- [ ] **Step 5: Local smoke build (no metric, just "it builds + wires up")**

Run: `cd docs && make html-fast`
Then:
```bash
grep -c "dataTables.scroller.min.js" _build/html/_static/dataset_generated/dataset_summary_table.html
grep -c "scroller: true\|scroller:true" _build/html/_static/js/dataset_table.*.js
```
Expected: both ≥ 1. Open `_build/html/dataset_summary.html` in a browser; the table renders, scrolls, sort/filter/SearchPanes/ColVis work, no console errors.

- [ ] **Step 6: Commit**

```bash
git add docs/prepare_summary_tables.py docs/source/_static/js/dataset_table.js
git commit -m "perf(docs): virtual-scroll dataset table with DataTables Scroller"
```

- [ ] **Step 7: Metric verification (after push → PR preview)**

Lighthouse `<preview>/dataset_summary.html`. Expected vs baseline (CLS 0.91 / TBT 1070ms / TTI 4.9s): **CLS < 0.1, TBT < 300ms, TTI < 2.5s.** Also confirm column alignment visually. If columns misalign under Scroller, add `dataTable.columns.adjust()` in an `initComplete` callback and re-measure.

### Task 1.3: Lazy-load Plotly + defer chart init to tab activation

**Why:** `dataset_summary.rst:93` loads Plotly (1.45MB gz) eagerly, and all 8 tab charts call `Plotly.newPlot` on page load even though only the *table* tab is visible. Capture those calls in a stub queue; load real Plotly and replay only when the user opens a chart tab. Default view (table) then loads zero chart JS. Mirrors the existing `lazy-embed.js` pattern.

**Files:**
- Modify: `docs/source/dataset_summary.rst:89-93`
- Create: `docs/source/_static/js/lazy-charts.js`
- Modify: `docs/source/conf.py:170-179` (`html_js_files`)

**Interfaces:**
- Produces: a global `window.Plotly` stub + `window.__plotlyQueue` at parse time; `lazy-charts.js` consumes the queue after loading the real library.

- [ ] **Step 1: Replace the eager Plotly script with a capture-stub**

In `docs/source/dataset_summary.rst`, replace:
```rst
   .. raw:: html

      <script src="https://cdn.plot.ly/plotly-3.1.0.min.js"></script>
```
with:
```rst
   .. raw:: html

      <script>
      /* Defer Plotly (~1.45MB gz) until a chart tab is opened. The chart
         fragments call Plotly.newPlot at parse; capture them in a queue while
         Plotly is a stub, then lazy-charts.js loads the real library and
         replays them on first chart-tab activation. Default tab is the table
         (no chart) so Plotly never loads unless the user explores charts. */
      (function () {
        var q = (window.__plotlyQueue = []);
        window.Plotly = {
          newPlot: function () { q.push([].slice.call(arguments)); },
          __stub: true
        };
      })();
      </script>
```

- [ ] **Step 2: Create the loader**

Create `docs/source/_static/js/lazy-charts.js`:
```javascript
/* lazy-charts.js — load Plotly and render the summary-page charts only when
   the user opens a chart tab. Pairs with the capture-stub inlined at the top
   of dataset_summary.rst. See docs/maintenance/docs-perf-plan.md. */
(function () {
  'use strict';
  var PLOTLY_SRC = 'https://cdn.plot.ly/plotly-3.1.0.min.js';
  var state = 'idle'; // idle -> loading -> loaded

  function drain() {
    var q = window.__plotlyQueue || [];
    while (q.length) {
      var args = q.shift();
      try { window.Plotly.newPlot.apply(window.Plotly, args); } catch (e) { /* skip one bad chart */ }
    }
  }
  function loadPlotly() {
    if (state !== 'idle') return;
    state = 'loading';
    var s = document.createElement('script');
    s.src = PLOTLY_SRC;
    s.onload = function () { state = 'loaded'; drain(); };
    s.onerror = function () { state = 'idle'; };
    document.head.appendChild(s);
  }
  function bind() {
    var set = document.querySelector('.sd-tab-set');
    if (!set) return;
    // The first tab ("Dataset Table") needs no Plotly; any other does.
    set.addEventListener('change', function (ev) {
      var input = ev.target;
      if (!input || input.type !== 'radio') return;
      var label = set.querySelector('label[for="' + input.id + '"]');
      var name = label && label.textContent ? label.textContent.trim() : '';
      if (name && name !== 'Dataset Table') loadPlotly();
    });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', bind);
  else bind();
})();
```

- [ ] **Step 3: Register the loader (deferred)**

In `docs/source/conf.py`, add to `html_js_files` (after the `lazy-embed.js` entry, ~line 178):
```python
    # Lazy-load Plotly for the dataset-summary chart tabs (see
    # dataset_summary.rst capture-stub + _static/js/lazy-charts.js).
    ("js/lazy-charts.js", {"defer": "defer"}),
```

- [ ] **Step 4: Local smoke build**

Run: `cd docs && make html-fast`
Then load `_build/html/dataset_summary.html`: on the default (table) tab, DevTools Network shows **no** `plot.ly` request; click "Dataset Treemap" → Plotly loads once and the chart renders; other chart tabs render without reloading Plotly. No console errors on the table tab.
```bash
grep -c "plot.ly/plotly" _build/html/dataset_summary.html   # expect 0 (eager script gone)
grep -c "__plotlyQueue" _build/html/dataset_summary.html      # expect >=1 (stub present)
```

- [ ] **Step 5: Commit**

```bash
git add docs/source/dataset_summary.rst docs/source/_static/js/lazy-charts.js docs/source/conf.py
git commit -m "perf(docs): lazy-load Plotly + defer summary charts to tab activation"
```

- [ ] **Step 6: Metric verification (PR preview)**

Lighthouse `<preview>/dataset_summary.html`: total transferred bytes and TBT drop further vs Task 1.2 (Plotly no longer in the initial load). Confirm each chart tab still renders.

> **Note (moabb tab):** the "Subject Distribution" chart uses d3 (already `defer`), not Plotly — it's unaffected by this stub. Verify it still renders.

---

## Phase 2 — Every page: drop `fontawesome.js` (~1.49MB raw / ~540KB)

### Task 2.1: Override the theme `css` block to skip the fontawesome preload

**Why:** pydata-sphinx-theme 0.16.1 emits `<script src="_static/scripts/fontawesome.js">` unconditionally from the `head_js_preload()` macro (no config flag). The theme *also* bundles a CSS/woff2 fallback (`fa-solid-900.woff2`, 156KB) for every glyph the site uses, so dropping the JS keeps the icons. eegdash already overrides theme blocks in this file, so this follows precedent.

**Files:**
- Modify: `docs/source/_templates/layout.html`

**Interfaces:**
- Consumes: theme macro `head_pre_assets()` and Sphinx's `css()` (both reproduced verbatim from the theme's `css` block minus the fontawesome line).

- [ ] **Step 1: Import the theme's webpack macros**

In `docs/source/_templates/layout.html`, immediately after line 1 (`{% extends "!layout.html" %}`) add (path must match the theme's own import at `<theme>/layout.html:8`):
```jinja
{# Needed to reproduce the theme's `css` block below (minus fontawesome.js). #}
{%- import "static/webpack-macros.html" as _webpack with context %}
```

- [ ] **Step 2: Override the `css` block without the fontawesome preload**

Add this block to `docs/source/_templates/layout.html` (next to the existing `extrahead` override). It is the theme's own `css` block (0.16.1) verbatim, minus the final `{{ _webpack.head_js_preload() }}` line:
```jinja
{# Reproduce pydata-sphinx-theme 0.16.1's `css` block but omit
   head_js_preload() — it unconditionally loads _static/scripts/fontawesome.js
   (~1.49MB) on every page. Every icon we use has a CSS/woff2 fallback in the
   theme's already-loaded stylesheet, so icons still render. RE-AUDIT on theme
   upgrade — see docs/maintenance/docs-perf-plan.md. #}
{% block css %}
  <script data-cfasync="false">
    document.documentElement.dataset.mode = localStorage.getItem("mode") || "{{ default_mode }}";
    document.documentElement.dataset.theme = localStorage.getItem("theme") || "{{ default_mode }}";
  </script>
  <noscript>
    <style>
      .pst-js-only { display: none !important; }
    </style>
  </noscript>
  {{ _webpack.head_pre_assets() }}
  {{- css() }}
{% endblock css %}
```

- [ ] **Step 3: Build and verify the script is gone but icons remain**

Run: `cd docs && make html-fast`
```bash
grep -rl "scripts/fontawesome.js" _build/html/index.html   # expect: no match
```
Open `_build/html/index.html`: verify the theme-switcher (sun/moon), search icon, back-to-top arrow, and prev/next arrows all still render (they now come from the woff2 fallback). Check DevTools console for no missing-glyph errors.

- [ ] **Step 4: Commit**

```bash
git add docs/source/_templates/layout.html
git commit -m "perf(docs): stop loading 1.49MB fontawesome.js on every page"
```

- [ ] **Step 5: Metric verification (PR preview)**

Lighthouse any page (e.g. home): total byte weight drops ~1.49MB raw; "Reduce unused JavaScript" no longer lists fontawesome. Visually confirm all icons across nav/footer/admonitions.

> **Fallback if a glyph is missing:** re-add only that icon as an inline SVG, or upgrade to pydata-sphinx-theme 0.19.0 (adds `defer` to the script — non-blocking but keeps the payload).

---

## Phase 3 — API reference: shrink the two 7.4MB `automodule` dumps

### Task 3.1: Restrict the giant automodule pages to hand-written classes

**Why:** `sphinx-apidoc` emits `.. automodule:: eegdash.dataset[.dataset] :members:`. Because `register_openneuro_datasets` stamps `__module__="eegdash.dataset.dataset"` on ~753 dynamic dataset classes and adds them to `__all__`, autodoc dumps all of them — twice (once per module page) — producing `eegdash.dataset.dataset.html` (7.43MB) and `eegdash.dataset.html` (7.07MB). The real per-dataset docs are the *Brief* pages (`eegdash.dataset.DS*.html`, ~225–358KB), already generated and linked. Restrict the two dumps to the hand-written classes (`EEGDashDataset`, `EEGChallengeDataset`; plus `EEGDashRaw` at the package level). These pages are near-orphans (`:noindex:`, no inbound links to their anchors), so URL/SEO risk is low. (SEO_AUDIT_LOG.md already flags them as the unidentified >2MB orphans.)

**Files:**
- Create: `docs/prune_apidoc.py`
- Modify: `docs/Makefile:18-22` (`apidoc` target)

- [ ] **Step 1: Create the prune script**

Create `docs/prune_apidoc.py`:
```python
#!/usr/bin/env python3
"""Shrink the two giant auto-generated dataset API pages.

`sphinx-apidoc` writes ``.. automodule:: eegdash.dataset[.dataset] :members:``,
which dumps the ~753 dynamically-registered per-dataset classes (they set
``__module__="eegdash.dataset.dataset"`` and land in ``__all__``), producing two
~7MB near-orphan pages. The canonical per-dataset docs are the Dataset Brief
pages (``eegdash.dataset.DS*.html``), already linked from the ``source_*.rst``
toctrees. Restrict these two ``automodule`` blocks to the hand-written classes.

Runs after ``sphinx-apidoc`` in the Makefile ``apidoc`` target.
"""

from pathlib import Path

API_DIR = Path(__file__).parent / "source" / "api" / "dataset"

# Exact ``:members:`` allow-lists, keyed by generated file name.
PATCHES = {
    "eegdash.dataset.dataset.rst": ":members: EEGDashDataset, EEGChallengeDataset",
    "eegdash.dataset.rst": ":members: EEGDashDataset, EEGChallengeDataset, EEGDashRaw",
}


def main() -> None:
    for name, members_line in PATCHES.items():
        path = API_DIR / name
        if not path.exists():
            print(f"prune_apidoc: {name} not found (skipped)")
            continue
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        patched = []
        changed = False
        for line in lines:
            if line.strip() == ":members:":
                indent = line[: len(line) - len(line.lstrip())]
                patched.append(f"{indent}{members_line}\n")
                changed = True
            else:
                patched.append(line)
        path.write_text("".join(patched), encoding="utf-8")
        print(f"prune_apidoc: {'patched' if changed else 'no :members: line in'} {name}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Wire it into the Makefile**

In `docs/Makefile`, the `apidoc` target currently ends with the `python -m sphinx.ext.apidoc ...` line. Append a prune step so it becomes:
```makefile
apidoc:
	# Generate full API docs, then prune duplicates covered by autosummary
	@rm -f "$(APIDIR)"/dataset/eegdash.features*
	@SPHINX_APIDOC_OPTIONS=members,undoc-members,show-inheritance,noindex \
		python -m sphinx.ext.apidoc -f -e -T -o "$(APIDIR)/dataset" "../$(PKG)" "../$(PKG)/features"
	# Restrict the two giant dataset automodule dumps to hand-written classes
	# (the ~753 dynamic dataset classes are documented on the Brief pages).
	@python prune_apidoc.py
```

- [ ] **Step 3: Verify the prune ran and pages shrank**

Run: `cd docs && make apidoc`
```bash
grep -n ":members:" source/api/dataset/eegdash.dataset.dataset.rst
```
Expected: `:members: EEGDashDataset, EEGChallengeDataset` (not a bare `:members:`).

Then build the API pages and measure (a full or `html-fast` build; the API pages don't depend on dataset count):
```bash
cd docs && make html-fast
du -h _build/html/api/dataset/eegdash.dataset.dataset.html _build/html/api/dataset/eegdash.dataset.html
```
Expected: each well under **300KB** (from 7.4MB). Spot-check a Brief page still exists: `ls _build/html/api/dataset/eegdash.dataset.DS000117.html`.

- [ ] **Step 4: Confirm no broken references**

Run: `cd docs && make html-fast 2>&1 | grep -iE "eegdash.dataset.*(WARNING|reference target not found)"`
Expected: no new warnings about `eegdash.dataset` targets. The module pages still document the hand-written classes; per-dataset classes remain on their Brief pages.

- [ ] **Step 5: Commit**

```bash
git add docs/prune_apidoc.py docs/Makefile
git commit -m "perf(docs): stop automodule from dumping 753 dataset classes (7.4MB -> <300KB)"
```

---

## Phase 4 (OPTIONAL) — Server-offloaded table via `chart-data`

Only do this if Phase 1 isn't enough or you want the ~1.3MB table HTML out of the page. It trades a smaller/faster page for a runtime dependency on `data.eegdash.org`.

**Why / feasibility (verified):** `GET https://data.eegdash.org/api/eegdash/datasets/chart-data?include=rows` returns all 824 rows (~705KB, gzips smaller) and is public + CORS-open (no server change). `dataset-explorer.js` already fetches this API off the critical path via `requestIdleCallback`. Caveat: `chart-data` has `limit` but no `skip`, so this is a fetch-all-once client-side table (SearchPanes still works over all rows), not true server-side paging.

**Approach (single task, larger):**
- Modify `docs/prepare_summary_tables.py` to emit a small **shell** (`<table id="datasets-table">` with `<thead>` only, no `<tbody>` rows) plus a `<script type="application/json" id="datasets-columns">` describing column order/renderers.
- New `docs/source/_static/js/dataset_table_remote.js`: on `requestIdleCallback`, `fetch(<api>/datasets/chart-data?include=rows)`, map `row` objects → DataTables `data`/`columns`, init with `deferRender:true` + Scroller (Task 1.2 config). Reuse the sparkbar/tag render logic from `dataset_table.js`.
- **Stale-cache fallback** (mirror `DatasetSnapshot` live→cache→CSV): keep a build-time-generated `datasets.json` in `_static/` and fall back to it if the fetch fails, so the page survives API downtime.
- Keep the build-time `df` path as the source for that fallback JSON.

**Verify:** page HTML no longer contains 800+ `<tr>`; Lighthouse total bytes drop; table still renders when offline (fallback JSON) — test by blocking `data.eegdash.org` in DevTools.

**Decision gate:** confirm the `chart-data` rate limit with the API owner (not published in `/.well-known/api-catalog`) before shipping a per-view fetch.

---

## Phase 5 (lower priority) — Home CLS 0.43

### Task 5.1: Fix the home page's largest layout shift

**Why:** Home is Perf 79 but CLS 0.43. Existing hero `min-height`s (`custom.css:3045-3051`) aren't enough; the residual shift is almost certainly an image/logo without reserved dimensions.

- [ ] **Step 1: Identify the shifting element from Lighthouse**

From a home-page Lighthouse JSON:
```bash
python3 -c "import json;d=json.load(open('lh.json'));[print(round(i.get('score',0),3), i['node'].get('selector','')) for i in d['audits']['layout-shift-elements']['details']['items']]"
```
This prints the exact selectors that shift.

- [ ] **Step 2: Reserve space for that element**

For each shifting `<img>`, set explicit `width`/`height` attributes (in the source `.rst`/template) or `aspect-ratio` + fixed width in `custom.css`. For a shifting text/hero block, extend the existing `body[data-page="index"] .hf-*` `min-height` rules to cover it.

- [ ] **Step 3: Verify + commit**

Lighthouse home: CLS < 0.1. Commit `perf(docs): reserve space for home hero to fix CLS`.

---

## Verify loop (all phases)

1. Implement a task on the `perf/docs-no-regret` branch (or a fresh `perf/docs-phaseN`).
2. `cd docs && make html-fast` for the functional smoke + `grep`/`du` assertions in each task.
3. Push → the `docs` workflow builds the PR preview.
4. Lighthouse the affected page(s) on the preview; compare to the baselines table.
5. Merge when green and metrics improve; revert the single task if a regression appears.

## Self-Review

- **Coverage:** summary CLS/TBT (1.2), eager Plotly (1.3), fontawesome (2.1), 7.4MB API pages (3.1), inert CSS cleanup (1.1), server option (4), home CLS (5.1) — all measured problems have a task.
- **URL safety:** Phase 3 leaves the Brief-page URLs (`eegdash.dataset.<ID>.html`) untouched; only the orphan dumps shrink.
- **No new runtime deps:** Scroller is a DataTables plugin from the same CDN already in use; Plotly/d3 unchanged; fontawesome uses the theme's bundled woff2.
- **Consistency:** `scrollY:'70vh'` in `dataset_table.js` must equal `.dt-loading-skeleton` height in `prepare_summary_tables.py` (called out in both edits).
- **Risk order:** ship 1.1 → 1.2 → 1.3 → 2.1 → 3.1 (each independently revertible); 4 and 5.1 are optional/last.
</content>
</invoke>
