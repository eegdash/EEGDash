# Docs site performance — audit + fix plan

Live Lighthouse audit of `eegdash.org` (desktop preset). The measured build is
**current** — the latest `gh-pages` deploy commit is `deploy: 9fb3fcb32…`, which
is `develop` HEAD, so these numbers reflect the code in the repo now.

## Measured (live)

| Page | Perf | LCP | TBT | CLS | TTI | Notes |
|---|---|---|---|---|---|---|
| Home | 79 | 0.9s | 0ms | **0.43** | 0.9s | hero layout shift |
| `dataset_summary.html` | **39** | 1.0s | **1070ms** | **0.91** | **4.9s** | 825 `<tr>` inline, `paging:false` |
| `api/dataset/eegdash.dataset.html` | 59 | 0.9s | 390ms | 0.57 | 2.0s | 7.43MB DOM (autodoc dump) |

LCP is fine everywhere (CDN + gzip). The problems are **DOM size, main-thread
work, and layout shift** — not download size.

Confirmed by inspecting the live pages:
- `_static/scripts/fontawesome.js` loads on **every** page.
- Summary page loads jQuery + **plotly 3.1MB** + d3 + DataTables **+ 4 plugins**,
  and inlines **825 table rows** (`paging: false` in `datatables-init.js`).
- API page is 7.43MB of HTML with **0 tables** — one giant `automodule` dump.

Already in place (don't redo): `dt-loading-skeleton` FOUC guard, hero
`min-height` reservations, deferred global JS, lazy-loaded search index, local
SVG icon_links.

## Fix plan (ranked; each verifiable via the PR surge preview + Lighthouse)

### 1. `dataset_summary.html` — the 825-row DOM  (biggest win; **product decision**)
Drives CLS 0.91 + TBT 1070ms + TTI 4.9s. Pick one:
- **(a) Paginate** — `paging: true, pageLength: 25, deferRender: true` in
  `datatables-init.js` (lines 118 & 208). ~30× smaller initial DOM. Changes UX
  from "scroll all" to paged.
- **(b) Keep all rows, cut render cost** — add to `custom.css`:
  `#datasets-table tbody tr { content-visibility: auto; contain-intrinsic-size: auto 44px; }`
  Native browser skip-offscreen; big TBT/TTI drop, no UX change. Test scroll +
  DataTables column-width calc (can interact).
- **(c) Ajax** — emit rows as JSON and load via DataTables `ajax` instead of
  inlining 3.8MB HTML. Most work, best result.

Kill the CLS regardless: make the `dt-loading-skeleton` height equal the final
rendered table height so the swap doesn't shift.

### 2. Defer plotly/d3 on the summary page  (−~3MB eager JS)
`plotly-3.1.0.min.js` (3.1MB) + `d3.v7.min.js` load eagerly for charts that are
below the fold. Lazy-load on scroll — reuse the existing `lazy-embed.js` pattern.

### 3. `api/dataset` page — split the 7.4MB dump  (**URL/SEO risk — plan first**)
Use `autosummary` per-class stub pages instead of one `automodule`.
⚠️ `prepare_summary_tables.py` generates links to
`/api/dataset/eegdash.dataset.{ID}.html`; a structure change needs a redirect
map (or keep those exact paths) or it creates 404s. Ship with redirects.

### 4. `fontawesome.js` (~540KB/page)  (**higher risk than it looks**)
The pydata theme still bundles it despite the local-SVG `icon_links`. It's
*partially used* by theme chrome (admonition/nav icons), so removing it outright
breaks those. Real options: bump `pydata-sphinx-theme` to a version that loads
FA conditionally, or replace remaining theme icon usages. Verify the built page
no longer requests `fontawesome.js`.

### 5. Home CLS 0.43
Existing hero min-heights aren't enough. Add `width`/`height` (or
`aspect-ratio`) to hero images/logo and reserve space for any JS-injected block.

### 6. Cheap global wins
`content-visibility: auto` on large static API sections; minify HTML; drop
unused theme CSS (~75KB/page).

## Verify loop

1. Change on a branch, open a PR → `doc.yaml` builds a surge.sh preview.
2. `npx lighthouse <preview-url>/dataset_summary.html --preset=desktop --only-categories=performance`
3. Compare CLS/TBT/TTI against the table above. Target: CLS < 0.1, TBT < 200ms.
