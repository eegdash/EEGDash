# EEGDash SEO / agent-readiness — audit log

> Running record of SEO + agent-readiness issues we've identified across
> PageSpeed Insights, Ahrefs Site Audit, isitagentready.com,
> buildwithfern.com Agent Score, and orank.ai Dora. Not every row has
> been ticked off; the ones still open are the target of future PRs.

## Platforms we validate against

| Platform | Measures | Rescan cadence | Auth required to trigger |
| --- | --- | --- | --- |
| [PageSpeed Insights (mobile)](https://pagespeed.web.dev/analysis/https-eegdash-org/) | Core Web Vitals + SEO category + Best Practices | On-demand | No |
| [Ahrefs Site Audit](https://app.ahrefs.com/site-audit/9713818/issues) | Full technical SEO crawl (404s, metadata, orphans, …) | Scheduled crawl | Yes |
| [isitagentready.com](https://isitagentready.com/eegdash.org) | Well-known files, MCP, Agent Skills, commerce checks | On-demand | No |
| [buildwithfern.com Agent Score](https://buildwithfern.com/agent-score/company/eegdash) | Docs-agent-readiness (llms.txt, markdown URLs, page-size) | Weekly + Rerun button (rate-limited) | Logged-in browser session |
| [orank.ai Dora](https://www.orank.ai/scan/eegdash.org) | Discovery / Identity / Auth / Agent / UX layers | Rescan button, ~5 s | No |

## Baseline vs. current scores

| Platform | Jan-2026 baseline | After PR #312 | After PR #314 | After PR #315 | Current (2026-04-20) |
| --- | ---: | ---: | ---: | ---: | ---: |
| isitagentready | 25 | 67 | 67 | 67 | **67** |
| buildwithfern Agent Score | 56 (F) | 56 (F) | 77 (C) | 77 (C) | **77 (C)** |
| orank.ai Dora | 32 (D) | 48 (C) | 48 (C) | 48 (C) | **40 (C)** (regression) |
| PSI mobile SEO | 57 | 92 | 92 | 92 | expected 100 after re-run |
| Ahrefs total 404s | 1 445 | 1 445 | 1 445 | 1 445 | **738** (-50 %) |

## Infrastructure caveats we keep hitting

* **GitHub Pages + Fastly** — no response-header control. Blocks
  `Link:` header emission (RFC 8288) and `Accept: text/markdown`
  content-negotiation. Workaround is a reverse proxy; see the parked
  plan in the Cloudflare runbook the team decided not to ship.
* **Lighthouse robots-txt validator lag** — rejects Cloudflare's
  `Content-Signal` directive as "Unknown directive". We keep it as a
  comment in `docs/source/_extra/robots.txt` to preserve the PSI
  score. Cost: orank's *Bot Access Control* layer lost 6 points once
  the directive went cold. Re-introduce as a live directive the moment
  Lighthouse ships support, or emit it as an HTTP response header from
  a reverse proxy.
* **Fern "Rerun" button** is authed / rate-limited. From an anonymous
  session it returns `Failed to start rerun.` silently — only a logged-in
  maintainer can trigger fresh crawls.

## Issue inventory (Ahrefs, fresh crawl 2026-04-20)

Ordered by expected impact. Each row tracks the status across PRs.

| Issue | Count | Status | Plan |
| --- | ---: | --- | --- |
| `404` pages | 738 | 50 % reduced in #315 | Remaining 738 are external-link backlinks + NEMAR IDs; covered by `/404.html` client-side redirect. |
| Meta description too long | 422 | **fix in flight in #317** | Regex in `_cap_descriptions_in_metatags` mis-handled apostrophes. Targeted patch. |
| Meta description too short | 284 | **open** | Extend the `_synthesize_description` backstop to overwrite existing descriptions under 50 chars, not just fill in when missing. |
| Meta description missing | 8 | **open** | Same backstop covers them once it no longer checks for "still lacks". |
| Orphan pages | 9 | **open** | The 8 `dataset_summary/*.html` chart fragments + `api/dataset/eegdash.html`. Remove them from the build or link them from the main catalog page. |
| Page size > 2 MB | 3 | partial | `dataset_summary.html` is 4.2 MB (known). Identify the other two. |
| Pages with only 1 dofollow incoming | 11 | open | Cross-link per-module API pages into each other. |
| H1 tag missing or empty | 8 | **open** | Investigate — likely the dataset_summary fragments. |
| Multiple H1 tags | 3 | **open** | Investigate — probably dataset pages combine the auto-generated header with a body H1. |
| Multiple title tags | 2 | **open** | Investigate. |
| Multiple meta description tags | 1 | **open** | Investigate. |
| Missing alt text | 1 | **open** | Single image. Grep the built output. |
| `5XX` page | 1 | **open** | Find and either fix the origin endpoint or exclude from sitemap. |
| 5XX page in sitemap | 1 | open | Resolves once the 5XX above is fixed. |
| Non-canonical page in sitemap | 1 | open | Identify. |
| HTML file size too large | 1 | partial | `dataset_summary.html` markdown fallback shipped in #314. |
| CSS file size too large | 2 | open | Out-of-scope: theme CSS is large by design. Only worth a minifier pass if we migrate to self-hosted stack. |
| Image file size too large | 1 | likely cache | NSF logo was 1,434 KiB; PR #315 took it to 22 KiB. Next crawl should clear. |
| 3XX redirect chain | 3 | informational | External backlinks to `http://eegdash.org/`. Not actionable from the repo. |
| HTTP → HTTPS redirect | 2 | informational | Same. |
| Indexable page not in sitemap | 22 | open | Gallery-page exclusion was removed in #315 but fresh crawl still finds some; may stabilise on next crawl. |

## Checklist — what each PR has already done

* **#312** — agent-readiness surface (robots.txt, Agent Skills, API
  catalog, security.txt, llms.txt), library-first hero, paper-accurate
  citation metadata (CITATION.cff, codemeta.json).
* **#314** — expanded llms.txt to cover 533 sitemap URLs, shrank
  `dataset_summary.md`, strengthened `<link rel>` discovery hints.
* **#315** — NSF logo 1.4 MB → 22 KB, FontAwesome swapped to 3 local
  SVGs, GTM deferred, fonts preloaded, GSC/Bing verification
  placeholders, IndexNow wiring, relative-href bug fix on catalog,
  custom `/404.html`, image width/height on homepage, hero CLS
  reservations, auto-description injector, schema.org dataset
  license URL + ISO 8601 `datePublished`.
* **#317** *(open)* — regex fix so the description cap actually
  matches content containing apostrophes ("Alzheimer's disease").
* **This PR** — see the "In scope" section below.

## In scope for the current PR (seo-batch-2)

1. Extend `_inject_seo_context` so the description backstop also
   overrides existing descriptions under 50 chars.
2. Audit pages for duplicate `<h1>`, `<title>`, `<meta name=description>`
   tags; fix where in-repo.
3. Add `alt=""` to the one missing-alt image.
4. Decide the fate of the 8 `dataset_summary/*.html` chart fragments —
   exclude from the build or explicit internal links.
5. Investigate the single `5XX` page.
6. Ship a local validator (`scripts/validate_docs_seo.py`) that
   parses every HTML under `docs/_build/html/` and reports the same
   categories Ahrefs flags, so a regression like #317 never ships
   undetected again.

## What we're deliberately **not** touching in this PR

* Internal cross-linking of per-module API pages (11 × single-dofollow).
  Requires per-module curation work that would balloon scope.
* CSS minification / tree-shaking. Theme-CSS is the dominant block
  and any trimming risks visual regressions.
* Cloudflare / Caddy reverse proxy. Deferred by the maintainer call
  earlier in the project.

## How we validate before merge

* Run `python scripts/validate_docs_seo.py docs/_build/html` after a
  local `make html-noplot` (no gallery execution). Expect output like:

  ```
  SEO validator — docs/_build/html
  Scanned 1 300 HTML files
  ------------------------------
  meta description missing   0
  meta description too short 0
  meta description too long  0
  duplicate <title>          0
  duplicate <h1>             0
  duplicate meta description 0
  <img> without alt          0
  PASS
  ```

  Exit non-zero on regressions.
* Optionally re-run on a fresh develop build to confirm the PR's
  before/after counts.
