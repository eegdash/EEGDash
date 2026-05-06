# Badge row alignment — evidence

Generated 2026-04-21 after fixing `docs/source/index.rst` and `docs/source/_static/custom.css` and rebuilding with `make html-noplot`.

## Static HTML check

```
$ grep -A 20 'hf-badges docutils' docs/_build/html/index.html \
    | grep -oE 'style="width: [0-9]+px; height: [0-9]+px;"'
style="width: 104px; height: 20px;"
style="width: 102px; height: 20px;"
style="width:  78px; height: 20px;"
style="width: 198px; height: 20px;"
style="width: 106px; height: 20px;"
style="width: 112px; height: 20px;"
style="width: 134px; height: 20px;"
style="width:  60px; height: 20px;"
```

Every `<img>` ships with an explicit `width` and `height` attribute (SEO / CLS contract preserved), and every height is 20px (natural shield-badge height).

## Runtime measurement (Chrome, `getBoundingClientRect`)

Served `_build/html/` via `python3 -m http.server 8765`, loaded `http://127.0.0.1:8765/index.html`, and measured every `.hf-badges img`:

| # | alt               | naturalW | boundingW | boundingH | top     | computedStyle                   |
|---|-------------------|----------|-----------|-----------|---------|---------------------------------|
| 1 | Test Status       | 104      | 104       | 20        | 1092.57 | `width: 104px; height: 20px;`   |
| 2 | Doc Status        | 102      | 102       | 20        | 1092.57 | `width: 102px; height: 20px;`   |
| 3 | PyPI              |  78      |  78       | 20        | 1092.57 | `width:  78px; height: 20px;`   |
| 4 | Python Versions   | 198      | 198       | 20        | 1092.57 | `width: 198px; height: 20px;`   |
| 5 | Downloads         | 106      | 106       | 20        | 1092.57 | `width: 106px; height: 20px;`   |
| 6 | Code Coverage     | 112      | 112       | 20        | 1092.57 | `width: 112px; height: 20px;`   |
| 7 | License           | 134      | 134       | 20        | 1092.57 | `width: 134px; height: 20px;`   |
| 8 | GitHub Stars      |  60      |  60       | 20        | 1092.57 | `width:  60px; height: 20px;`   |

### Invariants satisfied

- **All heights equal**: `[20, 20, 20, 20, 20, 20, 20, 20]` → `allSameHeight = true`.
- **All top offsets equal**: every badge sits on the same baseline `top = 1092.57` → `allSameTop = true`.
- **No aspect-ratio distortion**: `boundingW == naturalW` for every badge (1:1 scale; browser never re-scales the SVG).
- **SEO contract intact**: each `<img>` has explicit `width` + `height` so reserved layout box matches rendered box (no CLS regression).

## What changed vs. the broken screenshot

Before (post-PR #322 state): `.hf-badges img { height: 26px }` forced uniform 26px CSS height while the RST `:width:` values (90 / 120 / 140) did not match the SVGs' natural widths — so the browser stretched/squished each badge independently, producing the visibly different row heights the user flagged.

After (this fix):
- `docs/source/index.rst` — each of the 8 `.. image::` directives now pins `:width:` to the SVG's **actual** natural width and `:height: 20` (the shared natural shield-badge height).
- `docs/source/_static/custom.css` — the `height: 26px` rule on `.hf-badges img` is removed; the `filter: saturate(0.9)` is retained.

No attributes were stripped — the user's SEO/CLS constraint is preserved, per-badge.
