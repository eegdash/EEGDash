// dataset_table.js — DataTables UI for the dataset summary page.
//
// Source of truth: this file. Loaded as a sibling <script src="..."> from
// docs/source/_static/dataset_generated/dataset_summary_table.html, which
// is emitted by docs/prepare_summary_tables.py and inlined into
// docs/source/dataset_summary.rst via the `.. dataset-figure:: table`
// directive (see docs/source/_extensions/dataset_figure.py).
//
// The Python generator only writes the table HTML; everything below is
// behaviour (sort, filter, ColVis, SearchPanes, click-to-filter,
// Total-row recomputation, FOUC guard). The script reads column metadata
// from the rendered <thead> text — it does not depend on any data-*
// attributes injected by Python.

function tagsArrayFromHtml(html) {
    if (html == null) return [];
    if (typeof html === 'number') return [String(html)];
    if (typeof html === 'string' && html.indexOf('<') === -1) return [html.trim()];
    const tmp = document.createElement('div');
    tmp.innerHTML = html;
    const tags = Array.from(tmp.querySelectorAll('.tag')).map(el => (el.textContent || '').trim());
    const text = tmp.textContent.trim();
    return tags.length ? tags : (text ? [text] : []);
}

function parseSizeToBytes(text) {
    if (!text) return 0;
    const m = String(text).trim().match(/([\d,.]+)\s*(TB|GB|MB|KB|B)/i);
    if (!m) return 0;
    const value = parseFloat(m[1].replace(/,/g, ''));
    const unit = m[2].toUpperCase();
    const factor = { B:1, KB:1024, MB:1024**2, GB:1024**3, TB:1024**4 }[unit] || 1;
    return value * factor;
}

document.addEventListener('DOMContentLoaded', function () {
    const table = document.getElementById('datasets-table');
    if (!table || !window.jQuery || !window.jQuery.fn.DataTable) return;

    const $table = window.jQuery(table);
    if (window.jQuery.fn.DataTable.isDataTable(table)) return;

    const $tbody = $table.find('tbody');
    // Cell 0 now reads "Total 735 datasets" (rendered by prepare_table);
    // match on a leading "Total" token so the row moves into <tfoot> and
    // survives DataTables search/filter.
    const $total = $tbody.find('tr').filter(function(){
        return /^\s*Total\b/.test(window.jQuery(this).find('td').eq(0).text());
    });
    if ($total.length) {
        let $tfoot = $table.find('tfoot');
        if (!$tfoot.length) $tfoot = window.jQuery('<tfoot/>').appendTo($table);
        $total.appendTo($tfoot);
    }

    // Columns that expose a SearchPanes filter. Limit to low-cardinality
    // categorical fields (the four tag columns + Source); numeric columns
    // and free-text identity columns would produce a pick-list with
    // hundreds of options and swamp the filter panel.
    const FILTER_HEADERS = new Set(['source', 'recording', 'pathology', 'modality', 'type']);
    const FILTER_COLS = (function(){
        const cols = [];
        document.querySelectorAll('#datasets-table thead th').forEach((th, i) => {
            const t = (th.textContent || '').trim().toLowerCase();
            if (FILTER_HEADERS.has(t)) cols.push(i);
        });
        return cols;
    })();
    const TAG_COLS = (function(){
        const tagHeaders = new Set(['recording', 'pathology', 'modality', 'type']);
        const cols = [];
        $table.find('thead th').each(function(i){
            if (tagHeaders.has(window.jQuery(this).text().trim().toLowerCase())) cols.push(i);
        });
        return cols;
    })();
    const sizeIdx = (function(){
        let idx = -1;
        $table.find('thead th').each(function(i){
            const t = window.jQuery(this).text().trim().toLowerCase();
            if (t === 'size on disk' || t === 'size') idx = i;
        });
        return idx;
    })();

    // Power-user / rarely-distinguishing columns are hidden by default.
    // They stay available via the Columns button.
    const hiddenByDefaultIdxs = (function() {
        const targets = [];
        const wanted = new Set([
            'author (year)',
            'canonical',
            'source',          // only 2 values (openneuro / nemar) — low info
            'sessions',        // most datasets are 1 — low info density
        ]);
        const headers = table.querySelectorAll('thead th');
        for (let i = 0; i < headers.length; i++) {
            if (wanted.has((headers[i].textContent || '').trim().toLowerCase())) targets.push(i);
        }
        return targets;
    })();

    const dataTable = $table.DataTable({
        dom: 'Blfrtip',
        paging: false,
        searching: true,
        info: false,
        // Default sort = Dataset column (0) ascending; Canonical is hidden.
        order: [[0, 'asc']],
        language: {
            search: 'Filter dataset:',
            searchPanes: { collapse: { 0: 'Filters', _: 'Filters (%d)' } },
            emptyTable: '<span class="no-match">No datasets in this catalogue yet.</span>',
            zeroRecords: '<span class="no-match"><strong>No matches.</strong> Try clearing the search box or the Filters chip, or broaden your query.</span>'
        },
        buttons: [
            {
                extend: 'searchPanes',
                text: 'Filters',
                config: {
                    cascadePanes: true,
                    viewTotal: true,
                    layout: 'columns-4',
                    initCollapsed: false,
                    // Restrict the filter panel to the categorical columns.
                    // The `columns` array is resolved at runtime from the
                    // current table, below.
                    columns: FILTER_COLS,
                }
            },
            {
                extend: 'colvis',
                text: 'Columns',
                columns: ':not(:first-child)'  // don't let users hide the Dataset column
            }
        ],
        columnDefs: (function(){
            const findIdx = (label) => {
                let i = -1;
                $table.find('thead th').each(function (k) {
                    if ((this.textContent || '').trim().toLowerCase() === label) i = k;
                });
                return i;
            };
            // By default SearchPanes shows a pane for every column. That
            // would include Records (734 unique values), Canonical (~300),
            // and other high-cardinality columns — swamping the filter UI.
            // Only include the low-cardinality categorical columns that
            // are in FILTER_COLS; omit columnDefs for the rest and instead
            // turn SearchPanes off at the button level with a `columns`
            // whitelist. We still attach searchPanes opts on FILTER_COLS so
            // each pane can customise orthogonal / threshold.
            const defs = [
                { searchPanes: { show: true }, targets: FILTER_COLS },
            ];
            if (TAG_COLS.length) {
                defs.push({
                    targets: TAG_COLS,
                    searchPanes: { show: true, orthogonal: 'sp' },
                    render: function(data, type) { return type === 'sp' ? tagsArrayFromHtml(data) : data; }
                });
            }
            if (sizeIdx !== -1) {
                defs.push({
                    targets: sizeIdx,
                    render: function(data, type) {
                        return (type === 'sort' || type === 'type') ? parseSizeToBytes(data) : data;
                    }
                });
            }

            // Records / Subjects columns render as sparkbar HTML; the raw
            // `data` is that HTML, whose innerText is a comma-formatted
            // number. DataTables' default sort treats it as a string, so
            // "79" beats "40,360" in descending order. Parse it as an int
            // for sort/type, keep the HTML for display.
            ['records', 'subjects'].forEach(label => {
                const idx = findIdx(label);
                if (idx === -1) return;
                defs.push({
                    targets: idx,
                    render: function(data, type) {
                        if (type !== 'sort' && type !== 'type') return data;
                        const m = String(data).match(/>(\s*[\d,]+|—)\s*</);
                        if (!m) return 0;
                        const token = m[1].trim();
                        return token === '—' ? -1 : parseInt(token.replace(/,/g, ''), 10);
                    }
                });
            });

            // Channels + Sampling rate carry a trailing "*" for values that
            // are medians across recordings. DataTables sees the star as a
            // non-numeric suffix, falls back to string sort, and "999" ends
            // up above "8192". Strip the "*" and sort as int.
            ['channels', 'sampling rate'].forEach(label => {
                const idx = findIdx(label);
                if (idx === -1) return;
                defs.push({
                    targets: idx,
                    render: function(data, type) {
                        if (type !== 'sort' && type !== 'type') return data;
                        const s = String(data ?? '').trim();
                        if (!s || s === '—') return -1;
                        const n = parseInt(s.replace(/[*,\s]/g, ''), 10);
                        return Number.isNaN(n) ? -1 : n;
                    }
                });
            });
            if (hiddenByDefaultIdxs.length) {
                defs.push({ targets: hiddenByDefaultIdxs, visible: false });
            }
            // Reserve width for columns whose worst-case content is two
            // chips on one line (Type: "Clinical Intervention"; Recording:
            // "EEG + MEG"). Without a reservation these columns collapse to
            // ~80px and the multi-chip rows wrap to 2 lines.
            // Column-width reservations — pinned widths force DataTables to
            // give each column enough room for its worst-case content:
            //   - Dataset ID column fits the longest ID ("EEG2025R10MINI")
            //   - Sparkbar columns (records / subjects / size) need >= the
            //     ~150px sparkbar min-width plus padding
            //   - Tag columns need room for multi-chip values on one line
            //   - Scalar numeric columns (tasks / channels / sampling-rate)
            //     stay compact.
            const widthMap = {
                dataset: '8rem',
                // Hidden-by-default but reserve width anyway; without this,
                // revealing the column via ColVis collapses its cell to
                // ~80 px and long compound values (e.g. Canonical lists
                // like "BNCI2015_P300 BNCI2014_004 Schirrmeister2017 …")
                // wrap into 15-line cells that break the row rhythm.
                'author (year)': '9rem',
                canonical: '16rem',
                source: '7rem',
                recording: '9rem',
                pathology: '14rem',
                modality: '12rem',
                type: '12rem',
                records: '10rem',
                subjects: '10rem',
                tasks: '5rem',
                sessions: '6rem',
                channels: '6rem',
                'sampling rate': '6rem',
                size: '10rem',
            };
            Object.entries(widthMap).forEach(([label, w]) => {
                const idx = findIdx(label);
                if (idx !== -1) defs.push({ targets: idx, width: w });
            });
            return defs;
        })()
    });

    // Recompute column widths whenever a column is shown/hidden so newly
    // revealed columns don't overflow into their neighbours.
    dataTable.on('column-visibility.dt', function () {
        dataTable.columns.adjust();
    });

    // When the ColVis popup opens it is appended inside the .dt-buttons
    // container below the Columns chip. On short viewports the list
    // (14 items, ~580 px) overflows below the fold with most rows
    // unreachable. When that would happen, flip the popup so it opens
    // *upward* from the chip's top edge and cap its height to stay within
    // the viewport; also publish --dtc-top-offset so the CSS can size it.
    const positionCollection = function (node) {
        if (!(node instanceof HTMLElement)) return;
        if (!node.classList.contains('dt-button-collection')) return;
        // Reset any previous flip so re-measuring reflects natural position.
        node.style.removeProperty('transform');
        node.classList.remove('dtc-flipped');
        const rect = node.getBoundingClientRect();
        const vh = window.innerHeight;
        const safeBottom = 16; // 1rem gap before viewport edge
        if (rect.bottom > vh - safeBottom) {
            // Need to shift the menu up by (overflow amount + safeBottom)
            // so its bottom lands inside the viewport.
            const shift = Math.ceil(rect.bottom - vh + safeBottom);
            node.style.transform = `translateY(-${shift}px)`;
            node.classList.add('dtc-flipped');
        }
        // Publish the final top offset so CSS max-height can guarantee the
        // list fits even if content grew after the shift.
        const finalTop = node.getBoundingClientRect().top;
        node.style.setProperty('--dtc-top-offset', Math.max(8, finalTop) + 'px');
    };
    new MutationObserver(function (records) {
        records.forEach(function (rec) {
            rec.addedNodes.forEach(function (node) {
                if (node instanceof HTMLElement) {
                    positionCollection(node);
                    node.querySelectorAll?.('.dt-button-collection').forEach(positionCollection);
                }
            });
        });
    }).observe(document.body, { childList: true, subtree: true });

    // Search input: add a placeholder so first-time users know what the
    // field filters on, and an aria-label for screen readers.
    const $searchInput = $table.closest('.dataTables_wrapper').find('.dataTables_filter input');
    if ($searchInput.length) {
        $searchInput.attr('placeholder', 'e.g. DS000117, healthy, visual');
        $searchInput.attr('aria-label', 'Filter datasets by any field');
    }

    // Tag each column's td+th with a data-col-key attribute derived from
    // its header text. CSS can then hook specific columns (e.g. the Type
    // column with multi-tag cells) without fragile nth-child indices.
    //
    // Use the DataTables API (not `$table.find('thead th')`): by this
    // point DataTables has already removed hidden columns from the DOM,
    // so a raw DOM query would skip them. `dataTable.columns()` walks ALL
    // columns, visible or not, so when ColVis reveals one later its cells
    // already carry the expected attribute.
    dataTable.columns().every(function () {
        const headerEl = this.header();
        const key = (headerEl.textContent || '').trim().toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
        if (!key) return;
        headerEl.setAttribute('data-col-key', key);
        this.nodes().each(function (cell) {
            if (cell) cell.setAttribute('data-col-key', key);
        });
    });

    // Screen-reader hint on each Dataset ID link — without it, SRs just
    // announce the raw identifier ("DS000117") with no context.
    $table.find('tbody td:first-child a').each(function () {
        const id = (this.textContent || '').trim();
        if (id) this.setAttribute('aria-label', 'Open dataset ' + id + ' details');
    });

    // Click-to-filter on tag chips: clicking a tag filters the table to
    // rows whose same column contains that EXACT tag (not a substring).
    // Each tag in a multi-tag cell is wrapped in its own `<span>`, so the
    // ">TAG<" bracket pair uniquely identifies a complete tag text and
    // won't bleed "Clinical" into "Clinical Intervention" rows (which
    // contain `>Clinical</span> <span …>Intervention<` — "Clinical" is
    // followed by `<` either way, but also `>Intervention<` is there
    // separately, so we match on the bracketed text specifically).
    const escapeRe = s => String(s).replace(/[-/\\^$*+?.()|[\]{}]/g, '\\$&');
    $table.on('click', 'tbody .tag', function (e) {
        e.preventDefault();
        const cell = this.closest('td');
        if (!cell) return;
        const colKey = cell.dataset.colKey;
        if (!colKey) return;
        const col = dataTable.column('th[data-col-key="' + colKey + '"]');
        if (!col.length) return;
        const val = (this.textContent || '').trim();
        if (!val) return;
        // Match `>VALUE<` which only appears around the text of a
        // complete tag span. For cells containing a single plain text
        // token (no <span>) we also allow the value as the entire cell.
        const regex = '>' + escapeRe(val) + '<|^' + escapeRe(val) + '$';
        const current = col.search();
        if (current === regex) {
            col.search('').draw();
        } else {
            col.search(regex, true, false).draw();
        }
    });

    // Mount a "Clear filters" chip in the toolbar. Hidden at rest, shown
    // whenever any filter narrows the view (tag click, search box, column
    // search, or SearchPanes selection). One click wipes all three so the
    // user always has an obvious escape hatch. Added once; draw.dt below
    // only toggles its visibility.
    const clearChip = document.createElement('button');
    clearChip.type = 'button';
    clearChip.className = 'dt-button dt-clear-filters';
    clearChip.innerHTML = '\u2715&nbsp;Clear filters';
    clearChip.hidden = true;
    clearChip.setAttribute('aria-label', 'Clear all filters');
    clearChip.addEventListener('click', function () {
        // Reset column searches
        dataTable.columns().every(function () { this.search(''); });
        // Reset global search
        dataTable.search('');
        // Reset SearchPanes selections if the plugin is initialised
        if (dataTable.searchPanes && typeof dataTable.searchPanes.clearSelections === 'function') {
            try { dataTable.searchPanes.clearSelections(); } catch (e) {}
        }
        dataTable.draw();
    });
    const buttonsHost = $table.closest('.dataTables_wrapper').find('.dt-buttons')[0];
    if (buttonsHost) buttonsHost.appendChild(clearChip);

    // Capture the original Total row values on first draw so we can restore
    // them when all filters are cleared. The Tfoot is in the DOM before
    // DataTables takes over, so these reads give us the "unfiltered" state.
    const $tfootCells = $table.find('tfoot tr').children();
    const originalTotal = {};
    $tfootCells.each(function (i) { originalTotal[i] = this.innerText; });

    // Helper: sum of integers parsed from a column's rendered cells that
    // are currently in the filtered / searched subset. Note the options
    // go on the `column()` selector — passing them to .nodes() is a no-op
    // in DataTables 1.13 and quietly iterates every row.
    const sumColumn = (colKey, parser) => {
        let total = 0;
        const col = dataTable.column('th[data-col-key="' + colKey + '"]',
                                     { search: 'applied' });
        if (!col.length) return null;
        col.nodes().each(function (cell) {
            if (!cell) return;
            const v = parser(cell);
            if (Number.isFinite(v)) total += v;
        });
        return total;
    };
    const formatInt = n => n.toLocaleString('en-US');
    const formatBytes = n => {
        const units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB'];
        let v = n, u = 0;
        while (v >= 1024 && u < units.length - 1) { v /= 1024; u++; }
        return u <= 1 ? Math.round(v) + ' ' + units[u] : v.toFixed(1) + ' ' + units[u];
    };
    const parseIntFromSparkbar = cell => {
        const lab = cell.querySelector('.sparkbar-label');
        if (!lab) return 0;
        const t = lab.textContent.trim();
        if (!t || t === '—') return 0;
        return parseInt(t.replace(/,/g, ''), 10);
    };
    const parseBytesFromSparkbar = cell => {
        const lab = cell.querySelector('.sparkbar-label');
        if (!lab) return 0;
        const t = lab.textContent.trim();
        if (!t || t === '—') return 0;
        const m = t.match(/([\d,.]+)\s*(TB|GB|MB|KB|B)/i);
        if (!m) return 0;
        const factor = { B:1, KB:1024, MB:1024**2, GB:1024**3, TB:1024**4, PB:1024**5 }[m[2].toUpperCase()] || 1;
        return parseFloat(m[1].replace(/,/g, '')) * factor;
    };
    const parseIntFromText = cell => {
        const t = (cell.textContent || '').trim().replace(/\*/g, '').replace(/,/g, '');
        return parseInt(t, 10);
    };

    // Resolve column key -> DOM tfoot cell on demand. The tfoot is reordered
    // the first time DataTables redraws (row sort) and its cells need to be
    // matched via the header order, not via "whatever tbody's first row
    // looks like at init".
    const getFootCell = (key) => {
        const visibleHeaders = [...$table.find('thead tr:last-child th')];
        const idx = visibleHeaders.findIndex(th => th.dataset.colKey === key);
        if (idx === -1) return null;
        return $table.find('tfoot tr').children()[idx] || null;
    };

    // Re-render Total row to reflect current (possibly filtered) subset.
    const recomputeTotalRow = () => {
        const visibleCount = dataTable.rows({ search: 'applied' }).count();
        const allCount = dataTable.rows().count();
        const isFiltered = visibleCount < allCount;
        const firstCell = $tfootCells[0];
        if (firstCell) {
            firstCell.innerText = isFiltered
                ? `Showing ${formatInt(visibleCount)} of ${formatInt(allCount)}`
                : originalTotal[0];
        }
        // Re-aggregate numeric columns from visible cells.
        const colMap = {
            records:  { parse: parseIntFromSparkbar,   fmt: formatInt },
            subjects: { parse: parseIntFromSparkbar,   fmt: formatInt },
            tasks:    { parse: parseIntFromText,       fmt: formatInt },
            // Channels / Sampling rate use medians per-dataset; summing across
            // datasets is meaningless, so leave them blank while filtered.
            size:     { parse: parseBytesFromSparkbar, fmt: formatBytes },
        };
        Object.entries(colMap).forEach(([key, spec]) => {
            const footCell = getFootCell(key);
            if (!footCell) return;
            const domIdx = [...footCell.parentElement.children].indexOf(footCell);
            if (!isFiltered) {
                footCell.innerText = originalTotal[domIdx];
                return;
            }
            const sum = sumColumn(key, spec.parse);
            footCell.innerText = sum ? spec.fmt(sum) : '';
        });
    };

    // After every draw, reflect active column filters on:
    //  (a) the column header (adds .dt-col-filtered so CSS can tint it)
    //  (b) body tags whose text matches the active filter (adds
    //     .tag-filter-active so they read as "currently applied").
    //  (c) the Clear-filters chip visibility + label ("Clear 2 filters").
    //  (d) the Total row numbers reflecting the currently visible subset.
    dataTable.on('draw.dt', function () {
        let activeColCount = 0;
        dataTable.columns().every(function () {
            const header = this.header();
            const colKey = header.dataset.colKey;
            const raw = this.search();
            const active = raw !== '';
            if (active) activeColCount++;
            header.classList.toggle('dt-col-filtered', active);
            if (!colKey) return;
            // The search value is now a tag-boundary regex of the form
            // ">Foo<|^Foo$"; recover the plain tag text "Foo" to match
            // against tags in the column.
            let filterText = raw;
            const m = raw.match(/^>(.+?)<\|\^\1\$$/);
            if (m) filterText = m[1].replace(/\\([-/\\^$*+?.()|\[\]{}])/g, '$1');
            $table.find('tbody td[data-col-key="' + colKey + '"] .tag').each(function () {
                this.classList.toggle('tag-filter-active',
                    active && (this.textContent || '').trim() === filterText);
            });
        });
        const globalSearchActive = dataTable.search() !== '';
        const visibleCount = dataTable.rows({ search: 'applied' }).count();
        const allCount = dataTable.rows().count();
        const subsetActive = visibleCount < allCount;
        // Treat global search / panes / subset narrowing as one additional
        // "implicit" filter for the count purposes if no column filter
        // explains the row reduction.
        let labelCount = activeColCount;
        if (!activeColCount && subsetActive) labelCount = 1;
        if (globalSearchActive && !activeColCount) labelCount = 1;
        clearChip.innerHTML = '\u2715&nbsp;Clear '
            + (labelCount > 1 ? labelCount + ' filters' : 'filter' + (labelCount === 1 ? '' : 's'));
        // Show chip if any filter signal is live.
        clearChip.hidden = !(activeColCount || globalSearchActive || subsetActive);
        recomputeTotalRow();
    });

    // Force a few columns to reserve space for their worst-case content
    // (compound tags like "Clinical Intervention" or "EEG + MEG"). The
    // columnDefs `width` option is ignored once DataTables has auto-sized,
    // so we set inline styles directly on the <th> and re-adjust.
    const $typeTh = $table.find('thead th[data-col-key="type"]');
    if ($typeTh.length) $typeTh.css('min-width', '13rem');
    const $recTh = $table.find('thead th[data-col-key="recording"]');
    if ($recTh.length) $recTh.css('min-width', '9rem');
    dataTable.columns.adjust();

    // Drive the right-edge "scroll to see more" fade via an `.is-overflowing`
    // class on BOTH the DataTables wrapper and the outer <figure>. The
    // ::after affordance is anchored on the figure (outside the scroll
    // container) so it stays pinned to the visible right edge when users
    // scroll the table horizontally.
    const wrapper = table.closest('.dataTables_wrapper') || document.getElementById(table.id + '_wrapper');
    const figure = wrapper && wrapper.closest('.eegdash-figure');
    const syncOverflow = function () {
        if (!wrapper) return;
        const sx = wrapper.scrollWidth > wrapper.clientWidth + 1;
        wrapper.classList.toggle('is-overflowing', sx);
        if (figure) figure.classList.toggle('is-overflowing', sx);
    };
    syncOverflow();
    window.addEventListener('resize', syncOverflow, { passive: true });
    dataTable.on('column-visibility.dt', function () { window.setTimeout(syncOverflow, 0); });

    // Header clicks follow DataTables' default: sort. A previous iteration
    // added a second click handler on the tag-column headers that opened
    // SearchPanes — it conflicted with the sort-on-click convention and its
    // tooltip ("Click to filter this column") lied about what clicking did.
    // Filtering is now exclusively reachable via the Filters chip above.

    // FOUC guard: tear down the loading skeleton once DataTables has wrapped
    // the table. The raw <table> reveal is driven by CSS (it gains the
    // `.dataTable` class at init), so all we need to do here is drop the
    // placeholder. Done last so any thrown error above leaves the skeleton
    // up — a visible placeholder beats a blank card.
    document
      .querySelectorAll('.dt-loading-skeleton')
      .forEach(function (el) { el.hidden = true; });
});
