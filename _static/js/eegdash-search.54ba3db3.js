/**
 * EEGDash global search palette ("acquisition console").
 *
 * One search surface for the whole site, opened with Ctrl/Cmd+K, the
 * navbar search button, or any [data-open-eegdash-search] element.
 * Searches three sources:
 *
 *   1. Datasets   — _static/dataset_generated/search_index.json
 *                   (enriched: task names, license, channels, rates).
 *   2. Site pages — _static/docs_index.json (titles + sections only;
 *                   replaces the 2.3 MB searchindex.js for live search).
 *   3. Deep API   — GET {API}/datasets/search?q= ($text over name,
 *                   tasks, tags, README). Optional: the palette works
 *                   fully offline, the API only adds "deep matches".
 *
 * Structured filter tokens narrow datasets before/with free text:
 *   modality:eeg source:openneuro pathology:epilepsy type:visual
 *   task:rest license:CC0 channels:>=64 sfreq:1000 subjects:>100
 *
 * All assets (two JSON indexes) load lazily on first open, so every
 * other page view pays only this script (~14 KB) and its CSS.
 */
(function () {
  'use strict';

  // Overridable via <html data-eegdash-api="…"> so a staging build can
  // point the palette elsewhere without editing this file.
  var API_BASE = document.documentElement.getAttribute('data-eegdash-api')
    || 'https://data.eegdash.org/api/eegdash';
  var DEBOUNCE_MS = 90;
  var DEEP_DEBOUNCE_MS = 380;
  // Intentionally stricter than the server's min_length=2 — fewer
  // requests, not a mirror of the server contract.
  var DEEP_MIN_CHARS = 3;
  var MAX_PER_GROUP = { datasets: 6, docs: 5, examples: 5, api: 5, deep: 4 };
  var SCOPES = ['all', 'data', 'docs', 'api'];
  var SCOPE_LABELS = { all: 'All', data: 'Datasets', docs: 'Docs', api: 'API' };
  var RECENT_KEY = 'eegdash:recent-searches';

  var NUMERIC_OPS = /^(>=|<=|>|<|=)?(\d+(?:\.\d+)?)$/;

  // Filter-token facets. `list` facets substring-match any of the
  // entry's values; `numeric` facets apply the operator to any value.
  var FACETS = {
    modality: { type: 'list', hint: 'eeg, meg, ieeg, visual, auditory…', values: function (e) { return (e.recordModality || []).concat(e.modality || []); } },
    source: { type: 'list', hint: 'openneuro, nemar…', values: function (e) { return e.source ? [e.source] : []; } },
    pathology: { type: 'list', hint: 'epilepsy, parkinson…', values: function (e) { return e.pathology || []; } },
    type: { type: 'list', hint: 'visual, memory, resting…', values: function (e) { return e.type || []; } },
    task: { type: 'list', hint: 'rest, oddball, facerecognition…', values: function (e) { return e.taskNames || []; } },
    license: { type: 'list', hint: 'CC0, CC-BY…', values: function (e) { return e.license ? [e.license] : []; } },
    channels: { type: 'numeric', hint: '>=64, 128…', values: function (e) { return e.nchans || []; } },
    sfreq: { type: 'numeric', hint: '500, >=1000…', values: function (e) { return e.sfreq || []; } },
    subjects: { type: 'numeric', hint: '>100, >=20…', values: function (e) { return e.subjects != null ? [e.subjects] : []; } },
  };

  // Strips only KNOWN facet tokens out of the free text sent to the deep
  // API — an unknown `word:value` (e.g. "signal:noise") is legitimate
  // free text and must reach the server.
  var FACET_TOKEN_RE = new RegExp(
    '(?:^|\\s)(?:' + Object.keys(FACETS).join('|') + '):[^\\s]*', 'gi'
  );

  var state = {
    built: false,
    open: false,
    loading: false,
    datasets: null,
    docs: null,
    tokens: [],
    scope: 'all',
    query: '',
    items: [],          // flat list of navigable result anchors
    selected: -1,
    deep: { status: 'idle', results: [], controller: null, query: '' },
    facetValues: {},    // facet key -> sorted unique values (for suggestions)
  };

  var els = {};
  var searchTimer = null;
  var deepTimer = null;

  /* ------------------------------------------------------------------ *
   * Utilities
   * ------------------------------------------------------------------ */

  function contentRoot() {
    return document.documentElement.getAttribute('data-content_root') || '';
  }

  function esc(str) {
    return String(str == null ? '' : str)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function norm(str) {
    return String(str == null ? '' : str).toLowerCase();
  }

  var REGEX_CHARS = /[\\^$.*+?()[\]{}|]/;

  // Lowercased term array used for literal scoring. When the query
  // contains regex metacharacters AND compiles, the RegExp rides along
  // as `terms.rx` — same array everywhere downstream, no signature
  // churn. Power-user queries like `eeg|meg`, `vis.*` or `^ds00` work
  // the way the old theme-modal search did.
  function buildTerms(query) {
    var terms = norm(query).split(/\s+/).filter(function (w) { return w.length >= 2; });
    terms.rx = null;
    if (REGEX_CHARS.test(query)) {
      try {
        var rx = new RegExp(query.trim(), 'i');
        // Patterns matching the empty string match everything and break
        // highlighting — treat them as literal text instead.
        if (!rx.test('')) terms.rx = rx;
      } catch (err) { /* invalid regex — score as literal terms */ }
    }
    return terms;
  }

  function highlight(text, terms) {
    var safe = esc(text);
    try {
      var re;
      if (terms.rx) {
        re = new RegExp('(' + terms.rx.source + ')', 'gi');
      } else if (terms.length) {
        var pattern = terms.map(function (t) {
          return t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        }).join('|');
        re = new RegExp('(' + pattern + ')', 'gi');
      } else {
        return safe;
      }
      return safe.replace(re, '<mark>$1</mark>');
    } catch (err) {
      return safe;
    }
  }

  function datasetUrl(id) {
    return contentRoot() + 'api/dataset/eegdash.dataset.' + String(id).toUpperCase() + '.html';
  }

  // Local index rows carry a root-relative `url` chosen by
  // prepare_summary_tables.py for SEO reasons (see its comment) — prefer
  // it, but accept root-relative paths ONLY ("/x", not "//host" or any
  // scheme), so a poisoned index can never produce a javascript: href.
  // Deep-API rows have no url field and always take the computed path.
  function datasetHref(entry) {
    var u = entry.url;
    if (typeof u === 'string' && u.charAt(0) === '/' && u.charAt(1) !== '/') return u;
    return datasetUrl(entry.id);
  }

  function readRecent() {
    try {
      var raw = window.localStorage.getItem(RECENT_KEY);
      var list = raw ? JSON.parse(raw) : [];
      return Array.isArray(list) ? list.slice(0, 6) : [];
    } catch (err) { return []; }
  }

  function pushRecent(query) {
    var q = query.trim();
    if (q.length < 2) return;
    try {
      var list = readRecent().filter(function (item) { return item !== q; });
      list.unshift(q);
      window.localStorage.setItem(RECENT_KEY, JSON.stringify(list.slice(0, 6)));
    } catch (err) { /* private mode: fine */ }
  }

  /* ------------------------------------------------------------------ *
   * Index loading (lazy, once)
   * ------------------------------------------------------------------ */

  function loadIndexes() {
    if (state.datasets !== null || state.loading) return;
    state.loading = true;
    setTrace(true);

    var root = contentRoot();
    var datasetReq = fetch(root + '_static/dataset_generated/search_index.json')
      .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
      .catch(function () { return []; });
    var docsReq = fetch(root + '_static/docs_index.json')
      .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
      .catch(function () { return []; });

    Promise.all([datasetReq, docsReq]).then(function (loaded) {
      state.datasets = loaded[0] || [];
      state.docs = loaded[1] || [];
      state.loading = false;
      prepareDatasets();
      prepareDocs();
      buildFacetValues();
      setTrace(false);
      updateStatus();
      runSearch();
    });
  }

  // Lowercase every searched field once at load. scoreDataset and
  // tokenMatches run over all ~750 entries per keystroke; without this
  // they would rebuild and re-lowercase the same arrays tens of
  // thousands of times per search.
  function prepareDatasets() {
    (state.datasets || []).forEach(function (entry) {
      entry._lc = {
        id: norm(entry.id),
        title: norm(entry.title),
        tags: (entry.pathology || [])
          .concat(entry.type || [], entry.modality || [], entry.recordModality || [])
          .map(norm),
        tasks: (entry.taskNames || []).map(norm),
        aliases: (entry.canonical || [])
          .concat(entry.author ? [entry.author] : [])
          .map(norm),
      };
      // Materialize facet value arrays once so tokenMatches doesn't
      // re-concat per entry per keystroke. Deep-API rows skip this and
      // take the live accessor path.
      entry._fv = {};
      Object.keys(FACETS).forEach(function (key) {
        entry._fv[key] = FACETS[key].values(entry);
      });
    });
  }

  // Same one-time lowering for the docs index (title + section titles).
  function prepareDocs() {
    (state.docs || []).forEach(function (entry) {
      entry._t = norm(entry.t);
      entry._s = (entry.s || []).map(function (s) { return norm(s[0]); });
    });
  }

  function buildFacetValues() {
    Object.keys(FACETS).forEach(function (key) {
      var facet = FACETS[key];
      if (facet.type !== 'list') return;
      var seen = {};
      (state.datasets || []).forEach(function (entry) {
        facet.values(entry).forEach(function (value) {
          var v = String(value).trim();
          if (v) seen[v] = (seen[v] || 0) + 1;
        });
      });
      state.facetValues[key] = Object.keys(seen).sort(function (a, b) {
        return seen[b] - seen[a];
      });
    });
  }

  /* ------------------------------------------------------------------ *
   * Token grammar
   * ------------------------------------------------------------------ */

  function parseToken(raw) {
    var idx = raw.indexOf(':');
    if (idx <= 0) return null;
    var key = norm(raw.slice(0, idx));
    var value = raw.slice(idx + 1).trim();
    var facet = FACETS[key];
    if (!facet || !value) return null;
    if (facet.type === 'numeric') {
      var m = value.match(NUMERIC_OPS);
      if (!m) return null;
      return { key: key, op: m[1] || '=', num: parseFloat(m[2]), raw: key + ':' + value };
    }
    return { key: key, value: value, raw: key + ':' + value };
  }

  function tokenMatches(entry, token) {
    var facet = FACETS[token.key];
    var values = (entry._fv && entry._fv[token.key]) || facet.values(entry);
    if (facet.type === 'numeric') {
      return values.some(function (v) {
        var n = parseFloat(v);
        if (isNaN(n)) return false;
        switch (token.op) {
          case '>': return n > token.num;
          case '>=': return n >= token.num;
          case '<': return n < token.num;
          case '<=': return n <= token.num;
          default: return n === token.num;
        }
      });
    }
    var needle = norm(token.value);
    return values.some(function (v) { return norm(v).indexOf(needle) !== -1; });
  }

  // Stats `val` is typed float|int|str|null server-side — keep finite
  // positive numbers only, so "null ch" never renders and the numeric
  // facets never compare against NaN.
  function numericVals(counts) {
    return (counts || [])
      .map(function (c) { return Number(c && c.val); })
      .filter(function (n) { return isFinite(n) && n > 0; });
  }

  // Convert a deep-API row into the shape FACETS read, so committed
  // tokens filter deep matches identically to local ones.
  function deepRowToEntry(row) {
    var tags = row.tags || {};
    var mods = row.recording_modality;
    if (typeof mods === 'string') mods = [mods];
    return {
      id: row.dataset_id,
      title: row.computed_title || row.name || '',
      source: row.source || '',
      subjects: (row.demographics || {}).subjects_count,
      recordModality: mods || [],
      modality: tags.modality || [],
      pathology: tags.pathology || [],
      type: tags.type || [],
      taskNames: row.tasks || [],
      license: row.license || '',
      nchans: numericVals(row.nchans_counts),
      sfreq: numericVals(row.sfreq_counts),
      snippet: row.snippet || '',
    };
  }

  /* ------------------------------------------------------------------ *
   * Scoring
   * ------------------------------------------------------------------ */

  // These weights are deliberately independent of the Mongo $text index
  // weights in ops/migrations/0003 (server repo): the client index has no
  // README field, and this positional bonus scheme is not comparable to
  // $meta textScore — do not try to "sync" them.
  function scoreDataset(entry, terms) {
    var lc = entry._lc; // precomputed in prepareDatasets()
    if (terms.rx) {
      var rx = terms.rx;
      if (rx.test(lc.id)) return 90;
      if (rx.test(lc.title)) return 60;
      for (var k = 0; k < lc.tasks.length; k++) if (rx.test(lc.tasks[k])) return 55;
      for (k = 0; k < lc.aliases.length; k++) if (rx.test(lc.aliases[k])) return 50;
      for (k = 0; k < lc.tags.length; k++) if (rx.test(lc.tags[k])) return 35;
      return 0;
    }
    if (!terms.length) return 1; // token-only query: every filtered entry counts
    var id = lc.id;
    var title = lc.title;
    var haystackTags = lc.tags;
    var tasks = lc.tasks;
    var aliases = lc.aliases;

    var total = 0;
    for (var i = 0; i < terms.length; i++) {
      var t = terms[i];
      var s = 0;
      if (id === t) s = 120;
      else if (id.indexOf(t) === 0) s = 90;
      else if (id.indexOf(t) !== -1) s = 50;
      if (title.indexOf(t) !== -1) s = Math.max(s, title.indexOf(' ' + t) !== -1 || title.indexOf(t) === 0 ? 60 : 40);
      for (var j = 0; j < tasks.length && s < 55; j++) {
        if (tasks[j].indexOf(t) !== -1) s = Math.max(s, 55);
      }
      for (j = 0; j < aliases.length && s < 50; j++) {
        if (aliases[j].indexOf(t) !== -1) s = Math.max(s, 50);
      }
      for (j = 0; j < haystackTags.length && s < 35; j++) {
        if (haystackTags[j].indexOf(t) !== -1) s = Math.max(s, 35);
      }
      if (s === 0) return 0; // every term must match somewhere
      total += s;
    }
    return total;
  }

  function scoreDoc(entry, terms) {
    if (!terms.length && !terms.rx) return 0;
    var title = entry._t; // precomputed in prepareDocs()
    var sections = entry.s || [];
    var lowered = entry._s || [];
    var total = 0;
    var sectionHit = null;
    if (terms.rx) {
      var s0 = 0;
      if (terms.rx.test(title)) s0 = 70;
      for (var m = 0; m < lowered.length; m++) {
        if (terms.rx.test(lowered[m])) {
          s0 = Math.max(s0, 38);
          if (!sectionHit) sectionHit = sections[m];
        }
      }
      entry._section = sectionHit;
      return s0;
    }
    for (var i = 0; i < terms.length; i++) {
      var t = terms[i];
      var s = 0;
      if (title === t) s = 100;
      else if (title.indexOf(t) === 0) s = 70;
      else if (title.indexOf(t) !== -1) s = 45;
      for (var j = 0; j < lowered.length; j++) {
        if (lowered[j].indexOf(t) !== -1) {
          s = Math.max(s, 38);
          if (!sectionHit) sectionHit = sections[j];
        }
      }
      if (s === 0) return 0;
      total += s;
    }
    entry._section = sectionHit;
    return total;
  }

  /* ------------------------------------------------------------------ *
   * Deep (server) search
   * ------------------------------------------------------------------ */

  function scheduleDeepSearch(query) {
    clearTimeout(deepTimer);
    if (state.deep.controller) {
      state.deep.controller.abort();
      state.deep.controller = null;
    }
    if (query.length < DEEP_MIN_CHARS || state.scope === 'docs' || state.scope === 'api') {
      state.deep.results = [];
      state.deep.status = state.deep.status === 'off' ? 'off' : 'idle';
      return;
    }
    deepTimer = setTimeout(function () { fetchDeep(query); }, DEEP_DEBOUNCE_MS);
  }

  function fetchDeep(query) {
    var controller = ('AbortController' in window) ? new AbortController() : null;
    state.deep.controller = controller;
    state.deep.status = 'loading';
    state.deep.query = query;
    updateStatus();

    var url = API_BASE + '/datasets/search?q=' + encodeURIComponent(query) + '&limit=8';
    fetch(url, controller ? { signal: controller.signal } : {})
      .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
      .then(function (body) {
        if (state.deep.query !== query) return; // stale response
        state.deep.results = (body.results || []).map(deepRowToEntry);
        state.deep.status = 'live';
        renderResults();
        updateStatus();
      })
      .catch(function (err) {
        if (err && err.name === 'AbortError') return;
        state.deep.results = [];
        state.deep.status = 'off';
        updateStatus();
      });
  }

  /* ------------------------------------------------------------------ *
   * Search + render
   * ------------------------------------------------------------------ */

  function runSearch() {
    if (state.datasets === null) { loadIndexes(); return; }

    var query = state.query.trim();
    var hasTokens = state.tokens.length > 0;

    if (!query && !hasTokens) {
      state.deep.results = [];
      renderIdle();
      return;
    }

    // Deep search runs on free text only — committed tokens are already
    // structured filters, and a half-typed `modality:` fragment is not a
    // meaningful $text query.
    // Regex-mode queries never reach the deep API — $text would search
    // the pattern as literal words and return noise.
    var deepQuery = buildTerms(query).rx
      ? ''
      : query.replace(FACET_TOKEN_RE, ' ').replace(/\s+/g, ' ').trim();
    scheduleDeepSearch(deepQuery);
    renderResults();
  }

  function collectDatasetResults(terms) {
    var hasTokens = state.tokens.length > 0;
    var scored = [];
    var list = state.datasets || [];
    for (var i = 0; i < list.length; i++) {
      var entry = list[i];
      var ok = true;
      for (var j = 0; j < state.tokens.length; j++) {
        if (!tokenMatches(entry, state.tokens[j])) { ok = false; break; }
      }
      if (!ok) continue;
      var score = scoreDataset(entry, terms);
      if (score > 0 || (hasTokens && !terms.length && !terms.rx)) {
        scored.push({ entry: entry, score: score || (entry.subjects || 0) / 1e6 });
      }
    }
    scored.sort(function (a, b) {
      return b.score - a.score || (b.entry.subjects || 0) - (a.entry.subjects || 0);
    });
    return scored;
  }

  function collectDocResults(terms, group) {
    if (!terms.length && !terms.rx) return [];
    var scored = [];
    var list = state.docs || [];
    for (var i = 0; i < list.length; i++) {
      var entry = list[i];
      var inGroup = group === 'docs'
        ? (entry.g === 'docs' || entry.g === 'examples')
        : entry.g === group;
      if (!inGroup) continue;
      var score = scoreDoc(entry, terms);
      if (score > 0) scored.push({ entry: entry, score: score });
    }
    scored.sort(function (a, b) { return b.score - a.score; });
    return scored;
  }

  function renderResults() {
    var query = state.query.trim();
    var terms = buildTerms(query);
    var hasQuery = terms.length > 0 || !!terms.rx;
    var hasTokens = state.tokens.length > 0;
    var root = contentRoot();
    var html = '';
    state.items = [];

    // Token-value suggestions while typing `facet:prefix`
    var pending = pendingTokenInput();
    if (pending) {
      html += renderTokenSuggestions(pending);
    }

    var showData = state.scope === 'all' || state.scope === 'data';
    var showDocs = state.scope === 'all' || state.scope === 'docs';
    var showApi = state.scope === 'all' || state.scope === 'api';

    var totalShown = 0;

    if (showData && (hasQuery || hasTokens)) {
      var datasets = collectDatasetResults(terms);
      var capData = state.scope === 'data' ? 14 : MAX_PER_GROUP.datasets;
      if (datasets.length) {
        html += groupHeader('Datasets', datasets.length);
        datasets.slice(0, capData).forEach(function (item) {
          html += renderDatasetRow(item.entry, terms, false);
        });
        html += renderCatalogLink(datasets.length, query);
        totalShown += Math.min(datasets.length, capData);
      }

      // Deep matches the local index missed (README/task hits). Dedup
      // against ALL local matches, not just the rendered slice — a
      // dataset ranked past the cap is still locally found, and showing
      // it again under "Deep matches" would read as a duplicate.
      if (state.deep.results.length) {
        var localIds = {};
        datasets.forEach(function (item) { localIds[norm(item.entry.id)] = 1; });
        var fresh = state.deep.results.filter(function (entry) {
          if (localIds[norm(entry.id)]) return false;
          for (var j = 0; j < state.tokens.length; j++) {
            if (!tokenMatches(entry, state.tokens[j])) return false;
          }
          return true;
        });
        if (fresh.length) {
          html += groupHeader('Deep matches', fresh.length, 'README & task text — live from the EEGDash API');
          fresh.slice(0, MAX_PER_GROUP.deep).forEach(function (entry) {
            html += renderDatasetRow(entry, terms, true);
          });
          totalShown += Math.min(fresh.length, MAX_PER_GROUP.deep);
        }
      }
    }

    if (showDocs && hasQuery) {
      var docs = collectDocResults(terms, 'docs');
      if (docs.length) {
        html += groupHeader('Docs & tutorials', docs.length);
        docs.slice(0, MAX_PER_GROUP.docs).forEach(function (item) {
          html += renderDocRow(item.entry, terms);
        });
        totalShown += Math.min(docs.length, MAX_PER_GROUP.docs);
      }
    }

    if (showApi && hasQuery) {
      var api = collectDocResults(terms, 'api');
      if (api.length) {
        html += groupHeader('API reference', api.length);
        api.slice(0, MAX_PER_GROUP.api).forEach(function (item) {
          html += renderDocRow(item.entry, terms);
        });
        totalShown += Math.min(api.length, MAX_PER_GROUP.api);
      }
    }

    if (hasQuery) {
      html += '<a class="eds-row eds-row--escape" href="' + esc(root + 'search.html?q=' + encodeURIComponent(query)) + '">' +
        '<span class="eds-escape-icon">≋</span>' +
        '<span class="eds-escape-text">Full-text search for “' + esc(query) + '” across all pages</span>' +
        '<span class="eds-escape-hint">Sphinx index</span></a>';
    }

    if (!totalShown && !pending && (hasQuery || hasTokens)) {
      var deepNote = state.deep.status === 'loading'
        ? '<div class="eds-empty-deep">listening for deep matches…</div>' : '';
      html = '<div class="eds-empty">' +
        '<div class="eds-empty-title">No matches in the local index</div>' +
        '<div class="eds-empty-hint">Try fewer terms, filter tokens like ' +
        '<code>modality:eeg</code> <code>task:rest</code> <code>channels:&gt;=64</code>, ' +
        'or a regex like <code>eeg|meg</code> <code>^ds00</code></div>' +
        deepNote + '</div>' + html;
    }

    els.results.innerHTML = html;
    state.selected = -1;
    state.items = Array.prototype.slice.call(els.results.querySelectorAll('[data-eds-item]'));
    if (state.items.length) select(0);
    announce(totalShown);
  }

  function renderIdle() {
    var recent = readRecent();
    var html = '';
    state.items = [];

    html += '<div class="eds-idle">';
    if (recent.length) {
      html += groupHeader('Recent', recent.length);
      recent.forEach(function (q) {
        html += '<button type="button" class="eds-row eds-row--recent" data-eds-item data-eds-query="' + esc(q) + '">' +
          '<span class="eds-recent-glyph">↺</span><span>' + esc(q) + '</span></button>';
      });
    }
    html += groupHeader('Try', 4, 'free text + filter tokens compose');
    [
      ['face recognition', 'find datasets by experiment'],
      ['task:rest channels:>=64', 'high-density resting state'],
      ['pathology:epilepsy', 'clinical cohorts'],
      ['eeg|meg', 'regex works too'],
    ].forEach(function (pair) {
      html += '<button type="button" class="eds-row eds-row--try" data-eds-item data-eds-query="' + esc(pair[0]) + '">' +
        '<span class="eds-try-query">' + esc(pair[0]) + '</span>' +
        '<span class="eds-try-note">' + esc(pair[1]) + '</span></button>';
    });
    html += '</div>';

    els.results.innerHTML = html;
    state.items = Array.prototype.slice.call(els.results.querySelectorAll('[data-eds-item]'));
    state.selected = -1;
    announce(0);
  }

  // count == null renders a hint-only header (no dangling "0").
  function groupHeader(label, count, note) {
    return '<div class="eds-group" role="presentation">' +
      '<span class="eds-group-label">' + esc(label) + '</span>' +
      (count == null ? '' : '<span class="eds-group-count">' + count + '</span>') +
      (note ? '<span class="eds-group-note">' + esc(note) + '</span>' : '') +
      '</div>';
  }

  function renderDatasetRow(entry, terms, isDeep) {
    var chips = '';
    (entry.recordModality || []).slice(0, 2).forEach(function (m) {
      chips += '<span class="eds-chip eds-chip--modality">' + esc(m) + '</span>';
    });
    if (entry.source) chips += '<span class="eds-chip eds-chip--source">' + esc(entry.source) + '</span>';
    (entry.pathology || []).slice(0, 1).forEach(function (p) {
      chips += '<span class="eds-chip eds-chip--pathology">' + esc(p) + '</span>';
    });

    var stats = [];
    if (entry.subjects) stats.push(entry.subjects + ' sub');
    if (entry.nchans && entry.nchans.length) stats.push(entry.nchans[0] + ' ch');
    if (entry.sfreq && entry.sfreq.length) stats.push(entry.sfreq[0] + ' Hz');

    var snippet = '';
    if (isDeep && entry.snippet) {
      snippet = '<div class="eds-snippet">' + highlight(entry.snippet, terms) + '</div>';
    }

    return '<a class="eds-row eds-row--dataset' + (isDeep ? ' eds-row--deep' : '') + '" data-eds-item href="' + esc(datasetHref(entry)) + '">' +
      '<span class="eds-ds-id">' + highlight(String(entry.id).toUpperCase(), terms) + '</span>' +
      '<span class="eds-ds-main">' +
      '<span class="eds-ds-title">' + (entry.title ? highlight(entry.title, terms) : '<em>Untitled dataset</em>') + '</span>' +
      '<span class="eds-ds-chips">' + chips + '</span>' +
      snippet +
      '</span>' +
      '<span class="eds-ds-stats">' + esc(stats.join(' · ')) + '</span>' +
      '</a>';
  }

  function renderDocRow(entry, terms) {
    var root = contentRoot();
    var section = entry._section;
    var href = root + entry.u + '.html' + (section ? section[1] : '');
    var label = section ? section[0] : entry.t;
    var crumb = section ? entry.t : entry.u;
    var glyph = entry.g === 'api' ? '{ }' : (entry.g === 'examples' ? '▶' : '§');
    return '<a class="eds-row eds-row--doc" data-eds-item href="' + esc(href) + '">' +
      '<span class="eds-doc-glyph" aria-hidden="true">' + glyph + '</span>' +
      '<span class="eds-ds-main">' +
      '<span class="eds-doc-title">' + highlight(label, terms) + '</span>' +
      '<span class="eds-doc-crumb">' + esc(crumb) + '</span>' +
      '</span></a>';
  }

  function renderCatalogLink(total, query) {
    var root = contentRoot();
    var params = [];
    if (query) params.push('q=' + encodeURIComponent(query));
    state.tokens.forEach(function (token) {
      if (token.key === 'modality' && token.value) params.push('modality=' + encodeURIComponent(token.value));
      if (token.key === 'type' && token.value) params.push('task=' + encodeURIComponent(token.value));
    });
    // The catalog table only understands q/modality/task params — when
    // any other token is active the count promise would be a lie (the
    // table would show more rows than "all N"), so drop the count.
    var forwardable = state.tokens.every(function (token) {
      return token.key === 'modality' || token.key === 'type';
    });
    var label = forwardable
      ? 'Open all ' + total + ' in the catalog table'
      : 'Open the catalog table';
    var href = root + 'dataset_summary/table.html' + (params.length ? '?' + params.join('&') : '');
    return '<a class="eds-row eds-row--more" data-eds-item href="' + esc(href) + '">' +
      label + ' <span class="eds-more-arrow">→</span></a>';
  }

  /* ----- token suggestion UI ----- */

  function pendingTokenInput() {
    var value = els.input.value;
    var match = value.match(/(?:^|\s)([a-z]+):([^\s]*)$/i);
    if (!match) return null;
    var key = norm(match[1]);
    if (!FACETS[key]) return null;
    return { key: key, prefix: match[2] };
  }

  function renderTokenSuggestions(pending) {
    var facet = FACETS[pending.key];
    var html = '';
    if (facet.type === 'numeric') {
      html += groupHeader(pending.key + ':', null, facet.hint);
      return html;
    }
    var values = (state.facetValues[pending.key] || []).filter(function (v) {
      return !pending.prefix || norm(v).indexOf(norm(pending.prefix)) !== -1;
    }).slice(0, 6);
    if (!values.length) return '';
    html += groupHeader('Filter ' + pending.key + ':', values.length, facet.hint);
    values.forEach(function (v) {
      html += '<button type="button" class="eds-row eds-row--facet" data-eds-item data-eds-token="' +
        esc(pending.key + ':' + v) + '"><span class="eds-chip eds-chip--' + esc(pending.key) + '">' +
        esc(pending.key) + '</span><span class="eds-facet-value">' + esc(v) + '</span></button>';
    });
    return html;
  }

  /* ------------------------------------------------------------------ *
   * Tokens (commit / render / pop)
   * ------------------------------------------------------------------ */

  function commitToken(raw) {
    var token = parseToken(raw);
    if (!token) return false;
    state.tokens.push(token);
    renderTokens();
    return true;
  }

  function popToken() {
    if (!state.tokens.length) return;
    state.tokens.pop();
    renderTokens();
    runSearch();
  }

  function removeToken(index) {
    state.tokens.splice(index, 1);
    renderTokens();
    runSearch();
    els.input.focus();
  }

  function renderTokens() {
    var html = '';
    state.tokens.forEach(function (token, i) {
      var valueLabel = token.value != null ? token.value : (token.op === '=' ? '' + token.num : token.op + token.num);
      html += '<span class="eds-token eds-token--' + esc(token.key) + '">' +
        '<span class="eds-token-key">' + esc(token.key) + '</span>' +
        '<span class="eds-token-value">' + esc(valueLabel) + '</span>' +
        '<button type="button" class="eds-token-x" data-eds-remove="' + i + '" aria-label="Remove filter ' + esc(token.raw) + '">×</button>' +
        '</span>';
    });
    els.tokens.innerHTML = html;
  }

  // Commit a trailing "key:value" token out of the input into a chip.
  // requireSpace=true is the mid-typing form ("key:value " with trailing
  // whitespace, fired on every input event) and keeps a trailing space
  // after any remaining free text so continued typing doesn't glue words;
  // requireSpace=false is the Enter form (no trailing space needed).
  function commitTrailingToken(requireSpace) {
    var value = els.input.value;
    var re = requireSpace ? /(?:^|\s)([a-z]+:[^\s]+)\s$/i : /(?:^|\s)([a-z]+:[^\s]+)\s*$/i;
    var match = value.match(re);
    if (!match || !parseToken(match[1])) return false;
    commitToken(match[1]);
    var rest = value.slice(0, match.index).trimEnd();
    els.input.value = requireSpace && rest ? rest + ' ' : rest;
    state.query = els.input.value;
    return true;
  }

  /* ------------------------------------------------------------------ *
   * Selection + keyboard
   * ------------------------------------------------------------------ */

  function select(index) {
    if (!state.items.length) { state.selected = -1; return; }
    if (state.selected >= 0 && state.items[state.selected]) {
      state.items[state.selected].classList.remove('eds-selected');
      state.items[state.selected].removeAttribute('aria-selected');
    }
    state.selected = (index + state.items.length) % state.items.length;
    var el = state.items[state.selected];
    el.classList.add('eds-selected');
    el.setAttribute('aria-selected', 'true');
    el.scrollIntoView({ block: 'nearest' });
  }

  function activate(el) {
    if (!el) return;
    var token = el.getAttribute('data-eds-token');
    if (token) {
      // Replace the pending `key:prefix` fragment with the committed token
      var pending = pendingTokenInput();
      if (pending) {
        els.input.value = els.input.value.replace(/([a-z]+):([^\s]*)$/i, '');
      }
      commitToken(token);
      state.query = els.input.value;
      els.input.focus();
      runSearch();
      return;
    }
    var query = el.getAttribute('data-eds-query');
    if (query) {
      setQueryFromString(query);
      return;
    }
    if (el.href) {
      pushRecent(serializeQuery());
      window.location.href = el.href;
    }
  }

  function serializeQuery() {
    var parts = state.tokens.map(function (t) { return t.raw; });
    if (state.query.trim()) parts.push(state.query.trim());
    return parts.join(' ');
  }

  function setQueryFromString(full) {
    state.tokens = [];
    var rest = [];
    full.trim().split(/\s+/).forEach(function (part) {
      if (!commitToken(part)) rest.push(part);
    });
    els.input.value = rest.join(' ');
    state.query = els.input.value;
    renderTokens();
    els.input.focus();
    runSearch();
  }

  function cycleScope(dir) {
    var idx = SCOPES.indexOf(state.scope);
    state.scope = SCOPES[(idx + dir + SCOPES.length) % SCOPES.length];
    renderScopes();
    runSearch();
  }

  function setScope(scope) {
    if (SCOPES.indexOf(scope) === -1) return;
    state.scope = scope;
    renderScopes();
    runSearch();
    els.input.focus();
  }

  function renderScopes() {
    Array.prototype.forEach.call(els.scopes.querySelectorAll('[data-eds-scope]'), function (btn) {
      var active = btn.getAttribute('data-eds-scope') === state.scope;
      btn.classList.toggle('eds-scope--active', active);
      btn.setAttribute('aria-selected', active ? 'true' : 'false');
    });
  }

  /* ------------------------------------------------------------------ *
   * Shell (DOM) + status
   * ------------------------------------------------------------------ */

  function buildShell() {
    if (state.built) return;
    state.built = true;

    var overlay = document.createElement('div');
    overlay.className = 'eds-overlay';
    overlay.hidden = true;
    overlay.innerHTML =
      '<div class="eds-palette" role="dialog" aria-modal="true" aria-label="EEGDash search">' +
      '  <div class="eds-head">' +
      '    <div class="eds-input-row">' +
      '      <span class="eds-sigil" aria-hidden="true">' +
      '        <svg viewBox="0 0 28 14" width="28" height="14"><path class="eds-sigil-path" d="M0 7 H6 L8 7 10 2 12 12 14 4 16 10 18 7 H28" fill="none" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>' +
      '      </span>' +
      '      <div class="eds-tokens" aria-live="polite"></div>' +
      '      <input class="eds-input" type="text" role="combobox" aria-expanded="true" aria-autocomplete="list" aria-controls="eds-listbox" autocomplete="off" autocorrect="off" autocapitalize="off" spellcheck="false" placeholder="Search datasets, tasks, docs…  try modality:eeg channels:>=64" />' +
      '      <kbd class="eds-esc">esc</kbd>' +
      '    </div>' +
      '    <div class="eds-trace" aria-hidden="true">' +
      '      <svg preserveAspectRatio="none" viewBox="0 0 600 12"><path class="eds-trace-path" d="M0 6 H80 L90 6 95 1 100 11 105 3 110 9 115 6 H230 L240 6 245 2 250 10 255 4 260 8 265 6 H400 L410 6 415 1 420 11 425 3 430 9 435 6 H600" fill="none" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"/></svg>' +
      '    </div>' +
      '    <div class="eds-scopes" role="tablist" aria-label="Search scope">' +
      SCOPES.map(function (scope) {
        return '<button type="button" role="tab" class="eds-scope" data-eds-scope="' + scope + '" aria-selected="' + (scope === 'all') + '">' + SCOPE_LABELS[scope] + '</button>';
      }).join('') +
      '      <span class="eds-scope-hint">tab to switch</span>' +
      '    </div>' +
      '  </div>' +
      '  <div class="eds-results" id="eds-listbox" role="listbox" aria-label="Search results"></div>' +
      '  <div class="eds-status">' +
      '    <span class="eds-status-counts">indexing…</span>' +
      '    <span class="eds-status-deep" title="Deep search hits the live EEGDash API for README and task text"><span class="eds-dot"></span><span class="eds-deep-label">deep search</span></span>' +
      '    <span class="eds-status-keys"><kbd>↑↓</kbd> navigate <kbd>↵</kbd> open <kbd>⌫</kbd> pop filter</span>' +
      '  </div>' +
      '  <div class="eds-sr-live" role="status" aria-live="polite"></div>' +
      '</div>';

    document.body.appendChild(overlay);

    els.overlay = overlay;
    els.palette = overlay.querySelector('.eds-palette');
    els.input = overlay.querySelector('.eds-input');
    els.tokens = overlay.querySelector('.eds-tokens');
    els.results = overlay.querySelector('.eds-results');
    els.scopes = overlay.querySelector('.eds-scopes');
    els.status = overlay.querySelector('.eds-status-counts');
    els.deepDot = overlay.querySelector('.eds-status-deep');
    els.trace = overlay.querySelector('.eds-trace');
    els.live = overlay.querySelector('.eds-sr-live');

    // -- events --
    overlay.addEventListener('mousedown', function (e) {
      if (e.target === overlay) { close(); return; }
      // Keep focus pinned to the input: the palette is aria-modal and
      // fully keyboard-driven, so no other element inside it may take
      // focus (clicks still fire — only the focus transfer is blocked).
      // Otherwise Tab escapes to the page behind the overlay.
      if (e.target !== els.input) e.preventDefault();
    });

    // Belt and braces for programmatic focus moves: anything that lands
    // focus outside the palette while it is open gets pulled back.
    document.addEventListener('focusin', function (e) {
      if (state.open && els.palette && !els.palette.contains(e.target)) {
        els.input.focus();
      }
    });

    els.input.addEventListener('input', function () {
      commitTrailingToken(true);
      state.query = els.input.value;
      setTrace(true);
      clearTimeout(searchTimer);
      searchTimer = setTimeout(function () {
        setTrace(false);
        runSearch();
      }, DEBOUNCE_MS);
    });

    els.input.addEventListener('keydown', function (e) {
      if (e.key === 'ArrowDown') { e.preventDefault(); select(state.selected + 1); }
      else if (e.key === 'ArrowUp') { e.preventDefault(); select(state.selected - 1); }
      else if (e.key === 'Enter') {
        e.preventDefault();
        if (state.selected >= 0 && state.items[state.selected]) activate(state.items[state.selected]);
        else if (els.input.value.trim() && commitTrailingToken(false)) runSearch();
      }
      else if (e.key === 'Tab') { e.preventDefault(); cycleScope(e.shiftKey ? -1 : 1); }
      else if (e.key === 'Backspace' && !els.input.value) { popToken(); }
      else if (e.key === 'Escape') { e.preventDefault(); close(); }
    });

    els.tokens.addEventListener('click', function (e) {
      var btn = e.target.closest('[data-eds-remove]');
      if (!btn) return;
      var idx = parseInt(btn.getAttribute('data-eds-remove'), 10);
      // NaN would splice(0, 1) and silently remove the wrong token.
      if (!isNaN(idx)) removeToken(idx);
    });

    els.scopes.addEventListener('click', function (e) {
      var btn = e.target.closest('[data-eds-scope]');
      if (btn) setScope(btn.getAttribute('data-eds-scope'));
    });

    // mousedown (not click) so the input never loses focus first
    els.results.addEventListener('mousedown', function (e) {
      var item = e.target.closest('[data-eds-item]');
      if (item) { e.preventDefault(); activate(item); }
    });
    els.results.addEventListener('mousemove', function (e) {
      var item = e.target.closest('[data-eds-item]');
      if (item) {
        var idx = state.items.indexOf(item);
        if (idx !== -1 && idx !== state.selected) select(idx);
      }
    });
  }

  function announce(count) {
    if (els.live) {
      els.live.textContent = count > 0
        ? count + ' results. Use arrow keys to navigate.'
        : '';
    }
  }

  function updateStatus() {
    if (!els.status) return;
    var nDatasets = state.datasets ? state.datasets.length : 0;
    var nDocs = state.docs ? state.docs.length : 0;
    els.status.textContent = state.loading
      ? 'indexing…'
      : nDatasets + ' datasets · ' + nDocs + ' pages indexed';
    if (els.deepDot) {
      els.deepDot.setAttribute('data-state', state.deep.status);
      var label = els.deepDot.querySelector('.eds-deep-label');
      if (label) {
        label.textContent = state.deep.status === 'off' ? 'deep search offline'
          : state.deep.status === 'loading' ? 'deep search…'
          : 'deep search';
      }
    }
  }

  function setTrace(active) {
    if (els.trace) els.trace.classList.toggle('eds-trace--active', !!active);
  }

  /* ------------------------------------------------------------------ *
   * Open / close / global wiring
   * ------------------------------------------------------------------ */

  var lastFocused = null;

  // The pydata theme (0.16.x) binds its own Ctrl/Cmd+K on *window* in
  // CAPTURE phase (verified in its minified bundle: `window.add…,!0)`),
  // which fires before any document-level listener regardless of
  // registration order — so its #pst-search-dialog may already be open
  // by the time the palette reacts. Close it: the palette owns search.
  function closeThemeSearchDialog() {
    var dlg = document.getElementById('pst-search-dialog');
    if (dlg && dlg.open && typeof dlg.close === 'function') dlg.close();
  }

  function open(prefill) {
    buildShell();
    closeThemeSearchDialog();
    // The homepage hero autocomplete hides itself on blur after 200ms —
    // hide it immediately so it doesn't linger under the overlay.
    var heroDropdown = document.querySelector('.hf-autocomplete-dropdown');
    if (heroDropdown) heroDropdown.style.display = 'none';
    if (state.open) { els.input.focus(); return; }
    state.open = true;
    lastFocused = document.activeElement;
    els.overlay.hidden = false;
    document.documentElement.classList.add('eds-lock');
    // Re-trigger the entrance animation
    els.palette.classList.remove('eds-enter');
    void els.palette.offsetWidth;
    els.palette.classList.add('eds-enter');
    if (typeof prefill === 'string' && prefill) {
      setQueryFromString(prefill);
    } else {
      els.input.focus();
      if (serializeQuery()) runSearch(); else renderIdle();
    }
    loadIndexes();
    updateStatus();
  }

  function close() {
    if (!state.open) return;
    state.open = false;
    els.overlay.hidden = true;
    document.documentElement.classList.remove('eds-lock');
    if (lastFocused && lastFocused.focus) lastFocused.focus();
  }

  function init() {
    // Global shortcut — capture phase. The theme's own Ctrl+K handler
    // sits on window-capture and therefore ALWAYS fires before this one
    // (window precedes document in the capture descent), so we cannot
    // suppress it; instead closeThemeSearchDialog() undoes its dialog in
    // both toggle directions. The cleaner long-term seam is overriding
    // _templates/searchbox.html so the theme never binds search at all.
    document.addEventListener('keydown', function (e) {
      if ((e.ctrlKey || e.metaKey) && !e.altKey && (e.key === 'k' || e.key === 'K')) {
        e.preventDefault();
        e.stopImmediatePropagation();
        closeThemeSearchDialog();
        if (state.open) close(); else open();
      } else if (e.key === 'Escape' && state.open) {
        e.preventDefault();
        close();
      }
    }, true);

    // Intercept the theme's navbar search buttons.
    document.addEventListener('click', function (e) {
      var trigger = e.target.closest('.search-button__button, [data-open-eegdash-search]');
      if (!trigger) return;
      e.preventDefault();
      e.stopImmediatePropagation();
      var prefill = trigger.getAttribute('data-eds-prefill') || '';
      open(prefill);
    }, true);

    // ?eds-q= deep link (lets other pages open the palette pre-filled)
    try {
      var params = new URLSearchParams(window.location.search);
      var q = params.get('eds-q');
      if (q) open(q);
    } catch (err) { /* very old browsers: ignore */ }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
