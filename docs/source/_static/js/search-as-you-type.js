/**
 * Search-as-you-type enhancement for PyData Sphinx Theme
 * Uses Sphinx's search index to show live results with regex support
 */
(function () {
  'use strict';

  const CONFIG = {
    debounceMs: 150,
    minChars: 2,
    maxResults: 20,
  };

  let indexData = null;
  let resultsContainer = null;
  let searchInput = null;
  let selectedIndex = -1;

  // Debounce utility
  function debounce(fn, ms) {
    let timer = null;
    return function (...args) {
      clearTimeout(timer);
      timer = setTimeout(() => fn.apply(this, args), ms);
    };
  }

  // Escape HTML
  function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str || '';
    return div.innerHTML;
  }

  // Escape regex special chars for literal matching
  function escapeRegex(str) {
    return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }

  // Load the Sphinx search index
  async function loadSearchIndex() {
    if (indexData) return;

    try {
      const contentRoot = document.documentElement.dataset.content_root || '';
      const response = await fetch(contentRoot + 'searchindex.js');
      const text = await response.text();

      // Parse the JSON from Search.setIndex({...})
      const jsonStr = text.replace(/^Search\.setIndex\(/, '').replace(/\)$/, '');
      indexData = JSON.parse(jsonStr);
      console.log('[EEGDash] Search index loaded:', indexData.docnames.length, 'documents');
    } catch (err) {
      console.error('[EEGDash] Failed to load search index:', err);
    }
  }

  // Build regex from query - supports basic patterns
  function buildSearchRegex(query) {
    // Check if query looks like a regex (contains special chars)
    const hasRegexChars = /[.*+?^${}()|[\]\\]/.test(query);

    if (hasRegexChars) {
      try {
        return new RegExp(query, 'i');
      } catch (e) {
        // Invalid regex, fall back to escaped literal
        return new RegExp(escapeRegex(query), 'i');
      }
    }

    // For normal queries, create a fuzzy-ish pattern
    // Split into words and require all to match
    const words = query.toLowerCase().split(/\s+/).filter(w => w.length >= 2);
    if (words.length === 0) return null;

    // Create pattern that matches any of the words
    const pattern = words.map(w => escapeRegex(w)).join('|');
    return new RegExp(pattern, 'gi');
  }

  // Search the index
  function performSearch(query) {
    if (!indexData || !query || query.length < CONFIG.minChars) {
      return [];
    }

    const regex = buildSearchRegex(query);
    if (!regex) return [];

    const results = [];
    const seen = new Set();
    const queryLower = query.toLowerCase();
    const queryWords = queryLower.split(/\s+/).filter(w => w.length >= 2);

    const { docnames, titles, terms, titleterms, alltitles } = indexData;

    // Helper to add result
    const addResult = (docIdx, score, matchType, matchText) => {
      if (docIdx < 0 || docIdx >= docnames.length) return;
      const key = docnames[docIdx];
      if (seen.has(key)) {
        // Update score if higher
        const existing = results.find(r => r.docName === key);
        if (existing && score > existing.score) {
          existing.score = score;
          existing.matchType = matchType;
          existing.matchText = matchText;
        }
        return;
      }
      seen.add(key);
      results.push({
        docName: docnames[docIdx],
        title: titles[docIdx] || docnames[docIdx],
        score: score,
        matchType: matchType,
        matchText: matchText,
        anchor: '',
      });
    };

    // 1. Search document titles (highest priority)
    for (let i = 0; i < titles.length; i++) {
      const title = titles[i] || '';
      if (regex.test(title)) {
        let score = 100;
        // Boost for exact word match
        if (queryWords.some(w => title.toLowerCase().includes(w))) {
          score += 20;
        }
        addResult(i, score, 'title', title);
      }
    }

    // 2. Search titleterms (words in titles)
    if (titleterms) {
      for (const [term, docIndices] of Object.entries(titleterms)) {
        if (regex.test(term)) {
          const docs = Array.isArray(docIndices) ? docIndices : [docIndices];
          for (const docIdx of docs) {
            addResult(docIdx, 80, 'titleterm', term);
          }
        }
      }
    }

    // 3. Search section titles (alltitles)
    if (alltitles) {
      for (const [sectionTitle, locations] of Object.entries(alltitles)) {
        if (regex.test(sectionTitle)) {
          if (Array.isArray(locations)) {
            for (const loc of locations.slice(0, 3)) {
              const docIdx = Array.isArray(loc) ? loc[0] : loc;
              const anchor = Array.isArray(loc) ? loc[1] : '';
              if (docIdx < docnames.length) {
                const key = docnames[docIdx] + '#' + anchor;
                if (!seen.has(key)) {
                  seen.add(key);
                  results.push({
                    docName: docnames[docIdx],
                    title: sectionTitle,
                    parentTitle: titles[docIdx],
                    score: 60,
                    matchType: 'section',
                    matchText: sectionTitle,
                    anchor: anchor ? '#' + anchor : '',
                  });
                }
              }
            }
          }
        }
      }
    }

    // 4. Search body terms (words in content)
    if (terms) {
      for (const [term, docIndices] of Object.entries(terms)) {
        if (regex.test(term)) {
          const docs = Array.isArray(docIndices) ? docIndices : [docIndices];
          for (const docIdx of docs) {
            addResult(docIdx, 40, 'content', term);
          }
        }
      }
    }

    // Sort by score descending
    results.sort((a, b) => b.score - a.score);

    return results.slice(0, CONFIG.maxResults);
  }

  // Highlight matching text
  function highlightMatch(text, query) {
    if (!query || !text) return escapeHtml(text || '');

    const regex = buildSearchRegex(query);
    if (!regex) return escapeHtml(text);

    // Reset regex lastIndex
    regex.lastIndex = 0;

    let result = '';
    let lastIndex = 0;
    let match;

    // Create a new regex for iteration
    const iterRegex = new RegExp(regex.source, 'gi');

    while ((match = iterRegex.exec(text)) !== null) {
      result += escapeHtml(text.slice(lastIndex, match.index));
      result += '<mark class="sayt-highlight">' + escapeHtml(match[0]) + '</mark>';
      lastIndex = iterRegex.lastIndex;

      // Prevent infinite loop for zero-length matches
      if (match[0].length === 0) break;
    }

    result += escapeHtml(text.slice(lastIndex));
    return result;
  }

  // Get URL for a result
  function getResultUrl(result) {
    const contentRoot = document.documentElement.dataset.content_root || '';
    return contentRoot + result.docName + '.html' + (result.anchor || '');
  }

  // Get icon for match type
  function getMatchIcon(type) {
    switch (type) {
      case 'title': return '📄';
      case 'titleterm': return '📑';
      case 'section': return '§';
      case 'content': return '📝';
      default: return '•';
    }
  }

  // Render search results
  function renderResults(results, query) {
    if (!resultsContainer) return;

    if (results.length === 0) {
      if (query && query.length >= CONFIG.minChars) {
        resultsContainer.innerHTML = `
          <div class="sayt-no-results">
            No results found for "<strong>${escapeHtml(query)}</strong>"
            <div class="sayt-tip">Tip: Use regex patterns like <code>vis.*</code> or <code>eeg|meg</code></div>
          </div>
        `;
      } else if (query) {
        resultsContainer.innerHTML = `
          <div class="sayt-hint">
            Type at least ${CONFIG.minChars} characters to search...
          </div>
        `;
      } else {
        resultsContainer.style.display = 'none';
        return;
      }
      resultsContainer.style.display = 'block';
      return;
    }

    const html = results.map((r, i) => {
      const icon = getMatchIcon(r.matchType);
      const subtitle = r.parentTitle && r.parentTitle !== r.title
        ? `<span class="sayt-parent">in ${escapeHtml(r.parentTitle)}</span>`
        : '';

      return `
        <a href="${escapeHtml(getResultUrl(r))}" class="sayt-result sayt-type-${r.matchType}" data-index="${i}">
          <span class="sayt-icon">${icon}</span>
          <div class="sayt-content">
            <div class="sayt-title">${highlightMatch(r.title, query)}</div>
            <div class="sayt-meta">
              <span class="sayt-path">${escapeHtml(r.docName)}</span>
              ${subtitle}
              <span class="sayt-badge">${r.matchType}</span>
            </div>
          </div>
        </a>
      `;
    }).join('');

    resultsContainer.innerHTML = `
      <div class="sayt-count">${results.length} result${results.length !== 1 ? 's' : ''}</div>
      ${html}
    `;
    resultsContainer.style.display = 'block';
    selectedIndex = -1;
  }

  // Handle search input
  async function handleSearch(query) {
    await loadSearchIndex();
    const results = performSearch(query);
    renderResults(results, query);
  }

  const debouncedSearch = debounce(handleSearch, CONFIG.debounceMs);

  // Create results container
  function createResultsContainer() {
    const container = document.createElement('div');
    container.id = 'sayt-results';
    container.className = 'sayt-results';
    container.style.display = 'none';
    return container;
  }

  // Update keyboard selection
  function updateSelection(items, index) {
    items.forEach((item, i) => {
      item.classList.toggle('sayt-selected', i === index);
    });
    if (index >= 0 && items[index]) {
      items[index].scrollIntoView({ block: 'nearest' });
    }
  }

  // Initialize search-as-you-type
  function init() {
    // Wait for the search dialog to be in the DOM
    const dialog = document.getElementById('pst-search-dialog');
    if (!dialog) {
      setTimeout(init, 500);
      return;
    }

    // Find the search input inside the dialog
    searchInput = dialog.querySelector('input[type="search"], input[name="q"], input[type="text"]');
    if (!searchInput) {
      console.warn('[EEGDash] Search input not found in dialog');
      return;
    }

    // Create and insert results container
    resultsContainer = createResultsContainer();
    const form = searchInput.closest('form');
    if (form) {
      form.after(resultsContainer);
    } else {
      searchInput.after(resultsContainer);
    }

    // Pre-load search index
    loadSearchIndex();

    // Add event listeners
    searchInput.addEventListener('input', (e) => {
      const query = (e.target.value || '').trim();
      debouncedSearch(query);
    });

    searchInput.addEventListener('focus', () => {
      const query = (searchInput.value || '').trim();
      if (query.length >= CONFIG.minChars) {
        handleSearch(query);
      }
    });

    // Handle keyboard navigation
    searchInput.addEventListener('keydown', (e) => {
      const items = resultsContainer.querySelectorAll('.sayt-result');
      if (!items.length) return;

      if (e.key === 'ArrowDown') {
        e.preventDefault();
        selectedIndex = Math.min(selectedIndex + 1, items.length - 1);
        updateSelection(items, selectedIndex);
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        selectedIndex = Math.max(selectedIndex - 1, -1);
        updateSelection(items, selectedIndex);
      } else if (e.key === 'Enter' && selectedIndex >= 0) {
        e.preventDefault();
        items[selectedIndex].click();
      }
    });

    // Handle mouse hover
    resultsContainer.addEventListener('mouseover', (e) => {
      const item = e.target.closest('.sayt-result');
      if (item) {
        selectedIndex = parseInt(item.dataset.index, 10);
        const items = resultsContainer.querySelectorAll('.sayt-result');
        updateSelection(items, selectedIndex);
      }
    });

    // Observer to show results when dialog opens
    const observer = new MutationObserver(() => {
      if (dialog.open && searchInput.value) {
        handleSearch(searchInput.value.trim());
      }
    });
    observer.observe(dialog, { attributes: true, attributeFilter: ['open'] });

    console.log('[EEGDash] Search-as-you-type initialized');
  }

  // Start when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
