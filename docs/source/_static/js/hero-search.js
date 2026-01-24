(function () {
  'use strict';

  // Configuration
  const CONFIG = {
    debounceMs: 150,
    maxResults: 8,
    minChars: 2,
    fuseOptions: {
      keys: [
        { name: 'id', weight: 2.0 },
        { name: 'title', weight: 1.5 },
        { name: 'source', weight: 0.8 },
        { name: 'pathology', weight: 1.0 },
        { name: 'modality', weight: 1.0 },
        { name: 'type', weight: 1.0 },
        { name: 'recordModality', weight: 0.9 },
      ],
      threshold: 0.4,
      distance: 100,
      ignoreLocation: true,
      includeMatches: true,
      includeScore: true,
      minMatchCharLength: 2,
    },
  };

  let fuse = null;
  let searchIndex = [];
  let dropdownEl = null;
  let selectedIndex = -1;

  // Utility: debounce function calls
  function debounce(fn, ms) {
    let timer = null;
    return function (...args) {
      clearTimeout(timer);
      timer = setTimeout(() => fn.apply(this, args), ms);
    };
  }

  // Utility: escape HTML to prevent XSS
  function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  // Highlight matched text with <mark> tags
  function highlightMatches(text, indices) {
    if (!indices || !indices.length || !text) return escapeHtml(text || '');

    const sorted = [...indices].sort((a, b) => a[0] - b[0]);
    const merged = [];
    for (const [start, end] of sorted) {
      if (merged.length && start <= merged[merged.length - 1][1] + 1) {
        merged[merged.length - 1][1] = Math.max(merged[merged.length - 1][1], end);
      } else {
        merged.push([start, end]);
      }
    }

    let result = '';
    let lastEnd = 0;
    for (const [start, end] of merged) {
      result += escapeHtml(text.slice(lastEnd, start));
      result += `<mark class="hf-highlight">${escapeHtml(text.slice(start, end + 1))}</mark>`;
      lastEnd = end + 1;
    }
    result += escapeHtml(text.slice(lastEnd));
    return result;
  }

  // Get match indices for a specific key from Fuse.js results
  function getMatchIndicesForKey(matches, key) {
    if (!matches) return null;
    const match = matches.find(m => m.key === key);
    return match ? match.indices : null;
  }

  // Create the dropdown container
  function createDropdown(inputEl) {
    const wrapper = inputEl.closest('.hf-search-input-wrap') || inputEl.parentElement;

    const dropdown = document.createElement('div');
    dropdown.className = 'hf-autocomplete-dropdown';
    dropdown.setAttribute('role', 'listbox');
    dropdown.setAttribute('aria-label', 'Search suggestions');
    dropdown.style.display = 'none';

    wrapper.parentElement.insertBefore(dropdown, wrapper.nextSibling);
    return dropdown;
  }

  // Render a single search result item
  function renderResult(result, index) {
    const item = result.item;
    const matches = result.matches || [];

    const idIndices = getMatchIndicesForKey(matches, 'id');
    const titleIndices = getMatchIndicesForKey(matches, 'title');

    const idHtml = highlightMatches(item.id.toUpperCase(), idIndices);
    const titleHtml = item.title
      ? highlightMatches(item.title, titleIndices)
      : '<em>Untitled dataset</em>';

    // Build tags
    const tags = [];
    if (item.source) {
      tags.push(`<span class="hf-tag hf-tag-source">${escapeHtml(item.source)}</span>`);
    }
    if (item.recordModality && item.recordModality.length) {
      item.recordModality.slice(0, 2).forEach(mod => {
        tags.push(`<span class="hf-tag hf-tag-modality">${escapeHtml(mod)}</span>`);
      });
    }
    if (item.pathology && item.pathology.length) {
      item.pathology.slice(0, 1).forEach(path => {
        tags.push(`<span class="hf-tag hf-tag-pathology">${escapeHtml(path)}</span>`);
      });
    }

    // Stats line
    const stats = [];
    if (item.subjects) stats.push(`${item.subjects} subjects`);
    if (item.records) stats.push(`${item.records} recordings`);
    if (item.size) stats.push(item.size);

    return `
      <a href="${escapeHtml(item.url)}"
         class="hf-autocomplete-item"
         role="option"
         data-index="${index}"
         aria-selected="false">
        <div class="hf-ac-header">
          <span class="hf-ac-id">${idHtml}</span>
          <div class="hf-ac-tags">${tags.join('')}</div>
        </div>
        <div class="hf-ac-title">${titleHtml}</div>
        ${stats.length ? `<div class="hf-ac-stats">${stats.join(' · ')}</div>` : ''}
      </a>
    `;
  }

  // Render the dropdown with results
  function renderDropdown(results, query) {
    if (!dropdownEl) return;

    if (!results.length || query.length < CONFIG.minChars) {
      dropdownEl.style.display = 'none';
      selectedIndex = -1;
      return;
    }

    const limited = results.slice(0, CONFIG.maxResults);
    const html = limited.map((r, i) => renderResult(r, i)).join('');

    const viewAllHtml = results.length > CONFIG.maxResults
      ? `<a href="dataset_summary/table.html?q=${encodeURIComponent(query)}" class="hf-ac-view-all">
           View all ${results.length} results →
         </a>`
      : '';

    dropdownEl.innerHTML = html + viewAllHtml;
    dropdownEl.style.display = 'block';
    selectedIndex = -1;
  }

  // Update keyboard selection
  function updateSelection(direction) {
    const items = dropdownEl.querySelectorAll('.hf-autocomplete-item');
    if (!items.length) return;

    if (selectedIndex >= 0 && items[selectedIndex]) {
      items[selectedIndex].classList.remove('hf-ac-selected');
      items[selectedIndex].setAttribute('aria-selected', 'false');
    }

    if (direction === 'down') {
      selectedIndex = selectedIndex < items.length - 1 ? selectedIndex + 1 : 0;
    } else {
      selectedIndex = selectedIndex > 0 ? selectedIndex - 1 : items.length - 1;
    }

    items[selectedIndex].classList.add('hf-ac-selected');
    items[selectedIndex].setAttribute('aria-selected', 'true');
    items[selectedIndex].scrollIntoView({ block: 'nearest' });
  }

  // Perform fuzzy search
  function performSearch(query) {
    if (!fuse || query.length < CONFIG.minChars) {
      renderDropdown([], query);
      return;
    }
    const results = fuse.search(query);
    renderDropdown(results, query);
  }

  const debouncedSearch = debounce(performSearch, CONFIG.debounceMs);

  // Event handlers
  function handleInput(e) {
    const query = (e.target.value || '').trim();
    debouncedSearch(query);
  }

  function handleKeydown(e) {
    if (!dropdownEl || dropdownEl.style.display === 'none') return;

    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        updateSelection('down');
        break;
      case 'ArrowUp':
        e.preventDefault();
        updateSelection('up');
        break;
      case 'Enter':
        if (selectedIndex >= 0) {
          e.preventDefault();
          const items = dropdownEl.querySelectorAll('.hf-autocomplete-item');
          if (items[selectedIndex]) {
            window.location.href = items[selectedIndex].href;
          }
        }
        break;
      case 'Escape':
        dropdownEl.style.display = 'none';
        selectedIndex = -1;
        break;
    }
  }

  function handleBlur() {
    setTimeout(() => {
      if (dropdownEl && !dropdownEl.contains(document.activeElement)) {
        dropdownEl.style.display = 'none';
        selectedIndex = -1;
      }
    }, 200);
  }

  function handleFocus(e) {
    const query = (e.target.value || '').trim();
    if (query.length >= CONFIG.minChars && fuse) {
      performSearch(query);
    }
  }

  // Load search index from JSON
  async function loadSearchIndex() {
    try {
      // Determine base path for static files
      let basePath = '_static/dataset_generated/';

      // Check if we're in a subdirectory
      const pathParts = window.location.pathname.split('/');
      const depth = pathParts.filter(p => p && !p.includes('.')).length;
      if (depth > 0) {
        basePath = '../'.repeat(depth) + basePath;
      }

      const response = await fetch(basePath + 'search_index.json');
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      searchIndex = await response.json();

      if (typeof Fuse !== 'undefined') {
        fuse = new Fuse(searchIndex, CONFIG.fuseOptions);
        console.log(`[EEGDash] Search index loaded: ${searchIndex.length} datasets`);
      } else {
        console.warn('[EEGDash] Fuse.js not loaded');
      }
    } catch (err) {
      console.warn('[EEGDash] Failed to load search index:', err.message);
    }
  }

  // Initialize hero search
  function initHeroSearch() {
    if (typeof document === 'undefined') return;

    const input = document.querySelector('.hf-search-input, #hf-search-input');
    if (!input) return;

    // Hide static suggestions (replaced by dynamic autocomplete)
    const staticSuggest = document.querySelector('.hf-search-suggest');
    if (staticSuggest) {
      staticSuggest.style.display = 'none';
    }

    // Create dropdown
    dropdownEl = createDropdown(input);

    // Attach event listeners
    input.addEventListener('input', handleInput);
    input.addEventListener('keydown', handleKeydown);
    input.addEventListener('blur', handleBlur);
    input.addEventListener('focus', handleFocus);

    // ARIA attributes
    input.setAttribute('role', 'combobox');
    input.setAttribute('aria-autocomplete', 'list');
    input.setAttribute('aria-expanded', 'false');

    // Load search index
    loadSearchIndex();
  }

  // Start initialization
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initHeroSearch);
  } else {
    initHeroSearch();
  }
})();
