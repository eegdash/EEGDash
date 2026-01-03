(function () {
  function initHeroSearch() {
    if (typeof document === 'undefined') return;
    const body = document.body;
    if (!body || !body.classList.contains('bd-page-index')) return;

    const input = document.querySelector('.hf-search-input');
    const suggestions = Array.from(document.querySelectorAll('.hf-suggest-link'));
    const container = document.querySelector('.hf-search-suggest');

    if (!input || !suggestions.length || !container) return;

    function updateSuggestions() {
      const value = (input.value || '').trim().toLowerCase();
      let matches = 0;

      suggestions.forEach((link) => {
        const haystack = (link.dataset.query || link.textContent || '').toLowerCase();
        const visible = !value || haystack.includes(value);
        link.style.display = visible ? '' : 'none';
        if (visible) matches += 1;
      });

      container.style.display = matches ? '' : 'none';
    }

    input.addEventListener('input', updateSuggestions);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initHeroSearch);
  } else {
    initHeroSearch();
  }
})();
