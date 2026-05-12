/* ============================================================
   lazy-embed.js — swap `data-src` → `src` on first <details>
   expansion so embedded iframes only load when users ask.

   Each dataset page may carry one or more <details> blocks with a
   single child <iframe data-src="…"> and no src yet. Expanding the
   <details> triggers the first (and only) load.

   Recognised selectors:
     details.electrode-explorer  — electrodes.eegdash.org topomap
     details.trace-viewer        — eegdash.github.io/eegdash-viewer
   Adding a third is just a matter of appending to LAZY_SELECTORS.
   ============================================================ */
(function () {
  'use strict';

  const LAZY_SELECTORS = [
    'details.electrode-explorer',
    'details.trace-viewer',
  ];

  function hydrateIframe(detailsEl) {
    const frame = detailsEl.querySelector('iframe[data-src]');
    if (!frame || frame.src) return;
    frame.src = frame.dataset.src;
  }

  function bind() {
    document.querySelectorAll(LAZY_SELECTORS.join(',')).forEach(d => {
      // If a user lands on an anchor that auto-expands the details,
      // the `open` flip fires before this script runs. Handle both.
      if (d.open) hydrateIframe(d);
      d.addEventListener('toggle', () => {
        if (d.open) hydrateIframe(d);
      });
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', bind);
  } else {
    bind();
  }
})();
