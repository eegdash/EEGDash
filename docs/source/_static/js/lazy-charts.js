/* lazy-charts.js — load Plotly and render the summary-page charts only when
   the user opens a chart tab. Pairs with the capture-stub inlined at the top
   of dataset_summary.rst. See docs/maintenance/docs-perf-plan.md. */
(function () {
  'use strict';
  var PLOTLY_SRC = 'https://cdn.plot.ly/plotly-3.1.0.min.js';
  var state = 'idle'; // idle -> loading -> loaded

  function drain() {
    var q = window.__plotlyQueue || [];
    var rec;
    while ((rec = q.shift())) {
      try {
        var p = window.Plotly.newPlot.apply(window.Plotly, rec.args);
        // Replay captured .then callbacks (e.g. the kde/ridgeline spinner-hide)
        // against the real render promise.
        if (rec.then && rec.then.length && p && typeof p.then === 'function') {
          rec.then.forEach(function (cb) { p = p.then(cb); });
        }
      } catch (e) { /* skip one bad chart */ }
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
