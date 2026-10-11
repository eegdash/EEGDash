/* EEGDash Corpus pages: one tooltip for every [data-tip], the toggles (hours/datasets, order), the tables.
   Each part is a no-op when its elements are absent. Loaded only on corpus/ pages (layout.html gate). */
(function () {
  "use strict";
  function all(sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); }

  // --- tooltip ---------------------------------------------------------------------------------------------
  var tip = document.createElement("div");
  tip.className = "corpus corpus-tip"; tip.hidden = true; tip.setAttribute("role", "tooltip");
  document.body.appendChild(tip);
  function place(x, y) {
    var w = tip.offsetWidth, h = tip.offsetHeight, l = x + 14, t = y + 14;
    if (l + w > innerWidth - 8) l = x - w - 14;
    if (t + h > innerHeight - 8) t = y - h - 14;
    tip.style.left = Math.max(4, l) + "px"; tip.style.top = Math.max(4, t) + "px";
  }
  window.corpusTip = function (text, x, y) {  // shared with corpus-3d.js
    if (!text) { tip.hidden = true; return; }
    tip.textContent = text; tip.hidden = false; place(x, y);
  };
  function fromEvent(ev) {
    var t = ev.target.closest && ev.target.closest(".corpus [data-tip]");
    if (t) window.corpusTip(t.getAttribute("data-tip"), ev.clientX, ev.clientY);
    else if (!ev.target.closest || !ev.target.closest(".corpus-3d-canvas")) tip.hidden = true;
  }
  document.addEventListener("mouseover", fromEvent);
  document.addEventListener("pointerdown", fromEvent);
  document.addEventListener("mousemove", function (ev) { if (!tip.hidden) place(ev.clientX, ev.clientY); });
  document.addEventListener("scroll", function () { tip.hidden = true; }, true);

  // --- toggles: every toggle of a group stays in step; targets carry data-corpus-group ----------------------
  function reorder(box, by) {
    var pitch = +box.getAttribute("data-pitch"), rows = all("g.corpus-row", box);
    rows.sort(function (a, b) {
      var x = a.getAttribute("data-k-" + by), y = b.getAttribute("data-k-" + by);
      if (by === "name") return x.localeCompare(y);
      if (x === "" || y === "") return (x === "") - (y === "");
      return +y - +x;
    });
    rows.forEach(function (r, j) { r.setAttribute("transform", "translate(0," + j * pitch + ")"); });
  }
  function setGroup(group, value) {
    all('.corpus-toggle[data-corpus-group="' + group + '"] button').forEach(function (b) {
      b.setAttribute("aria-pressed", String(b.getAttribute("data-value") === value));
    });
    all('[data-corpus-group="' + group + '"]:not(.corpus-toggle)').forEach(function (el) {
      el.setAttribute("data-corpus-state", value);
      all("circle.corpus-el", el).forEach(function (c) {
        c.setAttribute("class", "corpus-el corpus-q" + c.getAttribute("data-q-" + value));
        c.setAttribute("r", c.getAttribute("data-r-" + value));
      });
    });
    if (group === "order") all(".corpus-sortable").forEach(function (box) { reorder(box, value); });
    document.dispatchEvent(new CustomEvent("corpus:toggle", { detail: { group: group, value: value } }));
  }
  document.addEventListener("click", function (ev) {
    var b = ev.target.closest && ev.target.closest(".corpus-toggle button");
    if (b) setGroup(b.parentNode.getAttribute("data-corpus-group"), b.getAttribute("data-value"));
  });

  // --- tables: text + status filter, numeric sort with blanks last ---------------------------------------
  all(".corpus-table-block").forEach(function (block) {
    var tbody = block.querySelector("tbody"), rows = all("tr", tbody), q = block.querySelector(".corpus-q"),
        st = block.querySelector(".corpus-status"), cnt = block.querySelector(".corpus-count");
    function apply() {
      var text = q ? q.value.trim().toLowerCase() : "", s = st ? st.value : "", n = 0;
      rows.forEach(function (r) {
        var status = r.getAttribute("data-status");
        var ok = (!s || (s === "not-converted" ? status !== st.getAttribute("data-converted") : status === s))
          && (!text || r.textContent.toLowerCase().indexOf(text) >= 0);
        r.hidden = !ok; if (ok) n++;
      });
      if (cnt) cnt.textContent = n + " of " + rows.length + " datasets";
    }
    if (q) q.addEventListener("input", apply);
    if (st) st.addEventListener("change", apply);
    all("th", block).forEach(function (th, k) {
      function sort() {
        var asc = th.getAttribute("aria-sort") !== "ascending", num = th.getAttribute("data-num") === "1";
        all("th", block).forEach(function (h) { h.removeAttribute("aria-sort"); });
        th.setAttribute("aria-sort", asc ? "ascending" : "descending");
        rows.sort(function (a, b) {
          var x = a.cells[k].getAttribute("data-v"), y = b.cells[k].getAttribute("data-v");
          if (x === "" || y === "") return (x === "") - (y === "");  // blanks last either way
          if (num) return asc ? x - y : y - x;
          return asc ? x.localeCompare(y) : y.localeCompare(x);
        });
        rows.forEach(function (r) { tbody.appendChild(r); });
      }
      th.addEventListener("click", sort);
      th.addEventListener("keydown", function (ev) { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); sort(); } });
    });
    apply();
  });
})();
