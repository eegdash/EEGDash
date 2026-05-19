/* Compact display for autodoc attribute blocks.
 *
 * Sphinx emits each class attribute as:
 *   <dl class="py attribute">
 *     <dt class="sig sig-object py">data_dir<a class="headerlink">#</a></dt>
 *     <dd>
 *       <p>Description...</p>
 *       <dl class="field-list">
 *         <dt class="field-odd">Type:</dt>
 *         <dd class="field-odd"><p>Path</p></dd>
 *       </dl>
 *     </dd>
 *   </dl>
 *
 * The "Type:" field-list adds two extra lines of vertical space per
 * attribute and looks redundant when the type is short. Pull the type
 * value up into the signature row ("data_dir : Path") and hide the
 * field-list. Description paragraph stays.
 */
(function () {
  "use strict";

  function compactOne(dl) {
    if (dl.dataset.compact === "1") return;
    dl.dataset.compact = "1";

    var dt = dl.querySelector(":scope > dt.sig, :scope > dt");
    var fieldList = dl.querySelector(":scope > dd > dl.field-list");
    if (!dt || !fieldList) return;

    var fieldDt = fieldList.querySelector("dt");
    if (!fieldDt) return;
    var label = (fieldDt.textContent || "").trim().replace(/:\s*$/, "");
    if (label.toLowerCase() !== "type") return;

    var typeValue = fieldList.querySelector("dd");
    if (!typeValue) return;

    // Clone the type-value children (preserving <a> / <code> markup) and
    // insert them inline next to the descname.
    var typeWrap = document.createElement("span");
    typeWrap.className = "eegdash-attr-type";
    typeWrap.appendChild(document.createTextNode(" : "));
    // The dd usually wraps its content in a <p>. Lift those children out
    // so we don't introduce a block-level <p> into an inline dt.
    var src = typeValue.querySelector("p") || typeValue;
    Array.prototype.forEach.call(src.childNodes, function (child) {
      typeWrap.appendChild(child.cloneNode(true));
    });

    // Insert before the headerlink anchor so the "#" stays at the end.
    var headerlink = dt.querySelector(".headerlink");
    if (headerlink) {
      dt.insertBefore(typeWrap, headerlink);
    } else {
      dt.appendChild(typeWrap);
    }

    fieldList.style.display = "none";
  }

  function run() {
    document.querySelectorAll("dl.py.attribute").forEach(compactOne);
    document.querySelectorAll("dl.py.property").forEach(compactOne);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", run);
  } else {
    run();
  }
})();
