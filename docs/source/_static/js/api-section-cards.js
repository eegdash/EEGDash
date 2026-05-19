/* Wrap Sphinx autoclass output (Parameters / Attributes / Methods) in
 * the same editorial card pattern used by the top "Signature" card
 * (.eegdash-ed-apicard). One card per section, each with a left gutter
 * label + body. Pure DOM post-processor — runs after
 * api-attribute-compact.js (defer order matters in conf.py).
 *
 * Contract:
 * - Gated by the presence of .eegdash-ed-apicard inside the same
 *   <section>; non-dataset autodoc pages keep their default look.
 * - Sections with no items are skipped silently (user choice).
 * - Methods render expanded inline (user choice).
 * - Parameter names use the same inline code chip look as the
 *   "Importable as" chips above (user choice).
 *
 * Idempotent via the data-api-grouped flag on the host <dd>.
 */
(function () {
  "use strict";

  function pluralize(n, singular, plural) {
    return `${n} ${n === 1 ? singular : plural}`;
  }

  function buildCardShell(kind, gutterLabel, gutterSub) {
    const card = document.createElement("div");
    card.className = `eegdash-ed-apicard is-section ${kind}`;

    const gutter = document.createElement("div");
    gutter.className = "apicard-gutter";

    const lbl = document.createElement("div");
    lbl.className = "lbl";
    lbl.textContent = gutterLabel;
    gutter.appendChild(lbl);

    if (gutterSub) {
      const sub = document.createElement("div");
      sub.className = "cls";
      sub.textContent = gutterSub;
      gutter.appendChild(sub);
    }

    const body = document.createElement("div");
    body.className = "apicard-body";

    card.appendChild(gutter);
    card.appendChild(body);
    return { card, body };
  }

  /* --- Parameters: turn dl.field-list bullets into .id-row grid ----- */

  function findParamsFieldList(dd) {
    const fls = dd.querySelectorAll(":scope > dl.field-list");
    for (const fl of fls) {
      const dt = fl.querySelector(":scope > dt");
      if (!dt) continue;
      const label = dt.textContent.trim().replace(/[:：]\s*$/, "").toLowerCase();
      if (label === "parameters" || label === "parameter") return fl;
    }
    return null;
  }

  function parseParamLi(li) {
    const p = li.querySelector(":scope > p") || li;
    const strong = p.querySelector(":scope > strong, strong");
    const em = p.querySelector(":scope > em, em");
    const name = strong ? strong.textContent.trim() : "";

    // Type lives in the first <em> at root; may be wrapped in parens.
    let type = em ? em.textContent.trim() : "";
    // Strip surrounding "(" and ")" pulled in from the autodoc rendering.
    type = type.replace(/^[\(\[]\s*|\s*[\)\]]$/g, "");

    // Description = everything after the first em-dash / en-dash / hyphen.
    // Use innerHTML so we keep inline links and code.
    const html = p.innerHTML;
    const sep = html.match(/([—–-])\s*/);
    let descHtml = html;
    if (sep) {
      descHtml = html.slice(sep.index + sep[0].length).trim();
    } else {
      // Fall back: drop the leading <strong>...</strong> + " (" + <em>...</em> + ") "
      descHtml = html
        .replace(/^\s*<strong[^>]*>[^<]*<\/strong>\s*/i, "")
        .replace(/^\(\s*<em[^>]*>[^<]*<\/em>\s*\)\s*/i, "");
    }

    return { name, type, descHtml };
  }

  function buildParamsCard(dd, params) {
    const items = [...params.querySelectorAll(":scope > dd > ul > li")];
    if (!items.length) {
      // Single-param shape: <dd><p>...</p></dd>
      const onlyP = params.querySelector(":scope > dd > p");
      if (onlyP) {
        items.push(onlyP.parentElement);
      } else {
        return;
      }
    }

    // Gutter sub-label: try to read "of __init__" from the closest
    // class signature. Fall back to "__init__" — autoclass always
    // emits the constructor's parameter list at the top.
    const gutterSub = "__init__";

    const { card, body } = buildCardShell("is-params", "Parameters", gutterSub);

    const ids = document.createElement("div");
    ids.className = "apicard-ids";
    body.appendChild(ids);

    for (const li of items) {
      const parsed = parseParamLi(li);
      if (!parsed.name && !parsed.descHtml) continue;

      const row = document.createElement("div");
      row.className = "id-row is-param";

      const k = document.createElement("span");
      k.className = "k";
      const code = document.createElement("code");
      code.textContent = parsed.name;
      k.appendChild(code);

      const v = document.createElement("span");
      v.className = "v";
      if (parsed.type) {
        const t = document.createElement("span");
        t.className = "param-type";
        t.textContent = parsed.type;
        v.appendChild(t);
      }
      const d = document.createElement("span");
      d.className = "param-desc";
      d.innerHTML = parsed.descHtml;
      v.appendChild(d);

      row.appendChild(k);
      row.appendChild(v);
      ids.appendChild(row);
    }

    params.replaceWith(card);
  }

  /* --- Attributes: wrap dl.py.attribute + property in is-attrs ------ */

  function buildAttrsCard(dd, attrs) {
    const { card, body } = buildCardShell(
      "is-attrs",
      "Attributes",
      pluralize(attrs.length, "FIELD", "FIELDS")
    );

    const wrap = document.createElement("div");
    wrap.className = "apicard-attrs";
    body.appendChild(wrap);

    // Insert the card at the location of the first attr, then move
    // every attr into the wrap (preserves source order).
    attrs[0].parentNode.insertBefore(card, attrs[0]);
    for (const a of attrs) wrap.appendChild(a);
  }

  /* --- Methods: wrap dl.py.method|classmethod|staticmethod --------- */

  function tagMethodKind(methodDl) {
    const dt = methodDl.querySelector(":scope > dt.sig, :scope > dt");
    if (!dt) return;
    const propEm = dt.querySelector(":scope > em.property");
    let kindText = "def";
    if (methodDl.classList.contains("classmethod")) kindText = "classmethod";
    else if (methodDl.classList.contains("staticmethod")) kindText = "staticmethod";
    else if (propEm) {
      const raw = propEm.textContent.trim().toLowerCase();
      if (raw.includes("classmethod")) kindText = "classmethod";
      else if (raw.includes("staticmethod")) kindText = "staticmethod";
      else if (raw.includes("async")) kindText = "async def";
    }
    if (propEm) propEm.remove();

    const kind = document.createElement("span");
    kind.className = "sig-kind";
    kind.textContent = kindText;
    dt.insertBefore(kind, dt.firstChild);
  }

  function buildMethodsCard(dd, methods) {
    const { card, body } = buildCardShell(
      "is-methods",
      "Methods",
      pluralize(methods.length, "ENTRY", "ENTRIES")
    );

    const wrap = document.createElement("div");
    wrap.className = "apicard-methods";
    body.appendChild(wrap);

    methods[0].parentNode.insertBefore(card, methods[0]);
    for (const m of methods) {
      tagMethodKind(m);
      wrap.appendChild(m);
    }
  }

  /* --- Entry point -------------------------------------------------- */

  function groupOneClass(dd) {
    if (dd.dataset.apiGrouped === "1") return;

    const section = dd.closest("section");
    if (!section || !section.querySelector(".eegdash-ed-apicard")) return;

    dd.dataset.apiGrouped = "1";

    // (a) Parameters card
    const params = findParamsFieldList(dd);
    if (params) buildParamsCard(dd, params);

    // (b) Attributes card — collect AFTER parameters mutation so we
    // don't pick up nested field-lists inside attribute dd's.
    const attrs = [
      ...dd.querySelectorAll(":scope > dl.py.attribute, :scope > dl.py.property"),
    ];
    if (attrs.length) buildAttrsCard(dd, attrs);

    // (c) Methods card
    const methods = [
      ...dd.querySelectorAll(
        ":scope > dl.py.method, :scope > dl.py.classmethod, :scope > dl.py.staticmethod"
      ),
    ];
    if (methods.length) buildMethodsCard(dd, methods);
  }

  function run() {
    document
      .querySelectorAll("article.bd-article dl.py.class > dd")
      .forEach(groupOneClass);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", run);
  } else {
    run();
  }
})();
