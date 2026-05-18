/* Dataset File Explorer — gated to per-dataset doc pages via layout.html.
 * Initializes for each `.eegdash-explorer[data-dataset-id="…"]` container
 * found on the page, fetches that dataset's records + dataset document
 * from the EEGDash API, and renders a collapsible BIDS file tree that
 * includes real sidecar files (from each record's `storage.dep_keys`). */
(function () {
  "use strict";

  const API_BASE = "https://data.eegdash.org";
  const DATABASE = "eegdash";
  const RECORD_LIMIT = 1000;

  const KIND = { DIR: "dir", FILE: "file" };
  const FILE_KIND = { DATA: "data", SIDECAR: "sidecar", ROOTFILE: "rootfile" };

  const TITLE_BY_FILE_KIND = {
    [FILE_KIND.DATA]: "Recording",
    [FILE_KIND.SIDECAR]: "BIDS sidecar",
    [FILE_KIND.ROOTFILE]: "Dataset metadata",
  };

  // Sort order within a directory: directories, then data files, then
  // dataset-root metadata, then per-recording sidecars. Anything unknown
  // sorts last.
  const FILE_KIND_RANK = {
    [FILE_KIND.DATA]: 1,
    [FILE_KIND.ROOTFILE]: 2,
    [FILE_KIND.SIDECAR]: 3,
  };

  function init() {
    document
      .querySelectorAll(".eegdash-explorer:not([data-eegdash-init])")
      .forEach(initExplorer);
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init, { once: true });
  } else {
    init();
  }

  function initExplorer(root) {
    root.setAttribute("data-eegdash-init", "1");
    const datasetId = root.dataset.datasetId;
    if (!datasetId) return;

    const els = {
      tree: root.querySelector(".ee-tree"),
      search: root.querySelector(".ee-search"),
      info: root.querySelector(".ee-info"),
      recCount: root.querySelector(".ee-rec-count"),
      fileCount: root.querySelector(".ee-file-count"),
      subjectCount: root.querySelector(".ee-subject-count"),
      modalities: root.querySelector(".ee-modalities"),
    };

    let loaded = false;
    let records = [];
    let nodeTree = null;
    let pendingIdle = null;
    let pendingTimeout = null;
    let searchActive = false;

    const trigger = () => { if (!loaded) load(); };
    els.tree.addEventListener("click", trigger, { once: true });
    if ("requestIdleCallback" in window) {
      pendingIdle = requestIdleCallback(trigger, { timeout: 4000 });
    } else {
      pendingTimeout = setTimeout(trigger, 1500);
    }

    function cancelPendingTriggers() {
      if (pendingIdle != null && "cancelIdleCallback" in window) {
        cancelIdleCallback(pendingIdle);
      }
      if (pendingTimeout != null) clearTimeout(pendingTimeout);
      pendingIdle = null;
      pendingTimeout = null;
    }

    async function load() {
      loaded = true;
      cancelPendingTriggers();
      els.tree.innerHTML = '<div class="ee-status">Loading file structure…</div>';
      try {
        const filter = encodeURIComponent(JSON.stringify({ dataset: datasetId }));
        const recordsUrl = `${API_BASE}/api/${DATABASE}/records?filter=${filter}&limit=${RECORD_LIMIT}`;
        const datasetUrl = `${API_BASE}/api/${DATABASE}/datasets/${encodeURIComponent(datasetId)}`;
        const [recResp, dsResp] = await Promise.all([
          fetch(recordsUrl),
          fetch(datasetUrl).catch(() => null),
        ]);
        if (!recResp.ok) throw new Error(`HTTP ${recResp.status}`);
        const body = await recResp.json();
        records = (body.data || []).filter(r =>
          r && (r.bidspath || r.bids_relpath || (r.storage && r.storage.raw_key))
        );
        if (!records.length) {
          els.tree.innerHTML = '<div class="ee-status">No file records indexed for this dataset yet.</div>';
          return;
        }
        let datasetDoc = null;
        if (dsResp && dsResp.ok) {
          try { datasetDoc = (await dsResp.json()).data || null; } catch (_) {}
        }
        nodeTree = buildTree(records, datasetDoc, datasetId);
        renderStats();
        renderTree();
        wireSearch();
        els.search.disabled = false;
      } catch (err) {
        els.tree.innerHTML =
          '<div class="ee-status ee-error">Could not load file list: ' +
          escapeText(String((err && err.message) || err)) +
          '</div>';
      }
    }

    function renderStats() {
      const subjects = new Set();
      const modalities = new Set();
      let totalFiles = 0;
      records.forEach(r => {
        if (r.subject) subjects.add(r.subject);
        if (r.datatype) modalities.add(String(r.datatype).toLowerCase());
        const dep = r.storage && r.storage.dep_keys;
        totalFiles += 1 + (dep ? dep.length : 0);
      });
      for (const c of nodeTree.children.values()) {
        if (c.kind === KIND.FILE && c.fileKind === FILE_KIND.ROOTFILE) totalFiles += 1;
      }
      els.recCount.textContent = String(records.length);
      els.fileCount.textContent = String(totalFiles);
      els.subjectCount.textContent = String(subjects.size);
      if (modalities.size) {
        els.modalities.innerHTML = "";
        [...modalities].sort().forEach(m => {
          els.modalities.appendChild(createBadge("ee-badge ee-" + m, m.toUpperCase()));
        });
      } else {
        els.modalities.textContent = "—";
      }
    }

    function renderTree() {
      els.tree.innerHTML = "";
      const rootRow = createRow(nodeTree, 0);
      els.tree.appendChild(rootRow);
      if (nodeTree.open) renderChildren(nodeTree, 1, rootRow);
    }

    function renderChildren(node, depth, after) {
      let cursor = after;
      for (const child of node.children.values()) {
        const row = createRow(child, depth);
        cursor.after(row);
        cursor = row;
        if (child.kind === KIND.DIR && child.open) {
          cursor = renderChildren(child, depth + 1, cursor);
        }
      }
      return cursor;
    }

    function createRow(node, depth) {
      const row = document.createElement("div");
      const classes = ["ee-row"];
      if (node.kind === KIND.DIR) classes.push("ee-folder");
      else if (node.fileKind) classes.push("ee-" + node.fileKind);
      if (node.kind === KIND.DIR && node.open) classes.push("ee-open");
      row.className = classes.join(" ");
      row.dataset.depth = String(depth);
      row.style.setProperty("--ee-depth", String(depth));

      row.appendChild(createSpan("ee-chev", "›"));
      row.appendChild(createSpan("ee-mark", glyphFor(node)));
      row.appendChild(createSpan("ee-name", node.name));
      row.appendChild(buildRowMeta(node));

      const handler = node.kind === KIND.DIR
        ? () => toggleDir(node, row, depth)
        : () => selectFile(node, row);
      row.addEventListener("click", handler);
      return row;
    }

    function buildRowMeta(node) {
      const meta = document.createElement("span");
      meta.className = "ee-meta";
      if (node.kind === KIND.DIR) {
        const n = node.children.size;
        meta.textContent = `${n} item${n === 1 ? "" : "s"}`;
        return meta;
      }
      const modality = node.fileKind === FILE_KIND.DATA
        && node.records && node.records[0] && node.records[0].datatype;
      if (modality) {
        meta.appendChild(createBadge(
          "ee-badge ee-" + String(modality).toLowerCase(),
          String(modality).toUpperCase()
        ));
      } else if (node.fileKind === FILE_KIND.SIDECAR) {
        meta.appendChild(createBadge("ee-badge ee-tag", "sidecar"));
      } else if (node.fileKind === FILE_KIND.ROOTFILE) {
        meta.appendChild(createBadge("ee-badge ee-tag", "metadata"));
      }
      return meta;
    }

    function toggleDir(node, row, depth) {
      node.open = !node.open;
      if (node.open) {
        row.classList.add("ee-open");
        renderChildren(node, depth + 1, row);
      } else {
        row.classList.remove("ee-open");
        let n = row.nextSibling;
        while (n && Number(n.dataset.depth || 0) > depth) {
          const next = n.nextSibling;
          n.remove();
          n = next;
        }
      }
    }

    function selectFile(node, row) {
      const prev = els.tree.querySelector(".ee-row.ee-selected");
      if (prev) prev.classList.remove("ee-selected");
      row.classList.add("ee-selected");
      els.info.hidden = false;
      els.info.innerHTML = "";

      const title = document.createElement("div");
      title.className = "ee-info-title";
      title.textContent = TITLE_BY_FILE_KIND[node.fileKind] || "File";
      els.info.appendChild(title);

      const dl = document.createElement("dl");
      const add = (k, v) => {
        if (v === undefined || v === null || v === "") return;
        const dt = document.createElement("dt"); dt.textContent = k;
        const dd = document.createElement("dd"); dd.textContent = String(v);
        dl.appendChild(dt); dl.appendChild(dd);
      };
      add("File", node.name);
      const r = node.records && node.records[0];
      if (r) {
        const s = r.storage || {};
        const fullPath = s.raw_key || r.bids_relpath || r.bidspath || "";
        if (node.fileKind === FILE_KIND.DATA) {
          add("Path", fullPath);
          add("Subject", r.subject);
          add("Session", r.session);
          add("Task", r.task);
          add("Run", r.run);
          add("Acquisition", r.acquisition || (r.entities_mne && r.entities_mne.acquisition));
          add("Modality", r.datatype && String(r.datatype).toUpperCase());
          add("Channels", r.nchans);
          const sfreq = r.sampling_frequency || r.sfreq;
          if (sfreq) add("Sample rate", sfreq + " Hz");
          if (r.duration) add("Duration", formatDuration(r.duration));
        } else if (node.fileKind === FILE_KIND.SIDECAR) {
          add("Belongs to", fullPath);
          if (node.records.length > 1) {
            add("Shared by", `${node.records.length} recordings`);
          }
        }
      }
      els.info.appendChild(dl);
    }

    function wireSearch() {
      let timer;
      els.search.addEventListener("input", () => {
        clearTimeout(timer);
        timer = setTimeout(applySearch, 80);
      });
    }

    function applySearch() {
      const q = els.search.value.toLowerCase().trim();
      const becameActive = !!q && !searchActive;
      searchActive = !!q;
      if (becameActive) expandAllAndRerender();
      els.tree.querySelectorAll(".ee-row").forEach(row => {
        if (!q) { row.style.display = ""; return; }
        const name = row.querySelector(".ee-name");
        const match = name && name.textContent.toLowerCase().includes(q);
        row.style.display = match ? "" : "none";
      });
    }

    function expandAllAndRerender() {
      let changed = false;
      const walk = n => {
        if (n.kind === KIND.DIR && !n.open) { n.open = true; changed = true; }
        for (const c of n.children.values()) walk(c);
      };
      walk(nodeTree);
      if (changed) renderTree();
    }
  }

  // ---------- Pure helpers (no per-instance state) ----------

  function buildTree(records, datasetDoc, datasetId) {
    const root = makeNode(datasetId, KIND.DIR);
    root.open = true;

    if (datasetDoc && datasetDoc.storage) {
      const s = datasetDoc.storage;
      if (s.raw_key) addPath(root, s.raw_key, FILE_KIND.ROOTFILE, null);
      for (const k of (s.dep_keys || [])) addPath(root, k, FILE_KIND.ROOTFILE, null);
    }

    for (const r of records) {
      const storage = r.storage || {};
      const rel = storage.raw_key
        || r.bids_relpath
        || stripPrefix(r.bidspath || r.filepath || r.path || "", datasetId);
      if (rel) addPath(root, rel, FILE_KIND.DATA, r);
      for (const k of (storage.dep_keys || [])) {
        addPath(root, k, FILE_KIND.SIDECAR, r);
      }
    }

    sortNode(root);
    const firstSubject = [...root.children.values()]
      .find(c => c.kind === KIND.DIR && /^sub-/.test(c.name));
    if (firstSubject) {
      firstSubject.open = true;
      const firstChild = [...firstSubject.children.values()]
        .find(c => c.kind === KIND.DIR);
      if (firstChild) firstChild.open = true;
    }
    return root;
  }

  function makeNode(name, kind) {
    return {
      name,
      kind,
      fileKind: null,
      children: new Map(),
      // Always an array for files: a data file holds its own record,
      // a sidecar holds every recording that references it (BIDS sidecars
      // can be shared up the directory tree by inheritance).
      records: null,
      open: false,
    };
  }

  function ensure(parent, name, kind) {
    let n = parent.children.get(name);
    if (!n) { n = makeNode(name, kind); parent.children.set(name, n); }
    return n;
  }

  function addPath(root, relPath, fileKind, record) {
    const parts = String(relPath || "").split("/").filter(Boolean);
    if (!parts.length) return null;
    let cursor = root;
    for (let i = 0; i < parts.length - 1; i++) cursor = ensure(cursor, parts[i], KIND.DIR);
    const name = parts[parts.length - 1];
    const node = ensure(cursor, name, KIND.FILE);
    // Data records always win the file-kind assignment if a path collides
    // with a sidecar reference elsewhere.
    if (fileKind === FILE_KIND.DATA || !node.fileKind) node.fileKind = fileKind;
    if (record) {
      if (fileKind === FILE_KIND.DATA) {
        node.records = [record];
      } else if (!node.records) {
        node.records = [record];
      } else {
        node.records.push(record);
      }
    }
    return node;
  }

  function stripPrefix(p, prefix) {
    if (!p || !prefix) return p;
    const re = new RegExp("^" + prefix.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + "/", "i");
    return p.replace(re, "");
  }

  function sortNode(node) {
    if (!node.children.size) return;
    const sorted = [...node.children.entries()].sort(([a, na], [b, nb]) => {
      const r = rankNode(na) - rankNode(nb);
      return r || a.localeCompare(b, undefined, { numeric: true });
    });
    node.children = new Map(sorted);
    for (const c of node.children.values()) sortNode(c);
  }

  function rankNode(n) {
    if (n.kind === KIND.DIR) return 0;
    return FILE_KIND_RANK[n.fileKind] || 4;
  }

  function glyphFor(node) {
    if (node.kind === KIND.DIR) return "";
    return node.fileKind === FILE_KIND.DATA ? "■" : "·";
  }

  function createSpan(className, text) {
    const el = document.createElement("span");
    el.className = className;
    el.textContent = text;
    return el;
  }

  function createBadge(className, text) {
    return createSpan(className, text);
  }

  function formatDuration(seconds) {
    seconds = Number(seconds) || 0;
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    return h > 0 ? `${h}h ${m}m` : `${m}m`;
  }

  function escapeText(s) {
    const d = document.createElement("div");
    d.textContent = s;
    return d.innerHTML;
  }
})();
