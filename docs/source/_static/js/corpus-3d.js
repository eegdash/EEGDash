/* EEGDash Corpus: the electrode positions in 3-D on a canvas (orthographic, yaw/pitch, painter's order).
   Data: the inline JSON next to the canvas ({p: xyz in head radii, q: ramp steps, r: radii, t: tooltip, l: landmark label
   or absent}, one per position, [hours, datasets] for q and r). Colours are the ramp variables of corpus.css, re-read on a theme
   change. Drag (mouse, touch, pen) or arrow keys rotate; Home resets; no auto-rotation. */
(function () {
  "use strict";
  var REST_YAW = 0, REST_PITCH = 0.35, VIEW = 560;

  function setup(box) {
    var canvas = box.querySelector("canvas"), src = box.querySelector("script.corpus-3d-data");
    if (!canvas || !src) return;
    var pts = JSON.parse(src.textContent), ctx = canvas.getContext("2d");
    var yaw = REST_YAW, pitch = REST_PITCH, mode = box.getAttribute("data-corpus-state") === "datasets" ? 1 : 0;
    var ramp = [], fg = "#888", halo = "#fff", size = 0, proj = [], drag = null;

    function readColours() {
      var cs = getComputedStyle(box);
      ramp = [];
      for (var k = 0; k < 8; k++) ramp.push(cs.getPropertyValue("--corpus-q" + k).trim() || "#2f9e44");
      fg = getComputedStyle(canvas).color || fg;
      halo = cs.getPropertyValue("--corpus-card").trim() || halo;
    }
    function rotate(p) {
      var cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
      var x1 = p[0] * cy - p[1] * sy, y1 = p[0] * sy + p[1] * cy, z1 = p[2];
      return [x1, y1 * cp - z1 * sp, y1 * sp + z1 * cp];  // screen x, screen y (up), depth (towards the viewer)
    }
    function draw() {
      if (!size) return;
      var dpr = window.devicePixelRatio || 1, half = size / 2, R = size * 0.4, k = size / VIEW;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, size, size);
      ctx.strokeStyle = fg; ctx.globalAlpha = 0.35; ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.arc(half, half, R, 0, 2 * Math.PI); ctx.stroke();
      ctx.globalAlpha = 0.18; ctx.setLineDash([3, 4]); ctx.beginPath();  // the equator (z = 0)
      for (var a = 0; a <= 64; a++) {
        var e = rotate([Math.cos(a / 32 * Math.PI), Math.sin(a / 32 * Math.PI), 0]);
        a ? ctx.lineTo(half + e[0] * R, half - e[1] * R) : ctx.moveTo(half + e[0] * R, half - e[1] * R);
      }
      ctx.stroke(); ctx.setLineDash([]);
      function outline(points, close) {  // a polyline of head-frame points, rotated into the view
        ctx.beginPath();
        points.forEach(function (p, i) {
          var s = rotate(p);
          i ? ctx.lineTo(half + s[0] * R, half - s[1] * R) : ctx.moveTo(half + s[0] * R, half - s[1] * R);
        });
        if (close) ctx.closePath();
        ctx.stroke();
      }
      ctx.globalAlpha = 0.45; ctx.lineWidth = 1.5;
      outline([[-0.08, 0.99, 0], [0, 1.1, 0], [0.08, 0.99, 0]], true);  // the nose, a triangle on the equator
      [-1, 1].forEach(function (side) {  // the ears, small ellipses in the head plane, as in the top view
        var ear = [];
        for (var a = 0; a <= 24; a++) {
          var t = a / 24 * 2 * Math.PI;
          ear.push([side * (1.035 + 0.04 * Math.cos(t)), 0.13 * Math.sin(t), 0]);
        }
        outline(ear, true);
      });
      proj = pts.map(function (p, i) {
        var s = rotate(p.p);
        return { i: i, x: half + s[0] * R, y: half - s[1] * R, z: s[2], r: p.r[mode] * k * 1.25 };
      });
      proj.sort(function (a, b) { return a.z - b.z; });  // painter's order: far first
      ctx.lineWidth = 1;
      proj.forEach(function (q) {
        ctx.globalAlpha = q.z < 0 ? 0.35 : 1;
        ctx.fillStyle = ramp[pts[q.i].q[mode]];
        ctx.beginPath(); ctx.arc(q.x, q.y, q.r, 0, 2 * Math.PI); ctx.fill();
      });
      ctx.font = "600 " + Math.round(11 * k) + "px sans-serif"; ctx.textAlign = "center";  // the landmark labels
      ctx.lineJoin = "round"; ctx.lineWidth = 3; ctx.strokeStyle = halo; ctx.fillStyle = fg;
      proj.forEach(function (q) {
        var lab = pts[q.i].l;
        if (!lab) return;
        ctx.globalAlpha = q.z < 0 ? 0.35 : 1;
        ctx.strokeText(lab, q.x, q.y - q.r - 3); ctx.fillText(lab, q.x, q.y - q.r - 3);
      });
      ctx.globalAlpha = 1;
    }
    function resize() {
      var w = canvas.clientWidth;
      if (!w) return;  // hidden tab: draw when it is shown
      var dpr = window.devicePixelRatio || 1;
      size = w; canvas.width = Math.round(w * dpr); canvas.height = Math.round(w * dpr);
      readColours(); draw();
    }
    function hit(ev) {
      var b = canvas.getBoundingClientRect(), x = ev.clientX - b.left, y = ev.clientY - b.top;
      for (var j = proj.length - 1; j >= 0; j--) {  // front-most first
        var q = proj[j], dx = q.x - x, dy = q.y - y;
        if (dx * dx + dy * dy <= (q.r + 3) * (q.r + 3)) return pts[q.i];
      }
      return null;
    }
    canvas.addEventListener("pointerdown", function (ev) {
      drag = { x: ev.clientX, y: ev.clientY }; canvas.setPointerCapture(ev.pointerId);
    });
    canvas.addEventListener("pointermove", function (ev) {
      if (drag) {
        yaw += (ev.clientX - drag.x) * 0.01;
        pitch = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, pitch + (ev.clientY - drag.y) * 0.01));
        drag = { x: ev.clientX, y: ev.clientY }; draw();
        if (window.corpusTip) window.corpusTip(null);
        return;
      }
      var p = hit(ev);
      if (window.corpusTip) window.corpusTip(p ? p.t : null, ev.clientX, ev.clientY);
    });
    function release() { drag = null; }
    canvas.addEventListener("pointerup", release);
    canvas.addEventListener("pointercancel", release);
    canvas.addEventListener("pointerleave", function () { if (!drag && window.corpusTip) window.corpusTip(null); });
    canvas.addEventListener("keydown", function (ev) {
      var step = 0.1, used = true;
      if (ev.key === "ArrowLeft") yaw -= step;
      else if (ev.key === "ArrowRight") yaw += step;
      else if (ev.key === "ArrowUp") pitch = Math.max(-Math.PI / 2, pitch - step);
      else if (ev.key === "ArrowDown") pitch = Math.min(Math.PI / 2, pitch + step);
      else if (ev.key === "Home") { yaw = REST_YAW; pitch = REST_PITCH; }
      else used = false;
      if (used) { ev.preventDefault(); draw(); }
    });
    document.addEventListener("corpus:toggle", function (ev) {
      if (ev.detail.group === "encode") { mode = ev.detail.value === "datasets" ? 1 : 0; draw(); }
    });
    new MutationObserver(function () { readColours(); draw(); })
      .observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    if (window.ResizeObserver) new ResizeObserver(resize).observe(canvas);
    else window.addEventListener("resize", resize);
    resize();
  }

  function init() { Array.prototype.forEach.call(document.querySelectorAll(".corpus-3d"), setup); }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
