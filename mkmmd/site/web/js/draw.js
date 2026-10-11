// Drawing on a picture: the picture fills the pane; pen, arrow, ellipse, box, text and eraser in eight colours, undo
// and redo, zoom (wheel, pinch, + -, 0 fits) and pan (space-drag, right-drag, two fingers). Marks are kept in the
// picture's own pixels ({kind, color, from/to | box | points, text}), as mkmmd/review.py checks and draws them.
import { h, layer, toast, topLayer } from "./ui.js";

export const COLORS = ["#e5383b", "#f77f00", "#fcbf49", "#2a9d8f", "#4361ee", "#9d4edd", "#ffffff", "#111111"];
export const baseWidth = (w, h) => Math.max(2, Math.round(Math.max(w, h) / 380));     // as mkmmd.review.render_marks
export const fontSize = (h) => Math.max(14, Math.round(h / 38));

const ICON = {
  ink: '<path d="M4 20c4-1 5-4 7-8s4-6 6-7 3 1 2 3-5 6-7 9-2 4 0 4 4-2 6-4"/>',
  arrow: '<path d="M5 19 19 5M10 5h9v9"/>',
  ellipse: '<ellipse cx="12" cy="12" rx="8" ry="6"/>',
  box: '<rect x="4" y="6" width="16" height="12" rx="1.5"/>',
  text: '<path d="M6 6h12M12 6v13M9 19h6"/>',
  erase: '<path d="m7 21-4-4 11-11 7 7-8 8H7z"/><path d="M11 21h9"/>',
  undo: '<path d="M9 14 4 9l5-5"/><path d="M4 9h10a6 6 0 0 1 0 12h-3"/>',
  redo: '<path d="m15 14 5-5-5-5"/><path d="M20 9H10a6 6 0 0 0 0 12h3"/>',
  fit: '<path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/>',
};
export const icon = (name) => `<svg viewBox="0 0 24 24">${ICON[name]}</svg>`;

// The marks, drawn in picture coordinates on a context already transformed to them; `lw` is the line width there and
// `fs` the text size.
export function paintMarks(ctx, marks, lw, fs, glow = null) {
  ctx.save();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  for (const m of marks) {
    if (m === glow) {
      ctx.save();
      ctx.shadowColor = "rgba(255,255,255,0.95)";
      ctx.shadowBlur = lw * 4;
    }
    ctx.strokeStyle = ctx.fillStyle = m.color || COLORS[0];
    ctx.lineWidth = lw;
    if (m.kind === "line" || m.kind === "arrow") {
      ctx.beginPath(); ctx.moveTo(...m.from); ctx.lineTo(...m.to); ctx.stroke();
      if (m.kind === "arrow") arrowHead(ctx, m.from, m.to, lw);
    } else if (m.kind === "ink") {
      ctx.beginPath();
      m.points.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])));
      ctx.stroke();
    } else if (m.kind === "box") {
      ctx.strokeRect(...m.box);
    } else if (m.kind === "ellipse") {
      const [x, y, w, hh] = m.box;
      ctx.beginPath(); ctx.ellipse(x + w / 2, y + hh / 2, Math.max(w / 2, 0.5), Math.max(hh / 2, 0.5), 0, 0, Math.PI * 2); ctx.stroke();
    }
    if (m.text) {
      const at = m.box ? m.box.slice(0, 2) : (m.to || m.points[m.points.length - 1]);
      ctx.font = `600 ${fs}px system-ui, sans-serif`;
      ctx.textBaseline = "top";
      ctx.lineWidth = Math.max(4, lw);
      ctx.strokeStyle = "white";
      ctx.strokeText(m.text, at[0], at[1]);
      ctx.fillText(m.text, at[0], at[1]);
    }
    if (m === glow) ctx.restore();
  }
  ctx.restore();
}

function arrowHead(ctx, a, b, lw) {
  const dx = b[0] - a[0], dy = b[1] - a[1], len = Math.hypot(dx, dy);
  if (!len) return;
  const ux = -dx / len, uy = -dy / len, size = 4 * lw + 8;
  ctx.beginPath();
  for (const s of [1, -1]) {
    const c = Math.cos(s * 0.4887), sn = Math.sin(s * 0.4887);     // 28 degrees
    ctx.moveTo(b[0], b[1]);
    ctx.lineTo(b[0] + (c * ux - sn * uy) * size, b[1] + (sn * ux + c * uy) * size);
  }
  ctx.stroke();
}

// Distance in picture pixels from point p to a mark (for the eraser).
export function distanceTo(m, p) {
  const seg = (a, b) => {
    const dx = b[0] - a[0], dy = b[1] - a[1], l2 = dx * dx + dy * dy;
    const t = l2 ? Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l2)) : 0;
    return Math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy);
  };
  if (m.kind === "line" || m.kind === "arrow") return seg(m.from, m.to);
  if (m.kind === "ink") return Math.min(...m.points.slice(1).map((q, i) => seg(m.points[i], q)));
  const [x, y, w, hh] = m.box;
  if (m.kind === "text") return p[0] >= x && p[0] <= x + w && p[1] >= y && p[1] <= y + hh ? 0 : Infinity;
  if (m.kind === "box") {
    const c = [[x, y], [x + w, y], [x + w, y + hh], [x, y + hh]];
    return Math.min(...c.map((q, i) => seg(q, c[(i + 1) % 4])));
  }
  const rx = Math.max(w / 2, 1), ry = Math.max(hh / 2, 1);                  // ellipse: a radial estimate
  return Math.abs(Math.hypot((p[0] - x - rx) / rx, (p[1] - y - ry) / ry) - 1) * Math.min(rx, ry);
}

export function openDrawing({ image, marks: start, onDone }) {
  const iw = image.w, ih = image.h, lw = baseWidth(iw, ih), fs = fontSize(ih);
  const S = { marks: start.map((m) => ({ ...m })), undo: [], redo: [], tool: "ink", color: COLORS[0], view: null,
    fitted: true, draft: null, glow: null, pointers: new Map(), pinch: null, pan: null, space: false, busy: false,
    dirty: false };
  const onKeyUp = (e) => { if (e.key === " ") { S.space = false; refresh(); } };
  const { el, close } = layer("draw", () => {
    document.removeEventListener("keydown", onKey, true);
    document.removeEventListener("keyup", onKeyUp);
  });
  const canvas = h("canvas");
  const stage = h("div", { class: "stage" }, canvas, h("div", { class: "hint", text: "scroll to zoom · space-drag to pan · Esc is Done" }));
  const count = h("span", { class: "badge mark" });
  const done = h("button", { class: "btn primary", text: "Done", onclick: () => finish() });
  el.append(h("div", { class: "bar top" }, h("span", { class: "title", text: image.name, title: image.path }),
    image.sheet ? h("span", { class: "badge accent", text: "lab sheet · mm" }) : null, count));
  el.append(stage);
  const tools = {}, bar = h("div", { class: "toolbar" });
  const tool = (name, title, onclick) => {
    const b = h("button", { class: "tool", title, onclick });
    b.innerHTML = icon(name);
    return b;
  };
  for (const [name, title] of [["ink", "Pen (p)"], ["arrow", "Arrow (a)"], ["ellipse", "Ellipse (o)"], ["box", "Box (b)"], ["text", "Text (t)"], ["erase", "Eraser (e)"]]) {
    tools[name] = tool(name, title, () => pick(name));
    bar.append(tools[name]);
  }
  const dot = h("span", { class: "dot" });
  const colorBtn = h("button", { class: "tool", title: "Colour", onclick: () => swatches() }, dot);
  bar.append(h("span", { class: "sep" }), colorBtn, h("span", { class: "sep" }),
    tool("undo", "Undo (Ctrl+Z)", () => undo()), tool("redo", "Redo (Ctrl+Shift+Z)", () => redo()),
    tool("fit", "Fit (0)", () => { fit(); paint(); }), h("span", { class: "sep" }), done);
  el.append(h("div", { class: "tools" }, bar));

  const picture = new Image();
  picture.decoding = "async";
  picture.src = image.url;
  picture.onload = () => { if (S.fitted) fit(); paint(); };
  picture.onerror = () => toast("Cannot read " + image.name, "bad");

  const ctx = canvas.getContext("2d");
  function refresh() {
    Object.entries(tools).forEach(([n, b]) => b.classList.toggle("on", n === S.tool));
    dot.style.background = S.color;
    count.textContent = S.marks.length === 1 ? "1 mark" : `${S.marks.length} marks`;
    count.hidden = !S.marks.length;
    canvas.style.cursor = S.space ? "grab" : S.tool === "erase" ? "pointer" : S.tool === "text" ? "text" : "crosshair";
  }
  function pick(name) { S.tool = name; refresh(); }
  function swatches() {
    const old = el.querySelector(".swatches");
    if (old) { old.remove(); return; }
    const box = h("div", { class: "swatches" }, COLORS.map((c) => h("button", { class: c === S.color ? "on" : "", style: { background: c }, title: c,
      onclick: () => { S.color = c; box.remove(); refresh(); } })));
    const r = colorBtn.getBoundingClientRect(), er = el.getBoundingClientRect();
    box.style.left = Math.max(6, r.left - er.left - 90) + "px";
    box.style.bottom = (er.bottom - r.top + 8) + "px";
    el.append(box);
  }

  // ---------------------------------------------------------------------------------------------- view
  function size() {
    const dpr = devicePixelRatio || 1, w = stage.clientWidth, hh = stage.clientHeight;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(hh * dpr)) {
      canvas.width = Math.round(w * dpr); canvas.height = Math.round(hh * dpr);
      canvas.style.width = w + "px"; canvas.style.height = hh + "px";
    }
    return [w, hh, dpr];
  }
  function fit() {
    const [w, hh] = size();
    const s = Math.min((w - 16) / iw, (hh - 16) / ih);
    S.view = { s, ox: (w - iw * s) / 2, oy: (hh - ih * s) / 2 };
    S.fitted = true;
  }
  function zoomAt(x, y, k) {
    const v = S.view, s = Math.min(40, Math.max(0.05, v.s * k));
    v.ox = x - (x - v.ox) * (s / v.s); v.oy = y - (y - v.oy) * (s / v.s); v.s = s;
    S.fitted = false;
  }
  const toPic = (x, y) => [(x - S.view.ox) / S.view.s, (y - S.view.oy) / S.view.s];
  const local = (e) => { const r = canvas.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
  let raf = 0;
  function paint() {
    if (raf) return;
    raf = requestAnimationFrame(() => {
      raf = 0;
      if (!S.view) return;
      const [w, hh, dpr] = size();
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, hh);
      const v = S.view;
      ctx.setTransform(dpr * v.s, 0, 0, dpr * v.s, dpr * v.ox, dpr * v.oy);
      ctx.imageSmoothingQuality = "high";
      if (picture.complete && picture.naturalWidth) ctx.drawImage(picture, 0, 0, iw, ih);
      paintMarks(ctx, S.draft ? [...S.marks, S.draft] : S.marks, lw, fs, S.glow);
    });
  }
  new ResizeObserver(() => { if (S.view) { if (S.fitted) fit(); paint(); } }).observe(stage);

  // ---------------------------------------------------------------------------------------------- editing
  function commit(next) { S.undo.push(S.marks); S.redo = []; S.marks = next; S.dirty = true; refresh(); paint(); }
  function undo() { if (!S.undo.length) return; S.redo.push(S.marks); S.marks = S.undo.pop(); S.dirty = true; refresh(); paint(); }
  function redo() { if (!S.redo.length) return; S.undo.push(S.marks); S.marks = S.redo.pop(); S.dirty = true; refresh(); paint(); }
  function nearest(p) {
    let best = null, bd = 10 / S.view.s;
    for (const m of S.marks) { const d = distanceTo(m, p); if (d < bd) { bd = d; best = m; } }
    return best;
  }
  function placeText(p, sx, sy) {
    const input = h("input", { class: "notes", placeholder: "Type, then Enter",
      style: { position: "absolute", left: sx + "px", top: sy + "px", width: "200px", zIndex: 6 } });
    stage.append(input);
    input.focus();
    let ended = false;
    const end = (keep) => {
      if (ended) return;
      ended = true;
      const text = input.value.trim();
      input.remove();
      if (keep && text) {
        ctx.font = `600 ${fs}px system-ui, sans-serif`;
        const w = ctx.measureText(text).width;
        commit([...S.marks, { kind: "text", color: S.color, box: [p[0], p[1], Math.round(w), Math.round(fs * 1.3)], text }]);
      }
      el.focus({ preventScroll: true });
    };
    input.addEventListener("keydown", (e) => {
      e.stopPropagation();
      if (e.key === "Enter") end(true);
      else if (e.key === "Escape") end(false);
    });
    input.addEventListener("blur", () => end(true));
  }

  canvas.addEventListener("pointerdown", (e) => {
    if (!S.view) return;
    canvas.setPointerCapture(e.pointerId);
    const [x, y] = local(e);
    S.pointers.set(e.pointerId, [x, y]);
    if (S.pointers.size === 2) {                           // two fingers: pinch and pan, no drawing
      S.draft = null;
      const [a, b] = [...S.pointers.values()];
      S.pinch = { d: Math.hypot(a[0] - b[0], a[1] - b[1]), c: [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2] };
      paint();
      return;
    }
    if (e.button === 1 || e.button === 2 || S.space) { S.pan = [x, y]; return; }
    const p = toPic(x, y);
    if (S.tool === "erase") { const m = nearest(p); if (m) commit(S.marks.filter((k) => k !== m)); return; }
    if (S.tool === "text") { placeText(p, x, y); return; }
    S.draft = S.tool === "ink" ? { kind: "ink", color: S.color, points: [p] }
      : S.tool === "arrow" ? { kind: "arrow", color: S.color, from: p, to: p }
      : { kind: S.tool, color: S.color, box: [p[0], p[1], 0, 0], start: p };
    paint();
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!S.view) return;
    const [x, y] = local(e);
    if (S.pointers.has(e.pointerId)) S.pointers.set(e.pointerId, [x, y]);
    if (S.pinch && S.pointers.size === 2) {
      const [a, b] = [...S.pointers.values()];
      const d = Math.hypot(a[0] - b[0], a[1] - b[1]), c = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
      S.view.ox += c[0] - S.pinch.c[0]; S.view.oy += c[1] - S.pinch.c[1];
      zoomAt(c[0], c[1], d / (S.pinch.d || d));
      S.pinch = { d, c };
      paint();
      return;
    }
    if (S.pan) { S.view.ox += x - S.pan[0]; S.view.oy += y - S.pan[1]; S.pan = [x, y]; S.fitted = false; paint(); return; }
    if (!S.draft) {
      if (S.tool === "erase") { const g = nearest(toPic(x, y)); if (g !== S.glow) { S.glow = g; paint(); } }
      return;
    }
    const evs = e.getCoalescedEvents ? e.getCoalescedEvents() : [];
    const pts = (evs.length ? evs : [e]).map((ev) => toPic(...local(ev)));
    const d = S.draft, p = pts[pts.length - 1];
    if (d.kind === "ink") {
      for (const q of pts) {
        const last = d.points[d.points.length - 1];
        if (Math.hypot(q[0] - last[0], q[1] - last[1]) * S.view.s >= 1.5) d.points.push(q);
      }
    } else if (d.kind === "arrow") d.to = p;
    else {
      let w = p[0] - d.start[0], hh = p[1] - d.start[1];
      if (e.shiftKey) { const m = Math.max(Math.abs(w), Math.abs(hh)); w = Math.sign(w || 1) * m; hh = Math.sign(hh || 1) * m; }
      d.box = [Math.min(d.start[0], d.start[0] + w), Math.min(d.start[1], d.start[1] + hh), Math.abs(w), Math.abs(hh)];
    }
    paint();
  });
  const up = (e) => {
    S.pointers.delete(e.pointerId);
    if (S.pointers.size < 2) S.pinch = null;
    if (S.pan) { S.pan = null; return; }
    const d = S.draft;
    S.draft = null;
    if (!d) return;
    const small = 3 / S.view.s;
    if (d.kind === "ink" && d.points.length > 1) commit([...S.marks, d]);
    else if (d.kind === "arrow" && Math.hypot(d.to[0] - d.from[0], d.to[1] - d.from[1]) > small) commit([...S.marks, d]);
    else if (d.box && (d.box[2] > small || d.box[3] > small)) { delete d.start; commit([...S.marks, d]); }
    paint();
  };
  canvas.addEventListener("pointerup", up);
  canvas.addEventListener("pointercancel", up);
  canvas.addEventListener("contextmenu", (e) => e.preventDefault());
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    if (!S.view) return;
    const [x, y] = local(e);
    if (e.ctrlKey || Math.abs(e.deltaY) >= Math.abs(e.deltaX)) zoomAt(x, y, Math.exp(-e.deltaY * (e.deltaMode ? 0.05 : 0.0015)));
    else { S.view.ox -= e.deltaX; S.view.oy -= e.deltaY; S.fitted = false; }
    paint();
  }, { passive: false });

  // ---------------------------------------------------------------------------------------------- keys and the end
  function onKey(e) {
    if (topLayer() !== el || e.target instanceof HTMLInputElement) return;
    const k = e.key.toLowerCase();
    if ((e.ctrlKey || e.metaKey) && k === "z") { if (e.shiftKey) redo(); else undo(); }
    else if ((e.ctrlKey || e.metaKey) && k === "y") redo();
    else if (e.ctrlKey || e.metaKey || e.altKey) return;
    else if (e.key === "Escape") finish();
    else if (e.key === " ") { S.space = true; refresh(); }
    else if ("paocbte".includes(k) && k.length === 1) pick({ p: "ink", a: "arrow", o: "ellipse", c: "ellipse", b: "box", t: "text", e: "erase" }[k]);
    else if (e.key === "0") { fit(); paint(); }
    else if (e.key === "+" || e.key === "=") { const [w, hh] = size(); zoomAt(w / 2, hh / 2, 1.25); paint(); }
    else if (e.key === "-") { const [w, hh] = size(); zoomAt(w / 2, hh / 2, 0.8); paint(); }
    else return;
    e.preventDefault();
    e.stopPropagation();
  }
  document.addEventListener("keydown", onKey, true);
  document.addEventListener("keyup", onKeyUp);

  async function finish() {
    if (S.busy) return;
    if (!S.dirty) { close(); return; }
    S.busy = true;
    done.disabled = true;
    done.textContent = "Saving…";
    try {
      await onDone(S.marks.map(({ start, ...m }) => m));
      close();
    } catch (e) {
      toast("Not saved: " + e.message, "bad");
      done.disabled = false;
      done.textContent = "Done";
      S.busy = false;
    }
  }
  refresh();
  requestAnimationFrame(() => { if (!S.view) fit(); paint(); });       // its size is known: fit before it loads
}
