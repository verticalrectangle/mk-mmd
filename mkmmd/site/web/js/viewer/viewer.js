// The 3D viewer: a review's model filling the pane, in six views (Shaded, Grey, Lines, Silhouette, X-ray, Parts), with
// the camera (drag turns, right- or shift-drag pans, wheel zooms, double-click zooms to a spot; F S B T, ortho, turntable),
// Cut (a plane through it: each loop's girth, as `mk model lab` measures; a .pmx's or a spec's named girths), Measure
// (two points, mm), Compare (another model of the question, side by side or laid over: Tab), Pose (a .pmx or a spec)
// and Mark (the view as a picture to draw on, kept with its camera).
import { api, bytes } from "../api.js";
import { h, layer, toast, topLayer, clear } from "../ui.js";
import { parseGLB } from "./glb.js";
import { Renderer, partColor } from "./render.js";
import { Orbit } from "./camera.js";
import { project, sub, len } from "./math.js";

const VIEWS = [["shaded", "Shaded"], ["grey", "Grey"], ["lines", "Lines"], ["silhouette", "Silhouette"], ["xray", "X-ray"], ["parts", "Parts"]];
const POSES = [["tpose", "T-pose"], ["rest", "Rest"], ["arms_down", "Arms down"], ["sit", "Sit"]];
const CAMS = [["F", "front", "Front (1)"], ["S", "side", "Side (3)"], ["B", "back", "Back (Ctrl+1)"], ["T", "top", "Top (7)"]];
const AXES = { height: [0, 1, 0], front: [0, 0, 1], side: [1, 0, 0] };

function cssColor(name, alpha = 1) {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const m = v.match(/^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i);
  return m ? [parseInt(m[1], 16) / 255, parseInt(m[2], 16) / 255, parseInt(m[3], 16) / 255, alpha] : [0.5, 0.5, 0.5, alpha];
}

const mm = (v) => `${v.toFixed(1)} mm`;

export function openViewer({ review, model, family = [], question = null, onMark = null }) {
  const S = { mode: "shaded", items: [], cam: new Orbit(), tool: null, cut: null, measure: [], overlay: false,
    turntable: false, hidden: new Set(), solo: null, drag: null, pointers: new Map(), pinch: null, raf: 0, seq: 0, last: 0 };
  const onKey = (e) => key(e);
  const { el, close } = layer("viewer", () => {
    document.removeEventListener("keydown", onKey, true);
    cancelAnimationFrame(S.raf);
    const lose = R && R.gl.getExtension("WEBGL_lose_context");
    if (lose) lose.loseContext();
  });

  // ---------------------------------------------------------------------------------------------- layout
  const canvas = h("canvas");
  const labels = h("div", { class: "labels" });
  const spinner = h("div", { class: "busy" }, h("span", { class: "spinner" }), h("span", { class: "busy-text" }));
  const hud = h("div", { class: "hud" });
  const panel = h("div", { class: "panel", hidden: true });
  const camCol = h("div", { class: "camcol" });
  const stage = h("div", { class: "stage" }, canvas, labels, hud, camCol, panel, spinner);
  const title = h("span", { class: "title", text: model.label });
  const poseSel = h("select", { class: "select", title: "Pose", hidden: true, onchange: () => setPose(poseSel.value) },
    POSES.map(([v, l]) => h("option", { value: v, text: l })));
  const others = family.filter((m) => m.id !== model.id);
  const cmpSel = h("select", { class: "select", title: "Compare with", hidden: !others.length, onchange: () => compare(cmpSel.value) },
    h("option", { value: "", text: "Compare…" }), others.map((m) => h("option", { value: m.id, text: m.label })));
  el.append(h("div", { class: "bar top" }, title, poseSel, cmpSel), stage);
  const viewBtns = VIEWS.map(([id, label]) => h("button", { text: label, "data-v": id, onclick: () => setMode(id) }));
  const tool = (id, label, title) => h("button", { class: "btn small", text: label, title, "data-t": id, onclick: () => setTool(id) });
  const toolBtns = [tool("cut", "Cut", "Cut (c)"), tool("measure", "Measure", "Measure (m)")];
  const markBtn = h("button", { class: "btn small", text: "Mark", title: "Draw on this view", onclick: () => mark(), hidden: !onMark });
  el.append(h("div", { class: "vbar" }, h("div", { class: "seg" }, viewBtns)),
    h("div", { class: "vbar" }, toolBtns, markBtn, h("span", { class: "grow" }), h("button", { class: "btn primary small", text: "Done", onclick: () => close() })));
  for (const [label, name, tip] of CAMS) camCol.append(h("button", { class: "cam", text: label, title: tip, "data-c": name, onclick: () => preset(name) }));
  const orthoBtn = h("button", { class: "cam", text: "⊡", title: "Orthographic (5)", onclick: () => { S.cam.ortho = !S.cam.ortho; refresh(); invalidate(); } });
  const spinBtn = h("button", { class: "cam", text: "↻", title: "Turntable (r)", onclick: () => { S.turntable = !S.turntable; refresh(); invalidate(); } });
  camCol.append(orthoBtn, spinBtn, h("button", { class: "cam", text: "⤢", title: "Fit (f)", onclick: () => { fit(); invalidate(); } }));

  let R;
  try {
    R = new Renderer(canvas, () => invalidate());
  } catch (e) {
    stage.append(h("div", { class: "error-card", text: "No 3D here: " + e.message }));
    return;
  }
  const colors = () => ({ bg: cssColor("--sunk"), flat: cssColor("--fg").slice(0, 3), edge: cssColor("--muted", 0.22),
    open: [0.98, 0.55, 0.1, 1], flipped: [0.92, 0.2, 0.2, 1], ghost: cssColor("--accent").slice(0, 3), accent: cssColor("--accent"),
    second: [0.27, 0.53, 0.93, 1], faint: cssColor("--muted", 0.8) });
  let C = colors();
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { C = colors(); invalidate(); });

  // ---------------------------------------------------------------------------------------------- loading
  const busy = (text) => { spinner.hidden = !text; spinner.querySelector(".busy-text").textContent = text || ""; };
  async function load(entry, pose) {
    busy(entry.kind === "build" ? `Posing ${entry.label}…` : entry.kind === "convert" ? `Turning ${entry.label} into 3D…` : `Loading ${entry.label}…`);
    try {
      const got = await api("GET", "model", { query: { path: review, id: entry.id, ...(pose ? { pose } : {}) } });
      const data = parseGLB(await bytes(got.url));
      return { entry, pose: pose || null, posable: got.posable, data, h: R.upload(data), offset: [0, 0, 0], sections: null, measures: null };
    } finally {
      busy(null);
    }
  }
  function bounds() {
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (const it of S.items) for (let k = 0; k < 3; k++) {
      lo[k] = Math.min(lo[k], it.data.bounds.min[k] + it.offset[k]);
      hi[k] = Math.max(hi[k], it.data.bounds.max[k] + it.offset[k]);
    }
    return { min: lo, max: hi };
  }
  function layoutCompare() {
    const [a, b] = S.items;
    if (!b) return;
    if (S.overlay) { b.offset = [0, 0, 0]; return; }
    const wa = a.data.bounds.max[0] - a.data.bounds.min[0], wb = b.data.bounds.max[0] - b.data.bounds.min[0];
    b.offset = [a.data.bounds.max[0] - b.data.bounds.min[0] + 0.2 * Math.max(wa, wb), 0, 0];
  }
  function fit() { S.cam.fit(bounds()); }

  async function start() {
    try {
      S.items = [await load(model, null)];
    } catch (e) {
      stage.append(h("div", { class: "error-card" }, h("strong", { text: `${model.label} cannot be shown` }), h("pre", { text: e.message })));
      return;
    }
    poseSel.hidden = !S.items[0].posable;
    fit();
    S.cam.preset("front");
    refresh();
    invalidate();
  }
  async function setPose(pose) {
    try {
      const loads = [load(S.items[0].entry, pose)];
      if (S.items[1] && S.items[1].posable) loads.push(load(S.items[1].entry, pose));      // compare in the same pose
      const got = await Promise.all(loads);
      got.forEach((it, i) => { S.items[i] = it; });
      S.hidden.clear(); S.solo = null;
      layoutCompare();
      if (S.cut) { await measuresFor(); await recut(); }
      refresh();
      invalidate();
    } catch (e) { toast(e.message, "bad"); }
  }
  async function compare(id) {
    if (!id) { S.items.length = 1; S.overlay = false; fit(); refresh(); invalidate(); return; }
    try {
      const entry = family.find((m) => m.id === id);
      const it = await load(entry, entry.kind === "build" && S.items[0].pose ? S.items[0].pose : null);
      S.items[1] = it;
      layoutCompare();
      fit();
      if (S.cut) { await measuresFor(); await recut(); }
      refresh();
      invalidate();
    } catch (e) { toast(e.message, "bad"); cmpSel.value = ""; }
  }

  // ---------------------------------------------------------------------------------------------- drawing
  function invalidate() { if (!S.raf) S.raf = requestAnimationFrame(draw); }
  function frame() {
    const [w, hh] = R.size();
    const f = S.cam.frame(w / hh);
    const items = S.items.map((it, i) => {
      let clip = null;
      if (S.cut && it.plane) {
        const n = it.plane.d, p = it.plane.p.map((v, k) => v + it.offset[k]);
        clip = [-n[0], -n[1], -n[2], n[0] * p[0] + n[1] * p[1] + n[2] * p[2]];
      }
      return { h: it.h, offset: it.offset, hidden: i === 0 ? S.hidden : null, solo: i === 0 ? S.solo : null, ghost: S.overlay && i === 1, clip };
    });
    return { ...f, mode: S.mode, items, clip: null, overlays: overlays(), ...C };
  }
  function overlays() {
    const out = [];
    S.items.forEach((it, i) => {
      if (!it.sections) return;
      const o = it.offset, color = i === 0 ? C.accent : C.second;
      it.sections.loops.forEach((lp, k) => {
        const seg = lp.segments.map((v, j) => v + o[j % 3]);
        const main = k === it.sections.girth;
        out.push({ segments: seg, color: main ? color : C.faint, width: main ? 3 : 1.5 });
      });
    });
    if (S.measure.length) {
      const seg = [];
      const [a, b] = S.measure;
      const r = S.cam.radius * 0.012;
      for (const p of S.measure) seg.push(p[0] - r, p[1], p[2], p[0] + r, p[1], p[2], p[0], p[1] - r, p[2], p[0], p[1] + r, p[2]);
      if (b) seg.push(...a, ...b);
      out.push({ segments: seg, color: C.accent, width: 2.5 });
    }
    return out;
  }
  function draw(now) {
    S.raf = 0;
    if (!S.items.length) return;
    if (S.turntable) {
      S.cam.yaw += S.last ? (now - S.last) * 0.0005 : 0;
      S.last = now;
      S.cam.view = null;
    } else S.last = 0;
    const f = frame();
    R.draw(f);
    placeLabels(f);
    if (S.turntable) invalidate();
  }
  function screen(f, p) {
    const q = project(f.VP, p);
    if (q[3] <= 0) return null;
    return [(q[0] * 0.5 + 0.5) * canvas.clientWidth, (0.5 - q[1] * 0.5) * canvas.clientHeight];
  }
  function placeLabels(f) {
    clear(labels);
    S.items.forEach((it, i) => {
      if (!it.sections || it.sections.girth == null) return;
      const lp = it.sections.loops[it.sections.girth];
      const at = screen(f, lp.centroid.map((v, k) => v + it.offset[k]));
      if (at) labels.append(h("span", { class: "label" + (i ? " second" : ""), style: { left: at[0] + "px", top: at[1] + (i && S.overlay ? 24 : 0) + "px" },
        text: (S.cut.name ? S.cut.name.replace(/ girth$/, "") + " " : "") + mm(lp.perimeter_mm) }));
    });
    if (S.measure.length === 2) {
      const [a, b] = S.measure, mid = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2];
      const at = screen(f, mid);
      if (at) labels.append(h("span", { class: "label", style: { left: at[0] + "px", top: at[1] + "px" }, text: mm(len(sub(a, b)) * 1000) }));
    }
  }

  // ---------------------------------------------------------------------------------------------- views, camera, panels
  function setMode(mode) { S.mode = mode; refresh(); invalidate(); }
  function preset(name) { S.cam.preset(name); refresh(); invalidate(); }
  function refresh() {
    viewBtns.forEach((b) => b.classList.toggle("on", b.dataset.v === S.mode));
    toolBtns.forEach((b) => b.classList.toggle("on", b.dataset.t === S.tool));
    camCol.querySelectorAll("[data-c]").forEach((b) => b.classList.toggle("on", b.dataset.c === S.cam.view));
    orthoBtn.classList.toggle("on", S.cam.ortho);
    spinBtn.classList.toggle("on", S.turntable);
    const it = S.items[0];
    if (it && it.pose) poseSel.value = it.pose;
    const legend = S.mode === "lines" ? " · orange: open edges · red: flipped faces" : "";
    hud.textContent = it ? `${S.cam.view || "free"} · ${S.cam.ortho ? "ortho" : "perspective"}${S.items[1] ? (S.overlay ? " · laid over (Tab)" : " · side by side (Tab)") : ""}${legend}` : "";
    canvas.style.cursor = S.tool === "measure" ? "crosshair" : "grab";
    showPanel();
  }
  function showPanel() {
    clear(panel);
    if (S.tool === "cut") { cutPanel(); panel.hidden = false; return; }
    if (S.mode === "parts" && S.items[0]) { partsPanel(); panel.hidden = false; return; }
    if (S.tool === "measure") {
      panel.hidden = false;
      panel.append(h("div", { class: "muted", text: S.measure.length === 2 ? `${mm(len(sub(S.measure[0], S.measure[1])) * 1000)}: click again to start over` : "Click two points on the model" }));
      return;
    }
    panel.hidden = true;
  }
  function partsPanel() {
    const it = S.items[0];
    panel.append(h("div", { class: "panel-head" }, h("strong", { text: "Parts" }), h("span", { class: "grow" }),
      h("button", { class: "btn small", text: "All", onclick: () => { S.hidden.clear(); S.solo = null; refresh(); invalidate(); } })));
    const list = h("div", { class: "parts" });
    it.h.parts.forEach((p) => {
      const c = partColor(p.index);
      const eye = h("input", { type: "checkbox", checked: !S.hidden.has(p.index), title: "Show", onchange: () => {
        if (eye.checked) S.hidden.delete(p.index); else S.hidden.add(p.index);
        invalidate();
      } });
      list.append(h("div", { class: "part" + (S.solo === p.index ? " on" : "") }, eye,
        h("span", { class: "swatch", style: { background: `rgb(${c.slice(0, 3).map((v) => Math.round(v * 255)).join(",")})` } }),
        h("button", { class: "part-name", text: p.name, title: "Show it alone", onclick: () => { S.solo = S.solo === p.index ? null : p.index; refresh(); invalidate(); } })));
    });
    panel.append(list);
  }

  // ---------------------------------------------------------------------------------------------- cut
  function cutPanel() {
    const b = bounds(), axis = S.cut ? S.cut.axis : "height";
    const k = { height: 1, front: 2, side: 0 }[axis] ?? 1;
    const head = h("div", { class: "panel-head" }, h("strong", { text: "Cut" }), h("span", { class: "grow" }),
      ["height", "front", "side"].map((a) => h("button", { class: "btn small" + (axis === a ? " on" : ""), text: a, onclick: () => cutAt(a, null) })));
    const at = S.cut && S.cut.axis !== "named" ? S.cut.p[k] : (b.min[k] + b.max[k]) / 2;
    const slider = h("input", { type: "range", class: "range", min: b.min[k], max: b.max[k], step: (b.max[k] - b.min[k]) / 400, value: at,
      oninput: () => cutAt(axis === "named" ? "height" : axis, +slider.value) });
    panel.append(head, slider);
    const read = h("div", { class: "readout" });
    S.items.forEach((it, i) => {
      if (!it.sections) return;
      const g = it.sections.girth, loops = it.sections.loops;
      read.append(h("div", { class: i ? "second" : "" }, h("strong", { text: g == null ? "no loop here" : mm(loops[g].perimeter_mm) }),
        h("span", { class: "muted", text: ` ${S.items.length > 1 ? it.entry.label + " · " : ""}${loops.length} loop${loops.length === 1 ? "" : "s"}${loops.some((l) => !l.closed) ? " (some open)" : ""}` })));
    });
    panel.append(read);
    const named = S.items[0] && S.items[0].measures;
    if (named && named.length) {
      panel.append(h("div", { class: "chips" }, named.filter((m) => m.p).map((m) => h("button", { class: "btn small" + (S.cut && S.cut.name === m.name ? " on" : ""),
        text: `${m.name.replace(/ girth$/, "")} ${Math.round(m.mm)}`, title: `${m.name}: ${mm(m.mm)} (mk model lab)`, onclick: () => cutNamed(m) }))));
    } else if (S.items[0] && S.items[0].posable && !named) {
      panel.append(h("div", { class: "muted small", text: "Reading her measures…" }));
    }
  }
  // The named measures of every posable model shown (mk model lab's, at its pose), read once each.
  function measuresFor() {
    return Promise.all(S.items.filter((it) => it.posable && !it.measures).map((it) =>
      api("GET", "measures", { query: { path: review, id: it.entry.id, ...(it.pose ? { pose: it.pose } : {}) } })
        .then((got) => { it.measures = got.measures; })
        .catch((e) => { it.measures = []; toast("Measures: " + e.message, "bad"); })));
  }
  async function setTool(id) {
    S.tool = S.tool === id ? null : id;
    if (S.tool !== "measure") S.measure = [];
    if (S.tool === "cut") {
      if (!S.cut) {
        if (Math.abs(S.cam.pitch) < 0.2) { S.cam.pitch = 0.42; S.cam.view = null; }    // the ring seen from above, not edge-on
        await cutAt("height", null);
      }
      measuresFor().then(() => refresh());
    } else { S.cut = null; S.items.forEach((it) => { it.sections = null; it.plane = null; }); }
    refresh();
    invalidate();
  }
  async function cutAt(axis, value) {
    const b = bounds(), d = AXES[axis], k = d.indexOf(1);
    const c = [(b.min[0] + b.max[0]) / 2, (b.min[1] + b.max[1]) / 2, (b.min[2] + b.max[2]) / 2];
    const p = [...c];
    p[k] = value == null ? (axis === "height" ? b.min[1] + 0.62 * (b.max[1] - b.min[1]) : c[k]) : value;
    S.cut = { axis, p, d, target: p, name: null };
    await recut();
  }
  async function cutNamed(m) {
    S.cut = { axis: "named", p: m.p, d: m.d, target: m.target, name: m.name };
    await recut();
  }
  // Where the cut lies on one model, in its own coordinates: a named girth on its own plane when it has that measure,
  // else the shared plane moved by its offset, measured around its middle.
  function planeFor(it, cut) {
    if (cut.name) {
      const own = it === S.items[0] ? cut : (it.measures || []).find((m) => m.name === cut.name);
      if (own) return { p: own.p, d: own.d, target: own.target };
    }
    const p = cut.p.map((v, k) => v - it.offset[k]);
    const b = it.data.bounds, t = [0, 1, 2].map((k) => (b.min[k] + b.max[k]) / 2), k = cut.d.indexOf(1);
    if (k >= 0) t[k] = p[k];
    return { p, d: cut.d, target: cut.name ? cut.target.map((v, j) => v - it.offset[j]) : t };
  }
  async function recut() {
    const seq = ++S.seq, cut = S.cut;
    S.items.forEach((it) => { it.plane = planeFor(it, cut); });
    refresh();
    invalidate();
    try {
      const got = await Promise.all(S.items.map((it) => api("POST", "section", {
        query: { path: review, id: it.entry.id, ...(it.pose ? { pose: it.pose } : {}) }, body: it.plane })));
      if (seq !== S.seq) return;
      got.forEach((g, i) => { S.items[i].sections = g; });
    } catch (e) {
      if (seq === S.seq) toast("Cut: " + e.message, "bad");
    }
    refresh();
    invalidate();
  }

  // ---------------------------------------------------------------------------------------------- mark this view
  async function mark() {
    if (!onMark || !S.items.length) return;
    draw(performance.now());
    const png = canvas.toDataURL("image/png");
    const it = S.items[0];
    try {
      const view = await api("POST", "snapshot", { query: { path: review }, body: { png, model: it.entry.id, label: it.entry.label, question,
        camera: { target: S.cam.target, dist: S.cam.dist, yaw: S.cam.yaw, pitch: S.cam.pitch, ortho: S.cam.ortho, view: S.cam.view, mode: S.mode, pose: it.pose,
          compare: S.items[1] ? S.items[1].entry.id : null } } });
      onMark(view);
    } catch (e) { toast("Not kept: " + e.message, "bad"); }
  }

  // ---------------------------------------------------------------------------------------------- input
  const local = (e) => { const r = canvas.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
  canvas.addEventListener("pointerdown", (e) => {
    canvas.setPointerCapture(e.pointerId);
    const [x, y] = local(e);
    S.pointers.set(e.pointerId, [x, y]);
    if (S.pointers.size === 2) {
      const [a, b] = [...S.pointers.values()];
      S.pinch = { d: Math.hypot(a[0] - b[0], a[1] - b[1]), c: [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2] };
      S.drag = null;
      return;
    }
    S.drag = { x, y, x0: x, y0: y, pan: e.button === 1 || e.button === 2 || e.shiftKey };
  });
  canvas.addEventListener("pointermove", (e) => {
    const [x, y] = local(e);
    if (S.pointers.has(e.pointerId)) S.pointers.set(e.pointerId, [x, y]);
    if (S.pinch && S.pointers.size === 2) {
      const [a, b] = [...S.pointers.values()];
      const d = Math.hypot(a[0] - b[0], a[1] - b[1]), c = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
      S.cam.zoom(S.pinch.d / (d || S.pinch.d));
      S.cam.pan(c[0] - S.pinch.c[0], c[1] - S.pinch.c[1], canvas.clientHeight);
      S.pinch = { d, c };
      invalidate();
      return;
    }
    if (!S.drag) return;
    const dx = x - S.drag.x, dy = y - S.drag.y;
    S.drag.x = x; S.drag.y = y;
    if (S.drag.pan) S.cam.pan(dx, dy, canvas.clientHeight); else S.cam.orbit(dx, dy);
    if (!S.drag.pan) refresh();
    invalidate();
  });
  const up = (e) => {
    S.pointers.delete(e.pointerId);
    if (S.pointers.size < 2) S.pinch = null;
    const d = S.drag;
    S.drag = null;
    if (!d || Math.hypot(d.x - d.x0, d.y - d.y0) > 4 || S.tool !== "measure" || d.pan) return;
    const p = R.pick(frame(), d.x0, d.y0);
    if (!p) return;
    S.measure = S.measure.length >= 2 ? [p] : [...S.measure, p];
    refresh();
    invalidate();
  };
  canvas.addEventListener("pointerup", up);
  canvas.addEventListener("pointercancel", up);
  canvas.addEventListener("contextmenu", (e) => e.preventDefault());
  canvas.addEventListener("dblclick", (e) => {
    const [x, y] = local(e);
    const p = R.pick(frame(), x, y);
    if (p) { S.cam.focus(p); refresh(); invalidate(); }
  });
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    S.cam.zoom(Math.exp(e.deltaY * (e.deltaMode ? 0.05 : 0.0012)));
    invalidate();
  }, { passive: false });
  new ResizeObserver(() => invalidate()).observe(stage);

  function key(e) {
    if (topLayer() !== el || e.target instanceof HTMLSelectElement || e.target instanceof HTMLInputElement && e.target.type !== "range") return;
    const k = e.key.toLowerCase(), ctrl = e.ctrlKey || e.metaKey;
    if (k === "1") preset(ctrl ? "back" : "front");
    else if (k === "3") preset(ctrl ? "other side" : "side");
    else if (k === "7") preset(ctrl ? "bottom" : "top");
    else if (ctrl || e.altKey) return;
    else if (k === "5") { S.cam.ortho = !S.cam.ortho; refresh(); invalidate(); }
    else if (k === "f") { fit(); invalidate(); }
    else if (k === "r") { S.turntable = !S.turntable; refresh(); invalidate(); }
    else if (k === "v") { const i = VIEWS.findIndex(([v]) => v === S.mode); setMode(VIEWS[(i + (e.shiftKey ? VIEWS.length - 1 : 1)) % VIEWS.length][0]); }
    else if (k === "c") setTool("cut");
    else if (k === "m") setTool("measure");
    else if (e.key === "Tab" && S.items[1]) { S.overlay = !S.overlay; layoutCompare(); fit(); if (S.cut) recut(); refresh(); invalidate(); }
    else if (e.key === "Escape") { if (S.tool) setTool(S.tool); else close(); }
    else return;
    e.preventDefault();
    e.stopPropagation();
  }
  document.addEventListener("keydown", onKey, true);
  start();
}
