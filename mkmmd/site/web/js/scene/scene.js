// The scene (docs/design.md: The site): a project's bake played to its music. 3D draws the bake: through the cut's shot
// cameras for an output (Shot: the picture's aspect, lens and shift as Blender frames it) or orbiting freely (Free: drag
// turns, right-drag or shift-drag pans, the wheel zooms, double-click fits), shaded or grey. Draft shows mk post's draft
// of that output on the same clock (WebCodecs). The music timeline under it is the clock (space plays); the label names
// the shot, the Blender frame and, in a freeze, the frame it holds.
import { api, bytes } from "../api.js";
import { h } from "../ui.js";
import { musicView } from "../music.js";
import { Orbit } from "../viewer/camera.js";
import { invert, mul } from "../viewer/math.js";
import { SceneGL, mat4of } from "./draw.js";
import { DraftPlayer } from "./draft.js";

// glTF's axes back to Blender's camera axes (it looks down its -Z with +Y up): the bake's matrices are A M A^T
const UNAXES = new Float32Array([1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1]);
const NEAR = 0.05, FAR = 400;

// Blender's projection of a camera with `lens` (mm) on a `sensor` (mm) fitted to `fit` (AUTO: the larger side), its
// shift in fractions of that side, for a W x H picture.
export function cameraProjection(lens, sensor, fit, sx, sy, W, H) {
  const S = fit === "VERTICAL" ? H : fit === "HORIZONTAL" ? W : Math.max(W, H), f = lens / sensor;
  return new Float32Array([2 * S * f / W, 0, 0, 0, 0, 2 * S * f / H, 0, 0, 2 * S * sx / W, 2 * S * sy / H,
    -(FAR + NEAR) / (FAR - NEAR), -1, 0, 0, -2 * FAR * NEAR / (FAR - NEAR), 0]);
}

const norm3 = (v) => { const l = Math.hypot(v[0], v[1], v[2]) || 1; return [v[0] / l, v[1] / l, v[2] / l]; };

// The cut's shot at clip second `t` for an output, and its camera on bake frame `k` as a view-projection with a key
// light from over the camera's shoulder: {shot, VP, key}, or null where the cut has no shot.
export function shotView(bake, blob, output, t, k) {
  const cut = bake.cut[output] || [];
  const shot = cut.find((c) => t >= c.from && t < c.to) || (cut.length && t >= cut[cut.length - 1].to ? cut[cut.length - 1] : cut[0]);
  const c = shot && bake.cameras.find((x) => x.name === shot.camera);
  if (!c) return null;
  const out = bake.outputs.find((o) => o.name === output) || bake.outputs[0], o = c.offset + 15 * k;
  const VP = mul(cameraProjection(blob[o + 12], c.sensor, c.fit, blob[o + 13], blob[o + 14], out.size[0], out.size[1]),
    mul(UNAXES, invert(mat4of(blob, o))));
  const back = [blob[o + 6], blob[o + 7], blob[o + 8]], up = [blob[o + 3], blob[o + 4], blob[o + 5]];   // the camera's +Z, +Y
  return { shot, VP, key: norm3([back[0] * 0.6 + up[0] * 0.7, back[1] * 0.6 + up[1] * 0.7, back[2] * 0.6 + up[2] * 0.7]) };
}

// `at`: open on that clip second (a check card's worst moment).
export function sceneView(root, { at = null } = {}) {
  const el = h("div", { class: "scene", tabindex: "0" });
  const S = { gl: null, scene: null, bake: null, blob: null, drafts: [], players: new Map(), output: null, view: "3d",
    mode: "shaded", cam: "shot", orbit: new Orbit(), raf: 0, last: "", drag: null };
  const outputSel = h("select", { class: "select", title: "Output" });
  const seg = (name, items) => h("div", { class: "seg small" }, items.map(([v, label]) =>
    h("button", { text: label, "data-v": v, onclick: () => { S[name] = v; refresh(); } })));
  const viewSeg = seg("view", [["3d", "3D"], ["draft", "Draft"]]);
  const camSeg = seg("cam", [["shot", "Shot"], ["free", "Free"]]);
  const modeSeg = seg("mode", [["shaded", "Shaded"], ["grey", "Grey"]]);
  const label = h("div", { class: "scene-label" });
  const canvas = h("canvas", { class: "scene-canvas" });
  const flat = h("canvas", { class: "scene-canvas draft", hidden: true });
  const stage = h("div", { class: "scene-stage" }, canvas, flat, label);
  const note = h("div", { class: "scene-note" });
  const timeline = musicView(root);
  el.append(h("div", { class: "scene-bar" }, outputSel, viewSeg, camSeg, modeSeg), stage, note, timeline);

  function refresh() {
    const hasDraft = S.drafts.includes(S.output);
    if (!hasDraft) S.view = "3d";
    for (const [segEl, key] of [[viewSeg, "view"], [camSeg, "cam"], [modeSeg, "mode"]]) {
      segEl.querySelectorAll("button").forEach((b) => b.classList.toggle("on", b.dataset.v === S[key]));
    }
    viewSeg.querySelector("[data-v=draft]").disabled = !hasDraft;
    viewSeg.querySelector("[data-v=draft]").title = hasDraft ? "mk post's draft of this output" : "No draft of this output (mk post --preset draft)";
    camSeg.hidden = modeSeg.hidden = S.view === "draft";
    canvas.hidden = S.view === "draft";
    flat.hidden = S.view !== "draft";
    if (S.view === "draft") loadDraft(S.output);
    S.last = "";
  }

  async function loadDraft(output) {
    if (S.players.has(output)) return;
    S.players.set(output, null);
    try {
      const meta = await api("GET", "draft", { query: { path: root, output } });
      if (typeof VideoDecoder === "undefined") throw new Error("this browser has no WebCodecs");
      const p = new DraftPlayer(meta, await bytes(meta.url));
      p.onFrame = () => { S.last = ""; };
      S.players.set(output, p);
      S.last = "";
    } catch (e) {
      note.textContent = "The draft cannot be shown: " + e.message;
    }
  }

  // the canvases at the output's aspect, as wide as the pane allows and at most 62 % of its height
  function size() {
    const out = S.bake.outputs.find((o) => o.name === S.output) || S.bake.outputs[0];
    const [W, H] = out.size, avail = stage.clientWidth || 300, maxH = Math.max(160, innerHeight * 0.62);
    let w = avail, hh = avail * H / W;
    if (hh > maxH) { hh = maxH; w = hh * W / H; }
    const dpr = devicePixelRatio || 1;
    for (const c of [canvas, flat]) {
      c.style.width = Math.round(w) + "px";
      c.style.height = Math.round(hh) + "px";
      if (c.width !== Math.round(w * dpr) || c.height !== Math.round(hh * dpr)) {
        c.width = Math.round(w * dpr); c.height = Math.round(hh * dpr); S.last = "";
      }
    }
    return out;
  }

  function shotAt(t) {
    const cut = S.bake.cut[S.output] || [];
    return cut.find((c) => t >= c.from && t < c.to) || (cut.length && t >= cut[cut.length - 1].to ? cut[cut.length - 1] : cut[0]) || null;
  }

  function frame() {
    S.raf = requestAnimationFrame(frame);
    if (!el.isConnected || !S.scene) return;
    const out = size(), b = S.bake, t = timeline.player ? timeline.player.now() : 0;
    const k = Math.max(0, Math.min(b.frames - 1, Math.floor(t * b.fps + 1e-6)));
    const shot = shotAt(t), held = (b.freezes || []).find(([f0, f1]) => b.frame0 + k >= f0 && b.frame0 + k < f1);
    const key = [k, S.view, S.cam, S.mode, S.output, canvas.width, canvas.height,
      S.cam === "free" ? [S.orbit.yaw, S.orbit.pitch, S.orbit.dist, ...S.orbit.target].join() : ""].join("|");
    if (key === S.last) return;
    S.last = key;
    label.textContent = `${shot ? shot.shot : "–"} · f ${b.frame0 + k}${held ? ` (held on ${held[0]})` : ""}`;
    if (S.view === "draft") {
      const p = S.players.get(S.output), vf = p ? p.frame(k) : null;
      if (vf) flat.getContext("2d").drawImage(vf, 0, 0, flat.width, flat.height);
      if (p && p.error) note.textContent = "The draft cannot be decoded: " + p.error.message;
      return;
    }
    S.scene.setFrame(k);
    const gl = S.gl;
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(0.93, 0.92, 0.9, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    let VP, keyL, fillL = [-0.6, 0.2, 0.4];
    const sv = S.cam === "shot" ? shotView(b, S.blob, S.output, t, k) : null;
    if (sv) {
      VP = sv.VP; keyL = sv.key;
    } else {
      const f = S.orbit.frame(canvas.width / canvas.height);
      VP = f.VP; keyL = f.key; fillL = f.fill;
    }
    S.scene.draw(VP, S.mode, keyL, fillL);
  }

  canvas.addEventListener("pointerdown", (e) => {
    if (S.cam !== "free") return;
    canvas.setPointerCapture(e.pointerId);
    S.drag = { x: e.clientX, y: e.clientY, pan: e.button === 2 || e.shiftKey };
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!S.drag) return;
    const dx = e.clientX - S.drag.x, dy = e.clientY - S.drag.y;
    S.drag.x = e.clientX; S.drag.y = e.clientY;
    if (S.drag.pan) S.orbit.pan(dx, dy, canvas.clientHeight); else S.orbit.orbit(dx, dy);
  });
  canvas.addEventListener("pointerup", () => { S.drag = null; });
  canvas.addEventListener("contextmenu", (e) => e.preventDefault());
  canvas.addEventListener("wheel", (e) => { if (S.cam !== "free") return; e.preventDefault(); S.orbit.zoom(Math.exp(e.deltaY * 0.0015)); }, { passive: false });
  canvas.addEventListener("dblclick", () => { if (S.scene) { S.orbit.fit(S.scene.bounds); S.last = ""; } });
  el.addEventListener("keydown", (e) => {
    if (e.key !== " " || e.target.closest("button, select, input, textarea") || !timeline.player) return;
    e.preventDefault();
    timeline.player.toggle();
  });
  outputSel.addEventListener("change", () => { S.output = outputSel.value; refresh(); });
  el.dispose = () => {
    cancelAnimationFrame(S.raf);
    for (const p of S.players.values()) if (p) p.close();
    if (timeline.dispose) timeline.dispose();
  };

  (async () => {
    try {
      const R = await api("GET", "scene", { query: { path: root } });
      note.textContent = "Loading the scene…";
      const [json, bin, glb] = await Promise.all([bytes(R.json), bytes(R.bin), bytes(R.glb)]);
      S.bake = JSON.parse(new TextDecoder().decode(json));
      S.blob = new Float32Array(bin);
      S.drafts = R.drafts || [];
      S.output = S.bake.outputs[0].name;
      outputSel.append(...S.bake.outputs.map((o) => h("option", { value: o.name, text: `${o.name} · ${o.size[0]}×${o.size[1]}` })));
      S.gl = canvas.getContext("webgl2", { antialias: true });
      if (!S.gl) throw new Error("this browser has no WebGL2");
      S.scene = new SceneGL(S.gl, glb, S.bake, S.blob);
      S.scene.onChange = () => { S.last = ""; };
      S.orbit.fit(S.scene.bounds);
      note.textContent = R.stale ? "The .blend is newer than this bake: `mk build --bake` bakes it again." : "";
      refresh();
      size();
      frame();
      if (at !== null && timeline.ready) timeline.ready.then(() => timeline.player.seek(at));
    } catch (e) {
      note.textContent = "The scene cannot be shown: " + e.message;
      note.classList.add("bad");
    }
  })();
  return el;
}
