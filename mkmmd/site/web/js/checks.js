// The project's checks as cards (docs/design.md: The site): failing ones first, each with what it measures in plain
// words (its metric's), the rule, its value and its worst moment (who, what, when), with a picture of that moment drawn
// from the bake through the shot camera; Look opens the scene there. Run checks runs `mk check` in the background (the
// results stay in .mk/checks.json, and the music timeline marks each worst moment).
import { api, bytes } from "./api.js";
import { h, clear } from "./ui.js";
import { SceneGL } from "./scene/draw.js";
import { shotView } from "./scene/scene.js";

const UNIT = { contact: " mm", penetration: " mm", foot_slide: " mm/frame", joint_limits: "°", camera_inside: " frames",
  strum: " mm", prop_body: " mm", wrist_bend: "°", hand_mirror: " mm", hand_room: " mm" };
const THUMB_H = 150;

const sentence = (doc) => (doc || "").split(/(?<=\.)\s/)[0];
const num = (v) => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2));

function state(r) {
  if (!r.ran) return "unrun";
  if (r.error) return "error";
  return r.ok ? "ok" : "fail";
}

function rule(r) {
  const u = UNIT[r.metric] || "", parts = [];
  if (r.min !== undefined && r.min !== null) parts.push(`at least ${num(r.min)}${u}`);
  if (r.max !== undefined && r.max !== null) parts.push(`at most ${num(r.max)}${u}`);
  return parts.join(", ");
}

// Who, what and when of the worst moment, from the result's args and detail (whatever keys the metric gives).
function worst(r) {
  const d = r.detail || {}, out = [];
  if (r.args && r.args.cast) out.push(r.args.cast);
  for (const k of ["side", "move", "worst", "bone", "into", "nearest", "point", "camera"]) {
    if (typeof d[k] === "string" && d[k]) out.push(k === "side" ? (d[k] === "L" ? "left hand" : "right hand") : d[k]);
  }
  if (r.t !== null && r.t !== undefined) out.push(`f ${d.at_frame} · ${r.t.toFixed(2)} s`);
  return out.join(" · ");
}

export function checksView(root, { onLook = null } = {}) {
  const el = h("div", { class: "checks" });
  const summary = h("span", { class: "grow muted" });
  const runBtn = h("button", { class: "btn small primary", text: "Run checks" });
  const list = h("div", { class: "checks-list" });
  el.append(h("div", { class: "checks-head" }, summary, runBtn), list);
  const S = { R: null, poll: 0, sceneLoad: null, bake: null, blob: null, scene: null, gl: null, canvas: null };

  async function load() {
    const R = await api("GET", "checks", { query: { path: root } });
    S.R = R;
    render();
    clearTimeout(S.poll);
    if (R.running) S.poll = setTimeout(() => load().catch(() => {}), 2000);
  }

  function render() {
    const rs = S.R.results, order = { fail: 0, error: 1, ok: 2, unrun: 3 };
    const n = (s) => rs.filter((r) => state(r) === s).length;
    summary.textContent = S.R.running ? "Running the checks…" : !rs.length ? "This project has no [[check]] entries."
      : `${n("fail") + n("error")} failing · ${n("ok")} passing` + (n("unrun") ? ` · ${n("unrun")} not run yet` : "");
    runBtn.disabled = S.R.running || !rs.length;
    clear(list);
    for (const r of [...rs].sort((a, b) => order[state(a)] - order[state(b)])) {
      const st = state(r), t = r.t ?? null, thumb = h("canvas", { class: "check-thumb", hidden: true });
      list.append(h("div", { class: `check-card s-${st}` },
        h("div", { class: "check-top" }, h("span", { class: "check-name", text: r.name }),
          h("span", { class: "badge " + (st === "ok" ? "ok" : st === "unrun" ? "" : "mark"),
            text: { ok: "passes", fail: "fails", error: "cannot run", unrun: "not run" }[st] })),
        h("div", { class: "check-doc", text: `${r.metric}: ${sentence(r.doc)}` }),
        h("div", { class: "check-body" },
          h("div", { class: "check-facts" },
            st === "error" ? h("div", { class: "check-error", text: r.error }) : null,
            r.value !== undefined && r.value !== null ? h("div", { class: "check-value" },
              h("strong", { text: num(r.value) + (UNIT[r.metric] || "") }),
              h("span", { class: "muted", text: rule(r) ? " · " + rule(r) : "" })) : null,
            h("div", { class: "check-worst", text: worst(r) }),
            t !== null && onLook ? h("button", { class: "btn small", text: "Look", title: "Open the scene at this moment",
              onclick: () => onLook(t) }) : null),
          thumb)));
      if (t !== null) drawThumb(thumb, r.detail.at_frame);
    }
  }

  // the bake, once, to draw each card's worst moment through the shot camera of the first output
  function scene() {
    if (!S.sceneLoad) {
      S.sceneLoad = (async () => {
        const R = await api("GET", "scene", { query: { path: root } });
        const [json, bin, glb] = await Promise.all([bytes(R.json), bytes(R.bin), bytes(R.glb)]);
        S.bake = JSON.parse(new TextDecoder().decode(json));
        S.blob = new Float32Array(bin);
        const [W, H] = S.bake.outputs[0].size;
        S.canvas = h("canvas", { width: Math.round(THUMB_H * W / H) * 2, height: THUMB_H * 2 });
        S.gl = S.canvas.getContext("webgl2", { antialias: true, preserveDrawingBuffer: true });
        S.scene = new SceneGL(S.gl, glb, S.bake, S.blob);
        await S.scene.ready;
        return true;
      })().catch(() => false);
    }
    return S.sceneLoad;
  }

  async function drawThumb(thumb, at) {
    if (!(await scene())) return;
    const b = S.bake, k = Math.max(0, Math.min(b.frames - 1, at - b.frame0));
    const sv = shotView(b, S.blob, b.outputs[0].name, k / b.fps, k);
    if (!sv) return;
    S.scene.setFrame(k);
    const gl = S.gl;
    gl.viewport(0, 0, S.canvas.width, S.canvas.height);
    gl.clearColor(0.93, 0.92, 0.9, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    S.scene.draw(sv.VP, "shaded", sv.key, [-0.6, 0.2, 0.4]);
    thumb.width = S.canvas.width;
    thumb.height = S.canvas.height;
    thumb.style.height = THUMB_H + "px";
    thumb.getContext("2d").drawImage(S.canvas, 0, 0);
    thumb.hidden = false;
  }

  runBtn.addEventListener("click", async () => {
    runBtn.disabled = true;
    await api("POST", "checks", { query: { path: root } });
    load().catch(() => {});
  });
  el.dispose = () => clearTimeout(S.poll);
  load().catch((e) => { summary.textContent = "The checks cannot be shown: " + e.message; });
  return el;
}
