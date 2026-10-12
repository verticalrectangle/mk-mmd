// The exported page (`mk export web`, docs/design.md: Exporting the page): the project's scene played to its music
// timeline from static files, the 3D loaded on demand. Query: view=3d|draft (the view it opens on), theme=light|dark|mono
// (else the system's), maxh=PX (the stage's largest height). In an iframe it tells its host page its height and its
// clock ({source: "mk-export", type: "height", height} and {source: "mk-export", type: "clock", t, playing}), and takes
// {source: "mk-host", type: "theme", theme}.
import { sceneView } from "./scene/scene.js";

const params = new URLSearchParams(location.search);
const html = document.documentElement;
const THEMES = ["light", "dark", "mono"];

function setTheme(t) {
  if (THEMES.includes(t)) html.dataset.theme = t;
  else delete html.dataset.theme;
}
setTheme(params.get("theme"));

const view = sceneView("", { view: params.get("view") === "draft" ? "draft" : "3d", lazy: true, draftLabel: "Video",
  maxHeight: +params.get("maxh") || null });
document.getElementById("main").append(view);

const hosted = window.parent !== window;
const post = (msg) => { if (hosted) window.parent.postMessage({ source: "mk-export", ...msg }, "*"); };

addEventListener("message", (e) => {
  const m = e.data;
  if (!m || m.source !== "mk-host" || e.source !== window.parent) return;
  if (m.type === "theme") setTheme(m.theme);
});

new ResizeObserver(() => post({ type: "height", height: Math.ceil(html.scrollHeight) })).observe(document.body);

// The clock, posted when the transport changes, when the playhead jumps, and every half second while it plays.
let last = { playing: null, t: -1, at: 0 };
function watch() {
  requestAnimationFrame(watch);
  const p = view.player, now = performance.now(), playing = p.playing(), t = p.now();
  const expected = last.playing ? last.t + (now - last.at) / 1000 : last.t;
  if (playing !== last.playing || Math.abs(t - expected) > 0.08 || (playing && now - last.at > 500)) {
    post({ type: "clock", t, playing });
    last = { playing, t, at: now };
  }
}
if (hosted) watch();
