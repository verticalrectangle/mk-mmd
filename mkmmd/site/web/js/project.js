// A project's page: its tabs (Reviews; Music, Scene and Checks when the project has them), the first one open; `tab`
// (the page's ?tab=) opens another.
import { api } from "./api.js";
import { h, clear } from "./ui.js";
import { musicView } from "./music.js";
import { sceneView } from "./scene/scene.js";
import { checksView } from "./checks.js";

export async function openProject(root, frame, tab = null) {
  const P = await api("GET", "project", { query: { path: root } });
  document.title = P.name;
  const tabs = [["reviews", "Reviews", showReviews]];
  if (P.music) tabs.push(["music", "Music", showMusic]);
  if (P.scene) tabs.push(["scene", "Scene", showScene]);
  if (P.checks) tabs.push(["checks", "Checks", showChecks]);
  const bar = h("div", { class: "tabs" });
  const buttons = tabs.map(([id, label, fn]) => h("button", { text: label, "data-tab": id, onclick: () => go(id) }));
  bar.append(...buttons);
  clear(frame.top).append(h("div", { class: "title", text: P.name, title: P.root }), bar);
  frame.dock.hidden = true;
  let shown = null;                       // the open tab's view, disposed of when another opens

  // open a tab; `opts` go to its view (the scene's `at`, from a check card)
  async function go(id, opts = {}) {
    buttons.forEach((b) => b.classList.toggle("on", b.dataset.tab === id));
    const [, , fn] = tabs.find((t) => t[0] === id);
    if (shown && shown.dispose) shown.dispose();
    shown = null;
    clear(frame.main);
    await fn(opts);
  }

  function showMusic() {
    shown = musicView(P.root);
    frame.main.append(shown);
    shown.focus({ preventScroll: true });
  }

  function showScene(opts = {}) {
    shown = sceneView(P.root, opts);
    frame.main.append(shown);
    shown.focus({ preventScroll: true });
  }

  function showChecks() {
    shown = checksView(P.root, { onLook: P.scene ? (t) => go("scene", { at: t }) : null });
    frame.main.append(shown);
  }

  async function showReviews() {
    if (!P.reviews.length) {
      frame.main.append(h("p", { class: "muted", text: "No review files in this project yet (NAME.review.toml)." }));
      return;
    }
    frame.main.append(h("div", { class: "list" }, P.reviews.map((r) => h("a", { class: "row-link", href: "?review=" + encodeURIComponent(r.path) },
      h("span", { class: "grow", text: r.title || r.name, title: r.path }),
      r.error ? h("span", { class: "badge", text: "has problems", title: r.error }) : h("span", { class: "pill" + (r.total && r.answered === r.total ? " ok" : ""), text: `${r.answered} / ${r.total}` }),
      r.sent ? h("span", { class: "badge ok", text: "sent" }) : null))));
  }

  await go(tabs.some((t) => t[0] === tab) ? tab : tabs[0][0]);
}
