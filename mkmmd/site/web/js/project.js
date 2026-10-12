// A project's page: its tabs (Brief and Reviews; Music, Scene and Checks when the project has them), the brief open
// while it is written but not sent, else the reviews; `tab` (the page's ?tab=) opens another, `review` (?review=) opens
// that review in the Reviews tab. A review opens inside the page, under the tabs, and ‹ Reviews (or the Reviews tab)
// goes back to the list.
import { api } from "./api.js";
import { h, clear } from "./ui.js";
import { openReview } from "./review.js";
import { musicView } from "./music.js";
import { sceneView } from "./scene/scene.js";
import { checksView } from "./checks.js";
import { briefView } from "./brief.js";

export async function openProject(root, frame, tab = null, review = null) {
  let P = await api("GET", "project", { query: { path: root } });
  const tabs = [["brief", "Brief", showBrief], ["reviews", "Reviews", showReviews]];
  if (P.music) tabs.push(["music", "Music", showMusic]);
  if (P.scene) tabs.push(["scene", "Scene", showScene]);
  if (P.checks) tabs.push(["checks", "Checks", showChecks]);
  const bar = h("div", { class: "tabs" });
  const buttons = tabs.map(([id, label, fn]) => h("button", { text: label, "data-tab": id, onclick: () => go(id) }));
  bar.append(...buttons);
  clear(frame.top).append(h("div", { class: "title", text: P.name, title: P.root }), bar);
  frame.dock.hidden = true;
  let shown = null;                       // the open tab's view, disposed of when another opens
  let stale = false;                      // a review was open: the list's answer counts may have changed

  // open a tab; `opts` go to its view (the scene's `at`, from a check card; the review the Reviews tab opens)
  async function go(id, opts = {}) {
    buttons.forEach((b) => b.classList.toggle("on", b.dataset.tab === id));
    const [, , fn] = tabs.find((t) => t[0] === id);
    if (shown && shown.dispose) shown.dispose();
    shown = null;
    document.title = P.name;
    clear(frame.main);
    history.replaceState(null, "", "?" + new URLSearchParams({ project: root, tab: id, ...(opts.review ? { review: opts.review } : {}) }));
    await fn(opts);
  }

  function showBrief() {
    shown = briefView(P.root, { dock: frame.dock });
    frame.main.append(shown);
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

  async function showReviews(opts = {}) {
    if (opts.review) return showReview(opts.review);
    if (stale) {
      P = await api("GET", "project", { query: { path: root } });
      stale = false;
    }
    if (!P.reviews.length) {
      frame.main.append(h("p", { class: "muted", text: "No review files in this project yet (NAME.review.toml)." }));
      return;
    }
    frame.main.append(h("div", { class: "list" }, P.reviews.map((r) => h("a", { class: "row-link", href: "?" + new URLSearchParams({ project: root, review: r.path }),
      onclick: (e) => { if (e.button || e.ctrlKey || e.metaKey || e.shiftKey) return; e.preventDefault(); go("reviews", { review: r.path }); } },
      h("span", { class: "grow", text: r.title || r.name, title: r.path }),
      r.error ? h("span", { class: "badge", text: "has problems", title: r.error }) : h("span", { class: "pill" + (r.total && r.answered === r.total ? " ok" : ""), text: `${r.answered} / ${r.total}` }),
      r.sent ? h("span", { class: "badge ok", text: "sent" }) : null))));
  }

  // the review under the tabs: ‹ Reviews, its title and count, its questions; Send in the dock
  async function showReview(path) {
    stale = true;
    const slot = h("div", { class: "slot" });
    const body = h("div");
    frame.main.append(h("div", { class: "review-head" }, h("button", { class: "btn small", text: "‹ Reviews", onclick: () => go("reviews") }), slot), body);
    let open = null, gone = false;
    shown = { dispose: () => { gone = true; if (open) open.dispose(); } };
    try {
      open = await openReview(path, { top: slot, main: body, dock: frame.dock });
    } catch (e) {
      body.append(h("div", { class: "error-card" }, h("strong", { text: "This review cannot be shown" }), h("pre", { text: e.message })));
      frame.dock.hidden = true;
      clear(frame.dock);
      return;
    }
    if (gone) open.dispose();             // another tab opened while it loaded
  }

  const first = P.brief && P.brief.written && !P.brief.sent ? "brief" : "reviews";
  await go(review ? "reviews" : tabs.some((t) => t[0] === tab) ? tab : first, review ? { review } : {});
}
