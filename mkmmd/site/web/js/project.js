// A project's page: its tabs (Reviews; Music, Scene and Checks when the project has them), the first one open.
import { api } from "./api.js";
import { h, clear } from "./ui.js";

export async function openProject(root, frame) {
  const P = await api("GET", "project", { query: { path: root } });
  document.title = P.name;
  const tabs = [["reviews", "Reviews", showReviews]];
  const bar = h("div", { class: "tabs" });
  const buttons = tabs.map(([id, label, fn]) => h("button", { text: label, "data-tab": id, onclick: () => go(id) }));
  bar.append(...buttons);
  clear(frame.top).append(h("div", { class: "title", text: P.name, title: P.root }), bar);
  frame.dock.hidden = true;

  async function go(id) {
    buttons.forEach((b) => b.classList.toggle("on", b.dataset.tab === id));
    const [, , fn] = tabs.find((t) => t[0] === id);
    clear(frame.main);
    await fn();
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

  await go(tabs[0][0]);
}
