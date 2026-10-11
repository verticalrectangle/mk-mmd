// The site's entry: ?project=DIR opens a project's page (?tab= one of its tabs, ?review=PATH that review in its Reviews
// tab), ?review=PATH alone a review that is in no project. An open page pings the server so it stays up while anyone looks.
import { api } from "./api.js";
import { h, clear } from "./ui.js";
import { openReview } from "./review.js";
import { openProject } from "./project.js";

const frame = { top: document.getElementById("top"), main: document.getElementById("main"), dock: document.getElementById("dock") };
const params = new URLSearchParams(location.search);

function fail(title, err) {
  clear(frame.main).append(h("div", { class: "error-card" }, h("strong", { text: title }),
    h("pre", { text: String((err && err.message) || err) }),
    h("button", { class: "btn", text: "Read it again", onclick: () => location.reload() })));
}

setInterval(() => { api("GET", "ping").catch(() => {}); }, 30000);

(async () => {
  try {
    if (params.get("project")) await openProject(params.get("project"), frame, params.get("tab"), params.get("review"));
    else if (params.get("review")) await openReview(params.get("review"), frame);
    else fail("Nothing to show", "Open a page with `mk review open FILE` or `mk site`.");
  } catch (e) {
    fail(params.get("project") || !params.get("review") ? "This page cannot be shown" : "This review cannot be shown", e);
  }
})();
