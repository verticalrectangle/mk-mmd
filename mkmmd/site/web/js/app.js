// The site's entry: ?review=PATH opens a review, ?project=DIR a project's page. An open page pings the server so it
// stays up while anyone looks.
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
    if (params.get("review")) await openReview(params.get("review"), frame);
    else if (params.get("project")) await openProject(params.get("project"), frame);
    else fail("Nothing to show", "Open a page with `mk review open FILE` or `mk site`.");
  } catch (e) {
    fail(params.get("review") ? "This review cannot be shown" : "This page cannot be shown", e);
  }
})();
