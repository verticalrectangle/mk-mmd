// Small DOM helpers shared by the site's views.

// h("div", {class: "x", onclick: fn, text: "..."}, ...children): an element; children may be nodes, strings, arrays,
// or null/false (skipped). Properties that exist on the element (value, checked ...) are set as properties.
export function h(tag, props = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "text") el.textContent = v;
    else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
    else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
    else if (k in el && typeof v !== "string") el[k] = v;
    else el.setAttribute(k, v === true ? "" : String(v));
  }
  for (const kid of kids.flat(Infinity)) {
    if (kid === null || kid === undefined || kid === false) continue;
    el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return el;
}

export function clear(el) {
  while (el.firstChild) el.firstChild.remove();
  return el;
}

// A full-pane layer over the page (the drawing view, the 3D viewer): Esc or close() removes it and gives the keys back.
export function layer(className, onClose) {
  const el = h("div", { class: "layer " + className, tabindex: "-1" });
  document.body.append(el);
  const prevFocus = document.activeElement;
  const close = () => {
    if (!el.isConnected) return;
    el.remove();
    if (prevFocus && prevFocus.focus) prevFocus.focus({ preventScroll: true });
    if (onClose) onClose();
  };
  el.focus({ preventScroll: true });
  return { el, close };
}

// The topmost layer, if any: page-wide keys go to it first.
export function topLayer() {
  const all = document.querySelectorAll("body > .layer");
  return all.length ? all[all.length - 1] : null;
}

export function toast(text, tone = "") {
  const el = h("div", { class: "toast " + tone, text });
  document.body.append(el);
  setTimeout(() => el.classList.add("gone"), 2600);
  setTimeout(() => el.remove(), 3200);
}

// A dropdown: a button showing the chosen option, opening a list under it (above it near the bottom). Tern's browser
// block draws no native popups, so a <select>'s list never opens there; the site draws its own. `options` [[value,
// label]]; `onChange(value)` when the person picks another one; `placeholder` shows while the value is "" or no option's.
// The element is the button: .value reads and sets the choice (setting it calls no onChange), .setOptions(options).
export function dropdown({ title = "", options = [], value = "", placeholder = "", onChange = null } = {}) {
  const label = h("span", { class: "dd-label" });
  const btn = h("button", { class: "dropdown", type: "button", title, "aria-haspopup": "listbox", "aria-expanded": "false" },
    label, h("span", { class: "dd-caret", text: "▾" }));
  let opts = options, current = value, list = null, active = -1;
  const show = () => {
    const o = opts.find(([v]) => v === current);
    label.textContent = o && (current !== "" || !placeholder) ? o[1] : placeholder;
  };
  const items = () => [...list.children];
  function mark(i) {
    active = i;
    items().forEach((el, k) => el.classList.toggle("active", k === i));
    const el = items()[i];                                  // into view inside the list only: a page scroll closes it
    if (!el) return;
    if (el.offsetTop < list.scrollTop) list.scrollTop = el.offsetTop;
    else if (el.offsetTop + el.offsetHeight > list.scrollTop + list.clientHeight) list.scrollTop = el.offsetTop + el.offsetHeight - list.clientHeight;
  }
  function place() {
    const r = btn.getBoundingClientRect();
    const below = innerHeight - r.bottom - 10, above = r.top - 10;
    const down = below >= Math.min(list.scrollHeight, 220) || below >= above;
    list.style.minWidth = r.width + "px";
    list.style.maxHeight = Math.max(60, down ? below : above) + "px";
    list.style.left = Math.max(8, Math.min(r.left, innerWidth - list.offsetWidth - 8)) + "px";
    list.style.top = down ? r.bottom + 4 + "px" : "";
    list.style.bottom = down ? "" : innerHeight - r.top + 4 + "px";
  }
  function open() {
    if (list || btn.disabled || !opts.length) return;
    list = h("div", { class: "dd-list", role: "listbox" }, opts.map(([v, l], i) =>
      h("div", { class: "dd-item" + (v === current ? " on" : ""), role: "option", "aria-selected": String(v === current), text: l,
        onclick: () => pick(v), onpointermove: () => { if (active !== i) mark(i); } })));
    document.body.append(list);
    place();
    mark(Math.max(0, opts.findIndex(([v]) => v === current)));
    btn.setAttribute("aria-expanded", "true");
    window.addEventListener("keydown", onKey, true);       // before the page's own keys (a viewer's Esc closes it)
    window.addEventListener("pointerdown", onDown, true);
    window.addEventListener("scroll", onScroll, true);
    window.addEventListener("resize", onResize);
  }
  function close(refocus) {
    if (!list) return;
    list.remove();
    list = null;
    btn.setAttribute("aria-expanded", "false");
    window.removeEventListener("keydown", onKey, true);
    window.removeEventListener("pointerdown", onDown, true);
    window.removeEventListener("scroll", onScroll, true);
    window.removeEventListener("resize", onResize);
    if (refocus) btn.focus({ preventScroll: true });
  }
  function pick(v) {
    close(true);
    if (v === current) return;
    current = v;
    show();
    if (onChange) onChange(v);
  }
  function onKey(e) {
    if (e.key === "Tab") { close(false); return; }
    if (e.key === "ArrowDown") mark(Math.min(opts.length - 1, active + 1));
    else if (e.key === "ArrowUp") mark(Math.max(0, active - 1));
    else if (e.key === "Home") mark(0);
    else if (e.key === "End") mark(opts.length - 1);
    else if (e.key === "Enter" || e.key === " ") pick(opts[active][0]);
    else if (e.key === "Escape") close(true);
    e.preventDefault();                                     // the open list takes every key
    e.stopPropagation();
  }
  const onDown = (e) => { if (!list.contains(e.target) && !btn.contains(e.target)) close(false); };
  const onScroll = (e) => { if (!list.contains(e.target)) close(false); };
  const onResize = () => close(false);
  btn.addEventListener("click", () => (list ? close(true) : open()));
  btn.addEventListener("keydown", (e) => {
    if (list || (e.key !== "ArrowDown" && e.key !== "ArrowUp")) return;
    e.preventDefault();
    e.stopPropagation();
    open();
  });
  Object.defineProperty(btn, "value", { get: () => current, set: (v) => { close(false); current = v; show(); } });
  btn.setOptions = (next) => { close(false); opts = next; show(); };
  show();
  return btn;
}

export const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
