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

export const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
