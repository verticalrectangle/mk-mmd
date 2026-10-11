// The Markdown a review's texts use, as DOM nodes (never HTML strings): paragraphs (single line breaks are spaces),
// "- " / "* " / "1. " lists with indented continuation lines, "#" headings, `code`, **bold**, *italic* and [text](url).

const LIST = /^\s*([-*]|\d+\.)\s+/;

export function markdown(text, className = "md") {
  const root = document.createElement("div");
  root.className = className;
  for (const block of String(text || "").replace(/\r/g, "").trim().split(/\n\s*\n/)) {
    const lines = block.split("\n");
    if (LIST.test(lines[0])) {
      const ordered = /^\s*\d+\./.test(lines[0]);
      const list = document.createElement(ordered ? "ol" : "ul");
      let item = null, parts = [];
      const flush = () => { if (item) { inline(parts.join(" "), item); list.append(item); } };
      for (const line of lines) {
        if (LIST.test(line)) {
          flush();
          item = document.createElement("li");
          parts = [line.replace(LIST, "")];
        } else parts.push(line.trim());
      }
      flush();
      root.append(list);
    } else if (/^#{1,4}\s/.test(lines[0])) {
      const level = Math.min(4, lines[0].match(/^#+/)[0].length);
      const head = document.createElement("h" + (level + 2));
      inline(lines[0].replace(/^#+\s*/, ""), head);
      root.append(head);
      if (lines.length > 1) root.append(inline(lines.slice(1).join(" "), document.createElement("p")));
    } else {
      root.append(inline(lines.map((l) => l.trim()).join(" "), document.createElement("p")));
    }
  }
  return root;
}

const TOKEN = /(`[^`]+`)|(\*\*[^*]+\*\*)|(\*[^*\s][^*]*\*|_[^_\s][^_]*_)|(\[[^\]]+\]\([^)\s]+\))/g;

export function inline(text, parent) {
  let at = 0;
  for (const m of text.matchAll(TOKEN)) {
    if (m.index > at) parent.append(text.slice(at, m.index));
    const t = m[0];
    let el;
    if (m[1]) { el = document.createElement("code"); el.textContent = t.slice(1, -1); }
    else if (m[2]) { el = document.createElement("strong"); inline(t.slice(2, -2), el); }
    else if (m[3]) { el = document.createElement("em"); inline(t.slice(1, -1), el); }
    else {
      const [, label, href] = t.match(/^\[([^\]]+)\]\(([^)\s]+)\)$/);
      el = document.createElement("a");
      el.textContent = label;
      if (/^(https?:|mailto:)/.test(href)) { el.href = href; el.target = "_blank"; el.rel = "noreferrer"; }
    }
    parent.append(el);
    at = m.index + t.length;
  }
  if (at < text.length) parent.append(text.slice(at));
  return parent;
}
