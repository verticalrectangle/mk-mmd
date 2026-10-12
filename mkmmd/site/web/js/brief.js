// The Brief tab (docs/design.md: Briefs): what the person wants from the video, written before anything is built. The
// song on one canvas, every lane on one scale in song seconds: the part lane (the part of the song as a bar with a
// handle at each end: drag a handle, drag the bar, or draw a new one; it becomes the project's clip, with Undo), the
// time ruler (the playhead), the waveform, then once the words are timed the bars and the lines by number (never their
// text), and the parts lane, each part a line, a group of lines or any stretch with a card: what the person wants
// there, pictures, whether it comes back the same or bigger. Nothing zooms by itself: Song and Part (and Ctrl+wheel)
// do. The lyrics are pasted once and kept in the project (never shown again); Time the words runs `mk timeline
// analyze` in the background; the video as a whole has its own card. Saved as it changes; Send (in the dock) types it
// into the pane of the agent that opened the page.
import { api } from "./api.js";
import { h, clear, clamp, toast, dropdown } from "./ui.js";
import { decode, context, peaks, niceStep, colours, claim, release } from "./music.js";

const GUTTER = 50, PAD = 8;
const ROW = { part: 30, ruler: 20, wave: 38, beats: 18, lines: 24, parts: 28 };
const SNAP_PX = 7, GRAB_PX = 10, TAB_W = 11, PAN_PX = 16, MIN_PART = 0.1, MIN_CLIP = 1.0;
const STAGES = { stems: "separating the voice from the band", whisper: "hearing the words", align: "timing each word" };
const SIZE_LABEL = { "9x16": "9:16", "1x1": "1:1", "16x9": "16:9" };
const FIELDS = [
  ["story", "The idea", "What happens, what it is about, how it should feel", 3],
  ["cast", "Who's in it", "The characters, the models", 1],
  ["place", "Where", "The places, the sets", 1],
  ["look", "The look", "Colours, light, style, videos it should feel like", 2],
  ["posting", "Posted on", "TikTok, RedNote, X, YouTube…", 1],
];
const PART_KINDS = ["clip", "edge", "move"];                            // drags on the part lane
const PANNING_KINDS = [...PART_KINDS, "lines", "newpart", "resize", "scrub"];   // drags that pan the view at its edges

const mmss = (t) => {
  const m = Math.floor(Math.max(0, t) / 60), s = Math.max(0, t) - 60 * m;
  return `${m}:${s.toFixed(1).padStart(4, "0")}`;
};
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const readFile = (f) => new Promise((ok, bad) => { const r = new FileReader(); r.onload = () => ok(r.result); r.onerror = () => bad(r.error); r.readAsDataURL(f); });

export function briefView(root, { dock }) {
  const el = h("div", { class: "brief" }, h("p", { class: "faint", text: "Loading…" }));
  const S = {
    B: null, doc: null, clip: [0, 0], dur: 0, view: [0, 1], sel: null, word: null, lastNote: null,
    buf: null, norm: 1, src: null, t0: 0, p0: 0, at: 0, stopAt: null, layer: null, C: null, w: 0, L: null,
    height: 0, drag: null, px: null, hover: null, raf: 0, timer: 0, poll: 0, undo: 0, gone: false, cards: new Map(),
  };
  const status = h("span", { class: "status grow" });
  const sendBtn = h("button", { class: "btn primary", text: "Send to the agent", onclick: () => send() });
  dock.hidden = false;
  clear(dock).append(status, sendBtn);
  const say = (text, tone = "muted") => { status.textContent = text; status.className = "status grow " + tone; };

  // ------------------------------------------------------------------------------------------------- saving, Send
  const saveSoon = () => { clearTimeout(S.timer); S.timer = setTimeout(() => save().catch(() => {}), 450); say("Saving…"); };
  async function save() {
    clearTimeout(S.timer);
    S.timer = 0;
    try {
      const got = await api("PUT", "brief", { query: { path: root }, body: { brief: S.doc } });
      S.B.brief = got.brief;
      say(S.B.brief.sent && S.B.brief.saved > S.B.brief.sent ? "Saved · changed since it was sent" : "Saved");
    } catch (e) {
      say("Not saved: " + e.message, "bad");
      throw e;
    }
  }
  async function send() {
    sendBtn.disabled = true;
    try {
      await save();
      const got = await api("POST", "send_brief", { query: { path: root } });
      S.B.brief.sent = got.sent;
      const at = new Date(got.sent).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
      if (got.delivered) {
        say(`Sent at ${at} to the agent's pane: the treatment comes back in Reviews`, "ok");
        toast("Sent to the agent");
      } else {
        say(`Saved at ${at}, but no agent opened this page: tell yours to read it (mk brief show)` + (got.why ? ` (${got.why})` : ""), "bad");
        toast("Saved: no agent is waiting", "bad");
      }
    } catch (e) {
      say("Not sent: " + e.message, "bad");
    } finally {
      sendBtn.disabled = false;
      sendBtn.textContent = S.B && S.B.brief.sent ? "Send again" : "Send to the agent";
    }
  }

  // ------------------------------------------------------------------------------------------------- the song card
  const playBtn = h("button", { class: "btn small play", text: "▶", title: "Play (space)" });
  const clock = h("span", { class: "clock", text: "" });
  const songBtn = h("button", { class: "btn small", text: "Song", title: "Show the whole song" });
  const partBtn = h("button", { class: "btn small", text: "Part", title: "Show the part up close (f)" });
  const cv = h("canvas", { class: "lanes" });
  const readout = h("div", { class: "music-readout" });
  const wordBtn = h("button", { class: "btn small", hidden: true, onclick: () => putWord() });
  const undoBtn = h("button", { class: "btn small", hidden: true, text: "Undo" });
  const partLine = h("div", { class: "brief-part" });
  const songBox = h("div", { class: "music brief-song", tabindex: "0" },
    h("div", { class: "transport" }, playBtn, clock, h("span", { class: "grow" }), songBtn, partBtn), cv,
    h("div", { class: "brief-readout" }, readout, wordBtn, undoBtn),
    h("div", { class: "hint-line", text: "drag the part's handles or the part itself, or draw a new one on the part lane · click a line to write about it · drag across lines or along the parts lane for a section · Song and Part zoom · space plays" }));
  const lyricsRow = h("div", { class: "brief-row" });
  const langRow = h("div", { class: "brief-row" });
  const wordsRow = h("div", { class: "brief-row" });
  const cardsBox = h("div", { class: "brief-parts" });

  const x0 = () => GUTTER, x1 = () => S.w - PAD;
  const tx = (t) => x0() + (t - S.view[0]) / (S.view[1] - S.view[0]) * (x1() - x0());
  const xt = (x) => S.view[0] + (x - x0()) / (x1() - x0()) * (S.view[1] - S.view[0]);
  const local = (e) => { const r = cv.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
  const lane = (id) => S.L && S.L.lanes.find((l) => l.id === id);
  const laneAt = (y) => S.L && S.L.lanes.find((l) => y >= l.y && y < l.y + l.h);

  // ---- a message under the song, with Undo for what can be taken back
  function offerUndo(text, fn) {
    readout.textContent = text;
    undoBtn.hidden = false;
    undoBtn.onclick = () => { undoBtn.hidden = true; fn(); };
    clearTimeout(S.undo);
    S.undo = setTimeout(() => { undoBtn.hidden = true; }, 20000);
  }

  // ------------------------------------------------------------------------------------------------- the parts
  const T = () => S.B.timeline;
  const parts = () => S.doc.parts;
  const partOf = (id) => parts().find((p) => p.id === id);
  const sortParts = () => parts().sort((a, b) => a.from - b.from);
  const numberOf = (id) => parts().findIndex((p) => p.id === id) + 1;
  const linesIn = (a, b) => (T() ? T().lines.filter((ln) => (ln.start + ln.end) / 2 >= a && (ln.start + ln.end) / 2 <= b) : []);
  function whereText(p) {
    const ls = linesIn(p.from, p.to);
    const lines = !ls.length ? "" : ls.length === 1 ? `line ${ls[0].line} · ` : `lines ${ls[0].line}–${ls[ls.length - 1].line} · `;
    return `${lines}${mmss(p.from)}–${mmss(p.to)}`;
  }
  const inClip = (p) => p.from >= S.clip[0] - 0.01 && p.to <= S.clip[1] + 0.01;
  // [a, b] trimmed so that it overlaps no other part (the parts in time order), or null when nothing is left
  function free(a, b, except = null) {
    for (const p of [...parts()].sort((x, y) => x.from - y.from)) {
      if (p.id === except || p.to <= a || p.from >= b) continue;
      if (p.from <= a) a = p.to; else b = p.from;
    }
    return b - a >= MIN_PART ? [a, b] : null;
  }
  function addPart(a, b) {
    if (b <= S.clip[0] || a >= S.clip[1]) {
      readout.textContent = "Parts go inside the part of the song: set it on the part lane first.";
      return null;
    }
    [a, b] = [Math.max(a, S.clip[0]), Math.min(b, S.clip[1])];
    const span = b - a >= MIN_PART ? free(a, b) : null;
    if (!span) { readout.textContent = "There is no room for a part there: another part holds it."; return null; }
    let n = 0;
    for (const p of parts()) n = Math.max(n, +p.id.slice(1));
    const p = { id: "p" + (n + 1), from: +span[0].toFixed(3), to: +span[1].toFixed(3), note: "", pictures: [], echo: null };
    parts().push(p);
    sortParts();
    changed(true);
    select(p.id, true);
    return p;
  }
  function removePart(id) {
    const k = numberOf(id), p = partOf(id);
    if (!p) return;
    const echoes = parts().filter((q) => q.echo && q.echo.of === id).map((q) => [q.id, q.echo.how]);
    S.doc.parts = parts().filter((q) => q.id !== id);
    for (const q of parts()) if (q.echo && q.echo.of === id) q.echo = null;
    if (S.sel === id) S.sel = null;
    changed(true);
    offerUndo(`Part ${k} deleted.`, () => {
      const span = free(p.from, p.to);
      if (!span || span[0] !== p.from || span[1] !== p.to) { readout.textContent = "Another part is there now: the deleted one cannot come back."; return; }
      parts().push(p);
      for (const [qid, how] of echoes) { const q = partOf(qid); if (q && q.from > p.from) q.echo = { of: id, how }; }
      sortParts();
      changed(true);
      select(id, true);
      readout.textContent = `Part ${numberOf(id)} is back.`;
    });
  }
  // a line click: the part that holds the line, else a new part for it; with Shift the selected part reaches it
  function pickLine(ln, extend) {
    const mid = (ln.start + ln.end) / 2;
    const holder = parts().find((p) => p.from <= mid && p.to >= mid);
    if (extend && S.sel && partOf(S.sel)) {
      const p = partOf(S.sel), span = free(Math.min(p.from, ln.start), Math.max(p.to, ln.end), p.id);
      if (span) { p.from = +span[0].toFixed(3); p.to = +span[1].toFixed(3); sortParts(); changed(true); }
      return;
    }
    if (holder) select(holder.id, true);
    else addPart(ln.start, ln.end);
  }
  function select(id, focus = false) {
    S.sel = id;
    for (const [pid, c] of S.cards) c.el.classList.toggle("on", pid === id);
    const p = partOf(id);
    if (p) reveal(p.from, p.to);
    invalidate(true);
    showWordBtn();
    const c = S.cards.get(id);
    if (c && focus) {
      c.el.scrollIntoView({ block: "nearest", behavior: "smooth" });
      c.note.focus({ preventScroll: true });
    }
  }
  // something about the parts changed: save; `structure` when cards must be laid out again
  function changed(structure = false) {
    if (structure) renderCards();
    invalidate(true);
    saveSoon();
  }

  // ------------------------------------------------------------------------------------------------- the canvas
  function layout() {
    const L = [{ id: "part", h: ROW.part, label: "part" }, { id: "ruler", h: ROW.ruler, label: "time" },
      { id: "wave", h: ROW.wave, label: "song" }];
    if (T() && T().beats.length) L.push({ id: "beats", h: ROW.beats, label: "bars" });
    if (T() && T().lines.length) L.push({ id: "lines", h: ROW.lines, label: "lines" });
    L.push({ id: "parts", h: ROW.parts, label: "parts" });
    let y = 0;
    for (const l of L) { l.y = y; y += l.h; }
    S.L = { lanes: L };
    S.height = y + 2;
  }
  // the view on the whole song (Song) or on the part with a little of the song around it (Part); nothing else zooms
  function showSong() { S.view = [0, S.dur]; invalidate(true); }
  function showPart() {
    const [a, b] = S.clip, pad = (b - a) * 0.06;
    S.view = b - a >= S.dur - 0.05 ? [0, S.dur] : [Math.max(0, a - pad), Math.min(S.dur, b + pad)];
    invalidate(true);
  }
  // pan the view, at the same zoom, so that a..b is in it (wider only when a..b does not fit)
  function reveal(a, b) {
    const [v0, v1] = S.view, w = v1 - v0;
    if (a >= v0 && b <= v1) return;
    if (b - a <= w * 0.9) { const c = (a + b) / 2, s = clamp(c - w / 2, 0, S.dur - w); S.view = [s, s + w]; }
    else { const pad = (b - a) * 0.06; S.view = [Math.max(0, a - pad), Math.min(S.dur, b + pad)]; }
    invalidate(true);
  }

  // ---- playing (song seconds; one view plays at a time)
  const offset = () => (S.B.song && S.B.song.offset) || 0;
  function now() {
    if (!S.src) return S.at;
    const ctx = context();
    const t = S.at + (ctx.state === "running" ? ctx.currentTime - S.t0 : (performance.now() - S.p0) / 1000);
    return Math.min(t, S.stopAt ?? S.dur);
  }
  function stopSrc() {
    if (!S.src) return;
    const s = S.src;
    S.src = null;
    try { s.stop(); } catch (e) { /* not started */ }
    s.disconnect();
  }
  function startAt(t, stopAt = null) {
    stopSrc();
    claim(player);
    const ctx = context(), src = ctx.createBufferSource();
    if (ctx.state !== "running") ctx.resume().catch(() => {});
    src.buffer = S.buf;
    src.connect(ctx.destination);
    const end = stopAt ?? S.dur;
    src.start(0, offset() + t, Math.max(0, end - t));
    src.onended = () => { if (S.src === src) { S.src = null; S.at = end; S.stopAt = null; refresh(); invalidate(); } };
    S.src = src; S.t0 = ctx.currentTime; S.p0 = performance.now(); S.at = t; S.stopAt = stopAt;
    refresh(); invalidate();
  }
  function play() {
    if (!S.buf) return;
    let t = now();
    if (t >= S.dur - 0.01) t = S.clip[0];
    startAt(t);
  }
  function pause() { if (!S.src) return; S.at = now(); stopSrc(); S.stopAt = null; refresh(); invalidate(); }
  function seek(t) { t = clamp(t, 0, S.dur); if (S.src) startAt(t, S.stopAt); else { S.at = t; refresh(); invalidate(); } }
  const playSpan = (a, b) => { if (S.buf) { reveal(a, b); startAt(a, b); } };
  const player = { pause };
  function refresh() {
    playBtn.textContent = S.src ? "❚❚" : "▶";
    playBtn.disabled = !S.buf;
  }

  // ---- drawing: a layer of what only changes with the view (redrawn on demand), the moving parts over it
  function paintLayer() {
    const C = S.C, dpr = devicePixelRatio || 1, w = S.w, H = S.height;
    const c = S.layer || (S.layer = document.createElement("canvas"));
    c.width = Math.round(w * dpr); c.height = Math.round(H * dpr);
    const g = c.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, H);
    const font = (px, weight = 400) => `${weight} ${px}px ${css("--font") || "sans-serif"}`;
    const [v0, v1] = S.view, visible = (a, b) => b >= v0 && a <= v1;
    const cols = Math.max(1, Math.round(x1() - x0()));
    g.textBaseline = "middle";
    for (const l of S.L.lanes) {
      g.fillStyle = C.line; g.fillRect(0, l.y + l.h - 0.5, w, 0.5);
      if (l.label) { g.font = font(10.5); g.fillStyle = C.muted; g.textAlign = "right"; g.fillText(l.label, GUTTER - 8, l.y + l.h / 2); }
      g.save();
      g.beginPath(); g.rect(x0() - 1, l.y, x1() - x0() + 2, l.h); g.clip();
      const mid = l.y + l.h / 2;
      if (l.id === "ruler") {
        const step = niceStep((v1 - v0) / (x1() - x0()));
        g.font = font(10); g.fillStyle = C.faint; g.strokeStyle = C.faint; g.textAlign = "left";
        for (let t = Math.ceil(v0 / step) * step; t <= v1 + 1e-9; t += step) {
          const x = Math.round(tx(t)) + 0.5;
          g.beginPath(); g.moveTo(x, l.y + l.h - 6); g.lineTo(x, l.y + l.h); g.stroke();
          g.fillText(step < 1 ? mmss(t) : mmss(t).replace(/\.0$/, ""), x + 3, l.y + l.h / 2 - 2);
        }
      } else if (l.id === "wave") {
        if (S.buf) {
          const p = peaks(S.buf, offset(), v0, v1, cols), amp = (l.h / 2 - 3) / S.norm;
          g.fillStyle = C.muted;
          for (let i = 0; i < cols; i++) g.fillRect(x0() + i, mid - p[2 * i + 1] * amp, 1, Math.max(1, (p[2 * i + 1] - p[2 * i]) * amp));
        } else {
          g.font = font(10.5); g.fillStyle = C.faint; g.textAlign = "left";
          g.fillText(S.B.song ? "Decoding the song…" : "No song", x0() + 6, mid);
        }
      } else if (l.id === "beats") {
        const down = new Set(T().downbeats.map((t) => t.toFixed(3)));
        g.font = font(9.5, 600); g.textAlign = "left";
        for (const t of T().beats) {
          if (!visible(t, t)) continue;
          const x = Math.round(tx(t)) + 0.5, d = down.has(t.toFixed(3));
          g.strokeStyle = d ? C.fg : C.faint; g.lineWidth = d ? 1.5 : 1;
          g.beginPath(); g.moveTo(x, d ? l.y + 2 : mid - 3); g.lineTo(x, d ? l.y + l.h - 2 : mid + 3); g.stroke();
        }
        g.fillStyle = C.fg;
        T().downbeats.forEach((t, k) => { if (visible(t, t)) g.fillText(String(k + 1), tx(t) + 3, l.y + 6); });
      } else if (l.id === "lines") {
        g.font = font(10, 650); g.textAlign = "center";
        for (const ln of T().lines) {
          if (!visible(ln.start, ln.end)) continue;
          const a = tx(ln.start), b = Math.max(tx(ln.end), a + 2);
          g.globalAlpha = 0.3; g.fillStyle = ln.line % 2 ? C.ok : C.accent; g.fillRect(a, l.y + 3, b - a, l.h - 6);
          g.globalAlpha = 0.55; g.fillStyle = C.fg;
          for (const [s] of ln.words.slice(1)) { const x = Math.round(tx(s)) + 0.5; g.fillRect(x, l.y + l.h - 8, 1, 4); }
          g.globalAlpha = 1;
          const s = String(ln.line), a0 = Math.max(a, x0()), b0 = Math.min(b, x1());
          if (g.measureText(s).width + 4 <= b0 - a0) { g.fillStyle = C.fg; g.fillText(s, (a0 + b0) / 2, mid - 1); }
        }
      } else if (l.id === "parts") {
        g.font = font(10.5, 650); g.textAlign = "left";
        parts().forEach((p, k) => {
          if (!visible(p.from, p.to)) return;
          const a = tx(p.from), b = Math.max(tx(p.to), a + 2), on = p.id === S.sel;
          g.fillStyle = on ? C["accent-soft"] : C.hover; g.fillRect(a, l.y + 3, b - a, l.h - 6);
          g.strokeStyle = on ? C.accent : C.line; g.lineWidth = on ? 1.5 : 1; g.strokeRect(a + 0.5, l.y + 3.5, b - a - 1, l.h - 7);
          if (on && b - a > 14) {                               // grips on the selected part's edges
            g.fillStyle = C.accent;
            for (const x of [a + 2, b - 5]) g.fillRect(x, mid - 6, 3, 12);
          }
          const label = String(k + 1) + (p.echo ? " ↺" : "");
          const a0 = Math.max(a, x0()) + (on ? 6 : 0);
          if (g.measureText(label).width + 6 <= Math.min(b, x1()) - a0) { g.fillStyle = C.fg; g.fillText(label, a0 + 4, mid); }
        });
      }
      g.restore();
    }
  }

  // the part of the song on the part lane: its bar's ends in pixels of the view, and whether two tabs fit inside it
  function barOf(span) {
    const xa = tx(span[0]), xb = tx(span[1]);
    return { xa, xb, wide: xb - xa >= 2 * TAB_W + 8 };
  }
  // the part: its bar with a handle at each end and its times on the part lane, the lanes under it dimmed outside it
  function drawPart(g, C) {
    const l = lane("part"), d = S.drag, top = lane("ruler").y;
    const span = d && PART_KINDS.includes(d.kind) && d.moved ? d.span : S.clip;
    const { xa, xb, wide } = barOf(span), H = S.height;
    g.save();
    g.beginPath(); g.rect(x0(), 0, x1() - x0(), H); g.clip();
    g.fillStyle = C.card; g.globalAlpha = 0.62;
    if (xa > x0()) g.fillRect(x0(), top, xa - x0(), H - top);
    if (xb < x1()) g.fillRect(xb, top, x1() - xb, H - top);
    g.globalAlpha = 1;
    const y = l.y + 4, hh = l.h - 8;
    g.fillStyle = C["accent-soft"]; g.fillRect(xa, y, xb - xa, hh);
    g.strokeStyle = C.accent; g.lineWidth = 1.5; g.strokeRect(xa + 0.75, y + 0.75, Math.max(0, xb - xa - 1.5), hh - 1.5);
    const tabs = wide ? [xa, xb - TAB_W] : [xa - TAB_W / 2, xb - TAB_W / 2];
    for (const x of tabs) {
      g.fillStyle = C.accent;
      g.beginPath(); if (g.roundRect) g.roundRect(x, y, TAB_W, hh, 3); else g.rect(x, y, TAB_W, hh); g.fill();
      g.fillStyle = C.card;
      g.fillRect(x + TAB_W / 2 - 2, y + hh / 2 - 6, 1, 12); g.fillRect(x + TAB_W / 2 + 1, y + hh / 2 - 6, 1, 12);
    }
    g.font = `650 10.5px ${css("--font") || "sans-serif"}`; g.textBaseline = "middle"; g.textAlign = "center";
    const text = `${mmss(span[0])} – ${mmss(span[1])} · ${(span[1] - span[0]).toFixed(1)} s`;
    const a0 = Math.max(xa + (wide ? TAB_W : 0), x0()), b0 = Math.min(xb - (wide ? TAB_W : 0), x1());
    if (g.measureText(text).width + 10 <= b0 - a0) { g.fillStyle = C.fg; g.fillText(text, (a0 + b0) / 2, l.y + l.h / 2); }
    g.restore();
  }

  function draw() {
    S.raf = 0;
    if (!S.B || !S.B.song || S.gone) return;
    const w = cv.clientWidth;
    if (!w) return;
    if (w !== S.w || !S.layer) { S.w = w; S.layer = null; }
    const d = S.drag;
    if (d && d.moved && PANNING_KINDS.includes(d.kind) && S.px !== null) {   // a drag held at an edge pans the view
      const span = S.view[1] - S.view[0], step = span * 0.02;
      const dv = S.px > x1() - PAN_PX ? Math.min(step, S.dur - S.view[1]) : S.px < x0() + PAN_PX ? -Math.min(step, S.view[0]) : 0;
      if (dv) { S.view = [S.view[0] + dv, S.view[1] + dv]; S.layer = null; dragTo(S.px); }
    }
    const dpr = devicePixelRatio || 1;
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(S.height * dpr)) {
      cv.width = Math.round(w * dpr); cv.height = Math.round(S.height * dpr); cv.style.height = S.height + "px";
    }
    if (!S.layer) { S.C = colours(); paintLayer(); }
    const g = cv.getContext("2d"), C = S.C;
    g.setTransform(1, 0, 0, 1, 0, 0);
    g.clearRect(0, 0, cv.width, cv.height);
    g.drawImage(S.layer, 0, 0);
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    drawPart(g, C);
    const top = lane("ruler").y;
    if (d && d.moved && ["newpart", "lines", "resize"].includes(d.kind)) {
      const [a, b] = d.span, l = lane(d.kind === "lines" ? "lines" : "parts");
      const xa = clamp(tx(a), x0(), x1()), xb = clamp(tx(b), x0(), x1());
      g.fillStyle = C["accent-soft"]; g.fillRect(xa, l.y, xb - xa, l.h);
      g.fillStyle = C.accent; g.fillRect(xa, l.y, xb - xa, 2);
    }
    if (S.hover !== null && !S.drag) { g.fillStyle = C.faint; g.fillRect(Math.round(S.hover), top + ROW.ruler, 1, S.height - top - ROW.ruler); }
    const t = now(), x = tx(t);
    if (S.src && t >= (S.stopAt ?? S.dur) - 1e-3) { S.at = S.stopAt ?? S.dur; stopSrc(); S.stopAt = null; refresh(); }
    if (x >= x0() - 1 && x <= x1() + 1) {
      g.fillStyle = C.accent;
      g.fillRect(x - 0.75, top, 1.5, S.height - top);
      g.beginPath(); g.moveTo(x - 5, top); g.lineTo(x + 5, top); g.lineTo(x, top + 7); g.fill();
    }
    clock.textContent = timeText(t);
    if (S.src || S.drag) S.raf = requestAnimationFrame(draw);
  }
  function invalidate(relayer = false) {
    if (relayer) S.layer = null;
    if (!S.raf) S.raf = requestAnimationFrame(draw);
  }
  function timeText(t) {
    let s = `${mmss(t)} / ${mmss(S.dur)}`;
    if (T()) {
      const d = T().downbeats.filter((b) => b <= t + 1e-6);
      if (d.length && t <= T().end) s += ` · bar ${d.length}.${T().beats.filter((b) => b >= d[d.length - 1] - 1e-6 && b <= t + 1e-6).length || 1}`;
    }
    return s;
  }

  // ---- what is under the pointer
  function lineAt(t) {
    if (!T()) return null;
    return T().lines.find((ln) => t >= ln.start - 0.05 && t <= ln.end + 0.05) || null;
  }
  function partAt(t) { return parts().find((p) => t >= p.from && t <= p.to) || null; }
  function edgeAt(x) {                               // the edge of a part near x (the selected part's first): {part, side}
    const near = [];
    for (const p of parts()) {
      const da = Math.abs(tx(p.from) - x), db = Math.abs(tx(p.to) - x);
      if (da <= GRAB_PX - 2) near.push({ part: p, side: "from", d: da - (p.id === S.sel ? 4 : 0) });
      if (db <= GRAB_PX - 2) near.push({ part: p, side: "to", d: db - (p.id === S.sel ? 4 : 0) });
    }
    near.sort((a, b) => a.d - b.d);
    return near[0] || null;
  }
  // the part's handle at x on the part lane (0: start, 1: end) or null: each one takes its tab and GRAB_PX past the
  // bar's end, where a hand reaching for an edge lands; on a bar too narrow for two tabs the nearer end wins
  function handleAt(x) {
    const { xa, xb, wide } = barOf(S.clip), inner = wide ? TAB_W + 4 : (xb - xa) / 2;
    const atA = x >= xa - GRAB_PX && x <= xa + inner, atB = x <= xb + GRAB_PX && x >= xb - inner;
    if (atA && atB) return Math.abs(x - xa) <= Math.abs(x - xb) ? 0 : 1;
    return atA ? 0 : atB ? 1 : null;
  }
  // `t` pulled to the nearest line edge, downbeat, edge of the part of the song or of another part within SNAP_PX
  function snap(t, except = null) {
    const c = [S.clip[0], S.clip[1]];
    if (T()) { for (const ln of T().lines) c.push(ln.start, ln.end); c.push(...T().downbeats); }
    for (const p of parts()) if (p.id !== except) c.push(p.from, p.to);
    let best = t, bd = SNAP_PX;
    for (const v of c) { const dd = Math.abs(tx(v) - tx(t)); if (dd < bd) { bd = dd; best = v; } }
    return best;
  }
  function describeWord(t) {
    const ln = lineAt(t);
    if (!ln) return null;
    const k = ln.words.findIndex(([s, e]) => t >= s - 0.02 && t <= e + 0.02);
    return k < 0 ? { line: ln.line, word: null, ln } : { line: ln.line, word: k + 1, ln, at: ln.words[k] };
  }
  function cursorAt(x, y) {
    const l = laneAt(y);
    if (!l) return "default";
    if (l.id === "part") {
      if (handleAt(x) !== null) return "ew-resize";
      const { xa, xb } = barOf(S.clip);
      return x > xa && x < xb ? "grab" : "crosshair";
    }
    if (l.id === "parts") return edgeAt(x) ? "ew-resize" : partAt(xt(x)) ? "pointer" : "crosshair";
    if (l.id === "lines") return lineAt(xt(x)) ? "pointer" : "crosshair";
    return "crosshair";
  }

  // ---- input
  cv.addEventListener("pointerdown", (e) => {
    if (!S.B || !S.B.song || e.button > 0) return;
    cv.setPointerCapture(e.pointerId);
    songBox.focus({ preventScroll: true });
    const [x, y] = local(e), l = laneAt(y);
    if (!l) return;
    S.px = x;
    const t = clamp(xt(x), 0, S.dur);
    if (l.id === "part") {
      const side = handleAt(x), { xa, xb } = barOf(S.clip);
      if (side !== null) S.drag = { kind: "edge", x, side, span: [...S.clip], moved: false };
      else if (x > xa && x < xb) { S.drag = { kind: "move", x, t0: t, span: [...S.clip], moved: false }; cv.style.cursor = "grabbing"; }
      else S.drag = { kind: "clip", x, a: snap(t), span: [t, t], moved: false };
    } else if (l.id === "lines") S.drag = { kind: "lines", x, a: t, span: [t, t], moved: false, shift: e.shiftKey };
    else if (l.id === "parts") {
      const edge = edgeAt(x);
      if (edge) S.drag = { kind: "resize", x, edge, span: [edge.part.from, edge.part.to], moved: false };
      else if (partAt(t)) S.drag = { kind: "pick", x, part: partAt(t), moved: false };
      else S.drag = { kind: "newpart", x, a: snap(t), span: [t, t], moved: false };
    } else {
      S.drag = { kind: "scrub", x, moved: false, wasPlaying: !!S.src };
      pause();
      S.at = t;
    }
    invalidate();
  });
  // follow the pointer at x with the drag going on
  function dragTo(x) {
    const d = S.drag;
    if (Math.abs(x - d.x) > (d.kind === "clip" ? 8 : 3)) d.moved = true;   // a new part of the song takes a real drag
    const t = clamp(xt(x), 0, S.dur);
    if (d.kind === "clip") { const b = snap(t); d.span = [Math.min(d.a, b), Math.max(d.a, b)]; }
    else if (d.kind === "edge") { const s = [...S.clip]; s[d.side] = snap(t); d.span = [Math.min(...s), Math.max(...s)]; }
    else if (d.kind === "move") { const len = S.clip[1] - S.clip[0], a = clamp(S.clip[0] + t - d.t0, 0, S.dur - len); d.span = [a, a + len]; }
    else if (d.kind === "lines") {
      const ls = linesIn(Math.min(d.a, t) - 1e-3, Math.max(d.a, t) + 1e-3).concat([lineAt(d.a), lineAt(t)].filter(Boolean));
      d.span = ls.length ? [Math.min(...ls.map((ln) => ln.start)), Math.max(...ls.map((ln) => ln.end))] : [Math.min(d.a, t), Math.max(d.a, t)];
    } else if (d.kind === "newpart") { const b = snap(t); d.span = [Math.min(d.a, b), Math.max(d.a, b)]; }
    else if (d.kind === "resize") {
      const p = d.edge.part, v = snap(t, p.id);
      d.span = d.edge.side === "from" ? [Math.min(v, p.to - MIN_PART), p.to] : [p.from, Math.max(v, p.from + MIN_PART)];
    } else if (d.kind === "scrub") S.at = t;
  }
  cv.addEventListener("pointermove", (e) => {
    const [x, y] = local(e);
    S.px = x;
    if (!S.drag) {
      const l = laneAt(y);
      S.hover = l && l.id !== "part" && x >= x0() && x <= x1() ? x : null;
      cv.style.cursor = cursorAt(x, y);
      invalidate();
      return;
    }
    dragTo(x);
  });
  cv.addEventListener("pointerleave", () => { S.hover = null; invalidate(); });
  cv.addEventListener("pointerup", (e) => {
    const d = S.drag;
    if (!d) return;
    S.drag = null;
    const [x, y] = local(e), t = clamp(xt(x), 0, S.dur);
    cv.style.cursor = cursorAt(x, y);
    if (PART_KINDS.includes(d.kind)) {
      const [a, b] = d.span;
      if (d.moved && b - a >= MIN_CLIP) {
        if (d.kind === "edge") {                       // a handle let go past the view: pan (never zoom) to show it
          const kept = S.clip[1 - d.side], end = Math.abs(a - kept) < 1e-9 ? b : a, w = S.view[1] - S.view[0], m = w * 0.04;
          const shift = end > S.view[1] - m ? end - (S.view[1] - m) : end < S.view[0] + m ? end - (S.view[0] + m) : 0;
          if (shift) { const s = clamp(S.view[0] + shift, 0, S.dur - w); S.view = [s, s + w]; }
        }
        setClip(a, b);
      } else if (d.moved) readout.textContent = `A part of the song is at least ${MIN_CLIP} s long.`;
      else seek(t);                                                           // a click: listen there
    } else if (d.kind === "lines") {
      const w = describeWord(t);
      if (!d.moved) {
        const ln = lineAt(t);
        if (ln) {
          pickLine(ln, d.shift);
          S.word = w && w.word ? w : null;
          readout.textContent = w && w.word ? `line ${w.line}, word ${w.word} · ${mmss(w.at[0])}` : `line ${ln.line} · ${mmss(ln.start)}–${mmss(ln.end)}`;
          showWordBtn();
          seek(w && w.at ? w.at[0] : ln.start);
        } else seek(t);
      } else if (d.span[1] - d.span[0] >= MIN_PART) addPart(d.span[0], d.span[1]);
    } else if (d.kind === "newpart") {
      if (d.moved && d.span[1] - d.span[0] >= MIN_PART) addPart(d.span[0], d.span[1]);
      else seek(t);
    } else if (d.kind === "resize") {
      if (d.moved) {
        const p = d.edge.part, span = free(Math.max(d.span[0], 0), Math.min(d.span[1], S.dur), p.id);
        if (span) { p.from = +span[0].toFixed(3); p.to = +span[1].toFixed(3); sortParts(); changed(true); select(p.id); }
      } else { const p = partAt(t); if (p) select(p.id, true); }
    } else if (d.kind === "pick") {
      select(d.part.id, true);
      seek(d.part.from);
    } else if (d.kind === "scrub" && d.wasPlaying) startAt(S.at);
    refresh(); invalidate(true);
  });
  cv.addEventListener("wheel", (e) => {
    if (!S.B || !S.B.song) return;
    const span = S.view[1] - S.view[0];
    if (e.ctrlKey) {
      e.preventDefault();
      const [x] = local(e), t = xt(x), wide = clamp(span * Math.exp(e.deltaY * 0.01), 0.5, S.dur);
      const a = clamp(t - (t - S.view[0]) * (wide / span), 0, S.dur - wide);
      S.view = [a, a + wide];
      invalidate(true);
    } else if (Math.abs(e.deltaX) > Math.abs(e.deltaY) || e.shiftKey) {
      e.preventDefault();
      const dd = (e.shiftKey && !e.deltaX ? e.deltaY : e.deltaX) * span / (x1() - x0());
      const a = clamp(S.view[0] + dd, 0, S.dur - span);
      S.view = [a, a + span];
      invalidate(true);
    }
  }, { passive: false });
  function stepBeat(dir) {
    const t = now(), bs = T() ? T().beats : [];
    const next = dir > 0 ? bs.find((b) => b > t + 1e-3) : [...bs].reverse().find((b) => b < t - 1e-3);
    seek(next === undefined ? t + dir : next);
  }
  songBox.addEventListener("keydown", (e) => {
    if (!S.B || e.ctrlKey || e.metaKey || e.altKey || (e.target !== songBox && e.target.closest("button, input, textarea"))) return;
    if (e.key === " ") S.src ? pause() : play();
    else if (e.key === "ArrowRight") stepBeat(1);
    else if (e.key === "ArrowLeft") stepBeat(-1);
    else if (e.key === "f") showPart();
    else if (e.key === "Home") seek(S.clip[0]);
    else return;
    e.preventDefault();
    e.stopPropagation();
  });
  playBtn.addEventListener("click", () => (S.src ? pause() : play()));
  songBtn.addEventListener("click", showSong);
  partBtn.addEventListener("click", () => { showPart(); if (!S.src) seek(S.clip[0]); });
  const ro = new ResizeObserver(() => { if (cv.clientWidth !== S.w) invalidate(true); });
  ro.observe(cv);
  const scheme = matchMedia("(prefers-color-scheme: dark)");
  const relayer = () => invalidate(true);
  scheme.addEventListener("change", relayer);

  // ---- (line, word) into the note of the selected part
  function showWordBtn() {
    const p = S.sel && partOf(S.sel);
    wordBtn.hidden = !(S.word && p);
    if (!wordBtn.hidden) wordBtn.textContent = `Put (${S.word.line}, ${S.word.word}) in part ${numberOf(p.id)}'s note`;
  }
  function putWord() {
    const p = S.sel && partOf(S.sel), c = p && S.cards.get(p.id);
    if (!c || !S.word) return;
    const ref = `(${S.word.line}, ${S.word.word})`, ta = c.note;
    const at = S.lastNote && S.lastNote.id === p.id ? S.lastNote : { start: ta.value.length, end: ta.value.length };
    const before = ta.value.slice(0, at.start), after = ta.value.slice(at.end);
    const pre = before && !/\s$/.test(before) ? " " : "", post = after && !/^\s/.test(after) ? " " : "";
    ta.value = before + pre + ref + post + after;
    p.note = ta.value;
    const caret = (before + pre + ref + post).length;
    S.lastNote = { id: p.id, start: caret, end: caret };
    ta.focus({ preventScroll: true });
    ta.setSelectionRange(caret, caret);
    ta.dispatchEvent(new Event("grow"));
    changed();
  }

  // ---- the part of the song: the project's clip (the view stays where it is)
  async function setClip(a, b, force = false, undoing = false) {
    a = +clamp(a, 0, S.dur).toFixed(3); b = +clamp(b, 0, S.dur).toFixed(3);
    const before = [...S.clip];
    try {
      const got = await api("PUT", "clip", { query: { path: root }, body: { start: a, end: b, force } });
      S.clip = [got.clip.start, got.clip.end];
      S.B.clip = got.clip;
      renderPartLine(); renderWords(); renderCards(); invalidate(true);
      if (!S.src) seek(S.clip[0]);
      const text = `The part is ${mmss(S.clip[0])}–${mmss(S.clip[1])} now.`;
      if (undoing) { readout.textContent = text; undoBtn.hidden = true; }
      else offerUndo(text, () => setClip(before[0], before[1], force, true));
    } catch (e) {
      if (!force && / on its clip: /.test(e.message)) {
        clear(partLine).append(h("span", { class: "bad", text: e.message + "." }),
          h("button", { class: "btn small", text: "Move it anyway", onclick: () => setClip(a, b, true) }),
          h("button", { class: "btn small", text: "Keep it", onclick: () => renderPartLine() }));
      } else readout.textContent = "The part did not change: " + e.message;
      invalidate(true);
    }
  }
  function renderPartLine() {
    const [a, b] = S.clip, whole = a < 0.01 && b > S.dur - 0.05;
    clear(partLine).append(h("span", { class: "lbl", text: "The part" }),
      h("span", { text: whole ? `the whole song · ${mmss(S.dur)}` : `${mmss(a)} – ${mmss(b)} · ${(b - a).toFixed(1)} s of ${mmss(S.dur)}` }),
      whole ? null : h("button", { class: "link", text: "use the whole song", title: "Make the whole song the part (Undo takes it back)", onclick: () => setClip(0, S.dur) }));
  }

  // ---- lyrics, language, timing the words
  function renderLyrics(editing = false) {
    clear(lyricsRow);
    const n = S.B.lyrics;
    if (n && !editing) {
      lyricsRow.append(h("span", { class: "lbl", text: "Lyrics" }), h("span", { text: `${n} line${n === 1 ? "" : "s"} kept` }),
        h("span", { class: "grow" }),
        h("button", { class: "btn small", text: "Replace", onclick: () => renderLyrics(true) }),
        h("button", { class: "btn small", text: "Remove", onclick: () => keepLyrics("") }));
      return;
    }
    const ta = h("textarea", { class: "notes lyrics", rows: 4, spellcheck: false,
      placeholder: "Paste the lyrics of the part here, one sung line per line. They stay in this project: mk never shows them again, it numbers the lines." });
    lyricsRow.append(h("div", { class: "grow col" }, h("span", { class: "lbl", text: "Lyrics" }), ta,
      h("div", { class: "row-end" },
        n ? h("button", { class: "btn small", text: "Cancel", onclick: () => renderLyrics() }) : null,
        h("button", { class: "btn small", text: "Keep them", onclick: () => keepLyrics(ta.value) }))));
  }
  async function keepLyrics(text) {
    try {
      const got = await api("PUT", "lyrics", { query: { path: root }, body: { text } });
      S.B.lyrics = got.lines || null;
      renderLyrics(); renderWords();
      toast(got.lines ? `${got.lines} lines kept` : "Lyrics removed");
    } catch (e) { toast("Not kept: " + e.message, "bad"); }
  }
  function renderLang() {
    clear(langRow).append(h("span", { class: "lbl", text: "Sung in" }), dropdown({
      title: "The language of the words: mk times English words; for another language it finds the bars only",
      options: [["en", "English: the words are timed"], ["other", "another language: the bars only"]], value: S.doc.language,
      onChange: (v) => { S.doc.language = v; saveSoon(); renderWords(); } }));
  }
  function renderWords() {
    clear(wordsRow);
    const W = S.B.words || {}, tl = T();
    const btn = h("button", { class: "btn", text: tl ? "Time the words again" : "Time the words", disabled: !!W.running || !S.B.song, onclick: () => timeWords() });
    const what = h("span", { class: "status grow" });
    if (W.running) {
      what.append(h("span", { class: "spinner" }), ` Timing the words: ${STAGES[W.stage] || "starting"}… ${mmss(W.seconds || 0).replace(/\.\d$/, "")}`);
    } else if (W.exit) {
      what.className += " bad"; what.textContent = "The words could not be timed: " + (W.error || "");
      what.title = W.error || "";
    } else if (tl) {
      const stale = Math.abs(tl.start - S.clip[0]) > 0.01 || Math.abs(tl.end - S.clip[1]) > 0.01;
      what.textContent = stale ? "The part changed since the words were timed: time them again."
        : `${tl.lines.length} lines, ${tl.downbeats.length} bars` + (tl.bpm ? `, ${tl.bpm.toFixed(0)} bpm` : "") + (tl.words_source ? ` (${tl.words_source})` : "");
      if (stale) what.className += " warn";
    } else {
      what.textContent = S.doc.language === "en" ? (S.B.lyrics ? "Times each word of the lyrics: a minute or two for a short part, ten for a whole song."
        : "Without lyrics mk hears the words itself; pasting them gives better lines.") : "Finds the beats and bars.";
    }
    wordsRow.append(btn, what);
  }
  async function timeWords() {
    try {
      await save();
      S.B.words = await api("POST", "words", { query: { path: root } });
      followWords();
    } catch (e) { toast(e.message, "bad"); }
  }
  function followWords() {
    clearTimeout(S.poll);
    renderWords();
    if (!S.B.words.running || S.gone) return;
    S.poll = setTimeout(async () => {
      try { S.B.words = await api("GET", "words", { query: { path: root } }); } catch (e) { S.B.words = { running: false, exit: 1, error: e.message }; }
      if (S.gone) return;
      if (S.B.words.running) followWords();
      else {
        if (!S.B.words.exit) {
          const fresh = await api("GET", "brief", { query: { path: root } });
          Object.assign(S.B, { timeline: fresh.timeline, clip: fresh.clip, lyrics: fresh.lyrics, words: fresh.words });
          layout();
          toast("The words are timed");
        }
        renderWords(); renderCards(); invalidate(true);
      }
    }, 1500);
  }

  // ------------------------------------------------------------------------------------------------- pictures
  async function upload(files, add) {
    for (const f of files) {
      if (!f.type.startsWith("image/")) continue;
      try {
        const got = await api("POST", "picture", { query: { path: root }, body: { data: await readFile(f) } });
        S.B.pictures[got.path] = got.url;
        add(got.path);
      } catch (e) { toast(`${f.name}: ` + e.message, "bad"); }
    }
  }
  // a strip of thumbnails with a remove button each, an add button, and drops and pastes onto `zone`
  function pictureStrip(list, zone, onChange) {
    const box = h("div", { class: "brief-pics" });
    const input = h("input", { type: "file", accept: "image/*", multiple: true, hidden: true });
    const add = (rel) => { if (!list.includes(rel)) list.push(rel); paint(); onChange(); };
    input.addEventListener("change", () => { upload([...input.files], add); input.value = ""; });
    function paint() {
      clear(box).append(...list.map((rel) => h("div", { class: "brief-pic" },
        h("img", { src: S.B.pictures[rel] || "", alt: "", draggable: false }),
        h("button", { class: "x", title: "Remove", text: "×", onclick: () => { list.splice(list.indexOf(rel), 1); paint(); onChange(); } }))),
      h("button", { class: "btn small add", text: "+ pictures", title: "Add pictures (or drop them here, or paste)", onclick: () => input.click() }), input);
    }
    zone.addEventListener("dragover", (e) => { if ([...e.dataTransfer.types].includes("Files")) { e.preventDefault(); zone.classList.add("drop"); } });
    zone.addEventListener("dragleave", () => zone.classList.remove("drop"));
    zone.addEventListener("drop", (e) => { e.preventDefault(); zone.classList.remove("drop"); upload([...e.dataTransfer.files], add); });
    zone.addEventListener("paste", (e) => {
      const files = [...(e.clipboardData ? e.clipboardData.files : [])];
      if (files.length) { e.preventDefault(); upload(files, add); }
    });
    paint();
    return box;
  }
  const grow = (ta) => { const f = () => { ta.style.height = "auto"; ta.style.height = ta.scrollHeight + 2 + "px"; }; ta.addEventListener("input", f); ta.addEventListener("grow", f); requestAnimationFrame(f); return ta; };

  // ------------------------------------------------------------------------------------------------- part cards
  function renderCards() {
    S.cards.clear();
    clear(cardsBox);
    if (!parts().length) {
      cardsBox.append(h("p", { class: "muted empty", text: T() && T().lines.length
        ? "Click a line on the song to write what you want there; drag across lines, or along the parts lane, for a section."
        : "Drag along the parts lane to write about any stretch of the song; once the words are timed, click a line." }));
      return;
    }
    parts().forEach((p, k) => {
      const note = grow(h("textarea", { class: "notes", rows: 2, spellcheck: true, value: p.note,
        placeholder: "What do you want here? As loose or as exact as you like: a feeling, a move, a shot, a colour. Refer to a word as (line, word)." }));
      note.addEventListener("input", () => { p.note = note.value; S.lastNote = { id: p.id, start: note.selectionStart, end: note.selectionEnd }; saveSoon(); });
      const keepCaret = () => { S.lastNote = { id: p.id, start: note.selectionStart, end: note.selectionEnd }; };
      note.addEventListener("keyup", keepCaret);
      note.addEventListener("click", keepCaret);
      note.addEventListener("focus", () => { if (S.sel !== p.id) select(p.id); });
      const where = h("span", { class: "pc-where", text: whereText(p) });
      const earlier = parts().slice(0, k);
      const echo = earlier.length ? h("div", { class: "pc-echo" }, h("span", { class: "lbl", text: "Comes back from" }),
        dropdown({ title: "A part that comes back (a chorus): the same shots, or bigger", value: p.echo ? p.echo.of : "",
          options: [["", "— (nothing before)"], ...earlier.map((q, j) => [q.id, `part ${j + 1} · ${whereText(q)}`])],
          onChange: (v) => { p.echo = v ? { of: v, how: p.echo ? p.echo.how : "same" } : null; changed(true); } }),
        p.echo ? h("div", { class: "seg" }, ["same", "bigger"].map((how) => h("button", { class: p.echo.how === how ? "on" : "", text: how === "same" ? "the same" : "bigger",
          onclick: () => { p.echo.how = how; changed(true); } }))) : null) : null;
      const el_ = h("section", { class: "part-card" + (p.id === S.sel ? " on" : "") + (inClip(p) ? "" : " outside"), "data-id": p.id },
        h("div", { class: "pc-head" }, h("span", { class: "pc-num", text: String(k + 1) }), where, h("span", { class: "grow" }),
          h("button", { class: "btn small icon", text: "▶", title: "Play this part", onclick: () => playSpan(p.from, p.to) }),
          h("button", { class: "btn small icon", text: "×", title: "Delete this part (Undo brings it back)", onclick: () => removePart(p.id) })),
        inClip(p) ? null : h("div", { class: "pc-warn", text: "Outside the part of the song: widen the part to take it in, or delete it." }),
        note);
      el_.append(pictureStrip(p.pictures, el_, () => changed()), echo || "");
      el_.addEventListener("pointerdown", (e) => { if (!e.target.closest("button, textarea, .dropdown") && S.sel !== p.id) select(p.id); });
      cardsBox.append(el_);
      S.cards.set(p.id, { el: el_, note, where });
    });
    showWordBtn();
  }

  // ------------------------------------------------------------------------------------------------- the video card
  function videoCard() {
    const v = S.doc.video;
    const card = h("section", { class: "brief-card" }, h("h3", { text: "The video as a whole" }),
      h("p", { class: "muted", text: "Whatever you already know. The agent asks about the rest." }));
    for (const [key, label, hint, rows] of FIELDS) {
      const field = rows > 1 ? grow(h("textarea", { class: "notes", rows, value: v[key], placeholder: hint }))
        : h("input", { class: "field", value: v[key], placeholder: hint });
      field.addEventListener("input", () => { v[key] = field.value; saveSoon(); });
      card.append(h("label", { class: "brief-field" }, h("span", { class: "lbl", text: label }), field));
    }
    const chips = h("div", { class: "chips" }, S.B.sizes.map((s) => {
      const b = h("button", { class: "btn small" + (v.sizes.includes(s) ? " on" : ""), text: SIZE_LABEL[s] || s, onclick: () => {
        v.sizes = v.sizes.includes(s) ? v.sizes.filter((x) => x !== s) : S.B.sizes.filter((x) => x === s || v.sizes.includes(x));
        b.classList.toggle("on", v.sizes.includes(s));
        saveSoon();
      } });
      return b;
    }));
    card.append(h("div", { class: "brief-field" }, h("span", { class: "lbl", text: "Sizes" }), chips),
      h("div", { class: "brief-field" }, h("span", { class: "lbl", text: "Pictures" }), pictureStrip(v.pictures, card, () => saveSoon())));
    return card;
  }

  // ------------------------------------------------------------------------------------------------- loading
  el.ready = (async () => {
    try {
      S.B = await api("GET", "brief", { query: { path: root } });
    } catch (e) {
      clear(el).append(h("div", { class: "error-card" }, h("strong", { text: "This brief cannot be shown" }), h("pre", { text: e.message })));
      return;
    }
    const b = S.B.brief;
    S.doc = { language: b.language, video: { ...b.video, sizes: [...b.video.sizes], pictures: [...b.video.pictures] },
      parts: b.parts.map((p) => ({ ...p, pictures: [...p.pictures], echo: p.echo ? { ...p.echo } : null })) };
    sendBtn.textContent = b.sent ? "Send again" : "Send to the agent";
    say(b.sent ? `Sent ${new Date(b.sent).toLocaleString([], { dateStyle: "medium", timeStyle: "short" })}` + (b.saved > b.sent ? " · changed since" : "")
      : "Saved as you write: Send when it says what you want");
    const intro = h("p", { class: "intro", text: "Tell the agent what you want from this video: choose the part of the song, then write what you want for each line or group of lines, as loose or as exact as you like. Send it, and the treatment and the agent's questions come back in Reviews." });
    clear(el).append(intro);
    if (S.B.song) {
      S.dur = S.B.song.seconds;
      S.clip = [S.B.clip.start, S.B.clip.end];
      S.at = S.clip[0];
      layout();
      el.append(h("section", { class: "brief-card" }, h("h3", { text: "The song" }), partLine, songBox, lyricsRow, langRow, wordsRow),
        h("section", { class: "brief-card" }, h("h3", { text: "Part by part" }), cardsBox));
      renderPartLine(); renderLyrics(); renderLang(); renderWords(); renderCards(); showPart(); refresh();
      if (S.B.words && S.B.words.running) followWords();
      if (S.B.song.problem) readout.textContent = "The song: " + S.B.song.problem;
      decode(S.B.song.url).then((buf) => {
        S.buf = buf;
        let peak = 1e-6;
        for (let ch = 0; ch < buf.numberOfChannels; ch++) { const d = buf.getChannelData(ch); for (let i = 0; i < d.length; i += 7) peak = Math.max(peak, Math.abs(d[i])); }
        S.norm = peak;
        refresh(); invalidate(true);
      }).catch((e) => { readout.textContent = "The song cannot be played: " + e.message; });
    } else {
      el.append(h("div", { class: "brief-card" }, h("h3", { text: "The song" }),
        h("p", { class: "muted", text: "This project has no song ([audio] file in mk.toml). `mk brief new DIR --song FILE` starts a video from one." })));
    }
    el.append(videoCard());
  })();

  el.dispose = () => {
    S.gone = true;
    pause();
    clearTimeout(S.poll);
    clearTimeout(S.undo);
    if (S.timer) save().catch(() => {});
    ro.disconnect();
    scheme.removeEventListener("change", relayer);
    release(player);
    dock.hidden = true;
    clear(dock);
  };
  return el;
}
