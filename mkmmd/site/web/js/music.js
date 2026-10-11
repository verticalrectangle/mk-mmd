// The music timeline (docs/design.md: The site): the clip's audio through Web Audio and its lanes on one canvas, the
// beats and bars, the project's hits and the timeline's onsets, the words by (line, word) (never their text), the shots
// of the cut and the effects (transitions, inserts, glitches, freezes, rings, texts), all in clip seconds. Space plays,
// a click or a drag moves the playhead (a click on a hit, a word, a shot or an effect jumps to its start and names it), a
// drag along the ruler loops that span (a click on it clears the loop), Ctrl+wheel or a pinch zooms, a sideways wheel
// pans, a double-click fits; ← → step a beat, l loops, f fits, Home goes back to the start.
import { api, bytes } from "./api.js";
import { h, clamp } from "./ui.js";

const GUTTER = 58, PAD = 8;
const ROW = { ruler: 20, wave: 38, beats: 18, hits: 16, words: 18, shots: 22, fx: 15 };
const FX_KINDS = ["transition", "insert", "glitch", "freeze", "ring", "text"];

let audioCtx = null;
const decoded = new Map();          // url -> Promise<AudioBuffer>
let playing = null;                 // the view whose audio plays: one at a time

// The context that plays is made by the first Play, inside its click or key press, and its resume() is not waited
// for: WebKit holds a context made outside a gesture "interrupted" for good, and settles resume() only once a source
// plays. Decoding needs no context of the page's own (an OfflineAudioContext decodes).
const context = () => audioCtx || (audioCtx = new AudioContext());
const decode = (url) => {
  if (!decoded.has(url)) decoded.set(url, bytes(url).then((b) => new OfflineAudioContext(2, 1, 44100).decodeAudioData(b)));
  return decoded.get(url);
};
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

function colours() {
  const c = {};
  for (const k of ["fg", "muted", "faint", "line", "sunk", "card", "hover", "accent", "accent-soft", "ok", "warn", "bad"]) c[k] = css("--" + k);
  c.lanes = [c.accent, c.warn, c.ok, c.bad, c.muted];
  c.fx = { transition: c.accent, insert: c.warn, glitch: c.bad, freeze: c.ok, ring: c.muted, text: c.faint };
  return c;
}

// Peaks of the clip's audio per column (min, max of both channels averaged), over clip seconds a..b.
function peaks(buf, offset, a, b, cols) {
  const sr = buf.sampleRate, L = buf.getChannelData(0), R = buf.numberOfChannels > 1 ? buf.getChannelData(1) : null;
  const n = L.length, out = new Float32Array(cols * 2);
  for (let i = 0; i < cols; i++) {
    const s0 = clamp(Math.floor((offset + a + (b - a) * i / cols) * sr), 0, n);
    const s1 = clamp(Math.max(Math.floor((offset + a + (b - a) * (i + 1) / cols) * sr), s0 + 1), 0, n);
    const step = Math.max(1, Math.floor((s1 - s0) / 400));
    let lo = 0, hi = 0;
    for (let s = s0; s < s1; s += step) {
      const v = R ? 0.5 * (L[s] + R[s]) : L[s];
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    out[2 * i] = lo; out[2 * i + 1] = hi;
  }
  return out;
}

function niceStep(secPerPx) {
  for (const s of [0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60]) if (s / secPerPx >= 64) return s;
  return 120;
}

// The lanes of a payload, top to bottom, each {id, label, y, h}; effects packed into as few rows as overlap allows.
function layout(M) {
  const L = [{ id: "ruler", h: ROW.ruler, label: "" }, { id: "wave", h: ROW.wave, label: "audio" }];
  if (M.beats.length) L.push({ id: "beats", h: ROW.beats, label: "beats" });
  M.lanes.forEach((lane, k) => L.push({ id: "hits", h: ROW.hits, label: lane.name, lane, k }));
  if (M.words.length) L.push({ id: "words", h: ROW.words, label: "words" });
  if (M.shots.length) L.push({ id: "shots", h: ROW.shots, label: "shots" });
  if ((M.checks || []).length) L.push({ id: "checks", h: ROW.hits, label: "checks" });
  const ends = [];
  for (const e of [...M.effects].sort((a, b) => a.from - b.from)) {
    let r = ends.findIndex((end) => end <= e.from + 1e-6);
    if (r < 0) { r = ends.length; ends.push(0); }
    ends[r] = e.to;
    e.row = r;
  }
  ends.forEach((_, r) => L.push({ id: "fx", h: ROW.fx, label: r ? "" : "effects", row: r }));
  let y = 0;
  for (const l of L) { l.y = y; y += l.h; }
  return { lanes: L, height: y + 2 };
}

export function musicView(root, { from = null, to = null } = {}) {
  const playBtn = h("button", { class: "btn small play", text: "▶", title: "Play (space)" });
  const clock = h("span", { class: "clock", text: "Loading…" });
  const loopBtn = h("button", { class: "btn small", text: "Loop", title: "Loop the span dragged along the ruler (l)" });
  const fitBtn = h("button", { class: "btn small", text: "Fit", title: "Fit (f, or double-click)" });
  const cv = h("canvas", { class: "lanes" });
  const readout = h("div", { class: "music-readout" });
  const el = h("div", { class: "music", tabindex: "0" },
    h("div", { class: "transport" }, playBtn, clock, h("span", { class: "grow" }), loopBtn, fitBtn), cv, readout,
    h("div", { class: "hint-line", text: "space plays · drag to scrub · drag the ruler to loop · Ctrl+wheel zooms" }));
  const S = { M: null, span: [0, 1], view: [0, 1], L: null, height: 0, w: 0, layer: null, C: null, buf: null, norm: 1,
    src: null, t0: 0, at: 0, loop: null, looping: false, drag: null, hover: null, raf: 0 };

  const x0 = () => GUTTER, x1 = () => S.w - PAD;
  const tx = (t) => x0() + (t - S.view[0]) / (S.view[1] - S.view[0]) * (x1() - x0());
  const xt = (x) => S.view[0] + (x - x0()) / (x1() - x0()) * (S.view[1] - S.view[0]);
  const local = (e) => { const r = cv.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
  const laneAt = (y) => S.L && S.L.lanes.find((l) => y >= l.y && y < l.y + l.h);

  // ---------------------------------------------------------------------------------------------------- playing
  const offset = () => (S.M && S.M.audio && S.M.audio.offset) || 0;
  // The playhead follows the audio clock while the context runs, else the page's clock (a browser that holds the sound
  // back still plays the timeline, silently).
  function now() {
    if (!S.src) return S.at;
    const ctx = context();
    let t = S.at + (ctx.state === "running" ? ctx.currentTime - S.t0 : (performance.now() - S.p0) / 1000);
    if (S.looping && S.loop && t >= S.loop[1]) t = S.loop[0] + ((t - S.loop[0]) % (S.loop[1] - S.loop[0]));
    return Math.min(t, S.span[1]);
  }
  function stopSrc() {
    if (!S.src) return;
    const s = S.src;
    S.src = null;
    try { s.stop(); } catch (e) { /* not started */ }
    s.disconnect();
  }
  function startAt(t) {
    stopSrc();
    if (playing && playing !== api_) playing.pause();
    playing = api_;
    const ctx = context(), src = ctx.createBufferSource();
    if (ctx.state !== "running") ctx.resume().catch(() => {});
    src.buffer = S.buf;
    src.connect(ctx.destination);
    if (S.looping && S.loop) {
      src.loop = true; src.loopStart = offset() + S.loop[0]; src.loopEnd = offset() + S.loop[1];
      src.start(0, offset() + t);
    } else src.start(0, offset() + t, Math.max(0, S.span[1] - t));
    src.onended = () => { if (S.src === src) { S.src = null; S.at = S.span[1]; refresh(); invalidate(); } };
    S.src = src; S.t0 = ctx.currentTime; S.p0 = performance.now(); S.at = t;
    refresh(); tick();
    setTimeout(() => {
      if (S.src === src && ctx.state !== "running") readout.textContent = "No sound: this browser keeps the page's audio from starting, so the timeline plays silently.";
    }, 1200);
  }
  function play() {
    if (!S.buf) return;
    let t = now();
    if (S.looping && S.loop && (t < S.loop[0] || t >= S.loop[1])) t = S.loop[0];
    if (!(S.looping && S.loop) && t >= S.span[1] - 0.01) t = S.span[0];
    startAt(t);
  }
  function pause() { if (!S.src) return; S.at = now(); stopSrc(); refresh(); invalidate(); }
  function seek(t) { t = clamp(t, S.span[0], S.span[1]); if (S.src) startAt(t); else { S.at = t; refresh(); invalidate(); } }
  const api_ = { pause };

  // ---------------------------------------------------------------------------------------------------- drawing
  function paintLayer() {
    const M = S.M, C = S.C, dpr = devicePixelRatio || 1, w = S.w, H = S.height;
    const c = S.layer || (S.layer = document.createElement("canvas"));
    c.width = Math.round(w * dpr); c.height = Math.round(H * dpr);
    const g = c.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, H);
    const font = (px, weight = 400) => `${weight} ${px}px ${css("--font") || "sans-serif"}`;
    const [v0, v1] = S.view, visible = (a, b) => b >= v0 && a <= v1;
    // `s` at x (the current alignment), cut with an ellipsis to `room` px; nothing when not even one letter fits
    const label = (s, x, y, room) => {
      let t = s;
      if (g.measureText(t).width > room) {
        while (t.length > 1 && g.measureText(t + "…").width > room) t = t.slice(0, -1);
        t += "…";
        if (g.measureText(t).width > room) return;
      }
      g.fillText(t, x, y);
    };
    g.textBaseline = "middle";
    for (const l of S.L.lanes) {
      g.fillStyle = C.line; g.fillRect(0, l.y + l.h - 0.5, w, 0.5);
      if (l.label) { g.font = font(10.5); g.fillStyle = C.muted; g.textAlign = "right"; label(l.label, GUTTER - 8, l.y + l.h / 2, GUTTER - 12); }
      g.save();
      g.beginPath(); g.rect(x0() - 1, l.y, x1() - x0() + 2, l.h); g.clip();
      const mid = l.y + l.h / 2;
      if (l.id === "ruler") {
        const step = niceStep((v1 - v0) / (x1() - x0()));
        g.font = font(10); g.fillStyle = C.faint; g.strokeStyle = C.faint; g.textAlign = "left";
        for (let t = Math.ceil(v0 / step) * step; t <= v1 + 1e-9; t += step) {
          const x = Math.round(tx(t)) + 0.5;
          g.beginPath(); g.moveTo(x, l.y + l.h - 6); g.lineTo(x, l.y + l.h); g.stroke();
          g.fillText(step < 0.1 ? t.toFixed(2) : step < 1 ? t.toFixed(1) : t.toFixed(0), x + 3, l.y + l.h / 2 - 1);
        }
      } else if (l.id === "wave") {
        if (S.buf) {
          const cols = Math.max(1, Math.round(x1() - x0())), p = peaks(S.buf, offset(), v0, v1, cols), amp = (l.h / 2 - 3) / S.norm;
          g.fillStyle = C.muted;
          for (let i = 0; i < cols; i++) g.fillRect(x0() + i, mid - p[2 * i + 1] * amp, 1, Math.max(1, (p[2 * i + 1] - p[2 * i]) * amp));
        } else {
          g.font = font(10.5); g.fillStyle = C.faint; g.textAlign = "left";
          g.fillText(M.audio && M.audio.url ? "Decoding the audio…" : "No audio", x0() + 6, mid);
        }
      } else if (l.id === "beats") {
        const down = new Set(M.downbeats.map((t) => t.toFixed(3)));
        g.font = font(9.5, 600); g.textAlign = "left";
        for (const t of M.beats) {
          if (!visible(t, t)) continue;
          const x = Math.round(tx(t)) + 0.5, d = down.has(t.toFixed(3));
          g.strokeStyle = d ? C.fg : C.faint; g.lineWidth = d ? 1.5 : 1;
          g.beginPath(); g.moveTo(x, d ? l.y + 2 : mid - 3); g.lineTo(x, d ? l.y + l.h - 2 : mid + 3); g.stroke();
        }
        g.fillStyle = C.fg;
        M.downbeats.forEach((t, k) => { if (visible(t, t)) g.fillText(String(k + 1), tx(t) + 3, l.y + 6); });
      } else if (l.id === "hits") {
        g.fillStyle = C.lanes[l.k % C.lanes.length];
        for (const hit of l.lane.hits) {
          if (!visible(hit.t, hit.t)) continue;
          const r = hit.db === null ? 2.6 : 1.6 + 3 * clamp((hit.db + 30) / 30, 0, 1);
          g.beginPath(); g.arc(tx(hit.t), mid, r, 0, 2 * Math.PI); g.fill();
        }
      } else if (l.id === "words") {
        g.font = font(9.5, 600); g.textAlign = "center";
        for (const wd of M.words) {
          if (!visible(wd.start, wd.end)) continue;
          const a = tx(wd.start), b = Math.max(tx(wd.end), a + 2);
          g.globalAlpha = 0.28; g.fillStyle = wd.line % 2 ? C.ok : C.accent; g.fillRect(a, l.y + 3, b - a, l.h - 6);
          g.globalAlpha = 1;
          const s = `${wd.line}.${wd.word}`, a0 = Math.max(a, x0()), b0 = Math.min(b, x1());
          if (g.measureText(s).width + 4 <= b0 - a0) { g.fillStyle = C.fg; g.fillText(s, (a0 + b0) / 2, mid); }
        }
      } else if (l.id === "shots") {
        g.font = font(10, 600); g.textAlign = "left";
        M.shots.forEach((s, k) => {
          if (!visible(s.from, s.to)) return;
          const a = tx(s.from), b = tx(s.to);
          g.fillStyle = k % 2 ? C.hover : C.sunk; g.fillRect(a, l.y + 2, b - a, l.h - 4);
          g.fillStyle = C.line; g.fillRect(a, l.y + 2, 1, l.h - 4);
          const a0 = Math.max(a, x0());
          g.fillStyle = C.fg; label(s.name, a0 + 4, mid, Math.min(b, x1()) - a0 - 8);
        });
      } else if (l.id === "checks") {
        for (const c of M.checks) {                         // failing checks red at their worst moment, passing ones faint
          if (!visible(c.t, c.t)) continue;
          const x = tx(c.t);
          g.fillStyle = c.ok ? C.faint : C.bad;
          g.beginPath(); g.moveTo(x, l.y + 3); g.lineTo(x + 4.5, l.y + l.h - 3); g.lineTo(x - 4.5, l.y + l.h - 3); g.fill();
        }
      } else if (l.id === "fx") {
        g.font = font(9.5, 600); g.textAlign = "left";
        for (const e of M.effects) {
          if (e.row !== l.row || !visible(e.from, e.to)) continue;
          const a = tx(e.from), b = Math.max(tx(e.to), a + 2);
          g.globalAlpha = 0.35; g.fillStyle = C.fx[e.kind] || C.muted; g.fillRect(a, l.y + 2, b - a, l.h - 4);
          g.globalAlpha = 1;
          const a0 = Math.max(a, x0());
          g.fillStyle = C.fg; label(e.label, a0 + 3, mid, Math.min(b, x1()) - a0 - 6);
        }
      }
      g.restore();
    }
  }

  function draw() {
    S.raf = 0;
    if (!S.M) return;
    const w = cv.clientWidth;
    if (!w) return;
    if (w !== S.w || !S.layer) { S.w = w; S.layer = null; }
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
    const span = S.drag && S.drag.kind === "loop" && S.drag.moved ? [Math.min(S.drag.a, S.drag.b), Math.max(S.drag.a, S.drag.b)] : S.loop;
    if (span) {
      const a = clamp(tx(span[0]), x0(), x1()), b = clamp(tx(span[1]), x0(), x1());
      g.fillStyle = C["accent-soft"]; g.fillRect(a, 0, b - a, S.height);
      g.fillStyle = S.looping || S.drag ? C.accent : C.faint; g.fillRect(a, 0, b - a, 3);
    }
    if (S.hover !== null && !S.drag) { g.fillStyle = C.faint; g.fillRect(Math.round(S.hover), ROW.ruler, 1, S.height - ROW.ruler); }
    const t = now(), x = tx(t);
    if (S.src && !(S.looping && S.loop) && t >= S.span[1] - 1e-3) { S.at = S.span[1]; stopSrc(); refresh(); }   // a silent run ends here
    if (x >= x0() - 1 && x <= x1() + 1) {
      g.fillStyle = C.accent; g.fillRect(x - 0.75, 0, 1.5, S.height);
      g.beginPath(); g.moveTo(x - 5, 0); g.lineTo(x + 5, 0); g.lineTo(x, 7); g.fill();
    }
    clock.textContent = timeText(t);
    if (S.src || S.drag) S.raf = requestAnimationFrame(draw);
  }
  const invalidate = () => { if (!S.raf) S.raf = requestAnimationFrame(draw); };
  const tick = invalidate;
  const relayer = () => { S.layer = null; invalidate(); };

  function timeText(t) {
    const M = S.M;
    let s = `${t.toFixed(2)} / ${M.duration.toFixed(2)} s · f ${M.frame0 + Math.round(t * M.fps)}`;
    const d = M.downbeats.filter((b) => b <= t + 1e-6);
    if (d.length) s += ` · bar ${d.length}.${M.beats.filter((b) => b >= d[d.length - 1] - 1e-6 && b <= t + 1e-6).length || 1}`;
    return s;
  }
  function refresh() {
    playBtn.textContent = S.src ? "❚❚" : "▶";
    playBtn.disabled = !S.buf;
    loopBtn.classList.toggle("on", S.looping && !!S.loop);
    loopBtn.disabled = !S.loop;
  }

  // ---------------------------------------------------------------------------------------------------- picking
  function itemAt(x, y) {
    const l = laneAt(y), M = S.M, t = xt(x), near = (tt) => Math.abs(tx(tt) - x) <= 6;
    if (!l) return null;
    if (l.id === "beats") {
      const k = M.beats.findIndex(near);
      return k < 0 ? null : { t: M.beats[k], text: `beat ${k + 1} · ${M.beats[k].toFixed(3)} s` };
    }
    if (l.id === "hits") {
      let best = null;
      for (const hit of l.lane.hits) if (near(hit.t) && (!best || Math.abs(hit.t - t) < Math.abs(best.t - t))) best = hit;
      return best && { t: best.t, text: `${l.lane.name} · ${best.t.toFixed(3)} s` + (best.db === null ? "" : ` · ${best.db.toFixed(1)} dB`) };
    }
    if (l.id === "words") {
      const wd = M.words.find((w) => t >= w.start && t <= w.end);
      return wd && { t: wd.start, text: `word ${wd.line}.${wd.word} · ${wd.start.toFixed(2)}–${wd.end.toFixed(2)} s` };
    }
    if (l.id === "shots") {
      const s = M.shots.find((s) => t >= s.from && t < s.to);
      return s && { t: s.from, text: `shot ${s.name} · ${s.from.toFixed(3)}–${s.to.toFixed(3)} s · ${Math.round((s.to - s.from) * M.fps)} frames` };
    }
    if (l.id === "checks") {
      let best = null;
      for (const c of M.checks) if (near(c.t) && (!best || Math.abs(c.t - t) < Math.abs(best.t - t))) best = c;
      return best && { t: best.t, text: `check ${best.name} · ${best.ok ? "passes" : "fails"} · worst at ${best.t.toFixed(2)} s` };
    }
    if (l.id === "fx") {
      const e = M.effects.find((e) => e.row === l.row && t >= e.from && t <= e.to);
      return e && { t: e.from, text: `${e.kind}${e.label !== e.kind ? " " + e.label : ""} · ${e.from.toFixed(3)}–${e.to.toFixed(3)} s` };
    }
    return null;
  }

  // ---------------------------------------------------------------------------------------------------- input
  cv.addEventListener("pointerdown", (e) => {
    if (!S.M || e.button > 0) return;
    cv.setPointerCapture(e.pointerId);
    el.focus({ preventScroll: true });
    const [x, y] = local(e), t = clamp(xt(x), S.span[0], S.span[1]);
    if (y < ROW.ruler) S.drag = { kind: "loop", a: t, b: t, x, moved: false };
    else {
      S.drag = { kind: "scrub", x, moved: false, wasPlaying: !!S.src };
      pause();
      S.at = t;
    }
    tick();
  });
  cv.addEventListener("pointermove", (e) => {
    const [x] = local(e);
    if (!S.drag) { S.hover = x >= x0() && x <= x1() ? x : null; invalidate(); return; }
    if (Math.abs(x - S.drag.x) > 3) S.drag.moved = true;
    const t = clamp(xt(x), S.span[0], S.span[1]);
    if (S.drag.kind === "loop") S.drag.b = t;
    else S.at = t;
  });
  cv.addEventListener("pointerleave", () => { S.hover = null; invalidate(); });
  cv.addEventListener("pointerup", (e) => {
    const d = S.drag;
    if (!d) return;
    S.drag = null;
    const [x, y] = local(e);
    if (d.kind === "loop") {
      const a = Math.min(d.a, d.b), b = Math.max(d.a, d.b);
      if (d.moved && b - a >= 0.05) {
        S.loop = [a, b]; S.looping = true;
        readout.textContent = `loop ${a.toFixed(2)}–${b.toFixed(2)} s`;
        if (S.src) startAt(a);
      } else { S.loop = null; S.looping = false; seek(d.a); }
    } else {
      if (!d.moved) {
        const it = itemAt(x, y);
        if (it) { S.at = clamp(it.t, S.span[0], S.span[1]); readout.textContent = it.text; }
      }
      if (d.wasPlaying) startAt(S.at);
    }
    refresh(); invalidate();
  });
  cv.addEventListener("dblclick", () => fit());
  cv.addEventListener("wheel", (e) => {
    if (!S.M) return;
    const span = S.view[1] - S.view[0];
    if (e.ctrlKey) {
      e.preventDefault();
      const [x] = local(e), t = xt(x), wide = clamp(span * Math.exp(e.deltaY * 0.01), 0.25, S.span[1] - S.span[0]);
      const a = clamp(t - (t - S.view[0]) * (wide / span), S.span[0], S.span[1] - wide);
      S.view = [a, a + wide];
      relayer();
    } else if (Math.abs(e.deltaX) > Math.abs(e.deltaY) || e.shiftKey) {
      e.preventDefault();
      const d = (e.shiftKey && !e.deltaX ? e.deltaY : e.deltaX) * span / (x1() - x0());
      const a = clamp(S.view[0] + d, S.span[0], S.span[1] - span);
      S.view = [a, a + span];
      relayer();
    }
  }, { passive: false });
  function fit() { S.view = [...S.span]; relayer(); }
  function stepBeat(dir) {
    const t = now(), bs = S.M.beats;
    const next = dir > 0 ? bs.find((b) => b > t + 1e-3) : [...bs].reverse().find((b) => b < t - 1e-3);
    seek(next === undefined ? (dir > 0 ? S.span[1] : S.span[0]) : next);
  }
  el.addEventListener("keydown", (e) => {
    if (!S.M || e.ctrlKey || e.metaKey || e.altKey || (e.target !== el && e.target.closest("button, input, select, textarea"))) return;
    if (e.key === " ") S.src ? pause() : play();
    else if (e.key === "ArrowRight") stepBeat(1);
    else if (e.key === "ArrowLeft") stepBeat(-1);
    else if (e.key === "l") { if (S.loop) { S.looping = !S.looping; if (S.src) startAt(now()); refresh(); invalidate(); } }
    else if (e.key === "f") fit();
    else if (e.key === "Home") seek(S.span[0]);
    else return;
    e.preventDefault();
    e.stopPropagation();
  });
  playBtn.addEventListener("click", () => (S.src ? pause() : play()));
  loopBtn.addEventListener("click", () => { if (S.loop) { S.looping = !S.looping; if (S.src) startAt(now()); refresh(); invalidate(); } });
  fitBtn.addEventListener("click", fit);
  const ro = new ResizeObserver(() => { if (cv.clientWidth !== S.w) relayer(); });
  ro.observe(cv);
  const scheme = matchMedia("(prefers-color-scheme: dark)");
  scheme.addEventListener("change", relayer);
  el.dispose = () => { pause(); ro.disconnect(); scheme.removeEventListener("change", relayer); if (playing === api_) playing = null; };
  // The clock other views follow (the scene plays to it): the playhead in clip seconds, and the transport.
  el.player = { now: () => now(), playing: () => !!S.src, play, pause, seek, toggle: () => (S.src ? pause() : play()) };

  // ---------------------------------------------------------------------------------------------------- loading
  el.ready = (async () => {
    try {
      const M = await api("GET", "music", { query: { path: root } });
      S.M = M;
      const a = from === null ? 0 : clamp(+from, 0, M.duration), b = to === null ? M.duration : clamp(+to, 0, M.duration);
      S.span = b > a ? [a, b] : [0, M.duration];
      S.view = [...S.span];
      S.at = S.span[0];
      S.L = layout(M);
      S.height = S.L.height;
      const facts = [M.bpm ? `${M.bpm.toFixed(1)} bpm` : null, `${M.beats.length} beats`, `${M.words.length} words`,
        `${M.shots.length} shots`, `${M.effects.length} effects`].filter(Boolean);
      readout.textContent = facts.join(" · ");
      if (M.problems.length) el.append(h("ul", { class: "music-problems" }, M.problems.map((p) => h("li", { text: p }))));
      refresh(); relayer();
      if (M.audio && M.audio.url) {
        try {
          S.buf = await decode(M.audio.url);
          let peak = 1e-6;
          for (let ch = 0; ch < S.buf.numberOfChannels; ch++) { const d = S.buf.getChannelData(ch); for (let i = 0; i < d.length; i += 7) peak = Math.max(peak, Math.abs(d[i])); }
          S.norm = peak;
          refresh(); relayer();
        } catch (e) { readout.textContent = "The audio cannot be played: " + e.message; }
      } else readout.textContent = (M.audio && M.audio.problem) ? "audio: " + M.audio.problem : "This project has no [audio] file.";
    } catch (e) {
      clock.textContent = "";
      readout.textContent = "The music cannot be shown: " + e.message;
      readout.classList.add("bad");
    }
  })();
  return el;
}
