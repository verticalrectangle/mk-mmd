// The review: its questions as cards (pictures, models, music, options, notes), saved as they change, and Send.
import { api } from "./api.js";
import { h, clear, topLayer, toast } from "./ui.js";
import { markdown } from "./md.js";
import { openDrawing, paintMarks, baseWidth, fontSize } from "./draw.js";
import { openViewer } from "./viewer/viewer.js";
import { musicView } from "./music.js";

const CUBE = '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"><path d="M12 2 3 7v10l9 5 9-5V7z"/><path d="M3 7l9 5 9-5M12 12v10"/></svg>';

export async function openReview(path, frame) {
  const R = await api("GET", "review", { query: { path } });
  const S = { answers: {}, current: 0, marks: R.marks || { images: [] }, sent: R.sent, timer: 0, pending: null, keyhelp: null };
  for (const [qid, a] of Object.entries(R.answers || {})) S.answers[qid] = { choice: a.choice ?? null, notes: a.notes || "" };
  const answer = (qid) => (S.answers[qid] ||= { choice: null, notes: "" });
  const withOptions = R.questions.filter((q) => q.options.length);
  const answered = () => withOptions.filter((q) => S.answers[q.id] && S.answers[q.id].choice).length;
  const cards = [], thumbs = [];

  // ---------------------------------------------------------------------------------------------- header and dock
  const pill = h("span", { class: "pill" });
  clear(frame.top).append(h("div", { class: "title", text: R.title, title: R.title }), pill);
  const status = h("span", { class: "status grow" });
  const sendBtn = h("button", { class: "btn primary", onclick: () => send() });
  const helpBtn = h("button", { class: "btn icon", title: "Keys", text: "?", onclick: () => toggleHelp() });
  frame.dock.hidden = false;
  clear(frame.dock).append(status, helpBtn, sendBtn);
  const refreshCounts = () => {
    const n = answered(), total = withOptions.length;
    pill.textContent = `${n} / ${total}`;
    pill.classList.toggle("ok", total > 0 && n === total);
    document.title = `${R.title} (${n}/${total})`;
    sendBtn.textContent = S.sent ? "Send again" : "Send";
  };

  // ---------------------------------------------------------------------------------------------- saving and Send
  const save = async () => {
    clearTimeout(S.timer);
    S.timer = 0;
    try {
      await api("PUT", "answers", { query: { path: R.path }, body: { answers: S.answers } });
    } catch (e) {
      status.textContent = "Cannot save the answers: " + e.message;
      status.className = "status grow bad";
      throw e;
    }
  };
  const saveSoon = () => { clearTimeout(S.timer); S.timer = setTimeout(() => save().catch(() => {}), 350); };
  const send = async () => {
    sendBtn.disabled = true;
    try {
      await save();
      const got = await api("POST", "send", { query: { path: R.path } });
      S.sent = got.sent;
      const at = new Date(got.sent);
      status.textContent = `Sent at ${at.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}: the agent has it`;
      status.className = "status grow ok";
      toast("Sent to the agent");
    } catch (e) {
      status.textContent = "Not sent: " + e.message;
      status.className = "status grow bad";
    } finally {
      sendBtn.disabled = false;
      refreshCounts();
    }
  };

  // ---------------------------------------------------------------------------------------------- marks on pictures
  const marksOf = (img) => (S.marks.images.find((x) => x.path === img.path) || { marks: [] }).marks;
  const paintThumbs = () => thumbs.forEach((t) => t());
  const draw = (img) => openDrawing({
    image: img,
    marks: marksOf(img),
    onDone: async (marks) => {
      const images = S.marks.images.filter((x) => x.path !== img.path);
      if (marks.length) images.push({ path: img.path, w: img.w, h: img.h, marks });
      S.marks = await api("PUT", "marks", { query: { path: R.path }, body: { images } });
      paintThumbs();
    },
  });

  function picture(img) {
    const im = h("img", { src: img.url, alt: img.name, draggable: false });
    const cv = h("canvas", { class: "marks" });
    const count = h("span", { class: "badge mark" });
    const paint = () => {
      const marks = marksOf(img);
      count.textContent = marks.length === 1 ? "1 mark" : `${marks.length} marks`;
      count.hidden = !marks.length;
      const w = im.clientWidth, hh = im.clientHeight;
      if (!w || !hh) return;
      const dpr = devicePixelRatio || 1;
      cv.width = Math.round(w * dpr);
      cv.height = Math.round(hh * dpr);
      const ctx = cv.getContext("2d");
      ctx.setTransform(dpr * w / img.w, 0, 0, dpr * hh / img.h, 0, 0);
      paintMarks(ctx, marks, Math.max(baseWidth(img.w, img.h), 2.2 * img.w / w), fontSize(img.h));
    };
    im.addEventListener("load", paint);
    new ResizeObserver(paint).observe(im);
    thumbs.push(paint);
    return h("div", { class: "pic" },
      h("div", { class: "pic-frame", title: "Draw on it", onclick: () => draw(img) }, im, cv,
        h("button", { class: "draw", title: "Draw on it", text: "✎", onclick: (e) => { e.stopPropagation(); draw(img); } })),
      h("div", { class: "pic-bar" }, h("span", { class: "name", text: img.name, title: img.path }),
        img.sheet ? h("span", { class: "badge accent", text: "lab sheet · mm", title: "Lines drawn on it come back in millimetres on the model" }) : null,
        count));
  }
  const pictures = (images) => images.length ? h("div", { class: "pics" + (images.length > 1 ? " many" : "") }, images.map(picture)) : null;

  // ---------------------------------------------------------------------------------------------- models and views
  const views = R.views || [];
  const viewBoxes = new Map();
  const showViews = (qid) => {
    const box = viewBoxes.get(qid);
    if (!box) return;
    clear(box);
    const mine = views.filter((v) => v.question === qid);
    if (mine.length) box.append(h("div", { class: "pic-bar muted", text: "Views kept from 3D" }), pictures(mine));
  };
  function models(list, family, q) {
    if (!list.length) return null;
    return h("div", { class: "models" }, list.map((m) => {
      const b = h("button", { class: "btn", title: "Turn it in 3D", onclick: () => openViewer({ review: R.path, model: m, family, question: q.id,
        onMark: (view) => { views.push(view); showViews(q.id); draw(view); } }) });
      b.innerHTML = CUBE;
      b.append(m.label);
      return b;
    }));
  }

  // ---------------------------------------------------------------------------------------------- questions
  function card(q, i) {
    const a = answer(q.id);
    const family = [...q.models, ...q.options.flatMap((o) => o.models)];
    const chosen = h("span", { class: "q-chosen" });
    const opts = q.options.map((o, k) => {
      const radio = h("input", { type: "radio", name: "q-" + q.id, value: o.id, checked: a.choice === o.id, tabindex: "-1" });
      radio.addEventListener("change", () => choose(i, o.id));
      return h("div", { class: "opt", "data-id": o.id },
        h("label", { class: "opt-row" }, radio, h("span", { class: "opt-key", text: k < 9 ? String(k + 1) : "" }),
          h("span", { class: "opt-label", text: o.label }), h("span", { class: "opt-id", text: o.id }),
          q.recommended === o.id ? h("span", { class: "badge accent", text: "recommended" }) : null),
        o.text ? h("div", { class: "opt-text" }, markdown(o.text)) : null,
        pictures(o.images), models(o.models, family, q));
    });
    const notes = h("textarea", { class: "notes", rows: 1, placeholder: "Notes for the agent", value: a.notes, spellcheck: true });
    const grow = () => { notes.style.height = "auto"; notes.style.height = notes.scrollHeight + 2 + "px"; };
    notes.addEventListener("input", () => { answer(q.id).notes = notes.value; grow(); saveSoon(); });
    notes.addEventListener("focus", () => setCurrent(i, false));
    notes.addEventListener("keydown", (e) => { if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); notes.blur(); } });
    requestAnimationFrame(grow);
    const viewBox = h("div", { class: "views" });
    viewBoxes.set(q.id, viewBox);
    const el = h("section", { class: "q", id: "q-" + q.id, onpointerdown: () => setCurrent(i, false) },
      h("div", { class: "q-head" }, h("span", { class: "q-num", text: String(i + 1) }), h("span", { class: "q-ask", text: q.ask }), chosen),
      q.text ? h("div", { class: "q-text" }, markdown(q.text)) : null,
      q.music ? musicView(q.music.project, { from: q.music.from ?? null, to: q.music.to ?? null }) : null,
      pictures(q.images), models(q.models, family, q), viewBox,
      q.options.length ? h("div", { class: "opts" + (q.options.length <= 3 ? " few" : "") }, opts) : null,
      notes);
    requestAnimationFrame(() => showViews(q.id));
    const refresh = () => {
      const c = answer(q.id).choice;
      const o = q.options.find((x) => x.id === c);
      chosen.textContent = o ? `${o.id}: ${o.label}` : "";
      el.classList.toggle("done", !!o);
      for (const optEl of opts) optEl.classList.toggle("on", optEl.dataset.id === c);
    };
    refresh();
    return { el, refresh, notes, q };
  }
  const choose = (i, oid) => {
    const c = cards[i];
    answer(c.q.id).choice = oid;
    for (const r of c.el.querySelectorAll("input[type=radio]")) r.checked = r.value === oid;
    c.refresh();
    setCurrent(i, false);
    refreshCounts();
    save().catch(() => {});
  };
  const setCurrent = (i, scroll = true) => {
    S.current = Math.max(0, Math.min(cards.length - 1, i));
    cards.forEach((c, k) => c.el.classList.toggle("current", k === S.current));
    if (scroll) cards[S.current].el.scrollIntoView({ block: "nearest", behavior: "smooth" });
  };

  // ---------------------------------------------------------------------------------------------- page
  clear(frame.main);
  if (R.text) {
    const intro = h("div", { class: "intro" }, markdown(R.text));
    frame.main.append(intro);
    requestAnimationFrame(() => {
      if (intro.scrollHeight > 90) {
        intro.classList.add("clamped");
        const more = h("button", { class: "more", text: "more", onclick: () => {
          const open = intro.classList.toggle("clamped");
          more.textContent = open ? "more" : "less";
        } });
        intro.append(more);
      }
    });
  }
  R.questions.forEach((q, i) => { const c = card(q, i); cards.push(c); frame.main.append(c.el); });
  const first = R.questions.findIndex((q) => q.options.length && !(S.answers[q.id] && S.answers[q.id].choice));
  setCurrent(first >= 0 ? first : 0, false);
  refreshCounts();
  if (S.sent) { status.textContent = "Sent before: Send again sends the answers as they are now"; status.className = "status grow muted"; }

  // ---------------------------------------------------------------------------------------------- keys
  const KEYS = [["j  k", "next / previous question"], ["1 – 9", "choose an option"], ["n", "notes (Esc leaves)"],
    ["Esc", "closes drawing and 3D"], ["?", "these keys"]];
  function toggleHelp() {
    if (S.keyhelp) { S.keyhelp.remove(); S.keyhelp = null; return; }
    S.keyhelp = h("div", { class: "keyhelp", onclick: () => toggleHelp() }, h("table", {}, KEYS.map(([k, v]) => h("tr", {}, h("td", { text: k }), h("td", { text: v })))));
    document.body.append(S.keyhelp);
  }
  document.addEventListener("keydown", (e) => {
    if (topLayer() || e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.target instanceof HTMLTextAreaElement || e.target instanceof HTMLInputElement && e.target.type !== "radio") return;
    const c = cards[S.current];
    if (e.key === "j" || e.key === "ArrowDown") setCurrent(S.current + 1);
    else if (e.key === "k" || e.key === "ArrowUp") setCurrent(S.current - 1);
    else if (e.key === "n" && c) { e.preventDefault(); c.notes.focus(); c.notes.scrollIntoView({ block: "nearest" }); }
    else if (/^[1-9]$/.test(e.key) && c && c.q.options[+e.key - 1]) choose(S.current, c.q.options[+e.key - 1].id);
    else if (e.key === "?") toggleHelp();
    else if (e.key === "Escape" && S.keyhelp) toggleHelp();
    else return;
    e.preventDefault();
  });
  addEventListener("beforeunload", () => { if (S.timer) save().catch(() => {}); });
}
