"""Vocal stems and word timings (torch, demucs, faster-whisper; CPU).

Words come from a lyrics file when one is given (one line per sung line; blank lines ignored) and otherwise from
Whisper's own transcription, split into lines at its segment boundaries and at pauses. Either way the text is
aligned to the vocal stem with wav2vec2 CTC forced alignment, then each word start is snapped to the nearest strong
vocal onset (or the first sung sound after a breath when the word opens a phrase), and its end follows the voiced
run. Whisper's own word times are kept for comparison. Nothing here prints lyric text."""
import difflib
import hashlib
import os
import re

import numpy as np

from . import signal as S


def _norm(w):
    return re.sub(r"[^a-z']", "", w.lower())


def stems(mix, cache_dir, sr=S.SR):
    """Demucs htdemucs stems {drums, bass, other, vocals}: (2, n) float32, cached by the audio's hash."""
    key = hashlib.sha256(mix.tobytes()).hexdigest()[:20]
    path = os.path.join(cache_dir, f"stems-{key}.npz")
    if os.path.exists(path):
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 2))
    torch.manual_seed(0)
    m = get_model("htdemucs")
    m.eval()
    with torch.no_grad():                                 # shifts=0: deterministic (no random time shift)
        src = apply_model(m, torch.from_numpy(mix)[None], device="cpu", shifts=0, split=True, overlap=0.25,
                          progress=False)[0]
    out = {n: src[i].numpy().astype(np.float32) for i, n in enumerate(m.sources)}
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(path, **out)
    return out


def transcribe(voc16, model="large-v3", language="en", prompt=None, threads=8):
    """Whisper word timestamps on the 16 kHz vocal: [{"start", "end", "words": [(word, start, end)]}]."""
    from faster_whisper import WhisperModel
    try:                                                  # cached models load without touching the network
        wm = WhisperModel(model, device="cpu", compute_type="int8", cpu_threads=threads, local_files_only=True)
    except Exception:                                     # noqa: BLE001 - not cached yet: download it
        wm = WhisperModel(model, device="cpu", compute_type="int8", cpu_threads=threads)
    segs, _ = wm.transcribe(voc16, language=language, word_timestamps=True, beam_size=5, vad_filter=False,
                            condition_on_previous_text=False, initial_prompt=prompt)
    out = []
    for s in segs:
        ws = [(w.word.strip(), float(w.start), float(w.end)) for w in (s.words or []) if _norm(w.word)]
        if ws:
            out.append({"start": float(s.start), "end": float(s.end), "words": ws})
    return out


def lines_from_whisper(segments, pause=0.45):
    """Lines of words from Whisper segments, also split where a pause between words exceeds `pause` s. Whisper often
    merges two sung lines into one segment: resplit_periodic() fixes that after alignment."""
    lines = []
    for seg in segments:
        cur = []
        for w in seg["words"]:
            if cur and w[1] - cur[-1][2] > pause:
                lines.append(cur)
                cur = []
            cur.append(w)
        if cur:
            lines.append(cur)
    return [[w[0] for w in ln] for ln in lines]


def resplit_periodic(lines, long_factor=1.15, snap=0.15, share=0.6):
    """Aligned lines ({"words": [{start, voiced_end, ...}]}): sung lines usually start on a regular grid (every one or
    two bars), while Whisper's segments merge lines or break them in the middle. The grid: period p = the median
    interval between line starts, phase from the start most others agree with (refined on those). It is trusted when
    at least `share` of the starts lie within snap x p of it; then (1) a line longer than long_factor x p is split at the
    words starting within snap x p of each grid start inside it, and (2) a line starting off the grid in the same grid
    slot as the previous line is merged into it. Lines keep at least two words. Without a trusted grid the lines come
    back unchanged. Returns new lines with start / end."""
    lines = [ln for ln in lines if ln["words"]]
    starts = np.array([ln["words"][0]["start"] for ln in lines], float)
    if len(starts) < 3:
        return lines
    p = float(np.median(np.diff(starts)))
    if p <= 0:
        return lines

    def resid(g):
        r = (starts - g) / p
        return np.abs(r - np.round(r)) * p

    g0 = max(starts, key=lambda g: int((resid(g) <= snap * p).sum()))
    on = resid(g0) <= snap * p
    if on.mean() < share:
        return lines
    r = (starts[on] - g0) / p
    g0 += float(np.median((r - np.round(r)) * p))

    def slot(t):                                        # a start a little before its grid line belongs to it
        return int(np.floor((t - g0) / p + snap))

    def off_grid(t):
        r = (t - g0) / p
        return abs(r - round(r)) * p > snap * p

    def end_of(w):
        return w["voiced_end"] if "voiced_end" in w else w["end"]

    pieces = []
    for ln in lines:
        ws = ln["words"]
        if end_of(ws[-1]) - ws[0]["start"] <= long_factor * p:
            pieces.append(ws)
            continue
        cuts = []
        for k in range(slot(ws[0]["start"]) + 1, slot(ws[-1]["start"]) + 1):
            want = g0 + k * p
            j = min(range(1, len(ws)), key=lambda i: abs(ws[i]["start"] - want))
            if abs(ws[j]["start"] - want) <= snap * p and j - (cuts[-1] if cuts else 0) >= 2 and len(ws) - j >= 2:
                cuts.append(j)
        pieces += [ws[a:b] for a, b in zip([0] + cuts, cuts + [len(ws)])]
    merged = []
    for ws in pieces:
        s = ws[0]["start"]
        if merged and off_grid(s) and slot(s) == slot(merged[-1][0]["start"]):
            merged[-1] = merged[-1] + ws
        else:
            merged.append(ws)
    return [{"words": ws, "start": ws[0]["start"], "end": end_of(ws[-1])} for ws in merged]


def lines_from_file(path):
    with open(path, encoding="utf-8") as fh:
        return [ln.split() for ln in (x.strip() for x in fh) if ln]


def ctc_align(voc, sr, words, t0, t1):
    """CTC forced alignment of words on the vocal between t0 and t1 (s): {(word, char): (start, end, score)}."""
    import torch
    import torchaudio
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    labels = bundle.get_labels()
    lab = {c: i for i, c in enumerate(labels)}
    w16 = torchaudio.functional.resample(torch.from_numpy(voc)[None], sr, bundle.sample_rate)
    model = bundle.get_model().eval()
    with torch.inference_mode():
        em, _ = model(w16)
        em = torch.log_softmax(em, dim=-1)
    spf = w16.shape[1] / bundle.sample_rate / em.shape[1]
    tokens, tw, tc = [], [], []
    for wi, w in enumerate(words):
        if wi:
            tokens.append(lab["|"])
            tw.append(-1)
            tc.append(-1)
        for ci, ch in enumerate(w):
            c = ch.upper()
            if c in lab and c not in ("-", "|"):
                tokens.append(lab[c])
                tw.append(wi)
                tc.append(ci)
    f0, f1 = int(max(0.0, t0) / spf), min(em.shape[1], int(t1 / spf))
    ali, scores = torchaudio.functional.forced_align(em[:, f0:f1], torch.tensor([tokens], dtype=torch.int32), blank=0)
    spans = torchaudio.functional.merge_tokens(ali[0], scores[0].exp())
    out = {}
    for sp, wi, ci in zip(spans, tw, tc):
        if wi >= 0:
            out[(wi, ci)] = ((sp.start + f0) * spf, (sp.end + f0) * spf, float(sp.score))
    return out, w16[0].numpy().astype(np.float32)


def word_timeline(voc, sr, lines, clip_end, whisper=None, hop=S.HOP):
    """Lines of {text, start, end, voiced_end, ctc_start, whisper} from the vocal stem (times in s of `voc`)."""
    rel = S.rms_db(voc, sr) - S.rms_db(voc, sr).max()
    voiced = S.voicing(voc, rel, sr)
    m = min(len(voiced), len(rel))
    gap = ~voiced[:m] | (rel[:m] < -30)
    sung = voiced[:m] & (rel[:m] > -25)
    run = np.convolve(voiced.astype(float), np.ones(10), "valid")
    t_first = float(np.argmax(run >= 10) * hop) if (run >= 10).any() else 0.0
    words = [w for ln in lines for w in ln]
    line_of = [li for li, ln in enumerate(lines) for _ in ln]
    wh_of = {}
    if whisper:
        wh = [(_norm(w), s, e) for seg in whisper for (w, s, e) in seg["words"]]
        sm = difflib.SequenceMatcher(None, [w[0] for w in wh], [_norm(w) for w in words], autojunk=False)
        for blk in sm.get_matching_blocks():
            for k in range(blk.size):
                wh_of[blk.b + k] = wh[blk.a + k]
    # the alignment window ends shortly after the last of these words as Whisper heard it: vocals after the chosen
    # lines (the next phrase, a chorus) would otherwise absorb the last word
    t_end = clip_end
    tail = [wh_of[i] for i in range(len(words) - 3, len(words)) if i in wh_of]
    if tail:
        t_end = min(clip_end, tail[-1][2] + 0.8 + (len(words) - 1 - max(i for i in wh_of if i >= len(words) - 3)) * 0.6)
    ctc, _ = ctc_align(voc, sr, words, t_first - 0.15, t_end)
    onsets = S.vocal_onsets(voc, lambda h: S.rms_db(voc, sr, h) - S.rms_db(voc, sr, h).max(), sr)

    def voiced_at(t):
        return bool(voiced[min(len(voiced) - 1, max(0, int(round(t / hop))))])

    def phrase_onset(s_ctc, prev_start):
        i1 = min(int((s_ctc + 0.05) / hop), m - 6)
        i0 = max(int((s_ctc - 0.6) / hop), int((prev_start + 0.05) / hop), 1)
        best, gap_run = None, 0
        for i in range(i0, i1):
            if gap[i]:
                gap_run += 1
                continue
            if gap_run >= 12 and sung[i:i + 6].sum() >= 4:
                best = i * hop
            gap_run = 0
        return best if best is not None and s_ctc - 0.45 <= best <= s_ctc + 0.05 else None

    out = [{"words": []} for _ in lines]
    prev = -1.0
    stats = {"snapped": 0, "phrase": 0, "unaligned": 0}
    for wi, w in enumerate(words):
        letters = [ctc[(wi, ci)] for ci in range(len(w)) if (wi, ci) in ctc]
        if not letters:
            stats["unaligned"] += 1
            s_ctc = e = max(prev + 0.05, 0.0)
        else:
            s_ctc, e = letters[0][0], letters[-1][1]
        cand = [(t, st * np.exp(-((t - s_ctc) / 0.08) ** 2)) for t, st in onsets if s_ctc - 0.15 <= t <= s_ctc + 0.10]
        s = max(cand, key=lambda c: c[1])[0] if cand else s_ctc
        if cand:
            stats["snapped"] += 1
        po = phrase_onset(s_ctc, prev)
        if po is not None:
            s = po
            stats["phrase"] += 1
        s = max(s, prev + 0.04)
        prev = s
        whw = wh_of.get(wi)
        out[line_of[wi]]["words"].append({"text": w, "start": round(s, 3), "end": round(max(e, s), 3),
                                          "ctc_start": round(s_ctc, 3),
                                          "whisper": [round(whw[1], 3), round(whw[2], 3)] if whw else None})
    flat = [w for ln in out for w in ln["words"]]
    for k, w in enumerate(flat):                              # the sung sound lasts while the vocal stays voiced
        nxt = flat[k + 1]["start"] if k + 1 < len(flat) else clip_end
        t, g, last = max(w["start"], w["end"] - 0.05), 0.0, w["start"]
        while t < nxt - hop and g <= 0.04:
            if voiced_at(t):
                last, g = t, 0.0
            else:
                g += hop
            t += hop
        w["voiced_end"] = round(min(max(last + hop, w["start"] + 0.05), nxt), 3)
        w["end"] = w["voiced_end"]
    for ln in out:
        if ln["words"]:
            ln["start"], ln["end"] = ln["words"][0]["start"], ln["words"][-1]["voiced_end"]
    bias = [w["whisper"][0] - w["start"] for w in flat if w["whisper"]]
    stats["whisper_minus_final_ms_median"] = round(float(np.median(bias)) * 1000) if bias else None
    stats["t_first_voice"] = round(t_first, 3)
    return out, rel, stats
