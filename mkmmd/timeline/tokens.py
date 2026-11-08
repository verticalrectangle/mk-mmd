"""Word onsets from Whisper's tokens (numpy only): the fallback when CTC forced alignment of a known lyric picks the
wrong place, e.g. an earlier occurrence of a repeated line (docs/design.md: Timeline, `--align tokens`).

1. `match`: Whisper's tokens (text, start, end) against the known words, a semi-global Needleman-Wunsch on text
   similarity: tokens before the first and after the last word cost nothing, so the occurrence of the lines that
   matches them all in order wins over a partial repeat, and a token sung where the vocal is silent scores less.
2. `candidates`: where a word may start: the vocal's syllable onsets (spectral flux peaks) where it is sung, and the
   moments the voice starts again after a stop.
3. `place`: one candidate per word by a monotone dynamic programme: near its token's start and the previous token's
   end (Whisper's ends sit on the next sung onset; its starts drift, most at a phrase start), strong onsets preferred,
   at least `min_gap` apart, and no word of a line after a pause of `max_pause` or more since the one before. A word
   no token was matched to falls where the onsets and its neighbours put it.
`word_lines` runs the three on a vocal stem and returns lines as `words.word_timeline` does. Never prints a word."""
import difflib
import re

import numpy as np

from . import signal as S

SUNG_DB = -24.0                 # dB under the vocal's peak a sung frame reaches (a delay's echo stays below)
RESTART_Z = 5.0                 # strength of "the voice starts again": a new syllable for sure
TOKEN_Z = 1.0                   # strength of a Whisper word boundary that no onset backs


def norm(w):
    return re.sub(r"[^a-z0-9']", "", w.lower())


def match(tokens, words, gap=0.4, sung=None):
    """{word index: (token index, similarity)} for the words a token was aligned to. `tokens` [(text, start, end)],
    `words` [text]; `sung(start, end)` -> share of the token's span that is sung (optional: a token in silence
    scores up to 0.3 less)."""
    nT, nW = len(tokens), len(words)
    if not nT or not nW:
        return {}
    tw = [norm(t[0]) for t in tokens]
    ww = [norm(w) for w in words]
    sim = np.array([[difflib.SequenceMatcher(None, a, b).ratio() if a and b else 0.0 for b in ww] for a in tw])
    bonus = np.array([0.3 * (sung(t[1], t[2]) - 1.0) if sung else 0.0 for t in tokens])
    D = np.zeros((nT + 1, nW + 1))
    D[0, 1:] = -gap * np.arange(1, nW + 1)                  # a word before any token: a gap
    # D[1:, 0] = 0: tokens before the first word are free (semi-global)
    for i in range(1, nT + 1):
        for j in range(1, nW + 1):
            D[i, j] = max(D[i - 1, j - 1] + sim[i - 1, j - 1] - 0.5 + bonus[i - 1],
                          D[i - 1, j] - gap, D[i, j - 1] - gap)
    i, out = int(np.argmax(D[:, nW])), {}                   # tokens after the last word are free
    j = nW
    while i > 0 and j > 0:
        if np.isclose(D[i, j], D[i - 1, j - 1] + sim[i - 1, j - 1] - 0.5 + bonus[i - 1]):
            out[j - 1] = (i - 1, float(sim[i - 1, j - 1]))
            i, j = i - 1, j - 1
        elif np.isclose(D[i, j], D[i - 1, j] - gap):
            i -= 1
        else:
            j -= 1
    return out


class Evidence:
    """The vocal on a grid of `hop` s from `t0`: `rel` dB under its peak, `voiced` (pitched and loud enough),
    `onsets` [(t, z)] its syllable onsets (flux peaks, z their strength over the median), `midi` its pitch (optional).
    `sung`: voiced within SUNG_DB of the peak, less the echoes of a delay when `midi` is given (`drop_echoes`)."""

    def __init__(self, t0, hop, rel, voiced, onsets, midi=None):
        n = min(len(rel), len(voiced))
        self.t0, self.hop = float(t0), float(hop)
        self.rel, self.voiced = np.asarray(rel[:n], float), np.asarray(voiced[:n], bool)
        self.sung = self.voiced & (self.rel > SUNG_DB)
        if midi is not None:
            self.sung = drop_echoes(self.sung, self.rel, np.asarray(midi[:n], float), hop)
        self.onsets = [(float(a), float(z)) for a, z in onsets]

    def index(self, t):
        return int(np.clip(round((t - self.t0) / self.hop), 0, len(self.sung) - 1))

    def time(self, i):
        return self.t0 + i * self.hop

    def sung_share(self, a, b):
        i, j = self.index(a), self.index(b)
        return float(self.sung[i:max(j, i + 1)].mean())

    def pauses(self, min_s):
        """[(start, end)] of the stretches of at least min_s s that are not sung."""
        out, s = [], None
        for i, v in enumerate(self.sung):
            if not v and s is None:
                s = i
            if (v or i == len(self.sung) - 1) and s is not None:
                if (i - s) * self.hop >= min_s:
                    out.append((self.time(s), self.time(i)))
                s = None
        return out


def drop_echoes(sung, rel, midi, hop, min_gap=0.15, semis=0.6, quieter_db=6.0):
    """`sung` without the runs that come back after a gap of at least `min_gap` s on the pitch of the previous run's
    last second (within `semis`: a held note sags as it dies, its delayed copy repeats the body) and peak at least
    `quieter_db` under it: a delay repeating the last note, not a new word."""
    sung = sung.copy()
    runs, s = [], None
    for i, v in enumerate(sung):
        if v and s is None:
            s = i
        if (not v or i == len(sung) - 1) and s is not None:
            runs.append((s, i if not v else i + 1))
            s = None
    tail = max(1, int(round(1.0 / hop)))
    prev = None
    for a, b in runs:
        if prev is not None and (a - prev[1]) * hop >= min_gap:
            p_pitch = np.median(midi[max(prev[0], prev[1] - tail):prev[1]])
            if abs(np.median(midi[a:b]) - p_pitch) < semis and rel[a:b].max() < rel[prev[0]:prev[1]].max() - quieter_db:
                sung[a:b] = False
                continue                                     # the echo is not the run the next one is compared with
        prev = (a, b)
    return sung


def evidence(voc, sr=S.SR, t0=0.0, z_min=1.4):
    """Evidence from a mono vocal stem whose sample 0 is at `t0` s."""
    hop = S.HOP
    rel = S.rms_db(voc, sr, hop)
    rel = rel - rel.max()
    voiced = S.voicing(voc, rel, sr, hop)
    n, hp = 2048, int(hop * sr)                             # pitch: the autocorrelation peak S.voicing reads
    fr = S.frames(voc, n, hp) * np.hanning(n)
    ac = np.fft.irfft(np.abs(np.fft.rfft(fr, n=2 * n, axis=1)) ** 2, axis=1)[:, :n]
    lo, hi = int(sr / 1000.0), int(sr / 90.0)
    midi = 69 + 12 * np.log2(sr / (np.argmax(ac[:, lo:hi], axis=1) + lo) / 440.0)
    n, hp = 1024, int(0.005 * sr)
    fr = S.frames(voc, n, hp) * np.hanning(n)
    spec = np.abs(np.fft.rfft(fr, axis=1))
    f = np.fft.rfftfreq(n, 1 / sr)
    lg = np.log1p(100 * spec[:, (f >= 150) & (f <= 5000)])
    nov = np.maximum(0, np.diff(lg, axis=0, prepend=lg[:1])).sum(1)
    w = np.hanning(7)
    nov = np.convolve(nov, w / w.sum(), "same")
    med = np.median(nov)
    mad = np.median(np.abs(nov - med)) + 1e-9
    z = (nov - med) / mad
    ons = [(t0 + i * hp / sr, float(z[i])) for i in range(6, len(nov) - 6)          # hp / sr: the hop as sampled
           if z[i] >= z_min and nov[i] == nov[i - 6:i + 7].max()]
    return Evidence(t0, hop, rel, voiced, ons, midi)


def candidates(ev, t_from=-np.inf, t_to=np.inf):
    """[(t, z)] sorted: the onsets that are sung (within 60 ms) and every restart of the voice after >= 50 ms."""
    out = []
    for a, z in ev.onsets:
        i = ev.index(a)
        if ev.sung[i:i + 7].any() and t_from <= a <= t_to:
            out.append((a, min(z, 8.0)))
    s = ev.sung
    for i in range(5, len(s)):
        if s[i] and not s[i - 5:i].any() and t_from <= ev.time(i) <= t_to:
            out.append((ev.time(i), RESTART_Z))
    return sorted(out)


def place(lines, tokens, matched, cands, ev, min_gap=0.12, max_pause=0.35, pause_cost=2.0):
    """One onset per word (flat, in order) from `cands`: see the module docstring. Falls back to a 0.05 s grid of
    sung frames when the candidates cannot hold the words."""
    words = [(li, wi) for li, ln in enumerate(lines) for wi, _ in enumerate(ln)]
    pauses = ev.pauses(max_pause)
    p_end = np.array([b for _, b in pauses])
    p_start = np.array([a for a, _ in pauses])

    def paused(a, b):                                       # a pause lies wholly between a and b
        k = np.searchsorted(p_start, a)
        return k < len(p_start) and p_end[k] <= b + 0.06                # an onset may lead the sung frames by 60 ms

    def run(cs):
        x = np.array([c[0] for c in cs])
        zb = -0.04 * np.array([c[1] for c in cs])
        W, C = len(words), len(cs)
        cost = np.zeros((W, C))
        for k, (li, wi) in enumerate(words):
            c = zb.copy()
            if k in matched:
                tok, s = matched[k]
                c += (0.15 if wi == 0 else 0.5 if s >= 0.8 else 0.2) * np.abs(x - tokens[tok][1])
            if wi > 0 and k - 1 in matched:
                tok, s = matched[k - 1]
                c += (1.0 if s >= 0.8 else 0.4) * np.abs(x - tokens[tok][2])
            cost[k] = c
        acc = np.full((W, C), np.inf)
        back = np.zeros((W, C), int)
        acc[0] = cost[0]
        for k in range(1, W):
            same_line = words[k][0] == words[k - 1][0]
            for j in range(C):
                lim = np.searchsorted(x, x[j] - min_gap + 1e-9)           # predecessors at least min_gap earlier
                if lim == 0:
                    continue
                prev = acc[k - 1, :lim]
                if same_line and len(pauses):
                    prev = prev + np.array([pause_cost if paused(x[i], x[j]) else 0.0 for i in range(lim)])
                i = int(np.argmin(prev))
                if np.isfinite(prev[i]):
                    acc[k, j] = prev[i] + cost[k, j]
                    back[k, j] = i
        j = int(np.argmin(acc[-1]))
        if not np.isfinite(acc[-1, j]):
            return None
        pick = [0] * W
        for k in range(W - 1, -1, -1):
            pick[k] = j
            j = back[k, j]
        return [float(x[p]) for p in pick]

    on = run(cands) if len(cands) >= len(words) else None
    if on is None:
        grid = [(ev.time(i), 0.0) for i in range(0, len(ev.sung), max(1, int(round(0.05 / ev.hop)))) if ev.sung[i]]
        on = run(sorted(cands + grid))
    if on is None:
        raise ValueError("the vocal has fewer sung moments than there are words")
    return on


def voiced_end(ev, x, limit, gap=0.12):
    """End of the singing from x on: the last sung hop before a stretch of more than `gap` s without one, at most
    `limit`."""
    g = max(1, int(round(gap / ev.hop)))
    k = ev.index(x)
    while k < len(ev.sung) - 1 and ev.time(k) < limit and (ev.sung[k] or ev.sung[k + 1:k + 1 + g].any()):
        k += 1
    return float(min(ev.time(k), limit))


def word_lines(lines, tokens, ev, clip_end):
    """(lines, stats) as words.word_timeline gives them: [{"words": [{text, start, end, voiced_end, ctc_start,
    whisper}], start, end}] from the known `lines` (lists of words), Whisper's `tokens` [(text, start, end)] and the
    evidence, all in the same seconds."""
    flat = [w for ln in lines for w in ln]
    matched = match(tokens, flat, sung=ev.sung_share)
    cands = candidates(ev, 0.0, clip_end)
    known = np.array([c[0] for c in cands])
    for _, s, e in tokens:                                  # Whisper's word boundaries where the voice sings: weak
        for x in (s, e):
            sung = ev.sung[ev.index(x):ev.index(x) + 7].any()
            if 0.0 <= x <= clip_end and sung and not (np.abs(known - x) < 0.03).any():
                cands.append((float(x), TOKEN_Z))
    on = place(lines, tokens, matched, sorted(cands), ev)
    out, k = [], 0
    for li, ln in enumerate(lines):
        ws = []
        for wi, text in enumerate(ln):
            nxt = on[k + 1] if k + 1 < len(on) else clip_end
            end = nxt if wi < len(ln) - 1 else voiced_end(ev, on[k], nxt)
            m = matched.get(k)
            span = [round(tokens[m[0]][1], 3), round(tokens[m[0]][2], 3)] if m else None
            ws.append({"text": text, "start": round(on[k], 3), "end": round(end, 3), "voiced_end": round(end, 3),
                       "ctc_start": None, "whisper": span})
            k += 1
        out.append({"words": ws, "start": ws[0]["start"], "end": ws[-1]["end"]} if ws else {"words": []})
    off = [abs(on[k] - tokens[m[0]][1]) for k, m in matched.items() if m[1] >= 0.8]
    stats = {"anchored": sum(1 for m in matched.values() if m[1] >= 0.8),
             "fuzzy": sum(1 for m in matched.values() if m[1] < 0.8), "unanchored": len(flat) - len(matched),
             "anchor_offset_ms_median": round(float(np.median(off)) * 1000) if off else None,
             "t_first_voice": round(on[0], 3) if on else None}
    return out, stats
