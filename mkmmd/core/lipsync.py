"""Lip sync: timeline words -> espeak-ng IPA phonemes -> the five MMD vowel morphs (semantic a i u e o).

Each word spans [start, voiced_end]; consonants take a short fixed slice, vowels share the rest (the last vowel gets the
held note). The mouth closes for m/b/p and returns to rest between words further apart than `gap_close`. Mouth size
follows the vocal loudness (`vocal_db` per frame, optional). The words come from a local timeline file and are
phonemised at build time; nothing here prints or stores lyric text.

Timeline (docs/design.md: Timeline): {"fps": 30, "lines": [{"words": [{"text", "start", "voiced_end"}, ...]}, ...],
"vocal_db": [...]} with times in clip seconds."""
import functools
import subprocess

VOWELS = ("a", "i", "u", "e", "o")
TIE = "\u200d"
DROP = set("ˈˌ.|‖-")

VOWEL_SHAPES = {
    "i": {"i": 0.75}, "ɪ": {"i": 0.55, "e": 0.15}, "e": {"e": 0.75}, "ɛ": {"e": 0.75},
    "æ": {"a": 0.7, "e": 0.3}, "a": {"a": 0.9}, "ɑ": {"a": 1.0}, "ɐ": {"a": 0.7}, "ʌ": {"a": 0.65},
    "ɒ": {"o": 0.8, "a": 0.2}, "ɔ": {"o": 0.85}, "o": {"o": 0.85}, "u": {"u": 0.85}, "ʊ": {"u": 0.65},
    "ə": {"a": 0.4, "e": 0.15}, "ɚ": {"u": 0.3, "e": 0.25}, "ɜ": {"e": 0.45, "u": 0.2}, "ᵻ": {"i": 0.45},
}
CONS_SHAPES = {
    "m": {}, "b": {}, "p": {},
    "f": {"i": 0.18}, "v": {"i": 0.18}, "w": {"u": 0.7}, "j": {"i": 0.4}, "ɹ": {"u": 0.3}, "r": {"u": 0.3},
    "ʃ": {"u": 0.35, "i": 0.15}, "ʒ": {"u": 0.35, "i": 0.15}, "l": {"i": 0.22, "a": 0.12},
    "h": {"a": 0.3}, "k": {"a": 0.25}, "ɡ": {"a": 0.25}, "ŋ": {"a": 0.25},
}
DEFAULT_CONS = {"i": 0.22, "a": 0.1}


@functools.lru_cache(maxsize=None)
def ipa(word, voice="en-gb"):
    w = "".join(ch for ch in word.lower() if ch.isalpha() or ch == "'")
    if not w:
        return ""
    r = subprocess.run(["espeak-ng", "-q", "--ipa=3", "-v", voice, w], capture_output=True, text=True)
    return r.stdout.strip()


def phonemes(word, voice="en-gb"):
    """IPA string -> tokens; diphthongs (tied with ZWJ) and long marks stay in one token."""
    s = [ch for ch in ipa(word, voice) if ch not in DROP and not ch.isspace()]
    out, i = [], 0
    while i < len(s):
        tok = s[i]
        i += 1
        while i < len(s) and (s[i] in "ːˑ" or s[i - 1] == TIE or s[i] == TIE):
            tok += s[i]
            i += 1
        tok = tok.replace(TIE, "")
        if tok:
            out.append(tok)
    return out


def shapes(tok):
    """Token -> (list of vowel-weight dicts (a diphthong yields two), is_vowel)."""
    base = tok.replace("ː", "").replace("ˑ", "")
    vs = [VOWEL_SHAPES[c] for c in base if c in VOWEL_SHAPES]
    if vs:
        return vs[:2], True
    for c in base:
        if c in CONS_SHAPES:
            return [CONS_SHAPES[c]], False
    return [DEFAULT_CONS], False


def select_words(timeline, lines=None, words=None):
    """Words of the chosen lines (1-based inclusive range [a, b] or list), or the chosen `words`: [[line, word], ...]
    (1-based; a negative word counts from the end of its line, so [2, -1] is line 2's last word), in time order.
    Raises ValueError naming a choice the timeline does not have."""
    lns = timeline["lines"]
    if words is not None:
        if lines is not None:
            raise ValueError("give `lines` or `words`, not both")
        out = []
        for pick in words:
            if not (isinstance(pick, (list, tuple)) and len(pick) == 2 and all(isinstance(x, int) for x in pick)):
                raise ValueError(f"words: {pick!r} is not [line, word]")
            ln, wd = pick
            if not 1 <= ln <= len(lns):
                raise ValueError(f"words: {pick!r}: no line {ln} (the timeline has {len(lns)})")
            ws = lns[ln - 1]["words"]
            if wd == 0 or abs(wd) > len(ws):
                raise ValueError(f"words: {pick!r}: line {ln} has {len(ws)} words")
            out.append(ws[wd - 1 if wd > 0 else wd])
        return sorted({id(w): w for w in out}.values(), key=lambda w: w["start"])
    if lines is None:
        idx = range(len(lns))
    elif isinstance(lines, (list, tuple)) and len(lines) == 2 and all(isinstance(x, int) for x in lines):
        idx = range(lines[0] - 1, lines[1])
    else:
        idx = [i - 1 for i in lines]
    return [w for i in idx for w in lns[i]["words"]]


def keyframes(timeline, lines=None, mouth=0.55, gap_close=0.14, cons_t=0.055, fps=None, voice="en-gb",
              offset=0.0, words=None):
    """vowel -> [(clip_time, value)] for the chosen lines or words (select_words). mouth: peak weight at full voice
    (0.5 mouthing, 0.9 singing out). offset shifts every key (s; a small negative lead reads better on screen)."""
    vdb = timeline.get("vocal_db")
    fps = fps or timeline.get("fps", 30)

    def loud(t):
        if not vdb:
            return 1.0
        k = min(len(vdb) - 1, max(0, int(t * fps)))
        return max(0.35, min(1.0, (vdb[k] + 32.0) / 24.0))

    keys = {m: [] for m in VOWELS}

    def put(t, shape, amp):
        for m in VOWELS:
            keys[m].append((t + offset, shape.get(m, 0.0) * amp))

    prev_end = -1.0
    for w in select_words(timeline, lines, words):
        t0, t1 = w["start"], max(w["voiced_end"], w["start"] + 0.08)
        if t0 - prev_end > gap_close:
            put(prev_end + 0.06 if prev_end > 0 else t0 - 0.12, {}, 0.0)
            put(t0 - 0.05, {}, 0.0)
        toks = [shapes(p) for p in phonemes(w["text"], voice)] or [([{"a": 0.5}], True)]
        n_c = sum(1 for _, v in toks if not v)
        n_v = sum(1 for _, v in toks if v) or 1
        span = t1 - t0
        ct = min(cons_t, 0.35 * span / max(n_c, 1))
        vt_total = span - ct * n_c
        vw = [1.0] * n_v
        vw[-1] = 2.2                                    # the held note goes to the last vowel
        vsum = sum(vw)
        t, vi = t0, 0
        for shp, is_v in toks:
            d = vt_total * vw[vi] / vsum if is_v else ct
            amp = mouth * loud(t + d * 0.3)
            if len(shp) == 2:
                put(t + d * 0.25, shp[0], amp)
                put(t + d * 0.75, shp[1], amp * 0.9)
            else:
                put(t + d * (0.35 if is_v else 0.5), shp[0], amp if is_v else amp * 0.8)
            if is_v:
                vi += 1
            t += d
        prev_end = t1
    if prev_end > 0:
        put(prev_end + 0.08, {}, 0.0)
    for m in VOWELS:
        pts = sorted(keys[m])
        dedup = []
        for t, v in pts:
            if dedup and t - dedup[-1][0] < 1.0 / 60:
                dedup[-1] = (t, max(v, dedup[-1][1]))
            else:
                dedup.append((t, v))
        keys[m] = dedup
    return keys
