"""Hit lanes (docs/design.md: Timeline, hit lanes): the moments a project keys things on, from the cached Demucs stems of
a timeline's song span, in clip seconds, as `[audio] hits` takes them (a JSON object of named lists).

Three kinds of lane:
  onsets  when a band of a stem is struck: the spectral-flux detector of note onsets (signal.band_onsets) with a shorter
          separation, so 16th-note rolls stay apart; each hit {t, db}, `db` the band's level just after the hit against
          the lane's loudest (0 = the loudest, -24 quieter). A hit must lift the band's level by MIN_JUMP dB over the
          moment before it (a fluctuation in the decay of a hit is none), hits more than `range_db` under the loudest
          are dropped, and between the onset lanes of one stem a hit that another lane has at the same moment (within
          SAME_S) far more prominently (by DOMINANCE dB, each against its own lane's loudest) is that lane's, leaking
          into this band: a kick's thump in the snare's band. Hits of two lanes at once (a kick and a snare together)
          both stay.
  gaps    when a stem (or the whole mix) falls silent: its level more than `drop_db` under its own running median (over
          `win_s`) for at least `min_s`; each {t, dur, db}: when the silence starts, how long it lasts, and how deep it is
          (dB under the median, negative).
  notes   onsets of a melodic stem with the pitch each note starts on: {t, db, midi} (`midi` the fundamental, a fraction
          of a semitone; the harmonic product spectrum of the 70 ms after the onset). A pitch is an estimate: an octave off
          on a weak fundamental happens.

Default lanes (LANES): kick, snare, hat (drums), stop (the mix), bass_out (bass), notes (other)."""
import numpy as np

from . import signal as S

ONSETS = {"kick": ("drums", 35.0, 150.0), "snare": ("drums", 180.0, 1000.0), "hat": ("drums", 7000.0, 16000.0)}
GAPS = {"stop": ("mix", 10.0, 0.08), "bass_out": ("bass", 14.0, 0.12)}
NOTES = {"notes": ("other", 200.0, 4000.0)}
STEMS = ("drums", "bass", "other", "vocals", "mix")
MIN_SEP = 0.06                 # s: hits closer than this are one (16ths stay apart up to 250 bpm)
RANGE_DB = 24.0                # dB: hits quieter than the lane's loudest by more than this are leakage
LEVEL_N = 1024                 # samples of the window the band's level is read with (23 ms: a 16th roll stays apart)
LEVEL_S = 0.05                 # s after a hit over which its level is read ...
BEFORE_S = (0.035, 0.012)      # ... and the stretch before it (s before the hit) it must rise from ...
MIN_JUMP = 4.0                 # ... by at least this (dB)
SAME_S = 0.015                 # s: hits of two lanes this close are one moment
DOMINANCE = 12.0               # dB: a hit this much more prominent in a sibling lane is that lane's
GAP_WIN_S = 3.0                # s: the running median a gap is measured against
PITCH_S = (0.01, 0.08)         # s after a note onset whose spectrum gives its pitch
PITCH_RANGE = (55.0, 2000.0)   # Hz: the fundamentals a note may have


class HitsError(ValueError):
    """A lane that cannot be computed: an unknown stem, a band upside down."""


def _clip(t, pre, duration):
    """Window seconds -> clip seconds, and a mask of those inside [0, duration]."""
    c = np.asarray(t, float) - pre
    return c, (c >= 0.0) & (c <= duration)


def onset_lane(x, lo, hi, pre=0.0, duration=None, sr=S.SR, min_sep=MIN_SEP, range_db=RANGE_DB):
    """[{t, db}] of a band's hits in a mono signal `x` (see the module docstring); t in clip seconds, the clip starting
    `pre` s into `x` and lasting `duration` (None: to the end)."""
    if not 0 < lo < hi:
        raise HitsError(f"band {lo:g}-{hi:g} Hz: the floor must be positive and below the ceiling")
    duration = len(x) / sr - pre if duration is None else duration
    t, _ = S.band_onsets(x, sr, lo, hi, min_sep=min_sep, start_frac=1.01)    # a hit is no strum: no walking back
    if not len(t):
        return []
    lvl = S.band_db(x, lo, hi, sr, n=LEVEL_N)
    k = int(round(LEVEL_S / S.HOP))
    idx = np.clip(np.round(t / S.HOP).astype(int), 0, len(lvl) - 1)
    peak = np.array([lvl[i:i + k + 1].max() for i in idx])
    b0, b1 = int(round(BEFORE_S[0] / S.HOP)), int(round(BEFORE_S[1] / S.HOP))
    before = np.array([lvl[max(i - b0, 0):max(i - b1, 1)].mean() for i in idx])
    c, inside = _clip(t, pre, duration)
    inside &= peak - before >= MIN_JUMP                         # a rise, not a wobble in the decay of the last hit
    if not inside.any():
        return []
    db = peak - peak[inside].max()
    keep = inside & (db >= -range_db)
    return [{"t": round(float(a), 3), "db": round(float(b), 1)} for a, b in zip(c[keep], db[keep])]


def separate(lanes_, stems):
    """The onset lanes `lanes_` {name: [{t, db}]} with each stem's leakage between its lanes removed: a hit that a
    sibling lane (same stem in `stems` {name: stem}) has within SAME_S and DOMINANCE dB more prominently is dropped."""
    out = {}
    for name, lane in lanes_.items():
        sibs = [lanes_[o] for o in lanes_ if o != name and stems[o] == stems[name]]
        out[name] = [h for h in lane
                     if not any(abs(g["t"] - h["t"]) <= SAME_S and g["db"] - h["db"] >= DOMINANCE
                                for sib in sibs for g in sib)]
    return out


def gap_lane(x, drop_db, min_s, pre=0.0, duration=None, sr=S.SR, win_s=GAP_WIN_S):
    """[{t, dur, db}] where a mono signal `x` falls more than `drop_db` under its running median for at least `min_s`
    (see the module docstring); t in clip seconds. A gap that starts before the clip or runs past it is cut to it."""
    if drop_db <= 0 or min_s <= 0:
        raise HitsError(f"gap: drop {drop_db:g} dB and length {min_s:g} s must be positive")
    duration = len(x) / sr - pre if duration is None else duration
    lvl = S.rms_db(x, sr)
    w = max(3, int(round(win_s / S.HOP)) | 1)
    pad = np.pad(lvl, w // 2, mode="edge")
    med = np.median(np.lib.stride_tricks.sliding_window_view(pad, w), axis=1)
    low = lvl < med - drop_db
    out = []
    i, n = 0, len(low)
    while i < n:
        if not low[i]:
            i += 1
            continue
        j = i
        while j < n and low[j]:
            j += 1
        a, b = i * S.HOP - pre, j * S.HOP - pre
        a, b = max(a, 0.0), min(b, duration)
        if b - a >= min_s - 1e-9:
            out.append({"t": round(a, 3), "dur": round(b - a, 3),
                        "db": round(float((lvl[i:j] - med[i:j]).min()), 1)})
        i = j
    return out


def pitch(x, t, sr=S.SR, span=PITCH_S, fmin=PITCH_RANGE[0], fmax=PITCH_RANGE[1]):
    """The fundamental (MIDI note number, fractional) of a mono signal just after `t` s: the harmonic product spectrum
    (the spectrum times itself compressed 2, 3 and 4 times) of `span` after it, peaked between fmin and fmax; None when
    the stretch is silent or too short."""
    a, b = int(round((t + span[0]) * sr)), int(round((t + span[1]) * sr))
    seg = np.asarray(x[max(a, 0):b], float)
    if len(seg) < 256 or not np.any(seg):
        return None
    nfft = 1 << 15
    X = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), nfft))
    f = np.fft.rfftfreq(nfft, 1.0 / sr)
    hps = X.copy()
    for h in (2, 3, 4):
        d = X[::h]
        hps[:len(d)] *= d
        hps[len(d):] = 0.0
    sel = (f >= fmin) & (f <= fmax)
    if not hps[sel].any():
        return None
    i = int(np.flatnonzero(sel)[np.argmax(hps[sel])])
    if 0 < i < len(hps) - 1:                                     # the peak's vertex, between bins
        y0, y1, y2 = np.log(hps[i - 1:i + 2] + 1e-300)
        den = y0 - 2.0 * y1 + y2
        i = i + (0.5 * (y0 - y2) / den if den < 0 else 0.0)
    return 69.0 + 12.0 * np.log2(i * sr / nfft / 440.0)


def note_lane(x, lo, hi, pre=0.0, duration=None, sr=S.SR, min_sep=MIN_SEP, range_db=RANGE_DB):
    """[{t, db, midi}]: the onsets of a melodic stem (onset_lane over lo..hi) with the pitch each note starts on."""
    out = []
    for h in onset_lane(x, lo, hi, pre, duration, sr, min_sep, range_db):
        m = pitch(x, h["t"] + pre, sr)
        if m is not None:
            out.append({**h, "midi": round(float(m), 1)})
    return out


def lanes(st, mix, pre, duration, onsets=None, gaps=None, notes=None, sr=S.SR):
    """(hits, meta): every lane of an analysed window (stems `st` {name: (2, n)} and the `mix` (2, n) they were made from;
    the clip starts `pre` s into it and lasts `duration`). `onsets`, `gaps` and `notes` map lane names to (stem, lo, hi),
    (stem, drop_db, min_s) and (stem, lo, hi), default ONSETS, GAPS and NOTES; `mix` names the mix as a stem. meta has
    per lane its kind, stem and settings."""
    src = dict(st, mix=mix)
    hits, meta = {}, {}

    def mono(stem, lane):
        if stem not in src:
            raise HitsError(f"lane {lane!r}: no stem {stem!r} (have {', '.join(sorted(src))})")
        return np.asarray(src[stem], np.float32).mean(0)

    onsets = ONSETS if onsets is None else onsets
    found = {name: onset_lane(mono(stem, name), float(lo), float(hi), pre, duration, sr)
             for name, (stem, lo, hi) in onsets.items()}
    hits.update(separate(found, {name: spec[0] for name, spec in onsets.items()}))
    for name, (stem, lo, hi) in onsets.items():
        meta[name] = {"kind": "onsets", "stem": stem, "band": [float(lo), float(hi)], "range_db": RANGE_DB}
    for name, (stem, drop, min_s) in (GAPS if gaps is None else gaps).items():
        hits[name] = gap_lane(mono(stem, name), float(drop), float(min_s), pre, duration, sr)
        meta[name] = {"kind": "gaps", "stem": stem, "drop_db": float(drop), "min_s": float(min_s)}
    for name, (stem, lo, hi) in (NOTES if notes is None else notes).items():
        hits[name] = note_lane(mono(stem, name), float(lo), float(hi), pre, duration, sr)
        meta[name] = {"kind": "notes", "stem": stem, "band": [float(lo), float(hi)], "range_db": RANGE_DB}
    return hits, meta


def timeline_hits(tl, cache_dir, context=20.0, onsets=None, gaps=None, notes=None, log=print):
    """`lanes` for an existing timeline (a dict from `analyze.analyse`), from its cached stems: the song span is read
    again the way `analyse` read it (analyze.timeline_onsets has the rule), so the stem cache hits."""
    from . import analyze as A
    au = tl.get("audio") or {}
    missing = [k for k in ("file", "start", "duration", "context") if k not in au]
    if missing:
        raise HitsError(f"the timeline has no audio.{' / audio.'.join(missing)}: run `mk timeline analyze`")
    mix, pre, span = A.load_window(au["file"], au["start"], au["duration"], context)
    post = max(0.0, span - pre - au["duration"])
    if abs(pre - au["context"][0]) > 0.002 or abs(post - au["context"][1]) > 0.002:
        raise HitsError(f"--context {context:g} reads a window of [{pre:.3f}, {post:.3f}] s around the clip, the timeline "
                        f"was analysed on {au['context']}: pass the context given to `mk timeline analyze`")
    from .words import stems
    log("stems")
    return lanes(stems(mix, cache_dir), mix, pre, au["duration"], onsets, gaps, notes)


def cached_stems(tl, cache_dir):
    """(stems, pre): the stems {drums, bass, other, vocals} and `mix` ((2, n) float32 each) of the window a timeline (a
    dict from `analyze.analyse`) was analysed on, and `pre`, the seconds of song before clip time 0 in them; from the stem
    cache only (a HitsError when they are not there: nothing is separated here)."""
    from . import analyze as A
    from .words import stems
    au = tl.get("audio") or {}
    missing = [k for k in ("file", "start", "duration", "context") if k not in au]
    if missing:
        raise HitsError(f"the timeline has no audio.{' / audio.'.join(missing)}: run `mk timeline analyze`")
    mix, pre, _ = A.load_window(au["file"], au["start"], au["duration"], max(au["context"]))   # the window analysed
    st = stems(mix, cache_dir, compute=False)
    if st is None:
        raise HitsError("the timeline's stems are not cached: run `mk timeline analyze` (it separates and caches them)")
    return {**st, "mix": mix}, pre


def parse(text, kind):
    """(name, spec) of a lane given on the command line: `NAME=STEM:LO-HI` for onsets and notes, `NAME=STEM:DROP:SECONDS`
    for gaps."""
    try:
        name, rest = text.split("=", 1)
        stem, val = rest.split(":", 1)
        if stem not in STEMS:
            raise HitsError(f"{text!r}: stem {stem!r} is none of {', '.join(STEMS)}")
        if kind == "gaps":
            drop, secs = val.split(":")
            return name.strip(), (stem, float(drop), float(secs))
        lo, hi = val.split("-")
        return name.strip(), (stem, float(lo), float(hi))
    except ValueError as e:
        if isinstance(e, HitsError):
            raise
        form = "NAME=STEM:DROP_DB:SECONDS" if kind == "gaps" else "NAME=STEM:LO-HI"
        raise HitsError(f"{text!r}: expected {form}") from None


def summary(hits, beats=()):
    """Numbers only, per lane: how many, per beat (with `beats`), and for gaps their total length."""
    out = {}
    span = (max(beats) - min(beats)) / max(len(beats) - 1, 1) if len(beats) > 1 else None
    for name, lane in hits.items():
        s = {"count": len(lane)}
        if span and lane:
            s["per_beat"] = round(len(lane) / ((max(beats) - min(beats)) / span + 1), 2)
        if lane and "dur" in lane[0]:
            s["seconds"] = round(sum(h["dur"] for h in lane), 3)
        if lane and "midi" in lane[0]:
            m = [h["midi"] for h in lane]
            s["midi"] = [round(min(m), 1), round(max(m), 1)]
        out[name] = s
    return out
