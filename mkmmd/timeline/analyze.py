"""Analyse a song span into a timeline (docs/design.md: Timeline). Times in the result are seconds from the span's
start (clip time 0).

The analysis runs on the span padded with `context` seconds of the song on each side (clamped to the file): tempo,
bar phase and the periodic line split need more than a short clip to be reliable (a 32 s clip alone can read the bar
phase half a bar off). Everything is then shifted and cropped to the clip."""
import numpy as np

from .. import config as CFG
from . import signal as S

TIME_KEYS = ("start", "end", "ctc_start", "voiced_end")
ONSET_STEMS = ("other",)                 # stems whose note onsets `analyse` stores (guitars and keys)


def load_window(audio, start, duration, context):
    """(mix, pre, span): the song window `analyse` analyses, read exactly as it reads it (the stem cache is keyed by a
    hash of these samples): the clip plus `context` s of song on each side, `pre` = min(context, start) s of it before
    the clip, `span` s read in all (shorter than duration + pre + context at the end of the file)."""
    pre = min(float(context), float(start))
    mix = S.load(audio, start - pre, duration + pre + float(context))
    return mix, pre, mix.shape[1] / S.SR


def _shift_words(lines, dt, duration, tol=0.02):
    """Lines with every time moved by -dt, keeping the words that lie inside [0, duration]; empty lines dropped."""
    out = []
    for ln in lines:
        words = []
        for w in ln["words"]:
            w = dict(w)
            for k in TIME_KEYS:
                if isinstance(w.get(k), (int, float)):
                    w[k] = round(float(w[k]) - dt, 3)
            if isinstance(w.get("whisper"), list):
                w["whisper"] = [round(float(x) - dt, 3) if isinstance(x, (int, float)) else x for x in w["whisper"]]
            if w["start"] >= -tol and w["end"] <= duration + tol:
                words.append(w)
        if words:
            out.append({**ln, "start": words[0]["start"], "end": words[-1]["end"], "words": words})
    return out


def analyse(audio, start=0.0, duration=30.0, fps=30, cache_dir=None, words=True, lyrics=None,
            whisper_model="large-v3", language="en", context=20.0, log=print):
    """The timeline of a song span (see the module docstring). `cache_dir` holds the cached Demucs stems (default:
    `timeline` in the user cache, `MK_CACHE`)."""
    cache_dir = CFG.cache_dir() / "timeline" if cache_dir is None else cache_dir
    mix, pre, span = load_window(audio, start, duration, context)     # span: the window read (short at the file's end)
    post = max(0.0, span - pre - duration)
    mono = mix.mean(0)
    out = {"fps": fps, "clip": {"start": 0.0, "end": round(duration, 3), "fps": fps},
           "audio": {"file": str(audio), "start": float(start), "duration": float(duration),
                     "context": [round(pre, 3), round(post, 3)]}}
    from .words import stems
    log("stems")
    st = stems(mix, cache_dir)
    drums = st["drums"].mean(0)
    voc = st["vocals"].mean(0)
    # ---- beats: kick + snare bands of the drum stem (hi-hats subdivide the beat and pull the tempo up an octave)
    #      plus the bass; tempo prior centred on 100 bpm. Window times; shifted to clip times at the end.
    env = S.onset_envelope(drums, lo=30.0, hi=2500.0) + 0.5 * S.onset_envelope(st["bass"].mean(0), lo=30.0, hi=400.0)
    env /= env.max() + 1e-12
    period = S.tempo(env, prior_bpm=100.0, prior_octaves=0.8)
    beats = S.track_beats(env, period)
    grid = S.fit_grid(beats)
    if grid:
        p, ph, resid = grid
        k0 = int(np.ceil(-ph / p))
        beats = ph + p * np.arange(k0, int((span - ph) / p) + 1)
        period = p
    kick = S.onset_envelope(drums, lo=30.0, hi=150.0, bands=6)
    harm = S.harmonic_change(S.chroma_per_beat(st["bass"].mean(0) + st["other"].mean(0), beats))
    dph, contrast = S.downbeat_phase(beats, kick, harm=harm)
    downbeats = beats[dph::4]
    beats, downbeats = beats - pre, downbeats - pre
    out.update({"bpm": round(60.0 / period, 3), "beat_s": round(period, 4),
                "beats": [round(float(b), 3) for b in beats if 0 <= b <= duration],
                "downbeats": [round(float(b), 3) for b in downbeats if 0 <= b <= duration],
                "grid": {"period": round(grid[0], 5), "phase": round((grid[1] - pre) % grid[0], 4),
                         "resid_ms": round(grid[2], 1)} if grid else None,
                "downbeat_contrast": round(contrast, 2)})
    # ---- loudness per frame of the clip, relative to the clip's loudest moment
    n = int(round(duration * fps))
    h0, h1 = int(pre / S.HOP), int((pre + duration) / S.HOP)

    def per_frame(db):
        db = db - db[h0:max(h1, h0 + 1)].max()
        vals = []
        for f in range(n):
            a, b = int((pre + f / fps) / S.HOP), int((pre + (f + 1) / fps) / S.HOP)
            vals.append(float(10 * np.log10(np.mean(10 ** (db[a:max(b, a + 1)] / 10)) + 1e-12)))
        return [round(v, 2) for v in np.convolve(np.pad(vals, 1, mode="edge"), np.ones(3) / 3, "valid")]

    out["vocal_db"] = per_frame(S.rms_db(voc))
    out["energy_db"] = per_frame(S.rms_db(mono))
    out["drums_db"] = per_frame(S.rms_db(drums))
    # ---- words: whisper hears the whole window (better phrase splits); a lyrics file is aligned on the clip only
    if words:
        from . import words as W
        import torch
        import torchaudio
        if lyrics:
            a, b = int(pre * S.SR), int((pre + duration) * S.SR)
            v_al, t0, length = voc[a:b], 0.0, duration
        else:
            v_al, t0, length = voc, pre, span
        v16 = torchaudio.functional.resample(torch.from_numpy(np.ascontiguousarray(v_al))[None], S.SR, 16000)[0].numpy()
        log("whisper")
        segs = W.transcribe(v16, whisper_model, language)
        lines = W.lines_from_file(lyrics) if lyrics else W.lines_from_whisper(segs)
        log("align")
        tl_lines, _, stats = W.word_timeline(v_al, S.SR, lines, length, whisper=segs)
        if not lyrics:
            tl_lines = W.resplit_periodic(tl_lines)
        out["lines"] = _shift_words(tl_lines, t0, duration)
        out["words_source"] = "lyrics file" if lyrics else "whisper"
        out["word_stats"] = stats
    out["onsets"], out["onsets_meta"] = stem_onsets(st, pre, duration)
    return out


def stem_onsets(st, pre, duration, names=ONSET_STEMS, lo=S.ONSET_LO, hi=S.ONSET_HI):
    """({stem: [t, ...]}, {stem: meta}) as a timeline stores them: the note onsets (signal.band_onsets) of the chosen
    stems of an analysed window `st`, in clip seconds (the clip starts `pre` s into the window), inside [0, duration],
    to the ms. meta: `band` (Hz), `latency_ms` (how early the detector's flux peaks on a synthetic pluck; already
    removed from the times) and `strength` (dB per onset: the biggest rise of its strum)."""
    lat = S.onset_latency(S.SR, lo, hi)
    onsets, meta = {}, {}
    for name in names:
        if name not in st:
            raise ValueError(f"no stem {name!r} (have {', '.join(sorted(st))})")
        t, s = S.band_onsets(st[name].mean(0), S.SR, lo, hi, latency=lat)
        t = t - pre
        keep = (t >= 0) & (t <= duration)
        onsets[name] = [round(float(v), 3) for v in t[keep]]
        meta[name] = {"band": [float(lo), float(hi)], "latency_ms": round(lat * 1000, 2),
                      "strength": [round(float(v), 1) for v in s[keep]]}
    return onsets, meta


def timeline_onsets(tl, cache_dir, names=ONSET_STEMS, lo=S.ONSET_LO, hi=S.ONSET_HI, context=20.0, log=print):
    """stem_onsets for an existing timeline (a dict from `analyse`) from the cached stems: the window is read again the
    way `analyse` read it, from the `audio` block (file, start, duration), so the stem cache -- keyed by a hash of the
    window's samples -- hits. `context` is the one given to `analyse`; a window whose [pre, post] differs from the one
    recorded is refused (it would be new stems, minutes of Demucs, on a time base that is not the timeline's)."""
    au = tl.get("audio") or {}
    missing = [k for k in ("file", "start", "duration", "context") if k not in au]
    if missing:
        raise ValueError(f"the timeline has no audio.{' / audio.'.join(missing)}: run `mk timeline analyze`")
    mix, pre, span = load_window(au["file"], au["start"], au["duration"], context)
    post = max(0.0, span - pre - au["duration"])
    if abs(pre - au["context"][0]) > 0.002 or abs(post - au["context"][1]) > 0.002:
        raise ValueError(f"--context {context:g} reads a window of [{pre:.3f}, {post:.3f}] s around the clip, the "
                         f"timeline was analysed on {au['context']}: pass the context given to `mk timeline analyze`")
    from .words import stems
    log("stems")
    return stem_onsets(stems(mix, cache_dir), pre, au["duration"], names, lo, hi)


def eighths(beats):
    """The 8th-note grid through a beat list: every beat and the midpoint to the next."""
    b = np.sort(np.asarray(beats, float))
    return np.sort(np.concatenate([b, (b[:-1] + b[1:]) / 2.0]))


def onset_stats(times, beats, tol=0.25):
    """Numbers on how onsets sit on the 8th-note grid of the beats (between the first and the last beat): `count` (all
    onsets), `per_beat`, `offset_ms` = {median, mad, p95} of the signed distance to the nearest 8th (positive: after
    it; the MAD is the robust spread; p95 is of the absolute distance), `empty_8ths` (grid points with no onset within
    tol x the 8th's length, the grid shifted by the median offset) and `off_8th` (onsets farther than that from every
    grid point)."""
    t = np.sort(np.asarray(times, float))
    out = {"count": int(len(t))}
    g = eighths(beats) if len(beats) >= 2 else np.zeros(0)
    t = t[(t >= g[0]) & (t <= g[-1])] if len(g) else t[:0]
    if len(t) == 0:
        return out
    j = np.clip(np.searchsorted(g, t), 1, len(g) - 1)
    off = t - np.where(t - g[j - 1] <= g[j] - t, g[j - 1], g[j])
    med = float(np.median(off))
    per = np.diff(g)                                          # length of the 8th that starts at each grid point
    gs = g + med
    near = np.abs(t[:, None] - gs[None, :])
    out.update({
        "per_beat": round(len(t) / ((len(g) - 1) / 2.0), 2),
        "offset_ms": {"median": round(med * 1000, 1), "mad": round(float(np.median(np.abs(off - med))) * 1000, 1),
                      "p95": round(float(np.percentile(np.abs(off), 95)) * 1000, 1)},
        "empty_8ths": int((near[:, :-1].min(0) > tol * per).sum()),
        "off_8th": int((near.min(1) > tol * per[np.minimum(near.argmin(1), len(per) - 1)]).sum())})
    return out


def summary(tl):
    """Numbers only (no text): what an agent needs to check a timeline."""
    lines = tl.get("lines", [])
    return {"bpm": tl.get("bpm"), "beats": len(tl.get("beats", [])), "downbeats": len(tl.get("downbeats", [])),
            "grid": tl.get("grid"), "downbeat_contrast": tl.get("downbeat_contrast"),
            "lines": [{"line": i + 1, "words": len(ln["words"]), "start": ln.get("start"), "end": ln.get("end")}
                      for i, ln in enumerate(lines)],
            "words_source": tl.get("words_source"), "word_stats": tl.get("word_stats"),
            "onsets": {k: onset_stats(v, tl.get("beats", [])) for k, v in (tl.get("onsets") or {}).items()} or None}
