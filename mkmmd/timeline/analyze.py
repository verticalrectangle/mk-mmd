"""Analyse a song span into a timeline (docs/design.md: Timeline). Times in the result are seconds from the span's
start (clip time 0).

The analysis runs on the span padded with `context` seconds of the song on each side (clamped to the file): tempo,
bar phase and the periodic line split need more than a short clip to be reliable (a 32 s clip alone can read the bar
phase half a bar off). Everything is then shifted and cropped to the clip."""
import numpy as np

from . import signal as S

TIME_KEYS = ("start", "end", "ctc_start", "voiced_end")


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


def analyse(audio, start=0.0, duration=30.0, fps=30, cache_dir="/tmp/mk-timeline", words=True, lyrics=None,
            whisper_model="large-v3", language="en", context=20.0, log=print):
    pre = min(float(context), float(start))
    mix = S.load(audio, start - pre, duration + pre + float(context))
    span = mix.shape[1] / S.SR                               # the window actually read (shorter at the file's end)
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
    return out


def summary(tl):
    """Numbers only (no text): what an agent needs to check a timeline."""
    lines = tl.get("lines", [])
    return {"bpm": tl.get("bpm"), "beats": len(tl.get("beats", [])), "downbeats": len(tl.get("downbeats", [])),
            "grid": tl.get("grid"), "downbeat_contrast": tl.get("downbeat_contrast"),
            "lines": [{"line": i + 1, "words": len(ln["words"]), "start": ln.get("start"), "end": ln.get("end")}
                      for i, ln in enumerate(lines)],
            "words_source": tl.get("words_source"), "word_stats": tl.get("word_stats")}
