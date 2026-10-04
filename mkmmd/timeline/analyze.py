"""Analyse a song span into a timeline (docs/design.md: Timeline). Times in the result are seconds from the span's
start (clip time 0)."""
import numpy as np

from . import signal as S


def analyse(audio, start=0.0, duration=30.0, fps=30, cache_dir="/tmp/mk-timeline", words=True, lyrics=None,
            whisper_model="large-v3", language="en", log=print):
    mix = S.load(audio, start, duration)
    mono = mix.mean(0)
    out = {"fps": fps, "clip": {"start": 0.0, "end": round(duration, 3), "fps": fps},
           "audio": {"file": str(audio), "start": float(start), "duration": float(duration)}}
    from .words import stems
    log("stems")
    st = stems(mix, cache_dir)
    drums = st["drums"].mean(0)
    voc = st["vocals"].mean(0)
    # ---- beats: kick + snare bands of the drum stem (hi-hats subdivide the beat and pull the tempo up an octave)
    #      plus the bass; tempo prior centred on 100 bpm
    env = S.onset_envelope(drums, lo=30.0, hi=2500.0) + 0.5 * S.onset_envelope(st["bass"].mean(0), lo=30.0, hi=400.0)
    env /= env.max() + 1e-12
    period = S.tempo(env, prior_bpm=100.0, prior_octaves=0.8)
    beats = S.track_beats(env, period)
    grid = S.fit_grid(beats)
    if grid:
        p, ph, resid = grid
        k0 = int(np.ceil(-ph / p))
        beats = ph + p * np.arange(k0, int((duration - ph) / p) + 1)
        period = p
    kick = S.onset_envelope(drums, lo=30.0, hi=150.0, bands=6)
    dph, contrast = S.downbeat_phase(beats, kick)
    downbeats = beats[dph::4]
    out.update({"bpm": round(60.0 / period, 3), "beat_s": round(period, 4),
                "beats": [round(float(b), 3) for b in beats if 0 <= b <= duration],
                "downbeats": [round(float(b), 3) for b in downbeats if 0 <= b <= duration],
                "grid": {"period": round(grid[0], 5), "phase": round(grid[1], 4), "resid_ms": round(grid[2], 1)}
                if grid else None, "downbeat_contrast": round(contrast, 2)})
    # ---- loudness per frame
    n = int(round(duration * fps))

    def per_frame(db):
        vals = []
        for f in range(n):
            a, b = int(f / fps / S.HOP), int((f + 1) / fps / S.HOP)
            vals.append(float(10 * np.log10(np.mean(10 ** (db[a:max(b, a + 1)] / 10)) + 1e-12)))
        return [round(v, 2) for v in np.convolve(np.pad(vals, 1, mode="edge"), np.ones(3) / 3, "valid")]

    vrel = S.rms_db(voc) - S.rms_db(voc).max()
    out["vocal_db"] = per_frame(vrel)
    out["energy_db"] = per_frame(S.rms_db(mono) - S.rms_db(mono).max())
    out["drums_db"] = per_frame(S.rms_db(drums) - S.rms_db(drums).max())
    # ---- words
    if words:
        from . import words as W
        import torch
        import torchaudio
        v16 = torchaudio.functional.resample(torch.from_numpy(voc)[None], S.SR, 16000)[0].numpy()
        log("whisper")
        segs = W.transcribe(v16, whisper_model, language)
        lines = W.lines_from_file(lyrics) if lyrics else W.lines_from_whisper(segs)
        log("align")
        tl_lines, _, stats = W.word_timeline(voc, S.SR, lines, duration, whisper=segs)
        out["lines"] = tl_lines if lyrics else W.resplit_periodic(tl_lines)
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
