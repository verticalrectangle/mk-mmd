import numpy as np
import pytest

from mkmmd.timeline import signal as S


def clicks(bpm=100.0, seconds=20.0, sr=S.SR, accent_every=4, accent_phase=1, offset=0.37):
    x = np.zeros(int(seconds * sr), np.float32)
    kick = np.zeros_like(x)
    period = 60.0 / bpm
    rng = np.random.default_rng(0)
    for k, t in enumerate(np.arange(offset, seconds - 0.1, period)):
        i = int(t * sr)
        burst = rng.normal(0, 1, 400).astype(np.float32) * np.exp(-np.arange(400) / 80.0)
        x[i:i + 400] += burst * 0.5
        if k % accent_every == accent_phase:
            low = np.sin(2 * np.pi * 60 * np.arange(2000) / sr).astype(np.float32) * np.exp(-np.arange(2000) / 600)
            x[i:i + 2000] += low
            kick[i:i + 2000] += low
    return x, kick, period, offset


def test_tempo_and_beats_on_clicks():
    x, kick, period, offset = clicks()
    env = S.onset_envelope(x)
    p = S.tempo(env)
    assert p == pytest.approx(period, rel=0.02)
    beats = S.track_beats(env, p)
    grid = S.fit_grid(beats)
    assert grid is not None
    assert grid[0] == pytest.approx(period, rel=0.01)
    k = np.round((beats - offset) / period)
    assert np.abs(beats - (offset + k * period)).max() < 0.03          # every beat within 30 ms of a click


def test_downbeat_phase_follows_the_kick():
    x, kick, period, offset = clicks(accent_phase=2)
    env = S.onset_envelope(x)
    beats = S.track_beats(env, S.tempo(env))
    kick_env = S.onset_envelope(kick, lo=30, hi=200, bands=6)
    first = int(round((beats[0] - offset) / period))                   # index of the first tracked beat
    ph, contrast = S.downbeat_phase(beats, kick_env)
    assert (first + ph) % 4 == 2 and contrast > 1.5


def test_voicing_detects_a_pitched_tone_not_noise():
    sr = S.SR
    t = np.arange(sr) / sr
    tone = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    noise = (0.3 * np.random.default_rng(1).normal(0, 1, sr)).astype(np.float32)
    x = np.concatenate([tone, noise])
    rel = S.rms_db(x) - S.rms_db(x).max()
    v = S.voicing(x, rel)
    half = len(v) // 2
    assert v[10:half - 10].mean() > 0.9 and v[half + 10:-10].mean() < 0.1


def test_resplit_periodic_splits_merged_lines_at_the_line_period():
    from mkmmd.timeline.words import resplit_periodic

    def line(t0, n, step=0.4, hold=0.3):
        return [{"start": round(t0 + k * step, 3), "voiced_end": round(t0 + k * step + hold, 3)} for k in range(n)]

    # lines every 3.8 s; the last two lines were merged and the second starts right after a held note (no pause)
    merged = line(11.4, 5) + [{"start": 13.5, "voiced_end": 15.2}] + line(15.25, 5)
    lines = [{"words": line(0.0, 5)}, {"words": line(3.8, 5)}, {"words": line(7.6, 5)}, {"words": merged}]
    out = resplit_periodic(lines)
    assert [ln["start"] for ln in out] == [0.0, 3.8, 7.6, 11.4, 15.25]
    assert [len(ln["words"]) for ln in out] == [5, 5, 5, 6, 5]


def test_resplit_periodic_leaves_no_fragment_at_the_end():
    from mkmmd.timeline.words import resplit_periodic

    def w(s, e):
        return {"start": s, "voiced_end": e}
    base = [{"words": [w(0.0 + 3.8 * i, 0.3 + 3.8 * i), w(1.0 + 3.8 * i, 2.0 + 3.8 * i)]} for i in range(4)]
    # two lines merged; the second ends with a word that starts 3.7 s after the merged line's start + p
    merged = [w(15.2, 15.5), w(16.0, 17.0), w(19.0, 19.3), w(20.0, 21.6), w(22.3, 22.7)]
    out = resplit_periodic(base + [{"words": merged}])
    assert [ln["start"] for ln in out][-2:] == [15.2, 19.0] and len(out[-1]["words"]) == 3
