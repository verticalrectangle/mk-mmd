"""Signal helpers and the beat tracker (numpy only)."""
import math
import subprocess

import numpy as np

SR = 44100
HOP = 0.01                       # envelope hop (s)


def load(path, start=0.0, duration=None, sr=SR, channels=2):
    """Audio as float32 (channels, samples) via ffmpeg."""
    cmd = ["ffmpeg", "-loglevel", "error", "-ss", f"{start:.6f}", "-i", str(path)]
    if duration:
        cmd += ["-t", f"{duration:.6f}"]
    cmd += ["-ac", str(channels), "-ar", str(sr), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, channels).T.copy()


def frames(x, n, hop):
    xp = np.pad(x, (n // 2, n // 2))
    return np.lib.stride_tricks.sliding_window_view(xp, n)[::hop]


def rms_db(x, sr=SR, hop_s=HOP, win_s=0.03):
    fr = frames(x, int(win_s * sr), int(hop_s * sr))
    return 20 * np.log10(np.sqrt((fr ** 2).mean(1)) + 1e-9)


def band_db(x, lo, hi, sr=SR, hop_s=HOP, n=2048):
    fr = frames(x, n, int(hop_s * sr)) * np.hanning(n)
    S = np.abs(np.fft.rfft(fr, axis=1)) ** 2
    f = np.fft.rfftfreq(n, 1 / sr)
    return 10 * np.log10(S[:, (f >= lo) & (f < hi)].sum(1) + 1e-12)


def onset_envelope(x, sr=SR, hop_s=HOP, n=2048, lo=30.0, hi=8000.0, bands=24):
    """Spectral flux on log band energies (half-wave rectified), smoothed and normalised to unit max."""
    fr = frames(x, n, int(hop_s * sr)) * np.hanning(n)
    S = np.abs(np.fft.rfft(fr, axis=1)) ** 2
    f = np.fft.rfftfreq(n, 1 / sr)
    edges = np.geomspace(lo, hi, bands + 1)
    B = np.stack([S[:, (f >= a) & (f < b)].sum(1) for a, b in zip(edges[:-1], edges[1:])], 1)
    L = np.log1p(1000 * B / (B.max() + 1e-12))
    flux = np.maximum(0, np.diff(L, axis=0, prepend=L[:1])).sum(1)
    w = np.hanning(5)
    flux = np.convolve(flux, w / w.sum(), "same")
    flux -= np.convolve(flux, np.ones(31) / 31, "same")           # remove the slow trend (local mean)
    flux = np.maximum(flux, 0)
    return flux / (flux.max() + 1e-12)


def comb_score(env, period, step=1.0):
    """Best-phase mean of the envelope sampled every `period` hops (the comb filter's response at that period)."""
    n = len(env)
    best = 0.0
    for ph in np.arange(0.0, period, step):
        idx = np.round(np.arange(ph, n - 1, period)).astype(int)
        if len(idx) >= 4:
            best = max(best, float(env[idx].mean()))
    return best


def tempo(env, hop_s=HOP, lo_bpm=60.0, hi_bpm=200.0, prior_bpm=100.0, prior_octaves=0.8):
    """Beat period (s): comb-filter response over candidate tempos weighted by a log-tempo prior, refined to
    0.05 bpm. The comb is robust where the autocorrelation is not (subdivisions and 2:3 patterns); the prior picks
    the octave."""
    def weighted(bpm):
        return comb_score(env, 60.0 / bpm / hop_s) * math.exp(-0.5 * (math.log2(bpm / prior_bpm) / prior_octaves) ** 2)
    coarse = np.arange(lo_bpm, hi_bpm + 1e-9, 0.5)
    s = np.array([weighted(b) for b in coarse])
    b0 = float(coarse[int(np.argmax(s))])
    fine = np.arange(max(lo_bpm, b0 - 0.6), min(hi_bpm, b0 + 0.6) + 1e-9, 0.05)
    sf = np.array([weighted(b) for b in fine])
    return 60.0 / float(fine[int(np.argmax(sf))])


def track_beats(env, period_s, hop_s=HOP, tightness=100.0):
    """Ellis-style dynamic programming: beats that land on strong onsets with spacing close to the period."""
    P = period_s / hop_s
    n = len(env)
    score = env.astype(float).copy()
    back = np.full(n, -1)
    lo, hi = int(round(P / 2)), int(round(2 * P))
    for t in range(lo, n):
        a, b = max(0, t - hi), t - lo
        if b <= a:
            continue
        taus = np.arange(a, b)
        pen = -tightness * (np.log((t - taus) / P)) ** 2
        cand = score[a:b] + pen
        k = int(np.argmax(cand))
        if cand[k] > 0:
            score[t] = env[t] + cand[k]
            back[t] = a + k
    tail = slice(max(0, n - int(round(P))), n)
    t = tail.start + int(np.argmax(score[tail]))
    beats = []
    while t >= 0:
        beats.append(t)
        t = back[t]
    return np.array(beats[::-1], float) * hop_s


def fit_grid(beats, max_resid_ms=35.0):
    """A constant-tempo grid through the beats when they are that steady (least squares on beat index); else None."""
    if len(beats) < 4:
        return None
    k = np.arange(len(beats))
    period, phase = np.polyfit(k, beats, 1)
    resid = beats - (phase + period * k)
    if np.abs(resid).max() * 1000 > max_resid_ms:
        return None
    return period, phase, float(np.abs(resid).max() * 1000)


def downbeat_phase(beats, kick_env, hop_s=HOP, per_bar=4):
    """Which of the `per_bar` beat phases carries the strongest kick accents (0..per_bar-1) and its contrast."""
    idx = np.clip(np.round(np.asarray(beats) / hop_s).astype(int), 0, len(kick_env) - 1)
    val = np.array([kick_env[max(0, i - 2):i + 3].max() for i in idx])
    means = [val[p::per_bar].mean() if len(val[p::per_bar]) else 0.0 for p in range(per_bar)]
    best = int(np.argmax(means))
    rest = [m for i, m in enumerate(means) if i != best]
    return best, float(means[best] / (np.mean(rest) + 1e-9))


def voicing(voc, rel_db, sr=SR, hop_s=HOP, f_lo=90.0, f_hi=1000.0, ac_min=0.5, db_min=-42.0):
    """Per hop: True where the vocal is pitched (normalised autocorrelation peak > ac_min) and louder than db_min."""
    n = 2048
    fr = frames(voc, n, int(hop_s * sr)) * np.hanning(n)
    F = np.fft.rfft(fr, n=2 * n, axis=1)
    ac = np.fft.irfft(np.abs(F) ** 2, axis=1)[:, :n]
    ac = ac / (ac[:, :1] + 1e-12)
    v = ac[:, int(sr / f_hi):int(sr / f_lo)].max(1)
    m = min(len(v), len(rel_db))
    return (v[:m] > ac_min) & (rel_db[:m] > db_min)


def vocal_onsets(voc, rel_db_fn, sr=SR, hop_s=0.005):
    """Syllable onsets of the vocal: spectral flux peaks (150-5000 Hz) that are clearly above the median and loud."""
    n, hop = 1024, int(hop_s * sr)
    fr = frames(voc, n, hop) * np.hanning(n)
    S = np.abs(np.fft.rfft(fr, axis=1))
    f = np.fft.rfftfreq(n, 1 / sr)
    Lg = np.log1p(100 * S[:, (f >= 150) & (f <= 5000)])
    nov = np.maximum(0, np.diff(Lg, axis=0, prepend=Lg[:1])).sum(1)
    w = np.hanning(7)
    nov = np.convolve(nov, w / w.sum(), "same")
    e = rel_db_fn(hop_s)[:len(nov)]
    med = np.median(nov)
    mad = np.median(np.abs(nov - med)) + 1e-9
    out = []
    for i in range(8, len(nov) - 8):
        if nov[i] == nov[i - 8:i + 9].max() and nov[i] > med + 2.0 * mad and e[min(len(e) - 1, i + 6)] > -35:
            out.append((i * hop_s, float((nov[i] - med) / mad)))
    return out
