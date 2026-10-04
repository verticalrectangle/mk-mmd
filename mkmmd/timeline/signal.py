"""Signal helpers and the beat tracker (numpy only)."""
import functools
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


def chroma_per_beat(x, beats, sr=SR, fmin=55.0, fmax=2000.0, n=16384):
    """(len(beats), 12) unit pitch-class profiles of the audio from each beat to the next (magnitude spectrum folded
    onto 12 semitone classes between fmin and fmax; the last beat's row is zero)."""
    beats = np.asarray(beats, float)
    f = np.fft.rfftfreq(n, 1 / sr)
    sel = (f >= fmin) & (f <= fmax)
    pc = (np.round(12 * np.log2(f[sel] / 440.0)) % 12).astype(int)
    out = np.zeros((len(beats), 12))
    for k in range(len(beats) - 1):
        a, b = max(int(beats[k] * sr), 0), min(int(beats[k + 1] * sr), len(x))
        seg = x[a:min(b, a + n)]
        if len(seg) < 512:
            continue
        mag = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), n))
        c = np.bincount(pc, weights=mag[sel], minlength=12)
        out[k] = c / (np.linalg.norm(c) + 1e-12)
    return out


def harmonic_change(chroma):
    """Per beat: how much the pitch content changes as that beat starts (1 - cosine to the previous beat's profile)."""
    nov = np.zeros(len(chroma))
    ok = (np.linalg.norm(chroma[1:], axis=1) > 0) & (np.linalg.norm(chroma[:-1], axis=1) > 0)
    nov[1:][ok] = 1.0 - np.einsum("ij,ij->i", chroma[1:][ok], chroma[:-1][ok])
    return nov


def downbeat_phase(beats, kick_env, hop_s=HOP, per_bar=4, harm=None, w_harm=1.0):
    """Which of the `per_bar` beat phases starts the bar (0..per_bar-1) and how clearly. Kick accents alone cannot tell
    beat 1 from beat 3 in a rock beat (both are kicked); chords change on beat 1, so `harm` (per-beat harmonic change,
    see harmonic_change) breaks the tie. Each cue is scored relative to its own mean, then summed."""
    idx = np.clip(np.round(np.asarray(beats) / hop_s).astype(int), 0, len(kick_env) - 1)
    val = np.array([kick_env[max(0, i - 2):i + 3].max() for i in idx])

    def rel(v):
        m = np.array([v[p::per_bar].mean() if len(v[p::per_bar]) else 0.0 for p in range(per_bar)])
        return m / (m.mean() + 1e-9)

    score = rel(val)
    if harm is not None and len(harm) == len(val) and np.asarray(harm).max() > 0:
        score = score + w_harm * rel(np.asarray(harm, float))
    best = int(np.argmax(score))
    rest = [m for i, m in enumerate(score) if i != best]
    return best, float(score[best] / (np.mean(rest) + 1e-9))


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


# ---------------------------------------------------------------------------------------------------- note onsets
ONSET_LO, ONSET_HI = 800.0, 6000.0           # Hz: pick noise and the upper harmonics of plucked / strummed strings


def _band_energies(x, sr, lo, hi, n, hop, bands):
    """((frames, bands) power of the Hann-windowed STFT of x summed over `bands` log-spaced slices of lo..hi, scale):
    the total over the bands divided by `scale` (3/8 n per bin) is the variance of white noise of the same level in the
    band. Frame i is centred on sample i * hop; the signal is mirrored at both ends, so a loud start is no pluck."""
    x = np.asarray(x, np.float64)
    if len(x) <= n // 2:
        return np.zeros((0, 1)), 1.0
    k0 = max(1, int(math.ceil(lo * n / sr)))
    k1 = max(k0, min(n // 2, int(math.floor(hi * n / sr))))
    edges = np.unique(np.round(np.geomspace(k0, k1 + 1, bands + 1)).astype(int))
    fr = np.lib.stride_tricks.sliding_window_view(np.pad(x, (n // 2, n // 2), mode="reflect"), n)[::hop]
    w = np.hanning(n)
    out = np.empty((len(fr), len(edges) - 1))
    for a in range(0, len(fr), 2048):                         # chunks: 70 s at a 2.5 ms hop would be ~0.5 GB at once
        F = np.fft.rfft(fr[a:a + 2048] * w, axis=1)[:, edges[0]:edges[-1]]
        out[a:a + 2048] = np.add.reduceat(F.real ** 2 + F.imag ** 2, edges[:-1] - edges[0], axis=1)
    return out, float((w ** 2).sum() * (edges[-1] - edges[0]))


def _flux(B, lag, floor_db, gate=0.0):
    """Per frame, in dB: the mean over the bands of the half-wave rectified rise of the log energy since `lag` frames
    ago. Energies are relative to the 99.5th percentile of the total and floored at `floor_db` (the same signal at any
    level gives the same flux; noise far below the music rises by nothing); frames whose total is below `gate` (power)
    are silence and have no flux."""
    tot = B.sum(1)
    ref = max(float(np.percentile(tot, 99.5)), 1e-30)
    L = 10.0 * np.log10(B / ref + 10.0 ** (floor_db / 10.0))
    fl = np.zeros(len(B))
    if len(B) > lag:
        fl[lag:] = np.maximum(L[lag:] - L[:-lag], 0.0).mean(1)
    fl[tot < gate] = 0.0
    return fl


def _local_threshold(fl, hw, step, k):
    """Rolling median + k * 1.4826 * MAD of the flux over +-hw frames, every `step` frames, interpolated."""
    pad = np.pad(fl, hw, mode="reflect" if len(fl) > hw else "edge")
    win = np.lib.stride_tricks.sliding_window_view(pad, 2 * hw + 1)[::step]
    med = np.median(win, axis=1)
    mad = np.median(np.abs(win - med[:, None]), axis=1)
    return np.interp(np.arange(len(fl)), np.arange(len(win)) * step, med + k * 1.4826 * mad)


def _subframe(fl, i):
    """Offset (-0.5..0.5 frames) of the vertex of the parabola through the flux at i - 1, i, i + 1."""
    den = fl[i - 1] - 2.0 * fl[i] + fl[i + 1]
    return float(np.clip(0.5 * (fl[i - 1] - fl[i + 1]) / den, -0.5, 0.5)) if den < 0 else 0.0


def _steps(sr, hop_s, lag_s):
    """(hop in samples, hop in s, lag in hops): the frame grid actually used for the requested hop and lag."""
    hop = max(1, int(round(hop_s * sr)))
    return hop, hop / sr, max(1, int(round(lag_s * sr / hop)))


def band_onsets(x, sr=SR, lo=ONSET_LO, hi=ONSET_HI, n=1024, hop_s=0.0025, lag_s=0.010, bands=8, floor_db=-60.0,
                gate_db=-70.0, k=2.5, win_s=1.0, min_rise=2.0, min_sep=0.10, start_frac=0.3, start_gap=0.04,
                latency=None):
    """When notes are plucked or strummed in a mono signal (a Demucs `other` stem): (times in s from the first sample,
    strengths in dB), sorted by time.

    The detector is the spectral flux of the band lo..hi: a Hann-windowed STFT (n samples, a frame every hop_s), the
    power summed into `bands` log-spaced slices, and per frame the mean over the slices of the rise in dB since lag_s
    ago (half-wave rectified: a pluck lifts every slice by tens of dB, a decaying string by nothing). Levels are
    relative to the signal's loud parts and floored `floor_db` below them, and frames quieter than `gate_db` (dB
    re full-scale white noise of the same level in the band) are silence: quiet noise is no onset. A flux peak counts
    when it exceeds the local median + k MAD of the flux (over win_s, so dense and sparse passages each get their own
    noise floor) and `min_rise` dB; the strongest peak within min_sep s wins (0.1 s: a strum spreads over up to ~90 ms
    and is one onset, while 16th notes up to 150 bpm stay apart) and its flux is the strength.

    The time is where the pluck or strum STARTS. A strum is not one step: its notes arrive over 20-90 ms, the energy
    climbs in steps and the biggest is often the last. From the winning peak the detector walks back through the flux
    peaks of its cluster (not more than min_sep before it, nor past the midpoint to the previous onset) while each is
    at least `start_frac` of the winner and within `start_gap` s of the one after it, and reports the earliest. Then
    the time is refined to a fraction of a frame, and is shifted by the detector's latency: frames are centred, so the
    flux peaks a few ms BEFORE the first sample of a pluck. `latency` (s, peak time minus pluck start; negative =
    early) is subtracted from every time: None measures it for these settings on synthetic plucks (onset_latency), 0
    returns the raw flux peaks."""
    hop, hs, lag = _steps(sr, hop_s, lag_s)
    B, scale = _band_energies(x, sr, lo, hi, n, hop, bands)
    if len(B) < 3:
        return np.zeros(0), np.zeros(0)
    fl = _flux(B, lag, floor_db, 10.0 ** (gate_db / 10.0) * scale)
    thr = np.maximum(min_rise, _local_threshold(fl, int(round(win_s / 2 / hs)), max(1, int(round(0.05 / hs))), k))
    cand = np.flatnonzero((fl[1:-1] > fl[:-2]) & (fl[1:-1] >= fl[2:]) & (fl[1:-1] > thr[1:-1])) + 1
    sep, gap = max(1, int(round(min_sep / hs))), max(1, int(round(start_gap / hs)))
    taken = np.zeros(len(fl), bool)
    main = []
    for i in cand[np.argsort(-fl[cand], kind="stable")]:     # strongest first: it claims +-min_sep around itself
        if not taken[i]:
            main.append(int(i))
            taken[max(0, i - sep + 1):i + sep] = True
    main.sort()
    start = []
    for j, p in enumerate(main):
        first = max(p - sep + 1, (main[j - 1] + p) // 2 + 1 if j else 0)
        s = p
        for c in cand[np.searchsorted(cand, first):np.searchsorted(cand, p)][::-1]:
            if s - c > gap:
                break
            if fl[c] >= start_frac * fl[p]:
                s = int(c)
        start.append(s)
    if latency is None:
        latency = onset_latency(sr, lo, hi, n, hop_s, lag_s, bands, floor_db)
    t = np.array([(s + _subframe(fl, s) - lag / 2.0) * hs for s in start]) - latency
    return np.maximum(t, 0.0), fl[np.array(main, int)]


@functools.lru_cache(maxsize=16)
def onset_latency(sr=SR, lo=ONSET_LO, hi=ONSET_HI, n=1024, hop_s=0.0025, lag_s=0.010, bands=8, floor_db=-60.0):
    """Seconds from the first sample of a plucked note to the flux peak of band_onsets (negative: early). Measured, not
    modelled: the detector's own flux is run on synthetic plucks (decaying white noise, 60 ms time constant, attacks of
    0.5, 2 and 5 ms, five start phases within one hop) and the mean of peak time minus start is returned. Frames are
    centred, so the window's leading edge meets a pluck about n / 2 before it, and the 10 ms difference and the log's
    steep rise give back part of that: -6.3 ms for the defaults at 44.1 kHz. The spread over other plucks is small
    (noise bursts -7.0 ms for a 0.5 ms attack .. -4.1 ms for 40 ms; decaying harmonic plucks from 196 Hz to 1.5 kHz
    -5.4 .. -6.4 ms), so the compensated time is within about 2 ms of the start whatever the attack."""
    hop, hs, lag = _steps(sr, hop_s, lag_s)
    rng = np.random.default_rng(0)
    t = np.arange(int(0.4 * sr)) / sr
    noise = rng.standard_normal(len(t))
    start = int(0.3 * sr)
    lat = []
    for attack in (0.0005, 0.002, 0.005):
        burst = noise * np.exp(-t / 0.06) * np.minimum(1.0, t / attack)
        for phase in range(5):
            i0 = start + phase * hop // 5
            x = np.zeros(start + len(burst) + 4 * n)
            x[i0:i0 + len(burst)] = burst
            fl = _flux(_band_energies(x, sr, lo, hi, n, hop, bands)[0], lag, floor_db)
            i = int(np.argmax(fl))
            lat.append((i + _subframe(fl, i) - lag / 2.0) * hs - i0 / sr)
    return float(np.mean(lat))
