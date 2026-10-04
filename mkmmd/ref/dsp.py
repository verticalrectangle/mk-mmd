"""Numerics on tracked series. A series is a 1-D float array sampled at the clip's frame rate; NaN marks frames
without a measurement (face not found, hand out of frame, cut). Everything here tolerates NaN and never fills a gap
silently: `interp_gaps` bridges short ones explicitly, event finders never report an event that touches a gap or the
ends of the series."""
import numpy as np

_trapz = getattr(np, "trapezoid", None) or np.trapz


def sig(x, d=4):
    """Round to d significant digits for JSON; None for non-finite."""
    if x is None:
        return None
    x = float(x)
    if not np.isfinite(x):
        return None
    return 0.0 if x == 0 else float(f"{x:.{d}g}")


def runs(mask):
    """[(i0, i1)) of consecutive True samples."""
    m = np.concatenate([[0], np.asarray(mask, np.int8), [0]])
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def interp_gaps(x, max_gap):
    """Linear interpolation across NaN gaps of at most max_gap samples; longer gaps and the series ends stay NaN."""
    x = np.asarray(x, float)
    good = np.isfinite(x)
    if good.sum() < 2:
        return x.copy()
    idx = np.arange(len(x))
    out = np.interp(idx, idx[good], x[good])
    for a, b in runs(~good):
        if (b - a) > max_gap or a == 0 or b == len(x):
            out[a:b] = np.nan
    return out


def gsmooth(x, sigma):
    """Gaussian smoothing (sigma in samples); NaN-aware normalised convolution, NaN samples stay NaN."""
    x = np.asarray(x, float)
    if len(x) < 3 or sigma < 0.3:
        return x.copy()
    k = max(1, min(int(np.ceil(4 * sigma)), (len(x) - 1) // 2))   # keep 'same' convolution at the signal length
    w = np.exp(-0.5 * (np.arange(-k, k + 1) / sigma) ** 2)
    m = np.isfinite(x)
    num = np.convolve(np.where(m, x, 0.0), w, mode="same")
    den = np.convolve(m.astype(float), w, mode="same")
    out = np.where(den > 1e-9, num / np.maximum(den, 1e-9), np.nan)
    out[~m] = np.nan
    return out


def sliding_extreme(x, w, fn):
    """Centred sliding max / min over w (odd) samples with edge padding."""
    xp = np.pad(x, w // 2, mode="edge")
    return fn(np.lib.stride_tricks.sliding_window_view(xp, w), axis=1)[:len(x)]


def opening(x, w):
    """Morphological opening (min then max over w samples): removes peaks narrower than w, keeps slow structure."""
    w = int(w) | 1
    return sliding_extreme(sliding_extreme(x, w, np.min), w, np.max)


def dilate(mask, k):
    """Binary dilation of a boolean series by k samples each side."""
    mask = np.asarray(mask, bool)
    if k <= 0:
        return mask.copy()
    n = len(mask)
    return (np.convolve(mask.astype(float), np.ones(2 * k + 1), mode="full")[k:k + n] > 0)      # 'same' misbehaves when k > n


def fft_bandpass(x, fs, lo, hi):
    """Zero-phase FFT band-pass of a gap-free series (lo/hi in Hz; lo=0 -> low-pass)."""
    X = np.fft.rfft(x - np.mean(x))
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    X[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(X, len(x))


def periodogram(x, fs, fmin, fmax, pad=16):
    """Hann-windowed, linearly detrended, zero-padded one-sided periodogram of a gap-free series -> (freqs Hz, power / Hz),
    normalised so that its integral over frequency is the variance (a sinusoid of amplitude A integrates to A^2 / 2)."""
    x = np.asarray(x, float)
    k = np.arange(len(x))
    x = x - np.polyval(np.polyfit(k, x, 1), k)
    w = np.hanning(len(x))
    nfft = int(2 ** np.ceil(np.log2(len(x) * pad)))
    p = 2.0 * np.abs(np.fft.rfft(x * w, nfft)) ** 2 / (fs * np.sum(w ** 2))
    f = np.fft.rfftfreq(nfft, 1.0 / fs)
    m = (f >= fmin) & (f <= fmax)
    return f[m], p[m]


def crossing_time(t, y, i, level, direction):
    """Sub-sample time where y crosses `level` going from index i in `direction` (+1 forward / -1 backward) until it
    leaves the region on the other side of the level; None if the series ends (or a gap starts) first."""
    j = i
    while 0 <= j + direction < len(y):
        a, b = y[j], y[j + direction]
        if np.isfinite(a) and np.isfinite(b) and (a - level) * (b - level) <= 0 and a != b:
            return float(t[j] + (t[j + direction] - t[j]) * (level - a) / (b - a))
        if not np.isfinite(b):
            return None
        j += direction
    return None


def describe(vals, unit=None):
    """Distribution summary used for every event list and every per-clip scalar: count, median and IQR (the headline),
    10th/90th percentiles, mean, std, min, max."""
    v = np.array([x for x in vals if x is not None and np.isfinite(x)], float)
    out = {"n": int(len(v))}
    if unit:
        out["unit"] = unit
    if len(v) == 0:
        return out
    p10, q1, med, q3, p90 = np.percentile(v, [10, 25, 50, 75, 90])
    out.update({"median": sig(med), "iqr": [sig(q1), sig(q3)], "p10": sig(p10), "p90": sig(p90), "mean": sig(v.mean()),
                "std": sig(v.std(ddof=1) if len(v) > 1 else 0.0), "min": sig(v.min()), "max": sig(v.max())})
    return out


def with_values(desc, vals, limit=100):
    """describe() output plus the raw event values (short lists only), which pooling over clips needs."""
    v = [float(x) for x in vals if x is not None and np.isfinite(x)]
    if 0 < len(v) <= limit:
        desc["values"] = [sig(x, 3) for x in v]
    return desc


def oscillation(x, fps, valid, fmin, fmax, min_seg_s, noise_hz=None):
    """Dominant oscillation of x over its valid, gap-free segments (>= min_seg_s): length-weighted mean Hann
    periodogram resampled on a common grid, band-limited to fmin..fmax Hz.
      period_s       the interior spectral peak if it is distinct, else the spectral-centroid period. Distinct = the typical
                     segment resolves it (two frequency cells <= a quarter of the band), the power within one cell of it is
                     at least 30 % of the band's, and it stands 2.5x above the band median: band-limited noise is not a rhythm
      amp            sinusoid-equivalent amplitude sqrt(2 * band variance), in the unit of x
      noise_amp      the same measure for the band 'noise_hz'..Nyquist rescaled to this band's width: what white
                     landmark jitter alone would give (only when noise_hz is set)
    None when no segment is long enough."""
    segs = [x[a:b] for a, b in runs(np.asarray(valid, bool) & np.isfinite(x)) if (b - a) >= max(min_seg_s * fps, 8)]
    if not segs:
        return None
    grid = np.linspace(fmin, fmax, 300)
    acc, tot = np.zeros_like(grid), 0.0
    zc, zt = 0, 0.0
    hf, hn = 0.0, 0
    for s in segs:
        f, p = periodogram(s, fps, fmin, fmax, pad=16)
        acc += np.interp(grid, f, p) * len(s)
        tot += len(s)
        bp = fft_bandpass(s - s.mean(), fps, fmin, fmax)
        zc += int(np.sum(np.diff(np.signbit(bp).astype(int)) != 0))
        zt += len(s) / fps
        if noise_hz is not None and noise_hz < fps / 2 - 0.5:
            hp = fft_bandpass(s - s.mean(), fps, noise_hz, fps / 2)
            hf += float(np.sum(hp ** 2))
            hn += len(hp)
    p = acc / tot
    var = float(_trapz(p, grid))
    ps = np.convolve(np.pad(p, 2, mode="edge"), np.ones(5) / 5, mode="valid")        # no zero-padding edge artefacts
    cell = 1.0 / float(np.median([len(s) / fps for s in segs]))                      # frequency resolution of a typical segment
    f_lo = max(fmin, cell)                                                           # a peak needs one full cycle in the segment
    pk = [i for i in range(2, len(grid) - 2) if grid[i] >= f_lo
          and ps[i] >= ps[i - 1] and ps[i] > ps[i + 1] and ps[i] >= ps[i - 2] and ps[i] > ps[i + 2]]
    i_pk = max(pk, key=lambda i: ps[i]) if pk else None
    prom = float(ps[i_pk] / max(np.median(ps), 1e-30)) if i_pk is not None else 0.0
    centroid = float((grid * p).sum() / p.sum()) if p.sum() > 0 else float(grid.mean())
    share = 0.0
    if i_pk is not None and var > 0:
        near = np.abs(grid - grid[i_pk]) <= cell
        share = float(_trapz(p[near], grid[near]) / var) if near.sum() > 1 else 0.0
    distinct = bool(i_pk is not None and 2 * cell <= 0.25 * (fmax - fmin) and share >= 0.3 and prom >= 2.5)
    f_dom = float(grid[i_pk]) if distinct else centroid
    out = {"period_s": 1.0 / f_dom, "peak_distinct": bool(distinct),
           "peak_period_s": float(1 / grid[i_pk]) if i_pk is not None else None, "peak_prominence": prom,
           "centroid_period_s": 1.0 / centroid, "amp": float(np.sqrt(2 * var)), "std": float(np.sqrt(var)),
           "zero_cross_period_s": (2 * zt / zc) if zc else None, "n_segments": len(segs),
           "seconds": float(sum(len(s) for s in segs) / fps)}
    if hn:
        # white noise: power per Hz is flat, so the share inside the band is (fmax - fmin) / (Nyquist - noise_hz)
        nvar = (hf / hn) * (fmax - fmin) / (fps / 2 - noise_hz)
        out["noise_amp"] = float(np.sqrt(2 * nvar))
    return out


def ev(vals, unit=None, limit=400):
    """describe() plus the raw event values: one event list of a clip's summary."""
    return with_values(describe(vals, unit), vals, limit)


def scalar(x, unit, n, basis, **kw):
    """A measured number with its sample count and what was counted ('frames', 'events', 'clips')."""
    return {"value": sig(x), "unit": unit, "n": int(n), "n_basis": basis, **kw}
