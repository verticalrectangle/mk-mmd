"""Mastering a clip's audio for small speakers (`mk post`, `[audio.master]`; docs/design.md: Post). A phone's speakers
play little below about 250 Hz, so a mix whose kick or 808 lives at 50-150 Hz loses its booms there while big speakers
thump. The fix is a mastering engineer's: give the booms overtones a phone can play (the low band saturated and
band-passed above it: the ear hears the missing fundamental from its harmonics), lift their attack, duck the rest of the
mix under each of them, cut the sub a phone wastes its travel on, and bring the whole to a loudness under a ceiling.
The booms are a lane of the project's hits file (`[audio] hits`, clip seconds), so it all happens on them and nowhere
else. numpy only: the filters are zero-phase masks in the frequency domain and the limiter looks ahead.

`master(x, sr, hits, cfg)` -> (y, report); `loudness(x, sr)` is ITU-R BS.1770-4 integrated loudness (LUFS);
`boom_contrast(x, sr, hits)` how far the booms stand out through a phone's speakers (`phone`)."""
import math

import numpy as np

SR = 48000                    # the rate everything is read and mastered at (BS.1770's K-weighting is given for it)
DEFAULTS = {"lane": "808", "split": 160.0, "harmonics": -3.0, "drive": 4.0, "tail": 0.40, "knock": 6.0, "duck": 4.0,
            "release": 0.15, "highpass": 35.0, "loudness": -12.0, "ceiling": -1.0}
KEYS = tuple(DEFAULTS)
OVERTONES = (1.1, 1500.0)     # the overtones kept: from 1.1 x `split` to this (Hz)
KNOCK_BAND = (1500.0, 5000.0)  # Hz: a boom's attack
KNOCK_S = 0.025               # s: the attack's lift fades over this
DUCK_HOLD = 0.04              # s: the rest of the mix stays down this long on a boom before it comes back over `release`
LOOKAHEAD = 0.005             # s: the limiter turns down this far ahead of a peak ...
LIMIT_RELEASE = 0.08          # s: ... and comes back over this
PHONE = (250.0, 14000.0)      # Hz: what a phone's speakers play
K_SHELF = ((1.53512485958697, -2.69169618940638, 1.19839281085285), (1.0, -1.69065929318241, 0.73248077421585))
K_RLB = ((1.0, -2.0, 1.0), (1.0, -1.99004745483398, 0.99007225036621))     # BS.1770's K-weighting at 48 kHz


class MasterError(ValueError):
    """A bad `[audio.master]`, or hits it cannot use."""


def settings(cfg):
    """`[audio.master]` (a table, or true for the defaults) with its defaults (DEFAULTS); MasterError names an unknown
    key or a bad value."""
    if cfg is True:
        cfg = {}
    if not isinstance(cfg, dict):
        raise MasterError(f"[audio] master = {cfg!r}: a table of {', '.join(KEYS)} (or true for the defaults)")
    cfg = dict(cfg)
    bad = sorted(set(cfg) - set(KEYS))
    if bad:
        raise MasterError(f"[audio.master]: unknown key{'s' if len(bad) > 1 else ''} {', '.join(map(repr, bad))} "
                          f"(known: {', '.join(KEYS)})")
    c = dict(DEFAULTS, **cfg)
    for k in KEYS[1:]:
        if not isinstance(c[k], (int, float)) or isinstance(c[k], bool):
            raise MasterError(f"[audio.master] {k} = {c[k]!r}: a number")
        c[k] = float(c[k])
    if not (20.0 <= c["split"] <= 400.0 and c["drive"] > 0 and c["tail"] > 0 and c["release"] > 0
            and c["duck"] >= 0 and c["knock"] >= 0 and c["highpass"] >= 0 and c["ceiling"] <= 0):
        raise MasterError("[audio.master]: split 20..400 Hz, drive, tail and release above 0, duck, knock and highpass "
                          "not below 0, ceiling at most 0 dBFS")
    return c


def boom_times(hits, lane):
    """[(t, weight)] of a hits file's lane (each entry a number or {t, db}: weight 10^(db/20), at most 1)."""
    if lane not in hits or not isinstance(hits[lane], list) or not hits[lane]:
        have = sorted(k for k, v in hits.items() if isinstance(v, list))
        raise MasterError(f"[audio.master] lane = {lane!r}: the hits file has no such lane (have {', '.join(have)})")
    out = []
    for h in hits[lane]:
        t, db = (h.get("t"), h.get("db", 0.0)) if isinstance(h, dict) else (h, 0.0)
        out.append((float(t), min(1.0, 10.0 ** (float(db) / 20.0))))
    return sorted(out)


def _band(f, lo=None, hi=None, order=4):
    """Butterworth magnitude of a band from `lo` to `hi` Hz (either open) at frequencies f."""
    m = np.ones_like(f)
    if lo:
        r = np.maximum(f, 1e-9) / lo
        m = m * r ** order / np.sqrt(1.0 + r ** (2 * order))
    if hi:
        m = m / np.sqrt(1.0 + (f / hi) ** (2 * order))
    return m


def _filtered(x, sr, gain):
    """x (n,) or (n, c) through the zero-phase response gain(frequencies), padded against wrap-around."""
    pad = int(0.5 * sr)
    n = len(x) + 2 * pad
    N = 1 << (n - 1).bit_length()
    X = np.fft.rfft(np.pad(x, [(pad, pad)] + [(0, 0)] * (x.ndim - 1)), N, axis=0)
    g = gain(np.fft.rfftfreq(N, 1.0 / sr))
    return np.fft.irfft(X * (g[:, None] if x.ndim > 1 else g), N, axis=0)[pad:pad + len(x)]


def band(x, sr, lo=None, hi=None, order=4):
    return _filtered(x, sr, lambda f: _band(f, lo, hi, order))


def phone(x, sr):
    """x as a phone's speakers play it: a 4th-order cut below PHONE[0] Hz and a 2nd-order one above PHONE[1]."""
    return _filtered(x, sr, lambda f: _band(f, PHONE[0], None, 4) * _band(f, None, PHONE[1], 2))


def _biquad(f, sr, b, a):
    z = np.exp(-2j * np.pi * f / sr)
    return np.abs((b[0] + b[1] * z + b[2] * z * z) / (a[0] + a[1] * z + a[2] * z * z))


def loudness(x, sr=SR):
    """Integrated loudness of stereo (or mono) x in LUFS, ITU-R BS.1770-4: K-weighting, 400 ms blocks every 100 ms,
    the absolute gate at -70 LUFS and the relative gate 10 LU under the absolute-gated loudness. -inf for silence."""
    if sr != SR:
        raise MasterError(f"loudness is measured at {SR} Hz, not {sr}")
    x = np.asarray(x, float).reshape(len(x), -1)
    k = _filtered(x, sr, lambda f: _biquad(f, sr, *K_SHELF) * _biquad(f, sr, *K_RLB))
    blk, hop = int(0.4 * sr), int(0.1 * sr)
    if len(k) < blk:
        z = (k ** 2).mean(0)[None]
    else:
        c = np.concatenate([np.zeros((1, k.shape[1])), np.cumsum(k ** 2, 0)])
        starts = np.arange(0, len(k) - blk + 1, hop)
        z = (c[starts + blk] - c[starts]) / blk                   # (blocks, channels) mean squares
    lj = -0.691 + 10.0 * np.log10(np.maximum(z.sum(1), 1e-30))
    keep = lj > -70.0
    if not keep.any():
        return -math.inf
    rel = -0.691 + 10.0 * math.log10(z[keep].mean(0).sum()) - 10.0
    keep &= lj > rel
    return -0.691 + 10.0 * math.log10(z[keep].mean(0).sum())


def _pulses(n, sr, times, rise, hold, fall):
    """(n,) the largest of one pulse per (t, weight): up over `rise` s to the weight at t, held `hold` s, down over `fall` s
    (smoothstep both ways)."""
    out = np.zeros(n)
    for t, w in times:
        i0, i1 = max(0, int((t - rise) * sr)), min(n, int((t + hold + fall) * sr) + 1)
        if i1 <= i0:
            continue
        tt = np.arange(i0, i1) / sr - t
        u = np.clip((tt + rise) / max(rise, 1e-9), 0.0, 1.0)
        v = np.clip((tt - hold) / fall, 0.0, 1.0)
        p = w * (u * u * (3 - 2 * u)) * (1.0 - v * v * (3 - 2 * v))
        out[i0:i1] = np.maximum(out[i0:i1], p)
    return out


def limit(x, sr, ceiling_db):
    """x under a look-ahead limiter: no sample above `ceiling_db` dBFS. The gain starts down LOOKAHEAD s before a peak
    (the smallest gain any sample of the window ahead needs, averaged over the window, so it ramps and still gets there)
    and comes back over LIMIT_RELEASE s. Returns (y, the most it turned down in dB)."""
    c = 10.0 ** (ceiling_db / 20.0)
    a = np.abs(x).max(1) if x.ndim > 1 else np.abs(x)
    need = np.minimum(1.0, c / np.maximum(a, 1e-12))
    if need.min() >= 1.0:
        return x.copy(), 0.0
    L = max(1, int(LOOKAHEAD * sr))
    ext = np.concatenate([np.ones(L), need, np.ones(L)])
    ahead = np.lib.stride_tricks.sliding_window_view(ext, L + 1).min(1)   # [j]: the least any of ext[j .. j + L] needs
    cs = np.concatenate([[0.0], np.cumsum(ahead)])
    g = (cs[L + 1:L + 1 + len(need)] - cs[:len(need)]) / (L + 1)  # [i]: the mean of ahead[i .. i + L], each <= need[i]
    blk = max(1, int(0.001 * sr))                                   # the release, block by block (1 ms)
    nb = -(-len(g) // blk)
    bmin = np.concatenate([g, np.ones(nb * blk - len(g))]).reshape(nb, blk).min(1)
    k = 1.0 - math.exp(-blk / (LIMIT_RELEASE * sr))
    rel = np.empty(nb)
    r = 1.0
    for i, m in enumerate(bmin):
        r = min(m, r + (1.0 - r) * k)
        rel[i] = r
    g = np.minimum(g, np.interp(np.arange(len(g)), np.arange(nb) * blk + blk // 2, rel))
    y = x * (g[:, None] if x.ndim > 1 else g)
    return y, round(float(-20.0 * math.log10(max(g.min(), 1e-12))), 2)


def boom_contrast(x, sr, times, win=0.2):
    """dB: how far the booms stand out through a phone's speakers, the median over the hits of the level just after each
    (`win` s) against the level just before it."""
    p = phone(np.asarray(x, float).reshape(len(x), -1), sr).mean(1)
    out = []
    for t, _ in times:
        i, w = int(t * sr), int(win * sr)
        if i - w < 0 or i + w > len(p):
            continue
        after, before = np.sqrt((p[i:i + w] ** 2).mean()), np.sqrt((p[i - w:i] ** 2).mean())
        out.append(20.0 * math.log10(max(after, 1e-12) / max(before, 1e-12)))
    return round(float(np.median(out)), 2) if out else None


def master(x, sr, hits, cfg):
    """x (n, 2) float at `sr` mastered for small speakers by `cfg` (`settings`) on the booms of `hits` (the hits file):
    the low band under `split` Hz around each boom saturated (`drive`) into overtones up to OVERTONES[1] Hz, mixed in
    `harmonics` dB against that band over `tail` s; the boom's attack (KNOCK_BAND) lifted `knock` dB, fading over KNOCK_S;
    the rest of the mix (above `split`) ducked `duck` dB, held DUCK_HOLD s and back over `release` s; everything under
    `highpass` Hz cut; then the whole brought to `loudness` LUFS under a `ceiling` dBFS limiter. Stronger hits ({t, db})
    get more of the attack and the duck. Returns (y, report)."""
    c = settings(cfg)
    times = boom_times(hits, c["lane"])
    x = np.asarray(x, float).reshape(len(x), -1)
    n = len(x)
    low = band(x, sr, hi=c["split"])
    rest = x - low
    gate = _pulses(n, sr, [(t, 1.0) for t, _ in times], 0.003, 0.05, c["tail"])
    mono = low.mean(1) * gate
    peak = float(np.percentile(np.abs(mono[gate > 0.5]), 99.5)) if (gate > 0.5).any() else 0.0
    over = np.zeros(n)
    if peak > 1e-9:
        s = np.tanh(c["drive"] * mono / peak)
        over = band(s + 0.35 * s * s, sr, lo=OVERTONES[0] * c["split"], hi=OVERTONES[1]) * gate
        on = gate > 0.5
        r_over, r_low = np.sqrt((over[on] ** 2).mean()), np.sqrt((mono[on] ** 2).mean())
        over *= 10.0 ** (c["harmonics"] / 20.0) * r_low / max(r_over, 1e-12)
    knock = band(x, sr, lo=KNOCK_BAND[0], hi=KNOCK_BAND[1]) * (
        _pulses(n, sr, times, 0.001, 0.0, KNOCK_S) * (10.0 ** (c["knock"] / 20.0) - 1.0))[:, None]
    duck = 1.0 - (1.0 - 10.0 ** (-c["duck"] / 20.0)) * _pulses(n, sr, times, 0.004, DUCK_HOLD, c["release"])
    y = low + rest * duck[:, None] + over[:, None] + knock
    if c["highpass"] > 0:
        y = band(y, sr, lo=c["highpass"])
    before = loudness(x, sr) if sr == SR else None
    lu = loudness(y, sr) if sr == SR else None
    if lu is not None and math.isfinite(lu):
        y = y * 10.0 ** ((c["loudness"] - lu) / 20.0)
    y, cut = limit(y, sr, c["ceiling"])
    after = loudness(y, sr) if sr == SR else None
    rep = {"lane": c["lane"], "booms": len(times),
           "loudness_in": None if before is None else round(before, 2),
           "loudness_out": None if after is None else round(after, 2),
           "peak_in_dbfs": round(20.0 * math.log10(max(float(np.abs(x).max()), 1e-12)), 2),
           "peak_out_dbfs": round(20.0 * math.log10(max(float(np.abs(y).max()), 1e-12)), 2), "limiter_db": cut,
           "boom_phone_db": {"before": boom_contrast(x, sr, times), "after": boom_contrast(y, sr, times)}}
    return y, rep
