"""Pooled statistics -> performance parameters.

`recommend` is shaped like a `[perform.<cast>]` table (see mkmmd/blender/build/perform.py) and only holds fields the
clips actually support; `why` has one entry per field with the reason, sample count and confidence; `skipped` says
why a parameter has no recommendation. Values are typical (median or pooled rate); `range` is the randomisation
range (10th-90th percentile with >= 8 samples, else min-max)."""
import math

import numpy as np

from .dsp import sig

MIN_FACE_SECONDS = 20.0       # less usable face time than this gives no blink or gaze rate
SHOULDER_WIDTH_M = 0.38       # to turn a shoulder rise into a chest pitch
CHEST_LEVER_M = 0.25          # distance from the chest pivot to the shoulders
NOISE_FACTOR = 1.5            # an oscillation below this many times the landmark jitter is 'almost still'
HEAD_FOLLOW_FRAMES = 3.0      # the performance stage's followers (mkmmd/blender/build/perform.py: lowpass(blend(W), 3.0) for the
EYE_FOLLOW_FRAMES = 1.5       # head, lowpass(blend(W_eye), 1.5) for the eyes): two cascaded exponentials of this many frames


def _conf(n, tentative=False):
    return "low" if tentative or n < 8 else ("medium" if n < 20 else "high")


def _rng(desc):
    if not desc or desc.get("n", 0) < 2 or desc.get("min") is None:
        return None
    return [desc["p10"], desc["p90"]] if desc["n"] >= 8 else [desc["min"], desc["max"]]


def _set(rec, path, value):
    cur = rec
    parts = path.split(".")
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = sig(value, 3)


def blink_curve_fwhm_ratio():
    """FWHM / duration of the performance stage's blink curve (so a measured width can be turned into its duration)."""
    from ..core import perform as PF
    ts = np.linspace(-0.05, 1.05, 4401)
    c = PF.blink_curve(ts, [(0.0, 1.0)])
    above = ts[c >= 0.5]
    return float(above.max() - above.min())


def turn_time(rise_s, fps, tau_frames):
    """10 -> 90 % time (s) of a smoothstep envelope of length rise_s after the performance stage's follower."""
    from ..core import perform as PF
    ts = np.arange(int(round((rise_s + 12 * tau_frames / fps + 2) * fps))) / fps
    y = PF.lowpass(PF.smooth(ts / max(rise_s, 1e-3)), tau_frames)
    return float(np.interp(0.9, y, ts) - np.interp(0.1, y, ts))


def rise_for_turn(t_1090, fps, tau_frames, lo=0.1, hi=3.0):
    """The envelope rise (s) whose turn, after the follower, takes t_1090 s from 10 to 90 %; `lo` when the follower alone is
    already that slow. Bisection on the actual curves of mkmmd.core.perform."""
    if turn_time(lo, fps, tau_frames) >= t_1090:
        return lo
    a, b = lo, hi
    for _ in range(30):
        m = (a + b) / 2
        a, b = (m, b) if turn_time(m, fps, tau_frames) < t_1090 else (a, m)
    return (a + b) / 2


class _Rec:
    def __init__(self):
        self.rec, self.why, self.skipped = {}, {}, {}

    def add(self, path, value, text, n, n_basis, rng=None, tentative=False, note=None):
        _set(self.rec, path, value)
        self.why[path] = {"why": text + (f" [{note}]" if note else ""), "range": [sig(x, 3) for x in rng] if rng else None, "n": int(n),
                          "n_basis": n_basis, "confidence": _conf(n, tentative)}

    def skip(self, path, reason):
        self.skipped[path] = reason


def recommend(P, results, fps=30.0):
    R = _Rec()
    n_face = len(P["clips"]["with_face"])
    face_s = P["clips"]["seconds"]["face_tracked"] or 0.0

    # ---------------------------------------------------------------- blink
    B = P["blinks"]
    sp = B["spontaneous_rate_per_min"]
    if n_face == 0:
        for k in ("blink.per_min", "lids", "gaze", "head_share", "eye_max", "nod"):
            R.skip(k, "no clip has a usable face (tracked in >= 50 % of the frames and >= 100 px wide)")
    else:
        if (sp.get("observed_seconds") or 0) < MIN_FACE_SECONDS:
            R.skip("blink.per_min", f"only {sp.get('observed_seconds') or 0:.0f} s of usable face time outside gaze shifts (needs {MIN_FACE_SECONDS:g} s)")
        elif sp["n"] == 0:
            R.skip("blink.per_min", f"no blink in {sp['observed_seconds']:.0f} s of usable face time: the rate is below ~{60 / sp['observed_seconds']:.0f}/min "
                                    "(staring, or eyes too small): keep the performance default")
        else:
            R.add("blink.per_min", sp["value"],
                  f"pooled blinks outside gaze shifts: {sp['n']} in {sp['observed_seconds']:.0f} s over {sp['n_clips']} clip(s) "
                  "(blinks within 0.4 s of a gaze shift are left out because the performance stage adds a blink to each gaze event)",
                  sp["n"], "events", _rng(B["spontaneous_rate_by_clip"]))
            f = B["duration_fwhm"]
            if f.get("n", 0) >= 3:
                ratio = blink_curve_fwhm_ratio()
                R.add("blink.duration_s", f["median"] / ratio,
                      f"median blink width at half height {f['median']:.3f} s, divided by {ratio:.3f} (the blink curve's FWHM per second of duration); "
                      "the performance stage does not read this key yet (BLINK_S)", f["n"], "events",
                      [f["p10"] / ratio, f["p90"] / ratio] if f["n"] >= 8 else [f["min"] / ratio, f["max"] / ratio])
        L = P["lids"]["rest_level"]
        if L.get("n", 0):
            R.add("lids", float(np.clip(L["median"], 0.0, 0.6)),
                  f"median resting lid closure score away from blinks (0 open .. 1 shut): {L['median']:.2f} over {L['n']} clip(s); eyes shut for "
                  f"{100 * (P['lids']['eyes_shut_share'].get('median') or 0):.0f} % of the time (median)", L["n"], "clips", _rng(L))
        else:
            R.skip("lids", "no clip had a resting lid level")

        # ---------------------------------------------------------------- gaze
        G = P["gaze"]
        rate = G["shift_rate_per_10s"]
        if (rate.get("observed_seconds") or 0) < MIN_FACE_SECONDS:
            R.skip("gaze", f"only {rate.get('observed_seconds') or 0:.0f} s of tracked gaze (needs {MIN_FACE_SECONDS:g} s; gaze is undefined while the eyes are shut)")
        else:
            by = G["shift_rate_by_clip"]
            rng = [by["min"] / 2, by["max"] / 2] if by.get("n", 0) >= 2 else None
            R.add("gaze.rate_per_10s", rate["value"] / 2,
                  f"{rate['n']} gaze shifts in {rate['observed_seconds']:.0f} s over {rate['n_clips']} clip(s) = {rate['value']:.2f} shifts / 10 s; a "
                  "performance gaze event is an out-and-back pair of shifts, hence half", rate["n"], "events", rng)
            if G["rise_gaze"].get("n", 0):
                hr, gr = G["head_rise"], G["rise_gaze"]
                use, what, tau = (hr, "head", HEAD_FOLLOW_FRAMES) if hr.get("n", 0) >= 3 else (gr, "gaze", EYE_FOLLOW_FRAMES)
                rise = rise_for_turn(use["median"], fps, tau)
                follower = turn_time(0.1, fps, tau)
                R.add("gaze.rise_s", rise,
                      f"median {what} 10-90 % time {use['median']:.2f} s over {use['n']} shifts; the performance stage's own follower already takes "
                      f"{follower:.2f} s at {fps:g} fps, so the smoothstep envelope needs this rise to reproduce the turn",
                      use["n"], "events", [rise_for_turn(x, fps, tau) for x in _rng(use)] if _rng(use) else None,
                      note="at the 0.1 s floor: the follower alone is as slow as the measured turns" if rise <= 0.1 else None)
                h = G["hold"]
                if h.get("n", 0):
                    R.add("gaze.hold_s", h["median"] + rise,
                          f"median time spent in a fixation between two shifts {h['median']:.2f} s over {h['n']} holds, plus the rise (the performance 'hold' "
                          "runs from the start of the turn to the start of the return)", h["n"], "events",
                          [x + rise for x in _rng(h)] if _rng(h) else None)
                else:
                    R.skip("gaze.hold_s", "no fixation was bounded by two shifts")
            else:
                R.skip("gaze.rise_s", "no gaze shift was measured")
                R.skip("gaze.hold_s", "no gaze shift was measured")
            hs = G["head_share"]
            if hs.get("n", 0) >= 3:
                R.add("head_share", float(np.clip(hs["median"], 0.0, 1.0)),
                      f"median share of the settled gaze change taken by the head over {hs['n']} shifts of >= 10 deg", hs["n"], "events", _rng(hs))
            else:
                R.skip("head_share", f"{hs.get('n', 0)} shift(s) of >= 10 deg with a settled head (needs 3)")
            e = G["eye_in_head_p95"]
            if e.get("n", 0):
                R.add("eye_max", float(np.clip(e["median"], 8.0, 40.0)),
                      f"95th percentile of the eye-in-head angle (eyes open), median over {e['n']} clip(s); full-scale blendshape score taken as 30 deg",
                      e["n"], "clips", _rng(e), note="blendshape angles are approximate")

        # ---------------------------------------------------------------- head oscillations
        H = P["head"]
        na, npd, nn = H["nod_amp"], H["nod_period"], H["nod_noise"]
        if na.get("n", 0) and npd.get("n", 0):
            near = na["median"] < NOISE_FACTOR * (nn.get("median") or 0)
            R.add("nod.deg", na["median"], f"sinusoid-equivalent amplitude of head pitch in the 0.2-1.5 Hz band away from gaze shifts, median over {na['n']} clip(s)",
                  na["n"], "clips", _rng(na), note=f"close to landmark jitter ({nn.get('median')} deg): almost still" if near else None)
            R.add("nod.period", npd["median"], f"dominant period of that band (spectral peak if distinct, else centroid), median over {npd['n']} clip(s)",
                  npd["n"], "clips", _rng(npd))
        else:
            R.skip("nod", "no clip had long enough stretches of tracked head pitch away from gaze shifts (3 s)")
        ba, bn = H["bob_amp"], H["bob_noise"]
        if ba.get("n", 0) and ba["median"] >= 2 * (bn.get("median") or 0) and ba["median"] > 0.3:
            R.add("bob.deg", ba["median"], f"head pitch amplitude in the 1-3.5 Hz band (period {H['bob_period'].get('median')} s), median over {ba['n']} clip(s): "
                  "rhythmic head bobbing well above landmark jitter (the beat comes from the timeline, not from the clip)", ba["n"], "clips", _rng(ba))
        else:
            R.skip("bob", "no rhythmic head motion above landmark jitter in the 1-3.5 Hz band")

        # ---------------------------------------------------------------- mouth
        M = P["mouth"]
        talking = {c: v for c, v in (M["moving_share"].get("by_clip") or {}).items() if v is not None and v >= 0.15}
        pk = M["peak_p90_when_moving"]
        vals = [v for c, v in (pk.get("by_clip") or {}).items() if c in talking and v is not None]
        if vals:
            arr = np.array(vals)
            R.add("sing.mouth", float(np.median(arr)),
                  f"90th percentile of jawOpen over frames where the mouth moves, median over the {len(vals)} clip(s) with the mouth moving >= 15 % of the time; "
                  "used 1:1 as the vowel morphs' peak weight", len(vals), "clips", [float(arr.min()), float(arr.max())] if len(arr) >= 2 else None)
        else:
            R.skip("sing.mouth", "no clip shows the mouth moving (talking / singing) for at least 15 % of the time")

    # ---------------------------------------------------------------- body
    Bd = P["body"]
    sa, sp_, sn = Bd["shoulder_sway_amp"], Bd["shoulder_sway_period"], Bd["shoulder_sway_noise"]
    if sa.get("n", 0):
        near = sa["median"] < NOISE_FACTOR * (sn.get("median") or 0)
        R.add("sway.deg", sa["median"], f"sinusoid-equivalent amplitude of the shoulder-line angle in the 0.15-1 Hz band, median over {sa['n']} clip(s)",
              sa["n"], "clips", _rng(sa), note=f"close to pose-landmark jitter ({sn.get('median')} deg): almost still" if near else None)
        R.add("sway.period", sp_["median"], f"dominant period of that band, median over {sp_['n']} clip(s)", sp_["n"], "clips", _rng(sp_))
    else:
        R.skip("sway", "no clip shows both shoulders for 60 % of its length")
    br, ba_ = Bd["breathing_rate"], Bd["breathing_amp"]
    if br.get("n", 0):
        amp_m = ba_["median"] / 100.0 * SHOULDER_WIDTH_M
        R.add("breath.per_min", br["median"], f"spectral peak (0.15-0.7 Hz) of the shoulder-midpoint height, median over {br['n']} clip(s) where it was distinct",
              br["n"], "clips", _rng(br), tentative=True)
        R.add("breath.deg", math.degrees(math.atan2(amp_m, CHEST_LEVER_M)),
              f"shoulder rise {ba_['median']:.2f} % of the shoulder width = {1000 * amp_m:.1f} mm on a {SHOULDER_WIDTH_M} m wide pair of shoulders, turned into a chest "
              f"pitch about a pivot {CHEST_LEVER_M} m below them", ba_["n"], "clips", None, tentative=True)
    else:
        R.skip("breath", "breathing is not resolvable in these clips (no distinct 0.15-0.7 Hz peak in the shoulder-midpoint height): keep the performance defaults")
    return R.rec, R.why, R.skipped
