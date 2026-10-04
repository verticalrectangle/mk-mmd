"""Body: shoulder sway and a breathing proxy from the pose landmarks, hand speed and stillness from the hand landmarks.
All positions are camera-stabilised (the cumulative global translation estimated during tracking is subtracted)."""
import numpy as np

from . import dsp
from .dsp import gsmooth, interp_gaps, runs, sig

HAND_WIDTH_MM = 80.0       # index-MCP to pinky-MCP breadth used to scale hand motion to millimetres (assumption, +-10 %)
REST_HW_S = 0.25           # palm speed below this = at rest (hand widths / s)
GESTURE_PEAK_HW_S = 0.6    # a gesture: palm speed above 0.2 hw/s with a peak of at least this ...
GESTURE_MIN_TRAVEL_HW = 0.4  # ... and a net travel of at least this many hand widths
PALM = [0, 5, 9, 13, 17]   # wrist + the four MCP knuckles
L_SHOULDER, R_SHOULDER = 11, 12
SHOULDER_BREATH_BAND = (0.15, 0.7)   # Hz
SHOULDER_SWAY_BAND = (0.15, 1.0)
BREATH_MAX_PCT = 1.5         # a breath lifts the shoulders by well under this share of their width


# ============================================================================================ shoulders
def shoulders(pose_lm, W, H, cam):
    """Shoulder-line angle (deg, an undirected line: folded into -90..90 so a person seen from behind does not wrap; + =
    subject-left shoulder lower on screen) and shoulder-midpoint height in shoulder widths (camera-stabilised), NaN where
    either shoulder is not confidently visible."""
    vis = np.minimum(pose_lm[:, L_SHOULDER, 3], pose_lm[:, R_SHOULDER, 3])
    p11 = pose_lm[:, L_SHOULDER, :2] * [W, H] - cam
    p12 = pose_lm[:, R_SHOULDER, :2] * [W, H] - cam
    d = p11 - p12
    width = np.hypot(*d.T)
    ok = (vis > 0.5) & (width > 0.06 * W)
    ang = np.where(ok, (np.degrees(np.arctan2(d[:, 1], d[:, 0])) + 90.0) % 180.0 - 90.0, np.nan)
    mid_y = np.where(ok, (p11[:, 1] + p12[:, 1]) / 2 / np.maximum(width, 1), np.nan)
    return ang, mid_y, np.where(ok, width, np.nan)


def analyze_body(pose_lm, W, H, fps, cam, usable):
    """Shoulder sway and breathing proxy. `usable` masks frames that may be used (no cuts). None when the shoulders
    are visible in less than 60 % of the clip (face close-ups)."""
    ang, mid_y, width = shoulders(pose_lm, W, H, cam)
    ok = np.isfinite(ang) & usable
    ang, mid_y = np.where(ok, ang, np.nan), np.where(ok, mid_y, np.nan)
    if ok.mean() < 0.6:
        return None
    res = {"coverage": {"value": sig(float(ok.mean())), "unit": "fraction of frames"},
           "shoulder_width_px": {"value": sig(np.nanmedian(width[ok])), "unit": "px"}}
    sw = dsp.oscillation(ang, fps, ok, *SHOULDER_SWAY_BAND, 6.0, noise_hz=3.0)
    if sw:
        idx = np.flatnonzero(ok)
        v = ang[ok] - np.polyval(np.polyfit(idx, ang[ok], 1), idx)
        sw["std_linear_detrended"] = float(v.std(ddof=1))
        sw["p5_p95"] = [float(np.percentile(v, 5)), float(np.percentile(v, 95))]
    res["shoulder_sway"] = _osc(sw, "deg", "shoulder-line angle in the image plane, " + _band(SHOULDER_SWAY_BAND))
    # breathing: shoulder-midpoint height (shoulder widths), slow posture drift removed; only ever 'tentative'
    hp = mid_y - gsmooth(mid_y, 2.0 * fps)
    br = dsp.oscillation(np.where(ok, hp, np.nan), fps, ok, *SHOULDER_BREATH_BAND, 8.0)
    if br:
        rate = 60.0 / br["peak_period_s"] if br["peak_period_s"] else None
        out = {"rate_per_min": sig(rate), "amp_pct_shoulder_width": sig(100 * br["amp"]),
               "peak_distinct": br["peak_distinct"], "peak_prominence": sig(br["peak_prominence"]),
               "seconds": sig(br["seconds"]),
               "visible": "tentative" if (br["peak_distinct"] and br["peak_prominence"] >= 4.0 and rate and 9 <= rate <= 35
                                          and 100 * br["amp"] <= BREATH_MAX_PCT) else False,
               "definition": "spectral peak (0.15-0.7 Hz) of the shoulder-midpoint height, plausible only when distinct (prominence >= 4), "
                             f"9-35 /min and <= {BREATH_MAX_PCT:g} % of the shoulder width (more is body motion); posture and arm motion mask "
                             "breathing at 720p, so 'tentative' is the best it gets"}
        res["breathing"] = out
    else:
        res["breathing"] = None
    return res


def _band(b):
    return f"{b[0]:g}-{b[1]:g} Hz band"


def _osc(o, unit, what):
    """JSON form of an oscillation() result."""
    if not o:
        return None
    out = {"amp": sig(o["amp"]), "unit": unit, "period_s": sig(o["period_s"]), "peak_distinct": o["peak_distinct"],
           "peak_period_s": sig(o["peak_period_s"]), "centroid_period_s": sig(o["centroid_period_s"]),
           "zero_cross_period_s": sig(o["zero_cross_period_s"]), "std": sig(o["std"]),
           "noise_amp": sig(o.get("noise_amp")), "seconds": sig(o["seconds"]), "segments": o["n_segments"],
           "definition": what + "; amp = sinusoid-equivalent amplitude sqrt(2 * band variance), noise_amp = what "
                                "landmark jitter alone would give in this band"}
    for k in ("std_linear_detrended", "p5_p95"):
        if k in o:
            out[k] = sig(o[k]) if not isinstance(o[k], list) else [sig(x) for x in o[k]]
    return out


# ============================================================================================ hands
def associate_hands(hand_lm, W, H, fps):
    """The landmarker returns up to two hands per frame in arbitrary order: chain them into two continuous slots by
    palm-centre continuity (stale memory after 1 s falls back to left-to-right order).
    Returns (n, 2) int: detection index feeding each slot per frame, -1 where the slot has no hand."""
    n = hand_lm.shape[0]
    present = np.isfinite(hand_lm[:, :, 0, 0])
    palm = hand_lm[:, :, PALM, :2].mean(2) * [W, H]
    out = np.full((n, 2), -1, np.int8)
    last, last_i, left_slot = [None, None], [-10 ** 9, -10 ** 9], 0

    def cost(s, p, i):
        if last[s] is None or i - last_i[s] > fps:
            return 0.5 * W
        return float(np.hypot(*(p - last[s])))

    for i in range(n):
        dets = [k for k in range(2) if present[i, k]]
        if len(dets) == 2:
            c_id = cost(0, palm[i, 0], i) + cost(1, palm[i, 1], i)
            c_sw = cost(0, palm[i, 1], i) + cost(1, palm[i, 0], i)
            if abs(c_id - c_sw) < 1e-6:
                lo = 0 if palm[i, 0, 0] <= palm[i, 1, 0] else 1          # tie: leftmost detection takes the left slot
                assign = {lo: left_slot, 1 - lo: 1 - left_slot}
            else:
                assign = {0: 0, 1: 1} if c_id < c_sw else {0: 1, 1: 0}
            left_slot = assign[0] if palm[i, 0, 0] <= palm[i, 1, 0] else assign[1]
        elif len(dets) == 1:
            k, p = dets[0], palm[i, dets[0]]
            c0, c1 = cost(0, p, i), cost(1, p, i)
            if abs(c0 - c1) < 1e-6:
                other = last[1 - left_slot]
                s = left_slot if (other is None or p[0] <= other[0]) else 1 - left_slot
            else:
                s = 0 if c0 < c1 else 1
            assign = {k: s}
        else:
            continue
        for k, s in assign.items():
            out[i, s] = k
            last[s], last_i[s] = palm[i, k], i
    return out


def by_slot(arr, slot_det):
    """Reorder a per-detection array (n, 2, ...) into slot order using associate_hands' map (NaN where empty)."""
    out = np.full(arr.shape, np.nan, arr.dtype)
    for s in range(2):
        for k in range(2):
            m = slot_det[:, s] == k
            out[m, s] = arr[m, k]
    return out


def smooth_xy(p, fps, sigma_s, max_gap_s=0.3):
    """NaN-gap-bridged Gaussian smoothing of an (n, 2) trajectory."""
    return np.column_stack([gsmooth(interp_gaps(p[:, j], int(round(max_gap_s * fps))), sigma_s * fps) for j in range(2)])


def palm_speed(h, W, H, fps, cam):
    """Camera-stabilised palm-centre speed of one hand slot (n, 21, 3) in hand widths per second (NaN where the hand
    is not tracked), the palm trajectory and the hand width in px (90th percentile of the knuckle breadth)."""
    px = h[..., :2] * [W, H] - cam[:, None, :]
    breadth = np.hypot(*(px[:, 5] - px[:, 17]).T)
    ok = np.isfinite(breadth)
    ref = float(np.percentile(breadth[ok], 90)) if ok.sum() > 5 else np.nan
    P = smooth_xy(px[:, PALM].mean(1), fps, 0.1)
    speed = np.hypot(*np.gradient(P, axis=0).T) * fps / ref if np.isfinite(ref) else np.full(len(P), np.nan)
    return speed, P, ref


def gestures(speed, P, ref, fps):
    """Runs of palm speed above 0.2 hw/s that peak above 0.6 hw/s and travel >= 0.4 hand widths."""
    out = []
    for a, b in runs(np.nan_to_num(speed, nan=0.0) > 0.2):
        if np.nanmax(speed[a:b]) < GESTURE_PEAK_HW_S:
            continue
        travel = float(np.nanmax(np.hypot(*(P[a:b] - P[a]).T)) / ref)
        if travel >= GESTURE_MIN_TRAVEL_HW:
            out.append({"t": a / fps, "dur_s": (b - a) / fps, "travel_hw": travel, "peak_hw_s": float(np.nanmax(speed[a:b]))})
    return out


def analyze_hands(hand_lm, W, H, fps, cam, usable):
    """Speed and stillness of the hands (both slots pooled). (summary, raw) ; summary['tracked'] is False when no hand
    was found in at least 10 % of the clip."""
    hand_lm = np.where(usable[:, None, None, None], hand_lm, np.nan)
    n = len(hand_lm)
    sd = associate_hands(hand_lm, W, H, fps)
    lm = by_slot(hand_lm, sd)
    slots, speeds, gest = [], [], []
    seen = np.zeros(n, bool)
    for s in range(2):
        speed, P, ref = palm_speed(lm[:, s], W, H, fps, cam)
        vis = np.isfinite(speed)
        if vis.mean() < 0.10:
            slots.append(None)
            continue
        seen |= vis
        g = gestures(speed, P, ref, fps)
        slots.append({"visible_s": sig(vis.sum() / fps), "hand_width_px": sig(ref),
                      "still_share": sig(float(np.mean(speed[vis] < REST_HW_S))),
                      "speed_median_mm_s": sig(float(np.median(speed[vis])) * HAND_WIDTH_MM), "gestures": len(g)})
        speeds.append(speed[vis])
        gest.extend(g)
    if not speeds:
        return ({"tracked": False, "visible_share": {"value": 0.0, "unit": "fraction of frames with at least one hand"}},
                {"gestures": [], "speed": np.array([]), "visible_s": 0.0})
    v = np.concatenate(speeds)
    vis_s = float(len(v) / fps)
    res = {"tracked": True, "visible_share": {"value": sig(float(seen.mean())), "unit": "fraction of frames with at least one hand"},
           "visible_hand_seconds": {"value": sig(vis_s), "unit": "s, both hands counted"},
           "still_share": {"value": sig(float(np.mean(v < REST_HW_S))),
                           "unit": f"fraction of visible hand time with palm speed < {REST_HW_S} hand widths/s"},
           "speed_hw_s": dsp.describe(v, "hand widths/s (palm centre)"),
           "speed_mm_s": dsp.describe(v * HAND_WIDTH_MM, f"mm/s (hand breadth = {HAND_WIDTH_MM:g} mm)"),
           "gestures": {"definition": f"palm speed > 0.2 hand widths/s, peak >= {GESTURE_PEAK_HW_S} hand widths/s, travel >= "
                                      f"{GESTURE_MIN_TRAVEL_HW} hand widths",
                        "count": len(gest), "per_10s": sig(10 * len(gest) / vis_s),
                        "duration": dsp.with_values(dsp.describe([g["dur_s"] for g in gest], "s"), [g["dur_s"] for g in gest]),
                        "travel": dsp.with_values(dsp.describe([g["travel_hw"] * HAND_WIDTH_MM for g in gest], "mm"),
                                                  [g["travel_hw"] * HAND_WIDTH_MM for g in gest])},
           "slots": slots}
    return res, {"gestures": gest, "speed": v, "visible_s": vis_s}
