"""Face: head pose, blinks, eyelid rest level, gaze shifts, mouth activity and head oscillations of one clip, from the
face landmarker's blendshapes, landmarks and facial transformation matrix.

Conventions (degrees): pitch_up + = chin up (looking down is negative), relative to the camera axis; yaw + = nose
towards image-right; roll + = top of the head towards image-left (counter-clockwise on screen). Eye-in-head angles use
the same signs (yaw + = towards image-right, pitch + = up). Gaze = head + eye-in-head."""
import numpy as np

from . import dsp
from .dsp import crossing_time, ev, gsmooth, interp_gaps, runs, scalar, sig

# 52 ARKit-style blendshape names in the order the face landmarker returns them (index 0 is "_neutral")
BLENDSHAPES = [
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight", "cheekPuff",
    "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight", "mouthFunnel",
    "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight", "mouthPucker",
    "mouthRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft",
    "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight"]
BS = {n: i for i, n in enumerate(BLENDSHAPES)}

EYE_FULL_DEG = 30.0      # assumed eye rotation at a look-blendshape score of 1.0 (+-30 %; iris landmarks suggest 16-40)
YAW_MAX = 55.0           # beyond this the face is near profile: lid, eye and mouth scores are unreliable
CLOSED = 0.5            # lid closure score above which the lids cover the iris: no eye direction (0.35-0.5 is a downcast gaze)
BLINK_DEPTH = 0.25       # minimum rise of the lid closure score over its local baseline for a blink
BLINK_FWHM_S = (0.06, 0.6)
FIX_RADIUS_DEG = 4.0    # a fixation: the gaze stays within this of its running median for ...
FIX_MIN_S = 0.2         # ... at least this long
SHIFT_MIN_DEG = 6.0     # a gaze shift: two neighbouring fixations at least this far apart
SHIFT_NEAR_S = 0.4       # a blink within this of a gaze shift counts as part of it
FACE_JUMP_WIDTHS = 0.7   # nose tip moving more than this many face widths between two tracked frames ...
FACE_JUMP_PER_S = 4.0    # ... plus this many per second of gap = the tracker changed person
MOUTH_OPEN = 0.12        # jawOpen above this = mouth open
MOUTH_MOVING = 0.08      # jawOpen range within 0.5 s above this = the mouth is moving (talking / singing)
SMILE = 0.3             # mean mouthSmileLeft/Right above this = smiling
SMOOTHSTEP_1090 = 0.608  # a smoothstep rise takes this share of its length to go from 10 % to 90 %
EYE_L = [362, 385, 387, 263, 373, 380]    # subject-left eye: inner corner, 2 upper-lid points, outer corner, 2 lower-lid points
EYE_R = [33, 160, 158, 133, 153, 144]     # subject-right eye: outer corner, 2 upper-lid points, inner corner, 2 lower-lid points


# ============================================================================================ head pose
def head_angles(mat):
    """(n, 3) pitch_up / yaw / roll in degrees from the facial transformation matrix (camera frame: x right, y up, z
    towards the viewer; scale removed by SVD). Euler order YXZ = yaw about the head's vertical axis, then pitch about its
    lateral axis, then roll about the nose axis."""
    R = mat[:, :3, :3].astype(float)
    out = np.full((len(R), 3), np.nan)
    ok = np.isfinite(R).all((1, 2))
    if ok.any():
        U, _, Vt = np.linalg.svd(R[ok])
        Rn = U @ Vt
        out[ok] = np.column_stack([np.degrees(np.arcsin(np.clip(Rn[:, 1, 2], -1, 1))),
                                   np.degrees(np.arctan2(Rn[:, 0, 2], Rn[:, 2, 2])),
                                   np.degrees(np.arctan2(Rn[:, 1, 0], Rn[:, 1, 1]))])
    return out


def eye_ear(lm, W, H):
    """Eye aspect ratio (n, 2) = [subject-left, subject-right]: (|p2-p6| + |p3-p5|) / (2 |p1-p4|) in aspect-correct px."""
    out = np.full((len(lm), 2), np.nan)
    for j, idx in enumerate((EYE_L, EYE_R)):
        p = lm[:, idx, :2] * [W, H]
        v = np.hypot(*(p[:, 1] - p[:, 5]).T) + np.hypot(*(p[:, 2] - p[:, 4]).T)
        out[:, j] = v / (2 * np.hypot(*(p[:, 0] - p[:, 3]).T))
    return out


# ============================================================================================ eyes
def closure_series(bs):
    """Lid closure score 0 (open) .. ~0.7 (shut): mean of eyeBlinkLeft / eyeBlinkRight."""
    return (bs[:, BS["eyeBlinkLeft"]] + bs[:, BS["eyeBlinkRight"]]) / 2.0


def detect_blinks(score, fps, win_s=0.5, depth=BLINK_DEPTH):
    """Blinks = brief peaks of the lid closure score above its local baseline. An opening (min then max over win_s)
    keeps closures wider than the window and slow changes of lid level (gaze moving down, squinting) as the baseline, so
    only events narrower than win_s remain: rise = score - baseline. An event is a run of rise above 40 % of `depth`
    whose peak reaches `depth`, with a width at half height inside BLINK_FWHM_S, clear of gaps and the series ends.
    Returns (events, rise series); an event has t, depth, level (score at the peak), fwhm_s and total_s (width at 20 %)."""
    n = len(score)
    t = np.arange(n) / fps
    x = interp_gaps(score, int(round(0.2 * fps)))
    valid = np.isfinite(x)
    if valid.sum() < 2 * fps:
        return [], np.zeros(n)
    xs = gsmooth(np.where(valid, x, np.nanmedian(x)), max(0.7, 0.025 * fps))
    base = dsp.opening(xs, int(round(win_s * fps)) | 1)
    rise = np.clip(xs - base, 0, None)
    rise[~valid] = 0
    ev_ = []
    for a, b in runs(rise > 0.4 * depth):
        pk = a + int(np.argmax(rise[a:b]))
        lo, hi = max(a - 2, 0), min(b + 2, n)
        if rise[pk] < depth or a == 0 or b == n or not valid[lo:hi].all():
            continue
        h0, h1 = (crossing_time(t, rise, pk, 0.5 * rise[pk], d) for d in (-1, 1))
        e0, e1 = (crossing_time(t, rise, pk, 0.2 * rise[pk], d) for d in (-1, 1))
        if h0 is None or h1 is None or not (max(BLINK_FWHM_S[0], 1.5 / fps) <= h1 - h0 <= BLINK_FWHM_S[1]):
            continue
        ev_.append({"t": float(t[pk]), "depth": float(rise[pk]), "level": float(xs[pk]), "fwhm_s": h1 - h0,
                    "total_s": (e1 - e0) if (e0 is not None and e1 is not None) else None})
    return ev_, rise


def eye_in_head(bs, closure, fps):
    """(n, 2) eye-in-head [yaw, pitch] in degrees from the look blendshapes (yaw + = towards image-right = the
    subject's left, pitch + = up), NaN where the lids cover the eye (the model reports 'looking down' for closed
    lids)."""
    g = lambda name: bs[:, BS[name]]
    yaw = ((g("eyeLookOutLeft") + g("eyeLookInRight")) - (g("eyeLookInLeft") + g("eyeLookOutRight"))) / 2.0
    pitch = ((g("eyeLookUpLeft") + g("eyeLookUpRight")) - (g("eyeLookDownLeft") + g("eyeLookDownRight"))) / 2.0
    out = np.column_stack([yaw, pitch]) * EYE_FULL_DEG
    out[dsp.dilate(np.nan_to_num(closure, nan=1.0) > CLOSED, max(1, int(round(0.06 * fps))))] = np.nan
    return out


# ============================================================================================ gaze shifts
def _median(x, i0, i1, min_n=2):
    """Per-component median of x[i0:i1] (n, k) over finite rows, None with fewer than min_n of them."""
    i0, i1 = max(i0, 0), min(i1, len(x))
    if i1 <= i0:
        return None
    seg = x[i0:i1]
    seg = seg[np.isfinite(seg).all(1)]
    return np.median(seg, axis=0) if len(seg) >= min_n else None


def _rise(t, s, centre, lo_i, hi_i):
    """10 % -> 90 % time of a normalised progress curve s around index `centre` (None if not both crossed)."""
    seg_t, seg = t[lo_i:hi_i], s[lo_i:hi_i]
    if len(seg) < 3:
        return None, None
    c = int(np.clip(centre - lo_i, 0, len(seg) - 1))
    t10 = crossing_time(seg_t, seg, c, 0.1, -1)
    t90 = crossing_time(seg_t, seg, c, 0.9, 1)
    return (t10, t90) if t10 is not None and t90 is not None and t90 >= t10 else (None, None)


def fixations(g, fps, radius=FIX_RADIUS_DEG, min_s=FIX_MIN_S):
    """Dispersion-threshold fixations of a gaze series g (n, 2) in degrees (NaN = undefined): stretches of at least
    min_s where the samples stay within `radius` of their running median (a lone outlier is tolerated), never
    spanning a gap. Returns [[i0, i1, centre], ...] with the stretch being samples i0 .. i1 - 1."""
    n = len(g)
    ok = np.isfinite(g).all(1)
    min_n = max(3, int(round(min_s * fps)))
    out, i = [], 0
    while i + min_n <= n:
        if not ok[i:i + min_n].all():
            i += 1
            continue
        c = np.median(g[i:i + min_n], axis=0)
        if np.hypot(*(g[i:i + min_n] - c).T).max() > radius:
            i += 1
            continue
        j = i + min_n
        while j < n and ok[j]:
            c = np.median(g[i:j + 1], axis=0)
            d0 = np.hypot(*(g[j] - c))
            d1 = np.hypot(*(g[j + 1] - c)) if j + 1 < n and ok[j + 1] else d0
            if d0 > radius and d1 > radius:
                break
            j += 1
        out.append([i, j, np.median(g[i:j], axis=0)])
        i = j
    return out


def gaze_shifts(gaze, head, fps, radius=FIX_RADIUS_DEG, amp_min=SHIFT_MIN_DEG):
    """Gaze shifts and holds. `gaze` and `head` are (n, 2) [yaw, pitch] in degrees (NaN where undefined; gaze = head +
    eye-in-head, undefined while the eyes are shut).

    The gaze is split into fixations (see `fixations`); neighbouring fixations closer than amp_min are one. A shift is the
    transition between two consecutive fixations that are amp_min or more apart and were tracked all the way between
    (at most 2.5 s). Per shift:
      amp            distance between the two fixation centres (deg)
      t_on, t_off    when the gaze has covered 10 % / 90 % of that distance; rise_gaze_s = t_off - t_on
      peak_deg_s     peak gaze speed in the transition
      head_amp       head change between the ends of the two fixations (deg), head_share = head_amp / amp (the head
                     needs time after the eyes: the last 0.3 s of a fixation of at least 0.4 s stand for 'settled')
      head_rise_s    head 10 -> 90 % time, when the head moved >= 3 deg
      dwell_s        how long the gaze then stayed (the fixation after the shift) when another shift ends it
    """
    n = len(gaze)
    t = np.arange(n) / fps
    sm = max(0.7, 0.025 * fps)
    gs = np.column_stack([gsmooth(interp_gaps(gaze[:, k], int(round(0.25 * fps))), sm) for k in range(2)])
    hs = np.column_stack([gsmooth(interp_gaps(head[:, k], int(round(0.25 * fps))), sm) for k in range(2)])
    fx = []
    for i0, i1, c in fixations(gs, fps, radius):
        if fx:
            p = fx[-1]
            if i0 - p[1] <= int(0.5 * fps) and np.isfinite(gs[p[1]:i0 + 1]).all() and np.hypot(*(c - p[2])) < amp_min:
                p[1], p[2] = i1, np.median(gs[p[0]:i1], axis=0)
                continue
        fx.append([i0, i1, c])
    m = max(2, int(round(0.3 * fps)))
    out = []
    for k in range(len(fx) - 1):
        A, B = fx[k], fx[k + 1]
        e, s0 = A[1], B[0]
        if s0 - e > 2.5 * fps or not np.isfinite(gs[e:s0 + 1]).all():
            continue
        d = B[2] - A[2]
        amp = float(np.hypot(*d))
        if amp < amp_min:
            continue
        s = ((gs - A[2]) @ (d / amp)) / amp                   # progress along the shift, 0 -> 1
        lo, hi = (A[0] + A[1]) // 2, (B[0] + B[1]) // 2
        mid = e + int(np.argmin(np.abs(s[e:s0 + 1] - 0.5)))
        t10, t90 = _rise(t, s, mid, lo, hi + 1)
        if t10 is None:
            t10, t90 = t[e], t[min(s0, n - 1)]
        lo2, hi2 = max(e - 2, 1), min(s0 + 2, n - 1)
        peak = float(np.max(np.hypot(*(gs[lo2 + 1:hi2 + 1] - gs[lo2 - 1:hi2 - 1]).T)) * fps / 2) if hi2 > lo2 else 0.0
        sh = {"amp": amp, "t_on": float(t10), "t_off": float(t90), "rise_gaze_s": float(t90 - t10), "peak_deg_s": peak,
              "vertical": bool(abs(d[1]) > abs(d[0])), "from": k, "to": k + 1, "head_amp": None, "head_rise_s": None,
              "head_share": None, "dwell_s": None}
        hpre = _median(hs, A[1] - min(m, A[1] - A[0]), A[1])
        hpost = _median(hs, B[1] - min(m, B[1] - B[0]), B[1]) if B[1] - B[0] >= 0.4 * fps else None
        if hpre is not None and hpost is not None:
            hd = hpost - hpre
            h_amp = float(np.hypot(*hd))
            sh["head_amp"], sh["head_share"] = h_amp, float(np.clip(h_amp / amp, 0.0, 1.0))
            if h_amp >= 3.0:
                hp = ((hs - hpre) @ (hd / h_amp)) / h_amp
                h_lo, h_hi = (A[0] + A[1]) // 2, B[1]
                hc = e + int(np.argmin(np.abs(np.nan_to_num(hp[e:min(B[1], n)], nan=9.0) - 0.5)))
                h10, h90 = _rise(t, hp, hc, h_lo, h_hi)
                if h10 is not None:
                    sh["head_rise_s"] = float(h90 - h10)
        out.append(sh)
    follow = {sh["from"]: sh for sh in out}
    for sh in out:
        if sh["to"] in follow:                                # the fixation between two shifts is a complete hold
            B = fx[sh["to"]]
            sh["dwell_s"] = (B[1] - B[0]) / fps
    return out


# ============================================================================================ mouth
def mouth_activity(jaw, valid, fps, smile=None):
    """Mouth openness and activity from the jawOpen score. 'Moving' = the score ranges by more than MOUTH_MOVING within
    0.5 s (talking / singing); 'open' = above MOUTH_OPEN. `smile` (mean mouthSmileLeft/Right) adds how much the person
    smiles. Returns (summary, jaw level over moving frames)."""
    j = interp_gaps(jaw, int(round(0.2 * fps)))
    ok = np.isfinite(j) & valid
    if ok.sum() < 2 * fps:
        return None, np.array([])
    filled = np.where(np.isfinite(j), j, np.nanmedian(j[ok]))
    w = int(round(0.5 * fps)) | 1
    rng = dsp.sliding_extreme(filled, w, np.max) - dsp.sliding_extreme(filled, w, np.min)
    moving = ok & (rng > MOUTH_MOVING)
    both = ok[1:] & ok[:-1]                                      # consecutive tracked frames only
    lvl = j[moving]
    res = {"definition": f"jawOpen blendshape (0 closed .. 1 wide open); moving = range > {MOUTH_MOVING} within 0.5 s, open = "
                         f"above {MOUTH_OPEN}",
           "observed_seconds": scalar(ok.sum() / fps, "s", int(ok.sum()), "frames"),
           "open_share": scalar(float(np.mean(j[ok] > MOUTH_OPEN)), "fraction of tracked time", int(ok.sum()), "frames"),
           "moving_share": scalar(float(moving.sum() / ok.sum()), "fraction of tracked time", int(ok.sum()), "frames"),
           "activity": scalar(float(np.mean(np.abs(np.diff(j))[both]) * fps) if both.any() else np.nan,
                              "jawOpen units / s (mean |d jawOpen / dt|)", int(ok.sum()), "frames")}
    if smile is not None and np.isfinite(smile[ok]).sum() > fps:
        s = smile[ok & np.isfinite(smile)]
        res["smile"] = {"level": dsp.describe(s, "mean mouthSmileLeft/Right"),
                        "share_smiling": scalar(float(np.mean(s > SMILE)), f"fraction of tracked time with the smile score above {SMILE}",
                                                len(s), "frames")}
    if moving.sum() >= fps:
        res["level_when_moving"] = dsp.describe(lvl, "jawOpen")
        res["peak_p90_when_moving"] = scalar(np.percentile(lvl, 90), "jawOpen (90th percentile over moving frames)",
                                             int(moving.sum()), "frames")
        osc = dsp.oscillation(j, fps, moving, 0.7, min(8.0, fps / 2 - 1), 1.5)
        if osc:
            res["cycle_period_s"] = scalar(osc["period_s"], "s (dominant open-close period while moving)", osc["n_segments"],
                                           "segments", peak_distinct=osc["peak_distinct"])
    return res, lvl


# ============================================================================================ the clip
def _stat(x, m, unit="deg"):
    v = x[m & np.isfinite(x)]
    if len(v) <= 5:
        return None
    return {"mean": sig(v.mean()), "std": sig(v.std(ddof=1)), "p5": sig(np.percentile(v, 5)),
            "p95": sig(np.percentile(v, 95)), "unit": unit, "n": int(len(v)), "n_basis": "frames"}


def _osc_json(o, unit, what, ratio_note=True):
    if not o:
        return None
    out = {"amp": sig(o["amp"]), "unit": unit, "period_s": sig(o["period_s"]), "peak_distinct": o["peak_distinct"],
           "peak_period_s": sig(o["peak_period_s"]), "centroid_period_s": sig(o["centroid_period_s"]),
           "zero_cross_period_s": sig(o["zero_cross_period_s"]), "std": sig(o["std"]), "noise_amp": sig(o.get("noise_amp")),
           "seconds": sig(o["seconds"]), "segments": o["n_segments"], "definition": what}
    return out


def analyze_face(d, fps, W, H, usable):
    """Everything about the face of one clip. `usable` masks frames that may be used (no cuts). Returns (summary, aux)
    or (None, {}) when the face is tracked in less than 30 % of the clip (profile views, tiny or hidden faces)."""
    n = int(d["n"])
    t = np.arange(n) / fps
    lm, bs = d["face_lm"], d["face_bs"]
    raw = np.isfinite(lm[:, 0, 0]) & usable
    face_w = np.hypot(*((lm[:, 454, :2] - lm[:, 234, :2]) * [W, H]).T)
    # two people in the frame: the tracker may hop from one face to the other, with or without a gap in between. No head moves
    # more than FACE_JUMP_WIDTHS face widths (plus FACE_JUMP_PER_S per second of gap) between two tracked frames: drop the
    # frames after such a jump for longer than any gap gets bridged, so no event spans two people
    v = np.flatnonzero(raw)
    jumps = []
    if len(v) > 1:
        hop = np.hypot(*(lm[v[1:], 1, :2] * [W, H] - lm[v[:-1], 1, :2] * [W, H]).T) / np.fmax(face_w[v[1:]], 1.0)
        jumps = v[1:][hop > FACE_JUMP_WIDTHS + FACE_JUMP_PER_S * np.diff(v) / fps]
    face_ok = raw.copy()
    for j in jumps:
        face_ok[j:j + int(np.ceil(0.3 * fps)) + 1] = False
    if face_ok.mean() < 0.30:
        return None, {}
    ang = head_angles(np.where(usable[:, None, None], d["face_mat"], np.nan))
    pitch, yaw, roll = ang.T
    frontal = face_ok & np.isfinite(yaw) & (np.abs(yaw) <= YAW_MAX)
    closure = np.where(frontal, closure_series(bs), np.nan)
    tracked_s = float(face_ok.sum() / fps)
    res = {"coverage": scalar(float(face_ok.mean()), "fraction of frames", n, "frames"),
           "tracked_seconds": scalar(tracked_s, "s", int(face_ok.sum()), "frames"),
           "face_width_px": scalar(np.nanmedian(face_w[face_ok]), "px (cheek to cheek)", int(face_ok.sum()), "frames"),
           "identity_jumps": int(len(jumps))}

    # ---- blinks, eyes, gaze shifts
    blinks, _ = detect_blinks(closure, fps)
    eyes = eye_in_head(bs, closure, fps)
    head2 = np.column_stack([yaw, pitch])
    gaze = np.where(frontal[:, None], head2 + eyes, np.nan)
    head2 = np.where(frontal[:, None], head2, np.nan)
    shifts = gaze_shifts(gaze, head2, fps)
    near = np.zeros(n, bool)
    for s in shifts:
        near |= (t >= s["t_on"] - SHIFT_NEAR_S) & (t <= s["t_off"] + SHIFT_NEAR_S)
    bt = [b["t"] for b in blinks]
    at_shift = [bool(near[min(int(round(x * fps)), n - 1)]) for x in bt]
    valid_blink = np.isfinite(interp_gaps(closure, int(round(0.2 * fps))))
    obs_s = float(valid_blink.sum() / fps)
    spont_obs = float((valid_blink & ~near).sum() / fps)
    n_sp = at_shift.count(False)
    ibi = [b - a for a, b in zip(bt[:-1], bt[1:]) if frontal[int(a * fps):int(b * fps) + 1].all()]
    res["blinks"] = {
        "definition": f"brief peaks of the lid closure score (mean eyeBlinkLeft/Right) over its local baseline (opening over 0.5 s): "
                      f"rise >= {BLINK_DEPTH}, width at half height {BLINK_FWHM_S[0]}-{BLINK_FWHM_S[1]} s; closures wider than 0.5 s "
                      f"(eyes shut while singing, laughing) are not blinks",
        "observed_seconds": scalar(obs_s, "s", int(valid_blink.sum()), "frames"),
        "count": scalar(len(blinks), "blinks", len(blinks), "events"),
        "rate_per_min": scalar(60 * len(blinks) / obs_s if obs_s else np.nan, "blinks/min over all observed time", len(blinks), "events"),
        "spontaneous": {
            "definition": f"blinks further than {SHIFT_NEAR_S} s from a gaze shift (the performance stage adds a blink to each gaze event itself)",
            "count": n_sp, "observed_seconds": sig(spont_obs),
            "rate_per_min": scalar(60 * n_sp / spont_obs if spont_obs >= 3 else np.nan, "blinks/min outside gaze shifts", n_sp, "events",
                                   observed_seconds=sig(spont_obs))},
        "at_gaze_shift": {"count": len(blinks) - n_sp,
                          "per_shift": scalar(((len(blinks) - n_sp) / len(shifts)) if shifts else np.nan,
                                              f"blinks within {SHIFT_NEAR_S} s of a shift, per shift", len(blinks) - n_sp, "events",
                                              shifts=len(shifts))},
        "duration_fwhm": ev([b["fwhm_s"] for b in blinks], "s (width at half height)"),
        "duration_total_20pct": ev([b["total_s"] for b in blinks if b["total_s"] is not None], "s (width at 20 % of the closure)"),
        "depth": ev([b["depth"] for b in blinks], "closure score rise"),
        "interval": ev(ibi, "s"), "times_s": [sig(x, 3) for x in bt]}
    # ---- eyelid rest level
    keep = frontal & np.isfinite(closure) & ~dsp.dilate(np.isin(np.arange(n), [int(round(b["t"] * fps)) for b in blinks]),
                                                         int(round(0.3 * fps)))
    rest = keep & (closure < CLOSED)
    shut = runs(frontal & (np.nan_to_num(closure, nan=0) > 0.5))
    shut = [(a, b) for a, b in shut if (b - a) / fps >= 0.4]
    if rest.sum() > 5:
        res["lids"] = {
            "definition": "lid closure score (0 open .. 1 shut) away from blinks, eyes visibly open (< %.2f); shut = above 0.5 for >= 0.4 s" % CLOSED,
            "rest_level": scalar(np.median(closure[rest]), "closure score (median)", int(rest.sum()), "frames"),
            "p10_p90": [sig(np.percentile(closure[rest], 10)), sig(np.percentile(closure[rest], 90))],
            "eyes_shut_share": scalar(sum(b - a for a, b in shut) / max(frontal.sum(), 1), "fraction of tracked time", len(shut), "episodes")}
    else:
        res["lids"] = None
    # ---- gaze
    gaze_ok = np.isfinite(gaze).all(1)
    gaze_s = float(gaze_ok.sum() / fps)
    sh_amp = [s["amp"] for s in shifts]
    big = [s for s in shifts if s["amp"] >= 10.0 and s["head_share"] is not None]
    eye_mag = np.hypot(*eyes.T)
    res["gaze"] = {
        "definition": f"gaze = head yaw/pitch + eye-in-head from the look blendshapes (x {EYE_FULL_DEG:g} deg per score unit, assumed); "
                      f"fixations = stays within {FIX_RADIUS_DEG:g} deg for >= {FIX_MIN_S:g} s; a shift = the transition between two fixations "
                      f">= {SHIFT_MIN_DEG:g} deg apart; a hold = the time spent in a fixation between two shifts; undefined while the eyes are shut",
        "observed_seconds": scalar(gaze_s, "s", int(gaze_ok.sum()), "frames"),
        "shifts": {"count": len(shifts),
                   "rate_per_10s": scalar(10 * len(shifts) / gaze_s if gaze_s >= 3 else np.nan, "gaze shifts / 10 s", len(shifts), "events",
                                          observed_seconds=sig(gaze_s)),
                   "amplitude": ev(sh_amp, "deg"), "peak_speed": ev([s["peak_deg_s"] for s in shifts], "deg/s"),
                   "rise_gaze": ev([s["rise_gaze_s"] for s in shifts], "s (gaze 10 -> 90 %)"),
                   "head_rise": ev([s["head_rise_s"] for s in shifts if s["head_rise_s"] is not None], "s (head 10 -> 90 %, shifts where the head moves >= 3 deg)"),
                   "head_amp": ev([s["head_amp"] for s in shifts if s["head_amp"] is not None], "deg"),
                   "head_share": ev([s["head_share"] for s in big], "share of the settled gaze change taken by the head (shifts >= 10 deg)"),
                   "vertical_share": scalar(np.mean([s["vertical"] for s in shifts]) if shifts else np.nan, "share of shifts mostly up/down",
                                            len(shifts), "events"),
                   "times_s": [sig(s["t_on"], 3) for s in shifts]},
        "hold": ev([s["dwell_s"] for s in shifts if s["dwell_s"] is not None], "s (time in a fixation between two shifts)"),
        "eye_in_head": {"yaw": _stat(eyes[:, 0], frontal), "pitch": _stat(eyes[:, 1], frontal),
                        "p95_magnitude": scalar(np.nanpercentile(eye_mag[frontal], 95) if np.isfinite(eye_mag[frontal]).sum() > 10 else np.nan,
                                                "deg (95th percentile of the eye-in-head angle, eyes open)", int(np.isfinite(eye_mag[frontal]).sum()), "frames")}}
    # ---- head pose and its oscillations away from gaze shifts
    spd = np.hypot(*(np.gradient(gsmooth(interp_gaps(a, int(round(0.2 * fps))), 1.0)) * fps for a in (pitch, yaw))) if frontal.sum() > 5 else None
    res["head_pose"] = {
        "convention": "pitch_up + = chin up (looking down is negative) relative to the camera axis; yaw + = nose towards image-right; roll + = top of head towards image-left",
        "pitch_up": _stat(pitch, frontal), "yaw": _stat(yaw, face_ok), "roll": _stat(roll, frontal & (np.abs(yaw) <= 45)),
        "speed_deg_s": ev(spd[frontal & np.isfinite(spd)], "deg/s (yaw+pitch angular speed)") if spd is not None else None}
    idle = frontal & ~near & np.isfinite(pitch)
    nz = 4.0 if fps >= 20 else None
    res["head_motion"] = {
        "frames_used": scalar(float(idle.sum() / fps), "s (face tracked, no gaze shift within 0.4 s)", int(idle.sum()), "frames"),
        "nod": _osc_json(dsp.oscillation(pitch, fps, idle, 0.2, 1.5, 3.0, nz), "deg", "head pitch, 0.2-1.5 Hz band, away from gaze shifts"),
        "sway_yaw": _osc_json(dsp.oscillation(yaw, fps, idle, 0.15, 1.0, 6.0, nz), "deg", "head yaw, 0.15-1 Hz band, away from gaze shifts"),
        "sway_roll": _osc_json(dsp.oscillation(roll, fps, idle & (np.abs(yaw) <= 45), 0.15, 1.0, 6.0, nz), "deg", "head roll, 0.15-1 Hz band, away from gaze shifts"),
        "bob": _osc_json(dsp.oscillation(pitch, fps, idle, 1.0, min(3.5, fps / 2 - 1), 2.0, nz), "deg", "head pitch, 1-3.5 Hz band (rhythmic head bobbing)")}
    # ---- mouth
    smile = (bs[:, BS["mouthSmileLeft"]] + bs[:, BS["mouthSmileRight"]]) / 2.0
    mouth, _ = mouth_activity(np.where(frontal, bs[:, BS["jawOpen"]], np.nan), frontal, fps, np.where(frontal, smile, np.nan))
    res["mouth"] = mouth
    aux = {"blinks": blinks, "shifts": shifts, "pitch": pitch, "yaw": yaw, "roll": roll, "closure": closure,
           "jaw": bs[:, BS["jawOpen"]], "gaze": gaze, "face_ok": face_ok}
    return res, aux
