"""Per-clip analysis from the tracked arrays, and pooling over the clips of a set.

A clip's result has `qc` (camera drift, coverage, face size, cuts, duplicated frames, warnings), `face`, `body` and
`hands`. Every number carries its unit, its sample count `n` and what was counted (`n_basis`: frames, events, clips),
so thin evidence is visible. Pooling uses clip-level scalars (median / IQR over clips), event lists (median / IQR over
every event of every clip) and rates (all events / all observed seconds)."""
import time

import numpy as np

from . import body as BO
from . import dsp
from . import face as FA
from .dsp import describe, runs, scalar, sig

FACE_MIN_WIDTH_PX = 100       # below this the lids are a few pixels: blinks and gaze are not trusted
CAMERA_DRIFT_WARN_PCT = 5.0
DUPLICATE_FDIFF = 0.05        # a frame this similar to the previous one is a repeat (frame-rate conversion)
LITERATURE = [{
    "what": "spontaneous blink rate by activity (150 healthy volunteers)",
    "source": "Bentivoglio et al. 1997, Movement Disorders 12(6):1028-1034, doi:10.1002/mds.870120629",
    "values_blinks_per_min": {"reading": 4.5, "rest": 17, "conversation": 26, "reading_5_to_95_percent": [0.7, 22]},
    "used_for": "sanity check: most people blink 8-30 times a minute while talking or looking around, far less while reading or staring"}]


# ============================================================================================ quality control
def find_cuts(fdiff, fps):
    """Hard cuts = isolated spikes of the thumbnail difference: at least 6x the typical frame-to-frame change (of the
    frames that do change), above 3 grey levels, and 2.5x any change within 3 frames either side. Returns frame indices."""
    f = np.asarray(fdiff, float)
    moving = f[np.isfinite(f) & (f > DUPLICATE_FDIFF)]
    if len(moving) < 10:
        return []
    med = float(np.median(moving))
    out = []
    for i in np.flatnonzero(np.nan_to_num(f) > max(6 * med, 3.0)):
        nb = np.concatenate([f[max(i - 3, 0):i], f[i + 1:i + 4]])
        if f[i] > 2.5 * np.nanmax(nb if len(nb) else [0.0]):
            out.append(int(i))
    return out


def usable_mask(n, cuts, fps, pad_s=0.1):
    """Frames that may be used: not within pad_s after / before a cut (tracking restarts on the new shot)."""
    ok = np.ones(n, bool)
    pad = max(1, int(round(pad_s * fps)))
    for c in cuts:
        ok[max(c - pad, 0):c + pad + 1] = False
    return ok


def camera_track(d, cuts):
    """Cumulative camera translation (n, 2) px with the steps at cuts removed (a cut is not a camera move)."""
    cam = d["cam"].astype(float)
    step = np.diff(cam, axis=0, prepend=cam[:1])
    step[cuts] = 0.0
    step[~np.isfinite(step)] = 0.0
    return np.cumsum(step, axis=0)


def qc_block(d, fps, W, cam, cuts, face, body, hands):
    n = int(d["n"])
    drift = np.hypot(*cam.T)
    p95 = float(np.percentile(drift, 95))
    inl = float(np.nanmedian(d["cam_inl"])) if np.isfinite(d["cam_inl"]).any() else float("nan")
    fd = d["fdiff"].astype(float)
    dup = float(np.nanmean(fd < DUPLICATE_FDIFF)) if np.isfinite(fd).any() else 0.0
    eff = fps * (1 - dup)
    qc = {"camera_drift_p95": scalar(p95, "px (cumulative global translation, stabilised out of the hand and shoulder tracks)", n, "frames",
                                     pct_of_width=sig(100 * p95 / W, 3)),
          "camera_fit_inlier_ratio": scalar(inl, "fraction of background features that follow the camera model", n, "frames"),
          "pose_coverage": scalar(float(np.isfinite(d["pose_lm"][:, 11, 0]).mean()), "fraction of frames", n, "frames"),
          "duplicate_frames": scalar(dup, "fraction of frames identical to the previous one", n, "frames", effective_fps=sig(eff)),
          "cuts_s": [sig(c / fps, 3) for c in cuts]}
    warn = []
    if 100 * p95 / W > CAMERA_DRIFT_WARN_PCT:
        warn.append(f"camera drift p95 is {100 * p95 / W:.1f} % of the width: tracks are stabilised, but look at the sheet")
    if np.isfinite(inl) and inl < 0.3:
        warn.append(f"the camera model fits only {100 * inl:.0f} % of the background features: drift numbers are unreliable")
    if cuts:
        warn.append(f"{len(cuts)} hard cut(s): frames around them are ignored, statistics span several shots")
    if dup > 0.1:
        warn.append(f"{100 * dup:.0f} % of the frames repeat the previous one (frame-rate conversion): effective {eff:.0f} fps")
    if face is None:
        warn.append("the face is tracked in under 30 % of the clip (profile view, hidden or tiny face): no face statistics")
    elif not face["usable"]:
        warn.append(f"face statistics are not trusted: face tracked in {100 * face['coverage']['value']:.0f} % of the frames, "
                    f"{face['face_width_px']['value']:.0f} px wide (needs >= 50 % and >= {FACE_MIN_WIDTH_PX} px)")
    if face is not None and face.get("identity_jumps"):
        warn.append(f"the face tracker jumped between people {face['identity_jumps']} time(s) (several faces in the frame): the frames after "
                    "each jump are dropped, so coverage and blink counts are lower")
    if body is None:
        warn.append("shoulders are visible in under 60 % of the clip: no body statistics")
    if not hands.get("tracked"):
        warn.append("no hand is tracked in at least 10 % of the clip")
    qc["warnings"] = warn
    return qc


# ============================================================================================ one clip
def analyze_clip(entry, d):
    """All per-clip metrics from the cached tracking arrays: (JSON-able result, aux with events for the sheet)."""
    fps, n, W, H = float(d["fps"]), int(d["n"]), int(d["W"]), int(d["H"])
    cuts = find_cuts(d["fdiff"], fps)
    usable = usable_mask(n, cuts, fps)
    cam = camera_track(d, cuts)
    face, face_aux = FA.analyze_face(d, fps, W, H, usable)
    if face is not None:
        face["usable"] = bool(face["coverage"]["value"] >= 0.5 and face["face_width_px"]["value"] >= FACE_MIN_WIDTH_PX)
    body = BO.analyze_body(d["pose_lm"], W, H, fps, cam, usable)
    hands, hand_raw = BO.analyze_hands(d["hand_lm"], W, H, fps, cam, usable)
    res = {"id": int(entry["id"]), "url": entry.get("url"), "author": entry.get("author"), "why": entry.get("why"),
           "fps": sig(fps), "duration_s": sig(n / fps), "frames": n, "size_px": [W, H],
           "qc": qc_block(d, fps, W, cam, cuts, face, body, hands), "face": face, "body": body, "hands": hands}
    return res, {"face": face_aux, "hands": hand_raw, "cuts": cuts, "cam": cam}


# ============================================================================================ pooling
def _get(d, *path):
    for p in path:
        if not isinstance(d, dict) or d.get(p) is None:
            return None
        d = d[p]
    return d


def _num(x):
    return x.get("value") if isinstance(x, dict) else x


def per_clip(results, *path, unit):
    """Clip-level scalar at `path` (number or {'value': ..}) -> describe() over clips plus the by-clip values."""
    by = {}
    for cid, r in results.items():
        x = _num(_get(r, *path))
        if x is not None and np.isfinite(x):
            by[str(cid)] = x
    out = describe(list(by.values()), unit)
    out["n_basis"] = "clips"
    out["by_clip"] = {k: sig(v) for k, v in by.items()}
    return out


def pooled_events(results, *path, unit):
    """Every event of an event list (describe() + values) pooled over clips."""
    vals, nc = [], 0
    for r in results.values():
        x = _get(r, *path)
        if isinstance(x, dict) and x.get("values"):
            vals.extend(x["values"])
            nc += 1
    out = describe(vals, unit)
    out["n_basis"] = "events"
    out["n_clips"] = nc
    return out


def pooled_rate(results, count_path, sec_path, per, unit):
    """per * sum(events) / sum(observed seconds) over the clips that have both."""
    n = t = 0.0
    nc = 0
    for r in results.values():
        c, s = _num(_get(r, *count_path)), _num(_get(r, *sec_path))
        if c is None or not s:
            continue
        n, t, nc = n + c, t + s, nc + 1
    return {"value": sig(per * n / t) if t else None, "unit": unit, "n": int(n), "n_basis": "events",
            "observed_seconds": sig(t, 4), "n_clips": nc}


def pooled_ratio(results, num_path, den_path, unit):
    """sum(numerator) / sum(denominator) over the clips that have both (blinks per gaze shift)."""
    a = b = 0.0
    nc = 0
    for r in results.values():
        x, y = _num(_get(r, *num_path)), _num(_get(r, *den_path))
        if x is None or y is None:
            continue
        a, b, nc = a + x, b + y, nc + 1
    return {"value": sig(a / b) if b else None, "unit": unit, "n": int(a), "n_basis": "events", "per": int(b), "n_clips": nc}


def pool(results):
    """Pool per-clip results (dict id -> result). Face statistics come from clips whose face is usable."""
    face = {c: r for c, r in results.items() if r.get("face") and r["face"].get("usable")}
    body = {c: r for c, r in results.items() if r.get("body")}
    hands = {c: r for c, r in results.items() if r["hands"].get("tracked")}
    P = {"clips": {"ids": [str(c) for c in results], "with_face": [str(c) for c in face], "with_body": [str(c) for c in body],
                   "with_hands": [str(c) for c in hands],
                   "seconds": {"all": sig(sum(r["duration_s"] for r in results.values()), 4),
                               "face_tracked": sig(sum(_num(r["face"]["tracked_seconds"]) for r in face.values()), 4)}}}
    pf = lambda *path, unit: per_clip(face, "face", *path, unit=unit)
    pe = lambda *path, unit: pooled_events(face, "face", *path, unit=unit)
    P["blinks"] = {
        "rate_per_min": pooled_rate(face, ("face", "blinks", "count"), ("face", "blinks", "observed_seconds"), 60, "blinks/min, all observed time"),
        "rate_by_clip": pf("blinks", "rate_per_min", unit="blinks/min"),
        "spontaneous_rate_per_min": pooled_rate(face, ("face", "blinks", "spontaneous", "count"),
                                                ("face", "blinks", "spontaneous", "observed_seconds"), 60,
                                                "blinks/min outside gaze shifts (what perform.blink.per_min should be)"),
        "spontaneous_rate_by_clip": pf("blinks", "spontaneous", "rate_per_min", unit="blinks/min"),
        "per_gaze_shift": pooled_ratio(face, ("face", "blinks", "at_gaze_shift", "count"), ("face", "gaze", "shifts", "count"),
                                       f"blinks within {FA.SHIFT_NEAR_S} s of a gaze shift, per shift"),
        "duration_fwhm": pe("blinks", "duration_fwhm", unit="s"),
        "duration_total_20pct": pe("blinks", "duration_total_20pct", unit="s"),
        "depth": pe("blinks", "depth", unit="closure score rise"),
        "interval": pe("blinks", "interval", unit="s")}
    P["lids"] = {"rest_level": pf("lids", "rest_level", unit="closure score (0 open .. 1 shut)"),
                 "eyes_shut_share": pf("lids", "eyes_shut_share", unit="fraction of tracked time")}
    P["gaze"] = {
        "shift_rate_per_10s": pooled_rate(face, ("face", "gaze", "shifts", "count"), ("face", "gaze", "observed_seconds"), 10, "gaze shifts / 10 s"),
        "shift_rate_by_clip": pf("gaze", "shifts", "rate_per_10s", unit="shifts / 10 s"),
        "amplitude": pe("gaze", "shifts", "amplitude", unit="deg"),
        "peak_speed": pe("gaze", "shifts", "peak_speed", unit="deg/s"),
        "rise_gaze": pe("gaze", "shifts", "rise_gaze", unit="s"),
        "head_rise": pe("gaze", "shifts", "head_rise", unit="s"),
        "head_share": pe("gaze", "shifts", "head_share", unit="share of the gaze change taken by the head"),
        "hold": pe("gaze", "hold", unit="s"),
        "eye_in_head_p95": pf("gaze", "eye_in_head", "p95_magnitude", unit="deg")}
    P["head"] = {
        "convention": "pitch_up + = chin up (looking down is negative) relative to the camera axis; yaw + = nose towards image-right; roll + = top of head towards image-left",
        "pitch_up_mean": pf("head_pose", "pitch_up", "mean", unit="deg"), "pitch_up_std": pf("head_pose", "pitch_up", "std", unit="deg"),
        "yaw_mean": pf("head_pose", "yaw", "mean", unit="deg"), "yaw_std": pf("head_pose", "yaw", "std", unit="deg"),
        "roll_mean": pf("head_pose", "roll", "mean", unit="deg"), "roll_std": pf("head_pose", "roll", "std", unit="deg"),
        "speed_median": pf("head_pose", "speed_deg_s", "median", unit="deg/s"),
        "nod_amp": pf("head_motion", "nod", "amp", unit="deg (sinusoid-equivalent, 0.2-1.5 Hz)"),
        "nod_period": pf("head_motion", "nod", "period_s", unit="s"),
        "nod_noise": pf("head_motion", "nod", "noise_amp", unit="deg (landmark jitter in the band)"),
        "sway_yaw_amp": pf("head_motion", "sway_yaw", "amp", unit="deg (0.15-1 Hz)"),
        "sway_yaw_period": pf("head_motion", "sway_yaw", "period_s", unit="s"),
        "sway_roll_amp": pf("head_motion", "sway_roll", "amp", unit="deg (0.15-1 Hz)"),
        "sway_roll_period": pf("head_motion", "sway_roll", "period_s", unit="s"),
        "bob_amp": pf("head_motion", "bob", "amp", unit="deg (1-3.5 Hz)"),
        "bob_period": pf("head_motion", "bob", "period_s", unit="s"),
        "bob_noise": pf("head_motion", "bob", "noise_amp", unit="deg (landmark jitter in the band)")}
    mouth = {c: r for c, r in face.items() if r["face"].get("mouth")}
    pm = lambda *path, unit: per_clip(mouth, "face", "mouth", *path, unit=unit)
    P["mouth"] = {"moving_share": pm("moving_share", unit="fraction of tracked time"), "open_share": pm("open_share", unit="fraction of tracked time"),
                  "activity": pm("activity", unit="jawOpen units / s"),
                  "peak_p90_when_moving": pm("peak_p90_when_moving", unit="jawOpen"),
                  "cycle_period": pm("cycle_period_s", unit="s"),
                  "smile_share": pm("smile", "share_smiling", unit="fraction of tracked time with a smile score above %g" % FA.SMILE)}
    tq = lambda *path, unit: per_clip(body, "body", *path, unit=unit)
    P["body"] = {
        "shoulder_sway_amp": tq("shoulder_sway", "amp", unit="deg (sinusoid-equivalent, 0.15-1 Hz)"),
        "shoulder_sway_period": tq("shoulder_sway", "period_s", unit="s"),
        "shoulder_sway_noise": tq("shoulder_sway", "noise_amp", unit="deg (landmark jitter in the band)"),
        "breathing_rate": per_clip({c: r for c, r in body.items() if _get(r, "body", "breathing", "visible")},
                                   "body", "breathing", "rate_per_min", unit="breaths/min (tentative)"),
        "breathing_amp": per_clip({c: r for c, r in body.items() if _get(r, "body", "breathing", "visible")},
                                  "body", "breathing", "amp_pct_shoulder_width", unit="% of shoulder width (tentative)"),
        "breathing_visible_clips": sum(1 for r in body.values() if _get(r, "body", "breathing", "visible"))}
    ph = lambda *path, unit: per_clip(hands, "hands", *path, unit=unit)
    P["hands"] = {"visible_share": ph("visible_share", unit="fraction of frames with a hand"),
                  "still_share": ph("still_share", unit="fraction of visible hand time at rest"),
                  "speed_median_mm_s": ph("speed_mm_s", "median", unit="mm/s (hand breadth = 80 mm)"),
                  "speed_p90_mm_s": ph("speed_mm_s", "p90", unit="mm/s"),
                  "gesture_rate": pooled_rate(hands, ("hands", "gestures", "count"), ("hands", "visible_hand_seconds"), 10, "gestures / 10 s of visible hand time"),
                  "gesture_duration": pooled_events(hands, "hands", "gestures", "duration", unit="s"),
                  "gesture_travel": pooled_events(hands, "hands", "gestures", "travel", unit="mm")}
    P["qc"] = {"camera_drift_pct_of_width": per_clip(results, "qc", "camera_drift_p95", "pct_of_width", unit="% of the frame width (p95)")}
    return P


def build_meta():
    return {
        "purpose": "how real people move in reference clips, as parameters for the performance stage (mk build, [perform.<cast>])",
        "tracking": "MediaPipe Tasks: face_landmarker (478 landmarks, 52 blendshapes, facial transformation matrix; run on an enlarged head "
                    "crop), hand_landmarker (2 hands), pose_landmarker_full; OpenCV: RANSAC similarity fit of sparse optical flow on the "
                    "background for the global camera translation (stabilised out of hand and shoulder tracks)",
        "units": "angles in degrees, times in seconds, hand lengths in mm assuming an 80 mm index-MCP to pinky-MCP breadth (+-10 %) or in hand "
                 "widths, rates per minute or per 10 s as named; every entry carries n and n_basis (frames, events, clips)",
        "conventions": {"pitch_up": "+ = chin up; looking down is negative; relative to the camera axis (depends on camera height)",
                        "yaw": "+ = nose towards image-right", "roll": "+ = top of the head towards image-left (counter-clockwise on screen)",
                        "gaze": "head yaw/pitch + eye-in-head from the look blendshapes; eye-in-head = look score x "
                                f"{FA.EYE_FULL_DEG:g} deg (assumed, +-30 %)"},
        "definitions": {
            "blink": "brief peak of the lid closure score (mean eyeBlinkLeft/Right) over its local baseline: rise >= %g, width at half height %g-%g s"
                     % (FA.BLINK_DEPTH, *FA.BLINK_FWHM_S),
            "gaze_shift": f"transition between two fixations (gaze within {FA.FIX_RADIUS_DEG:g} deg for >= {FA.FIX_MIN_S:g} s) at least "
                          f"{FA.SHIFT_MIN_DEG:g} deg apart; hold = time spent in a fixation between two shifts",
            "oscillation_amplitude": "sinusoid-equivalent amplitude sqrt(2 * band variance) of the stated band; noise_amp is what white landmark "
                                     "jitter alone would give in the band",
            "hand_rest": f"palm speed below {BO.REST_HW_S} hand widths/s"},
        "literature": LITERATURE,
        "limitations": [
            "faces in profile or under ~100 px are not trusted (qc.warnings says so per clip); car-interior clips shot from the side give no face data",
            "blendshape eye angles are a learned approximation: eye-in-head degrees assume a full-scale score of %g deg; closed lids read as 'looking down' "
            "and are excluded" % FA.EYE_FULL_DEG,
            "head angles are relative to each camera's axis; camera heights differ between clips",
            "blink timing is limited by the frame rate (a blink lasts ~5 frames at 30 fps); clips with repeated frames have a lower effective rate",
            "shoulder sway and breathing are near the pose-landmark jitter (see noise_amp); breathing is at best 'tentative'",
            "counts are small: read n and n_basis, and treat recommendations as tuning targets, not constants"]}


def run_measure(clips, load_track, project=None, set_name=None, sources=None, fps=30.0):
    """Analyse every clip (load_track(id) -> dict of arrays), pool, recommend. Returns (output, aux per clip)."""
    from . import recommend as RC
    results, aux = {}, {}
    for entry in clips:
        d = load_track(entry["id"])
        results[entry["id"]], aux[entry["id"]] = analyze_clip(entry, d)
    P = pool(results)
    rec, why, skipped = RC.recommend(P, results, fps)
    out = {"schema": 1, "generated": time.strftime("%Y-%m-%d"), "set": set_name, "project": project,
           "sources": sources if sources is not None else [
               {k: c.get(k) for k in ("id", "url", "author", "author_url", "license", "license_url", "why")} for c in clips],
           "meta": build_meta(), "clips": {str(k): v for k, v in results.items()}, "pooled": P, "recommend": rec, "why": why,
           "skipped": skipped}
    return out, aux


def _h(x):
    """Headline form of a pooled entry: {median, iqr, n, unit} for a distribution, {value, n, unit} for a rate."""
    if not isinstance(x, dict):
        return x
    if "median" in x:
        return {k: x[k] for k in ("median", "iqr", "n", "n_basis", "unit") if k in x}
    return {k: x[k] for k in ("value", "n", "n_basis", "unit", "observed_seconds") if k in x}


def digest(out):
    """What `mk ref measure` prints: the recommendation with its evidence, one headline row per clip, the headline of the
    pooled statistics. The full result (every metric, event lists, definitions) is in the file."""
    P = out["pooled"]
    rows = []
    for r in out["clips"].values():
        f, b, h = r["face"], r["body"], r["hands"]
        row = {"id": r["id"], "seconds": r["duration_s"], "fps": r["fps"], "face": ("usable" if f and f["usable"] else "partial" if f else "none")}
        if f and f["usable"]:
            row["blinks_per_min"] = f["blinks"]["rate_per_min"]["value"]
            row["gaze_shifts_per_10s"] = f["gaze"]["shifts"]["rate_per_10s"]["value"]
            row["nod_deg"] = (f["head_motion"]["nod"] or {}).get("amp")
            row["mouth_moving_share"] = ((f["mouth"] or {}).get("moving_share") or {}).get("value")
        if b:
            row["shoulder_sway_deg"] = (b["shoulder_sway"] or {}).get("amp")
        if h.get("tracked"):
            row["hands_still_share"] = h["still_share"]["value"]
        row["camera_drift_pct"] = r["qc"]["camera_drift_p95"]["pct_of_width"]
        row["warnings"] = r["qc"]["warnings"]
        rows.append(row)
    pooled = {"clips_with_face": P["clips"]["with_face"], "face_seconds": P["clips"]["seconds"]["face_tracked"],
              "blink_rate_per_min": _h(P["blinks"]["rate_per_min"]), "blink_rate_by_clip": _h(P["blinks"]["rate_by_clip"]),
              "blink_spontaneous_per_min": _h(P["blinks"]["spontaneous_rate_per_min"]), "blink_duration_fwhm": _h(P["blinks"]["duration_fwhm"]),
              "lids_rest_level": _h(P["lids"]["rest_level"]), "gaze_shifts_per_10s": _h(P["gaze"]["shift_rate_per_10s"]),
              "gaze_hold": _h(P["gaze"]["hold"]), "gaze_head_rise": _h(P["gaze"]["head_rise"]), "gaze_head_share": _h(P["gaze"]["head_share"]),
              "eye_in_head_p95": _h(P["gaze"]["eye_in_head_p95"]), "head_pitch_std": _h(P["head"]["pitch_up_std"]),
              "head_yaw_std": _h(P["head"]["yaw_std"]), "head_roll_std": _h(P["head"]["roll_std"]), "nod_amp": _h(P["head"]["nod_amp"]),
              "nod_period": _h(P["head"]["nod_period"]), "mouth_moving_share": _h(P["mouth"]["moving_share"]),
              "shoulder_sway_amp": _h(P["body"]["shoulder_sway_amp"]), "shoulder_sway_period": _h(P["body"]["shoulder_sway_period"]),
              "hands_still_share": _h(P["hands"]["still_share"]), "hand_speed_median_mm_s": _h(P["hands"]["speed_median_mm_s"])}
    return {"set": out["set"], "project": out["project"], "out": out.get("out"), "clips": rows, "recommend": out["recommend"], "why": out["why"],
            "skipped": out["skipped"], "pooled": pooled,
            "sources": [{k: s.get(k) for k in ("id", "author", "url", "license")} for s in out["sources"]],
            "untracked": out.get("untracked", []), "full": "`--full` prints everything; the file has it too (definitions, limitations, event lists)"}
