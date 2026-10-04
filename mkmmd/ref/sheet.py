"""Contact sheet: one row per clip, frames chosen at measured events (a blink, the biggest gaze shift, the widest mouth,
the extreme head turn, a hand gesture) with the tracked landmarks drawn and the frame's values written on them.
Frames come from the downloaded clip, so the sheet is made of copyrighted pictures: it stays in the cache."""
import os

import cv2
import numpy as np

from ..core import palette as PAL
from . import RefUsage, measure
from .face import EYE_L, EYE_R

FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
             172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
HAND_BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
              (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
ARM_BONES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
TILE = (512, 288)
GAP = 6


def _bgr(palette, name):
    h = palette[name].lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (4, 2, 0))


def _seg(img, p, q, col, halo, w=2):
    cv2.line(img, p, q, halo, w + 2, cv2.LINE_AA)          # a dark halo keeps thin lines readable on any footage
    cv2.line(img, p, q, col, w, cv2.LINE_AA)


def clip_crop(d, W, H):
    """One 16:9 crop per clip around the head, hands and shoulders: 1st-99th percentile extent over the clip."""
    pts = [d["face_lm"][:, [10, 152, 234, 454], :2].reshape(-1, 2), d["hand_lm"][..., :2].reshape(-1, 2)]
    sh = d["pose_lm"][:, [11, 12], :]
    pts.append(sh[sh[..., 3] > 0.5][:, :2])
    P = np.concatenate(pts) * [W, H]
    P = P[np.isfinite(P).all(1)]
    if len(P) < 5:
        return 0.0, 0.0, float(min(W, H * 16 / 9)), float(min(W, H * 16 / 9)) * 9 / 16
    lo, hi = np.percentile(P, 1, axis=0), np.percentile(P, 99, axis=0)
    lo[1] -= 0.15 * (hi[1] - lo[1])                          # headroom above the forehead
    c, size = (lo + hi) / 2, (hi - lo) * 1.25
    w = float(min(max(size[0], size[1] * 16 / 9, 0.3 * W), W, H * 16 / 9))
    h = w * 9 / 16
    return float(np.clip(c[0] - w / 2, 0, W - w)), float(np.clip(c[1] - h / 2, 0, H - h)), w, h


def pick_frames(aux, d, n, fps, k=4):
    """Frames that show measured events, >= 0.8 s apart, topped up with evenly spaced ones: [(time s, label)]."""
    fa, ha = aux["face"], aux["hands"]
    cand = []
    if fa:
        bl = sorted(fa["blinks"], key=lambda b: -b["depth"])[:1]
        cand += [(b["t"], f"blink ({b['fwhm_s']:.2f} s)") for b in bl]
        sh = sorted(fa["shifts"], key=lambda s: -s["amp"])[:1]
        cand += [((s["t_on"] + s["t_off"]) / 2, f"gaze shift {s['amp']:.0f} deg") for s in sh]
        jaw = np.where(fa["face_ok"], fa["jaw"], np.nan)
        if np.isfinite(jaw).any() and np.nanmax(jaw) > 0.2:
            i = int(np.nanargmax(jaw))
            cand.append((i / fps, f"mouth open {jaw[i]:.2f}"))
        yaw = np.where(fa["face_ok"], np.abs(fa["yaw"]), np.nan)
        if np.isfinite(yaw).any():
            i = int(np.nanargmax(yaw))
            cand.append((i / fps, f"head yaw {fa['yaw'][i]:+.0f} deg"))
    g = sorted(ha.get("gestures", []), key=lambda x: -x["peak_hw_s"])[:1]
    cand += [(x["t"] + x["dur_s"] / 2, "hand gesture") for x in g]
    t_end = n / fps
    chosen = []
    for t, lab in cand:
        if 0.2 < t < t_end - 0.2 and all(abs(t - u) >= 0.8 for u, _ in chosen):
            chosen.append((t, lab))
        if len(chosen) == k:
            break
    for t in np.linspace(0.1 * t_end, 0.9 * t_end, 8):
        if len(chosen) >= k:
            break
        if all(abs(t - u) >= 0.8 for u, _ in chosen):
            chosen.append((float(t), "typical"))
    return sorted(chosen)


def draw_tile(bgr, d, aux, fi, crop, label, cid, W, H, pal):
    x0, y0, w, h = crop
    s = TILE[0] / w
    img = cv2.resize(bgr[int(y0):int(y0 + h), int(x0):int(x0 + w)], TILE, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    pt = lambda xy: (int(round((xy[0] * W - x0) * s)), int(round((xy[1] * H - y0) * s)))
    halo, text = _bgr(pal, "base"), _bgr(pal, "text")
    fps = float(d["fps"])
    fa = aux["face"]
    lm = d["face_lm"][fi]
    if np.isfinite(lm[0, 0]):
        for ring, col in ((FACE_OVAL, "iris"), (LIPS, "rose"), (EYE_L, "foam"), (EYE_R, "foam")):
            pts = [pt(lm[i]) for i in ring]
            for a, b in zip(pts, pts[1:] + pts[:1]):
                _seg(img, a, b, _bgr(pal, col), halo, 1)
        cv2.circle(img, pt(lm[1]), 3, _bgr(pal, "gold"), -1, cv2.LINE_AA)
    pl = d["pose_lm"][fi]
    for a, b in ARM_BONES:
        if pl[a, 3] > 0.5 and pl[b, 3] > 0.5:
            _seg(img, pt(pl[a]), pt(pl[b]), _bgr(pal, "love"), halo, 2)
    for k in range(2):
        hl = d["hand_lm"][fi, k]
        if not np.isfinite(hl[0, 0]):
            continue
        for a, b in HAND_BONES:
            _seg(img, pt(hl[a]), pt(hl[b]), _bgr(pal, "pine"), halo, 1)
        cv2.circle(img, pt(hl[8]), 4, _bgr(pal, "gold"), -1, cv2.LINE_AA)
    lines = [f"{cid}  {fi / fps:.2f} s  {label}"]
    if fa and np.isfinite(fa["pitch"][fi]):
        lines.append(f"pitch {fa['pitch'][fi]:+.0f} yaw {fa['yaw'][fi]:+.0f} roll {fa['roll'][fi]:+.0f} deg")
        bits = []
        if np.isfinite(fa["closure"][fi]):
            bits.append(f"lids {fa['closure'][fi]:.2f}")
        if np.isfinite(fa["jaw"][fi]):
            bits.append(f"jaw {fa['jaw'][fi]:.2f}")
        if bits:
            lines.append("  ".join(bits))
    box_w = 8 + max(cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0] for ln in lines)
    over = img.copy()
    cv2.rectangle(over, (0, 0), (box_w, 6 + 16 * len(lines)), halo, -1)
    img = cv2.addWeighted(over, 0.8, img, 0.2, 0)
    for j, ln in enumerate(lines):
        cv2.putText(img, ln, (6, 17 + 16 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text, 1, cv2.LINE_AA)
    return img


def make_sheet(rs, out, per_row=4, palette="rose-pine-moon"):
    """Write the contact sheet of every clip of the set that has both its track and its video. Returns a report dict."""
    pal = PAL.get(palette)
    rows, shown, skipped = [], [], []
    for entry in rs.require_clips():
        cid = entry["id"]
        if not rs.track_path(cid).exists() or not rs.video_path(cid).exists():
            skipped.append({"id": cid, "why": "track missing" if not rs.track_path(cid).exists() else "video missing (cleaned?): `mk ref track` downloads it again"})
            continue
        d = dict(np.load(rs.track_path(cid)))
        fps, n, W, H = float(d["fps"]), int(d["n"]), int(d["W"]), int(d["H"])
        _, aux = measure.analyze_clip(entry, d)
        picks = pick_frames(aux, d, n, fps, per_row)
        want = {int(round(t * fps)): lab for t, lab in picks}
        crop = clip_crop(d, W, H)
        cap = cv2.VideoCapture(str(rs.video_path(cid)))
        tiles = []
        for i in range(min(max(want) + 1, n)):
            if not cap.grab():
                break
            if i in want:
                ok, bgr = cap.retrieve()
                if ok:
                    tiles.append(draw_tile(bgr, d, aux, i, crop, want[i], cid, W, H, pal))
        cap.release()
        while len(tiles) < per_row:
            tiles.append(np.full((TILE[1], TILE[0], 3), _bgr(pal, "base"), np.uint8))
        gap = np.full((TILE[1], GAP, 3), _bgr(pal, "base"), np.uint8)
        row = np.concatenate([x for t in tiles[:per_row] for x in (t, gap)][:-1], axis=1)
        rows += [row, np.full((GAP, row.shape[1], 3), _bgr(pal, "base"), np.uint8)]
        shown.append({"id": cid, "frames": [{"t": round(t, 2), "label": lab} for t, lab in picks]})
    if not rows:
        raise RefUsage("nothing to draw: no clip of the set has both a track and its downloaded video (`mk ref track`)")
    sheet = np.concatenate(rows[:-1], axis=0)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return {"out": str(out), "size": [sheet.shape[1], sheet.shape[0]], "bytes": os.path.getsize(out), "clips": shown, "skipped": skipped}
