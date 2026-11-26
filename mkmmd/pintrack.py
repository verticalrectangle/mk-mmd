"""A seat's track in a clip, for a pinned sticker (docs/design.md: Pinned stickers): the picture round a point of the clip
(a wing's edge, a wire, a tower's top) followed over the clip's frames, as the rows a table's `pin.track` takes,
[video second, x, y, scale, turn], one per clip frame.

    python -m mkmmd.pintrack CLIP --at X,Y --from S --to S [--ref S] [--area X,Y X,Y ...] [--radius R] [--sheet OUT.jpg]

`--at` is the seat (fractions of the clip's frame as the file has it) on the frame at `--ref` (default `--from`), the
reference. Corners of the reference inside `--area` (an outline, fractions) or within `--radius` frame heights of the
point are found and followed into every other frame straight from the reference (pyramidal Lucas-Kanade, starting
where the frame before put them, each checked by following it back), so the track does not drift; a similarity (a
move, a scale and a turn) is fitted to them with RANSAC, so corners that move otherwise (on clouds sliding past, a bird
taking off) are left out. Give the rigid thing the seat is part of, with its edges. The rows' scale and turn are that
similarity's (the size and the turn, degrees counter-clockwise on screen, of the picture there against the reference).
Each frame's fit is reported (`inliers`, the corners it kept, and `rms`, their distance from it in the clip's px), and
`--sheet` draws the tracked point and the outline on crops of every few frames to look at. The frames are read
through mkmmd.footage, as a table reads them. Prints JSON: the rows, the fit per frame, and the frames whose fit is
weak (fewer than `--min-inliers` corners or an `rms` over `--max-rms` px), which want a better outline."""
import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from . import footage as FT

WORK = 1080            # px: the short side the frames are followed at (finer detail is noise for a seat)
CORNERS = 800          # corners found in the reference at most
QUALITY = 0.001        # the weakest corner kept, a share of the strongest (a wing in shadow has faint ones only)
LK = {"winSize": (25, 25), "maxLevel": 4, "criteria": (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, 0.005)}
BACK = 0.4             # px at WORK: a corner followed back to the reference must land this near where it started
RANSAC = 0.8           # px at WORK: a corner this far from the similarity is not on the rigid thing


def _grey(rgb, k):
    g = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY)
    if k != 1.0:
        g = cv2.resize(g, (max(int(round(g.shape[1] * k)), 1), max(int(round(g.shape[0] * k)), 1)),
                       interpolation=cv2.INTER_AREA)
    return g


def _mask(shape, k, size, at, area, radius):
    """The reference's pixels whose corners take part (uint8 0/255) at the working scale `k`: inside `area` (an
    outline, fractions) or within `radius` frame heights of `at`."""
    m = np.zeros(shape, np.uint8)
    if area:
        q = np.array([[(x * size[0]) * k - 0.5, (y * size[1]) * k - 0.5] for x, y in area])
        cv2.fillPoly(m, [np.rint(q * 16).astype(np.int32)], 255, cv2.LINE_8, 4)
    else:
        c = ((at[0] * size[0]) * k - 0.5, (at[1] * size[1]) * k - 0.5)
        cv2.circle(m, (int(round(c[0] * 16)), int(round(c[1] * 16))), int(round(radius * size[1] * k * 16)), 255, -1,
                   cv2.LINE_8, 4)
    return m


def similarity_of(A):
    """(scale, turn degrees counter-clockwise on screen) of the similarity nearest a 2 x 2 linear map `A`."""
    a, b = 0.5 * (A[0, 0] + A[1, 1]), 0.5 * (A[0, 1] - A[1, 0])
    return math.hypot(a, b), math.degrees(math.atan2(b, a))


def _fit(R, F, P0, guess):
    """The similarity (2 x 3) the corners `P0` of `R` moved by into `F` (followed from `guess`, checked back, RANSAC);
    with the corners it kept and their rms distance from it (px). (None, 0, inf) when too few were followed."""
    P1, st, _ = cv2.calcOpticalFlowPyrLK(R, F, P0, guess.astype(np.float32), flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **LK)
    Pb, sb, _ = cv2.calcOpticalFlowPyrLK(F, R, P1, P0.copy(), flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **LK)
    good = (st[:, 0] == 1) & (sb[:, 0] == 1) & (np.linalg.norm(Pb - P0, axis=1) < BACK)
    if good.sum() < 4:
        return None, 0, math.inf
    M, inl = cv2.estimateAffinePartial2D(P0[good], P1[good], method=cv2.RANSAC, ransacReprojThreshold=RANSAC,
                                         maxIters=4000, confidence=0.999, refineIters=20)
    if M is None:
        return None, 0, math.inf
    keep = inl[:, 0] == 1
    a, b = P0[good][keep], P1[good][keep]
    return M.astype(np.float64), int(keep.sum()), float(np.sqrt(((a @ M[:, :2].T + M[:, 2] - b) ** 2).sum(1).mean()))


def _follow(R, F, P0, S):
    """The similarity (2 x 3, reference px -> frame px) that the reference's corners `P0` moved by into frame `F`,
    starting from `S`; with the corners it kept and their rms distance from it (px). Fitted twice: then the frame is
    turned back onto the reference by the first fit and the corners followed again there, where they barely move, so a
    scaled or turned picture is matched as precisely as a still one."""
    S1, n, rms = _fit(R, F, P0, P0 @ S[:, :2].T + S[:, 2])
    if S1 is None:
        return S1, n, rms
    back = cv2.warpAffine(F, cv2.invertAffineTransform(S1), (R.shape[1], R.shape[0]), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)
    Sr, n2, rms2 = _fit(R, back, P0, P0)
    if Sr is None:
        return S1, n, rms
    return S1 @ np.vstack([Sr, [0.0, 0.0, 1.0]]), n2, rms2


def track(clip, at, start, end, ref=None, area=None, radius=0.08, work=WORK):
    """The seat's rows over the clip frames from video second `start` to `end` (each frame's own second, as a table
    reads them: `Clip.index`), followed from the frame at `ref` (default `start`), on which `at` (fractions) is the
    seat: ([[second, x, y, scale, turn], ...], [{inliers, rms (clip px)}, ...])."""
    c = clip if isinstance(clip, FT.Clip) else FT.Clip(clip)
    i0, i1 = c.index(0.0, start), c.index(0.0, end)
    ir = c.index(0.0, start if ref is None else ref)
    if not i0 <= ir <= i1:
        raise ValueError(f"--ref {ref:g} s lies outside the frames tracked ({start:g} .. {end:g} s)")
    k = min(1.0, work / float(min(c.size)))
    R = _grey(c.frame(ir), k)
    P0 = cv2.goodFeaturesToTrack(R, CORNERS, QUALITY, 4, mask=_mask(R.shape, k, c.size, at, area, radius), blockSize=7)
    if P0 is None or len(P0) < 8:
        raise ValueError("fewer than 8 corners on the reference inside the area: give an --area round more of the rigid "
                         "thing the seat is part of")
    P0 = P0.reshape(-1, 2).astype(np.float32)
    p = np.array([at[0] * c.size[0] * k - 0.5, at[1] * c.size[1] * k - 0.5, 1.0])
    fits = {ir: (np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), len(P0), 0.0)}
    for order in (range(ir + 1, i1 + 1), range(ir - 1, i0 - 1, -1)):
        last = fits[ir][0]
        for i in order:
            S, n, rms = _follow(R, _grey(c.frame(i), k), P0, last)
            if S is None:
                raise RuntimeError(f"frame {i} ({i / c.fps:.3f} s): the corners were lost (fewer than 4 followed): a "
                                   f"larger --area, or track a shorter stretch")
            fits[i], last = (S, n, rms), S
    rows, quality = [], []
    for i in range(i0, i1 + 1):
        S, n, rms = fits[i]
        q = S @ p
        scale, turn = similarity_of(S[:, :2])
        rows.append([round(i / c.fps, 5), round(((q[0] + 0.5) / k) / c.size[0], 6),
                     round(((q[1] + 0.5) / k) / c.size[1], 6), round(scale, 5), round(turn, 4)])
        quality.append({"inliers": n, "rms": round(rms / k, 3)})
    return rows, quality


def sheet(clip, rows, area, out, every=4, half=0.12):
    """Crops round the tracked point of every `every`-th row, a crosshair on it and the outline carried with it, side by
    side, for looking at: a point that slides shows as a crosshair leaving its spot."""
    c = clip if isinstance(clip, FT.Clip) else FT.Clip(clip)
    tiles = []
    r0 = rows[0]
    for r in rows[::every]:
        img = cv2.cvtColor(c.frame(c.index(0.0, r[0])), cv2.COLOR_RGB2BGR)
        x, y = r[1] * c.size[0] - 0.5, r[2] * c.size[1] - 0.5
        h = int(round(half * c.size[1]))
        x0, y0 = int(round(x)) - h, int(round(y)) - h
        crop = np.ascontiguousarray(cv2.copyMakeBorder(img, h, h, h, h, cv2.BORDER_CONSTANT)[
            y0 + h:y0 + 3 * h, x0 + h:x0 + 3 * h])
        if area:
            s, a = r[3] / r0[3], math.radians(r[4] - r0[4])
            A = s * np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]])
            q0 = np.array([r0[1] * c.size[0] - 0.5, r0[2] * c.size[1] - 0.5])
            pts = (np.array([[u * c.size[0] - 0.5, v * c.size[1] - 0.5] for u, v in area]) - q0) @ A.T + (x, y)
            cv2.polylines(crop, [np.rint((pts - (x0, y0)) * 16).astype(np.int32)], True, (90, 220, 255), 2,
                          cv2.LINE_AA, 4)
        cx, cy = int(round((x - x0) * 16)), int(round((y - y0) * 16))
        arm, thick = int(round(0.12 * h)) * 16, max(int(round(h / 180)), 1)
        for d in ((1, 0), (0, 1)):
            cv2.line(crop, (cx - arm * d[0], cy - arm * d[1]), (cx + arm * d[0], cy + arm * d[1]), (255, 40, 255), thick,
                     cv2.LINE_AA, 4)
        crop = cv2.resize(crop, (360, 360), interpolation=cv2.INTER_AREA)
        cv2.putText(crop, f"{r[0]:.3f}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        tiles.append(crop)
    rows_img = [np.concatenate(tiles[j:j + 8] + [np.zeros_like(tiles[0])] * (8 - len(tiles[j:j + 8])), 1)
                for j in range(0, len(tiles), 8)]
    cv2.imwrite(str(out), np.concatenate(rows_img, 0), [cv2.IMWRITE_JPEG_QUALITY, 88])


def _points(s):
    return [tuple(float(v) for v in q.split(",")) for q in s.split()]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m mkmmd.pintrack", description=__doc__.split("\n\n")[0])
    ap.add_argument("clip", help="the video (a file)")
    ap.add_argument("--at", required=True, help="the seat X,Y (fractions of the frame) on the reference frame")
    ap.add_argument("--from", dest="start", type=float, required=True, help="the first video second tracked")
    ap.add_argument("--to", dest="end", type=float, required=True, help="the last video second tracked")
    ap.add_argument("--ref", type=float, default=None, help="the reference frame's video second (default --from)")
    ap.add_argument("--area", default=None, help="the outline of what takes part, 'X,Y X,Y ...' (fractions)")
    ap.add_argument("--radius", type=float, default=0.08, help="without --area: frame heights round the point")
    ap.add_argument("--min-inliers", type=int, default=12, help="a frame's fit on fewer corners is listed as weak")
    ap.add_argument("--max-rms", type=float, default=1.0, help="and one whose corners lie farther from it (clip px)")
    ap.add_argument("--sheet", default=None, help="a JPEG of crops round the tracked point, to look at")
    ap.add_argument("--every", type=int, default=4, help="the sheet's crops: every n-th frame")
    a = ap.parse_args(argv)
    cv2.setNumThreads(2)                                    # a tool run beside renders: two cores are plenty
    at = tuple(float(v) for v in a.at.split(","))
    area = _points(a.area) if a.area else None
    clip = FT.Clip(Path(a.clip).expanduser())
    rows, quality = track(clip, at, a.start, a.end, a.ref, area, a.radius)
    weak = [[r[0], q] for r, q in zip(rows, quality) if q["inliers"] < a.min_inliers or q["rms"] > a.max_rms]
    json.dump({"clip": str(clip.path), "fps": clip.fps, "size": list(clip.size), "rows": rows, "fit": quality,
               "weak": weak}, sys.stdout)
    print(flush=True)
    if a.sheet:
        sheet(clip, rows, area, a.sheet, a.every)


if __name__ == "__main__":
    main()
