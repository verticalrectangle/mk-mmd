"""Pinned stickers (mkmmd.core.pin, mkmmd.pin: a table's figure riding a seat its clip's camera moves, contact shadows,
pieces of the clip in front of her; mkmmd.pintrack: a seat's track measured in a clip) on small synthetic clips whose
camera drifts, zooms and turns."""
import math

import cv2
import numpy as np
import pytest

from mkmmd import framefx as FF
from mkmmd import footage as FT
from mkmmd import pintrack as PT
from mkmmd.core import palette as PAL
from mkmmd.core import transition as TR

FPS, F0, CLIP_FPS = 30, 1, 24.0
DAWN = PAL.get("rose-pine-dawn")
W, H = 90, 160                                   # a tall output
CW, CH = 160, 90                                 # the clip
N = 24                                           # its frames: one second


def seat(i):
    """The seat in clip frame i (px, its size against frame 0): the camera drifts and zooms in on it."""
    return (52.0 + 1.7 * i, 50.0 - 0.6 * i), 1.0 + 0.015 * i


def drifting_clip(path, wire=False):
    """A light sky with a dark ring round the seat (its outer rim 14 px at frame 0) and, with `wire`, a dark wire 2 px
    thick across it 3 px under the seat, moving with it."""
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), CLIP_FPS, (CW, CH))
    assert vw.isOpened()
    for i in range(N):
        (x, y), k = seat(i)
        f = np.full((CH, CW, 3), 200, np.uint8)
        cv2.circle(f, (int(round(x * 16)), int(round(y * 16))), int(round(12 * k * 16)), (30, 30, 30), int(round(4 * k)),
                   cv2.LINE_AA, 4)
        if wire:
            yy = y + 3 * k
            cv2.line(f, (0, int(round(yy * 16))), (CW * 16, int(round(yy * 16))), (25, 25, 25), 2, cv2.LINE_AA, 4)
        vw.write(f)
    vw.release()
    return path


def rows():
    out = []
    for i in range(N):
        (x, y), k = seat(i)
        out.append([i / CLIP_FPS, (x + 0.5) / CW, (y + 0.5) / CH, k, 0.0])
    return out


def drawer(tmp_path, fx, wire=False):
    drifting_clip(tmp_path / "clip.mp4", wire)
    data = {"project": {"duration": 4.0}, "output": [{"name": "o", "size": [W, H]}],
            "shot": [{"name": "a", "from": 0.0, "to": 4.0}, {"name": "plate", "plate": True}], "fx": [fx]}
    pl = TR.plan(data, FPS, F0, DAWN)

    def figure(g):                               # her plate: a square round its middle, 0.3 metres across
        ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
        return ((np.abs(xs - (W / 2 - 0.5)) <= 0.15 * 0.25 * H) &
                (np.abs(ys - (H / 2 - 0.5)) <= 0.15 * 0.25 * H)).astype(np.float32)

    under = np.full((H, W, 3), (1.0, 0.0, 1.0), np.float32)
    plate = np.full((H, W, 3), (0.2, 0.9, 0.2), np.float32)
    ffx = FF.FrameFx(pl["fx"], (W, H), fps=FPS, frame0=F0, frame_of=lambda g: under,
                     point_of=lambda key, g: ((W / 2 - 0.5, H / 2 - 0.5), 0.25), figure_of=lambda key, g: figure(g),
                     plate_of=lambda shot, g: plate, footage_of=FT.source(tmp_path))
    return ffx, under


def table(**kw):
    (x, y), _ = seat(0)
    return {"kind": "sheet", "from": 0.0, "to": 1.0, "clip": "clip.mp4", "drift": False, "objects": "Fig*",
            "shot": "plate", "edge": False, "shadow": False, "grade": {"map": ["text", "surface"]},
            "view": {"at": [0.45, 0.5], "height": 1.6},
            "track": {"point": "(0, 0, 0)", "at": [(x + 0.5) / CW, (y + 0.5) / CH], "metre": 0.4}, **kw}


def centroid(w):
    """The weighted middle (x, y) of a soft coverage `w`, and its area."""
    ys, xs = np.mgrid[0:w.shape[0], 0:w.shape[1]]
    return np.array([(xs * w).sum(), (ys * w).sum()]) / w.sum(), w.sum()


def test_a_pinned_figure_stays_on_its_seat_as_the_camera_moves_and_steps_with_the_clips_frames(tmp_path):
    ffx, under = drawer(tmp_path, table(pin={"track": rows()}))
    seen = {}
    for g in range(F0, F0 + FPS):
        out = ffx.frame(g, under)
        fig = np.clip((out[..., 1] - out[..., 0]) / 0.7, 0.0, 1.0)      # her sticker, green on a grey and plum table
        c, area = centroid(fig)
        near = np.hypot(*(np.mgrid[0:H, 0:W][::-1] - c[:, None, None])) < 24   # past the clip's edges it is reflected
        ring = np.clip((out.sum(-1) - 2.9) / (1.13 - 2.9), 0.0, 1.0) * near * (fig < 0.02)
        r, _ = centroid(ring)                                            # the seat's ring as the table shows it
        assert np.abs(c - r).max() < 0.25                                # on her seat, not a pixel off, on every frame
        i = int(np.floor((g - F0) / FPS * CLIP_FPS + 1e-6))
        seen.setdefault(i, []).append((c, area))
    for i, got in seen.items():
        assert all(np.allclose(c, got[0][0], atol=1e-4) for c, _ in got)  # she steps with the clip, never between
    k = seat(N - 1)[1] / seat(0)[1]
    assert seen[N - 1][0][1] / seen[0][0][1] == pytest.approx(k * k, rel=0.05)   # and grows as the camera zooms in


def test_contact_shadows_darken_the_seat_only_and_a_keyed_piece_of_the_clip_crosses_in_front_of_her(tmp_path):
    (x, y), _ = seat(0)
    sx, sy, wy = (x + 0.5) / CW, (y + 0.5) / CH, (y + 3.5) / CH          # the seat, and the wire 3 px under it
    pin = {"track": rows(),
           "shadow": {"at": [sx, sy], "size": [0.4, 0.2], "alpha": 0.8,            # straddles the wire: cut to its top
                      "on": [[0.0, wy], [1.0, wy], [1.0, 1.0], [0.0, 1.0]]},
           "front": {"outline": [[sx - 0.2, wy - 0.04], [sx + 0.2, wy - 0.04], [sx + 0.2, wy + 0.04],
                                 [sx - 0.2, wy + 0.04]], "key": {"luma": [0.0, 0.55]}}}
    ground = {k: v for k, v in table().items() if k not in ("objects", "shot", "edge", "shadow", "track")}
    bare, under = drawer(tmp_path, ground, wire=True)
    plain, _ = drawer(tmp_path, table(pin={"track": rows()}), wire=True)
    pinned, _ = drawer(tmp_path, table(pin=pin), wire=True)
    for g in (F0, F0 + 17):
        g0, a, b = bare.frame(g, under), plain.frame(g, under), pinned.frame(g, under)
        fig = (a[..., 1] > 0.8) & (a[..., 0] < 0.35)                       # her sticker, where the pin carried it
        L = g0 @ np.array([0.299, 0.587, 0.114], np.float32)
        rows_dark = np.nonzero((L < 0.5).mean(1) > 0.5)[0]                 # the wire: dark across the whole frame
        darker = (b.sum(-1) < a.sum(-1) - 0.03) & ~fig
        assert darker.any() and np.nonzero(darker)[0].min() >= rows_dark.min() - 1   # none above the seat's edge
        core, sky = fig & (L < 0.45), fig & (L > 0.6)
        assert core.any() and np.abs(b[core] - g0[core]).max() < 0.05       # the wire laid over her where it crosses
        assert np.abs(b[sky] - a[sky]).max() < 0.02                         # and never the sky round it


def test_the_tracker_follows_a_turning_zooming_seat_to_a_tenth_of_a_pixel_and_ignores_the_sky_moving_behind_it(tmp_path):
    rng = np.random.default_rng(3)
    S = 480
    tex = cv2.GaussianBlur(rng.uniform(0, 255, (S, S)).astype(np.float32), (0, 0), 3.0)
    tex = cv2.normalize(tex, None, 30, 225, cv2.NORM_MINMAX)
    sky = cv2.GaussianBlur(rng.uniform(0, 255, (S, 2 * S)).astype(np.float32), (0, 0), 6.0)
    sky = cv2.normalize(sky, None, 60, 200, cv2.NORM_MINMAX)
    cw, ch, n, fps = 320, 240, 12, 25.0
    seat0 = np.array([150.0, 110.0])                              # px of frame 0

    def motion(i):                                              # the camera: drifting, zooming in, turning
        return seat0 + (2.3 * i, -1.1 * i), 1.0 + 0.012 * i, 0.4 * i

    path = tmp_path / "seat.avi"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (cw, ch))
    for i in range(n):
        (x, y), k, deg = motion(i)
        f = cv2.warpAffine(sky, np.float32([[1, 0, -7.0 * i], [0, 1, 0]]), (cw, ch))    # clouds sliding past
        A = cv2.getRotationMatrix2D((S / 2 - 0.5, S / 2 - 0.5), deg, k)
        A[:, 2] += (x - (S / 2 - 0.5), y - (S / 2 - 0.5))         # the texture's middle on the seat
        plate = cv2.warpAffine(tex, A, (cw, ch), flags=cv2.INTER_CUBIC, borderValue=-1)
        thing = cv2.warpAffine(np.ones((S, S), np.float32), A, (cw, ch)) > 0.999
        disc = np.hypot(*(np.mgrid[0:ch, 0:cw][::-1] - np.array([x, y])[:, None, None])) < 60 * k
        f = np.where(thing & disc, plate, f)                       # a rigid round thing over the moving sky
        vw.write(cv2.cvtColor(np.clip(f, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
    vw.release()
    area = [((seat0[0] + 55 * math.cos(a)) / cw, (seat0[1] + 55 * math.sin(a)) / ch) for a in np.linspace(0, 6.2, 24)]
    rows, fit = PT.track(path, ((seat0[0] + 0.5) / cw, (seat0[1] + 0.5) / ch), 0.0, (n - 1) / fps, area=area)
    assert len(rows) == n and all(f["inliers"] >= 12 and f["rms"] < 0.3 for f in fit)
    for i, r in enumerate(rows):
        (x, y), k, deg = motion(i)
        assert r[0] == pytest.approx(i / fps, abs=1e-4)
        assert r[1] * cw - 0.5 == pytest.approx(x, abs=0.1) and r[2] * ch - 0.5 == pytest.approx(y, abs=0.1)
        assert r[3] == pytest.approx(k, rel=2e-3) and r[4] == pytest.approx(deg, abs=0.1)
