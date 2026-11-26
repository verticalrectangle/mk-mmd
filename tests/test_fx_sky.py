"""Skies (mkmmd.core.sky: segments, smears; mkmmd.sky: the clips through the grade by warmth, the smear between them) on
small synthetic clips."""
import cv2
import numpy as np
import pytest

from mkmmd import footage as FT
from mkmmd import framefx as FF
from mkmmd.core import framefx as FX
from mkmmd.core import palette as PAL
from mkmmd.core import table as TB
from mkmmd.core import transition as TR

FPS, F0 = 30, 1
DAWN = PAL.get("rose-pine-dawn")
W, H = 216, 384                                 # a tall output; the clips are its size, seen at their own size


def write(path, frames):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(FPS), (W, H))
    for f in frames:
        vw.write(cv2.cvtColor(np.ascontiguousarray(f, np.uint8), cv2.COLOR_RGB2BGR))
    vw.release()


def frames(tmp_path, sky):
    data = {"project": {"duration": 4.0}, "output": [{"name": "o", "size": [W, H]}],
            "shot": [{"name": "a", "from": 0.0, "to": 4.0}], "fx": [sky]}
    pl = TR.plan(data, FPS, F0, DAWN)
    under = np.zeros((H, W, 3), np.float32)
    return FF.FrameFx(pl["fx"], (W, H), fps=FPS, frame0=F0, frame_of=lambda g: under,
                      point_of=lambda key, g: ((W / 2, H / 2), None), footage_of=FT.source(tmp_path))


def ramp_at(stops, u):
    """A ramp of TB.RAMPS (its [u, colour] stops) at u, as the grade lays it: linear between the stops' colours."""
    us = [s[0] for s in stops]
    cs = np.array([FX._colour(s[1], DAWN, "test") for s in stops])
    return np.array([np.interp(u, us, cs[:, k]) for k in range(3)])


def test_cool_pixels_take_the_cool_ramp_warm_ones_the_warm_ramp_over_the_stretched_luma_and_a_lean_tips_the_rest(
        tmp_path):
    patches = {"dark": (20, 20, 20), "light": (235, 235, 235), "warm": (200, 110, 40), "cool": (40, 125, 200),
               "grey": (127, 127, 127)}
    pic = np.zeros((H, W, 3), np.uint8)
    band = H // len(patches)
    for i, c in enumerate(patches.values()):
        pic[i * band:(i + 1) * band] = c
    pic[len(patches) * band:] = patches["grey"]
    write(tmp_path / "c.mp4", [pic] * 3)
    lum = {k: 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2] for k, c in patches.items()}
    u = {k: (v - lum["dark"]) / (lum["light"] - lum["dark"]) for k, v in lum.items()}    # the stretch: 1st .. 99th

    def drawn(lean):
        fx = frames(tmp_path, {"kind": "sky", "from": 0.0, "to": 1.0,
                               "clips": [{"t": 0.0, "clip": "c.mp4", "lean": lean, "view": {"height": 1.0}}]})
        out = fx.frame(F0 + 1, np.zeros((H, W, 3), np.float32))
        return {k: out[i * band + band // 2, W // 2] for i, k in enumerate(patches)}

    got = drawn(0.0)
    cool, warm = TB.RAMPS["cool"], TB.RAMPS["warm"]
    tol = 0.03
    assert np.abs(got["dark"] - ramp_at(cool, 0.0)).max() < tol                      # text at the low end ...
    assert np.abs(got["light"] - ramp_at(cool, 1.0)).max() < tol                     # ... surface at the high one
    assert np.abs(got["warm"] - ramp_at(warm, u["warm"])).max() < tol                # red over blue: the warm ramp
    assert np.abs(got["cool"] - ramp_at(cool, u["cool"])).max() < tol                # blue over red: the cool ramp
    half = 0.5 * (ramp_at(cool, u["grey"]) + ramp_at(warm, u["grey"]))
    assert np.abs(got["grey"] - half).max() < tol                                    # neutral: half of each
    tipped = drawn(0.5)
    assert np.abs(tipped["grey"] - ramp_at(warm, u["grey"])).max() < tol             # a warm lean tips it warm
    assert np.abs(tipped["warm"] - got["warm"]).max() < tol                          # the warm stay warm


def test_a_smear_lands_on_its_frame_the_old_picture_moved_along_the_new_clips_motion_and_ends_on_the_new_clip(
        tmp_path):
    rng = np.random.default_rng(3)

    def texture(w, sigma):
        t = cv2.GaussianBlur(rng.random((H, w)).astype(np.float32), (0, 0), sigma)
        return (t - t.min()) / (t.max() - t.min())
    blobs = texture(W, 5)
    old = np.dstack([blobs * 180 + 60, blobs * 100 + 40, blobs * 40 + 20]).astype(np.uint8)        # warm, still
    wide = texture(W + 8 * 40, 8)

    def sliding(k):                                    # a cool texture running right 8 px a frame
        s = wide[:, 8 * 40 - 8 * k:8 * 40 - 8 * k + W]
        return np.dstack([s * 40 + 20, s * 100 + 60, s * 150 + 90]).astype(np.uint8)
    write(tmp_path / "old.mp4", [old] * 40)
    write(tmp_path / "new.mp4", [sliding(k) for k in range(40)])
    t_c = 0.5                                          # the change: clip frame 15
    sky = {"kind": "sky", "from": 0.0, "to": 1.2,
           "clips": [{"t": 0.0, "clip": "old.mp4", "view": {"height": 1.0}},
                     {"t": t_c, "clip": "new.mp4", "in": 0.5, "view": {"height": 1.0},
                      "smear": {"dur": 0.4, "block": 40, "gain": 2.0, "bleed": 0.0}}]}
    fx = frames(tmp_path, sky)
    zero = np.zeros((H, W, 3), np.float32)
    fc = FX.frame_of(t_c, FPS, F0)
    seq = {g: fx.frame(g, zero) for g in range(fc - 2, FX.frame_of(t_c + 0.4, FPS, F0) + 1)}
    ref = frames(tmp_path, sky)
    old_at = {g: ref.sky_segment(ref.fxs[0], 0, ref._t(g)) for g in (fc - 1, fc)}
    new_at = {g: ref.sky_segment(ref.fxs[0], 1, ref._t(g)) for g in (fc, max(seq))}
    assert np.abs(seq[fc - 1] - old_at[fc - 1]).max() < 1e-6                         # the frame before: the old clip
    assert np.abs(seq[fc] - seq[fc - 1]).mean() > 0.01                              # on the change's own frame it moves
    assert np.abs(seq[fc] - new_at[fc]).mean() > 0.05                                # still the old picture, not the new

    def luma(p):
        return np.ascontiguousarray(p @ np.float32([0.2126, 0.7152, 0.0722]), np.float64)
    (dx, dy), _ = cv2.phaseCorrelate(luma(seq[fc - 1]), luma(seq[fc]))
    assert dx == pytest.approx(16.0, abs=3.0) and abs(dy) < 1.0                       # along the new motion, gain 2
    assert np.abs(seq[max(seq)] - new_at[max(seq)]).max() < 1e-6                     # the window's end: the new clip
