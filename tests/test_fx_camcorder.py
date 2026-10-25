"""Camcorder ([[fx]] kind "camcorder", mkmmd.core.picturefx and mkmmd.picturefx): its keys, the stamp's clock (ticking
on with the clip in the style it was written in), the blinking REC dot, the tracking band rolling up, the font (a file, a
registry slug, the registry's first `mono` font) and a stamp drawn crisp after the tape's damage, in its corner."""
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from mkmmd import assets as A
from mkmmd import framefx as FF
from mkmmd import picturefx as PF
from mkmmd.core import framefx as FX
from mkmmd.core import palette as PAL
from mkmmd.core import picturefx as PC
from mkmmd.core import transition as TR

FPS, F0 = 30, 1
DAWN = PAL.get("rose-pine-dawn")


def system_font():
    """A TTF of this machine (the stamp needs a real font to draw), else the test is skipped."""
    if shutil.which("fc-match"):
        p = subprocess.run(["fc-match", "-f", "%{file}", "monospace"], capture_output=True, text=True).stdout.strip()
        if p.lower().endswith((".ttf", ".otf")) and Path(p).is_file():
            return p
    for p in ("/usr/share/fonts/TTF/DejaVuSansMono.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"):
        if Path(p).is_file():
            return p
    pytest.skip("no TrueType font on this machine")


def entry(**spec):
    return FX.normalize({"kind": "camcorder", "from": 2.0, "to": 6.0, **spec}, DAWN)


@pytest.mark.parametrize("spec, frag", [
    ({"tick": True}, "tick counts the seconds"),
    ({"time": "25:10", "tick": True}, "not a time of day"),
    ({"time": "8:61 PM", "tick": True}, "not a time of day"),
    ({"time": "AM 8:41 PM", "tick": True}, "expected a time"),
    ({"time": "soon", "tick": True}, "expected a time"),
    ({"corner": "middle"}, "none of bottom-right"),
    ({"indicator": "STOP"}, "is \"PLAY\", \"REC\" or \"none\""),
    ({"date": ""}, "expected a short text"),
    ({"tracking": {"amount": 0.2, "roll": 1}}, "expected a number or {amount, speed}"),
    ({"scanlines": 2}, "expected 0..1"),
])
def test_bad_entries_say_what_is_wrong(spec, frag):
    with pytest.raises(FX.FxError, match=frag.replace("{", r"\{").replace("}", r"\}")):
        entry(**spec)


def test_indicators_and_tracking_read_loosely():
    assert entry(indicator="rec")["indicator"] == "REC"
    assert entry(indicator="none")["indicator"] is None and entry(indicator=False)["indicator"] is None
    assert entry(tracking=0.5)["tracking"] == {"amount": 0.5, "speed": PC.TRACK_SPEED}


@pytest.mark.parametrize("text, later, shown", [
    ("8:41 PM", 65.2, "8:42:05 PM"),
    ("PM 8:41", 0.0, "PM 8:41:00"),
    ("11:59:59 pm", 1.0, "12:00:00 am"),
    ("12:00 AM", 3600.0, "1:00:00 AM"),
    ("08:05", 59.9, "08:05:59"),
    ("23:59:30", 45.0, "00:00:15"),
])
def test_the_clock_ticks_on_in_its_own_style(text, later, shown):
    assert PC.clock_text(PC.clock(text), later) == shown


def test_the_stamp_counts_up_from_the_span_start_and_rec_blinks():
    fx = entry(date="OCT. 23 1999", time="8:41 PM", tick=True, indicator="REC")
    assert PC.stamp(fx, 2.0) == {"lines": ["8:41:00 PM", "OCT. 23 1999"], "indicator": "REC", "dot": True}
    assert PC.stamp(fx, 3.7)["lines"][0] == "8:41:01 PM" and PC.stamp(fx, 3.7)["dot"] is False
    still = entry(time="8:41 PM")
    assert PC.stamp(still, 5.0)["lines"] == ["8:41 PM"]                     # without tick it is shown as written


def test_the_tracking_band_rolls_up_the_frame():
    fx = entry(tracking={"amount": 1.0, "speed": 0.2})
    ys = [PC.tracking_centre(fx, 2.0 + 0.1 * i) for i in range(20)]
    steps = np.diff(ys)
    assert np.all((steps < 0) | (steps > 1.0))                              # up, and round again from below
    assert all(-2 * PC.TRACK_BAND - 1e-9 <= y <= 1 + 2 * PC.TRACK_BAND + 1e-9 for y in ys)


def test_fonts_from_files_slugs_and_the_registry_mono(tmp_path):
    for name in ("deco.ttf", "mono.ttf", "proj.otf"):
        (tmp_path / name).write_bytes(b"x")
    reg = A.Registry(tmp_path / "assets")
    reg.add({"kind": "font", "slug": "deco", "path": str(tmp_path / "deco.ttf"), "tags": ["font"]})
    reg.add({"kind": "font", "slug": "vcr", "path": str(tmp_path / "mono.ttf"), "tags": ["font", "mono"]})
    reg.add({"kind": "model", "slug": "girl", "path": str(tmp_path / "deco.ttf")})
    get = PF.font_source(tmp_path, reg)
    assert get(None) == str(tmp_path / "mono.ttf")                          # the first font tagged mono
    assert get("deco") == str(tmp_path / "deco.ttf")
    assert get("proj.otf") == str(tmp_path / "proj.otf")                    # project-relative
    with pytest.raises(FX.FxError, match="not a font file and not a font in the asset registry"):
        get("girl")
    with pytest.raises(FX.FxError, match="is not a file"):
        get("fonts/none.ttf")
    empty = PF.font_source(tmp_path, A.Registry(tmp_path / "other"))
    with pytest.raises(FX.FxError, match="no font tagged `mono`"):
        empty(None)


def test_a_missing_font_fails_before_any_frame():
    pl = TR.plan({"project": {"duration": 8.0}, "output": [{"name": "t", "size": [64, 36]}],
                  "shot": [{"name": "a", "from": 0.0, "to": 8.0}],
                  "fx": [{"kind": "camcorder", "from": 2, "to": 6, "font": "/nowhere/vcr.ttf"}]}, FPS, F0, DAWN)
    with pytest.raises(FX.FxError, match="is not a file"):
        FF.FrameFx(pl["fx"], (64, 36), fps=FPS, frame0=F0, frame_of=lambda g: None, point_of=None)


def stamped(w, h, **spec):
    font = system_font()
    pl = TR.plan({"project": {"duration": 8.0}, "output": [{"name": "t", "size": [w, h]}],
                  "shot": [{"name": "a", "from": 0.0, "to": 8.0}],
                  "fx": [{"kind": "camcorder", "from": 2, "to": 6, "font": font, "color": "#FFFFFF", **spec}]},
                 FPS, F0, DAWN)
    img = np.full((h, w, 3), 0.35, np.float32)
    ffx = FF.FrameFx(pl["fx"], (w, h), fps=FPS, frame0=F0, frame_of=lambda g: None, point_of=None)
    return ffx.frame(FX.frame_of(3.0, FPS, F0), img)


def test_the_stamp_is_crisp_over_the_tape_in_its_corner():
    w, h = 480, 270
    out = stamped(w, h, date="OCT. 23 1999", time="8:41 PM", indicator=None, noise=0.08, soft=3.0)
    white = np.all(out > 0.999, axis=2)                                      # the stamp's own colour, untouched
    assert white.sum() > 150
    ys, xs = np.nonzero(white)
    assert xs.min() > w * 0.55 and ys.min() > h * 0.6                       # bottom right
    tape = out[: h // 2, : w // 2]
    assert 0.0 < tape.std() and not np.any(np.all(tape > 0.999, axis=2))    # the tape's noise, no stamp there


@pytest.mark.parametrize("corner", ["top-left", "bottom-left", "top-right"])
def test_the_stamp_goes_to_its_corner(corner):
    w, h = 480, 270
    out = stamped(w, h, time="8:41 PM", corner=corner, indicator=None, noise=0.0, tracking=0.0, scanlines=0.0)
    ys, xs = np.nonzero(np.all(out > 0.999, axis=2))
    assert (xs.mean() < w / 2) == corner.endswith("left") and (ys.mean() < h / 2) == corner.startswith("top")


def test_the_indicator_stays_clear_of_the_stamp():
    w, h = 480, 270
    out = stamped(w, h, time="8:41 PM", indicator="PLAY", noise=0.0, tracking=0.0, scanlines=0.0)
    ys, xs = np.nonzero(np.all(out > 0.999, axis=2))
    assert (ys < h / 3).any() and (xs[ys < h / 3] < w / 2).all()             # PLAY at the top left
    rec = stamped(w, h, time="8:41 PM", indicator="REC", corner="top-left", noise=0.0, tracking=0.0, scanlines=0.0)
    red = (rec[..., 0] > 0.8) & (rec[..., 1] < 0.3)
    assert red.any() and np.nonzero(red)[1].min() > w / 2                   # its dot at the top right, lit at 3.0 s
