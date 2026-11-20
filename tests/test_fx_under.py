"""`[[fx]] under = true` (docs/design.md: Frame effects): an effect that belongs to the cut's own picture, drawn on it
before the cut effects compose, so a transition carries it off with the outgoing shot; outside every window the frame is
the one the plain table order gives. Two flat shots peeled one off the other, tiny frames."""
import numpy as np
import pytest
from PIL import Image

from mkmmd import postframes as PF
from mkmmd.cli import common as CM
from mkmmd.core import framefx as FX
from mkmmd.core import palette as PAL
from mkmmd.core import transition as TR
from mkmmd.project import Project

FPS, W, H = 30, 160, 90
A, B = (200, 60, 40), (40, 160, 90)                 # the outgoing shot's flat colour, the incoming one's
FLAT = np.array([0x20, 0x40, 0x60])                 # what the tint makes of any picture
TINT = ('[[fx]]\nkind = "tint"\nfrom = 0.0\nto = 1.6\ncolor = "#204060"\ndark = "#204060"\nlight = "#204060"\n'
        '{under}')
MOSH = '[[fx]]\nkind = "mosh"\nfrom = 0.2\nto = 1.6\nblock = 8\n{under}'


def _output(root, fx, under, moving=False):
    """An Output over shots a (0 .. 0.8 s) and b (0.8 .. 1.6 s), peeled one off the other at 0.8 s over 0.4 s, with
    the `[[fx]]` table `fx` (its entries `under` or not); the cut's frames flat (with `moving`, a square crossing each),
    the incoming shot's plates over the window."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "mk.toml").write_text(
        f'[project]\nname = "t"\nfps = {FPS}\nframe0 = 1\nduration = 1.6\n[[output]]\nname = "wide"\nsize = [{W}, {H}]\n'
        '[look]\npalette = "rose-pine-dawn"\n[[shot]]\nname = "a"\nfrom = 0.0\nto = 0.8\n[[shot]]\nname = "b"\n'
        'from = 0.8\nto = 1.6\n[[transition]]\nat = 0.8\nkind = "peel"\ndur = 0.4\n'
        + fx.format(under="under = true\n" if under else "") +
        '[post]\nsaturation = 1.0\ncontrast = 1.0\nvignette = 0.0\nfloor = false\ngrain = { amount = 0.0 }\n',
        encoding="utf-8")
    src = root / "renders" / "draft" / "wide"
    src.mkdir(parents=True, exist_ok=True)
    proj = Project.load(root)
    plan = CM.cut_plan(proj)

    def picture(colour, f):
        im = Image.new("RGB", (W, H), colour)
        if moving:
            x = 4 + 3 * f
            im.paste((250, 250, 250), (x, 30, x + 20, 50))
        return im
    files = {}
    for f in range(1, 1 + 48):
        files[f] = src / f"{f:05d}.png"
        picture(A if TR.in_force(plan, "a", f) else B, f).save(files[f])
    tr, = plan["transitions"]
    for f in range(tr["first"], tr["last"] + 1):
        (src / TR.plate_rel("b", f)).parent.mkdir(parents=True, exist_ok=True)
        picture(B, f).save(src / TR.plate_rel("b", f))
    return PF.Output(proj, plan, "wide", src, files, sorted(files), (W, H)), tr


def _frames(out):
    """Every frame finished in order, (H, W, 3) ints, by Blender frame."""
    return {f: np.frombuffer(out.finish(k)[0], np.uint8).reshape(H, W, 3).astype(int) for k, f in enumerate(out.frames)}


def _near(pic, rgb, tol=2):
    return np.abs(pic - np.asarray(rgb)).max(axis=-1) <= tol


def test_an_under_effect_goes_off_with_the_outgoing_shot_and_spares_the_incoming_plate(tmp_path):
    under, tr = _output(tmp_path / "u", TINT, True)
    over, _ = _output(tmp_path / "o", TINT, False)
    got, plain = _frames(under), _frames(over)
    mid = (tr["first"] + tr["last"]) // 2 + 1                 # half peeled: both shots show
    pic = got[mid]
    tinted, bare = _near(pic, FLAT).mean(), _near(pic, B).mean()
    assert tinted > 0.1 and bare > 0.1                        # the sheet still on is tinted, the shot under it not
    assert not _near(pic, A, 30).any()                        # nothing of the outgoing shot's own colour is left
    assert _near(plain[mid], FLAT).mean() > 0.97              # without `under`: the tint covers both
    for f in got:                                             # outside the window: the very same frames
        if not tr["first"] <= f <= tr["last"]:
            assert np.array_equal(got[f], plain[f]), f
    assert _near(got[tr["last"] + 1], FLAT).all()             # past the cut the incoming shot is the cut's: tinted


def test_a_mosh_under_the_cut_keeps_its_frames_outside_the_window(tmp_path):
    under, tr = _output(tmp_path / "u", MOSH, True, moving=True)
    over, _ = _output(tmp_path / "o", MOSH, False, moving=True)
    got, plain = _frames(under), _frames(over)
    outside = [f for f in got if not tr["first"] <= f <= tr["last"]]
    assert any(not np.array_equal(plain[f], plain[1]) for f in outside if f > 7)   # the mosh drags something
    for f in outside:
        assert np.array_equal(got[f], plain[f]), f
    mid = (tr["first"] + tr["last"]) // 2 + 1
    assert _near(got[mid], B, 30).mean() > 0.1                # in the window the incoming plate shows, unmoshed


def test_under_is_true_or_false():
    with pytest.raises(FX.FxError, match="under"):
        FX.normalize({"kind": "tint", "from": 0, "to": 1, "under": "yes"}, PAL.get("rose-pine-dawn"))
