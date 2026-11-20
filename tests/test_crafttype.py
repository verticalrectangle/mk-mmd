"""Craft type (docs/design.md: Text, Craft type): a lyric line that arrives in units (the whole line, or chunks of its
words that land together on one frame), each unit one rigid piece through its arrival and its leaving, laid by `mk post`
over the frame effects or, with `under = true`, under them so they act on it. Synthetic timelines (no lyrics), tiny
frames, a font of this machine."""
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mkmmd import crafttype as CRF
from mkmmd import postframes as PF
from mkmmd.cli import common as CM
from mkmmd.core import crafttype as CT
from mkmmd.core import palette as PAL
from mkmmd.core.typeset import TextError
from mkmmd.project import Project

DAWN = PAL.get("rose-pine-dawn")
FPS = 30
TL = {"lines": [{"start": 0.5, "end": 2.0, "words": [
    {"text": "ab", "start": 0.50, "end": 0.70}, {"text": "cd", "start": 0.80, "end": 1.00},
    {"text": "ef", "start": 1.13, "end": 1.40}, {"text": "gh", "start": 1.50, "end": 2.00}]}]}


def system_font():
    """A TTF of this machine (the letters need a real font), else the test is skipped."""
    if shutil.which("fc-match"):
        p = subprocess.run(["fc-match", "-f", "%{file}", "sans:bold"], capture_output=True, text=True).stdout.strip()
        if p.lower().endswith((".ttf", ".otf")) and Path(p).is_file():
            return p
    for p in ("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if Path(p).is_file():
            return p
    pytest.skip("no TrueType font on this machine")


def entry(**over):
    e = {"name": "ly", "screen": True, "font": "x.ttf", "craft": {"kind": "sticker"},
         "lyrics": {"timeline": "tl.json", "line": 1}}
    for k, v in over.items():
        e[k] = {**e[k], **v} if isinstance(v, dict) and isinstance(e.get(k), dict) else v
    return e


# ------------------------------------------------------------------------------------------------------ units
def at_rest(p, u, f):
    if "center" not in u:
        CT.layout(p, [(4.0, 1.5)] * len(p["units"]))
    st = CT.state(p, u, f, FPS)
    return st is not None and np.allclose(st["center"], u["center"]) and abs(st["sx"] * st["sy"] - 1.0) < 0.05


def test_chunks_land_together_and_are_at_rest_on_their_moment_less_the_lead():
    late = {"words": [1, 2], "t": 0.9, "late": True}                          # 0.4 s after its first word, on purpose
    p = CT.plan(entry(lyrics={"units": [late, {"words": [3, 4]}]}), TL, DAWN, FPS, (320, 180))
    a, b = p["units"]
    assert [w.index for w in a["words"]] == [1, 2] and [w.index for w in b["words"]] == [3, 4]
    for u, t in ((a, 0.9), (b, 1.13)):                                        # a given time; its first word's onset
        f = math.floor((t - CT.LEAD) * FPS + 0.4)
        assert u["contact"] == f and at_rest(p, u, f) and not at_rest(p, u, f - 1)   # the approach is before it
        assert u["land"] < f and CT.on_screen(p, u["land"] - 1).count(u) == 0 and u in CT.on_screen(p, u["land"])
    assert CT.on_screen(p, b["land"] - 1) == [a] and CT.on_screen(p, b["land"]) == [a, b]
    whole = CT.plan(entry(lead=0), TL, DAWN, FPS, (320, 180))["units"]        # by default the whole line, at once
    assert len(whole) == 1 and len(whole[0]["words"]) == 4 and whole[0]["contact"] == math.floor(0.5 * FPS + 0.4)
    first = CT.plan(entry(lyrics={"units": [{"words": [1, 2], "t": 0.0}]}), TL, DAWN, FPS, (320, 180))
    u = first["units"][0]                                                     # sung on the clip's first frame:
    assert u["land"] < 0 and u in CT.on_screen(first, 0) and at_rest(first, u, 0)   # already down there
    with pytest.raises(TextError, match=r"words \[2\] are in another unit"):
        CT.plan(entry(lyrics={"units": [{"words": [1, 2]}, {"words": [2, 3]}]}), TL, DAWN, FPS, (320, 180))
    with pytest.raises(TextError, match=r"0.40 s after its first word"):      # a line is there when it is sung
        CT.plan(entry(lyrics={"units": [{"words": [1, 2], "t": 0.9}, {"words": [3, 4]}]}), TL, DAWN, FPS, (320, 180))


def test_the_pen_runs_ahead_and_each_word_is_written_by_its_own_moment():
    font = system_font()
    spec = entry(font=font, size=0.16, craft={"kind": "masking"}, lyrics={"to": 1.9})
    cr = CRF.Crafts([("ly", spec)], (640, 360), FPS, 0, DAWN, lambda r: r, lambda r: TL)
    p = cr.plans[0]
    u = p["units"][0]
    lead = p["craft"]["arrive"]["lead"]
    assert u["contact"] == math.floor((0.5 - CT.LEAD - lead) * FPS + 0.4)        # the pen starts ahead of the voice
    for i, w in enumerate(u["words"]):
        due = math.floor((w.start - CT.LEAD) * FPS + 0.4)
        assert CT.pen_progress(u, due, FPS) >= i + 1                         # each word laid by its moment
        assert CT.pen_progress(u, math.floor(w.start * FPS + 0.4), FPS) >= i + 1   # and so by its onset
    sp = cr._sprite(p, u)
    full = sp.at(len(u["words"]))
    first = (sp.ink[..., 3] > 0.05) & (sp.order <= sp.marks[0] - CRF.MARK)    # the first word's ink
    assert first.sum() > 50 and np.allclose(sp.at(1.0)[first], full[first], atol=1e-4)
    assert sp.at(0.5)[first][..., 3].sum() < 0.9 * full[first][..., 3].sum()   # and half of it at half way


def test_rows_split_words_into_rows_of_about_equal_width_in_order():
    assert CT.split_rows([3, 3, 3, 3], 2) == [[0, 1], [2, 3]]
    assert CT.split_rows([8, 1, 1, 1, 1], 2) == [[0], [1, 2, 3, 4]]
    assert CT.split_rows([2, 2], 5) == [[0], [1]]                              # never more rows than words
    assert CT.split_rows([5, 1, 1], 1) == [[0, 1, 2]]


def _centroid(patch, x0, y0, rgb):
    """The centroid (frame px) of the solid pixels of `patch` nearest the colour `rgb` (of the sticker's two inks)."""
    a = patch[..., 3]
    col = patch[..., :3] / np.maximum(a, 1e-6)[..., None]
    d = np.linalg.norm(col - np.asarray(rgb, np.float32), axis=-1)
    m = (a > 0.98) & (d < 0.12)
    ys, xs = np.nonzero(m)
    assert len(xs) > 30
    return np.array([xs.mean() + x0, ys.mean() + y0])


def test_a_unit_keeps_its_pieces_in_place_through_its_arrival_and_its_leaving():
    font = system_font()
    spec = entry(font=font, size=0.16, craft={"colors": ["love", "pine"], "rim": False, "shadow": False},
                 lyrics={"words": [1, 2], "to": 1.6, "leave": "drop"})
    cr = CRF.Crafts([("ly", spec)], (640, 360), FPS, 0, DAWN, lambda r: r, lambda r: TL)
    p = cr.plans[0]
    u = p["units"][0]
    rgb = [np.asarray(CT.resolve_colour(c, DAWN), np.float32) for c in ("love", "pine")]

    def pieces(f):
        (patch, x0, y0), = cr.patches(f, False)
        A = np.asarray(CT.matrix(CT.state(p, u, f, FPS), (640, 360)))[:, :2]
        return _centroid(patch, x0, y0, rgb[0]), _centroid(patch, x0, y0, rgb[1]), A

    a0, b0, A0 = pieces(u["land"] + 30)                                        # at rest
    rest = np.linalg.solve(A0, b0 - a0)                                       # the second word from the first, unit's own
    assert np.linalg.norm(rest) > 20
    leave = u["leave_frame"]
    for f in (u["land"], u["land"] + 1, u["land"] + 3, u["land"] + 5, leave + 2, leave + 5):
        a, b, A = pieces(f)
        st = CT.state(p, u, f, FPS)
        assert abs(st["sx"] - 1) + abs(st["sy"] - 1) + abs(st["rot"] - u["tilt"]) + abs(st["center"][1] - u["center"][1]) > 1e-3
        assert np.linalg.norm((b - a) - A @ rest) < 1.5, f                    # one map carries both words


# ------------------------------------------------------------------------------------------------------ under the effects
TINT = '[[fx]]\nkind = "tint"\nfrom = 0.0\nto = 1.6\ncolor = "#204060"\ndark = "#204060"\nlight = "#204060"\n'
HALF = ('[[fx]]\nname = "half"\nkind = "tint"\nfrom = 0.0\nto = 1.6\ncolor = "#a03020"\ndark = "#a03020"\n'
        'light = "#a03020"\nstrength = [[0.0, 0.5], [1.6, 0.5]]\n')


def _project(root, under, font, fx=TINT):
    root.mkdir(parents=True, exist_ok=True)
    (root / "tl.json").write_text(json.dumps(TL), encoding="utf-8")
    (root / "mk.toml").write_text(
        f'[project]\nname = "t"\nfps = {FPS}\nframe0 = 1\nduration = 1.6\n[[output]]\nname = "wide"\nsize = [160, 90]\n'
        '[look]\npalette = "rose-pine-dawn"\n[[shot]]\nname = "a"\nfrom = 0.0\nto = 1.6\n' + fx +
        '[post]\nsaturation = 1.0\ncontrast = 1.0\nvignette = 0.0\nfloor = false\ngrain = { amount = 0.0 }\n'
        f'[[text]]\nname = "ly"\nscreen = true\nunder = {json.dumps(under)}\nfont = "{font}"\n'
        'size = 0.2\ncraft = { kind = "sticker" }\nlyrics = { timeline = "tl.json", line = 1, words = [1, 2], to = 1.5 }\n',
        encoding="utf-8")
    src = root / "renders" / "draft" / "wide"
    src.mkdir(parents=True, exist_ok=True)
    files = {}
    for f in range(1, 1 + 48):
        files[f] = src / f"{f:05d}.png"
        Image.new("RGB", (160, 90), (200, 190, 180)).save(files[f])
    proj = Project.load(root)
    return PF.Output(proj, CM.cut_plan(proj), "wide", src, files, sorted(files), (160, 90))


def test_under_lays_the_type_before_the_frame_effects_and_the_default_after_them(tmp_path):
    font = system_font()
    k = 15 + 20                                     # Blender frame 36: the line at rest (it landed on clip frame 15)
    under = np.frombuffer(_project(tmp_path / "u", True, font).finish(k)[0], np.uint8).reshape(90, 160, 3)
    over = np.frombuffer(_project(tmp_path / "o", False, font).finish(k)[0], np.uint8).reshape(90, 160, 3)
    flat = np.array([0x20, 0x40, 0x60])
    assert np.abs(under.astype(int) - flat).max() <= 1                       # the tint took the type with the picture
    off = np.abs(over.astype(int) - flat).max(axis=-1)
    assert (off > 60).sum() > 200 and (off <= 1).mean() > 0.4                 # on top: the type stands on the tint


def test_under_a_named_effect_lays_the_type_after_the_effects_before_it_and_under_that_one(tmp_path):
    font = system_font()
    k = 15 + 20
    fx = TINT.replace('[[fx]]\n', '[[fx]]\nname = "flat"\n') + HALF

    def frame(name, under):
        raw = _project(tmp_path / name, under, font, fx).finish(k)[0]
        return np.frombuffer(raw, np.uint8).reshape(90, 160, 3).astype(float)
    over, half, first = frame("o", False), frame("h", "half"), frame("f", "flat")
    bg = (np.array([0x20, 0x40, 0x60]) + np.array([0xa0, 0x30, 0x20])) / 2.0   # both tints, half the second
    assert np.abs(first - bg).max() <= 1.5                                    # before the flat tint: swallowed by it
    for pic in (over, half):
        assert np.abs(pic - bg).max(axis=-1).mean() < 40                      # the rest is the two tints
    d_over, d_half = np.abs(over - bg).max(axis=-1), np.abs(half - bg).max(axis=-1)
    solid = d_over > 60
    assert solid.sum() > 150                                                  # after the flat tint, under the half one:
    assert 0.3 < np.median(d_half[solid] / d_over[solid]) < 0.85              # the type there, but tinted with it
    with pytest.raises(PF.SetupError, match="no \\[\\[fx\\]\\] has that name"):
        _project(tmp_path / "x", "nope", font, fx)
