"""`[[post.span]]`, a part of the clip graded its own way (docs/design.md: Post; mkmmd.post.spans, Spans): the frames a
[[shot]] with the same clip seconds holds, the span's grade keys laid over [post]'s, its fades a linear mix of the two
grades run on the same picture with the same grain, the errors that name a span (before mk post draws anything), the
breathing palette's colours in a span's grade too, `--gpu` leaving a project with spans to the CPU, and a frame outside
every span graded as it always was. Synthetic frames, no Blender."""
import json

import numpy as np
import pytest
from PIL import Image

from mkmmd import post as P
from mkmmd import postframes as PF
from mkmmd import postgpu as PG
from mkmmd.cli import common as CM
from mkmmd.cli import main as MAIN
from mkmmd.cli import post as POST
from mkmmd.core import palette as PAL
from mkmmd.project import Project

FPS, F0, W, H, N = 10, 1, 64, 36, 40            # a 4 s clip: Blender frames 1 to 40
LAST = F0 + N - 1
MOON, DAWN = PAL.get("rose-pine-moon"), PAL.get("rose-pine-dawn")
BASE = ('saturation = 1.08\ncontrast = 1.06\nvignette = 0.06\n'
        'split = { shadows = "iris", highlights = "gold", amount = 0.05 }\ngrain = { amount = 0.03, seed = 3 }\n')
HALATION = 'halation = { strength = 0.38, radius = 0.05, threshold = 0.52, tint = "#ff3ea5" }\n'
OWN = 'saturation = 1.4\ncontrast = 1.2\nvignette = 0.3\nsplit = false\n' + HALATION    # the span's keys
SPAN = '[[post.span]]\nfrom = 1.0\nto = 3.0\nfade = [0.4, 0.5]\n' + OWN                 # frames 11 to 30
WHOLE = OWN + 'grain = { amount = 0.03, seed = 3 }\n'          # a whole [post] that is the span's grade: BASE under OWN


def picture(f):
    """Frame `f`: slopes drifting with the frame, a lamp (for the halation) and a near-black patch (for the floor)."""
    x = np.linspace(0.0, 1.0, W)[None, :]
    y = np.linspace(0.0, 1.0, H)[:, None]
    r, g, b = np.broadcast_arrays(0.2 + 0.6 * x * (0.7 + 0.01 * f), 0.15 + 0.7 * y, 0.9 - 0.7 * x * y)
    pic = np.stack([r, g, b], axis=2).copy()
    pic[8:14, 10 + f // 2:18 + f // 2] = 0.97
    pic[24:32, 40:56] = (0.03, 0.02, 0.04)
    return (pic * 255 + 0.5).astype(np.uint8)


def write(root, post, spans="", look=""):
    """A project of the 4 s clip at `root`, one output with its frames on disk: `look` (TOML), [post] `post`, then
    `spans` ([[post.span]] tables)."""
    src = root / "renders" / "draft" / "wide"
    src.mkdir(parents=True, exist_ok=True)
    (root / "mk.toml").write_text(
        f'[project]\nname = "t"\nfps = {FPS}\nframe0 = {F0}\nduration = {N / FPS}\n'
        f'[[output]]\nname = "wide"\nsize = [{W}, {H}]\n[[shot]]\nname = "a"\nfrom = 0.0\nto = {N / FPS}\n'
        f'{look}[post]\n{post}{spans}', encoding="utf-8")
    for f in range(F0, F0 + N):
        Image.fromarray(picture(f)).save(src / f"{f:05d}.png")
    return root


def output(root):
    """mk post's pipeline of the project at `root` (mkmmd.postframes.Output, on the CPU)."""
    proj = Project.load(root)
    plan = CM.cut_plan(proj)
    src = root / "renders" / "draft" / "wide"
    files = {f: src / f"{f:05d}.png" for f in range(F0, F0 + N)}
    return PF.Output(proj, plan, "wide", src, files, sorted(files), (W, H), breath=CM.cut_breath(proj, plan))


# ---------------------------------------------------------------- frames and weights
@pytest.mark.parametrize("fps, frame0, a, b", [(10, 1, 1.0, 3.0), (30, 181, 0.05, 0.15), (30, 182, 0.05, 0.15),
                                               (30, 181, 7.656, 12.0)])
def test_a_span_holds_the_frames_a_shot_with_the_same_clip_seconds_holds(fps, frame0, a, b):
    """The shots stage's rounding, round(frame0 + t x fps), not frame0 + round(t x fps): at 30 fps from frame 181,
    0.05 s is frame 182.5, a shot's cut on 182 (half to even) where the other rule would say 183."""
    (e,) = P.spans({"span": [{"from": a, "to": b}]}, fps, frame0, frame0 + 1000)
    assert (e["from"], e["to"]) == (round(frame0 + a * fps), round(frame0 + b * fps))
    if (fps, frame0, a) == (30, 181, 0.05):
        assert e["from"] == 182 != frame0 + round(a * fps)


def test_a_span_s_grade_comes_in_and_goes_out_linearly_over_its_fades():
    (e,) = P.spans({"span": [{"from": 1.0, "to": 3.0, "fade": [0.4, 0.5]}]}, FPS, F0, LAST)
    assert (e["from"], e["to"], e["fade"]) == (11, 31, (0.4, 0.5))
    frames = (9, 10, 11, 12, 13, 14, 15, 20, 25, 26, 27, 28, 29, 30, 31, 32)
    assert [P.weight(e, f, FPS, F0) for f in frames] == pytest.approx(
        [0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.0])
    assert P.weight(e, 13, FPS, F0) == 0.5                                       # half way through the fade in
    (hard,) = P.spans({"span": [{"from": 1.0, "to": 3.0}]}, FPS, F0, LAST)      # no fade: whole on its first and last
    assert [P.weight(hard, f, FPS, F0) for f in (10, 11, 30, 31)] == [0.0, 1.0, 1.0, 0.0]
    (both,) = P.spans({"span": [{"from": 1.0, "to": 3.0, "fade": 1}]}, FPS, F0, LAST)
    assert both["fade"] == (1.0, 1.0) and P.weight(both, 21, FPS, F0) == 1.0    # fades that meet: whole only there


# ---------------------------------------------------------------- the span's grade
def test_a_span_s_grade_is_post_s_with_the_keys_it_gives_laid_over_them():
    keep = {"cast": "x", "materials": ["m"], "saturation": 0.5}
    post = {"saturation": 1.08, "contrast": 1.06, "vignette": 0.06, "split": {"shadows": "iris", "amount": 0.05},
            "halation": {"strength": 0.2}, "grain": {"amount": 0.02, "seed": 3}, "streaks": True, "jobs": 2,
            "keep": keep,
            "span": [{"from": 0.0, "to": 1.0, "saturation": 1.3, "split": False, "halation": False},
                     {"from": 1.0, "to": 2.0, "vignette": 0.2, "grain": {"amount": 0.05}}]}
    a, b = P.spans(post, FPS, F0, LAST)
    assert a["spec"] == {"saturation": 1.3, "contrast": 1.06, "vignette": 0.06, "split": False, "halation": False,
                         "grain": {"amount": 0.02, "seed": 3}, "streaks": True, "jobs": 2, "keep": keep}
    assert b["spec"]["grain"] == {"amount": 0.05} and b["spec"]["split"] == post["split"]   # a table replaced whole
    base = P.Grade(post, MOON, (W, H))
    ga, gb = P.Spans(post, MOON, (W, H), FPS, F0, LAST, base).grades
    assert base.split is not None and base.hal is not None and ga.split is None and ga.hal is None   # turned off
    assert ga.keep_sat == 0.5 and float(ga.spec["saturation"]) == 1.3                      # keep stays the project's
    assert ga.grain is base.grain and ga.vig is base.vig                    # made of the same numbers: [post]'s own
    assert gb.grain is not base.grain and gb.vig is not base.vig            # seed 7 (the default) and its vignette
    assert np.array_equal(np.stack(gb.grain), np.stack(P.Grade(b["spec"], MOON, (W, H)).grain))


@pytest.mark.parametrize("spans, says", [
    ([{"from": 1.0, "to": 2.0, "streaks": True}], "post span 0 (1.0-2.0 s): unknown key streaks (a span gives from, "),
    ([{"from": 2.0, "to": 2.0}], "post span 0 (2.0-2.0 s): from is not before to"),
    ([{"from": 3.0, "to": 1.0}], "post span 0 (3.0-1.0 s): from is not before to"),
    ([{"from": 1.0}], "post span 0: to = None: clip seconds"),
    ([{"from": 3.5, "to": 4.5}], "post span 0 (3.5-4.5 s): frames 36..45 reach outside the clip (1..40)"),
    ([{"from": -0.5, "to": 1.0}], "post span 0 (-0.5-1.0 s): frames -4..10 reach outside the clip (1..40)"),
    ([{"from": 1.0, "to": 2.0, "fade": [0.6, 0.5]}],
     "post span 0 (1.0-2.0 s): fade = [0.6, 0.5] is longer than the span (1.1 s in and out over 1 s)"),
    ([{"from": 1.0, "to": 2.0, "fade": -0.1}], "post span 0 (1.0-2.0 s): fade = -0.1: seconds, 0 or more"),
    ([{"from": 1.0, "to": 2.0}, {"from": 0.0, "to": 1.5}],
     "post span 1 (0.0-1.5 s) and post span 0 (1.0-2.0 s) overlap in frames 11..15: one grade at a time"),
])
def test_a_span_mk_post_cannot_use_is_an_error_that_names_it(spans, says):
    with pytest.raises(P.SpanError) as e:
        P.spans({"span": spans}, FPS, F0, LAST)
    assert says in str(e.value)


def test_spans_that_touch_are_one_after_the_other_and_a_bad_colour_names_its_span():
    a, b = P.spans({"span": [{"from": 2.0, "to": 4.0}, {"from": 0.0, "to": 2.0}]}, FPS, F0, LAST)
    assert (a["from"], a["to"], b["from"], b["to"]) == (1, 21, 21, 41)
    post = {"span": [{"from": 0.0, "to": 1.0, "split": {"shadows": "nope"}}]}
    with pytest.raises(P.SpanError, match=r"post span 0 \(0.0-1.0 s\): colour 'nope' is neither"):
        P.Spans(post, MOON, (W, H), FPS, F0, LAST, P.Grade(post, MOON, (W, H)))


def test_mk_post_says_which_span_is_wrong_before_anything_is_drawn(tmp_path, capsys, monkeypatch):
    root = write(tmp_path / "p", BASE, '[[post.span]]\nfrom = 0.0\nto = 2.0\n[[post.span]]\nfrom = 1.5\nto = 3.0\n')
    monkeypatch.setattr(POST, "encode", lambda *a, **k: pytest.fail("encoded before the error"))
    monkeypatch.setattr(PF.Output, "finish", lambda *a, **k: pytest.fail("a frame finished before the error"))
    with pytest.raises(SystemExit) as e:
        MAIN.main(["post", "--no-audio", "--project", str(root)])
    out = capsys.readouterr().out
    assert e.value.code == 2
    assert json.loads(out[out.index("{"):])["error"] == ("post span 0 (0.0-2.0 s) and post span 1 (1.5-3.0 s) overlap "
                                                         "in frames 16..20: one grade at a time")


# ---------------------------------------------------------------- frames graded
def test_outside_a_span_the_base_grade_inside_it_the_span_s_and_half_way_through_a_fade_the_mean(tmp_path):
    spanned = output(write(tmp_path / "spanned", BASE, SPAN))
    base = output(write(tmp_path / "base", BASE))                     # the same project without the span
    whole = output(write(tmp_path / "whole", WHOLE))                  # ... whose whole [post] is the span's grade
    e = spanned.spans.entries[0]
    seen = set()
    for k, f in enumerate(spanned.frames):
        w = P.weight(e, f, FPS, F0)
        got = spanned.finish(k)[0]
        if w == 0.0:                                                  # outside it, and the first frame of its fade in
            assert spanned.graded(f) is spanned.grade and got == base.finish(k)[0], f
        elif w == 1.0:
            assert spanned.graded(f) is spanned.spans.grades[0] and got == whole.finish(k)[0], f
        else:
            assert isinstance(spanned.graded(f), P.Faded)
        seen.add(w)
    assert {0.0, 0.25, 0.5, 0.75, 1.0} <= seen and len(seen) == 9
    for f, w in ((13, 0.5), (28, 0.6)):                               # the two grades on the same picture, mixed
        k = f - F0
        img = P.read(spanned.files[f])
        mixed = (1.0 - w) * base.grade(img, k) + w * whole.grade(img, k)
        assert np.abs(spanned.graded(f)(img, k) - mixed).max() <= 1e-6
    k = 13 - F0                                                       # half way: the mean, to the level
    img = P.read(spanned.files[13])
    half = np.frombuffer(spanned.finish(k)[0], np.uint8)
    assert np.array_equal(half, P.to_u8(0.5 * (base.grade(img, k) + whole.grade(img, k))).ravel())
    mean = (np.frombuffer(base.finish(k)[0], np.uint8).astype(float) + np.frombuffer(whole.finish(k)[0], np.uint8)) / 2
    assert np.abs(half - mean).max() <= 1.0


def test_a_fade_does_not_thin_the_grain():
    """A grey picture, which saturation leaves as it is: both grades give it the same grain, and so does any mix of
    the two (two fields of their own, mixed, would lose a fifth of its spread at a quarter)."""
    post = {"saturation": 1.0, "contrast": 1.0, "vignette": 0.0, "floor": False, "grain": {"amount": 0.05},
            "span": [{"from": 1.0, "to": 3.0, "fade": [0.4, 0.0], "saturation": 2.0}]}
    base = P.Grade(post, MOON, (W, H))
    spans = P.Spans(post, MOON, (W, H), FPS, F0, LAST, base)
    grey = np.full((H, W, 3), 0.5, np.float32)
    faded = spans.at(12, base)                                        # a quarter of the way in
    assert isinstance(faded, P.Faded) and faded.w == 0.25
    assert np.abs(faded(grey, 3) - base(grey, 3)).max() <= 1e-6 and base(grey, 3).std() > 0.03


def test_a_project_without_spans_grades_every_frame_with_its_one_grade(tmp_path):
    out = output(write(tmp_path / "p", BASE))
    assert out.spans is None
    for k, f in enumerate(out.frames):
        assert out.graded(f) is out.grade
        assert out.finish(k)[0] == P.to_u8(out.grade(P.read(out.files[f]), k)).tobytes()


def test_a_still_inside_a_span_is_the_span_s_grade_through_mk_post(tmp_path, capsys):
    pics = {}
    for name, post, spans in (("spanned", BASE, '[[post.span]]\nfrom = 0.0\nto = 0.1\n' + OWN), ("whole", WHOLE, "")):
        root = tmp_path / name
        (root / "renders" / "draft" / "wide").mkdir(parents=True)
        (root / "mk.toml").write_text(
            f'[project]\nname = "t"\nfps = {FPS}\nframe0 = {F0}\nduration = 0.1\n[[output]]\nname = "wide"\n'
            f'size = [{W}, {H}]\n[post]\n{post}{spans}', encoding="utf-8")
        Image.fromarray(picture(5)).save(root / "renders" / "draft" / "wide" / f"{F0:05d}.png")
        with pytest.raises(SystemExit) as e:
            MAIN.main(["post", "--project", str(root)])
        assert e.value.code == 0, capsys.readouterr().out
        pics[name] = (root / "out" / "t_wide_draft.png").read_bytes()
    assert pics["spanned"] == pics["whole"]


# ---------------------------------------------------------------- the breathing palette, the GPU
def test_a_span_s_colours_breathe_with_the_palette(tmp_path):
    look = ('[look]\npalette = "rose-pine-moon"\n'
            'breathe = { palette = "rose-pine-dawn", keys = [[0.0, 0.0], [2.0, 1.0]], ease = "linear" }\n')
    out = output(write(tmp_path / "p", BASE, '[[post.span]]\nfrom = 0.5\nto = 3.0\n' + OWN, look))   # frames 6 to 30
    for f, m in ((11, 0.5), (21, 1.0), (30, 1.0)):                   # inside the span, whole: its grade breathes
        g = out.graded(f)
        assert out.breath.mix(f) == pytest.approx(m)
        want = np.array(PAL.srgb(PAL.darkest(MOON))) * (1 - m) + np.array(PAL.srgb(PAL.darkest(DAWN))) * m
        assert g.floor == pytest.approx(want, abs=1e-6) and g.split is None and g.hal is not None
    img = P.read(out.files[21])
    assert np.array_equal(out.graded(21)(img, 20), P.Grade(out.spans.entries[0]["spec"], DAWN, (W, H))(img, 20))


def test_mkr_post_leaves_a_project_with_spans_to_the_cpu(tmp_path, monkeypatch):
    exe = tmp_path / "mkr-post"
    exe.write_bytes(b"")
    monkeypatch.setenv("MK_MKR_POST", str(exe))                     # a binary there: what else it says
    assert PG.refused({"saturation": 1.1}) is None
    out = output(write(tmp_path / "p", BASE, SPAN))
    assert PG.unusable(out, files=False) == "[[post.span]] (a part of the clip graded its own way) is drawn on the CPU"
