"""A palette that breathes, in mk post and mk look (docs/design.md: Post; mkmmd.core.breathe): with `[look] breathe`
every colour post takes from the palette is resolved with both palettes and mixed by each frame's mix (mkmmd.post.Breath:
the plan's entries as BR.Pairs; mkmmd.post.Mixed: the grade's and the streaks' colours), on the frames its keys say: a
frame effect's colour and a label's, a transition's, the grade's floor, split toning and halation, the streaks' tint, and
each frame's job for mkr-post. A project that does not breathe goes the way it always went; an entry that is not the same
thing with both palettes is the command's usage error, naming it. Synthetic frames, no Blender."""
import copy
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mkmmd import cutfx as CF
from mkmmd import framefx as FF
from mkmmd import post as P
from mkmmd import postframes as PF
from mkmmd import postgpu as PG
from mkmmd.cli import common as CM
from mkmmd.cli import look as LOOK
from mkmmd.cli import main as MAIN
from mkmmd.cli import post as POST
from mkmmd.core import breathe as BR
from mkmmd.core import palette as PAL
from mkmmd.core import transition as TR
from mkmmd.project import Project

FPS, F0, W, H = 10, 1, 64, 36
MOON, DAWN = PAL.get("rose-pine-moon"), PAL.get("rose-pine-dawn")
KEYS = [[0, 0], [1, 1], [2, 1], [3, 0]]          # Moon to Dawn by 1 s, Dawn until 2 s, back to Moon by 3 s
MIX = {1: 0.0, 6: 0.5, 11: 1.0, 16: 1.0, 21: 1.0, 26: 0.5, 31: 0.0, 40: 0.0}   # frames (10 fps from 1) and their mix
FLAT = {"saturation": 1.0, "contrast": 1.0, "vignette": 0.0, "grain": {"amount": 0.0}}
SLAB = {"kind": "slab", "from": 0.0, "to": 4.0, "at": [0.5, 0.5], "size": [4.0, 2.0], "slant": 0.0, "angle": 0.0,
        "in": 0.0, "out": 0.0, "shutter": 0.0, "color": "love"}          # the whole frame in one palette colour


def breathing(keys=KEYS):
    """[look]: Rosé Pine Moon, breathing to Dawn on `keys` (linear between them)."""
    return {"palette": "rose-pine-moon", "breathe": {"palette": "rose-pine-dawn", "keys": keys, "ease": "linear"}}


def project(fx=(), transition=(), look=None):
    """mk.toml data of a 4 s clip: two shots, one output."""
    return {"project": {"name": "t", "fps": FPS, "frame0": F0, "duration": 4.0},
            "output": [{"name": "wide", "size": [W, H]}],
            "shot": [{"name": "a", "from": 0.0, "to": 2.0}, {"name": "b", "from": 2.0, "to": 4.0}],
            "fx": list(fx), "transition": list(transition), "look": breathing() if look is None else look}


def plans(d):
    """(the cut's plan with the project's palette, its Breath: the same plan with the other palette too)."""
    a, b = BR.palettes(d["look"])
    plan = TR.plan(d, FPS, F0, a)
    return plan, P.Breath(BR.spec(d["look"]), FPS, F0, (a, b), plan, TR.plan(d, FPS, F0, b))


def between(a, b, m):
    """`m` of the way from colour `a` to colour `b` ("#rrggbb"), display-space floats mixed as "a:b:t" mixes them."""
    a, b = np.array(PAL.srgb(a)), np.array(PAL.srgb(b))
    return a + (b - a) * m


def srgb(slot, m):
    """Palette slot `slot` `m` of the way from Moon's colour to Dawn's."""
    return between(MOON[slot], DAWN[slot], m)


def system_font():
    """A TTF of this machine (a label needs a real font), else the test is skipped."""
    if shutil.which("fc-match"):
        p = subprocess.run(["fc-match", "-f", "%{file}", "monospace"], capture_output=True, text=True).stdout.strip()
        if p.lower().endswith((".ttf", ".otf")) and Path(p).is_file():
            return p
    for p in ("/usr/share/fonts/TTF/DejaVuSansMono.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"):
        if Path(p).is_file():
            return p
    pytest.skip("no TrueType font on this machine")


# ---------------------------------------------------------------- the frame effects, the cut effects
def test_an_effect_s_palette_colour_is_each_palette_s_at_the_ends_of_the_breath_and_the_srgb_mid_between():
    plan, br = plans(project([SLAB]))
    ffx = FF.FrameFx(plan["fx"], (W, H), fps=FPS, frame0=F0, frame_of=lambda g: None, point_of=None, breath=br)
    grey = np.full((H, W, 3), 0.5, np.float32)
    for f, m in MIX.items():
        assert br.mix(f) == pytest.approx(m)
        assert ffx.frame(f, grey)[H // 2, W // 2] == pytest.approx(srgb("love", m), abs=2e-6)
    hexmid = np.array(PAL.srgb(PAL.mix(MOON["love"], DAWN["love"], 0.5)))          # "a:b:0.5" as a palette mixes it
    assert np.abs(ffx.frame(6, grey)[0, 0] - hexmid).max() <= 0.5 / 255 + 1e-6
    assert ffx.entries(1)[0] is plan["fx"][0]                                       # the ends: the entries themselves
    assert ffx.entries(11)[0] is br.pairs["fx", 0].b
    assert br.mix(F0 - 5) == 0.0 and br.mix(F0 + 80) == 0.0                         # held before the clip and after it


def test_a_label_s_box_and_type_are_laid_in_the_frame_s_mix_of_their_colours():
    font = system_font()
    label = {"kind": "label", "from": 0.0, "to": 4.0, "labels": [{"text": "SIDE A", "at": [0.3, 0.6], "t": 0.0}],
             "box": {"pad": 30}, "leader": False}
    plan, br = plans(project([label]))
    size = (320, 180)
    ffx = FF.FrameFx(plan["fx"], size, fps=FPS, frame0=F0, frame_of=lambda g: None, point_of=None,
                     font_of=lambda ref: font, breath=br)
    black = np.zeros((size[1], size[0], 3), np.float32)
    for f in (6, 11, 16, 26, 31):
        m = MIX[f]
        (_, box, x0, y0, alpha), (_, text, *_) = ffx._label_inks(ffx.entries(f)[0], f)
        assert box == pytest.approx(list(srgb("base", m)), abs=2e-6) and alpha == pytest.approx(0.85)
        assert text == pytest.approx(list(srgb("text", m)), abs=2e-6)
        assert ffx.frame(f, black)[y0 + 2, x0 + 2] == pytest.approx(0.85 * srgb("base", m), abs=1e-5)   # its pad


@pytest.mark.parametrize("m", [0.0, 0.5, 1.0])
def test_a_transition_s_colour_is_the_frame_s_mix_of_its_two(m, tmp_path):
    spin = {"at": 2.0, "kind": "spin", "scale": 0.3, "blur": 0.0, "back": "love"}
    plan, br = plans(project(transition=[spin], look=breathing([[0, m]])))
    cut = plan["transitions"][0]["cut"]
    out = CF.CutFx(plan, tmp_path, (W, H), breath=br).frame(cut, np.full((H, W, 3), 0.5, np.float32))
    assert out[0, 0] == pytest.approx(srgb("love", m), abs=2e-6)                 # round the turned picture: its back
    assert out[H // 2, W // 2] == pytest.approx([0.5] * 3, abs=1e-6)             # the picture itself


# ---------------------------------------------------------------- the grade, the streaks
def test_the_grade_s_floor_split_toning_and_halation_and_the_streaks_tint_are_mixed_by_the_mix():
    spec = {**FLAT, "split": {"shadows": "iris", "highlights": "gold"}, "halation": {"tint": "rose"}}
    g = P.Grade(spec, MOON, (W, H))
    grades, floors = P.Mixed(g, DAWN), P.Mixed(P.Grade(FLAT, MOON, (W, H)), DAWN)
    black = np.zeros((H, W, 3), np.float32)
    for m in (0.0, 0.25, 0.5, 1.0):
        at = grades.at(m)
        assert at.floor == pytest.approx(between(PAL.darkest(MOON), PAL.darkest(DAWN), m))   # Moon's base, Dawn's text
        assert at.split[0] == pytest.approx(srgb("iris", m)) and at.split[1] == pytest.approx(srgb("gold", m))
        assert at.hal[3] == pytest.approx(srgb("rose", m)) and at.hal[:3] == g.hal[:3] and at.split[2] == g.split[2]
        assert at.grain is g.grain and at.vig is g.vig                            # shared, not made again
        fl = floors.at(m)                                                         # black lands on this frame's floor
        assert fl(black)[0, 0] == pytest.approx(np.clip(fl.floor, float(fl.floor @ P.Y_W) * 0.98, 1.0), abs=1e-6)
    assert grades.at(0.0) is g
    assert np.array_equal(grades.at(1.0)(black), P.Grade(spec, DAWN, (W, H))(black))
    streaks = P.Mixed(P.Streaks({}, MOON, (W, H)), DAWN)
    for m in (0.0, 0.5, 1.0):
        assert streaks.at(m).tint == pytest.approx(srgb("foam", m))


# ---------------------------------------------------------------- mk post, mk look
SLAB_TOML = """
[[fx]]
kind = "slab"
from = 0.0
to = 4.0
at = [0.25, 0.5]
size = [0.8888889, 2.0]
slant = 0.0
angle = 0.0
in = 0.0
out = 0.0
shutter = 0.0
color = "love"
"""
MOON_TOML = '[look]\npalette = "rose-pine-moon"\n'
DAWN_TOML = '[look]\npalette = "rose-pine-dawn"\n'
BREATHE_TOML = ('[look]\npalette = "rose-pine-moon"\n[look.breathe]\npalette = "rose-pine-dawn"\n'
                'keys = [[0, 0], [1, 1], [2, 1], [3, 0]]\nease = "linear"\n')


def write(root, look, extra=""):
    """A project at `root` (the slab over the frame's left half, a flat grade and `look`) and its draft frames, black."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "mk.toml").write_text(
        f'[project]\nname = "t"\nfps = {FPS}\nframe0 = {F0}\nduration = 4.0\nblend = "scene.blend"\n'
        f'[[output]]\nname = "wide"\nsize = [{W}, {H}]\n'
        '[[shot]]\nname = "a"\nfrom = 0.0\nto = 2.0\n[[shot]]\nname = "b"\nfrom = 2.0\nto = 4.0\n' + SLAB_TOML +
        '[post]\nsaturation = 1.0\ncontrast = 1.0\nvignette = 0.0\ngrain = { amount = 0.0 }\n' + look + extra,
        encoding="utf-8")
    d = root / "renders" / "draft" / "wide"
    d.mkdir(parents=True, exist_ok=True)
    for f in range(F0, F0 + 40):
        Image.fromarray(np.zeros((H, W, 3), np.uint8)).save(d / f"{f:05d}.png")
    return root


class Encoder:
    """Stands in for ffmpeg: keeps the frames mk post writes."""

    def __init__(self):
        self.frames, self.stdin = [], self

    def __call__(self, *a, **k):
        return self

    def write(self, data):
        self.frames.append(np.frombuffer(bytes(data), np.uint8).reshape(H, W, 3))

    def close(self):
        pass

    def wait(self, timeout=None):
        return 0

    def poll(self):
        return None

    def kill(self):
        pass


def run_post(root, capsys, monkeypatch, *extra):
    """(exit code, report, {Blender frame: rgb24 picture}) of `mk post` on the project at `root`."""
    enc = Encoder()
    monkeypatch.setattr(POST, "encode", enc)
    with pytest.raises(SystemExit) as e:
        MAIN.main(["post", "--preset", "draft", "--no-audio", "--project", str(root), *extra])
    out = capsys.readouterr().out
    return e.value.code, json.loads(out[out.index("{"):]), dict(zip(range(F0, F0 + 40), enc.frames))


def test_mk_post_makes_each_frame_as_its_mix_s_palette_would_on_every_frame_the_keys_say(tmp_path, capsys, monkeypatch):
    moon = run_post(write(tmp_path / "moon", MOON_TOML), capsys, monkeypatch, "--jobs", "1")
    dawn = run_post(write(tmp_path / "dawn", DAWN_TOML), capsys, monkeypatch, "--jobs", "1")
    one = run_post(write(tmp_path / "breathe", BREATHE_TOML), capsys, monkeypatch, "--jobs", "1")
    assert moon[0] == dawn[0] == one[0] == 0, (moon[1], dawn[1], one[1])
    a, b, got = moon[2], dawn[2], one[2]
    assert not np.array_equal(a[11][:, 8], b[11][:, 8]) and not np.array_equal(a[11][:, 56], b[11][:, 56])
    for f, m in MIX.items():
        if m == 0.0:
            assert np.array_equal(got[f], a[f]), f                               # the project's palette
        elif m == 1.0:
            assert np.array_equal(got[f], b[f]), f                               # the other one's
        else:                                                                    # the slab and the floor between them
            assert np.abs(got[f].astype(int) - (a[f].astype(int) + b[f].astype(int)) / 2.0).max() <= 1.0, f
            assert got[f][H // 2, 8] == pytest.approx(np.round(srgb("love", m) * 255), abs=1)
    monkeypatch.setattr(PF, "CHUNK", 3)                                          # worker processes: the breath with them
    two = run_post(tmp_path / "breathe", capsys, monkeypatch, "--jobs", "2")
    assert two[0] == 0 and two[1]["outputs"]["wide"]["jobs"] == 2
    assert all(np.array_equal(two[2][f], got[f]) for f in got)


def test_mkr_post_s_job_carries_each_frame_s_grade_and_effect_colours(tmp_path):
    root = write(tmp_path / "p", BREATHE_TOML)
    proj = Project.load(root)
    plan = CM.cut_plan(proj)
    br = CM.cut_breath(proj, plan)
    src = root / "renders" / "draft" / "wide"
    files = {f: src / f"{f:05d}.png" for f in range(F0, F0 + 40)}
    (tmp_path / "gpu").mkdir()
    out = PF.Output(proj, plan, "wide", src, files, sorted(files), (W, H), gpu=str(tmp_path / "gpu"), breath=br)
    for f, m in MIX.items():
        route, job, _ = out.plan(f - F0)
        assert route == "gpu"
        assert job["grade"]["floor"]["rgb"] == pytest.approx(list(between(PAL.darkest(MOON), PAL.darkest(DAWN), m)),
                                                             abs=1e-6)
        assert job["ops"][0]["op"] == "slab" and job["ops"][0]["colour"] == pytest.approx(list(srgb("love", m)),
                                                                                         abs=1e-6)


def test_mkr_post_grades_each_frame_in_its_own_colours_as_the_cpu_does(tmp_path):
    import gpu_fixtures as GF                     # skips without the binary or a GPU (run it as `mk gpu run -- ...`)
    why = GF.usable()
    if why:
        pytest.skip(why)
    root = write(tmp_path / "p", BREATHE_TOML)
    proj = Project.load(root)
    plan = CM.cut_plan(proj)
    br = CM.cut_breath(proj, plan)
    src = root / "renders" / "draft" / "wide"
    files = {f: src / f"{f:05d}.png" for f in range(F0, F0 + 40)}
    folder = tmp_path / "gpu"
    folder.mkdir()
    out = PF.Output(proj, plan, "wide", src, files, sorted(files), (W, H), gpu=str(folder), breath=br)
    cpu = PF.Output(proj, plan, "wide", src, files, sorted(files), (W, H), breath=br)
    server = PG.Server(out, folder)
    try:
        for f in MIX:
            route, job, _ = out.plan(f - F0)
            server.send(job)
            k, _, got = server.receive()
            want = np.frombuffer(cpu.finish(f - F0)[0], np.uint8).astype(int)
            assert k == f - F0 and np.abs(np.frombuffer(got, np.uint8).astype(int) - want).max() <= 1, f
    finally:
        server.close()


MANDALA = ('[[fx]]\nkind = "mandala"\nfrom = 0.0\nto = 30.0\nback = "pine"\n'
           '[[fx.rings]]\ncount = 6\nradius = 0.3\nscale = 0.3\nsource = { shape = ["clover", "heart"], '
           'colors = ["love", "gold"] }\n')
LONG = ('[look]\npalette = "rose-pine-moon"\n[look.breathe]\npalette = "rose-pine-dawn"\n'
        'keys = [[0, 0], [10, 1], [20, 1], [30, 0]]\nease = "linear"\n')           # 300 frames, a new mix on 200 of them


def long_breath(root, n=300):
    """A 30 s clip (n frames at 10 fps) of black frames under a mandala of palette shapes, breathing Moon -> Dawn -> Moon:
    (its Output for mkr-post, a CPU Output, the scratch folder)."""
    root.mkdir(parents=True)
    (root / "mk.toml").write_text(
        f'[project]\nname = "t"\nfps = {FPS}\nframe0 = {F0}\nduration = 30.0\n[[output]]\nname = "wide"\n'
        f'size = [{W}, {H}]\n[[shot]]\nname = "a"\nfrom = 0.0\nto = 30.0\n' + MANDALA + LONG, encoding="utf-8")
    src = root / "renders" / "draft" / "wide"
    src.mkdir(parents=True)
    files = {}
    for f in range(F0, F0 + n):
        files[f] = src / f"{f:05d}.png"
        Image.fromarray(np.zeros((H, W, 3), np.uint8)).save(files[f])
    proj = Project.load(root)
    plan = CM.cut_plan(proj)
    br = CM.cut_breath(proj, plan)
    folder = root / "gpu"
    folder.mkdir()
    return (PF.Output(proj, plan, "wide", src, files, sorted(files), (W, H), gpu=str(folder), breath=br),
            PF.Output(proj, plan, "wide", src, files, sorted(files), (W, H), breath=br), folder)


def tiles_of(job):
    return [t for op in job["ops"] for t in op.get("tiles") or [] if t["kind"] == "file"]


def test_mid_breath_shape_tiles_are_written_for_their_frame_alone_and_forgotten_when_not_sent(tmp_path):
    out, _, folder = long_breath(tmp_path / "p", n=120)
    kept, once = set(), []
    for k in range(120):
        mid = 0.0 < out.breath.mix(out.frames[k]) < 1.0
        for t in tiles_of(out.plan(k)[1]):
            assert t["once"] == mid                                               # the ends' shapes are kept
            if mid:
                once.append(t["rgba"])
            else:
                kept.add(t["rgba"])
    assert len(once) == len(set(once)) > 100 and len(kept) <= 8                  # a name each; the ends' kept, few
    for p in once:
        Path(p).unlink()                                                          # as mkr-post does once it read them
    job = out.plan(50)[1]                                                         # mid-breath, then not sent
    assert all(Path(t["rgba"]).exists() for t in tiles_of(job))
    PG.dropped(out.ffx, out.frames[50])
    assert not any(Path(t["rgba"]).exists() for t in tiles_of(job))


def test_a_long_breath_keeps_mkr_post_s_tiles_bounded_and_draws_what_the_unbounded_way_drew(tmp_path, monkeypatch):
    import gpu_fixtures as GF                     # skips without the binary or a GPU (run it as `mk gpu run -- ...`)
    why = GF.usable()
    if why:
        pytest.skip(why)

    def run(root):
        out, cpu, folder = long_breath(root)
        server = PG.Server(out, folder)
        frames, most, per = [], 0, 0
        try:
            for k in range(len(out.frames)):
                job = out.plan(k)[1]
                per = max(per, sum(1 + (t["ring"] is not None) for t in tiles_of(job)))
                server.send(job)
                frames.append(np.frombuffer(server.receive()[2], np.uint8).astype(int))
                most = max(most, len(list(folder.glob("tile-*"))))
                if k % 25 == 0:                                                  # the CPU's frame, within a level
                    assert np.abs(frames[-1] - np.frombuffer(cpu.finish(k)[0], np.uint8)).max() <= 1, k
        finally:
            server.close()
        return frames, most, per

    frames, most, per = run(tmp_path / "bounded")
    assert most <= 3 * per                                    # the two ends' tiles and one frame's: whatever the length
    monkeypatch.setattr(FF.FrameFx, "passing", lambda self, fx, frame: False)   # every tile kept: the unbounded way
    unbounded, grown, _ = run(tmp_path / "unbounded")
    assert grown > 20 * per                                   # (a tile for every mix it passed)
    assert all(np.abs(a - b).max() <= 1 for a, b in zip(frames, unbounded))


def test_mk_look_previews_the_effects_in_each_frame_s_mix(tmp_path, capsys, monkeypatch):
    root = write(tmp_path / "p", BREATHE_TOML)
    shots = tmp_path / "look"

    def render(op, job, **_):                                                    # stands in for Blender: black frames
        out = []
        for f in job["frames"]:
            for s in job["sizes"]:
                path = shots / f"cut_{s['name']}_{f}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.zeros((s["h"], s["w"], 3), np.uint8)).save(path)
                out.append({"path": str(path), "frame": f, "size": s["name"], "view": "cut"})
        return out
    monkeypatch.setattr(LOOK.bridge, "run", render)
    with pytest.raises(SystemExit) as e:
        MAIN.main(["look", "--engine", "eevee", "--frames", "1,6,11,26", "--out", str(shots), "--project", str(root)])
    out = capsys.readouterr().out
    assert e.value.code == 0, out
    for f in (1, 6, 11, 26):
        im = np.asarray(Image.open(shots / f"cut_wide_{f}.png").convert("RGB"))
        assert im[im.shape[0] // 2, im.shape[1] // 8] == pytest.approx(np.round(srgb("love", MIX[f]) * 255), abs=1)


# ---------------------------------------------------------------- what cannot breathe, what does not
ECHO = {"kind": "echo", "from": 0.0, "to": 4.0, "tint": "foam"}


def test_an_entry_that_is_not_the_same_thing_with_both_palettes_is_named():
    d = project([SLAB, ECHO])
    a, b = BR.palettes(d["look"])
    plan = TR.plan(d, FPS, F0, a)
    other = copy.deepcopy(TR.plan(d, FPS, F0, b))
    other["fx"][1]["blend"] = "darken"
    with pytest.raises(BR.BreatheError, match=r"fx 1 \(echo\)\['blend'\]: 'lighten' with one palette, 'darken'"):
        P.Breath(BR.spec(d["look"]), FPS, F0, (a, b), plan, other)


@pytest.mark.parametrize("command", ["post", "look"])
def test_the_command_says_so_as_its_usage_error_before_anything_is_drawn(command, tmp_path, capsys, monkeypatch):
    root = write(tmp_path / "p", BREATHE_TOML, "\n[[fx]]\nkind = \"echo\"\nfrom = 0.0\nto = 4.0\n")
    made = TR.plan

    def plan(data, fps, frame0, palette):                                        # a structure the palette decides
        out = made(data, fps, frame0, palette)
        if palette["base"] == DAWN["base"]:
            out["fx"][1]["blend"] = "darken"
        return out
    monkeypatch.setattr(TR, "plan", plan)
    monkeypatch.setattr(LOOK.bridge, "run", lambda *a, **k: pytest.fail("rendered before the error"))
    monkeypatch.setattr(POST, "encode", lambda *a, **k: pytest.fail("encoded before the error"))
    with pytest.raises(SystemExit) as e:
        MAIN.main([command, *(["--frames", "1"] if command == "look" else ["--no-audio"]), "--project", str(root)])
    out = capsys.readouterr().out
    assert e.value.code == 2
    assert "fx 1 (echo)['blend']" in json.loads(out[out.index("{"):])["error"]


@pytest.mark.parametrize("look", [MOON_TOML, MOON_TOML + "breathe = false\n"])
def test_a_project_that_does_not_breathe_goes_the_way_it_went(look, tmp_path):
    root = write(tmp_path / "p", look)
    proj = Project.load(root)
    plan = CM.cut_plan(proj)
    assert CM.cut_breath(proj, plan) is None
    src = root / "renders" / "draft" / "wide"
    files = {f: src / f"{f:05d}.png" for f in range(F0, F0 + 40)}
    (tmp_path / "gpu").mkdir()
    out = PF.Output(proj, plan, "wide", src, files, sorted(files), (W, H), gpu=str(tmp_path / "gpu"))
    assert out.graded(11) is out.grade and out.ffx.entries(11)[0] is plan["fx"][0]
    assert "grade" not in out.plan(10)[1]                                        # mkr-post grades with its setup's
