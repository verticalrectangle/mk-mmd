"""The loop's wrap through `mk post` and `mk look` (docs/design.md: Transitions and inserts, `page`): the clip's last
frames, as finished (their [[fx]] and screen type on), turn over onto frame 0 as finished, held still, and the grade runs on
the result. Past the roll the wrap's frames are exactly what frame 0 is when the loop starts again, whatever [[fx]] or type
either end carries, in both screen modes and with `[post] keep`; on the flat page they are the clip's end as it would be
without the wrap. Tiny synthetic frames: mk post's encoder keeps the frames it is handed, mk look's Blender op draws flat
pictures; no Blender, no ffmpeg."""
import json
import math
import tomllib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mkmmd import bridge
from mkmmd.cli import main as MAIN
from mkmmd.cli import post as POST
from mkmmd.core import keep as KP
from mkmmd.core import pageturn as PG
from mkmmd.core import palette as PAL
from mkmmd.core import screentype as SR
from mkmmd.core import transition as TR

W, H, FPS, F0 = 64, 36, 10, 1
END = F0 + 19                                    # a 2 s clip: frames 1 .. 20, the wrap on 15 .. 20
REST = (200, 60, 50)                             # every frame but frame 0
GRADE = 'saturation = 0.5\ncontrast = 1.2\npivot = 0.5\nvignette = 0.0\nfloor = false\ngrain = { amount = 0.0 }\n'
WRAP = '[[transition]]\nat = 2.0\nkind = "page"\nease = "out"\ndir = "left"\nangle = 0.0\nshadow = 0.0\n'
OWN = '[[fx]]\nkind = "tint"\nfrom = 0.0\nto = 0.1\ncolor = "#2040ff"\n'          # frame 0's own
LATE = '[[fx]]\nkind = "tint"\nfrom = 1.4\nto = 2.0\ncolor = "#ff8000"\n'         # over the clip's end
EVE = '[[cast]]\nname = "eve"\npmx = "eve.pmx"\n'


def project(root, post="", wrap=True, fx=OWN + LATE, cast=""):
    """Two shots, a 0-1 s and b 1-2 s, one output; the wrap (`wrap`), the [[fx]] and the [post] table (`post` after the
    grade). Returns the mk.toml as data."""
    text = (f'[project]\nname = "t"\nfps = {FPS}\nframe0 = {F0}\nduration = 2.0\n[[output]]\nname = "wide"\n'
            f'size = [{W}, {H}]\n[[shot]]\nname = "a"\nfrom = 0.0\nto = 1.0\n[[shot]]\nname = "b"\nfrom = 1.0\nto = 2.0\n'
            + (WRAP if wrap else "") + fx + cast + "[post]\n" + GRADE + post)
    (root / "mk.toml").write_text(text, encoding="utf-8")
    return tomllib.loads(text)


def first():
    """Frame 0 as rendered: red counts the columns, blue the rows."""
    a = np.empty((H, W, 3), np.uint8)
    a[..., 0] = np.linspace(30, 230, W).round().astype(np.uint8)[None, :]
    a[..., 1] = 90
    a[..., 2] = np.linspace(40, 220, H).round().astype(np.uint8)[:, None]
    return a


def picture(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(first() if frame == F0 else np.full((H, W, 3), REST, np.uint8)).save(path)


def typed(frame):
    """Where a frame's screen type lies and its colour: frame 0's at the top left, the clip's end's at the bottom right."""
    if frame == F0:
        return (4, 4, 24, 14), (40, 220, 90, 255)
    if frame >= END - 5:
        return (36, 22, 60, 32), (250, 250, 250, 200)
    return None


def layer(path, frame):
    """The frame's screen layer (straight RGBA) when it has type; its path, or None."""
    t = typed(frame)
    if t is None:
        return None
    (x0, y0, x1, y1), rgba = t
    a = np.zeros((H, W, 4), np.uint8)
    a[y0:y1, x0:x1] = rgba
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(a, "RGBA").save(path)
    return str(path)


def rendered(root, keep=None):
    """The cut's frames and screen layers as mk render leaves them, and frame 0 as a bare plate of its shot, which the
    page must not turn onto; with `keep` {frame: rows} the keep masks, 1 on those rows."""
    d = root / "renders" / "draft" / "wide"
    for f in range(F0, END + 1):
        picture(d / f"{f:05d}.png", f)
        layer(d / SR.layer_rel(f), f)
        if keep is not None:
            m = np.zeros((H, W), np.uint8)
            if f in keep:
                m[keep[f][0]:keep[f][1]] = 255
            (d / KP.LAYER_DIR).mkdir(exist_ok=True)
            Image.fromarray(m, "L").save(d / KP.rel(f))
    picture(d / TR.plate_rel("a", F0), F0)


class Encoder:
    """Stands in for ffmpeg: keeps the frames mk post writes."""

    def __init__(self):
        self.frames, self.stdin = [], self

    def write(self, data):
        self.frames.append(np.frombuffer(data, np.uint8).reshape(H, W, 3))

    def close(self):
        pass

    def wait(self):
        return 0


def run_post(root, capsys, monkeypatch):
    enc = Encoder()
    monkeypatch.setattr(POST, "encode", lambda *a, **k: enc)
    with pytest.raises(SystemExit) as e:
        MAIN.main(["post", "--preset", "draft", "--no-audio", "--project", str(root)])
    out = capsys.readouterr().out
    rep = json.loads(out[out.index("{"):])
    assert e.value.code == 0, rep
    return np.stack(enc.frames)


def regions(t, f):
    """(past the roll: frame 0 shows, the flat page) as column masks on `f` of the wrap's window (dir left, angle 0)."""
    k = f - t["first"]
    c, R, (c0, _) = PG.line(t, k, (W, H)), PG.roll(t, (W, H)), PG.travel(t, (W, H))
    xs = np.arange(W)
    return xs > c + R + 2.0, xs < c - max(c0 - c - math.pi * R, 0.0) - 2.0


def wrap_of(data):
    return TR.wrap_of(TR.plan(data, FPS, F0, PAL.get("rose-pine-moon")))


@pytest.mark.parametrize("screen", ["graded", "over"])
def test_the_clips_end_turns_over_onto_frame_0_as_it_finally_looks(tmp_path, capsys, monkeypatch, screen):
    """An [[fx]] over the clip's end and the end's type stay on the page; past the roll the wrap's frames are frame 0
    exactly as the loop starts again on it, its own [[fx]] and type on, from the start of the turn to its last frame."""
    data = project(tmp_path, f'screen = "{screen}"\n')
    rendered(tmp_path)
    v = run_post(tmp_path, capsys, monkeypatch)
    project(tmp_path, f'screen = "{screen}"\n', wrap=False)
    plain = run_post(tmp_path, capsys, monkeypatch)                          # the clip's end as finished, no wrap
    t = wrap_of(data)
    assert (t["first"], t["last"]) == (END - 5, END) and len(v) == len(plain) == END + 1 - F0
    assert np.array_equal(v[:t["first"] - F0], plain[:t["first"] - F0])      # up to the wrap nothing changes
    for f in (t["first"], t["last"]):
        past, page = regions(t, f)
        i = f - F0
        assert past.sum() >= 8
        assert not np.array_equal(plain[i][:, past], v[0][:, past])          # the end does not look like frame 0
        assert np.array_equal(v[i][:, past], v[0][:, past]), f"frame {f}: not frame 0 as it finally looks"
        assert np.array_equal(v[i][:, page], plain[i][:, page]), f"frame {f}: the page is not the end as finished"
    assert regions(t, t["first"])[1].sum() >= 20                              # the page lies flat at the start


def test_the_keep_masks_turn_with_the_page(tmp_path, capsys, monkeypatch):
    """With `[post] keep` frame 0 keeps its colour where its mask says, past the roll as on frame 0 itself, while the
    clip's end, inside its [[fx]], is graded whole on the page."""
    data = project(tmp_path, 'keep = { cast = "eve", materials = ["eyes"] }\n', fx=LATE, cast=EVE)
    rendered(tmp_path, keep={F0: (16, 26)})
    v = run_post(tmp_path, capsys, monkeypatch)
    t = wrap_of(data)
    band = v[0][16:26].astype(int)
    assert np.ptp(band, axis=-1).mean() > np.ptp(v[0][28:].astype(int), axis=-1).mean() + 20   # kept: more colour
    for f in (t["first"], t["last"]):
        past, _ = regions(t, f)
        assert np.array_equal(v[f - F0][:, past], v[0][:, past]), f"frame {f}: frame 0's kept colour is lost"


def test_mk_look_draws_frame_0_beside_the_wrap_and_turns_onto_it_as_finished(tmp_path, capsys, monkeypatch):
    """`mk look` on frames of the wrap asks the cut for frame 0 too, finishes it as mk post does (its [[fx]] and type
    on), turns each frame as finished over it, and leaves only the frames asked for."""
    data = project(tmp_path)
    t = wrap_of(data)
    calls = []

    def fake_look(op, args, **kw):
        """The Blender op: the frames each view asks for and their screen layers, and the plates the layers ask for."""
        calls.append(args)
        out, lay = Path(args["out"]), args.get("layers") or {"demands": {}}
        for size in args["sizes"]:
            for f, items in lay["demands"].items():
                for it in items:
                    if it["kind"] == "plate":
                        picture(Path(lay["dir"]) / size["name"] / TR.plate_rel(it["shot"], int(f)), int(f))
        items = []
        for view in args["views"]:
            for f in view.get("frames") or args["frames"]:
                for size in args["sizes"]:
                    path = out / f"{view['name']}_{size['name']}_{int(f):05d}.png"
                    picture(path, f)
                    items.append({"view": view["name"], "size": size["name"], "frame": int(f), "path": str(path),
                                  "camera": None, "screen": layer(out / size["name"] / SR.layer_rel(f), f)})
        return items
    monkeypatch.setattr(bridge, "run", fake_look)

    def look(frames, out):
        with pytest.raises(SystemExit) as e:
            MAIN.main(["look", str(tmp_path / "scene.blend"), "--engine", "eevee", "--frames", frames, "--size", str(W),
                       "--out", str(tmp_path / out), "--project", str(tmp_path)])
        rep = json.loads(capsys.readouterr().out)
        assert e.value.code == 0, rep
        return rep, {int(Path(p).stem.rsplit("_", 1)[1]): np.asarray(Image.open(p).convert("RGB")) for p in rep["images"]}

    rep, wrapped = look(f"{t['first']},{t['last']}", "wrap")
    asked = calls[-1]["views"][0]
    _, zero = look(str(F0), "zero")
    assert "frames" not in calls[-1]["views"][0]                             # asked for, frame 0 is drawn once
    project(tmp_path, wrap=False)
    _, plain = look(f"{t['first']},{t['last']}", "plain")
    for f in (t["first"], t["last"]):
        past, page = regions(t, f)
        assert np.array_equal(wrapped[f][:, past], zero[F0][:, past]), f"frame {f}: not frame 0 as finished"
        assert np.array_equal(wrapped[f][:, page], plain[f][:, page]), f"frame {f}: the page is not the end as finished"
    assert asked.get("frames") == [t["first"], t["last"], F0]                # the cut drew frame 0 for under the page
    assert sorted(wrapped) == [t["first"], t["last"]] and not (tmp_path / "wrap" / f"cut_wide_{F0:05d}.png").exists()
    assert [e["frame"] for e in rep["effects"]] == [t["first"], t["last"]]
