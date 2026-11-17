"""mkmmd.render.reuse: the keys that say which frames a rebuilt scene left as they were."""
import json

import numpy as np

from mkmmd.render import reuse as RU

N, FRAME0 = 12, 100
BONES, CAM_A, CAM_B, HAT, HAT_SHOWN, REST = 0, 2 * N, 5 * N, 6 * N, 7 * N, 8 * N      # float offsets in bake.bin


def bake(path, floats, bind="mmd_bind123", check=1.0, marks=True):
    """A bake folder: bones (2 floats a frame), camera A's rows (3), camera B's (1), a hat's matrix (1) and whether it
    shows (1: on the first four frames), then 4 floats that are every frame's."""
    path.mkdir(exist_ok=True)
    np.asarray(floats, "<f4").tofile(path / "bake.bin")
    (path / "lit.bin").write_bytes(b"mesh")
    doc = {"frames": N, "frame0": FRAME0,
           "lit": {"check": {"worst_mm": check}, "images": {}, "materials": {"skin": {"links": [bind, "out", bind]}},
                   "meshes": [{"name": "hat", "shown": HAT_SHOWN, "armature": None}]}}
    if marks:
        doc["per_frame"] = [[BONES, 2], [CAM_A, 3, "camera:A"], [CAM_B, 1, "camera:B"], [HAT, 1, "mesh:hat"],
                            [HAT_SHOWN, 1]]
    (path / "bake.json").write_text(json.dumps(doc))
    return path


def keys(path):
    sig = RU.signature(path)
    through = lambda f: ["A"] if f < FRAME0 + 6 else ["B"]          # the cut: A for the first half, then B
    return RU.keys(sig, FRAME0, range(FRAME0, FRAME0 + N), {"samples": 16}, lambda f: None, through)


def changed(a, b):
    return [f for f in a if a[f] != b[f]]


def test_a_frame_keeps_its_key_until_something_it_shows_changes(tmp_path):
    floats = np.arange(REST + 4, dtype=np.float32)
    floats[HAT_SHOWN:HAT_SHOWN + N] = [1, 1, 1, 1] + [0] * (N - 4)
    was = keys(bake(tmp_path / "a", floats))
    assert len(set(was.values())) == N

    def after(name, at, **how):
        new = floats.copy()
        new[at] += 1.0
        return changed(was, keys(bake(tmp_path / name, new, **how)))
    # built again: mmd_tools' node names and the self-check's numbers are new, the scene is not
    assert keys(bake(tmp_path / "b", floats, bind="mmd_bind-9223363", check=2.5)) == was
    assert after("c", BONES + 2 * 5) == [103, 104, 105, 106, 107]   # a bone moves on frame 5: the shutter's reach
    assert after("d", slice(CAM_B, CAM_B + N)) == list(range(106, 112))    # B's clip range, on every frame: its shot's
    assert after("e", HAT + 10) == []                               # the hat moves while nothing shows it
    assert after("f", HAT + 2) == [100, 101, 102, 103, 104]         # ... and while frames in the shutter's reach do
    assert len(after("g", REST + 3)) == N                           # what no mark covers is part of every frame
    assert RU.signature(bake(tmp_path / "h", floats, marks=False)) is None      # an older bake: nothing to go by


def test_a_spans_look_is_its_own_frames(tmp_path):
    """A [[look.span]] (bake.json lit.view `spans`) is part of its frames alone: a span added, or its LUT changed, draws
    its frames again and no others; the same look under another file name changes nothing; the view's own look is
    every frame's."""
    floats = np.arange(REST + 4, dtype=np.float32)

    def looked(name, spans=(), luts=None, own=b"own"):
        path = bake(tmp_path / name, floats)
        doc = json.loads((path / "bake.json").read_text())
        doc["lit"]["view"] = {"look": "None", "lut": {"file": "view_lut.bin", "n": 2, "lo": -1, "hi": 1}}
        if spans:
            doc["lit"]["view"]["spans"] = list(spans)
        (path / "bake.json").write_text(json.dumps(doc))
        (path / "view_lut.bin").write_bytes(own)
        for file, data in (luts or {}).items():
            (path / file).write_bytes(data)
        return keys(path)

    def span(first, last, file, exposure=-1.0):
        return {"frames": [first, last], "lut": {"file": file, "n": 2, "lo": -1, "hi": 1}, "zero": [0, 0, 0],
                "floor": [0, 0, 0], "look": "AgX - Punchy", "exposure": exposure, "gamma": 1.0}
    dark = {"view_lut_0.bin": b"dark"}
    plain = looked("plain")
    one = looked("one", [span(102, 104, "view_lut_0.bin")], dark)
    assert changed(plain, one) == [102, 103, 104]                   # a span added: its frames, its first and last too
    assert changed(one, looked("lut", [span(102, 104, "view_lut_0.bin")], {"view_lut_0.bin": b"darker"})) == \
        [102, 103, 104]                                             # its LUT changed
    assert changed(one, looked("rec", [span(102, 104, "view_lut_0.bin", -1.5)], dark)) == [102, 103, 104]
    two = looked("two", [span(102, 104, "view_lut_1.bin"), span(108, 109, "view_lut_0.bin", -2.0)],
                 {"view_lut_1.bin": b"dark", "view_lut_0.bin": b"night"})
    assert changed(one, two) == [108, 109]                          # another span: its frames; the first's look renamed
    assert len(changed(one, looked("own", [span(102, 104, "view_lut_0.bin")], dark, own=b"own2"))) == N   # every frame's
