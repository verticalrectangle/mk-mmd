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
