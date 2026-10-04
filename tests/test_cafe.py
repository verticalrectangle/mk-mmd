"""Pure maths of the cafe set and props: window panes, deterministic rain splats, the condensation noise image and the
palette-slot colours. (The Blender-side builders are checked by building and looking.)"""
import json

import numpy as np
import pytest

from mkmmd.blender.library.props.cafe_colors import DAWN, Colors
from mkmmd.blender.library.sets import cafe_maths as M


def test_panes_tile_the_opening_between_frame_and_bars():
    panes = M.pane_rects()
    assert len(panes) == 16 and {(p["row"], p["col"]) for p in panes} == {(r, c) for r in range(4) for c in range(4)}
    for p in panes:
        assert M.OPEN_Y[0] + M.FRAME_W - 1e-9 <= p["y0"] < p["y1"] <= M.OPEN_Y[1] - M.FRAME_W + 1e-9
        assert M.OPEN_Z[0] + M.FRAME_W - 1e-9 <= p["z0"] < p["z1"] <= M.OPEN_Z[1] - M.FRAME_W + 1e-9
    a, b = [p for p in panes if p["row"] == 0 and p["col"] in (0, 1)]
    assert b["y0"] - a["y1"] == pytest.approx(M.BAR_W)               # one mullion bar between neighbours


def test_default_ticks_and_splats_are_deterministic_and_land_in_panes():
    ticks = M.default_ticks(24.6)
    assert len(ticks) == 16 and ticks[0] == 1.5 and ticks[-1] == 24.0
    s1, s2 = M.splat_drops(ticks), M.splat_drops(ticks)
    assert s1 == s2 and len(s1) == 16
    panes = M.pane_rects()
    cols = []
    for (t, u, v, r), tk in zip(s1, ticks):
        assert t == tk and 0.013 <= r <= 0.020
        y, z = M.OPEN_Y[0] + u * M.OPEN_W, M.OPEN_Z[0] + v * M.OPEN_H
        hit = [p for p in panes if p["y0"] <= y <= p["y1"] and p["z0"] <= z <= p["z1"]]
        assert len(hit) == 1                                          # never on a bar
        cols.append(hit[0]["col"])
    assert all(a != b for a, b in zip(cols, cols[1:]))                # never the same column twice in a row
    assert M.splat_drops([]) == []


def test_splat_ticks_follow_the_timeline_a_project_names(tmp_path):
    """`timeline` reads the file's `ticks`, else its beats (the only tempo marks `mk timeline analyze` writes), and the
    default only when the file has no tempo marks at all."""
    (tmp_path / "audio").mkdir()

    def spec(tl):
        (tmp_path / "audio" / "t.json").write_text(json.dumps(tl), encoding="utf-8")
        return {"timeline": "audio/t.json"}
    analysed = {"bpm": 120, "beat_s": 0.5, "beats": [0.25, 0.75, 1.25], "downbeats": [0.25], "lines": []}
    assert M.splat_ticks(spec(analysed), str(tmp_path), 24.6) == [0.25, 0.75, 1.25]                 # beats: what analyze writes
    assert M.splat_ticks(spec({"tempo": {"beats": [2.0, 1.0]}}), str(tmp_path), 24.6) == [1.0, 2.0]    # the older layout
    assert M.splat_ticks(spec(dict(analysed, ticks=[3.0, 4.0])), str(tmp_path), 24.6) == [3.0, 4.0]    # ticks win when present
    assert M.splat_ticks(spec({"tempo": {"ticks": [5.0]}, "beats": [1.0]}), str(tmp_path), 24.6) == [5.0]
    assert M.splat_ticks(spec({"lines": []}), str(tmp_path), 24.6) == M.default_ticks(24.6)         # no tempo marks: the default
    assert M.splat_ticks({"timeline": str(tmp_path / "audio" / "t.json"), "ticks": [9.0]}, str(tmp_path / "nowhere"), 24.6) == [9.0]   # `ticks` first
    assert M.splat_ticks({}, str(tmp_path), 7.0) == M.default_ticks(7.0) == [1.5, 3.0, 4.5, 6.0]


def test_uv_roundtrip_corners():
    u, v = M.uv_from_world([M.OPEN_Y[0], M.OPEN_Y[1]], [M.OPEN_Z[0], M.OPEN_Z[1]])
    assert u.tolist() == pytest.approx([0.0, 1.0]) and v.tolist() == pytest.approx([0.0, 1.0])


def test_fog_noise_is_uniform_deterministic_and_the_png_reads_back(tmp_path):
    f = M.make_fog_noise((96, 80), seed=3)
    assert f.shape == (80, 96) and f.dtype == np.float32
    assert f.min() == 0.0 and f.max() == pytest.approx(1.0) and f.mean() == pytest.approx(0.5, abs=0.01)
    assert np.array_equal(f, M.make_fog_noise((96, 80), seed=3))
    assert not np.array_equal(f, M.make_fog_noise((96, 80), seed=4))
    # the bottom of the glass fogs first: more mass in the upper noise range near the bottom rows
    assert f[-20:].mean() > f[:20].mean()
    PIL = pytest.importorskip("PIL.Image")
    path = tmp_path / "fog.png"
    q = np.clip(np.round(f * 65535.0), 0, 65535).astype(np.uint16)
    M.write_png16(str(path), q)
    back = np.asarray(PIL.open(path))
    assert back.shape == q.shape and np.array_equal(back.astype(np.uint16), q)


def test_colors_blend_in_linear_light_and_follow_the_palette():
    C = Colors(None)
    assert C.slot("pine")[:3] == pytest.approx(Colors(DAWN).slot("pine")[:3])
    half = C.blend(base=1, text=1)
    assert half[:3] == pytest.approx([(a + b) / 2 for a, b in zip(C.slot("base")[:3], C.slot("text")[:3])])
    assert C.blend(gold=2)[:3] == pytest.approx(C.slot("gold")[:3])                     # weights are normalised
    assert C.blend(gold=1, k=0.5)[:3] == pytest.approx([v * 0.5 for v in C.slot("gold")[:3]])
    assert C.blend(gold=1, hue=0.0, chroma=1.0)[:3] == pytest.approx(C.slot("gold")[:3], abs=1e-6)
    assert max(C.light(gold=1, surface=1)[:3]) == pytest.approx(1.0)
    # a project palette overrides slots; non-slot keys (an image path) are ignored
    P = Colors({"pine": "#ffffff", "image": "poster.png"})
    assert P.slot("pine")[:3] == pytest.approx([1.0, 1.0, 1.0])
    assert set(P.resolved()) == {"pine"}
    with pytest.raises(KeyError):
        C.slot("nope")
