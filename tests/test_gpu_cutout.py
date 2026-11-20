"""The cut-out paper look finished on the GPU (`mkr cutout`, mkr/mkr/src/cutout.rs) against the CPU look it ports
(mkmmd.core.cutout through compose_look), on made-up passes: pieces of every kind of tone (flat, lit and in shadow,
drawn from the textures, cel) overlapping at several depths, two materials in one piece, a piece cut off by the frame's
edge and a sliver the scissors may cut away, under type; and which look jobs mkr finishes and which stay on the CPU
(mkmmd.render.mkr.finishes). The parity test is skipped without the mkr binary or a GPU; run it as
`mk gpu run -- pytest ...`."""
import json
import subprocess

import numpy as np
import pytest
from PIL import Image

from mkmmd.core import palette as PAL
from mkmmd.core import shotstyle as SS
from mkmmd.render import compose as CO
from mkmmd.render import mkr as MKR

DAWN = PAL.get("rose-pine-dawn")
VEC = {"colors": {"background": "#F0E8D8", "line": "#202020"},
       "tones": {"red": "#D03030", "blue": {"lit": "#3050D0", "shade": "#203090"},
                 "eye": {"colors": ["#101010", "#5070C0", "#F0F0F0"], "at": [0.3, 0.7], "grey": 0.25},
                 "cloth": {"texture": True, "shade": "#806070"}},
       "materials": [{"match": "skin", "tone": "red"}, {"match": "coat*", "tone": "blue", "group": "coat"},
                     {"match": "eye", "tone": "eye"}, {"match": "band", "tone": "cloth"},
                     {"match": "dot", "tone": "red"}]}
MATERIALS = ["skin", "coat", "coat_sleeve", "eye", "band", "dot"]
W, H = 480, 360                       # the frame; the passes are twice that


def _passes(seed=3):
    """The passes of a frame (top row first, as mkr writes them): ids, depth, shade, tex at twice the frame, type at it."""
    rng = np.random.default_rng(seed)
    h, w = 2 * H, 2 * W
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    ids = np.zeros((h, w), np.uint16)
    depth = np.full((h, w), 50.0, np.float32)

    def lay(mask, mid, z):
        ids[mask] = mid
        depth[mask] = z + 0.002 * (y[mask] - 300.0)              # a slope: the median is a piece's middle
    lay(((x - 120) / 300) ** 2 + ((y - 380) / 260) ** 2 < 1, 1, 4.0)           # skin, cut off by the left edge
    lay((x > 300) & (x < 640) & (y > 200) & (y < 560), 2, 3.6)                   # coat, over the skin
    lay((x > 600) & (x < 760) & (y > 420) & (y < 520), 3, 3.6)                   # its sleeve: the same piece
    lay(((x - 420) ** 2 + (y - 300) ** 2) < 55 ** 2, 4, 3.0)                     # an eye on the coat
    lay((y > 600) & (y < 660) & (x > 80) & (x < 900), 5, 3.3)                    # a band across, under the eye's
    lay((x > 860) & (x < 863) & (y > 100) & (y < 103), 6, 2.0)                   # a dot the scissors may cut away
    shade = np.clip(110.0 + 90.0 * np.sin(x / 37.0) * np.cos(y / 53.0) + rng.normal(0, 25, (h, w)), 0, 255)
    tex = np.clip(128 + 120 * np.sin(np.stack([x / 23.0, y / 31.0, (x + y) / 41.0], -1))
                  + rng.normal(0, 30, (h, w, 3)), 0, 255)
    typ = np.zeros((H, W, 4), np.uint8)
    typ[300:340, 40:260] = [250, 240, 20, 200]                                  # a caption, partly see-through
    return {"id": ids, "depth": depth, "shade": shade.astype(np.uint8), "tex": tex.astype(np.uint8), "type": typ}


def _write_look(path, job, passes):
    """A pass file as mkr writes it (mkr/mkr/src/looks.rs, mkmmd.render.compose.read_look)."""
    kinds = {np.dtype(np.uint8): "u8", np.dtype(np.uint16): "u16", np.dtype(np.float32): "f32"}
    recs, at = {}, 1024
    for name, a in passes.items():
        recs[name] = {"offset": at, "shape": list(a.shape), "type": kinds[a.dtype]}
        at = (at + a.nbytes + 15) // 16 * 16
    head = json.dumps({"job": job, "passes": recs}).encode()
    buf = bytearray(at)
    buf[:12 + len(head)] = CO.LOOK_MAGIC + len(head).to_bytes(4, "little") + head
    for name, a in passes.items():
        o = recs[name]["offset"]
        buf[o:o + a.nbytes] = np.ascontiguousarray(a).astype(a.dtype.newbyteorder("<")).tobytes()
    path.write_bytes(bytes(buf))


def _job(cutout, path, t=1.3, role="frame", look="vector", rings=()):
    spec = SS.normalize({"name": "v", "style": "vector", "cutout": cutout}, DAWN, VEC)["vector"]
    tones, groups = SS.vector_table(MATERIALS, spec)
    return {"id": 0, "look": look, "role": role, "spec": spec, "table": [list(tones), list(groups)], "rings": list(rings),
            "ring_t": t, "t": t, "files": [str(path)], "centres": {}}


@pytest.mark.parametrize("cutout, case", [
    ({"fps": 8}, "figure"),
    ({"rim": False, "shadow": False, "lift": 0.0, "rough": 3.0}, "figure"),
    # no grain, no scissors, one cut; a drop shadow cast 2^32 + 40 px right: past every window, so none on the CPU (a
    # shift narrowed to 32 bits once landed it 40 px right)
    ({"grain": 0.0, "rough": 0.0, "fps": 0.0, "shadow": {"offset": [(2 ** 32 + 40) / (2 * H), -0.02]}}, "figure"),
    ({"fps": 8}, "empty"),                                     # no piece at all: the paper and the type
    ({"fps": 8}, "bad id"),                                    # an id the tables do not hold: both stop
])
def test_the_gpu_cuts_the_paper_the_cpu_cuts(tmp_path, cutout, case):
    try:
        exe = MKR.binary()
    except MKR.EngineError as e:
        pytest.skip(str(e))
    passes = _passes()
    if case == "empty":
        passes["id"][:] = 0
    elif case == "bad id":
        passes["id"][10:12, 10:12] = len(MATERIALS) + 3
    job = _job(cutout, tmp_path / "gpu.png")
    _write_look(tmp_path / "0.look", 0, passes)
    rec, why = CO.gpu_finish(job)
    assert why is None
    (tmp_path / "finish.json").write_text(json.dumps(rec))
    r = subprocess.run([str(exe), "cutout", "--look", str(tmp_path / "0.look"), "--finish", str(tmp_path / "finish.json")],
                       capture_output=True, text=True, timeout=600)
    if r.returncode != 0 and any(w in r.stderr for w in ("Vulkan adapter", "needs a GPU", "cannot open the GPU",
                                                         "the adapter")):
        pytest.skip(f"mkr has no GPU: {r.stderr.strip()[-200:]}")
    head, raw = CO.read_look(tmp_path / "0.look")
    if case == "bad id":
        with pytest.raises(IndexError):
            CO.compose_look(CO.look_passes(raw, "vector"), job)
        assert r.returncode != 0 and not (tmp_path / "gpu.png").exists()
        assert "material id 9, and the finish record's tables hold 7" in r.stderr, r.stderr[-500:]
        return
    assert r.returncode == 0, r.stderr[-2000:]
    gpu = np.asarray(Image.open(tmp_path / "gpu.png").convert("RGB")).astype(np.int16)
    cpu = CO.compose_look(CO.look_passes(raw, "vector"), job)
    cpu = (np.clip(cpu, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8).astype(np.int16)
    assert gpu.shape == cpu.shape == (H, W, 3)
    d = np.abs(gpu - cpu).max(2)
    assert d.max() <= 1 and (d > 0).mean() < 1e-3, (int(d.max()), float((d > 0).mean()))
    if case == "figure":
        assert len(np.unique(cpu.reshape(-1, 3), axis=0)) > 1000              # paper, shadows, tones: a real picture


def test_mkr_finishes_the_cut_out_paper_and_leaves_the_rest_to_the_cpu(monkeypatch, tmp_path):
    jobs = {0: _job(True, tmp_path / "a.png"), 1: _job(True, tmp_path / "b.png", role="plate"),
            2: _job(True, tmp_path / "c.png", role="matte"), 3: _job(False, tmp_path / "d.png"),
            4: dict(_job(True, tmp_path / "e.png"), look="silhouette"),
            5: _job(True, tmp_path / "f.png", rings=[{"live": True}]),
            6: _job(True, tmp_path / "g.png", role="bare"),
            7: _job(True, tmp_path / "h.png", rings=[{"live": False}]),            # rings, none live on its frame
            8: _job({"seed": 1 << 60}, tmp_path / "i.png")}                        # numpy takes it; 64 bits do not
    monkeypatch.setattr(CO.RG, "live", lambda rings, t: [r for r in rings if r["live"]])
    finish, because = MKR.finishes(jobs)
    assert sorted(finish) == [0, 1, 6, 7]
    assert finish[1]["files"] == [str(tmp_path / "b.png")] and finish[0]["step"] == 10        # 1.3 s at 8 cuts a second
    assert because == {"a matte": 1, "a vector look without cutout": 1, "a silhouette look": 1,
                       "rings live on the frame": 1, "a cutout seed past 64 bits": 1}
    monkeypatch.setenv("MK_MKR_LOOKS", "cpu")
    finish, because = MKR.finishes(jobs)
    assert finish == {} and because == {"MK_MKR_LOOKS=cpu": 9}
