"""mk export web: the packed bake comes back as mk build wrote it (to its grid), the .glb keeps its meshes, the page
carries every module it imports, and nothing is written for assets whose licences are not reviewed or forbid it."""
import gzip
import io
import json
import shutil
import struct

import numpy as np
import pytest

from mkmmd.cli import main as MAIN
from mkmmd.site import export as EX
from mkmmd.site import meshes as ME

F = 6


def fake_bake(rng):
    """A bake description and its bake.bin in the layout mk build writes: an armature, two nodes (one still), a shown
    track, morphs, a camera, a cable whose two geometries share a vertex count, and a ring whose point is NaN off its
    span; offsets packed one after another."""
    at, b, parts = 0, {"frames": F, "armatures": [], "nodes": [], "shown": [], "morphs": [], "cameras": [],
                       "by_frame": [], "rings": {"points": {}}}, []

    def put(a):
        nonlocal at
        parts.append(np.asarray(a, np.float32).reshape(-1))
        at += parts[-1].size
        return at - parts[-1].size

    walk = np.cumsum(rng.normal(0, 0.01, (F, 2 * 12)), axis=0) + 0.5
    b["armatures"].append({"name": "arm", "bones": ["a", "b"], "offset": put(walk)})
    b["nodes"].append({"name": "moving", "frames": F, "offset": put(rng.normal(0, 2, (F, 12)))})
    b["nodes"].append({"name": "still", "frames": 1, "offset": put([1, 0, 0, 0, 1, 0, 0, 0, 1, 3.25, -1.5, 0.001])})
    b["shown"].append({"name": "moving", "offset": put([1, 1, 0, 0, 1, 1])})
    b["morphs"].append({"name": "moving", "count": 3, "offset": put(rng.uniform(0, 1, (F, 3)))})
    b["cameras"].append({"name": "a@9x16", "offset": put(np.c_[rng.normal(0, 1, (F, 12)), np.full(F, 35.0),
                                                               rng.uniform(-0.1, 0.1, (F, 2))])})
    v0 = rng.normal(0, 0.3, (5, 6))
    g = [[put(v0), 5, put([[0, 1, 2, 0], [2, 3, 4, 1]]), 2]]
    g.append([put(v0 + rng.normal(0, 0.002, (5, 6))), 5, g[0][2], 2])
    b["by_frame"].append({"name": "cable", "geoms": g, "frames": [0, 0, 1, 1, -1, -1]})
    ring = np.full((F, 3), np.nan)
    ring[2:4] = [[0.5, 1.0, -0.25], [0.6, 1.0, -0.25]]
    b["rings"]["points"]["0"] = put(ring)
    return b, np.concatenate(parts)


def test_the_packed_bake_comes_back_on_its_grid_with_nans_and_indices_exact():
    bake, blob = fake_bake(np.random.default_rng(3))
    data = EX.pack(blob, bake)
    out = EX.unpack(data)
    assert out.dtype == np.float32 and out.shape == blob.shape
    assert np.array_equal(np.isnan(out), np.isnan(blob))
    ok = ~np.isnan(blob)
    assert np.abs(out[ok] - blob[ok]).max() <= max(EX.CABLE_STEP) / 2 + 1e-6

    def block(off, n):
        return slice(off, off + n)
    arm = bake["armatures"][0]
    assert np.abs(out[block(arm["offset"], F * 24)] - blob[block(arm["offset"], F * 24)]).max() <= EX.STEP["joints"] / 2 + 1e-7
    tri = bake["by_frame"][0]["geoms"][0]
    for s in (block(tri[2], 8), block(bake["shown"][0]["offset"], F)):
        assert np.array_equal(out[s], blob[s])                       # triangle indices and visibility: not approximated
    pt = bake["rings"]["points"]["0"]
    assert np.array_equal(out[pt:pt + 3 * F][6:12], blob[pt:pt + 3 * F][6:12])   # a raw block stays float32


def test_a_block_too_large_for_its_grid_and_the_gaps_between_blocks_are_kept_as_floats():
    bake, blob = fake_bake(np.random.default_rng(5))
    blob = np.concatenate([blob, np.float32([7.5, -2.0])])            # floats past the last block the description names
    node = bake["nodes"][1]["offset"]
    blob[node + 9] = 1e9                                              # a translation far beyond the int32 grid
    out = EX.unpack(EX.pack(blob, bake))
    assert out[node + 9] == np.float32(1e9) and list(out[-2:]) == [7.5, -2.0]


def test_a_bake_whose_blocks_overlap_or_overrun_is_refused():
    bake, blob = fake_bake(np.random.default_rng(1))
    bake["nodes"][1]["offset"] = bake["nodes"][0]["offset"] + 3
    with pytest.raises(EX.ExportError, match="overlaps"):
        EX.pack(blob, bake)
    bake, blob = fake_bake(np.random.default_rng(1))
    with pytest.raises(EX.ExportError, match="holds"):
        EX.pack(blob[:-10], bake)


def glb(images):
    """A .glb with one triangle and the given embedded PNG images."""
    pos = np.float32([[0, 0, 0], [1, 0, 0], [0, 1, 0]]).tobytes()
    chunks, views = [pos], [{"buffer": 0, "byteOffset": 0, "byteLength": len(pos)}]
    off = len(pos)
    for png in images:
        views.append({"buffer": 0, "byteOffset": off, "byteLength": len(png)})
        chunks.append(png)
        off += len(png)
        pad = -off % 4
        chunks.append(b"\0" * pad)
        off += pad
    binary = b"".join(chunks)
    doc = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
           "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
           "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"}],
           "bufferViews": views, "buffers": [{"byteLength": len(binary)}],
           "images": [{"bufferView": i + 1, "mimeType": "image/png"} for i in range(len(images))]}
    js = json.dumps(doc).encode()
    js += b" " * (-len(js) % 4)
    return (struct.pack("<III", 0x46546C67, 2, 28 + len(js) + len(binary)) + struct.pack("<I4s", len(js), b"JSON") + js
            + struct.pack("<I4s", len(binary), b"BIN\0") + binary)


def test_the_web_glb_has_webp_textures_no_larger_than_asked_and_the_same_mesh(tmp_path):
    from PIL import Image
    rng = np.random.default_rng(0)
    pngs = []
    for size, alpha in (((512, 256), False), ((64, 64), True)):
        a = rng.integers(0, 255, (size[1], size[0], 4), np.uint8)
        a[..., 3] = 255 if not alpha else a[..., 3]
        buf = io.BytesIO()
        Image.fromarray(a, "RGBA").save(buf, "PNG")
        pngs.append(buf.getvalue())
    src, dst = tmp_path / "a.glb", tmp_path / "b.glb"
    src.write_bytes(glb(pngs))
    dst.write_bytes(EX.web_glb(src.read_bytes(), side=128))
    V0, T0 = ME.read_glb(src)
    V1, T1 = ME.read_glb(dst)
    assert np.array_equal(V0, V1) and np.array_equal(T0, T1)
    n = struct.unpack_from("<I", dst.read_bytes(), 12)[0]
    doc = json.loads(dst.read_bytes()[20:20 + n])
    data = dst.read_bytes()[20 + n + 8:]
    got = [Image.open(io.BytesIO(data[doc["bufferViews"][im["bufferView"]]["byteOffset"]:][:doc["bufferViews"][im["bufferView"]]["byteLength"]]))
           for im in doc["images"]]
    assert [im["mimeType"] for im in doc["images"]] == ["image/webp", "image/webp"]
    assert got[0].size == (128, 64) and got[0].mode == "RGB" and got[1].mode == "RGBA"
    assert all(v["byteOffset"] % 4 == 0 for v in doc["bufferViews"])


def test_the_page_carries_every_module_it_imports_with_the_static_api():
    files = EX.page_files()
    js = {p for p in files if p.endswith(".js")}
    for p in js:
        src = files[p].decode()
        for m in EX.IMPORT.finditer(src):
            dep = EX.posixpath.normpath(EX.posixpath.join(EX.posixpath.dirname(p), m.group(1)))
            assert dep in files, f"{p} imports {m.group(1)}, not in the page"
    assert files["app/js/api.js"] == (EX.WEB / "js" / "api_static.js").read_bytes()
    assert {"app/js/scene/scene.js", "app/js/music.js", "app/js/viewer/glb.js"} <= js
    assert not any(p.endswith(("review.js", "app.js", "checks.js")) for p in js)       # the site's own pages stay home


def registry(root, **entry):
    root.mkdir(parents=True, exist_ok=True)
    readme = root / "ReadMe.txt"
    readme.write_text("terms", encoding="utf-8")
    e = {"kind": "model", "slug": "m", "name": "M", "path": str(root / "m.pmx"), "readmes": [str(readme)], **entry}
    (root / "registry.json").write_text(json.dumps([e]), encoding="utf-8")


def project(root):
    root.mkdir(parents=True)
    (root / "mk.toml").write_text('[project]\nname = "p"\nfps = 30\nframe0 = 1\nduration = 0.2\nblend = "build/p.blend"\n'
                                  '[[cast]]\nname = "a"\nasset = "m"\n', encoding="utf-8")


def mk(argv, capsys):
    with pytest.raises(SystemExit) as e:
        MAIN.main(argv)
    return e.value.code, json.loads(capsys.readouterr().out)


@pytest.mark.parametrize("entry, why", [({"license": "unreviewed", "credit": "M by A"}, "license not reviewed"),
                                        ({"license": "free", "credit": "M by A", "restrictions": "non-commercial; no redistribution"},
                                         "forbid redistribution")])
def test_nothing_is_written_for_assets_unreviewed_or_not_to_be_redistributed(tmp_path, monkeypatch, capsys, entry, why):
    monkeypatch.setenv("MK_ASSETS", str(tmp_path / "assets"))
    monkeypatch.setenv("MK_CACHE", str(tmp_path / "cache"))
    registry(tmp_path / "assets", **entry)
    project(tmp_path / "proj")
    code, out = mk(["export", "web", "--project", str(tmp_path / "proj"), "--out", str(tmp_path / "web")], capsys)
    assert code == 1 and out["written"] is False and any(why in p for p in out["problems"])
    assert not (tmp_path / "web").exists()


@pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="no ffmpeg")
def test_an_export_holds_the_page_its_data_and_the_licences_and_writes_only_over_its_own(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MK_ASSETS", str(tmp_path / "assets"))
    monkeypatch.setenv("MK_CACHE", str(tmp_path / "cache"))
    registry(tmp_path / "assets", license="CC0", credit="M by A", restrictions="none")
    root = tmp_path / "proj"
    project(root)
    bake, blob = fake_bake(np.random.default_rng(2))
    bake.update({"fps": 30.0, "frame0": 1, "duration": 0.2, "outputs": [{"name": "9x16", "size": [540, 960]}],
                 "cut": {"9x16": []}})
    (root / "build" / "p.bake").mkdir(parents=True)
    (root / "build" / "p.bake" / "bake.json").write_text(json.dumps(bake), encoding="utf-8")
    blob.tofile(root / "build" / "p.bake" / "bake.bin")
    (root / "build" / "p.bake" / "scene.glb").write_bytes(glb([]))
    out = tmp_path / "web"
    code, got = mk(["export", "web", "--project", str(root), "--out", str(out), "--video", "none"], capsys)
    assert code == 0 and got["written"] is True
    scene = json.loads((out / "data" / "scene.json").read_text())
    assert scene["drafts"] == [] and scene["bin"] == "data/bake.mkz"
    assert np.array_equal(np.isnan(EX.unpack((out / "data" / "bake.mkz").read_bytes())), np.isnan(blob))
    assert gzip.decompress((out / "data" / "scene.glb.gz").read_bytes())[:4] == b"glTF"
    assert (out / "licences" / "m" / "ReadMe.txt").read_text() == "terms"
    assert "M by A" in (out / "CREDITS.md").read_text()
    assert str(tmp_path) not in "".join(p.read_text(errors="ignore") for p in (out / "data").glob("*.json"))
    assert (out / "index.html").is_file() and (out / "app" / "js" / "api.js").is_file()
    code, again = mk(["export", "web", "--project", str(root), "--out", str(out), "--video", "none"], capsys)
    assert code == 0                                                    # its own folder: written again
    other = tmp_path / "other"
    other.mkdir()
    (other / "keep.txt").write_text("mine")
    code, refused = mk(["export", "web", "--project", str(root), "--out", str(other), "--video", "none"], capsys)
    assert code == 2 and "not written by" in refused["error"] and (other / "keep.txt").read_text() == "mine"
