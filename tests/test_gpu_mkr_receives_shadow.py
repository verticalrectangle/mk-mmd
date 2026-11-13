"""mkr keeps every light's shadow off a material that receives none (the lit bake's `receives_shadow` false: an MMD
material whose PMX flag "self shadow" is off; docs/design.md: mkr). A bake written here: four tiles on the ground, seen
from straight above through an orthographic camera, each under a small card whose shadow falls on it, one light a
frame (a sun, a square area light, a point and a spot light); the tiles drawn by the deferred lighting pass
(`deferred`), by the blended pass (`blended`) and through Shader to RGB (`toon`), and a fourth that always receives
(`control`). The same scene with the three tiles' flag on and off, with shadow maps and with ray-traced shadows: off,
the ground under a card is as bright as beside it, and beside it as bright as with the flag on; on, it is dark (but
Shader to RGB, which ray-traced shadows never shadowed). Skipped without the mkr binary or a GPU; run it as
`mk gpu run -- pytest ...`."""
import json
import subprocess

import numpy as np
import pytest
from PIL import Image

from mkmmd.render import mkr as MKR

W, H, SPAN = 128, 64, 2.6                 # the picture, and the metres its width spans
TILES = {"deferred": -0.99, "blended": -0.33, "toon": 0.33, "control": 0.99}     # tile -> its centre's x
HALF, CARD, LIFT, CARD_Z = 0.3, 0.08, 0.3, 0.25         # tile half width; card half width, height, centre z
D = (0.0, -0.6, -0.8)                     # the way every light shines (glTF axes, y up)
AT = (0.0, 6.0, 8.0)                      # where the lights that have a place are: 10 m back along it
LIGHTS = (("sun", "SUN", None, 3.0, 1.0, 0.0, 0.02),       # name, type, shape, energy, size, spot size, radius
          ("area", "AREA", "SQUARE", 942.5, 1.0, 0.0, 0.0),
          ("point", "POINT", None, 3770.0, 0.0, 0.0, 0.05),
          ("spot", "SPOT", None, 3770.0, 0.0, 0.8, 0.05))


def _sock(identifier, kind, value=None, linked=False):
    return {"identifier": identifier, "name": identifier, "type": kind, "enabled": True, "linked": linked,
            "value": value}


def _material(method, receives, toon=False, grey=0.8):
    """A Diffuse BSDF (`toon`: through Shader to RGB into an Emission) drawn by render method `method`."""
    nodes = [{"name": "diffuse", "type": "BSDF_DIFFUSE",
              "inputs": [_sock("Color", "RGBA", [grey, grey, grey, 1.0]), _sock("Roughness", "VALUE", 0.0),
                         _sock("Normal", "VECTOR", [0.0, 0.0, 0.0])], "outputs": [_sock("BSDF", "SHADER")]},
             {"name": "out", "type": "OUTPUT_MATERIAL", "props": {"is_active_output": True, "target": "ALL"},
              "inputs": [_sock("Surface", "SHADER", linked=True)]}]
    links = [["diffuse", "BSDF", "out", "Surface"]]
    if toon:
        nodes += [{"name": "rgb", "type": "SHADERTORGB", "inputs": [_sock("Shader", "SHADER", linked=True)],
                   "outputs": [_sock("Color", "RGBA"), _sock("Alpha", "VALUE")]},
                  {"name": "glow", "type": "EMISSION", "inputs": [_sock("Color", "RGBA", [1.0] * 4, linked=True),
                                                                  _sock("Strength", "VALUE", 1.0)],
                   "outputs": [_sock("Emission", "SHADER")]}]
        links = [["diffuse", "BSDF", "rgb", "Shader"], ["rgb", "Color", "glow", "Color"],
                 ["glow", "Emission", "out", "Surface"]]
    return {"settings": {"surface_render_method": method}, "tree": {"nodes": nodes, "links": links}, "anim": [],
            "keyed": [], "receives_shadow": receives}


def _bake(folder, receives):
    """The scene's bake in `folder`, the three tiles' flag `receives`."""
    folder.mkdir(parents=True)
    lit_bin, floats = bytearray(), []

    def put(arr):
        off = len(lit_bin)
        lit_bin.extend(np.ascontiguousarray(arr).tobytes())
        return off

    def put_f(values):
        floats.extend(float(v) for v in values)
        return len(floats) - len(values)

    matrix = put_f([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0])

    def mesh(name, mat, rects, y):
        """Rectangles (x0, x1, z0, z1) at height y, facing up."""
        pos, tri = [], []
        for x0, x1, z0, z1 in rects:
            b = len(pos)
            pos += [(x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1)]
            tri += [b, b + 2, b + 1, b, b + 3, b + 2]
        n = len(tri) // 3
        return {"name": name, "kind": "static", "materials": [mat], "vertices": len(pos),
                "attributes": {"position": {"offset": put(np.float32(pos)), "comp": 3, "type": "f32"},
                               "normal": {"offset": put(np.tile(np.float32([0, 1, 0]), (len(pos), 1))), "comp": 3,
                                          "type": "f32"}},
                "triangles": {"offset": put(np.uint32(tri)), "count": n}, "slots": [[0, 0, n]],
                "matrix": {"offset": matrix, "frames": 1}}
    meshes = [mesh(k, k, [(c - HALF, c + HALF, -0.62, 0.62)], 0.0) for k, c in TILES.items()]
    meshes.append(mesh("cards", "card", [(c - CARD, c + CARD, CARD_Z - CARD, CARD_Z + CARD) for c in TILES.values()],
                       LIFT))
    materials = {"deferred": _material("DITHERED", receives), "blended": _material("BLENDED", receives),
                 "toon": _material("DITHERED", receives, toon=True), "control": _material("DITHERED", True),
                 "card": _material("DITHERED", True, grey=0.5)}
    n = len(LIGHTS)
    # an orthographic camera 5 m up looking straight down: the picture's right is +x, its top -z
    cam = put_f([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 5, 0, 50, 0, 0] * n)
    clip = put_f([0.1, 20.0] * n)
    lens = put_f([0, 10, 2.8, 0, 0, 1, SPAN, 36] * n)
    right, up = (1.0, 0.0, 0.0), (0.0, 0.8, -0.6)
    lights = []
    for i, (name, kind, shape, energy, size, spot, radius) in enumerate(LIGHTS):
        rows = []
        for k in range(n):
            rows += [*AT, *D, *right, *up, 1, 1, 1, energy, size, size, spot, 0.0, radius, float(k == i)]
        lights.append({"name": name, "type": kind, "shape": shape, "use_shadow": True, "frames": n,
                       "offset": put_f(rows)})
    lut_n, lo, hi = 33, -8.0, 2.0                    # the view: scene-linear values as they are, clamped to 1
    v = np.minimum(np.exp2(lo + (hi - lo) * np.arange(lut_n) / (lut_n - 1)), 1.0)
    b, g, r = np.meshgrid(v, v, v, indexing="ij")
    np.stack([r, g, b], -1).astype("<f4").tofile(folder / "view_lut.bin")
    doc = {"version": 5, "fps": 30.0, "frame0": 1, "frames": n, "outputs": [{"name": "o", "size": [W, H]}],
           "shots": [{"name": "s", "from": 1, "to": n + 1, "cameras": {"o": "cam"}}],
           "cameras": [{"name": "cam", "sensor": 36.0, "sensor_height": 24.0, "fit": "AUTO", "offset": cam,
                        "clip_at": clip}],
           "lit": {"view": {"transform": "Standard", "dither": 0.0, "zero": [0.0, 0.0, 0.0],
                            "lut": {"file": "view_lut.bin", "n": lut_n, "lo": lo, "hi": hi}},
                   "eevee": {"shadow_ray_count": 1, "shadow_step_count": 6, "use_raytracing": False},
                   "world": None, "lights": lights, "materials": materials,
                   "cameras": {"cam": {"type": "ORTHO", "offset": lens}}, "meshes": meshes}}
    (folder / "bake.json").write_text(json.dumps(doc))
    (folder / "lit.bin").write_bytes(bytes(lit_bin))
    np.float32(floats).tofile(folder / "bake.bin")
    return folder


def _render(bake, out, mode):
    """The bake's frames drawn by mkr with shadows `mode` ("map" or "rays"): one grey picture (0..1) per light."""
    try:
        exe = MKR.binary()
    except MKR.EngineError as e:
        pytest.skip(str(e))
    params = out.parent / f"{out.name}.json"
    params.write_text(json.dumps({"shadow_mode": mode}))
    r = subprocess.run([str(exe), "render", "--bake", str(bake), "--output", "o", "--frames", f"1:{len(LIGHTS)}",
                        "--out", str(out), "--samples", "16", "--params", str(params)],
                       capture_output=True, text=True, timeout=600)
    if r.returncode != 0 and any(w in r.stderr for w in ("Vulkan adapter", "needs a GPU", "cannot open the GPU",
                                                         "the adapter")):
        pytest.skip(f"mkr has no GPU: {r.stderr.strip()[-200:]}")
    assert r.returncode == 0, r.stderr[-2000:]
    return [np.asarray(Image.open(out / f"{f:05d}.png").convert("L"), float) / 255.0
            for f in range(1, len(LIGHTS) + 1)]


def _shadow(c, light):
    """Where the card over the tile centred at x = c throws the middle of its shadow on the ground: (x, z)."""
    if light == "sun":
        t = LIFT / -D[1]
        return c + t * D[0], CARD_Z + t * D[2]
    t = AT[1] / (AT[1] - LIFT)
    return AT[0] + t * (c - AT[0]), AT[2] + t * (CARD_Z - AT[2])


def _at(img, x, z):
    """The mean of the 3 x 3 pixels round ground point (x, z) (scene-linear: the view is a clamp)."""
    p = SPAN / W
    row, col = round(z / p + H / 2 - 0.5), round(x / p + W / 2 - 0.5)
    return img[row - 1:row + 2, col - 1:col + 2].mean()


@pytest.mark.parametrize("mode", ["map", "rays"])
def test_a_material_that_receives_no_shadow_takes_none_from_any_light(tmp_path, mode):
    got = {}
    for receives in (True, False):
        imgs = _render(_bake(tmp_path / f"bake_{receives}", receives), tmp_path / f"{mode}_{receives}", mode)
        for (light, *_), img in zip(LIGHTS, imgs):
            for tile, c in TILES.items():
                x, z = _shadow(c, light)
                beside = (_at(img, x - 0.17, z) + _at(img, x + 0.17, z)) / 2
                got[receives, light, tile] = (_at(img, x, z), beside)
    for light, *_ in LIGHTS:
        for tile in TILES:
            under, beside = got[False, light, tile]
            where = f"{mode}, {light}, {tile}"
            assert beside > 0.25, f"{where}: the tile is lit ({beside:.3f})"
            if tile == "control":                       # its flag stays on in both bakes
                assert under < 0.5 * beside, f"{where}: {under:.3f} under the card, {beside:.3f} beside it"
                continue
            assert abs(under - beside) <= 0.02 * beside, f"{where}: {under:.3f} under the card, {beside:.3f} beside"
            assert abs(beside - got[True, light, tile][1]) <= 1.5 / 255, f"{where}: lit otherwise as with the flag on"
            if tile == "toon" and mode == "rays":
                continue                                # ray-traced shadows never fell on Shader to RGB's light
            on_under, on_beside = got[True, light, tile]
            assert on_under < 0.5 * on_beside, f"{where}, flag on: {on_under:.3f} under, {on_beside:.3f} beside"
