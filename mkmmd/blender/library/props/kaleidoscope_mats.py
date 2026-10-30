"""Materials of the cardboard kaleidoscope toy (mkmmd/blender/library/props/kaleidoscope.py): the printed paper (its
print and wear as one packed image from mkmmd/core/kaleidoscope_print.py), the bare paper of the seam and the card, the
lacquered turned rims and the painted face of the eyepiece cap (chipped at their edges), the brass eyelet, the lens,
the dark tunnel and the glowing view behind it, the frosted and the clear glass of the chamber, the beads and the glass
bits. A helper module: it registers no builder.

Everything that sees the toy close up (a macro shot of a 50 mm tube) needs detail below the print's pixels: the paper's
tooth and fibres, the turning marks under the lacquer and the frosting are procedural noise in the object's own frame
(metres), so they stay put on a moving toy and stay sharp at any distance; every part has its origin on the toy's axis,
the part's own Y, so the turned rings and the chips know their radius. Transparent parts are BLENDED (EEVEE sorts them
per object, so each has its origin at its own centre) and mix a Transparent BSDF with their surface, which Cycles
renders the same way."""
import bpy
import numpy as np

from .cafe_kit import L, N, principled


def packed_image(name, rgba, colorspace="sRGB", key=None):
    """A packed Blender image from an (h, w, 4) float array whose row 0 is the image's top. An image of that name built
    from the same `key` (any string) is reused."""
    img = bpy.data.images.get(name)
    if img is not None and key is not None and img.get("mk_key") == key:
        return img
    if img is not None:
        bpy.data.images.remove(img)
    h, w = rgba.shape[:2]
    img = bpy.data.images.new(name, w, h, alpha=True)
    img.colorspace_settings.name = colorspace
    img.alpha_mode = "CHANNEL_PACKED"
    img.pixels.foreach_set(np.ascontiguousarray(rgba[::-1], np.float32).ravel())
    img.pack()
    if key is not None:
        img["mk_key"] = key
    return img


# ================================================================================================= node shorthands
def _coords(nt, loc=(-1900, 0)):
    """Object coordinates (metres in the prop's frame, as built)."""
    return N(nt, "ShaderNodeTexCoord", loc).outputs["Object"]


def _noise(nt, vec, scale, loc, detail=4.0, rough=0.6):
    n = N(nt, "ShaderNodeTexNoise", loc, inputs={"Scale": scale, "Detail": detail, "Roughness": rough})
    L(nt, vec, n.inputs["Vector"])
    return n.outputs["Fac"]


def _math(nt, op, a, b, loc, clamp=False):
    n = N(nt, "ShaderNodeMath", loc, operation=op, use_clamp=clamp)
    for sock, v in zip(n.inputs, (a, b)):
        if isinstance(v, (int, float)):
            sock.default_value = v
        else:
            L(nt, v, sock)
    return n.outputs[0]


def _ramp(nt, x, lo, hi, loc):
    """0 below lo, 1 above hi, a smoothstep between."""
    n = N(nt, "ShaderNodeMapRange", loc, interpolation_type="SMOOTHSTEP",
          inputs={"From Min": lo, "From Max": hi, "To Min": 0.0, "To Max": 1.0})
    L(nt, x, n.inputs["Value"])
    return n.outputs["Result"]


def _mix(nt, fac, a, b, loc):
    """Colour mix a -> b by fac (sockets or values)."""
    n = N(nt, "ShaderNodeMix", loc, data_type="RGBA")
    for sock, v in ((n.inputs[0], fac), (n.inputs[6], a), (n.inputs[7], b)):
        if isinstance(v, (int, float, tuple)):
            sock.default_value = v
        else:
            L(nt, v, sock)
    return n.outputs[2]


def _bump(nt, height, strength, distance, loc):
    b = N(nt, "ShaderNodeBump", loc, inputs={"Strength": strength, "Distance": distance})
    L(nt, height, b.inputs["Height"])
    return b.outputs["Normal"]


def _see_through(nt, out, surface, share, tint=(1.0, 1.0, 1.0, 1.0), loc=(300, 0)):
    """`surface` over a Transparent BSDF: `share` (a value or a socket) of it, the rest let through tinted."""
    clear = N(nt, "ShaderNodeBsdfTransparent", (loc[0] - 300, loc[1] - 250), inputs={"Color": tint})
    mix = N(nt, "ShaderNodeMixShader", loc)
    if isinstance(share, (int, float)):
        mix.inputs[0].default_value = share
    else:
        L(nt, share, mix.inputs[0])
    L(nt, clear.outputs[0], mix.inputs[1])
    L(nt, surface, mix.inputs[2])
    L(nt, mix.outputs[0], out.inputs["Surface"])


def _blended(m, both=False):
    """EEVEE's blended transparency; `both` draws the back faces too (a coloured glass bit), else only the faces turned
    to the camera (a pane has one frosted or glossy face toward the viewer, the other would double its veil)."""
    m.surface_render_method = "BLENDED"
    m.use_backface_culling = not both
    m.show_transparent_back = both


# ================================================================================================= paper
def paper(K, image, bare, wear):
    """The printed paper: the print's colour (image RGB) worn through to the `bare` paper where its alpha (the wear map)
    times `wear` says, ragged at the edge by the paper's tooth; a coated print's soft sheen; the tooth and the fibres as
    bump, rougher and more fibrous where it is rubbed."""
    m, nt, out = K.new_mat("paper")
    uv = N(nt, "ShaderNodeUVMap", (-1900, 400), uv_map="UVMap")
    tex = N(nt, "ShaderNodeTexImage", (-1650, 400), image=image, interpolation="Cubic", extension="EXTEND")
    L(nt, uv.outputs[0], tex.inputs["Vector"])
    P = _coords(nt)
    tooth = _noise(nt, P, 2600.0, (-1650, -100), detail=6.0, rough=0.65)          # ~0.4 mm
    fibre = _noise(nt, P, 9000.0, (-1650, -400), detail=3.0, rough=0.5)           # ~0.1 mm
    worn = _math(nt, "ADD", _math(nt, "MULTIPLY", tex.outputs["Alpha"], float(wear), (-1350, 250)),
                 _math(nt, "MULTIPLY", _math(nt, "SUBTRACT", tooth, 0.5, (-1450, 0)), 0.5, (-1300, 0)), (-1150, 150))
    edge = _ramp(nt, worn, 0.32, 0.62, (-950, 150))
    colour = _mix(nt, edge, tex.outputs["Color"], bare, (-650, 350))
    rough = _math(nt, "ADD", _math(nt, "MULTIPLY", edge, 0.34, (-700, 50)), 0.38, (-550, 50))
    lift = _math(nt, "ADD", _math(nt, "MULTIPLY", edge, 1.4, (-1150, -450)), 0.25, (-1000, -450))
    height = _math(nt, "ADD", _math(nt, "MULTIPLY", tooth, 0.6, (-1150, -200)),
                   _math(nt, "MULTIPLY", fibre, lift, (-850, -400)), (-700, -300))
    b = principled(nt, out, loc=(300, 100), **{"Specular IOR Level": 0.45, "Sheen Weight": 0.25, "Sheen Roughness": 0.4,
                                                "Coat Weight": 0.12, "Coat Roughness": 0.3})
    L(nt, colour, b.inputs["Base Color"])
    L(nt, rough, b.inputs["Roughness"])
    L(nt, _bump(nt, height, 0.22, 0.00012, (-400, -300)), b.inputs["Normal"])
    return m


def paper_edge(K, bare):
    """The paper's cut edge at the seam: bare paper."""
    m, nt, out = K.new_mat("paper_edge")
    principled(nt, out, **{"Base Color": bare, "Roughness": 0.75, "Specular IOR Level": 0.35})
    return m


def card(K, colour):
    """The cardboard of the tube's inside and its ends: matte, mottled."""
    m, nt, out = K.new_mat("card")
    n = _noise(nt, _coords(nt), 3000.0, (-900, 0), detail=5.0)
    b = principled(nt, out, **{"Roughness": 0.9, "Specular IOR Level": 0.3})
    L(nt, _mix(nt, _math(nt, "MULTIPLY", n, 0.35, (-600, 100)), colour, tuple(0.6 * c for c in colour[:3]) + (1.0,),
               (-300, 100)), b.inputs["Base Color"])
    return m


# ================================================================================================= turned and painted
def _radial(nt, P, loc):
    """(distance from the toy's axis, position along it) as sockets, from the part's own coordinates: its origin is on
    the axis, which is its Y."""
    sep = N(nt, "ShaderNodeSeparateXYZ", loc)
    L(nt, P, sep.inputs[0])
    rr = _math(nt, "ADD", _math(nt, "MULTIPLY", sep.outputs["X"], sep.outputs["X"], (loc[0] + 200, loc[1] + 120)),
               _math(nt, "MULTIPLY", sep.outputs["Z"], sep.outputs["Z"], (loc[0] + 200, loc[1] - 120)), (loc[0] + 400, loc[1]))
    return _math(nt, "SQRT", rr, 0.0, (loc[0] + 550, loc[1])), sep.outputs["Y"]


def lacquer(K, base, colour, chip_r):
    """Lacquered paint on turned wood: a glossy coat over a faintly mottled colour, the lathe's turning marks as rings
    in the bump, and chips through to the pale wood on the edges that stand out furthest (radius over `chip_r`)."""
    m, nt, out = K.new_mat(base)
    P = _coords(nt)
    r, s = _radial(nt, P, (-1900, -350))
    rings = N(nt, "ShaderNodeTexWave", (-1050, -450), wave_type="BANDS", bands_direction="X",
              inputs={"Scale": 1.0, "Distortion": 0.6, "Detail": 2.0})
    L(nt, _math(nt, "MULTIPLY", _math(nt, "ADD", r, s, (-1250, -450)), 2400.0, (-1150, -550)), rings.inputs["Vector"])
    mottle = _noise(nt, P, 450.0, (-1050, 250), detail=3.0)
    flecks = _noise(nt, P, 1400.0, (-1050, 0), detail=5.0, rough=0.7)
    exposed = _ramp(nt, r, chip_r, chip_r + 0.0004, (-850, -100))
    chip = _math(nt, "GREATER_THAN", _math(nt, "MULTIPLY", exposed, flecks, (-700, 0)), 0.57, (-550, 0))
    wood = (0.62, 0.48, 0.33, 1.0)
    paint = _mix(nt, _math(nt, "MULTIPLY", mottle, 0.12, (-800, 250)), colour,
                 tuple(min(1.0, 1.15 * c) for c in colour[:3]) + (1.0,), (-600, 250))
    height = _math(nt, "ADD", _math(nt, "MULTIPLY", rings.outputs["Fac"], 0.25, (-800, -450)),
                   _math(nt, "MULTIPLY", chip, -0.6, (-800, -300)), (-600, -400))
    b = principled(nt, out, loc=(300, 100), **{"Roughness": 0.34, "Coat Roughness": 0.07, "Specular IOR Level": 0.5})
    L(nt, _mix(nt, chip, paint, wood, (-350, 200)), b.inputs["Base Color"])
    L(nt, _math(nt, "SUBTRACT", 0.85, _math(nt, "MULTIPLY", chip, 0.8, (-350, -100)), (-200, -100)), b.inputs["Coat Weight"])
    L(nt, _bump(nt, height, 0.12, 0.00008, (-300, -400)), b.inputs["Normal"])
    return m


def brass(K, colour):
    """The eyelet: brass, a little worn."""
    m, nt, out = K.new_mat("ferrule")
    n = _noise(nt, _coords(nt), 5000.0, (-700, -100), detail=4.0)
    b = principled(nt, out, **{"Base Color": colour, "Metallic": 1.0, "Specular IOR Level": 0.5})
    L(nt, _math(nt, "ADD", _math(nt, "MULTIPLY", n, 0.18, (-400, -100)), 0.16, (-250, -100)), b.inputs["Roughness"])
    return m


# ================================================================================================= glass and light
def lens(K):
    """The eyepiece lens: clear, its sharp reflection's share growing at grazing angles (the glint on its dome)."""
    m, nt, out = K.new_mat("lens")
    _blended(m)
    fres = N(nt, "ShaderNodeFresnel", (-500, 250), inputs={"IOR": 1.52})
    glossy = N(nt, "ShaderNodeBsdfGlossy", (-300, 0), inputs={"Roughness": 0.03})
    _see_through(nt, out, glossy.outputs[0], _math(nt, "ADD", fres.outputs[0], 0.02, (-250, 250), clamp=True),
                 tint=(0.97, 0.98, 1.0, 1.0))
    return m


def tunnel(K, colour):
    """The mirrors' dark mouth behind the eyehole: black, no sheen (a light probe would light a closed tube)."""
    m, nt, out = K.new_mat("tunnel")
    principled(nt, out, **{"Base Color": tuple(0.03 * c for c in colour[:3]) + (1.0,), "Roughness": 0.6,
                           "Specular IOR Level": 0.0})
    return m


def view(K, image, glow):
    """The view at the end of the tunnel: the mirrored beads, lit from behind (emission) over a dark diffuse."""
    m, nt, out = K.new_mat("view")
    uv = N(nt, "ShaderNodeUVMap", (-900, 0), uv_map="UVMap")
    tex = N(nt, "ShaderNodeTexImage", (-650, 0), image=image, interpolation="Linear", extension="EXTEND")
    L(nt, uv.outputs[0], tex.inputs["Vector"])
    b = principled(nt, out, **{"Roughness": 0.5, "Emission Strength": float(glow)})
    L(nt, tex.outputs["Color"], b.inputs["Base Color"])
    L(nt, tex.outputs["Color"], b.inputs["Emission Color"])
    return m


def frost(K, colour, veil=0.32):
    """The chamber's frosted window: a milky veil (`veil` of it a rough pale surface, the rest let through) with a fine
    frosted grain, so the beads behind it show as soft colour in EEVEE and Cycles alike."""
    m, nt, out = K.new_mat("frost")
    _blended(m)
    grain = _noise(nt, _coords(nt), 7000.0, (-900, -300), detail=3.0, rough=0.7)
    surf = principled(nt, out, loc=(-150, 150), **{"Base Color": colour, "Roughness": 0.55, "Specular IOR Level": 0.6,
                                                   "Coat Weight": 0.3, "Coat Roughness": 0.25})
    L(nt, _bump(nt, grain, 0.35, 0.00005, (-500, -300)), surf.inputs["Normal"])
    _see_through(nt, out, surf.outputs[0], veil)
    return m


def clear_glass(K):
    """The chamber's clear inner glass: nearly invisible, a faint reflection."""
    m, nt, out = K.new_mat("glass")
    _blended(m)
    fres = N(nt, "ShaderNodeFresnel", (-500, 250), inputs={"IOR": 1.5})
    glossy = N(nt, "ShaderNodeBsdfGlossy", (-300, 0), inputs={"Roughness": 0.05})
    _see_through(nt, out, glossy.outputs[0], fres.outputs[0])
    return m


def bead(K, slot, colour, glow):
    """An opaque glass bead: glossy colour under a coat, glowing a little (light through the frosted end, `glow`)."""
    m, nt, out = K.new_mat(f"bead_{slot}")
    principled(nt, out, **{"Base Color": colour, "Roughness": 0.16, "Coat Weight": 0.7, "Coat Roughness": 0.05,
                           "Specular IOR Level": 0.6, "Emission Color": colour, "Emission Strength": 0.3 * glow})
    return m


def bit(K, slot, colour, glow):
    """A coloured glass bit or rod: tinted, mostly see-through, glossy, glowing with the light through the frosted end."""
    m, nt, out = K.new_mat(f"bit_{slot}")
    _blended(m, both=True)
    body = principled(nt, out, loc=(-150, 150), **{"Base Color": colour, "Roughness": 0.06, "Specular IOR Level": 0.8,
                                                   "Emission Color": colour, "Emission Strength": 0.8 * glow})
    _see_through(nt, out, body.outputs[0], 0.62, tint=colour)
    return m
