"""Materials of the bedroom soft furnishings: quilt (Memphis print, diamond quilting, stripes and binding), lining,
sheet, pillowcase, mattress ticking and painted wood. Procedural shader nodes only (EEVEE Next)."""
import math

from .bedroom_soft_nodes import NB, memphis
from .cafe_kit import principled

AA = 0.0022                           # anti-aliasing half width of printed edges (m)


def _tone(nb, nt, scale, detail=3.0):
    """A smooth 0..1 noise over the object coordinates (large-scale tone drift)."""
    n = nb.node("ShaderNodeTexNoise")
    n.inputs["Scale"].default_value = scale
    n.inputs["Detail"].default_value = detail
    n.inputs["Roughness"].default_value = 0.5
    tc = nb.node("ShaderNodeTexCoord")
    nt.links.new(tc.outputs["Object"], n.inputs["Vector"])
    return nb.wrap(n.outputs["Fac"])


def _noise(nb, nt, scale, vec, detail=2.0, rough=0.5):
    n = nb.node("ShaderNodeTexNoise")
    n.inputs["Scale"].default_value = scale
    n.inputs["Detail"].default_value = detail
    n.inputs["Roughness"].default_value = rough
    nt.links.new(vec, n.inputs["Vector"])
    return nb.wrap(n.outputs["Fac"])


def _bump(nb, nt, height, strength, distance, normal=None):
    b = nb.node("ShaderNodeBump")
    b.inputs["Strength"].default_value = strength
    b.inputs["Distance"].default_value = distance
    nb.link(height, b.inputs["Height"])
    if normal is not None:
        nt.links.new(normal, b.inputs["Normal"])
    return b.outputs["Normal"]


def relief(nb, u, v, e, pitch, line_w=0.08, fade=0.025):
    """Diamond quilting in the flat cloth coordinates: (puff 0..1, stitch-line mask 0..1). The puffs flatten toward the
    hem (distance e)."""
    s = 0.7071067811865476 / pitch
    a, b = (u + v) * s, (u - v) * s
    sa, sb = nb.sin(nb.fract(a) * math.pi), nb.sin(nb.fract(b) * math.pi)
    puff = sa * sb
    line = nb.sat(1.0 - nb.min(sa, sb) / line_w)
    return puff, line, nb.sat(e / fade)


def _principled(nt, out, base, normal, rough, sheen=0.5, sheen_rough=0.4, spec=0.25, sheen_tint=(0.95, 0.95, 1.0, 1.0)):
    b = principled(nt, out, loc=(1700, 0), **{"Roughness": rough, "Sheen Weight": sheen, "Sheen Roughness": sheen_rough,
                                              "Specular IOR Level": spec})
    b.inputs["Sheen Tint"].default_value = sheen_tint
    if isinstance(base, (tuple, list)):
        b.inputs["Base Color"].default_value = tuple(base)
    else:
        nt.links.new(base, b.inputs["Base Color"])
    nt.links.new(normal, b.inputs["Normal"])
    return b


def _voronoi(nb, nt, u, v, scale):
    """(distance in metres to the nearest feature point, and three random numbers of its cell) on the flat cloth."""
    n = nb.node("ShaderNodeTexVoronoi", voronoi_dimensions="3D", feature="F1")
    n.inputs["Scale"].default_value = scale
    n.inputs["Randomness"].default_value = 1.0
    nt.links.new(nb.vec(u, v, 0.0), n.inputs["Vector"])
    r, g, b = nb.csplit(n.outputs["Color"])
    return nb.wrap(n.outputs["Distance"]) / scale, r, g, b


def confetti(nb, nt, col, u, v, dm, zone, blob_cols, speck_cols, blob_mix=0.3, blob_scale=2.6, speck_scale=15.0):
    """Tone-on-tone blobs (big soft discs, a tint of `blob_cols` mixed into the ground) and small specks (solid dots of
    `speck_cols`) over the ground colour socket `col`; `dm` is the signed distance to the print's motifs (specks keep
    away from them), `zone` a 0..1 mask of where the field may be decorated."""
    if blob_cols:
        n = len(blob_cols)
        dist, r, g, b = _voronoi(nb, nt, u, v, blob_scale)
        m = nb.edge(dist - (0.07 + 0.08 * r), AA * 3.0) * nb.gt(b, 0.4) * zone
        pal = nb.table([(i / n, c) for i, c in enumerate(blob_cols)], (nb.floor(g * (n - 0.01)) + 0.5) / n)
        col = nb.cmix(m * blob_mix, col, pal.outputs["Color"])
    if speck_cols:
        n = len(speck_cols)
        dist, r, g, b = _voronoi(nb, nt, u + 3.7, v - 1.9, speck_scale)
        m = nb.edge(dist - (0.0065 + 0.006 * r), AA) * nb.gt(b, 0.55) * zone * nb.sat((dm - 0.022) / 0.004)
        pal = nb.table([(i / n, c) for i, c in enumerate(speck_cols)], (nb.floor(g * (n - 0.01)) + 0.5) / n)
        col = nb.cmix(m, col, pal.outputs["Color"])
    return col


def quilt(K, spec):
    """The quilt's pattern side. spec: nx, ny, sx, sy, tables (lookup tables of the print), ground (RGBA), colors (the
    palette the print indexes), ink, gold, binding (RGBA of the border stripes), pitch (quilting)."""
    m, nt, out = K.new_mat("quilt")
    nb = NB(nt)
    u, v = nb.uv("UVMap")
    e, _ = nb.uv("edge")
    mask, c1, _, dm = memphis(nb, u, v, spec["nx"], spec["ny"], spec["sx"], spec["sy"], spec["tables"])
    cols = spec["colors"]
    pal = nb.table([(i / len(cols), c) for i, c in enumerate(cols)], (c1 + 0.5) / len(cols))
    field = mask * nb.sat((e - 0.095) / 0.003)
    tone = _tone(nb, nt, 2.4)
    ground = nb.cmix(tone, spec["ground"], tuple(c * 0.92 if i < 3 else c for i, c in enumerate(spec["ground"])))
    zone = nb.sat((e - 0.095) / 0.003)
    ground = confetti(nb, nt, ground, u, v, dm, zone, spec["blobs"], spec["specks"])
    col = nb.cmix(field, ground, pal.outputs["Color"])
    col = nb.cmix(nb.edge(nb.abs(e - 0.046) - 0.017, AA), col, spec["ink"])
    col = nb.cmix(nb.edge(nb.abs(e - 0.0745) - 0.0045, AA), col, spec["gold"])
    col = nb.cmix(nb.edge(e - 0.017, AA), col, spec["binding"])
    puff, line, fade = relief(nb, u, v, e, spec["pitch"])
    col = nb.cmul(col, (0.42, 0.42, 0.5, 1.0), line * 0.6)
    col = nb.cmul(col, (0.9, 0.9, 0.95, 1.0), (1.0 - puff) * 0.5)
    tc = nb.node("ShaderNodeTexCoord")
    weave = _noise(nb, nt, 260.0, tc.outputs["Object"], detail=3.0, rough=0.6)
    n1 = _bump(nb, nt, nb.sqrt(puff) * fade, 0.85, 0.006)
    n2 = _bump(nb, nt, weave, 0.07, 0.0006, normal=n1)
    _principled(nt, out, col, n2, 0.78, sheen=0.35, sheen_rough=0.4, spec=0.22)
    return m


def lining(K, spec):
    """The quilt's reverse: a plain pastel with the stitch lines and the binding."""
    m, nt, out = K.new_mat("lining")
    nb = NB(nt)
    u, v = nb.uv("UVMap")
    e, _ = nb.uv("edge")
    tone = _tone(nb, nt, 2.0)
    base = spec["lining"]
    col = nb.cmix(tone, base, tuple(c * 0.9 if i < 3 else c for i, c in enumerate(base)))
    stripe = nb.sat(nb.sin(u * (math.pi / 0.03)) * 4.0 + 0.5)                       # tone-on-tone 3 cm stripes
    col = nb.cmul(col, (0.93, 0.93, 0.96, 1.0), stripe * 0.6)
    col = nb.cmix(nb.edge(e - 0.017, AA), col, spec["binding"])
    puff, line, fade = relief(nb, u, v, e, spec["pitch"])
    col = nb.cmul(col, (0.5, 0.5, 0.58, 1.0), line * 0.5)
    tc = nb.node("ShaderNodeTexCoord")
    weave = _noise(nb, nt, 240.0, tc.outputs["Object"], detail=3.0, rough=0.6)
    n1 = _bump(nb, nt, nb.sqrt(puff) * fade, 0.7, 0.005)
    n2 = _bump(nb, nt, weave, 0.07, 0.0006, normal=n1)
    _principled(nt, out, col, n2, 0.82, sheen=0.4, sheen_rough=0.45, spec=0.2)
    return m


def binding(K, color):
    m, nt, out = K.new_mat("binding")
    nb = NB(nt)
    tc = nb.node("ShaderNodeTexCoord")
    weave = _noise(nb, nt, 300.0, tc.outputs["Object"], detail=3.0, rough=0.6)
    n1 = _bump(nb, nt, weave, 0.12, 0.0006)
    _principled(nt, out, color, n1, 0.7, sheen=0.4, sheen_rough=0.4, spec=0.25)
    return m


def sheet(K, color):
    """The fitted sheet: a plain pastel, a little tone drift, a fine weave."""
    m, nt, out = K.new_mat("sheet")
    nb = NB(nt)
    tone = _tone(nb, nt, 3.0)
    col = nb.cmix(tone, color, tuple(c * 0.9 if i < 3 else c for i, c in enumerate(color)))
    tc = nb.node("ShaderNodeTexCoord")
    weave = _noise(nb, nt, 280.0, tc.outputs["Object"], detail=3.0, rough=0.6)
    n1 = _bump(nb, nt, weave, 0.08, 0.0006)
    _principled(nt, out, col, n1, 0.86, sheen=0.4, sheen_rough=0.5, spec=0.2)
    return m


def pillowcase(K, color):
    """A pillowcase with an Oxford flange: a stitch line round the pillow, 38 mm from the seam."""
    m, nt, out = K.new_mat("pillowcase")
    nb = NB(nt)
    e, _ = nb.uv("edge")
    tone = _tone(nb, nt, 4.0)
    col = nb.cmix(tone, color, tuple(c * 0.9 if i < 3 else c for i, c in enumerate(color)))
    line = nb.edge(nb.abs(e - 0.038) - 0.0009, AA)
    col = nb.cmul(col, (0.55, 0.52, 0.55, 1.0), line * 0.7)
    tc = nb.node("ShaderNodeTexCoord")
    weave = _noise(nb, nt, 300.0, tc.outputs["Object"], detail=3.0, rough=0.6)
    n1 = _bump(nb, nt, weave, 0.08, 0.0006)
    _principled(nt, out, col, n1, 0.84, sheen=0.45, sheen_rough=0.45, spec=0.2)
    return m


def ticking(K, color, stitch):
    """Mattress fabric: a pastel with faint tone drift and a diamond stitch."""
    m, nt, out = K.new_mat("ticking")
    nb = NB(nt)
    u, v = nb.uv("UVMap")
    tone = _tone(nb, nt, 3.5)
    col = nb.cmix(tone, color, tuple(c * 0.9 if i < 3 else c for i, c in enumerate(color)))
    big = nb.c(1.0)
    puff, line, _ = relief(nb, u, v, big, 0.085, line_w=0.1)
    col = nb.cmul(col, stitch, line * 0.6)
    tc = nb.node("ShaderNodeTexCoord")
    weave = _noise(nb, nt, 260.0, tc.outputs["Object"], detail=3.0, rough=0.6)
    n1 = _bump(nb, nt, nb.sqrt(puff), 0.4, 0.004)
    n2 = _bump(nb, nt, weave, 0.07, 0.0006, normal=n1)
    _principled(nt, out, col, n2, 0.85, sheen=0.35, sheen_rough=0.5, spec=0.2)
    return m


def paint(K, color):
    """Satin enamel over pine: grain along the board's UV (u = along, v = across, metres), faint tone drift."""
    m, nt, out = K.new_mat("paint")
    nb = NB(nt)
    u, v = nb.uv("UVMap")
    vec = nb.vec(u * 5.0, v * 120.0, 0.0)
    grain = _noise(nb, nt, 1.0, vec, detail=4.0, rough=0.6)
    tone = _tone(nb, nt, 5.0)
    col = nb.cmix(tone, color, tuple(c * 0.93 if i < 3 else c for i, c in enumerate(color)))
    col = nb.cmul(col, (0.82, 0.8, 0.84, 1.0), nb.sat((grain - 0.45) * 3.0) * 0.4)
    n1 = _bump(nb, nt, grain, 0.08, 0.0008)
    b = principled(nt, out, loc=(1700, 0), **{"Roughness": 0.42, "Coat Weight": 0.15, "Coat Roughness": 0.25,
                                              "Specular IOR Level": 0.5})
    nt.links.new(col, b.inputs["Base Color"])
    nt.links.new(n1, b.inputs["Normal"])
    return m


def accent(K, color):
    return K.simple_mat("finial", color, rough=0.32, coat=0.4)


# ================================================================================================== the rug
def rug_wool(K, spec):
    """The rug's top: a thick-wool pile with the Memphis print. spec: nx, ny, sx, sy, ox, oy (the print's lattice
    corner), tables, colors, ground, border (dict of RGBA: edge, band, line, saw, saw2, inner), width, height (the rug's
    size), plus the border's bands in metres (bands)."""
    m, nt, out = K.new_mat("wool")
    nb = NB(nt)
    u, v = nb.uv("UVMap")
    W, H = spec["width"], spec["height"]
    dxe, dye = W / 2 - nb.abs(u - W / 2), H / 2 - nb.abs(v - H / 2)
    e = nb.min(dxe, dye)
    along = u + (v - u) * nb.lt(dxe, dye)
    mask, c1, _, dm = memphis(nb, u - spec["ox"], v - spec["oy"], spec["nx"], spec["ny"], spec["sx"], spec["sy"],
                              spec["tables"])
    cols = spec["colors"]
    pal = nb.table([(i / len(cols), c) for i, c in enumerate(cols)], (c1 + 0.5) / len(cols))
    b = spec["bands"]                                           # edge line, light band, ink line, saw band, inner line
    field = mask * nb.sat((e - b["field"]) / 0.004)
    tone = _tone(nb, nt, 5.0)
    ground = nb.cmix(tone, spec["ground"], tuple(c * 0.9 if i < 3 else c for i, c in enumerate(spec["ground"])))
    zone = nb.sat((e - b["field"]) / 0.004)
    ground = confetti(nb, nt, ground, u, v, dm, zone, spec["blobs"], spec["specks"], blob_mix=0.22, blob_scale=3.2,
                      speck_scale=17.0)
    col = nb.cmix(field, ground, pal.outputs["Color"])
    bc = spec["border"]
    # light band, ink line, the saw band (triangles pointing inward), inner line, then the outer edge line
    b0, b1 = b["band"]
    col = nb.cmix(nb.edge(nb.abs(e - (b0 + b1) / 2) - (b1 - b0) / 2, AA), col, bc["band"])
    s0, s1 = b["saw"]
    t = (e - s0) / (s1 - s0)
    tri = 1.0 - nb.abs(nb.fract(along / b["saw_pitch"]) * 2.0 - 1.0)
    saw_in = nb.edge(nb.abs(e - (s0 + s1) / 2) - (s1 - s0) / 2, AA)
    tri_mask = nb.edge((t - tri) * (s1 - s0) * 0.8, AA)
    col = nb.cmix(saw_in, col, bc["saw2"])
    col = nb.cmix(saw_in * tri_mask, col, bc["saw"])
    col = nb.cmix(nb.edge(nb.abs(e - b["line"][0]) - b["line"][1], AA), col, bc["line"])
    col = nb.cmix(nb.edge(nb.abs(e - b["inner"][0]) - b["inner"][1], AA), col, bc["inner"])
    col = nb.cmix(nb.edge(e - b["edge"], AA), col, bc["edge"])
    # the pile: layered noise (fine fibres, coarser tufts), a darker tone down in the pile, a slow dye drift
    tc = nb.node("ShaderNodeTexCoord")
    fine = _noise(nb, nt, spec.get("tuft", 190.0), tc.outputs["Object"], detail=4.0, rough=0.68)
    tufts = _noise(nb, nt, 62.0, tc.outputs["Object"], detail=2.0, rough=0.5)
    pile = fine * 0.65 + tufts * 0.35
    drift = _noise(nb, nt, 22.0, tc.outputs["Object"], detail=2.0, rough=0.5)
    grey = nb.cmix(nb.sat((drift - 0.35) * 3.0), (0.9, 0.9, 0.92, 1.0), (1.08, 1.08, 1.05, 1.0))
    col = nb.cmul(col, grey, 1.0)
    col = nb.cmul(col, (0.55, 0.55, 0.62, 1.0), nb.sat((0.62 - pile) * 3.0) * 0.6)
    n1 = _bump(nb, nt, pile, 0.9, 0.0016)
    b_ = principled(nt, out, loc=(1700, 0), **{"Roughness": 0.95, "Sheen Weight": 0.8, "Sheen Roughness": 0.35,
                                               "Specular IOR Level": 0.12})
    b_.inputs["Sheen Tint"].default_value = (0.9, 0.92, 1.0, 1.0)
    nt.links.new(col, b_.inputs["Base Color"])
    nt.links.new(n1, b_.inputs["Normal"])
    return m


def rug_edge(K, color):
    """The rug's edge and underside: tape / felt, a fine weave."""
    m, nt, out = K.new_mat("edge")
    nb = NB(nt)
    tc = nb.node("ShaderNodeTexCoord")
    weave = _noise(nb, nt, 280.0, tc.outputs["Object"], detail=3.0, rough=0.6)
    n1 = _bump(nb, nt, weave, 0.25, 0.0006)
    _principled(nt, out, color, n1, 0.9, sheen=0.4, sheen_rough=0.5, spec=0.15)
    return m


def yarn(K, color):
    """Fringe yarn: a plied twist (a helical ridge from the angle of the object-space normal round the strand, which runs
    along x, plus the distance along it, UV u in metres), fine fibres and a soft sheen. No UV seam round the strand."""
    m, nt, out = K.new_mat("yarn")
    nb = NB(nt)
    u, _ = nb.uv("UVMap")
    tc = nb.node("ShaderNodeTexCoord")
    sp = nb.node("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Normal"], sp.inputs[0])
    ang = nb.op("ARCTAN2", nb.wrap(sp.outputs["Z"]), nb.wrap(sp.outputs["Y"]))
    ply = 0.5 + 0.5 * nb.sin(ang + u * (math.tau / 0.0075))                        # one ridge, a turn every 7.5 mm
    fib = _noise(nb, nt, 900.0, tc.outputs["Object"], detail=2.0, rough=0.6)
    col = nb.cmul(color, (0.78, 0.76, 0.8, 1.0), (1.0 - ply) * 0.6)
    col = nb.cmul(col, (0.9, 0.9, 0.92, 1.0), nb.sat((fib - 0.4) * 3.0) * 0.5)
    n1 = _bump(nb, nt, ply + fib * 0.3, 0.6, 0.0007)
    _principled(nt, out, col, n1, 0.85, sheen=0.7, sheen_rough=0.35, spec=0.2)
    return m


def padded(K, color, pitch=0.12):
    """Tufted velour (the padded headboard): diamond puffs, stitch lines, a button at every node. UV = (x, z) metres."""
    m, nt, out = K.new_mat("padded")
    nb = NB(nt)
    u, v = nb.uv("UVMap")
    puff, line, _ = relief(nb, u, v, nb.c(1.0), pitch, line_w=0.1)
    sa = nb.sin(nb.fract((u + v) * (0.7071067811865476 / pitch)) * math.pi)
    sb = nb.sin(nb.fract((u - v) * (0.7071067811865476 / pitch)) * math.pi)
    button = nb.sat(1.0 - nb.max(sa, sb) / 0.2)
    tone = _tone(nb, nt, 4.0)
    col = nb.cmix(tone, color, tuple(c * 0.88 if i < 3 else c for i, c in enumerate(color)))
    col = nb.cmul(col, (0.5, 0.45, 0.55, 1.0), nb.sat(line * 0.5 + button * 0.8))
    tc = nb.node("ShaderNodeTexCoord")
    nap = _noise(nb, nt, 500.0, tc.outputs["Object"], detail=2.0, rough=0.6)
    n1 = _bump(nb, nt, nb.sqrt(puff) - button * 0.8, 0.8, 0.012)
    n2 = _bump(nb, nt, nap, 0.1, 0.0005, normal=n1)
    _principled(nt, out, col, n2, 0.7, sheen=0.9, sheen_rough=0.3, spec=0.25, sheen_tint=(1.0, 0.95, 1.0, 1.0))
    return m


def tube(K, color):
    """Powder-coated steel tube: a smooth satin coat in the palette colour."""
    m, nt, out = K.new_mat("tube")
    nb = NB(nt)
    tone = _tone(nb, nt, 6.0)
    col = nb.cmix(tone, color, tuple(c * 0.94 if i < 3 else c for i, c in enumerate(color)))
    b = principled(nt, out, loc=(1700, 0), **{"Roughness": 0.3, "Coat Weight": 0.4, "Coat Roughness": 0.12,
                                              "Specular IOR Level": 0.6})
    nt.links.new(col, b.inputs["Base Color"])
    return m
