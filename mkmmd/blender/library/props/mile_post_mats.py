"""Materials of the mile post (Blender side), node trees built with the night kit's node builder (`sets/nightkit.NB`).
Every colour is an RGB node fed by a colour role (mile_post_layout.ROLES, resolved from the palette on every build) and
every shape of the weathering is a mask of the object's own coordinates (Texture Coordinate Object / Normal, no object
set), a noise, or the post's `mk_check` attribute: the node trees, their node names and the material names are the same
whatever the palette, so a palette breath only drives colours.

    railroad    `<name>_post`      white paint over timber: a faint grain telegraphing through, rain streaks and stains
                                   running down from a greyed top, the paint gone in blotches (more toward the ground)
                                   where the grey weathered wood shows, dark season checks (`mk_check`), dirt splashed
                                   up the foot and the tarred foot with a ragged edge
                `<name>_plate`     painted cast iron: a mottled satin black with a cast texture and specks of rust
                `<name>_numerals`  white paint on the raised numerals, worn on their tops
    interstate  `<name>_post`      galvanised steel: spangle, a soft sheen, the punched holes down the web
                `<name>_back`      the panel's mill-finish aluminium back and edges, the bolt heads
                `<name>_face`      the green retroreflective sheeting; `<name>_border` and `<name>_numerals` the white

Nothing renders black: the paint and the iron glow FLOOR of their colour, the sheeting SHEET (retroreflective: it
returns the light of whatever looks at it)."""
from ..sets import nightkit as NK
from . import mile_post_layout as LAY

FLOOR = 0.06                    # share of its colour a painted surface always glows
SHEET = 0.22                    # the retroreflective sheeting's own glow


def _rgb(nb, colour):
    """An RGB node holding a role's colour: the value a palette breath drives."""
    n = nb.node("ShaderNodeRGB")
    n.outputs[0].default_value = tuple(colour[:3]) + (1.0,)
    return n.outputs[0]


def _scale(nb, c, k):
    """A colour socket times k (a darker or lighter shade of a role's colour)."""
    return nb.mixc(1.0, c, (k, k, k, 1.0), blend="MULTIPLY", clamp=False)


def _glow(b, nb, colour, k):
    nb.set(b.inputs["Emission Color"], colour)
    b.inputs["Emission Strength"].default_value = k


def _post_rr(name, col):
    m, nb = NK.new_material(f"{name}_post")
    P = nb.texcoord("Object")
    _x, _y, z = nb.sep(P)
    _nx, _ny, nz = nb.sep(nb.texcoord("Normal"))
    paint, wood = _rgb(nb, col["post"]), _rgb(nb, col["wood"])
    grime, foot = _rgb(nb, col["grime"]), _rgb(nb, col["foot"])
    grain = nb.noise(nb.vmath("MULTIPLY", P, (70.0, 70.0, 2.2)), scale=1.0, detail=6.0, rough=0.62)
    blot = nb.noise(P, scale=7.5, detail=5.0, rough=0.58)
    up = nb.smooth(0.3, 0.7, nz)                                             # the hipped top: end grain, rain, sun
    low = nb.math("SUBTRACT", 1.0, nb.smooth(0.0, 1.1, z))                  # 1 at the ground .. 0 up the post
    edge = nb.math("SUBTRACT", 0.655, nb.math("ADD", nb.math("MULTIPLY", low, 0.07), nb.math("MULTIPLY", up, 0.09)))
    loss = nb.smooth(0.0, 0.025, nb.math("SUBTRACT", nb.math("ADD", blot, nb.math("MULTIPLY", grain, 0.05)), edge))
    painted = nb.mixc(nb.math("MULTIPLY", grain, 0.30), paint, _scale(nb, paint, 0.86))
    streak = nb.noise(nb.vmath("MULTIPLY", P, (16.0, 16.0, 0.8)), scale=1.0, detail=3.0, rough=0.55)
    weather = nb.math("ADD", nb.remap(streak, 0.5, 0.8, 0.0, 0.22), nb.math("MULTIPLY", up, 0.35))
    painted = nb.mixc(weather, painted, wood)                               # rain streaks down the paint, a grey top
    runs = nb.math("MULTIPLY", nb.remap(streak, 0.55, 0.85, 0.0, 0.45), nb.smooth(0.55, LAY.POST_H, z))
    painted = nb.mixc(runs, painted, grime)                                 # stains run down from the top
    bare = nb.mixc(grain, _scale(nb, wood, 0.78), _scale(nb, wood, 1.12))
    base = nb.mixc(loss, painted, bare)
    splash = nb.noise(P, scale=23.0, detail=4.0, rough=0.6)
    dirt = nb.math("MULTIPLY", nb.math("SUBTRACT", 1.0, nb.smooth(0.04, LAY.GRIME_Z,
                                                                 nb.math("ADD", z, nb.math("MULTIPLY", splash, 0.12)))),
                   nb.remap(splash, 0.3, 0.75, 0.25, 0.75))
    base = nb.mixc(dirt, base, grime)
    rag = nb.noise(P, scale=40.0, detail=3.0, rough=0.5)
    tar = nb.math("SUBTRACT", 1.0, nb.smooth(LAY.FOOT_Z - 0.004, LAY.FOOT_Z + 0.004,
                                             nb.math("ADD", z, nb.math("MULTIPLY", nb.math("SUBTRACT", rag, 0.5), 0.03))))
    base = nb.mixc(tar, base, _scale(nb, foot, 1.0))
    check = nb.smooth(0.15, 0.9, nb.attr("mk_check", "Fac"))
    base = nb.mixc(check, base, _scale(nb, wood, 0.22))
    rough = nb.mixf(loss, 0.52, 0.86)
    rough = nb.mixf(tar, rough, 0.38)
    h = nb.math("ADD", nb.math("MULTIPLY", grain, 0.5), nb.math("MULTIPLY", nb.math("SUBTRACT", 1.0, loss), 0.6))
    bump = nb.node("ShaderNodeBump", {"Strength": 0.22, "Distance": 0.0006, "Height": h}).outputs["Normal"]
    b = nb.principled(base, rough=rough, spec=0.45, normal=bump)
    _glow(b, nb, base, FLOOR)
    nb.output(b)
    return m


def _plate_rr(name, col):
    m, nb = NK.new_material(f"{name}_plate")
    P = nb.texcoord("Object")
    iron, rust = _rgb(nb, col["plate"]), _rgb(nb, col["grime"])
    cast = nb.noise(P, scale=420.0, detail=3.0, rough=0.6)
    mottle = nb.noise(P, scale=38.0, detail=4.0, rough=0.55)
    base = nb.mixc(nb.math("MULTIPLY", mottle, 0.3), iron, _scale(nb, iron, 1.3))
    speck = nb.smooth(0.68, 0.76, nb.noise(P, scale=95.0, detail=4.0, rough=0.6))
    base = nb.mixc(nb.math("MULTIPLY", speck, 0.5), base, rust)
    bump = nb.node("ShaderNodeBump", {"Strength": 0.12, "Distance": 0.0004, "Height": cast}).outputs["Normal"]
    b = nb.principled(base, rough=nb.remap(mottle, 0.3, 0.7, 0.55, 0.72), spec=0.3, normal=bump)
    _glow(b, nb, base, FLOOR)
    nb.output(b)
    return m


def _numerals_rr(name, col):
    m, nb = NK.new_material(f"{name}_numerals")
    P = nb.texcoord("Object")
    _nx, ny, _nz = nb.sep(nb.texcoord("Normal"))
    paint, iron = _rgb(nb, col["numerals"]), _rgb(nb, col["plate"])
    top = nb.smooth(0.9, 0.99, nb.math("ABSOLUTE", ny))                     # the strokes' tops face the front (-Y)
    rub = nb.smooth(0.70, 0.78, nb.noise(P, scale=140.0, detail=3.0, rough=0.6))
    base = nb.mixc(nb.math("MULTIPLY", nb.math("MULTIPLY", top, rub), 0.85), paint, iron)
    b = nb.principled(base, rough=0.46, spec=0.5)
    _glow(b, nb, base, FLOOR)
    nb.output(b)
    return m


def _galvanised(name, col):
    m, nb = NK.new_material(f"{name}_post")
    P = nb.texcoord("Object")
    x, y, z = nb.sep(P)
    steel = _rgb(nb, col["post"])
    cells = nb.voronoi(nb.vmath("MULTIPLY", P, (1.0, 1.0, 0.55)), scale=70.0, which="Color")
    spangle = nb.node("ShaderNodeSeparateColor", {0: cells}).outputs[0]       # a random value per spangle cell
    base = nb.mixc(spangle, _scale(nb, steel, 0.88), _scale(nb, steel, 1.1))
    dz = nb.math("MULTIPLY", nb.math("SUBTRACT", nb.math("FRACT", nb.math("DIVIDE", z, LAY.CH_PITCH)), 0.5), LAY.CH_PITCH)
    d = nb.vlen(nb.comb(x, dz, 0.0))
    web = nb.math("SUBTRACT", 1.0, nb.smooth(0.9 * LAY.CH_T, 1.3 * LAY.CH_T,
                                             nb.math("ABSOLUTE", nb.math("ADD", y, 0.5 * (LAY.CH_A - LAY.CH_T)))))
    hole = nb.math("MULTIPLY", web, nb.math("SUBTRACT", 1.0, nb.smooth(0.5 * LAY.CH_HOLE - 0.0004,
                                                                         0.5 * LAY.CH_HOLE + 0.0004, d)))
    base = nb.mixc(hole, base, _scale(nb, steel, 0.03))
    rough = nb.mixf(hole, nb.remap(spangle, 0.0, 1.0, 0.26, 0.42), 1.0)
    b = nb.principled(base, rough=rough, metal=nb.mixf(hole, 0.85, 0.0), spec=0.5)
    _glow(b, nb, base, FLOOR)
    nb.output(b)
    return m


def _aluminium(name, col):
    m, nb = NK.new_material(f"{name}_back")
    P = nb.texcoord("Object")
    metal = _rgb(nb, col["post"])
    brushed = nb.noise(nb.vmath("MULTIPLY", P, (900.0, 30.0, 30.0)), scale=1.0, detail=2.0, rough=0.5)
    b = nb.principled(_scale(nb, metal, 1.12), rough=nb.remap(brushed, 0.3, 0.7, 0.3, 0.42), metal=0.9, spec=0.5)
    _glow(b, nb, metal, FLOOR)
    nb.output(b)
    return m


def _sheeting(name, key, colour):
    """Retroreflective sign sheeting: a satin face with the faint sparkle of its prisms, glowing SHEET of its colour."""
    m, nb = NK.new_material(f"{name}_{key}")
    P = nb.texcoord("Object")
    c = _rgb(nb, colour)
    fine = nb.noise(P, scale=900.0, detail=2.0, rough=0.5)
    b = nb.principled(c, rough=nb.remap(fine, 0.3, 0.7, 0.36, 0.48), spec=0.4, coat=0.12, coat_rough=0.2)
    _glow(b, nb, c, SHEET)
    nb.output(b)
    return m


def materials(name, style, col):
    """{material key of mile_post_layout.MATS[style]: Blender material} from the roles' linear colours `col`."""
    if style == "railroad":
        return {"post": _post_rr(name, col), "plate": _plate_rr(name, col), "numerals": _numerals_rr(name, col)}
    return {"post": _galvanised(name, col), "back": _aluminium(name, col), "face": _sheeting(name, "face", col["plate"]),
            "border": _sheeting(name, "border", col["border"]), "numerals": _sheeting(name, "numerals", col["numerals"])}
