"""Materials of the giant carrot (Blender side), node trees built with the night kit's node builder (`sets/nightkit.NB`),
coloured from the roles of giant_carrot_layout.ROLES (linear RGBA).

  carrot   the body: the shading point's distance from the axis over the modelled profile's radius at its height (`rel`,
           object coordinates, so it follows the shape keys; the profile is a ramp of giant_carrot_layout.profile_ramp)
           picks the colour: the skin where rel is about 1 (and all of the crown), the cut flesh inside it and the paler
           core with its ring in the middle: a bite's face shows them, wet (a clear coat), the skin satin with faint streaks
           along it; the crown green-tinged. The chunks' skin too (their small object coordinates read as skin).
  cut      a chunk's broken flesh: flesh with core-coloured patches, wet.
  stem     satin green with faint streaks along it.
  leaf     green with a soft mottle and a little light through it.
`flesh_colour(nb, rel, col)` is the cut flesh's ramp for any part that knows its own `rel` (the plate's carrot coins)."""
from ..sets import nightkit as NK
from . import giant_carrot_layout as LAY

CORE, SKIN_IN = LAY.CORE, LAY.SKIN_IN      # the core's edge and where the skin starts, shares of the radius


def _ramp(nb, fac, stops):
    """A colour ramp (linear) with the stops [(position, rgba), ...] (sorted, distinct): its colour socket."""
    n = nb.node("ShaderNodeValToRGB", {0: fac})
    els = n.color_ramp.elements
    els[0].position, els[0].color = stops[0][0], tuple(stops[0][1])
    els[1].position, els[1].color = stops[-1][0], tuple(stops[-1][1])
    for p, c in stops[1:-1]:
        els.new(p).color = tuple(c)
    return n.outputs[0]


def _curve(nb, fac, stops):
    """A float ramp [(position, value)]: the red of a grey colour ramp."""
    return nb.sep(_ramp(nb, fac, [(p, (v, v, v, 1.0)) for p, v in stops]))[0]


def _scale(c, k):
    return tuple(min(1.0, v * k) for v in c[:3]) + (1.0,)


def _mix(a, b, t):
    return tuple(x * (1.0 - t) + y * t for x, y in zip(a[:3], b[:3])) + (1.0,)


def flesh_colour(nb, rel, col):
    """The colour of carrot flesh at `rel` (a socket: 0 on the axis .. 1 at the skin): the paler core with a light ring at
    its edge, the flesh, the skin's edge."""
    core, flesh, skin = col["core"], col["flesh"], col["skin"]
    return _ramp(nb, rel, [(0.0, _mix(core, (1.0, 1.0, 1.0), 0.12)), (0.3, core), (CORE - 0.035, core),
                           (CORE, _mix(core, (1.0, 1.0, 1.0), 0.3)), (CORE + 0.03, flesh), (SKIN_IN - 0.02, flesh),
                           (SKIN_IN + 0.01, skin), (1.0, skin)])


def _rel(nb):
    """(rel, z): the shading point's distance from the axis over the profile's radius at its height (anything above the
    shoulder counts as skin), and its height."""
    x, y, z = nb.sep(nb.texcoord("Object"))
    rho = nb.vlen(nb.comb(x, y, 0.0))
    prof = _curve(nb, nb.math("DIVIDE", z, LAY.L), [(p, r / LAY.R_S) for p, r in LAY.profile_ramp()])
    rel = nb.math("DIVIDE", rho, nb.math("MAXIMUM", nb.math("MULTIPLY", prof, LAY.R_S), 0.0004))
    return nb.math("ADD", rel, nb.math("MULTIPLY", nb.smooth(LAY.L - 0.003, LAY.L, z), 2.0)), z


def _streaks(nb, scale, lo, hi):
    """A noise stretched along the object's z: lo .. hi (faint streaks along a root or a stem)."""
    p = nb.node("ShaderNodeMapping", {"Vector": nb.texcoord("Object"), "Scale": scale}).outputs[0]
    return nb.remap(nb.noise(p, scale=1.0, detail=3.0, rough=0.55), 0.3, 0.7, lo, hi)


def _body(name, col):
    m, nb = NK.new_material(f"{name}_carrot")
    rel, z = _rel(nb)
    colour = flesh_colour(nb, rel, col)
    skin = nb.smooth(SKIN_IN - 0.02, SKIN_IN + 0.01, rel)
    band = _streaks(nb, (40.0, 40.0, 320.0), 0.86, 1.06)
    colour = nb.mixc(skin, colour, nb.mixc(1.0, colour, nb.comb(band, band, band), blend="MULTIPLY"))
    tinge = nb.math("MULTIPLY", nb.smooth(LAY.L - 0.012, LAY.L + 0.016, z),
                    nb.remap(nb.noise(nb.texcoord("Object"), scale=60.0, detail=2.0), 0.3, 0.7, 0.7, 1.0))
    colour = nb.mixc(tinge, colour, col["crown"])
    b = nb.principled(colour, rough=_curve(nb, rel, [(0.0, 0.34), (SKIN_IN - 0.02, 0.36), (SKIN_IN + 0.01, 0.52),
                                                     (1.0, 0.52)]),
                      coat=_curve(nb, rel, [(0.0, 0.3), (SKIN_IN - 0.02, 0.3), (SKIN_IN + 0.01, 0.0), (1.0, 0.0)]),
                      coat_rough=0.2, spec=0.5)
    b.inputs["Subsurface Weight"].default_value = 0.12
    b.inputs["Subsurface Radius"].default_value = (1.0, 0.4, 0.12)
    b.inputs["Subsurface Scale"].default_value = 0.004
    nb.output(b)
    m.diffuse_color = col["skin"]
    return m


def _cut(name, col):
    m, nb = NK.new_material(f"{name}_cut")
    patch = nb.smooth(0.42, 0.56, nb.noise(nb.texcoord("Object"), scale=140.0, detail=2.0))
    b = nb.principled(nb.mixc(patch, col["flesh"], col["core"]), rough=0.36, coat=0.3, coat_rough=0.2)
    nb.output(b)
    m.diffuse_color = col["flesh"]
    return m


def _stem(name, col):
    m, nb = NK.new_material(f"{name}_stem")
    k = _streaks(nb, (90.0, 90.0, 6.0), 0.85, 1.1)
    b = nb.principled(nb.mixc(1.0, col["stem"], nb.comb(k, k, k), blend="MULTIPLY"), rough=0.42, spec=0.45)
    b.inputs["Sheen Weight"].default_value = 0.2
    nb.output(b)
    m.diffuse_color = col["stem"]
    return m


def _leaf(name, col):
    m, nb = NK.new_material(f"{name}_leaf")
    mottle = nb.noise(nb.texcoord("Object"), scale=28.0, detail=3.0, rough=0.5)
    colour = nb.mixc(nb.remap(mottle, 0.3, 0.7), _scale(col["leaf"], 0.82), _scale(col["leaf"], 1.18))
    b = nb.principled(colour, rough=0.5, spec=0.45)
    b.inputs["Sheen Weight"].default_value = 0.3
    b.inputs["Sheen Roughness"].default_value = 0.4
    through = nb.node("ShaderNodeBsdfTranslucent", {"Color": colour})
    nb.output(nb.mix_shader(0.15, b, through))
    m.diffuse_color = col["leaf"]
    return m


def materials(name, col):
    """{material: Blender material} for giant_carrot_layout.MATS, from the roles' linear colours `col`."""
    return {"carrot": _body(name, col), "cut": _cut(name, col), "stem": _stem(name, col), "leaf": _leaf(name, col)}
