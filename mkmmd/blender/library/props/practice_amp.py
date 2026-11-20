"""The practice amp (`card = "library:practice_amp"`): a small tolex combo standing on the floor, its front toward -Y.
Objects `<name>_cabinet` (the tolex shell, its hidden baffle dark), `_grille` (the cloth), `_piping` (the welt round it),
`_panel`, `_print` (the panel's border and knob scales and the knobs' pointer lines: graphic layers, form-exempt),
`_knobs`, `_hardware` (the jack, the lamp's bezel, the switch, the handle's caps and screws, the feet, the corner
protectors), `_jewel` (the lamp, flat shaded), `_handle` (the strap) and the hidden collider box `<name>_col`, under the
root.

    [[prop]]
    name = "amp"
    card = "library:practice_amp"
    at = [0.55, 0.9, 0.0]
    yaw = -20
    slots = { tolex = "rose", grille = "text", knobs = 4, corners = true }

The tolex is a pebbled grain (a Voronoi bump in the object's space), the cloth a basket weave whose threads fade to their
mean colour beyond about two metres (a draft's few samples would shimmer on them), the panel a satin plate, the knobs
glossy plastic, the metal chrome, the jewel glowing (`glow`). Numbers, roles, options and the card:
practice_amp_layout.py."""
import math

import bpy

from .. import shell as SH
from ..mesh import box as _collider_box
from ..sets import nightkit as NK
from . import practice_amp_layout as LAY
from . import register
from . import showkit as SK


def _rgba(c, a=1.0):
    return (float(c[0]), float(c[1]), float(c[2]), a)


def _plain(name, rgb, rough, spec=0.5, coat=0.0, metal=0.0, sheen=0.0, emission=0.0):
    m, nb = NK.new_material(name)
    b = nb.principled(_rgba(rgb), rough=rough, metal=metal, spec=spec, coat=coat, coat_rough=0.2,
                      emission=_rgba(rgb) if emission > 0 else None, strength=emission)
    nb.set(b.inputs["Sheen Weight"], sheen)
    nb.output(b)
    m.diffuse_color = _rgba(rgb)
    return m


def _tolex(name, rgb):
    """Pebbled vinyl: domes of a Voronoi cell grid (about 2.6 mm) in the bump, their valleys a little darker, a slow
    mottle over the whole cabinet, a satin sheen."""
    m, nb = NK.new_material(name)
    P = nb.texcoord("Object")
    cell = nb.voronoi(P, scale=380.0, which="Distance")
    dome = nb.math("SUBTRACT", 1.0, nb.smooth(0.0, 0.62, cell))
    mott = nb.noise(P, scale=18.0, detail=3.0, rough=0.55)
    k = nb.math("MULTIPLY", nb.math("ADD", 0.90, nb.math("MULTIPLY", dome, 0.10)),
                nb.math("ADD", 0.97, nb.math("MULTIPLY", mott, 0.06)))
    col = nb.vmath("SCALE", _rgba(rgb), scale=k)
    bump = nb.node("ShaderNodeBump", {"Strength": 0.45, "Distance": 0.0005, "Height": dome})
    b = nb.principled(col, rough=0.52, spec=0.45, coat=0.12, coat_rough=0.35, normal=bump.outputs["Normal"])
    nb.set(b.inputs["Sheen Weight"], 0.15)
    nb.output(b)
    m.diffuse_color = _rgba(rgb)
    return m


def _grille(name, rgb, light):
    """Grille cloth: a basket weave of threads 2.6 mm apart (pairs of threads crossing in cells two threads wide), dark
    gaps between them; beyond about two metres the weave fades to its mean colour."""
    m, nb = NK.new_material(name)
    x, _y, z = nb.sep(nb.texcoord("Object"))
    pitch = 0.0026
    w = 2.0 * math.pi / pitch
    a = nb.math("MULTIPLY_ADD", nb.math("SINE", nb.math("MULTIPLY", x, w)), 0.5, 0.5)
    b = nb.math("MULTIPLY_ADD", nb.math("SINE", nb.math("MULTIPLY", z, w)), 0.5, 0.5)
    cell = 1.0 / (2.0 * pitch)
    ck = nb.math("FLOORED_MODULO", nb.math("ADD", nb.math("FLOOR", nb.math("MULTIPLY", x, cell)),
                                           nb.math("FLOOR", nb.math("MULTIPLY", z, cell))), 2.0)
    thread = nb.mixf(ck, a, b)
    fade = nb.math("SUBTRACT", 1.0, nb.smooth(1.4, 2.6, nb.cam_distance()))
    t = nb.math("MULTIPLY_ADD", nb.math("SUBTRACT", thread, 0.5), fade, 0.5)
    col = nb.mixc(nb.math("MULTIPLY", t, t), _rgba([0.45 * c for c in rgb]), _rgba(light))
    bump = nb.node("ShaderNodeBump", {"Strength": nb.math("MULTIPLY", fade, 0.5), "Distance": 0.0004, "Height": thread})
    bsdf = nb.principled(col, rough=0.86, spec=0.25, normal=bump.outputs["Normal"])
    nb.set(bsdf.inputs["Sheen Weight"], 0.35)
    nb.set(bsdf.inputs["Sheen Roughness"], 0.5)
    nb.output(bsdf)
    m.diffuse_color = _rgba(rgb)
    return m


def _leather(name, rgb):
    m, nb = NK.new_material(name)
    P = nb.texcoord("Object")
    grain = nb.noise(P, scale=420.0, detail=3.0, rough=0.6)
    col = nb.vmath("SCALE", _rgba(rgb), scale=nb.math("MULTIPLY_ADD", grain, 0.08, 0.96))
    bump = nb.node("ShaderNodeBump", {"Strength": 0.25, "Distance": 0.0004, "Height": grain})
    nb.output(nb.principled(col, rough=0.48, spec=0.45, coat=0.15, coat_rough=0.3, normal=bump.outputs["Normal"]))
    m.diffuse_color = _rgba(rgb)
    return m


def _jewel(name, rgb, glow):
    """A coloured glass jewel: a deep, glossy body and, at `glow`, its own light of its colour."""
    m, nb = NK.new_material(name)
    nb.output(nb.principled(_rgba([0.4 * c for c in rgb]), rough=0.08, spec=0.6, coat=1.0, coat_rough=0.05,
                            emission=_rgba([0.6 * c for c in rgb]) if glow > 0 else None, strength=glow))
    m.diffuse_color = _rgba(rgb)
    return m


def _materials(name, roles, slots, glow):
    c = {role: SK.rgb(spec, slots, f"practice_amp: {role}") for role, spec in roles.items()}
    light = [0.92 * g + 0.08 * p for g, p in zip(c["grille"], SK.rgb("paper", slots))]
    return {"tolex": _tolex(f"{name}_tolex", c["tolex"]),
            "grille": _grille(f"{name}_grille", c["grille"], light),
            "piping": _plain(f"{name}_piping", c["piping"], 0.42, coat=0.25),
            "panel": _plain(f"{name}_panel", c["panel"], 0.36, coat=0.2),
            "print": _plain(f"{name}_print", c["print"], 0.5, spec=0.35),
            "knobs": _plain(f"{name}_knobs", c["knobs"], 0.26, coat=0.45),
            "handle": _leather(f"{name}_handle", c["handle"]),
            "metal": _plain(f"{name}_metal", c["metal"], 0.24, metal=1.0),
            "lamp": _jewel(f"{name}_lamp", c["lamp"], glow),
            "rubber": _plain(f"{name}_rubber", c["rubber"], 0.8, spec=0.3),
            "corners": _plain(f"{name}_corners", c["corners"], 0.4, coat=0.15)}


@register("practice_amp")
def practice_amp(name, coll, root, slots=None):
    slots = dict(slots or {})
    O = LAY.options(slots)
    roles = LAY.roles(slots)
    mats = _materials(name, roles, slots, O["glow"])
    mat_of = lambda i: mats[LAY.MATS[i]]                                                       # noqa: E731
    for part, mesh in LAY.part_meshes(O).items():
        oname = f"{name}_{part}"
        if part == "jewel":                                      # facets: flat shaded
            SH.purge(oname)
            me, _ = SH.mesh_data(oname, mesh, mat_of, smooth=False, normals=False)
            o = bpy.data.objects.new(oname, me)
            coll.objects.link(o)
            o.parent = root
        else:
            o = SH.mesh_object(oname, mesh, coll, root, mat_of)
        if part in LAY.GRAPHIC:
            SH.exempt(o, "printed on the panel: a graphic layer")
    centre, size = LAY.collider_box()
    SH.purge(f"{name}_col")
    _collider_box(f"{name}_col", centre, size, coll, root, collider=True)
    return LAY.card(name, roles)
