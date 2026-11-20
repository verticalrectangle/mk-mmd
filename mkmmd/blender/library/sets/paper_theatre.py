"""paper_theatre: a toy theatre of construction paper and card, life-size round a figure 1.5 m tall (docs/design.md: Sets).

    [[set]]
    name = "theatre"
    kind = "paper_theatre"
    at = [60.0, 0.0, 0.0]
    arrive = 13.64              # the proscenium, the curtains and the hanging pieces drop in (absent: there all along)
    step = 12                   # the set's own motion held 12 times a second, as stop-motion

The layout, the motion and the card are mkmmd.core.paper_theatre (numpy, tested), the meshes paper_theatre_parts. Frame:
she stands at the origin facing -Y on the stage floor, whose top is z = 0.

Objects (`<name>_<part>`): `floor` (corrugated card under its paper, the flutes on its cut front edge, pencil lines),
`base` (the stage's front under it), `sky` (the backdrop bent round the stage, its stands and pencil lines), `sun` (the
face, pinned by a brass split pin), `sun_rays` (turning on the pin), `hills_back`, `hills_front`, `grass`, `grass_back`,
`flies` (the battens and their tape); `frame` (an empty: the carrier that drops in), under it `proscenium` (with the
valance) and `curtain_L` / `curtain_R` (their origin on their top's middle); per hanging piece (`cloud1`..`cloud3`,
`star1`..`star5`, `moon`) the pivot `<piece>_hang` (an empty on its batten), the piece `<piece>` and `<piece>_thread`.
Motion is keyed on them (LINEAR every frame, or CONSTANT steps with `step`); the proscenium's and the curtains' flex are
shape keys `flex`. Materials `<name>_<role>`, one per colour role of mkmmd.core.paper_theatre.ROLES."""
import bpy
import numpy as np

from ....core import paper_theatre as PT
from ....core import shotstyle as SS
from ... import keys as K
from .. import shell as SH
from ..props import parent_cutout_mats as MT
from . import nightkit as NK
from . import paper_theatre_parts as P
from . import register

FLAT = "flat paper and card: a toy theatre"


def _lin(rgb):
    return tuple(SS.srgb_to_linear(float(c)) for c in rgb)


def _materials(name, colours):
    """mat_of(index) for the shell toolkit: the role's material, made on first use."""
    made = {}

    def mat_of(i):
        role = PT.MATS[i]
        if role not in made:
            c = _lin(colours[role])
            nm = f"{name}_{role}"
            if role == "brad":
                m = MT.brass(nm, c)
            elif role in ("thread", "pencil"):
                m, nb = NK.new_material(nm)
                nb.output(nb.principled((*c, 1.0), rough=0.75 if role == "thread" else 0.45,
                                        spec=0.3 if role == "thread" else 0.45))
                m.diffuse_color = (*c, 1.0)
            elif role == "tape":
                m = MT.paper(nm, c, tuple(min(1.0, 1.05 * v) for v in c), rough=0.6, mottle=0.03, fleck=0.02, grain=0.05)
            else:
                m = MT.paper(nm, c, tuple(min(1.0, 1.12 * v + 0.02) for v in c))
            made[role] = m
        return made[role]
    return mat_of


def _empty(name, coll, parent, loc):
    old = bpy.data.objects.get(name)
    if old is not None:
        bpy.data.objects.remove(old)
    o = bpy.data.objects.new(name, None)
    o.empty_display_size = 0.15
    coll.objects.link(o)
    o.parent = parent
    o.location = loc
    o.rotation_mode = "XYZ"
    return o


def _flex(o, offsets):
    """A shape key `flex` on the object: its vertices moved by `offsets` at value 1 (keyed between -1 and 1)."""
    o.shape_key_add(name="Basis", from_mix=False)
    k = o.shape_key_add(name="flex", from_mix=False)
    k.slider_min, k.slider_max = -1.0, 1.0
    co = np.empty(len(o.data.vertices) * 3, np.float32)
    o.data.vertices.foreach_get("co", co)
    k.data.foreach_set("co", (co.reshape(-1, 3) + np.asarray(offsets, np.float32)).ravel())
    return k


def _key_hidden(o, frames, hidden):
    """hide_render / hide_viewport keyed CONSTANT where `hidden` changes (and on the first frame)."""
    hidden = np.asarray(hidden, bool)
    at = np.concatenate([[0], np.flatnonzero(hidden[1:] != hidden[:-1]) + 1])
    for attr in ("hide_render", "hide_viewport"):
        setattr(o, attr, bool(hidden[0]))
        K.set_fcurve(o, attr, 0, np.asarray(frames, float)[at], hidden[at].astype(float), interp="CONSTANT")


@register("paper_theatre")
def paper_theatre(name, coll, root, spec, palette):
    """spec: the [[set]] table (mkmmd.core.paper_theatre.DEFAULTS: arrive, stagger, step, step_from, sun_rpm, swing,
    twist, seed, colors, width, height, depth); an unknown key is an error that lists the known ones."""
    opts = PT.options(spec)
    colours = PT.colours(opts, palette)
    lay = PT.layout(opts["width"], opts["height"], opts["depth"], opts["seed"])
    sc = bpy.context.scene
    f0 = float(sc.get("mk_frame0", sc.frame_start))
    fps = sc.render.fps / sc.render.fps_base
    frames, times, interp = PT.key_times(sc.frame_start, sc.frame_end, f0, fps, opts["step"], opts["step_from"])
    for nm in PT.objects(name, lay):                      # a rebuild in the same session
        SH.purge(nm)
    mat_of = _materials(name, colours)
    seed = opts["seed"]
    obs = {}
    for part in lay.statics:
        o = SH.mesh_object(f"{name}_{part.name}", P.part_mesh(part, seed), coll, root, mat_of, loc=part.at)
        o.rotation_mode = "XYZ"
        obs[part.name] = SH.exempt(o, FLAT)
    pro, at = lay.frame
    carrier = _empty(f"{name}_frame", coll, root, at)
    obs["frame"] = carrier
    mesh, flex = P.proscenium_mesh(pro, seed, float(pro.special["flex_height"]))
    o = SH.exempt(SH.mesh_object(f"{name}_proscenium", mesh, coll, carrier, mat_of), FLAT)
    _flex(o, flex)
    obs["proscenium"] = o
    for c in lay.curtains:
        mesh, flex = P.curtain_mesh(c, P.seed_of(seed, c.name))
        o = SH.exempt(SH.mesh_object(f"{name}_{c.name}", mesh, coll, carrier, mat_of, loc=c.hinge), FLAT)
        o.rotation_mode = "XYZ"
        _flex(o, flex)
        obs[c.name] = o
    for h in lay.hangs:
        piv = _empty(f"{name}_{h.name}_hang", coll, root, h.anchor)
        piece = SH.exempt(SH.mesh_object(f"{name}_{h.name}", P.part_mesh(h.part, seed), coll, piv, mat_of,
                                         loc=(0.0, 0.0, -h.length)), FLAT)
        piece.rotation_mode = "XYZ"
        th = SH.mesh_object(f"{name}_{h.name}_thread", P.thread(), coll, piv, mat_of)
        th.scale = (1.0, 1.0, h.length)
        obs.update({f"{h.name}_hang": piv, h.name: piece, f"{h.name}_thread": th})

    channels, hidden = PT.motion(opts, lay, times)
    for ch in channels:
        o = obs[ch.object]
        if ch.path.startswith("shape:"):
            K.set_fcurve(o.data.shape_keys, f'key_blocks["{ch.path[6:]}"].value', 0, frames, ch.values, interp=interp)
        else:
            K.set_fcurve(o, ch.path, ch.index, frames, ch.values, interp=interp)
    for nm, hid in hidden.items():
        _key_hidden(obs[nm], frames, hid)
    return PT.card(name, opts, lay)
