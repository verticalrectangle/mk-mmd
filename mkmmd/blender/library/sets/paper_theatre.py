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
shape keys `flex`. Materials `<name>_<role>`, one per colour role of mkmmd.core.paper_theatre.ROLES.

The pond and its machinery (mkmmd.core.paper_theatre_pond: `pond`, `panorama`, `rise`, `night`, `moon`, `loose`, `sun`,
`day`) add `wave1`..`wave5`, the wings `wing_R1` .. `wing_L2`, the `lane` (an empty) with `pad<i>`, `leaf<i>`,
`reeds<i>` and `lily<i>` under it, `tab`, `crank` (under the frame) with `crank_knob`, the day halves `day_L` / `day_R`,
the spools, `starlight`, `stroke<i>`; what changes shape (a half sliding round the backdrop's bends, a lily opening, a
stroke drawn) is a flipbook: a shape key `f<k>` per held pose, one at 1 at a time."""
import bpy
import numpy as np

from ....core import paper_theatre as PT
from ....core import paper_theatre_pond as PP
from ....core import shell as S
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


def _key_hidden(o, frames, hidden, viewport=True):
    """hide_render (and hide_viewport) keyed CONSTANT where `hidden` changes (and on the first frame). A flipbook keys
    hide_render alone: an object off in the viewport is not evaluated, and the bake checks its shape there."""
    hidden = np.asarray(hidden, bool)
    at = np.concatenate([[0], np.flatnonzero(hidden[1:] != hidden[:-1]) + 1])
    for attr in ("hide_render", "hide_viewport") if viewport else ("hide_render",):
        setattr(o, attr, bool(hidden[0]))
        K.set_fcurve(o, attr, 0, np.asarray(frames, float)[at], hidden[at].astype(float), interp="CONSTANT")


def _flipbook(o, shapes, index, frames, interp):
    """A flipbook on the mesh object `o`, whose mesh is `shapes[index[0]]`: a shape key `f<k>` for every other shape (n,
    3 positions in its own frame), each at 1 on the keys `index` shows it and 0 elsewhere (only where it changes, and
    beside a change when LINEAR, so the blend runs from one key to the next)."""
    basis = int(index[0])
    o.shape_key_add(name="Basis", from_mix=False)
    frames = np.asarray(frames, float)
    for k, V in enumerate(shapes):
        if k == basis:
            continue
        kb = o.shape_key_add(name=f"f{k}", from_mix=False)
        kb.slider_min, kb.slider_max = 0.0, 1.0
        kb.data.foreach_set("co", np.asarray(V, np.float32).ravel())
        v = (np.asarray(index) == k).astype(float)
        ch = np.concatenate([[True], v[1:] != v[:-1]])
        if interp != "CONSTANT":
            ch = ch | np.concatenate([v[:-1] != v[1:], [True]])
        K.set_fcurve(o.data.shape_keys, f'key_blocks["f{k}"].value', 0, frames[ch], v[ch], interp=interp)


@register("paper_theatre")
def paper_theatre(name, coll, root, spec, palette):
    """spec: the [[set]] table (mkmmd.core.paper_theatre.DEFAULTS: arrive, stagger, step, step_from, sun_rpm, swing,
    twist, seed, colors, width, height, depth, and the pond's pond, panorama, rise, night, moon, loose, sun, day); an
    unknown key is an error that lists the known ones."""
    opts = PT.options(spec)
    colours = PT.colours(opts, palette)
    lay, pond = PT.scene(opts)
    sc = bpy.context.scene
    f0 = float(sc.get("mk_frame0", sc.frame_start))
    fps = sc.render.fps / sc.render.fps_base
    frames, times, interp = PT.key_times(sc.frame_start, sc.frame_end, f0, fps, opts["step"], opts["step_from"])
    for nm in (PT.objects(name, lay) if pond is None else PP.objects(name, lay, pond)):     # a rebuild in the session
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

    if pond is None:
        channels, hidden = PT.motion(opts, lay, times)
        flips, shapes = {}, {}
    else:
        channels, hidden, flips = PP.motion(opts, lay, pond, times)
        shapes = _pond(name, coll, root, carrier, pond, seed, mat_of, obs, flips)
    for ch in channels:
        o = obs[ch.object]
        if ch.path.startswith("shape:"):
            K.set_fcurve(o.data.shape_keys, f'key_blocks["{ch.path[6:]}"].value', 0, frames, ch.values, interp=interp)
        else:
            K.set_fcurve(o, ch.path, ch.index, frames, ch.values, interp=interp)
    for nm, hid in hidden.items():
        _key_hidden(obs[nm], frames, hid, viewport=nm not in flips)
    if pond is not None and opts["house"] is not None:
        _house(opts, pond, obs, frames, hidden, flips, f0, fps, sc.frame_start)
    for nm, (states, index) in flips.items():
        if len(states) > 1:
            _flipbook(obs[nm], [shapes[nm](s) for s in states], index, frames, interp)
    card = PT.card(name, opts, lay, pond)
    ride = opts["pond"]["ride"] if pond is not None and opts["pond"] is not None else None
    if ride:                                              # the main pad bobs with what she lies on, within the water's
        when = opts["pond"]["ride_when"]                  # give (its top at z = 0 on the lane): the props stage ties it
        card["ride"] = [{"object": f"{name}_pad1", "target": ride, "axes": "z", "frames": [float(f) for f in frames],
                         "influence": [1.0 if when is None or any(a <= t < b for a, b in when) else 0.0 for t in times],
                         "limit": {"z": [-PP.RIDE_GIVE, PP.RIDE_GIVE]}}]       # once that exists (docs/design.md: Sets)
    return card


def _house(opts, pond, obs, frames, hidden, flips, f0, fps, start):
    """With `house`: everything but the water and what floats on it is hidden outside the house's windows, switching on
    the cut frames themselves; the cover water shows only then. Each object's own hiding (on the set's steps) holds
    while the house is there."""
    ks, gone = PP.house_keys(opts, f0, fps, start)
    keep = PP.keeps(pond)
    frames = np.asarray(frames, float)
    for nm, o in obs.items():
        if o.type != "MESH" or (nm in keep and nm != "cover"):
            continue
        own = np.asarray(hidden.get(nm, np.zeros(len(frames), bool)), bool)
        keys = np.unique(np.concatenate([frames, ks]))
        i = np.clip(np.searchsorted(frames, keys, side="right") - 1, 0, None)
        g = gone[np.clip(np.searchsorted(ks, keys, side="right") - 1, 0, None)]
        _key_hidden(o, keys, ~g if nm == "cover" else (own[i] | g), viewport=nm not in flips)


def _pond(name, coll, root, carrier, pond, seed, mat_of, obs, flips):
    """The pond's objects (mkmmd.core.paper_theatre_pond) into `obs`; returns {flipbook object: shape(state) -> its
    positions}. A flipbook object's mesh is made in the shape its first key shows (the flipbook's basis)."""
    def mesh_ob(nm, mesh, parent, loc=(0.0, 0.0, 0.0)):
        o = SH.exempt(SH.mesh_object(f"{name}_{nm}", mesh, coll, parent, mat_of, loc=loc), FLAT)
        o.rotation_mode = "XYZ"
        obs[nm] = o
        return o

    def flip_ob(nm, mesh, shape, parent, loc):
        states, index = flips[nm]
        mesh_ob(nm, S.Mesh(shape(states[int(index[0])]), mesh.Q, mesh.T, mesh.Qm, mesh.Tm), parent, loc)
        shapes[nm] = shape
    shapes = {}
    for r in pond.rows:
        mesh_ob(r.name, P.part_mesh(r.part, seed), root, (0.0, 0.0, pond.lift))
    for w in pond.wings:
        mesh_ob(w.name, P.part_mesh(w, seed), root, (0.0, 0.0, pond.lift))
    if pond.lane:
        lane = _empty(f"{name}_lane", coll, root, (0.0, 0.0, 0.0))
        obs["lane"] = lane
        for p in pond.lane:
            loc = (p.x, p.y, 0.0) if p.kind == "pad" else (p.x, 0.0, pond.lift)
            mesh_ob(p.name, P.part_mesh(p.part, seed), lane, loc)
        for li in pond.lilies:
            mesh, positions = P.lily_mesh(li, seed)
            flip_ob(li.name, mesh, positions, lane, (li.x, li.y, li.z))
    if pond.tab is not None:
        mesh_ob("tab", P.part_mesh(pond.tab, seed), root, pond.tab_at)
    if pond.crank is not None:
        crank = mesh_ob("crank", P.part_mesh(pond.crank, seed), carrier, pond.crank_at)
        mesh_ob("crank_knob", P.part_mesh(pond.knob, seed), crank, pond.knob.at)
        for half in pond.halves:
            unrolled = P.half_mesh(half, seed)
            flip_ob(half.name, unrolled, lambda d, V=unrolled.V: PP.wind(V, d, pond.sky), root, (0.0, 0.0, pond.lift))
    for s in pond.strokes:
        mesh, positions = P.stroke_mesh(s, pond.sky)
        flip_ob(s.name, mesh, positions, root, (0.0, 0.0, pond.lift))
    for rp in pond.ripples:                               # on the water round the main pad, riding the lane
        mesh, positions = P.ripple_mesh(rp, PP.RIPPLE[3])
        flip_ob(rp.name, mesh, positions, obs["lane"], (PP.PAD_X, 0.0, -PP.W0 + 0.0004))
    return shapes
