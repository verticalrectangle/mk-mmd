"""The cardboard kaleidoscope toy (`card = "library:kaleidoscope"`): a printed-paper tube lying on its side on two turned
rims, the eyepiece toward -Y. Objects under the root: `<name>_tube` (the card tube and its paper, the seam's step at the
bottom), `<name>_cap` (the eyepiece's turned cap, its dish painted), `<name>_ferrule` (the brass eyelet), `<name>_lens`,
`<name>_tunnel` and `<name>_view` (what glows at the end of the dark tunnel behind the lens), the empty
`<name>_chamber` on the axis carrying `<name>_rim` (the far rim), `<name>_frost`, `<name>_glass` and `<name>_beads`
(the beads and glass bits), the empty `<name>_eyehole` (the eyehole's centre, its Z out of the eyepiece: an insert's
anchor) and the hidden cylinder `<name>_col`. The root property `turn` (degrees, for [[key]]) turns the chamber and the
view about the axis (drivers). Numbers, meshes and the card: mkmmd/core/kaleidoscope.py; the print:
mkmmd/core/kaleidoscope_print.py; the materials: kaleidoscope_mats.py."""
import json
import math

import numpy as np

from ....core import kaleidoscope as KD
from ....core import kaleidoscope_print as KP
from ....core import palette as PAL
from .. import shell as SH
from ..mesh import cylinder as _collider_cylinder
from . import kaleidoscope_mats as KM
from . import register
from .cafe_kit import Kit, clear_drivers, drive

PX_PER_MM = 10.0                 # the print's resolution (a macro frame of the tube sees 10-40 px per mm of paper)
VIEW_PX = 512


def _palette(slots):
    """{slot: "#hex"}: Rose Pine Dawn under the project's palette and the prop's own slot overrides."""
    pal = dict(PAL.PALETTES["rose-pine-dawn"])
    pal.update({k: v for k, v in slots.items() if k in pal and isinstance(v, str) and v.startswith("#")})
    return pal


def _linear(rgb):
    return tuple(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb[:3]) + (1.0,)


def _part(K, base, mesh, parent, mat_of, at=None):
    """A part from a prop-frame mesh, its origin at `at` (default its bounds' centre: blended glass is sorted by the
    origins) in the frame of `parent` (whose own place in the prop frame is subtracted)."""
    lo, hi = mesh.bbox()
    c = (lo + hi) / 2.0 if at is None else np.asarray(at, float)
    off = np.asarray(parent.get("mk_at", (0.0, 0.0, 0.0)), float) if parent is not K.root else np.zeros(3)
    return SH.mesh_object(f"{K.name}_{base}", mesh.moved(-c), K.coll, parent, mat_of, loc=tuple(c - off))


def _turning(obj, root):
    """`obj` turns about the toy's axis (its parent's Y) by the root's `turn`, in degrees."""
    clear_drivers(obj)
    drive(obj, "rotation_euler", "radians(t)", 1, var=("t", root, '["turn"]'))


@register("kaleidoscope")
def kaleidoscope(name, coll, root, slots=None):
    K = Kit(name, coll, root, slots)
    o = KD.options(K.slots, KP.PATTERNS)
    pal = _palette(K.slots)
    clear_drivers(root)
    K.prop("turn", o["turn"], doc="degrees: the far rim and the chamber turned about the axis, the view with them")

    # ---- colours and pictures
    inks = KP.colors(o["pattern"], pal, o["print"])
    size_mm = (2.0 * math.pi * KD.TUBE_R * 1000.0, (KD.PAPER_S[1] - KD.PAPER_S[0]) * 1000.0)
    key = json.dumps([o["pattern"], o["seed"], sorted((k, [round(c, 4) for c in v]) for k, v in inks.items()),
                      [round(v, 3) for v in size_mm], PX_PER_MM])
    rgb = KP.paper_print(o["pattern"], inks, seed=o["seed"], size_mm=size_mm, px_per_mm=PX_PER_MM)
    rgba = np.empty(rgb.shape[:2] + (4,), np.float32)
    rgba[..., :3] = rgb / 255.0
    rgba[..., 3] = KP.paper_wear(seed=o["seed"], size_mm=size_mm, px_per_mm=PX_PER_MM)
    print_img = KM.packed_image(f"{name}_print", rgba, key=key)
    view_rgb = KP.view(inks, seed=o["seed"], px=VIEW_PX)
    view_rgba = np.ones(view_rgb.shape[:2] + (4,), np.float32)
    view_rgba[..., :3] = view_rgb / 255.0
    view_img = KM.packed_image(f"{name}_view", view_rgba, key=key)

    bare = K.blend(surface=0.86, gold=0.10, base=0.04)                 # the paper under the ink, a little yellowed
    mats = {KD.m("paper"): KM.paper(K, print_img, bare, o["wear"]), KD.m("paper_edge"): KM.paper_edge(K, bare),
            KD.m("card"): KM.card(K, K.blend(k=0.8, gold=0.4, base=0.35, text=0.25)),
            KD.m("rim"): KM.lacquer(K, "rim", K.role("rim", KD.ROLES), KD.RIM_R - 0.0005),
            KD.m("face"): KM.lacquer(K, "face", K.role("face", KD.ROLES), 1.0),
            KD.m("ferrule"): KM.brass(K, K.role("ferrule", KD.ROLES)), KD.m("lens"): KM.lens(K),
            KD.m("tunnel"): KM.tunnel(K, K.role("inside", KD.ROLES)), KD.m("view"): KM.view(K, view_img, o["glow"]),
            KD.m("frost"): KM.frost(K, K.role("frost", KD.ROLES)), KD.m("glass"): KM.clear_glass(K)}
    for i, b in enumerate(o["beads"]):
        colour = (*PAL.linear(b), 1.0) if b.startswith("#") else K.slot(b)
        tag = b.lstrip("#")
        mats[KD.BEAD_MAT + i], mats[KD.BIT_MAT + i] = KM.bead(K, tag, colour, o["glow"]), KM.bit(K, tag, colour, o["glow"])
    mat_of = mats.__getitem__

    # ---- the tube, its normals the cylinder's own
    lathe = KD.tube_mesh()
    tube = _part(K, "tube", KD.to_prop(lathe), root, mat_of, at=KD.point(KD.GRIP_S))
    tube.data.normals_split_custom_set((KD.tube_normals(lathe) @ KD.ROT.T).tolist())

    # ---- the eyepiece
    eye = KD.eyepiece_meshes()
    for part in ("cap", "ferrule", "lens", "tunnel"):
        _part(K, part, KD.to_prop(eye[part]), root, mat_of)
    view = _part(K, "view", KD.to_prop(eye["view"]), root, mat_of, at=KD.point(KD.VIEW_S))
    _turning(view, root)
    K.empty("eyehole", loc=tuple(KD.point(KD.EYE_POINT_S)), rot=(0.5 * math.pi, 0.0, 0.0), kind="CIRCLE",
            size=KD.EYE_OPEN)

    # ---- the far end, on the chamber's empty that `turn` turns
    centre = KD.point(0.5 * (KD.CHAMBER[0] + KD.CHAMBER[1]))
    chamber = K.empty("chamber", loc=tuple(centre), kind="SINGLE_ARROW", size=0.02)
    chamber["mk_at"] = [float(v) for v in centre]
    _turning(chamber, root)
    far = KD.far_meshes()
    for part in ("rim", "frost", "glass", "mirrors"):
        _part(K, part, KD.to_prop(far[part]), chamber, mat_of)
    _part(K, "beads", KD.to_prop(KD.chamber_mesh(o["seed"], len(o["beads"]))), chamber, mat_of)

    SH.purge(f"{name}_col")
    _collider_cylinder(f"{name}_col", tuple(KD.point(0.5 * KD.LENGTH)), KD.AXIS_Z, KD.LENGTH, coll, root,
                       rot=(90.0, 0.0, 0.0), collider=True)
    out = KD.card(name, {**{r: K.slots.get(r, KD.ROLES[r]) for r in KD.ROLES}, "beads": ",".join(o["beads"]),
                         "pattern": o["pattern"]})
    out["colliders"] = [{"type": "cylinder", "object": f"{name}_col", "tag": name}]
    return out
