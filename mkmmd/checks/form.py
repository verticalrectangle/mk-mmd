"""form: how blocky a prop's modelled shapes are (docs/design.md: Checks; docs/AGENTS.md: Modelling props).

The Blender side hands over the evaluated geometry of what a render shows (mkmmd.blender.ops_sample, `meshes`); the maths
is mkmmd.core.form. Props are measured at one frame, in the frame of their root so that positions in the detail read like
the card's use points."""
from fnmatch import fnmatchcase

from ..core import form as FM
from . import CheckError, Metric, metric

FIX = ("build visible forms from lofts, profiles and subdivision with creases (mkmmd.core.shell), bevel every visible "
       "hard edge, never stack cuboids; look at a --ref side-by-side sheet (docs/AGENTS.md: Modelling props)")

FAR = 40.0                  # a measured thing wider than this (m) is no prop: the warning says so


def _list(v):
    if v is None:
        return []
    return [v] if isinstance(v, str) else list(v)


def _matches(name, patterns):
    return any(fnmatchcase(name, p) for p in patterns)


def _card_exempt(ctx, props):
    """`form_exempt` of the measured props' `card_extra` in the project's [[prop]] entries (names or patterns)."""
    out = []
    if ctx.project:
        for spec in ctx.project.data.get("prop", []):
            if spec.get("name") in props:
                out += _list((spec.get("card_extra") or {}).get("form_exempt"))
    return out


def _r(v, n=3):
    return round(float(v), n) + 0.0


def _vec(v, n=2):
    return [round(float(x), n) + 0.0 for x in v]


def report(r, worst=6):
    """The numbers and the parts to fix, from core.form.analyse's result."""
    detail = {"cuboid": _r(r["cuboid_share"]), "flat": _r(r["flat_share"]), "hard_edges": _r(r["hard_edge_share"]),
              "flat_sharp": _r(r["flat_hard_share"]), "area_m2": _r(r["area"], 2), "size_m": _vec(r["size"]),
              "diameter_m": _r(r["diameter"], 2), "objects": r["objects"], "parts": r["parts"],
              "triangles": r["triangles"],
              "worst_parts": [{"object": c["object"], "part": c["part"], "area_m2": _r(c["area"]),
                               "share": _r(c["share"]), "boxiness": _r(c["score"]), "why": FM.why(c),
                               "size_m": _vec(c["size"]), "at": _vec(c["at"])} for c in r["components"][:worst]],
              "worst_objects": {nm: {"area_m2": _r(o["area"]), "share": _r(o["share"]), "boxiness": _r(o["score"])}
                                for nm, o in list(r["by_object"].items())[:worst] if o["score"] > 0.01},
              "sharp_panels": [{"object": p["object"], "area_m2": _r(p["area"]), "share": _r(p["share"]),
                                "sharp_rim": _r(p["rim"], 2), "normal": _vec(p["normal"]), "at": _vec(p["at"])}
                               for p in r["patches"][:worst]]}
    if r["exempt"]:
        detail["exempt"] = {nm: {"area_m2": _r(o["area"]), "boxiness": _r(o["score"])} for nm, o in r["exempt"].items()}
    if r["ignored"]:
        detail["degenerate_triangles"] = r["ignored"]
    if r["diameter"] > FAR:
        detail["warning"] = (f"{r['diameter']:.0f} m across is a scene, not a prop: parts far smaller than that are too "
                             f"small to be tested for boxes. Measure its objects one by one (objects = [...])")
    if r["score"] > Form.default_max:
        detail["fix"] = FIX
    return detail


@metric
class Form(Metric):
    name = "form"
    doc = ("How blocky a prop or object set is, from its evaluated geometry at one frame (what a render shows: hidden "
           "colliders and objects hidden from render do not count). Value: boxiness 0..1, area-weighted over loose "
           "parts: 1 - (1 - cuboid)(1 - flat_sharp), where cuboid is how much a part is a box (its surface is flat faces "
           "on three orthogonal axes: stacked box primitives, bevelled or not) and flat_sharp the share of its surface "
           "in large flat patches (normals within 3 degrees, from 12% of the prop's diameter across) whose rims turn "
           "more than 65 degrees (no bevel). `flat` (all large flat area) and `hard_edges` (sharp share of the edge "
           "length around large patches) are reported for reading. Smooth, lofted or subdivided forms and rounded "
           "plates score ~0; a stack of boxes ~0.9; hero props must stay under 0.25. `worst_parts` names the objects "
           "and parts to fix. Architectural boxes: exclude / exempt them.")
    args = {"prop": "prop root name (or a list): every object a render shows under it, characters excluded",
            "objects": "object names measured as given (besides or instead of prop)",
            "exclude": "object names or fnmatch patterns left out entirely (hidden or internal geometry)",
            "exempt": "names or patterns measured and listed but weighed at exempt_weight (walls, gantries, barriers); "
                      "objects tagged mk_form_exempt in the scene and `card_extra.form_exempt` of the project's "
                      "[[prop]] count too",
            "exempt_weight": "weight of exempt parts, 0 (default: ignored) .. 1 (like the rest)",
            "frame": "Blender frame to measure at (default: the project's frame0, else the scene as it stands)",
            "hard_edge_deg": "dihedral angle above which an edge is hard (default 65)",
            "worst": "how many parts, objects and panels the detail lists (default 6)"}
    uses_frames = False
    default_max = FM.HERO_MAX

    def needs(self, args, ctx, need):
        props, objects = _list(args.get("prop")), _list(args.get("objects"))
        if not props and not objects:
            raise CheckError('form: give prop = "name" (a prop root: every object a render shows under it) or '
                             'objects = ["name", ...]')
        frame = args.get("frame")
        if frame is None and ctx.project:
            frame = ctx.project.frame0
        return {"k": need.mesh(objects, props, _list(args.get("exclude")), frame), "props": props}

    def compute(self, args, ctx, data, frames, st):
        mesh = data.mesh(st["k"])
        names, V = mesh["names"], mesh["vertices"]
        if len(mesh["triangles"]) == 0:
            raise CheckError("form: no visible geometry (colliders and objects hidden from render do not count)")
        space = "world"
        if len(mesh["roots"]) == 1:                     # prop frame: rigid inverse of the root (scale stays metres)
            V, space = FM.to_frame(V, next(iter(mesh["roots"].values()))), "prop"
        patterns = _list(args.get("exempt")) + _card_exempt(ctx, st["props"])
        exempt = [n for n in names if n in mesh["exempt"] or _matches(n, patterns)]
        worst = int(args.get("worst", 6))
        r = FM.analyse(V, mesh["triangles"], mesh["object"], names, exempt=exempt,
                       exempt_weight=float(args.get("exempt_weight", FM.EXEMPT_WEIGHT)),
                       hard_deg=float(args.get("hard_edge_deg", FM.HARD_DEG)), worst=worst)
        detail = {"at_frame": mesh["frame"], "space": space, **report(r, worst)}
        return r["score"], detail
