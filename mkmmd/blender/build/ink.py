"""ink: handwriting on a page that appears behind the nib, with no Python at render time (docs/design.md: Text, Ink).

The text stage hands every `[[text]]` that carries `ink = {...}` to `build`. The strokes become ONE ribbon mesh
(mkmmd.core.ink: a strip of quads per pen-down stroke) in the page's surface frame, parented to the owner's root so it
follows the prop; every vertex carries the time the nib passes it as the point attribute `tw`. The shader shows a
fragment once the object's clock `clip_t` (a custom property KEYED linearly over the frames: clip seconds, no driver,
no handler) has passed its `tw`; attributes interpolate, so the front sits exactly at the nib, between frames too. Fresh
ink is glossy and a touch deeper in colour and dries matte over `dry` seconds.

[[text]]
name = "ink"
ink = {strokes = "tracks/ink.json", on = "page:page", color = "pine"}

ink keys
  strokes = "tracks/ink.json"   the project's strokes file: {"unit": "mm", "strokes": [{"t": [clip s], "p": [[x, y]]}]}
                            in page millimetres from the top-left corner as read (x right, y down), points ~0.15 mm
                            apart; "page": [w, h] mm is checked against the surface
  track = "nib"             coarse fallback instead: the runs of `down` in tracks/nib.json (or a .json path), the nib
                            `target` per frame read on the surface; only while the nib is within 1 mm of the paper
  on = "page:page"          the surface to write on, `<prop or set>:<surface>` (a card's use.surface; the surface name may
                            go when the owner has one): page millimetres map onto its panel from the top-left corner
  width = 0.00042           ribbon width (m); lift = 0.00018  distance above the surface (m)
  color = "pine"            palette slot or #hex ("slot:slot:t" mixes two)
  dry = 0.55                seconds fresh ink stays glossy (0 or glossy = false: matte from the start)
  glossy = true             fresh ink shines and is deeper in colour
  from, to                  clip seconds: only ink written in this window is built (a stroke across a bound is cut there)
"""
import json
import os

import bpy
import numpy as np
from mathutils import Matrix

from ...core import ink as I
from ...core import pentrack as PT
from ...core import typeset as T
from .. import keys as K
from ..library.sets import nightkit as NK
from . import BuildError

KNOWN = {"strokes", "track", "on", "width", "lift", "color", "dry", "glossy", "from", "to"}
CLOCK = "clip_t"                # the ink object's custom property the shader reads: clip seconds, keyed linearly
DRY_S = 0.55                    # seconds fresh ink stays glossy
DEEPER = 0.82                   # fresh ink is this share of the dry colour
ROUGH_WET, ROUGH_DRY = 0.08, 0.62
COAT_ROUGH = 0.05
TOUCH = 0.001                   # a track's nib is on the paper within this of the surface (m)
PAGE_TOL = 1.0                  # mm a strokes file's page may differ from the surface before it is refused


# ------------------------------------------------------------------------------------------------------ inputs
def _surface(ctx, name, ref):
    """(parent object, parent-inverse or None, frame 4x4 in the parent's frame, (w, h) panel m, label) of `ref`."""
    owner, _, sname = str(ref).partition(":")
    if owner in ctx.props:
        kind, o = "prop", ctx.props[owner]
    elif owner in ctx.sets:
        kind, o = "set", ctx.sets[owner]
    else:
        raise BuildError(f"ink {name!r}: on {ref!r}: no prop or set {owner!r} (props {sorted(ctx.props)}, sets "
                         f"{sorted(ctx.sets)})")
    surfaces = o.card.get("use", {}).get("surface", [])
    names = [s["name"] for s in surfaces]
    if sname:
        s = next((s for s in surfaces if s["name"] == sname), None)
        if s is None:
            raise BuildError(f"ink {name!r}: {kind} {owner!r} has no surface {sname!r} (has {names})")
    elif len(surfaces) == 1:
        s = surfaces[0]
    else:
        raise BuildError(f"ink {name!r}: on {ref!r}: {kind} {owner!r} has " +
                         (f"{len(surfaces)} surfaces, name one ({names})" if surfaces else "no surfaces"))
    F = T.surface_matrix(s["center"], s["normal"], s.get("up", (0.0, 0.0, 1.0)))
    parent, inverse = o.root, None
    carrier = bpy.data.objects.get(s["object"]) if s.get("object") else None
    if carrier is not None and carrier.parent == o.root:         # the surface rides an object of its own
        parent = carrier
        inverse = (carrier.matrix_parent_inverse @ carrier.matrix_basis).inverted()
    return parent, inverse, F, (float(s["size"][0]), float(s["size"][1])), f"{owner}:{s['name']}"


def _track_strokes(ctx, name, ref, world, size):
    """Coarse strokes from a pen track: the `down` runs of the nib `target` per frame, read on the surface whose
    frame is `world` (4x4) and only while the nib is within TOUCH of its plane."""
    path = ref if ref.endswith(".json") else os.path.join("tracks", f"{ref}.json")
    try:
        frames, pos = PT.load(ctx.path(path))
        with open(ctx.path(path), encoding="utf-8") as fh:
            down = json.load(fh).get("down")
        if down is None or len(down) != len(frames):
            raise ValueError("needs a `down` flag (1 while the pen is down) for every frame")
        local = (np.linalg.inv(world) @ np.c_[pos, np.ones(len(pos))].T).T[:, :3]
        strokes = I.from_track((frames - ctx.frame0) / ctx.fps, I.to_page(local[:, :2], size),
                               np.asarray(down, bool) & (np.abs(local[:, 2]) <= TOUCH))
    except (OSError, ValueError) as e:                           # InkError is a ValueError
        raise BuildError(f"ink {name!r}: track {ref!r}: {e}")
    return strokes


def _strokes(ctx, name, ink, world, size):
    """(strokes, source label)."""
    if ("strokes" in ink) == ("track" in ink):
        raise BuildError(f"ink {name!r}: give exactly one of `strokes` (the strokes file) and `track` (a pen track)")
    if "track" in ink:
        return _track_strokes(ctx, name, str(ink["track"]), world, size), f"track {ink['track']}"
    try:
        strokes, page = I.load(ctx.path(ink["strokes"]))
    except I.InkError as e:
        raise BuildError(f"ink {name!r}: {e}")
    if page is not None and (abs(page[0] - size[0] * 1e3) > PAGE_TOL or abs(page[1] - size[1] * 1e3) > PAGE_TOL):
        raise BuildError(f"ink {name!r}: the strokes were made for a {page[0]:g} x {page[1]:g} mm page but the surface is "
                         f"{size[0] * 1e3:g} x {size[1] * 1e3:g} mm")
    return strokes, str(ink["strokes"])


# ------------------------------------------------------------------------------------------------------ blender
def _mesh(name, rb):
    """The ribbon as a mesh with the point attribute `tw`, filled in one go."""
    me = bpy.data.meshes.new(name)
    n = len(rb.faces)
    me.vertices.add(len(rb.verts))
    me.vertices.foreach_set("co", rb.verts.astype(np.float32).ravel())
    me.loops.add(n * 4)
    me.loops.foreach_set("vertex_index", rb.faces.astype(np.int32).ravel())
    me.polygons.add(n)
    me.polygons.foreach_set("loop_start", (np.arange(n) * 4).astype(np.int32))
    me.polygons.foreach_set("loop_total", np.full(n, 4, np.int32))
    me.update(calc_edges=True)
    me.attributes.new("tw", "FLOAT", "POINT").data.foreach_set("value", rb.tw.astype(np.float32))
    return me


def _material(name, rgb, dry, glossy):
    """Principled ink, alpha 1 once the clock has passed the fragment's `tw`. Glossy: fresh ink (wet 1 falling to 0
    over `dry` seconds) is deeper, shines (coat, low roughness) and dries matte."""
    m, nb = NK.new_material(name)
    m.surface_render_method = "DITHERED"
    tw = nb.attr("tw", "Fac")
    clock = nb.attr(CLOCK, "Fac", "OBJECT")
    shown = nb.math("GREATER_THAN", clock, tw)
    if glossy:
        wet = nb.remap(nb.math("SUBTRACT", clock, tw), 0.0, dry, 1.0, 0.0)
        col = nb.mixc(wet, (*rgb, 1.0), (*(c * DEEPER for c in rgb), 1.0))
        sh = nb.principled(col, rough=nb.remap(wet, 0.0, 1.0, ROUGH_DRY, ROUGH_WET), coat=wet, coat_rough=COAT_ROUGH,
                           alpha=shown)
    else:
        sh = nb.principled((*rgb, 1.0), rough=ROUGH_DRY, alpha=shown)
    nb.output(sh)
    return m


def build(ctx, coll, spec):
    """One ink object for a `[[text]]` entry with `ink = {...}`; returns its report (`strokes` = pen-down strokes)."""
    name = spec.get("name")
    if not name:
        raise BuildError("[[text]] needs a name")
    extra = sorted(set(spec) - {"name", "ink"})
    if extra:
        raise BuildError(f"ink {name!r}: unknown keys {extra} next to `ink` (everything goes inside ink = {{...}})")
    ink = spec["ink"]
    if not isinstance(ink, dict):
        raise BuildError(f"ink {name!r}: `ink` must be a table such as {{strokes = \"tracks/ink.json\", on = \"page\"}}")
    unknown = sorted(set(ink) - KNOWN)
    if unknown:
        raise BuildError(f"ink {name!r}: unknown keys {unknown} (known: {sorted(KNOWN)})")
    if not ink.get("on"):
        raise BuildError(f"ink {name!r}: `on` = \"<prop>:<surface>\" says which surface to write on")
    if bpy.data.objects.get(name) is not None:
        raise BuildError(f"ink {name!r}: an object of that name exists already")
    try:
        width, lift = float(ink.get("width", I.WIDTH)), float(ink.get("lift", I.LIFT))
        dry, glossy = float(ink.get("dry", DRY_S)), bool(ink.get("glossy", True))
        lo, hi = (None if ink.get(k) is None else float(ink[k]) for k in ("from", "to"))
    except (TypeError, ValueError):
        raise BuildError(f"ink {name!r}: width, lift, dry, from and to are numbers") from None
    wet_look = glossy and dry > 0
    if width <= 0 or lift < 0 or dry < 0:
        raise BuildError(f"ink {name!r}: width must be positive, lift and dry not negative")
    try:
        rgb = NK.rgb(ctx.palette, ink.get("color", "pine"))
    except KeyError as e:
        raise BuildError(f"ink {name!r}: {e.args[0]}")

    parent, inverse, F, size, label = _surface(ctx, name, ink["on"])
    bpy.context.view_layer.update()
    eye = Matrix.Identity(4) if inverse is None else inverse
    world = np.array(parent.matrix_world @ eye @ Matrix([list(map(float, r)) for r in F]))
    strokes, source = _strokes(ctx, name, ink, world, size)
    try:
        strokes = I.window(strokes, lo, hi)
        if not strokes:
            span = (f" written between {lo:g} and {hi:g} s" if lo is not None and hi is not None else
                    f" written from {lo:g} s on" if lo is not None else f" written up to {hi:g} s" if hi is not None else "")
            raise BuildError(f"ink {name!r}: {source} has no ink{span}")
        rb = I.ribbon(strokes, size, width, lift)
    except I.InkError as e:
        raise BuildError(f"ink {name!r}: {e}")

    mat = _material(f"{name}_ink", rgb, dry, wet_look)
    me = _mesh(name, rb)
    me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    coll.objects.link(ob)
    ob.parent = parent
    if inverse is not None:
        ob.matrix_parent_inverse = inverse
    ob.matrix_basis = Matrix([list(map(float, r)) for r in F])
    ob.visible_shadow = False
    ob[CLOCK] = 0.0                                              # clip seconds, keyed: the shader's clock
    K.set_fcurve(ob, f'["{CLOCK}"]', 0, [ctx.start, ctx.end], [ctx.time(ctx.start), ctx.time(ctx.end)], interp="LINEAR")
    ctx.texts = getattr(ctx, "texts", {})
    ctx.texts[name] = ob
    return {"on": label, "source": source, "strokes": rb.strokes, "points": rb.points, "verts": len(rb.verts),
            "faces": len(rb.faces), "ink_mm": round(I.length_mm(strokes), 1),
            "t": [round(float(min(s.t[0] for s in strokes)), 3), round(float(max(s.t[-1] for s in strokes)), 3)],
            "width_mm": round(width * 1e3, 3), "lift_mm": round(lift * 1e3, 3), "dry_s": dry if wet_look else 0.0,
            "color": str(ink.get("color", "pine"))}
