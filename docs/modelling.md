# Modelling props: forms, not boxes

A prop that is assembled from cuboids reads as a toy even when every dimension is right: flat faces, hard edges that catch no
light, no surface flowing into the next. This page is the standard for everything that is built in code (`library/props/*`,
`library/sets/*`, `mk model`): the rules, the toolkit that makes them cheap to follow, and the recipes the car
(`convertible_80s`) is built with. The `form` check in `mk check` measures the rules on the evaluated scene.

## The rules

1. **Never assemble a visible form from cuboids.** A box, a 6-face slab, an axis-aligned plate or a stack of them is not a
   form. A door, a seat back, a dash, a bumper, a lamp surround, a knob, a pedal, a mirror is a surface with curvature:
   build it from a profile, a loft, a lathe or a sweep. A part whose bounding box it fills (a cuboid in disguise) fails
   as much as `rounded_box` with a 0.2 mm radius.
2. **Every visible hard edge is a deliberate crease on a subdivided surface, or bevelled (>= 3 mm at car scale; 1 mm on a
   part smaller than a hand).** A crease (the shoulder line of a car) is a decision, kept sharp on a smooth surface; a
   silhouette that turns a corner without a radius is a mistake. Real sheet metal, plastic and leather have radii: they
   are what the highlights live on.
3. **Forms come from profiles and lofts.** Draw the section (a curve), then pull it along a path or between stations;
   let the Subdivision Surface do the rounding; put a crease only where the real part has a line. Fewer, well placed
   control points beat many.
4. **Work from real references and compare side by side at matching angles before calling anything done.** Look at the
   reference photo and the render at the same yaw, elevation and lens (`mk look ... --ref` puts them side by side; a 10 % grid drawn on both helps), correct proportion and silhouette first, then the surface detail. Numbers from a data sheet are
   starting values; the photo decides. A first render is never the answer.
5. **Graphic layers are not forms.** A display segment, a label stripe, a seam ribbon, a decal lifted a fraction of a
   millimetre off a surface may be flat quads: tag the object `exempt(obj)` (custom property `mk_form_exempt`) and the
   check skips it. Never use it to hide a form.

No logos, badges, lettering or brand names on a model; references are looked at, never copied or traced.

## The toolkit

`mkmmd/core/shell.py` is pure numpy (importable from tests, no `bpy`); `mkmmd/blender/library/shell.py` turns its meshes
into Blender objects. Code that runs inside Blender is Python 3.11 with numpy 1.24: no scipy, no numpy-2-only API.

### Forms (core)

| function | for |
|---|---|
| `loft(stations, spacing, ...)` | a body from a few half cross-sections along y, interpolated smoothly, mirrored, with creases, caps, gaps and open-top regions. Returns a `Loft` (the cage `.mesh` and addressing helpers). |
| `cage(rings, closed, loop, caps, crease_cols, crease_rings, ...)` | a quad cage from any grid of rings (n stations x m points): a dash along x, a seat back with pleats, a surround along y |
| `sweep(path, section, scale, up)` / `tube` / `ribbon` | a profile along a path: bumper strips, trim, piping, grab handles, rims. Per-point `scale` tapers a strip into its end caps |
| `lathe(profile)` / `skin(sections)` | surfaces of revolution (knobs, domes, bezel rings), sections skinned together |
| `rounded_panel(outline, thickness, r_edge, dome)` | a plate with a rolled rim and a domed face: buttons, labels, pads, switches |
| `rounded_frame(outer, inner, height, r_out, r_in, floor)` | a bezel, vent or surround with real rolled rims and an optional pocket floor |
| `rounded_box(size, r)` | a small block with real fillets (radius 3 mm and more). Use for hidden or tiny parts; never to build a visible form |
| `extrude_profile(bottom, top, x0, x1, ...)` | a side profile pulled across x with rounded rims |
| `arch_cutter(center, radius, x0, x1, flare, lip)` | the solid a wheel arch is cut with (a flared cylinder), for a Boolean |
| outlines | `rrect`, `superellipse(w, h, p)`, `stadium`, `ellipse`, `fillet(points, radius)`, `arc`, `resample_loop`, `offset_rings` |

`Mesh` holds vertices `V`, quads `Q`, triangles `T`, a material index per face (`Qm`, `Tm`), optional per-vertex `UV` and
edge creases (`Ce` pairs, `Cw` weights). `merge`, `weld`, `apply`/`rot`/`moved`, `mirrored_x`, `crease`, `volume`,
`is_closed`, `triangles`. Winding is outward for every primitive; faces are smooth shaded.

### Lofting a body

A station is a half section at one position along y: named points `pt(name, x, z)` from the underside on the centre line
(x = 0), out and around the +x side, to the top on the centre line. Every station has the same names; the polygon of
points is interpolated between stations with monotone cubics (no overshoot: a few stations give a long smooth form), and
the Subdivision Surface rounds the polygon off.

```python  # doc-test
import numpy as np
from mkmmd.core import shell as S


def section(half_w, belt, top):
    return [S.pt("keel", 0.0, 0.20), S.pt("sill", half_w - 0.04, 0.20), S.pt("rocker", half_w, 0.26),
            S.pt("flank", half_w, belt - 0.12), S.pt("shoulder", half_w - 0.01, belt, crease=1.0),   # a deliberate line
            S.pt("edge", half_w - 0.10, top - 0.02), S.pt("roof", 0.30, top), S.pt("crown", 0.0, top + 0.01)]


body = S.loft([S.station(-2.1, section(0.55, 0.62, 0.74)),                 # the nose: a smaller section
               S.station(-1.6, section(0.84, 0.76, 0.88)),
               S.station(0.0, section(0.86, 0.80, 0.94)),
               S.station(1.6, section(0.84, 0.76, 0.88)),
               S.station(2.1, section(0.55, 0.62, 0.74))],
              spacing=0.12, cap_bulge=(0.02, 0.02),
              gaps=[S.gap(-0.9, width=0.008, depth=0.004, mat=3)])           # a hood / door gap across the top and sides
mesh = body.mesh                                                             # the cage, with crease weights
assert mesh.is_closed() and mesh.volume() > 0
```

* `hard=True` on a `station(...)` breaks the interpolation there (a sharp turn along the loft: a cowl, a deck edge); put
  a ring crease on it with `ring=1.0` (or `(1.0, first_name, last_name)` for a part of the ring).
* `pt(..., y=dy)` moves a point along the loft (a raked grille, a chevron); `pt(..., crease=w)` creases the edges that
  run along the loft through that point.
* **Open-top regions (a cockpit):** keep the same point names but let the top points of the stations in that range run
  down a lip, the inner wall and across a floor; the section becomes a U. Hard stations with ring creases at the ends of
  the opening turn the top surface into its front and back walls.
* `Loft.assign(mat, y=(y0, y1), cols=(first, last), side=...)` sets the material of a patch (the rocker moulding, the
  two-tone), `Loft.x_at(y, z)`, `Loft.point(name, y)` read the cage back, `crease_ring`, `crease_col` add creases.
* Support loops: a subdivided corner rounds over about the length of its two adjacent cage edges. Put two points 20-40 mm
  apart where a radius of 10-30 mm is wanted, and one 100 mm apart where the surface should be flat.

### Cages from rings

`cage(rings, ...)` takes any `(stations, points, 3)` array. `closed=True` joins the last point to the first (a tube),
`loop=True` joins the last ring to the first (a torus), caps close the ends of a tube with a flat disc of rings and a fan
(`cap_bulge` domes it). `crease_cols` (weights per point) and `crease_rings` (weights per ring) are the lines that stay
sharp. A horizontally pleated seat back is rings at the pleat crests and troughs with the trough rings creased 0.6.

### Panels, frames, strips

```python  # doc-test
import numpy as np
from mkmmd.core import shell as S

pad = S.rounded_panel(S.superellipse(0.105, 0.07, 3.5, 40), 0.016, r_edge=0.005, dome=0.004)           # a horn pad
bezel = S.rounded_frame(S.rrect(0.118, 0.026, 0.010, 4), S.rrect(0.108, 0.016, 0.006, 4), 0.006, floor=0.004)  # a slot
path = np.stack([np.linspace(-0.8, 0.8, 40), np.full(40, -2.28), np.full(40, 0.34)], 1)
taper = np.minimum(1.0, np.sin(np.linspace(0.0, np.pi, 40)) * 6.0)                                     # round end caps
strip = S.sweep(path, S.rrect(0.07, 0.11, 0.025, 4), scale=taper[:, None] * [1.0, 1.0], up=(0, -1, 0))
for m in (pad, bezel, strip):
    assert m.is_closed()
```

### Blender side

```python
from mkmmd.blender.library import shell
body = shell.shell_object(f"{name}_body", cage, coll, root, mat_of, levels=2, bevel=(0.003, 2), cutters=cutters,
                          bake_it=True)                                   # bake what moves or is cut
part = shell.mesh_object(f"{name}_lamp", mesh, coll, root, mat_of)       # a plain part, smooth normals
sticker = shell.exempt(shell.mesh_object(f"{name}_label", ribbon, coll, root, mat_of))   # a graphic layer
surface = shell.probe(body)                                              # the evaluated shell as a ray target
hit = surface.ray((0.3, -3.0, 0.6), (0, 1, 0))                           # (t, point, normal): sit a lamp on it exactly
```

* `shell_object` = cage -> mesh with `crease_edge` -> Subdivision Surface (`levels` in the viewport, `levels + 1` on
  render) -> Boolean DIFFERENCE for every cutter -> Bevel on the edges the cut leaves hard. With `bake_it=True` the stack is
  applied at the render level into a plain mesh (the cutters are deleted). **Bake anything that moves or is cut by a
  Boolean**: a live modifier stack is re-evaluated whenever its object (or a cutter) moves, which is about a second per
  frame for a car body, i.e. an animated vehicle can no longer be built, simulated or rendered. Static furniture may stay
  live (the `.blend` then carries the small cage). Cost on a 2300-point cage at level 3: Subdivision 0.1 s, four arch
  Booleans 3 s with `solver="EXACT"`, 1 s with `"FAST"` (`S.smooth(..., solver=...)`).
* A section that crosses itself (a floor below the underside, a wall through a wall) is not a solid: `loft` refuses it, and
  a Boolean on such a body silently returns nothing.
* A Catmull-Clark surface pulls in from its control polygon (up to a few millimetres on a tight curve, a hand on a fat
  corner): never place a part on the cage polygon. Put it on `shell.probe(...)` hits and sink its back 4-15 mm into the
  body, so no sliver of gap shows. `shell.probe` is a ray cast on Blender's BVH tree (microseconds per ray on a million
  triangles); the numpy `core.shell.Probe` brute-forces every triangle and is for tests and small meshes.
* A material index stays on the quads when they are subdivided: assign materials on the cage, not on the result.

## Recipes

| to get | do |
|---|---|
| a crisp character line (a shoulder crease) | a point with `crease=1` on every station; the line is sharp where the cage is, soft nowhere; the Bevel makes it catch the light |
| a panel gap | `gap(y, width, depth, cols, mat)` in `loft`: a V groove between two creased rows, its faces in the dark gap material |
| a wheel arch with a lip | `arch_cutter(...)` + Boolean in `shell_object`: the flare rolls into a rim |
| a rolled edge (door top, dash brow) | points at the radius' tangents, 2-3 per 90 degrees, and no crease |
| a bumper strip with end caps | `sweep` of a `rrect` section along the plan path, `scale` tapering to 0 at both ends |
| a lamp, switch, button | `rounded_panel` (domed) in a `rounded_frame`; the lens is a separate panel with `dome` |
| a knob, hub, ring | `lathe` of a profile with `fillet`ed corners |
| a grille / louvres | `rounded_frame` for the surround, `sweep` blades of a lens section, never box bars |
| upholstery | `cage` with creased trough rings (pleats), subdivided, with a lighter roll on the edges |

## Worked example: the convertible

`mkmmd/blender/library/props/convertible*.py` (card `library:convertible_80s`), modelled in proportion on a 1984-86 Dodge 600
convertible with no badge or lettering. `convertible_layout.py` holds every number the parts agree on, measured on a
side-on photograph at the known length (4.59 m) and wheelbase (2.62 m): the shoulder crease at 0.83 m the whole way along
the side, the door top at 0.915 m, the foot of the glass at 0.925 m with the windshield raked 44 degrees up to a header at
1.32 m, a hood that rises from 0.80 m at the nose to 0.99 m at its rear edge and drops to the glass.

| part | how |
|---|---|
| `convertible_body` | ONE lofted shell: 19 named points per half section, stations every metre or so (a closed top for hood and deck, a U with a door-top cap, inner walls and a floor for the cockpit, hard stations between), creases on the shoulder and the rub strip, ring gaps for the hood, doors and deck lid, four flared arch cutters (Boolean). Subdivided, cut and bevelled by the builder, then baked |
| `convertible_exterior` | grille (chrome slats), headlamps (bezel, two lenses), corner lamps, swept bumpers with rub strips and guards, plates, the ribbed tail-lamp band: lofts, sweeps and `rounded_frame`/`rounded_panel`, put on the body with ray casts |
| `convertible_trim` | rub strips and rocker mouldings that follow the side, the lip round each opening, door pulls, fender vents, hood vents, mirrors (a lofted pod on a stalk), wipers, antenna, exhaust tip, the dark shut lines (graphic layers) |
| `convertible_seats`, `convertible_dash` | cages with creased pleats, a rolled dash brow, lathed knobs, swept spokes |

The builder (`convertible.py`) makes the shells first, bakes them, builds ray-cast probes on them, then asks the other modules for
their meshes with those probes (`static_parts(probes)`), so a lamp is placed on the real surface. Reference checks used: the
side silhouette laid over the photograph (orthographic render at the photo's scale), and the car beside the photographs at the
same angle.

## Worked example: the guitar

`mkmmd/blender/library/props/electric_guitar*.py` (card `library:electric_guitar`): an 80s Strat-type electric with no logo or
lettering. `electric_guitar_layout.py` holds every number and the card (the contract with the grip, strum and wear code), `_geo`, `_body`,
`_neck`, `_hw` and `_parts` the meshes (numpy only, so `tests/test_electric_guitar.py` checks the card against them), `_mats` and
`electric_guitar.py` the Blender side. It builds in 0.7 s and scores 0.014 on `form` (cuboid 0.008; worst parts: the hidden faces of
`guitar_fretboard` 0.04 and `guitar_neck` 0.02; before the board's underside corners were rolled it was 0.054, that flat sharp underside on top).

**What was measured.** A photograph is flat and has perspective, so the first step was a metric frame: the 20 visible fret wires of a
full-length front photo, fitted with 12-TET on the known 648 mm scale and a scale that shrinks along the neck, give the nut row, the pixels
per metre at every height (residual 0.4 mm; a plain scale left 2 mm and put the saddle line 25 px off) and, with the neck's tilt, an image
rectified to a grid in metres (a small script of your own: x across, z along the strings from the saddle line). Read off it
as landmarks, never traced: body 0.315 across the lower bout and 0.225 at the waist (z 0.125), tail edge z -0.1155, long horn tip z 0.314
(the 12.6th fret) and short horn 0.2565 (16th) with its bay down to 0.196, the pocket's end at 0.2545, pickups at z 0.159 / 0.100 / 0.045 (the
bridge one slanted 9 degrees, poles 10.5 mm apart), a 75 x 42 mm bridge plate, knobs on an arc, the jack's 71 x 33 mm boat plate at -36
degrees; the headstock (a flat photo at 6.19 px/mm) 0.1826 long, 0.088 wide, its six posts on a line 17 degrees off the neck. From the data
sheet: 22 frets, a 9.5 inch board radius, nut 42 mm and 56 mm at the last wire, neck depth 20.5 mm at the nut and 25 mm at the heel,
back exponent p = 2.6, a 10-46 set.

| part | how |
|---|---|
| body, guard | `electric_guitar_geo.slab`: the outline is a spline through 113 control points, the rim a stack of offset rings (4.5 mm radius, 4 segments, front and back), each face a Delaunay triangulation of the inset outline plus a lattice (horns and bays are not star-shaped: `rounded_panel`'s cap would fold), so a face is any height function: `face` = the forearm bevel (a smoothstep scoop on the bass edge), `back` = the belly cut. The guard is a second slab on `face` whose chamfer shows three plies |
| neck, headstock | ONE `skin` of 64 rings along z, each the lower half of |2x/w|^p + (v/d)^p = 1 cut under the board: along the neck the mesh IS the card's `section` (the test slices it to 1e-9). Past the nut the ring's centre and width follow the headstock outline (pchip tables) while p grows to 6, so the C runs out into the flat back and the face is the plane the board sits on |
| board, wires | a skin of radiused rings (rolled edges, walls that continue the neck's curve); 22 wires `sweep`t across along the radius with tapered ends, 1.2 mm proud; the dots are lathes |
| strings | tubes through the card's nut and saddle points with a graded action (2.4 to 1.6 mm over fret 12, so the saddles stand 4-7 mm over the board plane), coiled 2.5 turns on the posts |
| hardware | lathes (posts, knobs, strap buttons, plug, screws, poles), `rounded_panel`s (covers, plates, paddles), `sweep`s (saddles, lever), put on the guard or the face with `S.place` |
| materials | one Principled per colour role, a clear coat on the body, a faint grain on the woods, and for metal a facing-angle glow (a matcap) so it keeps its form in a dark world where it reflects nothing |

Checks used: the rectified photograph beside the orthographic render and the render's outline over it (the body is within 2 mm
all round, IoU 0.96 with the pick and cord in; the headstock within 2 mm), 3/4 views against a case photograph, and a `silhouette`
shot (`keep = ["guitar*"]`, `accent = ["guitar_cable"]`) upright and turned 65 degrees about +Y, where the long neck, headstock,
horns, strap buttons and the hanging cord read at once.

## Reviewing

* Render the prop alone from named views and set it beside the reference photo at the same angle and lens: front 3/4
  left, rear 3/4 left, side, front, and close-ups of every detail (`mk look --view yaw:elev --target ... --lens ...`).
* A proportion error is a silhouette error: fix it with the profile curves, not with details. For a side profile render the
  prop with an orthographic camera at the photograph's scale (the length and wheelbase give pixels per metre; the wheel
  centres give the ground line) and lay its outline over the photo: a few centimetres of hood height, glass rake or sill
  height show at once, where a perspective comparison hides them.
* Run `mk check` (the `form` metric): no cuboid parts, no unbevelled hard edges on the visible parts of the prop.
* Keep the build under about 3 s per prop; evaluate `levels` 2 first, add the render level last.
