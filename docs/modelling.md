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
| strings | one tube per string (`<name>_string_<n>`, 1 the highest, named in the card's neck `strings`), through the card's nut and saddle points with a graded action (2.4 to 1.6 mm over fret 12, so the saddles stand 4-7 mm over the board plane), coiled 2.5 turns on the posts; between the nut and the saddle a ring at every wire, at every press point and at most 6 mm apart (10 past the last wire), so a pressed string bends and a ringing one widens smoothly (`playable`, design.md: Playing a worn guitar, Strings that ring) |
| hardware | lathes (posts, knobs, strap buttons, plug, screws, poles), `rounded_panel`s (covers, plates, paddles), `sweep`s (saddles, lever), put on the guard or the face with `S.place` |
| materials | one Principled per colour role, a clear coat on the body, a faint grain on the woods, and for metal a facing-angle glow (a matcap) so it keeps its form in a dark world where it reflects nothing |

Checks used: the rectified photograph beside the orthographic render and the render's outline over it (the body is within 2 mm
all round, IoU 0.96 with the pick and cord in; the headstock within 2 mm), 3/4 views against a case photograph, and a `silhouette`
shot (`keep = ["guitar*"]`, `accent = ["guitar_cable"]`) upright and turned 65 degrees about +Y, where the long neck, headstock,
horns, strap buttons and the hanging cord read at once.

## Worked example: the offset guitar

`mkmmd/blender/library/props/offset_guitar*.py` (card `library:offset_guitar`): an offset-waist guitar that takes the electric
guitar's neck whole (`electric_guitar_neck`: the same scale, frets, strings and headstock, so the card's `neck` grip is that
guitar's entry) and has its own `_layout`, `_body`, `_hw` and `_parts`. Its numbers were read off one front-on photograph
(`mk ref photos`): the scale from the nut and the 12th fret, then the body's edges at stations 15 mm apart, the guard's the
same way, and the middles of the pickups, bridge, tremolo and controls; each was checked by drawing the layout's outlines over the
photograph at its scale (the first pickguard cut through the jack: its lower lobe had been read off a small picture, a zoom with a
millimetre grid put it right). The tremolo arm is a separate object so that a hand can carry it
([Playing a worn guitar](design.md#playing-a-worn-guitar)). It builds in 0.8 s and scores under 0.15 on `form`.

## Worked example: the bass

`mkmmd/blender/library/props/electric_bass*.py` (card `library:electric_bass`): a short-scale bass of the Mustang Bass kind,
built on the guitar's toolkit (`electric_guitar_geo.slab`, `ring_curve`, the guitar's materials) with its own `_layout`,
`_body`, `_neck`, `_hw`, `_parts`, `_mats`. It builds in 0.8 s and scores 0.009 on `form` (worst: the board's hidden faces, 0.036).

**What was measured.** One front photograph of a 1966 body showed ten fret wires; fitted with 12-TET on the 762 mm scale and a
scale that grows toward the neck (residual 0.7 mm) they gave a rectified image in metres. The fret numbering is not in the
fit (a geometric series fits any starting fret): the 12th fret's double dot and the 15th, 17th and 19th dots fixed it, and the
lower bout then measured 0.310 m against a dealer's 0.314 (x rescaled to it). The body's bass side and the bottom came out of
a colour mask of the rectified image, the shadowed treble side and the bays from column samples (a wall shadow next to an
edge is not the body: take where the body colour ends), the guard, plate, pickups and knobs as landmarks. The headstock is a
front photograph of a Fender bass headstock scaled from its 1.75 inch nut to this one's 1 5/8. The check: every outline drawn
over the rectified photograph (within a couple of millimetres), the render beside the photographs, and a flat silhouette.

| part | how |
|---|---|
| body, guard, plate | `slab`s of splines through the read points; the guard's notch round the neck has sharp corners. A long sliver of an outline (the guard's tip between the treble bay and the neck, 1.5 mm wide) breaks `fill`'s Delaunay triangulation (edges cross the sliver, the slab is not closed): keep any tip at least ~6 mm wide |
| stripes | three bands cut to the body's outline (Sutherland-Hodgman against each convex band) and laid over the face AND the rolled edge as a closed 0.3 mm film: the height comes from the slab's own quarter round at each point's distance from the wall |
| neck, headstock, keys | the guitar's skin of superellipse rings; past the nut the headstock outline (pchip) ends a little short of its tip (a ring of zero width is degenerate); clover keys are a polar outline (four lobes, the one toward the shaft pulled in) on `rounded_panel`, flat faces parallel to the headstock's |
| pickup, bridge, controls | `rounded_panel` covers and plate, swept saddles and the rolled tail, lathed knobs with metal caps, screws snapped 4 mm inside the guard's edge from their read positions |

## Worked example: the turntable

`mkmmd/blender/library/props/turntable*.py` (card `library:turntable`): a direct-drive DJ deck in SL-1200 MK2 proportions with
a 7-inch single on it, no logo or lettering. `turntable_layout.py` holds every number, the tonearm's kinematics and the drivers'
expressions (the tests evaluate those strings against the numpy maths), `_parts` the meshes, `_mats` and `turntable.py` the
Blender side. It builds in 0.5 s and scores 0.0005 on `form`.

**What was measured.** A front three-quarter photograph rectified by the top plate's four corners (a homography onto the 453 x
360 mm plate) gave the landmarks that lie on the plate: the spindle 179 mm from the left edge and centred front to back, the arm
well (56 mm in radius) round the pivot, which the data sheet's 230 mm effective length and 15 mm overhang put 215 mm from the
spindle, 24 degrees round from the right; the buttons, the knob, the target light, the slider, the adapter's well. A point above
the plate (a knob's top, the gimbal) rectifies too far back: read it at its foot. The single is RIAA / IEC 60098's 7-inch (174.6 mm,
the 38.2 mm hole, the recorded area 168.3 to 108 mm, an 89 mm label: half the diameter, as photographs of singles show).

| part | how |
|---|---|
| top plate | one `cage` of rings sampled on the same rays from the pivot: the rolled outer edge on outlines inset by each ring's offset (moving the points along their own normals lets two neighbours swap order round a corner, and the top face's long quads fold), then straight in to the arm well's lip, down its wall to a coved floor |
| platter, mat, record, adapter, knobs, feet | lathes of filleted profiles (an annulus is a closed loop, welded); knurling is the lathe's radius modulated by the angle |
| arm | a `tube` along a cubic from the gimbal into the collar, its end tangent the headshell's 22 degree offset; `rounded_panel` headshell and cartridge put in place by an explicit frame; the arm's origin on the axle, so one Euler (0, -lift, yaw) is the gimbal |
| grooves, sheen, strobe dots | drawn by the materials in the part's own frame, so they turn with it: bands of shine are a noise along the radius; the sheen is the highlight concentric grooves make (a bar along the radius toward the half vector of the view and a key light, both moved into the record's frame), which EEVEE's isotropic shading cannot give; the dots come from the angle and the height on the bevel |

Checks used: the landmarks on the rectified photograph, the deck beside the photographs at their angles (front three-quarter
with an LP, three-quarter from the left with a single on it), a straight-down view, and the stylus against the record: the
arm's lowest point is the stylus tip, on the playing surface at every `arm` past 1 and 10 mm over it with the cue up.

## Worked example: the butterfly

`mkmmd/core/butterfly.py` and `butterfly_pattern.py` (bpy-free, tested in `tests/test_butterfly.py`), `butterfly_parts.py` and
`butterfly.py` the Blender side (card `library:butterfly`; the swarm `library:butterflies` instances the same parts). It builds
in 0.5 s once the pattern atlas is cached and passes the form guard.

**What was measured.** Landmarks of the left fore and hind wings read off four top-view photographs (a monarch engraving in
the public domain, a blue morpho, a tiger swallowtail and a common blue basking, `mk ref photos`), in pixels from the thorax,
scaled by the half span: never traced. A basking butterfly holds its forewings swept back, so both wings are turned about the
forewing's root until its hind edge lies `sweep` (12-16) degrees behind square, the hindwing by a share of that turn (a set
specimen's hindwings come forward less). The hindwing's hidden leading edge is not read off the photograph: it runs a fixed
overlap ahead of the forewing's hind edge and comes out right behind its corner, so no pose opens a gap between the wings
(a test checks it). Scalloped edges are notches bitten into the spline between round lobes.

| part | how |
|---|---|
| wings | a membrane 0.35 % of the span thick: the outline inset by half the thickness, its faces ONE Delaunay triangulation of the inset ring and a lattice inside it (a tail or a deep scallop is not star-shaped: rings drawn in toward a centre fold there), a rolled edge of three rings, a gentle upward curve toward the tip; the forewing lies 0.6 % of the span over the hindwing. A per-vertex `mk_dorsal` (1 upper side, 0 underside) picks the pattern's half of the atlas |
| patterns | masks drawn from the wing's own anatomy (`butterfly_pattern.Tile`): a discal cell from near the base toward the outer edge and veins fanning from it to the edge, bands by the distance to the outline, dots and lunules by the arc length along it, eyespots on a curve inside it; four channels (sheen, accent, ink, spot) the material lays over the ground in the roles' colours, painting one clearing those above it |
| body | head, thorax and a ringed abdomen turned on a lathe, eyes, clubbed antennae, legs (four for the brush-foots) and the coiled tongue, all tubes and lathes |
| flap | each wing turns about its hinge line on the thorax (an axis along Y): mid + amp / 2 cos(2 pi phase), the body bobbing along its own +Z; drivers on the root's properties (simple expressions, checked at build) |

Checks used: the outlines over the photographs' landmarks, the patterns as flat previews against the photographs, and close-ups
from above, below and three-quarter in the project's own lighting (the undersides are dark in a scene lit from above: `glow`).

## Worked example: the spike mallet and the golden spike

`mkmmd/blender/library/props/spike_mallet*.py` (card `library:spike_mallet`) and `golden_spike*.py` (`library:golden_spike`),
tested in `tests/test_spike_mallet.py` and `tests/test_golden_spike.py`. Each `_layout` holds every number, the meshes and the
card (numpy only); the builder makes the materials. Both build in a fraction of a second and score 0.00 on `form`.

**What was measured.** The mallet: photographs of mochi pounding (`mk ref photos 'mochi pounding kine'`): a kine's head is a
plain turned cylinder with rolled rims, its slim handle passing through the head's side (the tenon's end grain shows on the
far side). The giant's head (0.46 x 0.24 m, a kine's 0.30 x 0.15 scaled up) and its frame were agreed with the move that
swings it; iron hoops near the faces, as a carpenter's beetle (kakeya) wears them, make it read as a maul too. The spike: the
1869 spike's 5 5/8 in length (the park service's figure), then a front photograph of the spike at its museum read at that
length: the shank 13.1-13.5 mm across (17/32 in), the head 25 mm wide and about 4.5 mm high over a 7.5 mm flare, the chisel's
bevel over the last 28 mm. Its engraved inscriptions are not copied: the band carries an ornament of its own.

| part | how |
|---|---|
| mallet head | ONE `lathe` about its axis, laid along X: the crowned face (a parabola), the rim rolled by `fillet` on the corner between the crown's and the barrel's tangents, the barrel (R_MID at the middle, R_END at the rims) |
| hoops | closed `lathe` loops (welded): the outside follows the barrel BAND_T over it, the inside 2 mm inside the wood, the outer edges filleted |
| handle | a `lathe` of `handle_r` (a straight shaft plus a gaussian swell at the end, a smoothstep flare into the head, the end rounded on an ellipse); its top vertices are moved onto the barrel's top, 0.6 mm proud: the tenon's end follows the head's curve |
| spike | ONE `cage` of superellipse rings: a thin slot at the chisel edge (its cap's dome puts the tip's middle at the origin), the square shank (exponent 4.5: flat faces, rounded arrises, which the box test does not count as a box), the flare (u^1.8 of the widening), the head's rolled edges; the crown's inner rings are lifted to a smooth pillow (a dome along the rounded rectangle's own rings creases on its diagonals) |
| beads | closed `tube`s round the shank's section at the band's edges, half sunk |
| materials | wood: growth rings (a wave texture's rings round a pith off the axis, its distortion stretched along the axis so the side shows straight figure and cathedrals while the faces' rings wander), end grain where the surface faces along the axis, faint rays, and a small glow of its colour; the hoops dark hammered iron (a Voronoi bump); the gold a studio reflection in its emission (the reflected ray's height picks a floor, a horizon streak or the sky) so it reads as polished metal in a black world, and the band's ornament (interlaced waves, dots, border lines) drawn from the angle round the shank and the height, cut in by an inverted bump and filled with the `engraving` colour |

Checks used: `form` in a scratch build (0.00 for both); the spike's front view beside its photograph at the same angle (head
width, flare, shank and point within a millimetre at the photograph's scale); both props alone from three sides and in
close-ups under `rose-pine-moon` and `rose-pine-dawn`; the spike standing in a tie of the library track; the mallet swung by
`pound` (both fists on the hold, its face meeting the floor on the hits).

## Worked example: the speaker stack and the penlight

`mkmmd/blender/library/props/speaker_stack*.py` (card `library:speaker_stack`) and `penlight*.py` (`library:penlight`), their
colours resolved by `showkit.py` (a slot, `ink` / `paper`, `#hex` or `a:b:t`, from the slots the props stage hands over).

| part | how |
|---|---|
| cabinet shell | ONE closed cage of rings round the depth: each ring the rounded rectangle inset by its d with the corner radius shrinking with the inset down to a few millimetres. An exact offset (`offset_rings`) past a corner's own radius turns the corner inside out, and the fan cap over such a ring shades as a dark wedge across the baffle. Support rings 4 mm beside every roll keep the big flat faces flat-shaded |
| holes | a driver's cone lies behind the baffle, so the baffle needs real holes: the drivers' holes, the ports' tunnels and the horn's mouth are cutters (`shell_object(..., levels=0, render_levels=0, cutters=..., bake_it=True)`), and `set_sharp_from_angle(35 deg)` splits the normals at the cuts' rims so the baffle round them stays flat. Each hole's edge hides under a part (a driver's frame ring, a port's flared lip, the horn's lip) |
| woofers | one lathe each, closed behind (out of sight in the cabinet); the pump is a shape key whose per-vertex weight comes from the profile (0 frame, 0..1 across the surround, 1 cone and cap), so the surround stretches as the cone pushes; a test keeps the pushed cone clear of the grille |
| grille | a welded wire mesh: straight tubes, the vertical ones in front of the horizontal ones, their ends in a perimeter wire |
| card | `layers` given from the layout: measuring them samples every surface (seconds on a dense prop) |
| penlight | lathes (the ribbed handle, the collar, the tube with its domed tip), a `rounded_panel` button stood on the front; the tube's material picks one of N RGB nodes by Math ROUND, FLOORED_MODULO and COMPARE of the driven index (all drawn by mkr) |

Checks used: close-ups of a woofer at pump 0 and 1 from the side, the stack under both palettes, Reisen's fists round the
handle from four sides (a fist closes round a 31 mm handle at curls of about 52, 74, 50 degrees). `form`: the cabinets
are boxes by nature (the stack 0.70 against its card's 0.8), the penlight 0.

## Worked example: the railroad track

`mkmmd/core/rail_track.py` (the plan, the laying schedule and its drivers, numpy, tested in `tests/test_rail_track.py`),
`mkmmd/blender/library/props/rail_track_layout.py` (the meshes and the card) and `rail_track.py` (the Blender side; card
`library:rail_track`; the build is checked in `tests/test_rail_track_build.py`): an 1860s standard-gauge track that lays itself.
It builds in under half a second for 20 m and scores 0.00 on `form` (0.03 for a two-tie stub, where the bed is the prop).

**What was measured.** Standard gauge, 1435 mm between the heads' inner faces 16 mm under the top (the test measures it on
the meshes, straight and curved); ties 8.5 ft long, 7 in thick, 2 ft apart (2640 to the mile: the Central Pacific's count);
a 56 lb/yd iron T rail of the time (4 in high, a 4 in foot, a 2 1/4 in head, 13 degree fishing slopes) in 30 ft lengths
with suspended joints and 20 in fish plates with four bolts; cut spikes with 5/8 in shanks, two a rail on every tie,
staggered, their heads hooked over the foot's edge.

| part | how |
|---|---|
| tie | ONE cage of hewn sections along its length: a flat adzed top and bottom, the log's rounded sides (a pole tie, not a sawn box: its sides carry no flat area, so `form` sees no cuboid), the top corners' wane growing and shrinking along it, small random bows and widths, the end edges rolled; six seeded variants shared by every tie, its top's high points at z = 0 where the rails sit |
| rail span | the rail's section (filleted foot, web, head, crowned running surface in its own material) carried along the track by `along`: every ring placed exactly square to the arc (a sweep's chord frames tilt the end rings on a curve), the ends capped by ear clipping (a T is not star-shaped: a fan cap folds outside it). Built at span 0's place about the circle's centre, so turning the object about the centre slides it along the arc |
| bed piece | the bed's profile carried one tie spacing, open underneath (a flat hidden underside with sharp rims scored 0.49 alone); both ends 1 : 1 ramps capped separately (the sides' smooth normals run on into the next piece with no seam) and domed 6 cm forward, so the laying front shows a heap's rounded face and no large flat panel, and the next piece's start is the same surface |
| spikes, plates, bolts | `rounded_box` heads and shanks (tiny parts), `rounded_panel` tie plates with low shoulders, hexagonal lathes for the bolts |
| card | `bounds` and `layers` from the layout: measuring them samples every triangle down to 3 cm, minutes on a track's 9 m rail faces |
| materials | gravel from a world-space Voronoi (one colour per stone, domed by the bump; world space so the pieces join seamlessly), wood streaked and checked along the tie in object space offset by Object Info Random, iron with a rust mix; all colours from roles |

Checks used: the schedule's invariants as tests (nothing past `laid`, rails only over landed ties, clean rests on every
tie spacing, whole from `done`), the drivers evaluated against their numpy twins, the built scene against the schedule on
several frames, and look sheets of a track laid a tie per hit and a rail per hit.

## Worked example: the mile post

`mkmmd/blender/library/props/mile_post*.py` (card `library:mile_post`) and `mkmmd/core/numerals.py` (the numerals, numpy),
tested in `tests/test_mile_post.py`: `mile_post_layout.py` holds every number, the meshes, the options and the card,
`mile_post_mats.py` the materials. A post builds in well under a second and scores 0.11 on `form` (railroad) and 0.00
(interstate), with no `form_max`.

**What was measured.** The railroad post: the Southern Pacific's common standard mile post drawing (a 6 x 6 in post;
numerals 10 in tall and 5 in wide with a 1 5/8 in stroke: half as wide as tall, the stroke 0.16 of the height) and
photographs of old posts (`mk ref photos 'milepost'`, railfan pages): white paint, a hipped point, a tarred foot about a
sixth of what stands out, stains running down from the top. The interstate marker: MUTCD 2009 Table 2H-1 and 2H.05 (a
freeway reference location sign is 12 in wide with 10 in numerals, 24 / 36 / 48 in tall for 1 / 2 / 3 digits under the word
MILE; without it each numeral takes 13 in: 17 / 30 / 43 in), the Standard Highway Signs D10 layout (3.5 in over and under
the legend, 3 in between rows, 1.5 in corners, a 0.5 in border), FHWA Standard Alphabets 2000, Series E (numerals 0.81 of
the height wide, the 1 0.30, the 4 0.94, the 0 0.84, the stroke 0.1725) and a 3 lb/ft U-channel post (3.5 in across the
flanges, 1.75 in deep, a 1.625 in bearing face, 0.16 in steel, 3/8 in holes on 1 in centres); photographs of markers on
I-587 and I-75.

| part | how |
|---|---|
| numerals | `core.numerals`: each digit drawn once in the unit box of its centre lines (lines, elliptical arcs, cubic Beziers, superellipse loops; a stroke is tangent-continuous and a corner starts a new stroke, so the round caps make a round join), mapped onto the ink box of the width and stroke asked (`condense` narrows a digit without thinning its stroke: 0.70 for the plate's condensed gothic) and resampled evenly. Drawn by eye beside the Series E sheets from landmarks (where the 2's spine lands, the 3's middle bar, the 4's crossbar at 0.28, the top of the 5's bowl, the loops' sizes), never traced |
| relief numerals | the stroke's section (a half superellipse on a foot sunk 0.8 mm into the plate) swept along every stroke; an open stroke runs on half a stroke past each end while its section narrows on a quarter circle (a round end in plan); a loop is a closed sweep, welded shut |
| sheeting legend | flat `ribbon`s the stroke's width with half-disc caps, and the border a closed ribbon 0.375 in in from the edge, 0.3-0.45 mm over the face: graphic layers, `exempt` |
| post | ONE cage of rings: the chamfered square's section with its eight corners eased (`fillet`, 3 mm) and three points on each face per season check (its lips and its floor); up the faces the floor sinks to the check's depth and back (smoothsteps over 0.16 m), then the shoulder (a 6 mm fillet) and the hips straight up to a rounded point. A square post is a box to the `form` test (two strong axes): its 1 1/8 in chamfers and the checks' walls take enough surface off the axes (0.11; chamfers alone needed about 1 3/8 in, an octagon). The check walls lean 59 degrees (6 mm wide, 5 mm deep): at 70 the turn passed the test's 65 degrees and the faces counted as sharp-rimmed panels (0.20) |
| plate, panel | `rounded_panel`s with both edges rolled (an unrolled back is a sharp-rimmed flat patch); the plate's bead a closed sweep of a half round, the bolts lathes; the panel's face takes the sheeting by its normal, its back and edges bare aluminium |
| U-channel | the sheet's middle line (flange, side, web, side, flange) with rounded bends, thickened into an outline and swept up Z; its ends closed by the strip between the sheet's two sides (the section is not star-shaped: a fan cap would cross it). Its sides lean 10.6 degrees, past the box test's 10 |
| materials | the roles' colours in RGB nodes, the weathering in masks of the object's own coordinates and normal: a grain (a noise stretched along Z) through the paint, rain streaks and stains from a greyed top, blotches of lost paint (more toward the ground) showing the bare wood, dirt up the foot, the tarred foot's ragged edge, the checks darkened by the post's `mk_check` attribute (how open a check is at each ring point); painted cast iron with rust specks; galvanised spangle (Voronoi cells) and the holes drawn from the web's coordinates; sheeting glowing a little of its colour, as retroreflective sheeting does |

Checks used: `form` in a scratch build (railroad 0.11, interstate 0.00), the post beside photographs of old posts at their
angles and the marker beside a photograph of a marker on its road, both styles side by side under `rose-pine-moon` and
`rose-pine-dawn`, blank plates written by `[[text]]` on their `plate` surface, and a frame drawn by mkr beside EEVEE's.

## Reviewing

* Render the prop alone from named views and set it beside the reference photo at the same angle and lens: front 3/4
  left, rear 3/4 left, side, front, and close-ups of every detail (`mk look --view yaw:elev --target ... --lens ...`).
* A proportion error is a silhouette error: fix it with the profile curves, not with details. For a side profile render the
  prop with an orthographic camera at the photograph's scale (the length and wheelbase give pixels per metre; the wheel
  centres give the ground line) and lay its outline over the photo: a few centimetres of hood height, glass rake or sill
  height show at once, where a perspective comparison hides them.
* Run `mk check` (the `form` metric): no cuboid parts, no unbevelled hard edges on the visible parts of the prop.
* Keep the build under about 3 s per prop; evaluate `levels` 2 first, add the render level last.
