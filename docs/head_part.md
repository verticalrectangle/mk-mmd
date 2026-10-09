# The head part (`mkmmd/model/parts/head*.py`) and the PMX importer (`mkmmd/model/pmx_take.py`)

One part, `head`, built by `mk model build` after the body (`needs = ["body"]`) and before hair and outfit. It makes the skin of the head
(face, eyes, mouth, brows, lashes, ears), the face morphs, and publishes what the hair and the outfit fit to (`Part.info`: closed
`skin` shell, `hairline`, `hairline_side`, `nape`, `ears`, `cat_ear_anchors`, `eyes`, `brows`, `neck_ring`, `landmarks` incl. `eye.L/R`,
`head_center`, `skull_top`, `chin`, `mouth`, `lid_margin`, `face_outline` ...). Everything is bpy-free (numpy, PIL) and driven by
`[head]` in `head.toml`. Two modes:

| `[head] source` | The face is | Modules |
|---|---|---|
| `"param"` (default) | generated: an implicit skull + grid skin with eye / mouth openings, anime lids that close by sheets and strips (the skin never moves for a blink), 28 morphs, own textures and toon | `head.py`, `head_shape/skin/eye/lid/mouth/brow/ear/decal/morph/tex/info.py` |
| `"body"` | imported with the body: the body part cuts face + body materials out of a PMX with `pmx_take.take` and publishes the `Take` as `info["take"]`; the head part builds the face meshes from it | `head_body.py`, `head_edit.py`, `head_cap.py`, `head_shell.py` (+ `head_ear.py`, `head_info.py`, `head.published`) |

## The importer: `mkmmd.model.pmx_take`

`take(path, materials, *, unit=0.08, scale=1.0, head=None, skip_morphs=(), bodies=False, keep_bones=(), drop_bones=()) -> Take`
reads a PMX (`pmx_io`) and returns, in mk model space (metres, Z up, -Y forward, +X her left; faces CCW outward; uv v UP):

- `verts normals uv edge src_ids` for the vertices the kept materials use (shared vertices stay shared), `faces[material]`;
- `weights {bone name: (n,)}` (rows sum to 1);
- `materials {name: TakeMaterial}` with the MMD settings as in the file and absolute texture / sphere / toon paths;
- `bones` (`part.Bone`, parents first): the bones the kept vertices use and their ancestors, plus every bone no vertex uses (control, IK,
  D, twist, tip), minus the bones that only serve dropped materials; IK targets / links and grant parents are closed over;
- `morphs`: vertex morphs restricted to the kept vertices (offsets scaled), material morphs restricted to the kept materials, group morphs
  restricted to surviving members; `bodies` / `joints` on kept bones when asked; `frames`; `info` (counts, dropped bones, pivot).

`scale` is a uniform scale about the origin (the floor). `head = {"bone": name, "factor": f}` then enlarges the head: each vertex goes to
`pivot + (p - pivot) * (1 + w (f - 1))`, `w` its weight on the bone's subtree (the bone and its descendants), `pivot` the scaled bone head;
the bones of the subtree get the full factor, morph offsets the same per-vertex factor, so the neck blends. `Take.piece(names)` gives
renumbered arrays for some materials (what a part makes a `Mesh` of). `part_materials(take, ctx, names, recolor=None, hooks=None)` makes
`part.Material`s and writes the (edited) texture copies with `ctx.save_png`; identical results are shared between parts.
`pmx_recolor.recolor` edits a copy of an image: hue / saturation / value maps over a rectangle (`protect` keeps whites), `warp` (scale a
painted shape about a centre) and `almond` (soft vertical lens in alpha). Rules and hooks are matched by the CONTENT of the source file.

Nothing about a particular model is in the repo: the PMX path, the kept materials, the scale and the colour edits are in the project's
`body.toml` / `head.toml`. The built model and its edited textures stay local.

## `source = "param"`: the generated face (`head.py`, `head_shape.py`)

The head is an implicit shape sampled by rays from the skull centre (`head_skin.Grid`). `head_shape.HeadShape` is the skull loft
(`ANCHORS`: widths at named heights, the face plane's midline depths, the back of the skull; `[proportions.head]` and `[head.shape]`
override them); with a `face` it adds the face's features, `FACE` merged with the spec's `[head.face]` (tables merge key by key):

| Key | Feature |
|---|---|
| `profile` | the midline depths of the ridge the face's planes meet at (Profiles anchors plus `front_extra` `(z, y)` knots): forward at the muzzle and under the nose, back where the eyes sit, forward again at the brow and the forehead; `loft_bottom` lets the loft go on under the chin point, so the jaw shapes the chin's underside |
| `front_n` | `(z, n)` knots of the exponent of the box the planes cut: square, so the sides stay full out to where the planes turn back into them (the cheekbones); squarest at the eyes, so the temples stay forward past the eyes' outer corners and the face turns onto the side of the head beyond them |
| `planes` | each half of the face is a plane, receding from the midline ridge to the depth `side` (y) at `x_ref`; `knots` `(z, side, round)` by height (`round`: the ridge's rounding either side of the midline), faded flat over `fade` above the top knot, `corner` the rounding into the box's sides. The side line is smooth in z whatever the ridge does (the slope is derived from both): it comes forward over the cheeks, goes back where the eyes sit and forward again at the brow |
| `nose` | `tip (y, z)` (default `[proportions.face]` or `[proportions.head]` `nose_tip`, model space; met exactly, over the other features), `root_z` / `base_z` where it rises out of the bridge and tucks into the upper lip, half `width` at root / tip / base, `round` (the point), `bridge` (> 1: a concave line), `under` |
| `lips` | midline bumps over the mouth line `z`, tapering sideways as exp(-(x / `width`)^2.5): from under the nose to the lip's edge the profile is one straight slope a little in front of the line from the nose's tip to the chin (no pout): the upper lip and the space under the nose filled up to that slope; under the mouth line (folded in, see Mouth) the lower lip comes out again, with a little fullness under it before the chin |
| `bridge` | the ridge stands `height` proud of the planes, `width` half wide, from `bottom_z` (into the nose's tip) up the brow, fading over `fade` below `top_z` as the forehead comes forward |
| `jaw` | the underside: from the chin's lowest point (`chin`, default the `chin_point` anchor) back to the jaw angle at `angle_y`, rising `rise_deg` on the midline (low and nearly flat under the chin, so the neck meets it low) and `lateral` x² towards the sides (seen from the side the jaw line climbs from the chin to the ear: a keel, not a box); behind the angle `behind` more, then falling away at `fall_deg` under the back of the skull, whose own underside curves into the nape; `round` is the edge where the face meets it. The underside must face away from the ray centre (no overhang steeper than about 50 deg) |
| `neck_k` | the fillet where the neck column meets the jaw |

Why planes: a plane lights evenly under a toon ramp, so the cheeks, the temples and the sides of the muzzle read as clean drawn
areas, and the nose is the end of the ridge where the planes meet, not a shape stuck on the face (from the front it is a hint).
The eyes sit on the planes, their outer corners further back than the inner ones. The face's front is the box cut by the planes
(`HeadShape.hull`, a smooth maximum); the nose, lips and bridge displace it along y (a relief); the jaw is a rounded cut.
`HeadShape(face=None)` is the plain skull, the prior an imported face's cap relaxes to (`head_body`), unchanged by the face's features.
`HeadShape.jaw_side(P)` is negative on the jaw's underside and positive on the face, zero along the middle of the rounded edge.

- **Grid** (`[head.grid]`): rows `spacing_face` apart along the front profile (finer `detail` rows where the nose is), columns
  `face_deg` apart in front and `mid_deg` within `mid_to` degrees of the midline (the nose, the mouth's middle); `n_cols = None` takes as
  many columns as the spacings ask for. The mouth block is the midline row nearest the mouth on the face (below the chin the midline
  comes back up the throat) and its slit points follow the block's columns, so the ring lines never cross.
- **Shading normals** (`skin_normals`): the gradient of the loft and the planes (no nose, bridge or lips shading: from the front the
  nose is the painted shade under its point, its outline in profile), leaning `cyl` of the way to its horizontal part below the eyes
  so the cheeks and chin stay lit towards the jaw; the true normals on the underside (where it faces down) and the neck. Further from
  the surface than that, shadow maps band on the lower face.
- **Texture map** (`head_tex`): cylindrical, `u` = azimuth; the face by height (`face_v`), the skin under the jaw and the neck in a strip
  by `jaw_side` (`under_v`): the chin and the throat behind it share azimuth and height. The strip is in shadow up to `SHADOW_EDGE` onto
  the face from the middle of the jaw's edge (with the body's neck-shadow weight per azimuth), so the drawn jaw shadow follows the
  jaw's edge; faces within `under_jaw`'s `reach` of the edge use it, so the seam between the two parts lies on unshaded skin. The
  nose's shade sits under its point.
- **Mouth**: the closed mouth line is folded into the face (`head_mouth` `fold`, `fold_share`): the slit lies `fold` back at the
  middle, (1 - (x / half_width)^2)^2 of it sideways, and the rings round it take their share, so seen from the side the upper lip
  ends in an edge over a notch and the lower lip comes out under it (a line painted on a smooth face disappears in profile; a
  fold catches the outline and the shadow). The mouth shapes move the lips in z and keep the fold's depth; the corner shapes
  (口角上げ / 口角下げ, the corners' share of い, え, にやり) are bumps round each corner, the middle of the lips left still. The
  interior bag's inner rings stay behind the skin in front of them at rest (`cavity(..., shape)`) and in every mouth morph
  (`build_mouth_morphs`): they follow the lips further than the skin under the lower lip does. The 照れ blush patches wait
  1.2 mm under the cheeks and follow the skin through every mouth shape (each vertex the mix of its four nearest skin vertices'
  offsets), so a shape that pulls the cheeks in (ω) never shows them.
- **Eyes** (`head_eye`, `[head.eye]`): an anime eye, not an eyeball. The opening (`CORNER_IN/OUT`, `UPPER`, `LOWER`): the upper lid
  a flat-topped arch that comes down an almost vertical outer side into the outer corner at mid-height, the lower lid a round bowl
  rising into it; seen from the side the lower lid runs straight back and the eye's outer end tucks into the head. Each lid moves
  steadily outward (u never decreases along it), so the lid sheets and the closed-eye drawings are functions of u; everything along
  the lids is placed by the lid parameter `lid_s` (0 at the inner corner, 1 at the outer). The lid margins and the lid sheets ride on
  the lid shell (`LidShell`), which follows the face: `centre_depth` behind it over the upper part of the eye, so the upper lid's
  edge sits on the face (a shell curving away above the eye would sink the skin round it into a crater, whose rim stands out over
  the eye), and `lower_depth` behind it at the lower lid (the tuck starts `tuck[0]` below the chord between the corners and is
  complete `tuck[1]` further down), so the lower lid tucks in behind the cheek. Round the opening the skin sinks into a socket
  (`Dip`) that fades out over `dip_length`; on the nose side `inner_recess` more sets it back, deepest half way out and flat at
  both ends, so the skin arrives at the inner corner facing forward instead of diving into it (it dropped 4 mm over the last
  6 mm). The sheet columns are spaced along the lids, not evenly in u. Behind the opening the white is a **pocket**: a rounded
  bowl from the inner lid rim (straight back from the margin) to
  `pocket.back` behind the iris, opening out by `pocket.undercut` behind the lids (the iris moves inside it) and always
  `pocket.clear` inside the head. The iris and the pupil are flat discs on the **iris plane** (`IrisPlane`, `iris_plane`),
  `iris_depth` behind the lid margin and turned `iris_yaw_deg` outward like the face, pushed back only as far as covering them
  needs: by the skin where they turn outside the opening (every `gaze`) and by the closed lid sheets (`SHEET_GAP` over the shell)
  inside it (every `blink_gaze`), each by `iris_clear`. Nothing covers them in the open eye, so they sit about as far behind the
  face as drawn eyes do (about 5 mm): from the front a drawn circle, from the side mostly iris with a crescent of white, not a deep
  white bowl. The highlights float `highlight_gap` in front of the iris on their own bones `左ハイライト` / `右ハイライト`, granted
  `highlight_follow` of the eyes' (`両目`) rotation: a reflection lags behind the gaze and shifts against the iris as the head
  turns. All eye layers carry the iris plane's normal (flat, evenly lit).
- **Lines** (`upper_lines`): the upper lash band starts in a point beside the inner corner (`UPPER_LASH` `tail`), meets the upper lid
  where it rises through `peel_v`, runs along it, turns at `turn` up to its top-outer corner (`elbow`, met and left at the angles
  `elbow_in` / `elbow_out`) and comes back down the eye's outer side over the lower lid to a point (`out_v`, `out_tip` degrees sharp);
  its lid parameters run below 0 along the tail and on to 1 + `OUTER_RUN` down the outer side. Two blades lie over it: the fork, a
  second point at the inner end (`LASH_FORK`: `length` and `angle` from the notch between it and the tail), and the wing on the
  top-outer corner (`LASH_WING`: `root` and `base` on the band's top edge between the turn and the elbow, `length` and `angle` from
  their middle); the inner lid rim under the band is lash-coloured. The crease (`CREASE`) is the double-eyelid line: a stroke over the
  inner half, pointed at both ends. Curves between these points leave and arrive along the given directions (`_hermite`), so every
  shape is set by points, lengths and angles. The lower lash lies under the bottom of the lower lid, heaviest towards its outer end;
  brows are crescents pointed at both ends. The closed-eye drawings (`head_lid`) start in a point `CLOSED_TAIL` from the inner
  corner and end in a flick down and out (`CLOSED_FLICK`, the band's run down the outer side), rounded over about 1 mm where they
  turn; the fork and the wing fold into the drawn line, the crease follows it (fades in hau). An eye that stays open (the moods'
  lowered lids, surprised) moves everything above it with the band: the band's lower edge onto the lowered lid, the rest as far
  as the lid drops at its lid parameter, held at the band's `hold` (where the band meets the lid, its turn), so the tail, the
  fork, the elbow and the wing move rigidly; a lowered lid leaves the eye 1.6 mm open, closing into the corners. `clear_skin`
  lifts a strip at most `max_lift` off the skin (at a hole's edge the lift can otherwise run away); a morph places strips on the
  lid shell inside the opening (`DISPLAY_GAP`, clear of the sheets' faces) and on the skin outside it, never behind the skin and
  `OUT_LIFT` further off it, and `clear_points` raises a target whose faces would still cut into the skin (they span several mm,
  and the skin round the eye's outer corner curves away).

## `source = "body"` (`head_body.py`)

`[head.take]`: `materials` (the face materials; one mesh each, the skin one is named `face`), `skin`, `head_bone`, `brow_bone` /
`brow_material`, `eyes {left, right, iris}`, `cap {rings, fade, pole_gap}`, `patch`, `shell_deg`, `[[head.take.recolor]]` rules.

An imported face is a front shell with an open back and top. `head_cap.build_cap` closes it with rings that start exactly on the shell's
outer boundary and relax to the generated skull (`head_shape`): hidden under the hair, and what the hair is fitted to. A radial map of the
closed shell (`head_shell.RadialShape`, same interface as the implicit shape) answers the hairline, ears, cat-ear anchors and face-outline
queries. The cap's colour is a generated gradient patch painted into a free corner of the skin texture copy (it starts in the colours
the shell ends in). The neck needs no stitching: the face skin and the body skin share the ring vertices (same weights, normals).

### Edits (`[head.edit]`, `head_edit.py`)

Small smooth deformations that keep the morphs working; each sub-table has `enabled`, and `[head.edit] enabled = false` turns all off.
Landmarks are found on the mesh (the skin's holes, the iris weights); positions are rotated / moved together with the morph offsets.

| Table | Edit |
|---|---|
| `eyes` | outer corners up by `tilt_deg` (default `[proportions.face] eye_outer_tilt_deg`): the skin and the listed `pieces` (white, lashes, closed-eye pieces) turn about each opening's centre, fully within `plateau` eye widths, falling to nothing at `reach`, nothing at the midline; brow-bone vertices stay; the iris stays round; **morph offsets turn with their vertices** so blinks / winks still close onto each other |
| `jaw` | the rest skin moves outward along its normals by `amount` at the lower cheeks and jaw line (zero at eyes, nose, lips, chin point, neck, back); morph offsets unchanged |
| `fang` | a yaeba as its own mesh `fang` (a white triangle in its own pure-white material `牙`, an additive white sphere map and full ambient as the eye highlights, so it stays white in the shadow of the lip, + a slightly larger pink `outline` triangle behind it, material `牙線`: a white tooth on pale skin does not read without it): tilted so its base hides in the upper lip and its tip hangs `peek` below the lip's lower edge, `over` in front of the skin there (the lower lip), found by a ray cast; in every vertex morph it rides the lip (the nearest upper-lip vertex's offset is copied rigidly), and `reveal = {morph = factor}` pulls the tip further down / forward in those morphs (`reveal_down`, `reveal_forward`): a hint at rest, reading in the smiles |
| `highlight` | the main highlight of each eye scaled in x by `scale_x` |
| `pupil` | texture rules applied after `[[head.take.recolor]]` (`almond` pupil, `warp` of the painted highlight) |

## Tests

`tests/test_model_pmx_take.py` (importer, recolour) runs on a tiny synthetic PMX (pruning, head blend, morph / material / group
carry-over, bodies, `part_materials`), `tests/test_model_head_body.py` (cap, radial shape, the builder on a synthetic take, edits on it),
`tests/test_model_head_edit.py` (each edit and the texture ops), `tests/test_model_head.py` (the generated mode). None of them needs
a real model.
