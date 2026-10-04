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
