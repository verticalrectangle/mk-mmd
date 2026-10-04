# Playbook: making a video with mk

For agents (and people) driving `mk`. Every command prints JSON; exit codes are 0 ok, 1 a check failed, 2 usage,
3 Blender/runtime error. Read `docs/design.md` for the contracts behind what follows.

## The loop

1. **Know the model.** `mk inspect MODEL.pmx` (or `mk assets add MODEL.pmx --kind model`) and read the summary:
   `kind`, `quirks`, chain `families`, `expressions`, `measure`. Quirks are things that will bite later (missing
   bones, non-bone vertex groups, locked physics joints, missing textures).
2. **Register and clear licenses.** `mk assets add` for everything you use, then `mk assets show SLUG --terms`,
   read the readme lines, and fill in `license`, `restrictions` and `credit` with `mk assets set`. Never guess a
   license; if a readme only links a page, record the link and say so.
3. **Set up the project.** An `mk.toml` with `[project]` (fps, frame0, duration, blend), one `[[output]]` per
   aspect, the `[[cast]]`, collider sets and the checks that define "done" (see the example below).
   `mk timeline analyze` turns the song span into `audio/timeline.json` (beats, downbeats, loudness, word timings by
   `(line, word)`); `mk timeline show` prints its numbers. Pick shot boundaries on downbeats from it.
4. **Measure real people (optional, for performances).** `mk ref search "woman singing in car"`, `mk ref add ID...`,
   `mk ref track`, `mk ref measure` give blink, gaze, nod, sway and mouth numbers shaped like `[perform.<cast>]`, each
   with a confidence; `mk ref sheet` shows what was tracked; `mk ref clean` frees the disk. Prefer front-on or
   three-quarter clips (side-view faces are rarely found), read `qc.warnings`, and treat values with low `n` as
   hints.
5. **Build, then measure.** `mk build`, then `mk check` runs every check in one Blender pass. Fix what fails; never
   loosen a threshold to make a check pass without saying why.
6. **Look.** Numbers miss things. `mk look --frames ... --sheet` for the cut in every aspect; `mk look --view
   front,left,back --target 'bone("wrist.R").head' --dist 0.4` to inspect a detail from all sides. Look at the
   images before claiming anything about how a shot looks.
7. Repeat 5-6 until checks pass and the sheets look right, then `mk render --preset draft`, `mk post --preset
   draft`, and the final render once the draft is approved.

## Asking questions of a scene

```sh
mk q scene.blend --list bones                         # names: Blender, PMX and semantic
mk q 'bone("head").head' --frames 181:280             # any expression, frame by frame
mk q 'dist(bone("index_tip.R").head, obj("Pen").loc)' --frames all --summary
mk serve scene.blend &                                # big scene: keep it loaded, q/sample answer in < 1 s
```

Bones can be named by their semantic name (`head`, `wrist.R`, `index2.L`, see `mkmmd/core/bonemap.py`), their
Blender name (`手首.R`) or their PMX name (`右手首`).

## Checks worth having in every project

| Want | Check |
|---|---|
| Hair, ears, tails do not shake | `jitter` per family, `max = 0.5` (side locks that swing: 0.8) |
| Hair stays out of the body and props | `penetration` with the project's collider set, `max = 5` (mm) |
| Feet do not skate | `foot_slide`, `max = 1.5` (mm per frame) |
| No broken elbows, knees, necks | `joint_limits`, `max = 0` |
| A hand really holds or touches something | `contact` between two points or a point and a track (see Grips below) |
| The subject is in frame in every aspect | `framing` per shot (frames of that shot), `min = 0` |
| Nothing blocks the subject | `occlusion`, `max = 0.3` |
| No near-black, stays in palette | `palette` on rendered frames |

## Grips

Hands that hold things are built, not placed: `grip = "car:wheel"` (or `"rest"` with `rest = "car:sill_R"`) in
`[pose.<cast>.hands.L]` solves the fingers on the model's own skin (docs/design.md: Grips), where `at` plus a finger
preset only guesses. Read the build log: `WARNING ... short of its goal` means the seat is out of the arm's reach
(`lean`, move the seat or bring the prop closer), a grip digest past 3 / 1 / 1 mm (gap, penetration, clash) means
the prop does not fit that hand. Pin it with a `contact` check between the fingertip bone's `.tail` and the prop.

## Modelling props

A prop assembled from boxes reads as a toy even when every dimension is right. For anything built in code (a library prop,
a vehicle, furniture, a set element) follow this loop, and do not call a prop done before step 6:

1. **References first.** `mk ref photos 'Dodge 600 convertible' --set dodge600 --n 12` fetches licensed photographs from
   Wikimedia Commons into `refs/dodge600/` (outside a project: `~/.cache/mk/ref/photos/dodge600/`) with `SOURCES.json`
   (licence, author, page) and `contact_sheet.jpg`. LOOK at the sheet and pick the angles that matter (front 3/4, side, rear
   3/4, top, interior, the detail you will model). Ask again for what is missing (`'Dodge 600 dashboard'`, `... wheel`):
   the same set takes more. The photos are visual references only: never textures, never in a render, never traced.
2. **Real dimensions.** Write down the true size and the proportions that matter (length, width, height, wheelbase, seat
   and sill heights; a mug's height against its width) before the first vertex. A data sheet or the Wikipedia page beats a
   guess; the photo decides what the data sheet cannot say (the slope of the hood, the radius of a fender).
3. **Build forms from profiles, lofts and subdivision with creases.** Draw a section, pull it along a path or between
   stations, let the Subdivision Surface round it, and put a crease only where the real part has a line. The toolkit is
   `mkmmd/core/shell.py` (`loft`, `cage`, `sweep`, `lathe`, `rounded_panel`, ...); the rules and recipes are in
   [docs/modelling.md](modelling.md).
4. **Bevel every visible hard edge; never stack cuboids for visible forms.** Real surfaces have radii, and the highlights
   live on them. A box, a slab or a plate with a 1 mm bevel is still a box: round it, crown it, or build it as a loft. Parts
   that really are flat graphic layers (a label, a display) are tagged `mk_form_exempt` (`shell.exempt(obj)`).
5. **Measure with `form`.** A `[[check]]` per hero prop, run after every build:

   ```toml
   [[check]]
   name = "car form"
   metric = "form"
   args = { prop = "car", exclude = ["car_underbody"] }     # exclude: objects nobody sees; hidden colliders never count
   max = 0.25                                               # the default for hero props
   ```

   `value` is the boxiness (0 smooth .. 1 boxes). Read `detail.worst_parts` first: each entry names the object, the part
   inside it, its area, its share of the prop, `why` (`a box: ...` or `flat panels with sharp edges`), its size and where
   it is in the prop's frame; `sharp_panels` lists the biggest flat panels with unbevelled rims. Fix the top entry,
   rebuild, check again: a prop usually needs three or four rounds. Furniture and gadgets that are boxes by nature (a
   boombox, a nightstand) may be judged at `max` 0.5-0.7 once you have looked at them; walls, gantries and barriers are
   `exempt = ["wall*"]` (measured and listed, not scored) or `exclude`d. Do not raise `max` to make a hero prop pass.
6. **Compare side by side and look.** `mk look build/scene.blend --frames 1 --view 35:12,90:8 --target 'obj("car").matrix @
   Vector((0, 0, 0.6))' --dist 7.5 --ref refs/dodge600/front_left.jpg,refs/dodge600/side.jpg` writes `ref.jpg`: photo i
   beside view i at the same height. Choose `yaw:elev` to match the photo's angle (the photo's horizon and perspective tell
   you), then compare silhouette and proportion first (wheelbase, hood length, roofline, stance), surface detail second.
   Open the image. A passing `form` number means no boxes, not the right car.

## Reading results

- `jitter` with a `static` note: the chain barely moves and the ratio is float32 noise; judge by `jerk_p95_mm`.
- `penetration` is measured on the baked bones (what renders), with each chain's own radius. It reports the frame,
  bone and the shape it is in (`body:<bone>` for the model's own bodies). Body capsules are fatter than the mesh, so
  a few millimetres rarely show; look before fixing.
- `joint_limits` does not flag sideways elbow bends: MMD rigs treat the elbow as a ball joint and professional
  motions put upper-arm rotation there. It flags fold-through, in-plane hyperextension, knees bending the wrong way
  or sideways, wrists past 110 degrees, and necks and spines past what dances do.
- `framing` margins are fractions of the frame; negative means outside the safe area. Configure it per shot: a page
  close-up is not supposed to contain the head.

## Debugging

- `MK_KEEP_JOBS=1 mk ...` keeps `~/.cache/mk/jobs/<id>/` (job, result, Blender log) even on success.
- `mk check --keep-sample` keeps the sampled `.npz` and its JSON reply in `~/.cache/mk/samples/` for your own numpy.
- A failing Blender job prints `error`, `log` (path) and `trace`; read the log before retrying.

## Pitfalls

- Timeline markers switch cameras every frame; `mk look` removes them for debug views and keeps them for the cut.
- mmd_tools gives bones without a tail target a 0.08 m tail; never use bone tails as physical lengths.
- mmd_tools adds `_dummy_` / `_shadow_` helper bones and non-bone vertex groups (`mmd_edge_scale`,
  `mmd_vertex_order`); rig.json marks the former and quirks list the latter.
- Many PMX hair joints are fully locked (zero angular range): Bullet swings such chains as rigid sticks.
- Never print lyric text anywhere: refer to words as `(line, word)`.

## Example `mk.toml`

```toml
[project]
name = "cafe"
fps = 30
frame0 = 181
duration = 24.6
blend = "build/cafe.blend"

[[output]]
name = "9x16"
size = [1080, 1920]

[[cast]]
name = "reisen"
armature = "Reisen_arm"
asset = "miy_reisen"

[colliders]
cafe = [{ type = "box", object = "ChairBack", rnd = 0.012 }, { type = "fingers" }, { type = "floor", z = 0.0 }]

[[check]]
name = "back hair is calm"
metric = "jitter"
args = { family = "back_hair" }
max = 0.5

[[check]]
name = "hair stays out of the body"
metric = "penetration"
args = { colliders = "cafe", radius = { scale = 0.5, max = 0.02 } }
max = 5.0
```
