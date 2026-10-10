# Playbook: making a video with mk

For agents (and people) driving `mk`. Every command prints JSON; exit codes are 0 ok, 1 a check, gate or verification failed, 2 usage,
3 Blender or runtime error. [design.md](design.md) is the reference for the contracts and every `mk.toml` key behind what follows;
[modelling.md](modelling.md) is how props are modelled. Times in `mk.toml` are clip seconds, frames are Blender frame numbers.

## Rules that never bend

- Never print or quote lyric text: not in code, logs, briefs, commit messages or agent transcripts. Refer to words as `(line, word)`.
- Never commit models, motions, audio, lyrics, reference clips, reference photos or renders; projects keep them in their own folders.
- Never guess a licence. Read the readme lines (`mk assets show SLUG --terms`), record what they say, and say so when a readme only links a page.
- Never loosen a threshold to make a check pass without saying why. Never call a shot good before you have looked at the images.
- Read the build log. `WARNING` lines are findings (a hand short of its goal, a prop over its form limit, a pattern matching nothing).

## The loop

| Phase | Do | Commands |
|---|---|---|
| 1. Brief | the song span, the outputs, the story, the palette, the cast, the places, the effects | |
| 2. Timeline | beats, downbeats, loudness, note onsets, word timings | `mk timeline analyze`, `show`, `onsets` |
| 3. Storyboard | shots on downbeats, the effect at each cut, what each output shows | |
| 4. Cast, sets, props | register and clear assets, model what is missing, place everything | `mk assets`, `mk inspect`, `mk ref photos`, `mk build --until props` |
| 5. Pose, perform, sim | seats, grips, gaze, blinks, lip sync, hair in the wind | `mk build`, `mk grip`, `mk ref measure` |
| 6. Shots, effects, type | cameras per output, looks, transitions, inserts, lyrics and signs | `mk build --until text` |
| 7. Checks | numbers that define done | `mk check` |
| 8. Look sheets | the cut in every output, details from all sides | `mk look` |
| 9. Render and post | draft, preview, final; grade and encode; review | `mk render`, `mk post`, `mk play` |
| 10. Credits | every asset named, nothing unreviewed | `mk assets credits` |

Go back as often as needed: build, check and look are cheap (seconds); render and post are not. Repeat 5-8 until the checks pass and
the sheets look right, then render a draft, post it, and render the final once the draft is approved.

## 1. Brief

Write down before touching a file: the song and the span (`start` in song seconds, `duration`), the outputs (`16x9`, `9x16`, `1x1`: each
gets its own camera for every shot), fps (30), the palette ([Palettes](design.md#palettes)), who is in the scene, where, with which
props, and the effects you want. Keep the brief free of lyric text. Choose `frame0` so that at least 3 seconds (90 frames at 30 fps)
come before it: that pre-roll is where poses ease in and hair settles (`[scene] settle_frames`, default 24).

## 2. Timeline

`mk timeline analyze song.flac --start 15.59 --duration 30.6 --out audio/timeline.json` (or just `mk timeline analyze` in a project with
`[audio] file` and `start`) turns the song span into beats, downbeats, per-frame loudness, note onsets and word timings by
`(line, word)`; `mk timeline show` prints its numbers. Pick shot boundaries on downbeats from it. A lyrics file (`--lyrics`, one sung line
per line) beats Whisper's line breaks. For a strummed or plucked part, `mk timeline onsets` recomputes the onsets of a Demucs stem. If
the transcription is wrong the words move, not the beats: check `words_source` and `word_stats` in the summary.

## 3. Storyboard

List the shots as a table (name, `from`, `to` in clip seconds on downbeats, subject, camera, what happens at the cut). Decide at each
cut whether it is a plain cut or a [transition or insert](design.md#transitions-and-inserts), and what each output shows: a 9:16 output
crops differently from a 16:9 one (`[shot.aspect.<output>]`, `frame = {...}` to solve the lens per output, or a lens shift for an exact
square crop of a vertical master). Put the plan in comments of `mk.toml` or a local file, never in the repository's tracked docs.

## 4. Cast, sets and props

1. **Know the model.** `mk inspect MODEL.pmx` (or `mk assets add MODEL.pmx --kind model`) and read the summary: `kind`, `quirks`, chain
   `families`, `expressions`, `measure`. Quirks are things that will bite later (missing bones, non-bone vertex groups, locked physics
   joints, missing textures). A character that does not exist yet is made from a model base: `mk model new NAME`, then
   [model_base.md](model_base.md); `mk model build` writes the PMX and the rig.json a `[[cast]]` takes.
2. **Register and clear licences.** `mk assets add` for everything you use, then `mk assets show SLUG --terms`, read the readme lines, and fill
   in `license`, `restrictions` and `credit` with `mk assets set`. `mk assets credits` fails while anything used is unreviewed.
3. **Set up the project.** An `mk.toml` with `[project]` (fps, frame0, duration, blend), one `[[output]]` per output, the `[[cast]]`,
   collider sets and the checks that define "done" (see the example below). The README has a project that builds.
4. **Build the places.** Library sets and props ([Sets](design.md#sets), [Props](design.md#props)), placed by rules instead of coordinates
   ([Placement](design.md#placement)): `place = { on = "room:floor", align = "edge:back" }`. A placement that cannot be met fails with the
   numbers and the prop in the way. Props from MMD accessory models need no card ([PMX props](design.md#pmx-props)). Run `mk build --until props` and
   look at the room with `mk look` (an orbit camera clips at 100 m: look through a shot camera to see a far skyline).
5. **Model what is missing** with the loop under [Modelling props](#modelling-props), and measure it with `form`.
6. **Measure real people (optional, for performances).** `mk ref search "woman singing in car"`, `mk ref add ID...`, `mk ref track`, `mk ref measure`
   give blink, gaze, nod, sway and mouth numbers shaped like `[perform.<cast>]`, each with a confidence; `mk ref sheet` shows what was
   tracked; `mk ref clean` frees the disk. Prefer front-on or three-quarter clips (side-view faces are rarely found), read `qc.warnings`, and
   treat values with low `n` as hints.

## 5. Pose, perform and sim

- **Pose** ([Posing](design.md#posing)): `sit = "car:driver"` puts the hips on a seat use point; a standing pose writes its feet as
  `{cast = "rin", point = [x, y, 0]}` (in the character's own frame, so it survives moving her). Hands that hold things are built, not placed
  (below). The pose stage reports each arm IK's miss in mm and warns past 5 mm: lean the character (`lean`), move the seat (`sit_offset`) or bring the prop closer.
  Look at the pose with `mk build --until pose --out build/pose_test.blend` and `mk look --view front,3q,left,back --frames N --sheet`.
- **Perform** ([Perform](design.md#perform)): gaze events over an idle target, eye-only glances, breathing, sway, nods, beat bob, blinks,
  expressions, lip sync from the timeline (`sing`; `words = [[2, -1]]` mouths only chosen words, a partner shouting the line endings), twitches,
  strumming (`strum`: down strokes on the downbeats, the pick meeting the string at each strike).
  Group and material morphs (vowels with a tongue bone, a blush) are keyed like any other: the cast stage binds the model's morph sliders
  (`bound_morphs` in its report).
  Take the numbers from `mk ref measure` or a timeline, not from taste.
- **Moves** ([Moves](design.md#moves)): `[[move.<cast>]]` puts named moves on the beats (`chest_pat`, `paws`, `point`, `heart_wink`,
  `hands_up`, `stank` ...) and `{name = "rest", hand = "R", place = "mic"}` keeps a mic hand at the mouth between them (a two-hand move then
  plays with the free hand only; `place = "dainty"` rests a hand lightly on the front of a skirt, elbows in); they compile to the pose's hand keys (riding the chest) and perform's lean, twitches and faces. The hands
  come in from outside and stop at the body, clothes and hair, and wrists stay within 60 degrees of the forearm: the pose report's
  `moves.places` says how far each place was kept out and how much its wrist bends. Draw a pose sheet of every move, front and 3/4, and look at
  it closely before building the shots.
- **Sim** ([Sim](design.md#sim)): hair, ears, tails and skirts are solved outside Blender and baked. List the families, give the colliders (the
  seat, the car, the fingers, the floor), and `wind = { carrier = "car", ... }` in a moving vehicle. `mk build --skip sim` is the fast loop for poses
  and performance.
- **Worn props** (a guitar on its strap, [Playing a worn guitar](design.md#playing-a-worn-guitar)): `wear = "rin"` on the prop, `grip = "guitar:neck"` and
  `grip = "guitar:strum"` on the hands, `strum = {...}` in `perform`; check `strum`, `prop_body` and the arm `contact`.

Hands. `grip = "car:wheel"` (or `"rest"` with `rest = "car:sill_R"`) in `[pose.<cast>.hands.L]` solves the fingers on the model's own skin
([Grips](design.md#grips)), where `at` plus a finger preset only guesses. A `WARNING ... short of its goal` means the seat is out of the arm's
reach; a grip digest past 3 / 1 / 1 mm (gap, penetration, clash) means the prop does not fit that hand. Pin it with a `contact` check between the
fingertip bone's `.tail` and the prop.

## 6. Shots, effects and type

- **Shots** ([Shots](design.md#shots)): one `[[shot]]` per cut, a camera per output is made for you. Mount the camera on a car or a set, aim it at a target
  (`"cast:rin.head"`), give lens and lag; `frame = {subject, fill, solve}` solves the lens or dolly per output. A shot that two outputs frame differently
  overrides keys in `[shot.aspect.<output>]`.
- **Looks**: `style = "silhouette"` (flat background, the scene in one colour, accents in another), `style = "vector"` (a flat-vector drawing: every material a tone
  of the project's `[vector]` table, two-tone shading, outlines) and `reflection` (her image in a pane) are composed at render time; `mk look` draws them as
  `mk render` will, `--no-styles` draws the shot as lit. A vector look starts with the model's material names: `mk q BLEND --list materials` lists them with
  the name a `[vector] materials` rule matches, and the shots stage warns about a rule that names none. `[[ring]]` entries flip a vector shot to the look's
  `opposite` palette inside discs and bands that sweep out from a point on a beat.
- **Freezes**: `[[freeze]] {from, to}` holds the world while the shot's camera goes on moving (a hit-stop orbit); `mk look` and `mk render` show it, `mk q`
  reads the timeline as it is.
- **Transitions and inserts**: composited by `mk post` from layers `mk render` draws next to the frames. Windows may not overlap and must lie inside one shot of the
  cut; the build refuses a window that starts before `[scene] start`. Preview a frame inside one with `mk look`.
- **Type**: text on a sign, a screen or a page (`[[text]] on = "road:gantry1_panel1"`), lyric type (one text per sung word on its onset), handwriting that appears behind a pen's
  nib, and screen type laid over the picture. Fonts are registered assets (`mk assets add FILE --kind font`); use static fonts. Words come from the timeline and never reach a log.
  Delete `renders/<preset>` when type changes: a stale `screen/` folder stays with the frames it belongs to.

## 7. Checks

A project's `[[check]]` entries define done; `mk check` runs them all in one Blender pass and exits 1 if one fails. Worth having in every project:

| Want | Check |
|---|---|
| Hair, ears, tails do not shake | `jitter` per family, `max = 0.5` (side locks that swing: 0.8) |
| Hair stays out of the body and props | `penetration` with the project's collider set, `max = 5` (mm) |
| Feet do not skate | `foot_slide`, `max = 1.5` (mm per frame) |
| No broken elbows, knees, necks | `joint_limits`, `max = 0` |
| A hand really holds or touches something | `contact` between two points or a point and a track (see Hands above) |
| A worn prop (a guitar) is played in time and stays out of the body | `strum` (the pick meets the strings at every down stroke, `max = 10` mm; `detail.timing` is the strike error in ms), `prop_body` (`max = 8` mm), and `contact` of `bone("wrist.R").tail` against `obj("<cast>_hand.R").loc` (the arm reached every goal, `max = 5`) |
| The subject is in frame in every output | `framing` per shot (`frames` of that shot), `min = 0` |
| Nothing blocks the subject | `occlusion`, `max = 0.3` |
| The camera is not inside a car body or a wall | `camera_inside`, `max = 0` |
| No near-black, stays in palette | `palette` on rendered frames |
| No shimmer | `flicker` on rendered frames, compared shot with shot |
| A hero prop is not a pile of boxes | `form`, default `max` 0.25 (see Modelling props) |

Reading results:

- `jitter` with a `static` note: the chain barely moves and the ratio is float32 noise; judge by `jerk_p95_mm`.
- `penetration` is measured on the baked bones (what renders), with each chain's own radius. It reports the frame, bone and the shape it is in (`body:<bone>` for the model's own
  bodies). Body capsules are fatter than the mesh, so a few millimetres rarely show; look before fixing.
- `joint_limits` does not flag sideways elbow bends: MMD rigs treat the elbow as a ball joint and professional motions put upper-arm rotation there. It flags fold-through, in-plane
  hyperextension, knees bending the wrong way or sideways, wrists past 110 degrees, and necks and spines past what dances do.
- `framing` margins are fractions of the frame; negative means outside the safe area. Configure it per shot: a page close-up is not supposed to contain the head.

## 8. Look sheets

Numbers miss things. `mk look --frames 200,400,600 --sheet` shows the cut in every output; `mk look --view front,left,back --target 'bone("wrist.R").head' --dist 0.4 --frames 300 --sheet`
inspects a detail from all sides; `--ab OTHER.blend` pairs two scenes; `--ref` puts reference photos beside the render. Look at the images before claiming anything about how a shot
looks: silhouettes and proportions first, surface detail second. Frames inside a transition or an insert and frames with screen type are previewed composited; `--no-transitions` and `--no-styles` leave effects and looks out.

### Asking questions of a scene

```sh
mk q scene.blend --list bones                         # names: Blender, PMX and semantic
mk q 'bone("head").head' --frames 181:280             # any expression, frame by frame
mk q 'dist(bone("index_tip.R").head, obj("Pen").loc)' --frames all --summary
mk serve scene.blend &                                # big scene: keep it loaded, q / check / grip answer in < 1 s
```

Bones can be named by their semantic name (`head`, `wrist.R`, `index2.L`, see `mkmmd/core/bonemap.py`), their Blender name (`手首.R`) or their PMX name (`右手首`).
The expression language is listed under [Expressions](design.md#expressions).

## 9. Render and post

`mk render --preset draft --jobs 2` renders every output (frames are claimed, so a stopped render resumes and `--jobs` share the work); `--output 9x16 --frames t=0:5` renders a
part. `mk post --preset draft` grades and encodes `out/<name>_<output>_draft.mp4` with the song. Review the draft (`mk play` opens it in the player of the `player` setting, with the shots as chapters), fix, render `--preset preview` for motion blur, then `final`. A missing
layer or frame is an error unless `--allow-gaps`; after changing text, delete the preset's frames. `mk post --to DIR` copies the videos. The grade is `[post]`
([Rendering and post](design.md#rendering-and-post)): check `min_luma` in the report if you care about black.

## 10. Credits

`mk assets credits --out CREDITS.md` builds the credits from the cast and `[credits] assets`; put what only the project knows (the song, a print made for it, the palette, tools) in
`[credits] lines`. It exits 1 while any asset is unreviewed or has no author or credit line.

## Modelling props

A prop assembled from boxes reads as a toy even when every dimension is right. For anything built in code (a library prop, a vehicle, furniture, a set element) follow this loop, and do
not call a prop done before step 6:

1. **References first.** `mk ref photos 'Dodge 600 convertible' --set dodge600 --n 12` fetches licensed photographs from Wikimedia Commons into `refs/dodge600/` (outside a project:
   `~/.cache/mk/ref/photos/dodge600/`) with `SOURCES.json` (licence, author, page) and `contact_sheet.jpg`. LOOK at the sheet and pick the angles that matter (front 3/4, side, rear 3/4,
   top, interior, the detail you will model). Ask again for what is missing (`'Dodge 600 dashboard'`, `... wheel`): the same set takes more. The photos are visual references only: never
   textures, never in a render, never traced.
2. **Real dimensions.** Write down the true size and the proportions that matter (length, width, height, wheelbase, seat and sill heights; a mug's height against its width) before the first
   vertex. A data sheet or the Wikipedia page beats a guess; the photo decides what the data sheet cannot say (the slope of the hood, the radius of a fender).
3. **Build forms from profiles, lofts and subdivision with creases.** Draw a section, pull it along a path or between stations, let the Subdivision Surface round it, and put a crease only
   where the real part has a line. The toolkit is `mkmmd/core/shell.py` (`loft`, `cage`, `sweep`, `lathe`, `rounded_panel`, ...); the rules and recipes are in [modelling.md](modelling.md).
4. **Bevel every visible hard edge; never stack cuboids for visible forms.** Real surfaces have radii, and the highlights live on them. A box, a slab or a plate with a 1 mm bevel is still a
   box: round it, crown it, or build it as a loft. Parts that really are flat graphic layers (a label, a display) are tagged `mk_form_exempt` (`shell.exempt(obj)`).
5. **Measure with `form`.** A `[[check]]` per hero prop, run after every build:

   ```toml
   [[check]]
   name = "car form"
   metric = "form"
   args = { prop = "car", exclude = ["car_underbody"] }     # exclude: objects nobody sees; hidden colliders never count
   max = 0.25                                               # the default for hero props
   ```

   `value` is the boxiness (0 smooth .. 1 boxes). Read `detail.worst_parts` first: each entry names the object, the part inside it, its area, its share of the prop, `why` (`a box: ...` or
   `flat panels with sharp edges`), its size and where it is in the prop's frame; `sharp_panels` lists the biggest flat panels with unbevelled rims. Fix the top entry, rebuild, check again: a
   prop usually needs three or four rounds. Furniture and gadgets that are boxes by nature (a boombox, a nightstand) may be judged at `max` 0.5-0.7 once you have looked at them; walls, gantries and
   barriers are `exempt = ["wall*"]` (measured and listed, not scored) or `exclude`d. Do not raise `max` to make a hero prop pass. `mk build` runs the same maths on every library prop it places
   and logs `WARNING prop 'car': form 0.51 > 0.25 (car_body part 25: a box: ...)` when one is over its limit. A prop's card says what is fair for it: `form_max` (its limit; a boombox says 0.7; 1
   switches the guard off) and `form_exempt` (object names or patterns that are architecture or graphic layers), from the builder's card or the project's `[[prop]] card_extra`.
6. **Compare side by side and look.** `mk look build/scene.blend --frames 1 --view 35:12,90:8 --target 'obj("car").matrix @ Vector((0, 0, 0.6))' --dist 7.5 --ref refs/dodge600/front_left.jpg,refs/dodge600/side.jpg`
   writes `ref.jpg`: photo i beside view i at the same height. Choose `yaw:elev` to match the photo's angle (the photo's horizon and perspective tell you), then compare silhouette and
   proportion first (wheelbase, hood length, roofline, stance), surface detail second. Open the image. A passing `form` number means no boxes, not the right car.

## Debugging

- `MK_KEEP_JOBS=1 mk ...` keeps `~/.cache/mk/jobs/<id>/` (job, result, Blender log) even on success.
- `mk check --keep-sample` keeps the sampled `.npz` and its JSON reply in `~/.cache/mk/samples/` for your own numpy.
- A failing Blender job prints `error`, `log` (path) and `trace`; read the log before retrying.
- `mk build --until STAGE` and `--skip STAGE` bisect a build; `mk doctor` checks the environment (`--fix` restores mmd_tools' wheels).

## Pitfalls

Models and rigs:

- mmd_tools gives bones without a tail target a 0.08 m tail; never use bone tails as physical lengths.
- mmd_tools adds `_dummy_` / `_shadow_` helper bones and non-bone vertex groups (`mmd_edge_scale`, `mmd_vertex_order`); rig.json marks the former and quirks list the latter.
- Many PMX hair joints are fully locked (zero angular range): Bullet swings such chains as rigid sticks. That is why a cast member's chains are solved by the sim stage instead (`physics = "mk"`).
- A Blender started with `--factory-startup` can delete mmd_tools' bundled wheels and break PMX import everywhere; run `mk doctor --fix`.

The build:

- Stage order matters: a target that names a cast member (`cast:rin`) or the camera does not exist while sets and props are built, and the camera does not exist while poses and performance are
  built (`look = "camera"` in `[perform.<cast>]` fails). Aim at a bone or a point instead.
- An empty `[pose.rin]` table does nothing, and a standing pose needs explicit feet targets (`{cast = "rin", point = [x, y, 0]}`): `feet = "floor"` or `"seat"` without a `sit` fails.
- `mk look` removes the timeline markers for debug views (they switch cameras every frame) and keeps them for the cut; its orbit camera clips at 100 m and cuts far skylines away.
- A silhouette, vector or reflection shot is a finished flat frame on disk (composed inside the render job): `--no-styles` shows the lit scene instead, in `mk look` and `mk render`.
- After editing a `[[transition]]` or `[[insert]]` run `mk build` again (the cameras of the shots that lend frames are keyed at build; `mk render` tells you when they are not).
- A scripted driver is an animation that needs no Python only while its expression is a Blender *simple expression*, and Blender stores at most 255 characters of it (a longer one is silently cut there and then fails).
  `%`, `**`, `sign` and `copysign` are not simple (`fmod`, `pow`, `floor`, `clamp`, `smoothstep`, `lerp`, `min`, `max`, `abs`, `sin`, `cos`, `radians`, `sqrt`, `if ... else`, `and`, `or` are); check
  `driver.is_simple_expression` and `is_valid`, and split a long formula over driven custom properties of the object (the hearts prop's `age`, `u` and `pin`).
- A prop over 600 000 triangles is not measured by the form guard; a pmx or appended `.blend` prop is never measured.
- `mk serve` keeps the scene it loaded: re-run it after the file changes on disk. Only `q`, `check` and `grip` use it.

Renders and post:

- Delete `renders/<preset>` when text changes; layers and frames are kept by name and a stale one is not redrawn.
- Frames and layers are claimed with an empty file; a stopped render resumes, but a frame rendered before a change stays until you delete it.
- Use static fonts: Blender reads a variable font's default instance (`python -m mkmmd.fontinst` pins the axes).

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
name = "rin"
armature = "Rin_arm"
asset = "my_model"

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
