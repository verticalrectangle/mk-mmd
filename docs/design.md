# mk-mmd design

mk-mmd is a command-line toolkit for making music videos with MMD models in Blender: understand a model's rig, pose
it, put things in its hands, perform, simulate hair, build sets, frame shots, render and grade. It is built for agents
first (JSON in and out, checks that fail loudly, fast retries) and is pleasant for people too. It does not model new
meshes and it does not export edited PMX/VMD files back to MikuMikuDance.

This document is the contract between the parts. Change it together with the code.

## Layers

| Layer | Package | Runs in | May import |
|---|---|---|---|
| core | `mkmmd.core` | the CLI and Blender | stdlib, numpy |
| solvers | `mkmmd.solvers` | the CLI only | numpy, scipy, numba (optional) |
| blender | `mkmmd.blender` | Blender only | bpy, mathutils, `mkmmd.core` |
| cli | `mkmmd.cli` | the CLI only | everything except bpy |
| ref | `mkmmd.ref` | the CLI only | numpy, scipy, opencv; MediaPipe with the `ref` extra |
| timeline | `mkmmd.timeline` | the CLI only | numpy; torch, torchaudio, demucs, faster-whisper with the `timeline` extra |
| post | `mkmmd.post` | the CLI only | numpy, opencv, pillow |

Rules:

- Blender imports, samples, keys and renders. Numerical work (posing maths, IK, grips, hair, gaze, placement,
  framing, metrics) lives in `core` or `solvers` and works on sampled arrays, so it is fast, testable and traceable.
- `core` and `blender` run under Blender 4.2's Python 3.11 with numpy 1.24: no Python-3.12-only syntax, no
  numpy-2-only API, no scipy.
- No model, project or prop names inside the library. Bones are addressed by semantic names (`head`, `wrist.R`,
  `index2.L`) resolved through the model's map; props through their cards; projects pass their own numbers in.
- Everything a command produces is reproducible from its inputs; slow results are cached by a hash of their inputs.

## The bridge

The CLI runs Blender headless for anything that needs a scene:

```
blender -b SCENE.blend -y --python-exit-code 3 --python-expr BOOT -- job.json result.json
```

- `job.json`: `{"op": str, "args": {...}, "project": {...} | null, "config": {...}}`
- `result.json`: `{"ok": bool, "data": ..., "error": str, "trace": str, "seconds": float}`
- Blender-side handlers register with `@runtime.op("name")` in `mkmmd/blender/ops_*.py`.
- The Blender log of every job goes to `~/.cache/mk/jobs/<id>/blender.log`; it is kept when a job fails.
- Jobs without a scene open `~/.cache/mk/empty.blend` (a factory-default empty scene made once with Blender's user
  folders pointed at a scratch place), never `--factory-startup`: without the user's preferences Blender's extension
  sync removes the wheels of extensions it sees as disabled (mmd_tools' opencc), breaking PMX import in every running
  Blender.
- `mk serve SCENE.blend` keeps one Blender alive on a Unix socket. The bridge uses it automatically for the ops that
  leave the scene as they found it (`ping`, `list`, `q`, `sample`, `visibility`), which matters for big files. Ops
  that change scene state (`look` hides objects, removes markers, adds a camera) always get a fresh Blender.
- `sample` is how numbers leave Blender: one pass over a set of frames writes posed bone and object matrices, rest
  matrices, expression values, the active camera, resolved colliders and evaluated meshes (world-space triangles of what
  a render shows, for `form`) to an `.npz`; checks and solvers work on that.

## Conventions

- Units are metres, Z is up. A character faces -Y unless a seat or placement says otherwise. MMD models import at
  scale 0.08 (one MMD unit = 8 cm).
- Time: `frame(t) = frame0 + t * fps`, fractional. Clip time 0 is `frame0`; earlier frames are pre-roll (the hair
  settles there).
- Output: JSON on stdout (UTF-8, Japanese names kept). Images are written to files and their paths printed.
- Exit codes: 0 ok, 1 a check failed, 2 usage or configuration error, 3 Blender or runtime error.
- Lyrics: never print lyric text. Timelines reference words as `(line, word)` indices; the text stays in local
  files and is read only when rendering type. This keeps agent transcripts clean of copyrighted text.
- Frame specs (`--frames`): `181:280` (inclusive), `181:280:5` (step), `100,140,200`, `t=1.5:3.0` (clip seconds,
  needs a project), `all` (the project's clip).

## Configuration

`~/.config/mk/config.toml` (all optional), overridden by environment variables:

| Key | Env | Default |
|---|---|---|
| `blender` | `MK_BLENDER` | `~/blender-portable/blender-4.2.3-linux-x64/blender` |
| `mmd_addon` | `MK_MMD_ADDON` | `bl_ext.user_default.mmd_tools` |
| `assets` | `MK_ASSETS` | `~/mk-assets` |

## Project file: `mk.toml`

A project is a folder with an `mk.toml`; commands find it by walking up from the working directory (or
`--project DIR`). Paths are relative to the project folder.

```toml
[project]
name = "stan"
fps = 30
frame0 = 181            # Blender frame of clip time 0
duration = 24.6         # clip length, seconds
blend = "build/stan.blend"
build = "build.py"      # runs inside Blender with mkmmd importable
audio = "audio/clip.wav"

[[output]]
name = "9x16"
size = [1080, 1920]

[[output]]
name = "16x9"
size = [1920, 1080]

[[cast]]                # who is in the scene
name = "reisen"
armature = "Reisen_arm" # the armature object in the .blend
asset = "miy_reisen"    # registry slug (its rig.json); or rig = "path/to/model.rig.json"

[credits]
assets = ["miy_reisen"] # everything else the credits must name (audio, props, motions)
lines = ["## Music", "A Band - A Song (used as the soundtrack)", "## Palette", "Rose Pine Dawn"]   # credits only the project knows

[colliders]            # named sets of scene collision shapes, see Colliders
cafe = [{ type = "box", object = "ChairColBack", rnd = 0.012 }, { type = "floor", z = 0.0 }]

[[check]]               # see Checks
name = "back hair is calm"
metric = "jitter"
args = { family = "back_hair" }
frames = "181:918"      # optional, default: the clip
max = 0.5
```

Project-specific reference data that only the project can compute (where a nib should be, when a hand writes) goes
to `tracks/<name>.json`: `{"frames": [...], "<channel>": [one value per frame (number, vector or null)], ...}`.
Checks refer to channels as `track:<name>.<channel>`. Dense geometry that a one-position-per-frame track cannot hold, such
as the strokes of handwriting, has a shape of its own (`tracks/ink.json`, see Text: Ink).

## Model description: `rig.json` (schema 1)

Written by `mk inspect MODEL.pmx` (or `mk assets add`), stored in the asset library as `<assets>/rigs/<slug>.rig.json`.
Bone names are the Blender names mmd_tools gives on import (left/right prefixes become `.L`/`.R` suffixes); `jp`
keeps the original PMX name. Positions are in armature space (metres at the import scale); body geometry is in its
bone's rest frame, so it follows the bone wherever the model is placed and posed.

```jsonc
{
  "schema": 1,
  "source": {"path": "...pmx", "sha1": "...", "name_j": "...", "name_e": "..."},
  "scale": 0.08,
  "comment": "the PMX comment (author notes), first 4000 characters",
  "map": {"head": "頭", "wrist.R": "手首.R", "index2.L": "人指２.L", ...},   // semantic -> Blender bone
  "missing": ["upper_body3", ...], "missing_required": [],                  // standard bones not found
  "bones": {"頭": {"jp": "頭", "parent": "首", "head": [x, y, z], "tail": [x, y, z], "deform": true,
                  "helper": false}, ...},                                    // helper: mmd_tools _dummy_/_shadow_
  "fingers": {"index.L": ["人指１.L", "人指２.L", "人指３.L", "人差指先.L"], ...},
  "bodies": [{"name": "...", "bone": "頭", "shape": "sphere", "size": [..], "type": 0, "group": 0,
              "no_collide": [1, 2], "loc": [..], "quat": [..],
              "geom": {"kind": "sphere", "c": [..], "R": 0.1}}, ...],       // capsule: a b R; box: M half
  "chains": [{"family": "back_hair", "root": "後髪1", "anchor": "頭", "bones": ["後髪1", ...],
              "parents": [-1, 0, ...], "ends": [[x, y, z], ...], "body_radius": [0.072, ...],
              "branching": false, "locked_joints": 14, "length": 1.009}],
  "morphs": {"blink": "まばたき", "a": "あ", "i": "い", "smile_eyes": "笑い", ...},   // semantic -> morph
  "morph_list": [{"name": "まばたき", "panel": "eye", "kind": "vertex", "shape_key": true}, ...],
  "measure": {"top": 1.71, "eye_height": 1.32, "upper_arm": 0.21, "forearm": 0.17, "hand": 0.17,
              "thigh": 0.34, "shin": 0.34, "hip_height": 0.78, "shoulder_width": 0.18, ...},
  "quirks": ["...: 2 vertex groups are not bones (mmd_edge_scale, mmd_vertex_order); skip them ...", ...],
  "stats": {"bones": 370, "bodies": 212, "dynamic_bodies": 188, "joints": 355, "chains": 25, ...}
}
```

A chain is every bone driven by a dynamic rigid body, grouped from each root that hangs from a non-simulated bone
(the anchor). `ends` are the rest segment ends: the head of the simulated child, or for a leaf its tail when the
chain is connected (mmd_tools gives unconnected bones a default 0.08 m tail, so then the leaf continues its chain's
last direction). `length` runs along the bone heads. Families come from bone names (`bangs`, `side_hair`,
`back_hair`, `twintail`, `braid`, `hair`, `ears`, `tail`, `skirt`, `breasts`, `ribbon`, `sleeve`, `coat`,
`accessory`, `other`); projects tune the secondary-motion solver per family.

## Prop card (schema 1)

Every prop has a card. Procedural props ship their cards in `mkmmd/library/props/`; downloaded props keep theirs
next to the files in the asset library.

```jsonc
{
  "schema": 1,
  "name": "bentwood_chair",
  "kind": "procedural",                    // procedural | pmx | blend | gltf | fbx | obj
  "builder": "mkmmd.blender.library.props.chair:build",   // procedural only
  "source": null,                          // file path for imported props
  "license": {"text": "...", "credit": "...", "url": "..."},
  "size": [0.43, 0.48, 0.85], "origin": "floor_center", "front": "-Y",
  "use": {                                 // points other tools act on, in the prop's local frame
    "sit":  [{"name": "seat", "hip": [0, 0.04, 0.52], "seat_z": 0.45, "facing": [0, -1, 0], "back_tilt_deg": 10}],
    "grip": [{"name": "rim", "type": "ring", "center": [..], "axis": [..], "radius": 0.19, "tube": 0.015}],
    "rest": [{"name": "door_edge", "type": "edge", "a": [..], "b": [..], "normal": [..]}],
    "look": [{"name": "mirror", "point": [..]}]
  },
  "colliders": [{"type": "box", "center": [..], "half": [..], "rot_deg": [..]}, {"type": "cylinder", ...}],
  "slots": {"wood": "#b4637a", "cane": "#ea9d34"}   // palette slots for look development
}
```

Fields the placement rules and the build add (all optional; see Placement and PMX props): `bounds` {min, max} (the
visible geometry in the prop's frame; the props stage measures it when a card has none), `front` (`-Y`: the axis a person
faces), `flat` (true: other props may stand on or overlap it, a rug), `blocks` (false: it does not occupy its footprint),
`origin` (`wall_center` for something hung on a wall: its origin lies on the wall plane), `use.rest` planes with `size`
[w, d] or `radius`, `use.surface` panels (`center`, `normal`, `up`, `size` [w, h]), and for imported props `parts`,
`armature` and `stats`. A **set** card may carry `use.rest` (its `floor`) and `obstacles`, `[{name, min, max}]`
axis-aligned boxes in the set's frame that placed props keep clear of (windows, doors, built-ins). A project can merge
its own entries over any card with `card_extra`.

## Asset registry

`<assets>/registry.json`: a list of entries, one per model, motion, prop, vehicle, audio file, reference clip,
texture or font.

```jsonc
{"slug": "miy_reisen", "kind": "model", "path": "/abs/path/model.pmx", "name": "Reisen Udongein Inaba",
 "author": "Miy", "source_url": "...", "license": "...", "restrictions": "...", "credit": "Reisen model: Miy",
 "rig": "rigs/miy_reisen.rig.json", "tags": ["touhou"], "readmes": ["/abs/path/readme.txt"], "extra": {...}}
```

mk never guesses a license. New entries are `unreviewed` and keep the paths of the readme files found next to the
asset; someone who has read them fills in `license`, `restrictions` and `credit`. `mk assets credits` builds the
credits from the project's cast and `[credits] assets` and exits 1 while any of them is unreviewed; `[credits] lines` are
the project's own lines (a song, a print made for it, the palette, tools), appended as bullets under `## Also` or under
the `## Heading` lines among them.

## Colliders

Scene collision shapes for the secondary-motion solver and the `penetration` check, as specs (inline or a named
`[colliders]` set). Shapes ride on their source; rounded edges (`rnd`, m) keep contact smooth.

| Spec | Shape |
|---|---|
| `{type = "box", object = O, rnd}` | O's mesh bounds (scaled), riding on O |
| `{type = "cylinder", object = O, R?, half_h?, center?, rnd}` | about O's local Z through its origin (R, half height default to the bounds) |
| `{type = "cylinder", center = [x, y, z], R, half_h, rnd}` | vertical, fixed in the world |
| `{type = "capsule", object = O, a, b, R}` | segment a-b in O's axes (metres) |
| `{type = "sphere", object = O, c?, R?}` | on O |
| `{type = "capsule", bone = B, to = B2, R, armature?}` | from B's head to B2's head, riding on B |
| `{type = "fingers", radius = {thumb, index, middle, ring, little, palm}?, sides?, armature?}` | every finger segment and three palm capsules (wrist to index1, middle1, little1) |
| `{type = "floor", z}` | the ground plane |
| `{type = "prop", prop}` | every collider of a prop's card, as the sim stage uses them (stored on the prop's root when the scene is saved) |

The model's own bodies (rig.json `bodies`) collide too. A chain skips a body it already overlaps in the rest pose
(the author's intended overlaps); within `anchor_free` (0.25 m) of its root it also skips the body it hangs from and
every pair the PMX collision masks exclude.

## Checks

A check is a named metric with a threshold. Metrics compute from sampled data or renders and return numbers; `mk
check` prints every result and exits 1 if any fails. All sampled metrics of one run share a single Blender pass over
the union of their frames.

```jsonc
{"name": "back hair is calm", "metric": "jitter", "value": 0.38, "max": 0.5, "ok": true, "detail": {...}}
```

| Metric | Value | Notes |
|---|---|---|
| `jitter` | median jerk / median speed of chain points in the head's frame | calm hair < 0.5; Bullet hair on a seated model ~1.4; `static` flag when a chain barely moves (the ratio is then float32 noise) |
| `contact` | largest distance (mm) between two points (expressions or tracks) where `when` holds | `component = "z"` for a signed axis difference |
| `penetration` | deepest chain point inside a body or collider (mm) | measured on the baked bones, i.e. what renders |
| `foot_slide` | 95th percentile horizontal speed of planted feet (mm/frame) | |
| `joint_limits` | worst excess over a limit (deg) | elbow fold-through and in-plane hyperextension, knees, wrists, neck, spine; limits calibrated on professional MMD dances |
| `framing` | smallest margin to each output's safe area | per output aspect through the active camera; matches Blender's projection |
| `occlusion` | largest share of subject points hidden from the camera | ray casts; the subject's own meshes do not count |
| `camera_inside` | frames with the camera inside a closed mesh a render shows | hidden objects (colliders) and `ignore` (glow and haze volumes) are passed through |
| `flicker` | worst frame's 99th percentile of temporal luma noise | on rendered frames |
| `palette` | near-black share (or median distance to a palette) | on rendered frames; `rose-pine-moon`, `rose-pine`, `rose-pine-dawn` built in |
| `form` | boxiness 0..1 of a prop's evaluated geometry: area-weighted over loose parts, `1 - (1 - cuboid)(1 - flat_sharp)` | what a render shows at one frame, in the prop's frame; hero props stay under the default `max` 0.25; `detail.worst_parts` names the objects to fix (see Form below) |

`mk check --list` prints every metric's arguments.

**Form** (`form`; docs/AGENTS.md: Modelling props) measures how blocky a prop is. `prop = "car"` takes every object a render
shows under that prop root (colliders, objects hidden from render and characters are left out), `objects = [...]` names
objects as they are, `exclude = [...]` drops names or fnmatch patterns, `frame` picks the frame (default the project's
`frame0`). The `sample` op evaluates modifiers as a render does (Subdivision Surface and Multires at render levels) and
returns world-space triangles (`meshes` in its job); the maths is `mkmmd.core.form` (numpy only, tested without Blender).
Triangles are welded per object, grouped into loose parts and grown into planar patches (neighbouring normals within 3
degrees of the patch's mean); a patch is *large* from (0.12 x the prop's diameter)^2 up, a flat panel at the size the prop
is looked at. Per part:

- `cuboid`: how much of it is a box. Flat faces (within 3 degrees of an axis, fading out by 10) on three orthogonal axes
  must make up 60-85 % of its surface and the second axis must carry 10-22 % (a thin plate is no box). A bevelled or
  filleted cuboid stays a box until its fillets reach a sixth to a fifth of its smallest side; a crowned panel is not flat.
- `flat_sharp`: its surface in large flat patches, each weighed by the share of its own rim that turns more than 65
  degrees (a chamfer turns 45 per edge, a cube corner 90): flat panels with unbevelled edges, such as a table top without
  a rounded rim.

The value is the area-weighted mean over parts. `detail` also gives `flat` (all large flat area), `hard_edges` (sharp share
of the edge length around large patches), `worst_parts` (`object`, `part`, `area_m2`, `share`, `boxiness`, `why`, `size_m`,
`at` in the prop's frame), `worst_objects`, `sharp_panels` (the biggest flat patches with unbevelled rims) and `exempt`.
Hidden or internal surfaces count (no occlusion test): `exclude` what nobody sees.

The props stage runs the same maths on every library prop it places (`mkmmd/blender/build/form_warn.py`; pmx and appended
.blend props are someone else's modelling and are skipped) and logs `WARNING prop 'car': form 0.51 > 0.25 (car_body part 25:
a box: ...)` when one is over its limit. The prop's card says what is fair: `form_max` (default 0.25; a thing that is a box
by nature says 0.7; 1 or more switches the guard off) and `form_exempt` (object names or fnmatch patterns counted at weight
0), in the builder's card or the project's `[[prop]] card_extra`. Results are cached in `<project>/.mk/cache/form/` by a
hash of the evaluated geometry in the prop's own frame and of `core/form.py`, so an unchanged prop costs a mesh evaluation
and no analysis; a prop over 600 000 triangles is not measured.

Limits: `max` defaults to 0.25, the limit for hero props (anything the camera sees large); a project overrides it per check.
Furniture and gadgets that are boxes by nature (a boombox, a nightstand) read 0.2-0.7 even when well made: judge them by
eye and give them their own `max`. Architecture (walls, gantries, barriers) is legitimately boxy: `exempt = [names or
patterns]` measures and lists those objects but weighs them at `exempt_weight` (default 0), as does an object tagged
`mk_form_exempt` (a custom property; `shell.exempt(obj)` in the shell toolkit) or named in `[[prop]] card_extra = {
form_exempt = [...] }`; `exclude` removes objects altogether. Calibration on the test scenes:

| Scene (`mk check ... form`) | value |
|---|---|
| the rejected car, `convertible_80s` modelled from cuboids: whole prop / exterior / body | 0.51 / 0.59 / 0.71 |
| `convertible_80s` rebuilt on the shell toolkit (one lofted body shell, swept and framed trim) | 0.14 |
| `car_mockup` (stacked boxes) | 0.90 |
| a car body lofted and subdivided with the shell toolkit (the docs/modelling.md example: long flat flanks, creased shoulder) | 0.08 |
| café iPod, the highest of the passing set | 0.09 |
| café mug, chair, table, vase, saucer, plants, page, poster, lights, pen; the MMD characters | 0.00-0.01 |

## Looking

`mk look` renders views without touching the file: the cut (scene camera with its timeline markers) per output
aspect, named cameras, or orbit presets around any target expression relative to a cast member's facing. It writes
one JPEG per view, aspect and frame, plus optional contact sheets, strips, A/B pairs against another scene and
framing guides. The cut shows a shot's render-time look (silhouette, reflection; see Shots) as `mk render` will draw it;
`--no-styles` draws every shot as it is lit.

## Building

`mk build` assembles the scene from `mk.toml` in one Blender session and saves it to `[project] blend`. Stages run
in order and each reads its own sections:

| Stage | Sections | Does |
|---|---|---|
| scene | `[scene]` (`start`, `end`, `settle_frames`) | empty scene, fps, frame range including the pre-roll before `frame0` |
| sets | `[[set]]` (`name`, `kind`, `at`, `yaw`, builder keys) | library set builders with their paths, surfaces and lights (see the list below) |
| props | `[[prop]]` (`name`, `card`, `at`, `yaw` or `rot`, `parent`, `slots`, `card_extra`, `place`, `attach`, `anchor_to`; `scale` and `origin` for PMX props), `[[scatter]]` | library props (`library:<name>`), card files, MMD accessory models (`pmx:PATH` or a registered prop slug, see PMX props); builders get the project palette as slots; their use points and colliders; `bounds` measured on every card; placement rules and clutter (see Placement) with a per-prop report (position, yaw, clearance) |
| vehicles | `[[vehicle]]` (`prop`, `path`, `lane`, `speed`, `at`, `roll`, `pitch`, `wheelbase`, `steer_ratio`) | a prop drives a set path's lane: position and heading per frame, body roll and pitch, wheels spinning, the steering wheel turning with the curvature |
| cast | `[[cast]]` (`name`, `asset` or `pmx`, `armature`, `at`, `yaw`, `parent`, `physics`) | models imported without Bullet (`physics = "mk"`), named, placed |
| pose | `[pose.<cast>]` | sit on a prop's seat (`sit_offset` slides the hips on it), feet on targets (leg IK), lean / turn / head (`lean_share`, `turn_share`, `head.neck` split them over the spine and neck), arm IK to points, edges and moving keys (targets can ride a prop part such as a steering wheel), finger presets or curl tables, grips (the hand holds a prop's `use.grip` entry or lies on a `use.rest` surface, a pen's nib can follow a track on every frame, see Grips), `[[pose.<cast>.drape]]` (a bone chain such as a skirt pointed along chosen directions, optionally bunched); `[[prop]] attach = "cast:bone"` puts props on bones. The stage reports each arm IK's miss in mm (`ik_error_mm`, the worst over a moving track) and logs a WARNING past 5 mm: a goal beyond the arm's reach leaves the hand short of the prop |
| motion | `[[motion.<cast>]]` | VMDs on NLA strips: source range, scale or `retime = "beats"`, body masks, blends |
| perform | `[perform.<cast>]` | gaze events over an idle target, eye-only glances (`glance`: the eyes lead, the head lifts a little, the lids open), breathing, sway, nod, beat bob, startles, keyed upper-body `lean` / `turn` (a reach that leans in and settles back), blinks, lids, expressions, lip sync, twitches |
| shots | `[[shot]]` | the cut, see Shots |
| lights | `[[light]]`, `[look]` | lights in palette colours (mounted, aimed, keyed); view transform, contrast look, exposure |
| text | `[[text]]` (`name`, `on` or `mount` / `at` / `facing` / `box`, `text` or `value`, `font`, `size` or `fit`, `align`, `offset`, `color`, `glow`, `depth`, `reveal`, `blink`, `flicker`, `lyrics`, `kinetic`, `backing`, `outline`, `ink`; see Text) | type on set and prop surfaces: fitted, palette-coloured, a typewriter reveal, a keyed number; geometry nodes with keyed inputs, so EEVEE needs no Python at render time; `lyrics` is one text per sung word, read from the timeline and landing on its onset (arrival, letter spread, weight, drip, tape behind it, all keyed); `ink` is handwriting that appears behind a pen's nib (a ribbon mesh with write times, a keyed clock) |
| keys | `[[key]]` (`target`, `prop`, `index`, `keys = [[t, v], ...]`, `interp`, `relative`) | keys on set, prop and object properties: a set's storm and fog, a car's lamps, any RNA path (`location`, `data.energy`); `relative = true` adds the values as offsets to what the property already does (a value another stage solved, such as a hand's grip target) and keeps its animation outside the keys' span |
| sim | `[sim.<cast>]` (`families`, `params`, `colliders`, `props`, `fingers`, `floor`, `wind`) | secondary motion solved outside Blender (`mkmmd.solvers.strands`) and baked to keys; branching chains solve as trunk then branches; `wind.carrier = "car"` makes the air relative to a vehicle. `params.<family>`: `sag`, `drag`, `zeta`, `radius`, `radius_max`, `friction`, `wind_drag`, and for sheets (skirts) `lateral` (0..1, default 0.5: the chains of a skirt are tied to their neighbours in a ring found from the rest positions, and the cloth between them is kept out of seats and thighs; 0 = independent chains), `lateral_collide`, `anchor_free`. Strands stay inextensible, so the baked bones render the solved particles |
| save | | the `.blend` |

Library sets (`kind`; every key is documented in the builder's docstring): `test_road`, `night_sky` (gradient dome,
stars, moon with light, horizon glow, clouds, EEVEE ray tracing), `highway` (divided road with lanes as card
paths, lamps with baked spill, gantries and billboards as `use.surface`, tunnels, trees, wet asphalt), `skyline`
(a city arc or band with lit windows and aviation lights), `cafe_room` (the rainy café: window with rain and fog,
street, storm; its animatable state is custom properties on the set root), `bedroom_80s` (an 80s bedroom at night:
striped wallpaper, parquet, trim, a door, a window with a half-raised venetian blind and a lit city behind it, a
neon tube; its floor, walls, `use.rest` and obstacles are what the placement rules work on; `blinds`, `slat_angle`,
`window_glow`, `city_glow` and `neon` are custom properties on the set root). Library props (`library:<key>`):
`car_mockup`, `chair`, the café props (`cafe_chair`, `cafe_table`, `cafe_page`, `cafe_pen`, `cafe_mug`,
`cafe_saucer`, `cafe_ipod`, `cafe_earbuds`, `cafe_vase`, `cafe_fairy_lights`, `cafe_poster`, `cafe_pothos`,
`cafe_haworthia`, `cafe_monstera`) and the bedroom props (`bedroom_desk`, `desk_chair` (seats a character exactly as
`cafe_chair` does), `desk_lamp` (owns its spot light, `power` and `on`), `cassette_player` (`glow`), `cassette_tape`
(label colour by name), `bed_single` (a Memphis quilt, pillows), `rug_80s` (`flat`), `poster_80s` (five designs),
`alarm_clock` (`glow`, `colon`, `alarm`), `bedroom_nightstand`, `wall_shelf`); their options are the non-colour
`slots` keys documented in each module's docstring (`mkmmd/blender/library/props/bedroom_*.py`). `convertible_80s` is an
80s convertible (Dodge 600 proportions, top folded) with two seats and the driver on +X, wheels, steering, a dash with
`SPEED_GRID` / `BAR_GRAPH` surfaces and a cassette, and `lamps` (0..1: headlamp lenses and spot lights with a warm-up
flicker); its card documents every `use` point.

The character eases from its rest pose (at `start`) into the base pose over `settle_frames`; secondary motion settles
in the same pre-roll. Rotations are composed in the armature's axes (the model faces -Y): a bone's posed rotation
relative to rest is `D_bone = D_parent . q`, and the bone-local key is `R^-1 q R` with `R` its armature-space rest
rotation, so characters riding a moving vehicle keep correct keys. Gaze blends unit directions from the eyes, so
targets at any distance (a mirror 0.5 m away, a road 30 m ahead) mix evenly.

Targets anywhere in the build spec are `[x, y, z]` (world), `{prop = "car", point = [..]}` (a prop's local frame),
`"car:road"` (a prop use point), `"cast:rin"` (eyes) or `"cast:rin.head"` (a bone), or `"camera"`.

Solvers run as `python -m <module> IN.npz OUT.npz` on the CLI's Python and are cached in `<project>/.mk/cache/`
by a hash of their inputs.

## Placement

`[[prop]] place = {...}` puts a prop's footprint on a surface instead of giving it `at` and `yaw`, and `[[scatter]]`
fills a surface with clutter. The maths is `mkmmd/core/place.py` (numpy only, tested without Blender); the props stage
(`mkmmd/blender/build/place.py`) gathers the scene and writes the result as the prop root's world matrix. A prop's
footprint is the plan rectangle of its card's `bounds` (measured from its visible geometry when the card has none); its
lowest point rests on the surface.

**Surfaces** (`on = "<prop or set>:<name>"`; the name may go when the owner has exactly one rest plane): a `use.rest`
plane (a table top, a set's `floor`: horizontal, `size` [w, d] or `radius`, rectangle or disc), a `use.rest` edge (the
footprint's centre goes on the segment), or a `use.surface` panel (a wall: `center`, `normal` pointing into the room,
`size` [w, h]; the prop hangs on it facing out, its back, or its origin when the card's `origin` is `wall_*`, on the
plane). `(u, v)` are metres from the surface's centre along its axes: on a plane u is the owner's +X and v its +Y (its
front is -v), on a wall u runs right as seen from the room and v up.

| `place` key | Meaning |
|---|---|
| `on` | the surface, as above |
| `at` | `[u, v]` (the footprint's centre; one number on an edge), `"center"`, `"near:<target>"` (below) or a region `[[u0, v0], [u1, v1]]` (a seeded spot inside it); default `"center"` |
| `distance`, `bearing` | with `near:` — the gap between footprints (m, default `clear`; negative = overlapping that much, a chair tucked under a desk) and where around the target (deg from its front: 0 in front of it, +90 toward its left; candidates in 15 degree steps, nearest the bearing first) |
| `facing` | a target (the prop's `front` axis turns toward it, per candidate position) or a number of degrees; default the prop's `yaw`. Yaws are relative to the surface's frame, so a prop on a turned desk turns with it; on a wall the prop always faces out |
| `align` | `"edge:front"` (or back, left, right; top, bottom on a wall), or a list for a corner: the footprint is pushed against that edge of the surface, `clear` inside it, replacing that coordinate of `at`. front / back are -v / +v, left / right -u / +u |
| `clear` | metres (0.02): the gap kept to other props and to the surface's edges |
| `avoid` | prop names (and `<set>:<obstacle>`) to keep clear of whatever the heights |
| `seed` | integer: breaks ties between mirror candidates, drives regions and scatters |

A placed prop blocks the new one when their plan footprints come closer than `clear` while their heights overlap
(standing on top of something does not overlap). A prop occupies its **height layers**, not one box: the stage slices its
visible geometry into layers of about 5 cm and gives each the plan bounding box of its surface points, kept on the card as
`layers` ([[z0, z1, x0, y0, x1, y1], ...] in the prop's frame), so a chair is wide at the seat and only a backrest above,
and a lamp is a small base under a long shade: a mug stands under the shade, the lamp clears the chair tucked under the
desk. The prop that owns the surface (`on = "desk:top"`) never blocks what stands on it, even where its bounds rise above
the surface (a headboard). A card with `flat` (a rug) never blocks and is never blocked (it lies under things),
`blocks = false` opts a prop out, a set's `obstacles` (windows, doors) block like props, and an `avoid` entry blocks
whatever the heights. The `near:` target may be overlapped (its gap is `distance`). Targets of `near:` and `facing` are a
prop name (its footprint and heading), `prop:use` or `set:use` points, or `[x, y]` / `[x, y, z]` in the world; `cast:`
and `camera` do not exist yet when props are built. A card can carry a second `use.rest` plane for the part of a surface
that is really free (a desk top under a window sill: `card_extra = { use = { rest = [{ name = "work", ... }] } }`), and
things that stand up are placed on that.

`[[scatter]]` (`name`, `props` [card refs], `on`, `count`, `region` [[u0, v0], [u1, v1]] for the footprint centres,
`min_dist` (m between scattered footprints), `yaw` [lo, hi] (deg, relative to the surface), `seed`, `clear`, `avoid`,
`slots`, `card_extra`) builds `count` props `<name>_0`, `<name>_1`, ... picking cards from `props` (seeded) and drops
them one after another at random spots that fit; it runs after every `[[prop]]`, works on walls too, and fails when an
item finds no spot in 300 tries.

Everything is deterministic for a seed. The stage output carries each placed prop's `placement` (`on`, `position` of the
origin, `yaw_deg`, `uv`, `clearance_mm`: the margin to the surface's edge and the nearest blocking prop) and the log a
line per placement. When a rule cannot be met the build fails with the numbers and the prop in the way: `place on
'desk:top': the footprint is 90 mm too close to the right edge (clear 20 mm)`, `overlaps 'lamp' by 12 mm`, `400
candidates tried (overlap 'lamp' x400)`. Cards read by the rules: `bounds`, `layers`, `front`, `flat`, `blocks`, `origin`,
`use.rest` planes with `size` or `radius`, `use.surface` panels, and on set cards `obstacles` ([{name, min, max}] boxes
in the set's frame). The `bedroom` test project (`~/Projects/mk-tests/bedroom`) is the worked example: desk on the
floor against the window wall, the chair tucked under its edge facing it, the lamp and the player pushed into the desk's
corners, tapes scattered, posters on the walls, the bed in a corner.

## PMX props

`[[prop]] card = "pmx:PATH"` (a `.pmx` / `.pmd` file or a folder holding one; relative paths from the project) or an
asset slug registered as kind `prop` (its path is a model, a card file or a folder with a `card.json`) imports an MMD
accessory or prop model with mmd_tools (scale 0.08) as a plain prop under its root; `scale` resizes it and `origin`
(`floor_center` default, `center`, `keep`) puts the origin at the bottom centre of its bounds, the centre, or where the
author had it. The model keeps its own materials and shape keys (`slots` are ignored, with a warning). Rigid body, joint
and mmd root objects are removed. A model with no moving parts becomes static (no armature); one whose bones move
separate parts (more than one bone carrying at least 8 vertices and 1% of them besides a single base bone) keeps its
armature as `<name>_arm`, listed in the card as `parts` (bone, head, tail in the prop frame, vertices), so a project can
pose them (`[[key]] target = "<name>_arm", prop = 'pose.bones["bone"].rotation_quaternion', index = 0`).

The card is made from the geometry (`mkmmd/core/propcard.py`): `size` and `bounds`, `front` `-Y`, a `look` point
`center`, a `use.rest` plane `top` (`size` [w, d], or `radius` for a round top) when the model is roughly flat-topped
(the highest level covering 40% of its footprint lies within max(2 cm, 12% of the height) of its highest point: a desk,
a mixer, a crate; not a cup, a cap, a figure), and `colliders` from the model's static rigid bodies (spheres and
capsules riding on the root, boxes as hidden `<name>_col<i>` objects; dynamic bodies are soft parts and are skipped),
else one box over the bounds. `stats` says what was found (triangles, rigid bodies, top face, textures missing or
relinked). `card_extra = {...}` (any prop) is merged over the card: tables key by key, lists of entries with a `name` by
name (an entry with a known name replaces it, a new name is appended, an empty list clears), anything else replaces:
`card_extra = { front = "+X", use = { rest = [{ name = "top", center = [0, 0, 0.74], size = [1, 0.5] }] } }`. Colliders
and moving parts are rest-pose approximations, they do not follow a moving part.

## Grips

A grip is how a character's hand holds a prop: the 15 finger rotations and the frame the hand takes on the prop, solved
on the model's own skin (`mkmmd/solvers/grip.py`, numpy + scipy, no Blender) until the pads touch the prop's surface
(contacts within about 0.5 mm), nothing of the hand is inside it and no finger is inside another. The hand is
exported from the scene by the Blender op `hand_model` (rest bones, rest skin near the wrist with its weights, and for
a missing `*_tip` bone a virtual tip). `mk grip STYLE ...` runs both and writes JSON; the build's pose stage calls the
same solver through `python -m mkmmd.solvers.grip IN.npz OUT.npz` (cached by a hash of the hand, the spec and the
solver source).

| Style | Prop (`use.grip` card entry, metres) | Grip frame (`target_in_wrist` is this frame) | Options |
|---|---|---|---|
| `pen` | `type = "pen"`: `length`, `radius` (number or `[[distance from nib, radius], ...]`), `tip`, `nib_offset` | the prop's own: origin at the nib (minus `nib_offset`), +Z nib to cap, +X the barrel side facing the back of the hand | lateral tripod; `posture` (nib, shoulder, pole, table, ...) adds the writing orientation |
| `wheel` | `type = "ring"`: `center`, `axis` (prop frame), `radius`, `tube` | origin on the tube's centreline where the palm sits, x radially outward, y tangent (counter-clockwise seen from +z), z the ring axis | `approach` (deg, palm side in the section plane, 0 = +x, 90 = +z), `wrap` (+1 / -1) |
| `pinch` | `type = "pinch"`: `width` (thickness between the pads), `span` (depth, becomes `depth`), `length`, `center`, `axis` (along the strap), `normal` (outward) | midway between the pads, z from the index pad to the thumb pad, x away from the wrist, y = z cross x | `edge` (pads' distance inside the edge, 4 mm) |
| `rest` | `use.rest` entry (`edge` with `a`, `b`, `normal`, or a `plane` with `center`) | origin on the plane below the palm centre, z the normal, x the hand's heading | `face` (`palm` or `back`) |

`pen` was tuned on one hand shape (Reisen's): the other two models tried, Una and Maki, also solve within the gates
(gaps 0.2-0.6 mm; Maki's thumb sits at the solver's bounds) but a third hand can still miss them (the CLI then exits 1).
The other three styles solve every hand tried within 0.5 mm.

**Result.** `{"style", "side", "bones": {blender_bone: [w, x, y, z]}, "target_in_wrist": 4x4, "report", "solver"}`.
`bones` are `pose_bone.rotation_quaternion` values (bone-local, relative to rest; the 15 finger joint bones, identity
where the grip leaves one alone). With the wrist posed, `frame_world = wrist_bone_world @ target_in_wrist`, so the wrist
goes where `grip_frame_world @ inv(target_in_wrist)` says. `report` has `contacts.<name>.gap_mm` (0 = touching),
`penetration_mm`, `finger_clash_mm`, `angles_deg` and style numbers; a pen with a posture adds `frame_world_quat`.
The solver's `python -m` interface writes `bones` (names), `quats`, `target_in_wrist` and `report` (JSON of the whole
result minus those) to the output `.npz`; exit code 0 ok, 2 bad input.

**In the build** (`[pose.<cast>.hands.<L|R>]`, replaces `at` / `rest` / `fingers`):

- `grip = "car:wheel"`: the prop's `use.grip` entry, by type. A ring needs `clock` (hours on a clock face as the
  character sees the wheel: 12 top, 3 its right; default 10 for L and 2 for R, read at the first frame, the hand then
  rides the wheel with `ride = "car_wheel"`); `approach = 90` puts the palm on the side of the rim facing the character
  and `wrap = -1` the fingers round the outside of the rim; `seeds`, `skin_radius` (0.16 m). The ring's axis is flipped
  to point at the character when the card gives it the other way. A pen grip needs a `posture` or a `track` (below), a
  pinch grip a `normal` on the card.
- `grip = "rest"` with `rest = "prop:edge"` (+ `along`, `offset`, `lift = 0.0`, `face`, `dir`): the relaxed hand lies on
  the surface, the palm centre at the rest point, `dir` its heading.
- The stage output and the build log carry each hand's digest (contact gaps, penetration, finger clash, seconds,
  warnings past 3 / 1 / 1 mm) and `ik_error_mm`. A wrist that ends more than 5 mm short of its goal is a WARNING: the
  seat is too far for the arm, so lean the character (`lean`), move the seat or bring the prop closer. Arm length is
  about 0.38 m for a 1.7 m model. Arms without a grip take `fingers` as a preset (`relaxed`, `curled`, `fist`, `flat`,
  `point`) or a table of curls, `{index = [8, 10, 0], middle = [10, 12], thumb = [0, 8]}`: degrees of each finger's
  first three joints toward the palm, anything left out stays straight (`mkmmd/core/fingers.py`).
- **A pen whose nib follows a path** (`grip = "pen:barrel"` with a `track`): the writing hand of a character who writes
  while the camera watches. Give the nib's path as a project track, `track = "nib"` for `tracks/nib.json` (or any
  `.json` path), `channel = "target"` (the default) naming the positions: `{"frames": [Blender frames], "target":
  [[x, y, z] world metres per frame]}`, held before its first and after its last frame. The stage solves ONE grip for
  the pen on the hand, with its writing orientation (the whole hand's rotation about the nib, solved against the arm
  and the desk), then keys the arm IK on every frame with the wrist goal `pen_frame @ inv(target_in_wrist)`, the pen
  frame's origin on the track and its rotation the solved one (LINEAR keys), and bone-parents the pen prop to the
  wrist in the solved grip: the pen object's origin, the nib, lands on the track within the IK's accuracy (a
  `contact` check of `obj("pen").loc` against `track:nib.target` pins it; tens of micrometres in the original).
  `posture` completes itself from the scene (`mkmmd/core/pentrack.py`): `nib` is the track's mean x, y at
  `paper_z` (the track's lowest z unless given), `shoulder` the shoulder in the seated pose at the end of the settle,
  `facing` the character's heading, `pole` the IK's elbow pole (`pole = [x, y, z]` of the hand), `upper` / `fore` the
  model's arm; the caller gives what only it knows: `table = "table:top"` (a prop's `use.rest` plane, or `{z, center,
  radius}`: the forearm stays above it) and `target = {elevation, azimuth, tilt, extension, ulnar: [deg, tolerance]}`
  to change the writing posture. `wobble = 2.2` (or `{deg, tau, seed}`) tilts the pen slowly about the world X and Y,
  as the original did; the stage logs the grip digest (with the `writing` posture numbers) and the worst IK miss over
  the track. The solve takes minutes on a pen (cached by a hash of the hand, the track-derived posture and the solver
  source), so keep the pose (`lean`, `head`, `pole`) steady while you tune the rest.

Frames: `mkmmd/core/gripframe.py` builds the grip frame in the world (`ring_frame`, `surface_frame`, `pinch_frame`) and
the wrist goal (`wrist_goal`); it is numpy only, so the maths is tested without Blender. Check a grip with `contact`
checks between a fingertip bone's tail and the prop (see AGENTS.md).

## Shots

`[[shot]]` (`name`, `from`, `to` in clip seconds) makes one camera per output aspect (`<shot>@<output>`), keyed on
every frame: `mount` (a prop, set or object the camera rides), `at` (in the mount's frame), `look` (any target),
`lens`, `roll`, `lag` (operator lag in seconds, applied in the mount's frame so a camera in a car lags the subject,
not the road), `shake` (handheld, degrees), `keys = [{t, at, look, lens, shift}]` for moves, `dof = {focus, fstop}`, and
`frame = {subject, fill, solve}` to solve the lens (or dolly) per aspect so the subject fills that share of the
frame height. `[shot.aspect.<output>]` overrides any key for one aspect. Timeline markers cut between shots; the
scene keeps the shot table in `scene["mk_shots"]`, and `mk look` / `mk render` point the markers at each aspect's
cameras before rendering it.

`shift = [x, y]` is Blender's lens shift: fractions of the larger image side, the picture moving the other way (a positive `y`
moves it down; probed in Blender 4.2.3), constant or keyed in `keys[].shift`. It is how a 1:1 output becomes an exact crop of a
9:16 master: the square camera keeps the master's position, aim, roll and keys and changes three numbers (`crop_camera` in
`mkmmd/core/shotstyle.py`): the lens × 1920/1080 (a pixel keeps its angular size), the `dof` f-stop × 1920/1080 (the blur keeps its
size in pixels) and `shift = [0, (420 − top) / 1080]` for a crop whose top row lies `top` px below the master's top, so the
master's axis sits `960 − top` px below the crop's top (`top = 420` is centred: no shift). Against a render of the master the crop
differs by 0.4/255 on average (anti-aliasing); pitching the aim to follow the crop instead is off by up to 88 px at the corners.
`roll` turns the picture around the shifted axis: keep rolled shots centred.

### Looks: silhouette and reflection

Two looks need more than the lit scene. They belong to the shot and are switched on and off frame by frame by `mk render` and
`mk look` (`mkmmd/blender/styles.py`; the normalised specs per output aspect are in the shot table), put back everything they
change, and leave `mk post` to grade the finished frames as usual. `mk render --no-styles` and `mk look --no-styles` draw every shot as lit. The build
resolves the colours, checks the glass object and logs a WARNING for an object pattern that matches nothing. Object patterns are
`name*` (fnmatch on the object name, case-sensitive), `@collection` (in that collection or below) and `prop:key` (that custom
property is set on the object).

| Key | Meaning |
|---|---|
| `style = "silhouette"` | the flat look of the lightning and sunbreak shots: a flat background, the scene as one colour, accents in another. `style = "none"` in `[shot.aspect.<output>]` switches an inherited look off |
| `colors = {background, subject, accent}` | palette slot, `#hex`, `"a:b:t"` mix or `[r, g, b]` (defaults base, text, surface) |
| `hide`, `keep` | objects not rendered (walls, outside, rain), minus `keep` |
| `accent` | objects in the accent colour. One with a transparent material (a lightning bolt) keeps its softness; the others (earbud cords) are painted over everything and grown by `grow` px (1, counted at 1080 wide) so a thin wire reads |
| `tint = [{object, prop, color, gain, glow}]` | the background moves toward `color` by min(1, gain × the custom property `prop` of `object`): a lightning flash. `glow = {at = [x, y], size = [sx, sy]}` makes it a gaussian bloom from that point (fractions of the frame from the top left) |
| `knockout = {objects, color}` | type reversed over the silhouette: `color` on the background, the background colour where it overlaps the silhouette. These objects are drawn over everything; other type is hidden behind what stands in front of it |
| `samples = 16` | EEVEE samples of the passes below |
| `reflection = {object, strength, dim, roughness, hide, only, bend, world, tint}` | her image in a window pane (below) |

Silhouette. Blender 4.2.3 ignores `view_layer.material_override` in EEVEE Next and Workbench (only Cycles honours it), so the
frame is composed inside the render call from passes that work, with the formulas of the original video's post
(`compose_silhouette`, numpy): Workbench with flat light and `Object.color` draws the subject and the cords in their colours
(exact to one level of 255, Workbench's colour transform); EEVEE with an AOV on the Mix Shader factor of the accent objects'
materials and the compositor (R = alpha, B = AOV, Raw) gives the bolt as the original's `sil` pass did (`sil = alpha − bolt`: the
AOV is painted in the accent colour, what its soft shells have beyond that stays a glow in the subject colour); type, the objects
whose modifier node group starts with `mk_text_`, is rendered by EEVEE with the lights off and a black world as its emission shows
it, keyed opacity included, and drawn over the silhouette; `knockout` objects likewise as a mask. Against the original's own sil
pass at three frames the bolt's alpha and AOV agree to 0.002 (mean absolute) and the composed frame to 1.2–2.1/255 (the rest is
the scene: her pose, the cords' route); the flat colours graded by `[post]` land within 1–2 levels of the delivered frames. A
frame costs about a second at 540 px wide (Workbench 0.2 s, the EEVEE passes the rest); silhouettes are always 8-bit.

Reflection. `reflection = {object = "<glass>"}` shows what stands in front of a window pane; screen-space tracing cannot see a person
behind the camera. EEVEE Next draws it through a plane light probe on the glass (the set's own probe at the pane, else one is made)
and a mirror layer mixed in front of the glass shader: `strength` (0.5) of a Glossy BSDF (`roughness`, tinted by `tint`) over the
glass dimmed by `dim`. The rain bends the reflection only with `bend`. `hide` and `only` choose what the pane reflects (the probe
skips the other objects) and the sky is black in it unless `world`, so a pane that reflects only the sitter lets the dimmed street
show through at 1 − `strength`. The original composed a mirrored-camera pass at 18 % for the room and 52 % where she is (street
at 57 %); one mirror weighs both the same, which is why the room is left out. Against that composite (type and grade left out) the
error of the frame drops from 32 to 17 of 255 with `strength` 0.3–0.4; the sitter's own region (24 → 21) is limited by her pose.

## Text

`[[text]]` puts type on a surface: a sign panel of a set (the `highway` gantries and billboards), a prop's screen, label
or page (any `use.surface` card entry), or a panel placed freely on anything. Each entry is one object named `name`: an
empty mesh with a geometry-nodes modifier (String to Curves, Fill Curve, Extrude Mesh for `depth`) and a palette-coloured
emissive material, parented to the owner's root (or to the surface's `object`), so it rides a moving car and sits `lift`
(2 mm) in front of the surface. Whatever changes over time is a key on a node input (the number, the typewriter's ramp,
the emission gain), never a frame handler, so `mk render` needs nothing but the saved `.blend`; the fonts are packed
into it. The text frame is the card's: x right (`up x normal`), y up, z out of the surface. The stage runs after `lights`
and before `keys`, so `[[key]]` can toggle a text's `hide_render` or move it.

| Key | Meaning |
|---|---|
| `name` | object name (unique) |
| `on` | `"<set or prop>:<surface>"`, a card `use.surface` (centre, normal, up, size); the surface name may go when the owner has just one |
| `mount`, `at`, `facing`, `up`, `box` | free placement instead of `on`: panel centre `at`, the direction it `facing`, and `up` (default +Z) for its top edge, in the frame of `mount` (a set, prop or object; default the world). `box = [w, h]` is the panel to fit and align in; without it text is placed around `at`. `box` also narrows a surface's panel |
| `text` or `value` | the string (`\n`, or a list of lines, for several lines), or `value = {keys = [[t, v], ...], format = "{:.0f} MPH", interp}`: a number keyed over clip time, formatted with literal text around one `{:[0][width][.decimals][f\|d]}` field (`"{:03d}"`, `"{:.1f}"`); zero padding is for non-negative numbers |
| `font` | asset registry slug (kind `font`) or a font file; default Blender's built-in font |
| `size`, `fit` | `size` is the cap height (m). `fit` is the share of the panel the ink of the widest string the text will ever show may fill (a keyed number is measured at every string it can show), also the margin text aligns in; default 0.9. With both, `size` is the largest cap height `fit` allows |
| `align`, `valign` | left / center / right and top / middle / bottom (`align = "left top"` works too): the ink block of the widest string meets that edge of the margin, or the centre; lines align inside the block by `align` |
| `offset` | `[u, v]` metres along the panel's right and up |
| `color`, `glow` | palette slot or `#hex` (`"slot:slot:0.3"` mixes two); emission strength (1.0; 0 = lit only by the scene) |
| `depth`, `lift` | extrusion toward the viewer (0 = flat); distance in front of the surface (0.002) |
| `tracking`, `word_spacing`, `leading` | character spacing and word gap (factors, 1.0); line pitch / cap height (1.5) |
| `reveal` | `{from, to}` clip seconds, typewriter: the first character appears at `from`, the last at `to`; spaces cost no time |
| `blink`, `flicker`, `fade` | multipliers on the emission: `blink = {period, duty, low, phase, from, to}` (on for `duty` of each `period`, `low` otherwise) and `flicker = {amount, rate, dips, seed, from, to}` (seeded random dips) step; `fade = [[t, gain], ...]` (or `{keys, interp}`) is a keyed ramp, e.g. a dash waking up. All need `glow` > 0 |
| `ghost` | `true`, a number or `{strength, text, color}`: for display fonts, the unlit segments behind the lit ones (the widest string with every letter and digit as an 8), added as light so they stay a faint hint. `strength` is their brightness as a share of the lit segments' as displayed (0.10; the emission is that share to the 2.2), their colour the text's pulled halfway to the palette's `muted` |
| `halo` | `true` or `{strength, size}`: a slight glow past the lit edges (copies of the lit shapes on three rings, additive, just behind them); `strength` the share of the lit emission it adds (0.3), `size` its reach in cap heights (0.04). EEVEE has no bloom; `mk post` halation comes on top |
| `haze` | `false` or `{distance, cap}`; text on a `highway` set fades into the road's haze like its signs |
| `lit` | share of the scene's light the letters also reflect (1.0); 0 is flat ink: only the emission shows |
| `back` | `true`: the text goes on the other side of a card surface (its normal reversed, `up` kept: a pane read from outside) |
| `backing` | `{color, pattern, pattern_color, pad, height, torn, glow, scale, dz, seed}`: a strip of tape behind the text, the ink box plus `pad` [x, y] (em; 0.55, 0.3) or `height` em, ends torn by `torn` x its height (0.1), `pattern` `plain` / `stripe` / `dots` / `check` drawn from the strip's own UV; it moves, rotates and fades with the word |
| `outline` | `{color, width, alpha, dz}`: a ring of another colour behind the letters (the filled shapes grown by `width` em, 0.05, at `alpha` 0.5) so they read on any background |
| `kinetic` | `{show, scale, sx, sy, dx, dy, dxp, dyp, rot, pivot, tracking, weight, opacity, tint, drip}`: motion keyed on node inputs, written by `lyrics` (see Lyrics) or by hand |
| `lyrics` | `{timeline, line, words, style, ...}`: one text per sung word, read from the timeline, see Lyrics below |
| `ink` | `{strokes \| track, on, width, lift, color, dry, glossy, from, to}`: handwriting instead of type, a pen's strokes that appear behind the nib; the entry carries only `name` and `ink`, see Ink below |

```toml
[[text]]                              # a highway sign panel: two lines, fitted to 80 % of the panel
name = "exit_sign"
on = "road:gantry1_panel2"
text = "NORTH\nEXIT 12"
font = "overpass_bold"
fit = 0.8

[[text]]                              # typed out on the lower strip of a billboard
name = "tagline"
on = "road:billboard1_panel"
box = [10, 1.1]
offset = [0, -1.55]
text = "open all night"
font = "permanent_marker"
reveal = { from = 3.0, to = 5.0 }

[[text]]                              # a speedometer: a keyed number in a 7-segment font, unlit segments behind it
name = "speedo"
on = "car:speed"                      # or free: mount = "car", at = [..], facing = [0, 1, 0], box = [0.3, 0.09]
value = { keys = [[6.0, 58], [7.9, 71]], format = "{:.0f}" }
font = "dseg7_classic_bold"
align = "right"
ghost = true
halo = true
fade = [[5.4, 0.0], [5.9, 1.0]]
```

Blender only measures: the ink box of every string the text can show is taken from the same String to Curves node at
em size 1, and `mkmmd/core/typeset.py` (numpy only, tested) does the rest: size and fit, alignment, number formats, the
typewriter's keys and character count, blink and flicker keys, the surface frame. The build reports each text's cap height,
ink size and whether it fits, and logs a WARNING when a given `size` overflows its panel. Read a keyed number with
`mk q 'bpy.data.node_groups["mk_text_<name>"].nodes["Value"].outputs[0].default_value' --frames ...`.

Fonts live in the asset library (`<assets>/fonts/<family>/`, licence files beside them) and are registered with
`mk assets add FILE --kind font --slug ...`, with licence and credit filled in; list the slugs in `[credits] assets`.
Registered: `dseg7_classic`, `dseg7_classic_bold` (7-segment digits, OFL, keshikan), `overpass_regular`,
`overpass_semibold`, `overpass_bold` (highway signage, OFL, Red Hat), `monoton`, `audiowide` (80s display, OFL),
`permanent_marker` (hand lettering, Apache 2.0), `fraunces_italic`, `fraunces_roman`, `fraunces_semibold`, `fraunces_black` (a
variable serif, OFL, Undercase Type). Use static fonts: Blender reads a variable font's default instance. `python -m
mkmmd.fontinst SRC.ttf OUT.ttf wght=420 opsz=72 SOFT=100` pins a variable font's axes into a static font (`--axes` lists
them; needs fontTools), which is how the Fraunces entries were made.

### Lyrics

A `[[text]]` with `lyrics = {...}` is lyric type: one text per sung word, read from the timeline and keyed so that every word
lands on its onset. `mkmmd/core/wordtype.py` (numpy only, tested on synthetic timelines) selects the words and decides when each
lands, arrives, spreads, weighs, leaves and where it stands; `mkmmd/blender/build/wordtype.py` reads the file and hands one
ordinary text per word to the stage, each with a `kinetic` table of keys. The entry's other keys (`on`, `mount` / `at` /
`facing` / `box`, `font`, `size` / `fit`, `color`, `glow`, `lit`, `depth`, `lift`, `offset`, `outline`, `backing`, ...) go to
every word; `on` may be a list of surfaces (word k goes to surface k mod len: a diagonal chain over window panes in lyric
order is such a list, each word fitted into its own pane). Words that must stay where they are on screen
while the camera moves (macro shots against glass) go on a plane that rides the shot's camera: `mount = "<shot>@<output>"`,
`at = [0, 0, -distance]`, `facing = [0, 0, 1]`, `up = [0, 1, 0]`.

The words never leave the local files: the log, the stage report, errors, object names (`<name>_l<line>w<word>`) and custom
properties (`mk_lyric = [line, word]`) carry numbers only; the string lives in the object's String to Curves input and nowhere
else. Lines and words are numbered from 1 as everywhere.

**Timing.** A word lands on clip frame `floor(start * fps + 0.4)`, never before the frame of `from`, and is on screen from that
frame (its arrival plays after it). It is gone from `round(frame0 + to * fps)`, the frame the shots stage cuts to the next shot,
or earlier when its line drips away (`leave`) or another word takes its slot (`recycle`); a word that would never be on screen
is left out (the report lists it by number). Frames in the stage report are Blender frames.

| Key (inside `lyrics = {...}`) | Meaning |
|---|---|
| `timeline` | the timeline JSON (project-relative, or `~/...`) |
| `line`, `lines = [a, b]`, `words = [i, j]` | 1-based and inclusive; `words` applies to every selected line (one number: one word) |
| `case`, `clean` | `keep` / `lower` / `upper` / `title`; `clean = true` keeps letters, digits and apostrophes |
| `style`, `arrive` | how a word arrives: `pop` (a damped spring about its centre: `arrive = {a, td, tp, rise, jitter}`), `slap` (`pop` plus a decaying wobble: `wobble`), `drop` (falls in from half a panel above and squashes on landing, about its bottom: `drop`), `rise` (steam: from `arrive.base` = [u, v] shares of the panel, default the last word's slot, to its slot, swaying, arriving when the last word lands: `sway`, `sway_rot`), `type` (a typewriter over the note), `none`. Every style takes `alpha0` and `alpha_frames`: the opacity at the landing (1.0) and the frames it takes to reach 1 |
| `from`, `to` | clip seconds the type may show; default `to`: the end of the last word's note |
| `leave` | `cut` (default: hold to the cut) or `drip`: when a tempo tick falls between the end of the last word's note (minus 30 ms) and `to - 0.4 s`, every letter stretches downward from its top and falls like a drop, the first one leaving on the tick's frame; no such tick: it holds to the cut. `drip = {g, stretch, life, gap, order, min_run}`: gravity in px/s² at a 150 px em (2600), extra height (3.2), seconds to fade (0.55), seconds between words, `none` / `bottom_first` / `last_first`, seconds of shot left (0.4) |
| `spread` | letters spread (tracking grows) while a note is held, by `0.07 + 0.27 smoothstep(0.12, 0.85, hold)` of the word's width with an ease-out cubic, and relax after it with a 0.13 s time constant; the fit leaves room for the widest it gets. `false` or `{scale}` |
| `weight` | `{lo, hi, breath}`: strokes grow by `lo` .. `hi` em with the word's own `vocal_db` level (the mean over its note), breathing with the voice (smoothed per frame, `breath` 0.65 of it) while the note is held and 0.1 s after |
| `colors`, `color_by` | accent colours cycled by `line` (default) or by `word` |
| `recolor` | `{color, from, over, ease, words}`: a keyed mix toward another palette colour (frost, ash) from clip seconds `from` (`"last"`: when the last word lands) over `over` seconds, for `words` = [first, last] of the selection only (default all) |
| `tilt`, `sizes`, `offsets` | per word: degrees, cap heights (m), `[u, v]` nudges (m) on top of where the word stands: a number or a list cycled over the words (`tilt = {random = deg}`: seeded by the word) |
| `backing` | a `backing` table, or a list of them cycled over the words |
| `recycle` | `true` or `{gap, fade, assign}`: a word is gone `gap` frames before the next word of its slot lands and fades over the `fade` frames before; `assign` lists group numbers cycled over the words, apart from where they stand |
| `layout` | `same` (default: every word in the entry's panel, centred), `flow`, `stack` or `slots`, below |

Layouts. `flow` (`gap` 0.28 em): words side by side in reading order, one row per lyric line, the block centred on the panel
and fitted to it (one cap height for all). `stack` (`rows`, `pitch` 1.3 em, `dir` `down` | `up`, `align` and `shift` cycled by
row, `tilt` per row): word k on row `k mod rows`, the first on top (`down`) or at the bottom (`up`). `slots` (`slots = [{at,
box, align, tilt, size}]`, `assign`): `at = [u, v]` and `box = [w, h]` are shares of the panel (u right, v up from its centre),
the word is fitted into its box and aligned there; word k takes slot `assign[k]` (default k mod len). A layout needs a panel
(`on`, or `box`). The words are measured in Blender with the node the text is drawn with; the rest is `core/wordtype.py`.

`kinetic` keys (clip seconds; `[[t, v], ...]` linear keys, one per frame where a value changes): `show = [on, off]` (on screen
from the frame of `on` until the frame of `off`), `scale`, `sx` / `sy` (squash about the pivot), `rot` (degrees), `dx`, `dy`
(em), `dxp`, `dyp` (shares of the panel), `pivot` (`center`, `bottom`, `top`), `tracking` (a factor), `weight` (em),
`opacity`, `tint = {color, keys}` and `drip = {t, g, stretch, life, seed}`. They become keys on Value nodes of the text's node
group (`Pop`, `SquashX`, `SquashY`, `Rot`, `Dx`, `Dy`, `DxPanel`, `DyPanel`, `Tracking`, `Weight`, `Opacity`, `Tint`, `Show`,
`Drip`; read one with `mk q 'bpy.data.node_groups["mk_text_<name>"].nodes["Pop"].outputs[0].default_value' --frames ...`), so
a render needs no Python. Letters stay filled shapes of the String to Curves instances: the drip scales and moves each
character about its own top, opacity and tint travel as the attributes `mk_alpha` and `mk_tint` that the shared material reads
(hashed alpha: a word fades without sorting against the glass), the weight is the shape repeated on a ring. A `kinetic` text
takes no `value`, `blink`, `flicker`, `fade`, `ghost` or `halo` (they need their own material).

```toml
[[text]]                              # steam words over a mug: they land on their onsets, rise to their slots and frost over
name = "mug"
mount = "mug"
at = [0, 0, 0.14]                     # the plane through the focus point, facing the camera
facing = [0.495, -0.869, 0.0]
box = [0.128, 0.137]
font = "fraunces_italic"
size = 0.0175
color = "love:text:0.3"
lit = 0                               # flat ink
[text.lyrics]
timeline = "audio/timeline.json"
line = 1
words = [1, 4]
case = "lower"
clean = true
style = "rise"
from = 1.3                            # the shot's own seconds: no word lands before it, the cut hides them at `to`
to = 2.6
arrive = { alpha0 = 0.55, base = [0.0, -0.208] }
recolor = { color = "foam", from = "last", over = 0.2 }
[text.lyrics.layout]
kind = "slots"
slots = [{ at = [-0.215, 0.278] }, { at = [0.187, 0.150] }, { at = [-0.195, -0.069] }, { at = [0.0, -0.199] }]

[[text]]                              # one word per window pane, falling in and dripping away on a tempo tick
name = "front"
on = ["cafe:pane_r1c3", "cafe:pane_r1c2", "cafe:pane_r0c2", "cafe:pane_r0c3"]
font = "fraunces_roman"
fit = 0.8
lyrics = { timeline = "audio/timeline.json", line = 1, words = [5, 8], style = "drop", from = 2.6, to = 5.0, leave = "drip", sizes = [0.13, 0.055, 0.12, 0.21] }
```

The stage report has one entry per `lyrics` text, `{words, lines, style, layout, skipped, first_frame, last_frame, cut_frame,
drip_from}` (Blender frames; `skipped` as `[line, word]` pairs), and per word the usual text numbers with `lyric` [line, word],
`chars`, `kinetic_keys` and `frames` [on, off); the string itself (`widest`) is not reported.

### Ink

A `[[text]]` with `ink = {...}` is handwriting: the strokes of a pen, drawn on a surface, that appear exactly behind the nib
with no Python at render time. `mkmmd/core/ink.py` (numpy only, tested) makes ONE ribbon mesh of all strokes in the surface
frame (x right, y up, z out): per pen-down stroke a strip of quads `width` wide, `lift` above the paper, with a square cap
half a width long at both ends, bends mitred (up to two half widths: sharp turns thin out, they never spike), repeated
points merged and a one-point stroke made a dot. Every vertex carries the time the nib is on its cross-section as the point
attribute `tw`. The material (dithered alpha, exactly 0 or 1) shows a fragment once the ink object's clock has passed its
`tw`; the clock is the object's custom property `clip_t`, KEYED linearly over the frames to clip seconds (no driver, no
handler: `mk q 'bpy.data.objects["ink"]["clip_t"]' --frames ...`). Attributes interpolate along every quad, so the front
sits at the nib between points and between frames, and a nib that moves 3-10 mm per frame still leaves legible letters.
Fresh ink is glossy (coat, low roughness) and 18 % deeper in colour, and dries matte over `dry` seconds. Like any text
the object is parented to the owner's root, so it follows the page; it casts no shadow and takes `[[key]]` like any object.

| Key (inside `ink = {...}`) | Meaning |
|---|---|
| `strokes` | the strokes file (project-relative): `{"unit": "mm", "page": [w, h], "strokes": [{"t": [...], "p": [[x, y], ...]}]}`: page millimetres from the top-left corner as read (x right, y down) and the clip seconds the nib is on each point (never decreasing); `page` (optional) is checked against the surface. Points ~0.15 mm apart make letters |
| `track` | instead of `strokes`, the coarse fallback: `"nib"` (the project's `tracks/nib.json`) or a `.json` path: the runs of `down` through the `target` positions, one point per frame (shapes, not letters), counted only while the nib is within 1 mm of the surface's plane (a pen lifted away for a pause draws nothing) |
| `on` | `"<prop or set>:<surface>"`, a card `use.surface` (required): page millimetres map onto its panel from the top-left corner, as `page_to_local` does for the cafe page (160 x 220 mm) |
| `width`, `lift` | ribbon width (0.00042 m) and distance above the surface (0.00018 m) |
| `color` | palette slot or `#hex` (`pine`) |
| `dry`, `glossy` | seconds fresh ink stays glossy (0.55); `glossy = false` (or `dry = 0`): matte from the first moment |
| `from`, `to` | clip seconds: only ink written inside the window is built, a stroke that crosses a bound is cut exactly there |

```toml
[[text]]                              # the handwriting of the writing schedule on the cafe page, in pine
name = "ink"
ink = { strokes = "tracks/ink.json", on = "page:page" }
```

Only the project can compute where the pen is at every moment, so the strokes are reference data like the pen track: one
nib position per frame (`tracks/nib.json`) cannot draw letters of 3 mm that the nib crosses in a frame; the writing
schedule's dense strokes can. The build reports `strokes`, `points`, `verts`, `faces`, the length of ink (mm) and its time
range.

## Palettes

`[look] palette` picks a named palette (`rose-pine-moon`, `rose-pine`, `rose-pine-dawn`), `[look.slots]` overrides
slots. Slots: base, surface, overlay, muted, subtle, text, love, gold, rose, pine, foam, iris, hl_low, hl_med,
hl_high. Sets, props, lights and the grade colour by slot, never by hard-coded values.

## Rendering and post

`mk render --preset draft|preview|final` renders each output's cut to `<project>/renders/<preset>/<output>/
<frame>.png`. Frames are claimed with an empty file first, so a stopped render resumes and `--jobs N` Blender
processes share the frames; a disk check refuses to start when the frames would not fit. Shots with a render-time look (Shots:
silhouette, reflection) are finished flat frames on disk, composed inside the render job; `--no-styles` renders them as lit.

`mk post` grades the frames (`[post]`: contrast around a pivot, saturation, split toning, a palette floor with a
soft toe so nothing is black, halation from blurred highlights, vignette, grain) and encodes
`<project>/out/<name>_<output>[_<preset>].mp4` with `[audio]` (`file`, `start` = song seconds at clip time 0).

## Timeline

`mk timeline analyze` turns the clip's song span into `audio/timeline.json`: `bpm`, `beat_s`, `beats`, `downbeats`,
`grid` (a constant-tempo fit when the beats are that steady), per-frame `vocal_db`, `energy_db`, `drums_db`, and
`lines[].words[]` with `start`, `end`, `voiced_end`, `ctc_start`, `whisper` (seconds from clip time 0). The analysis
runs on the span padded with `--context` seconds of song on each side (default 20, clamped to the file) and is then
shifted and cropped: a short clip alone can read the bar phase half a bar off or merge lines. Beats come from the kick
and snare bands of the Demucs drum stem (hi-hats would double the tempo) with a dynamic-programming tracker;
downbeats from kick accents plus harmonic change (chords change on beat 1; kicks alone cannot tell beat 1 from beat
3). Words come from a lyrics file (one sung line per line) or from Whisper's transcription, aligned with wav2vec2 CTC
on the vocal stem and snapped to sung onsets; Whisper's line breaks are then corrected on the song's line grid (line
starts recur every one or two bars: merged lines are split at the grid, mid-line breaks merged). Lines and words are
numbered from 1; tools refer to words as `(line, word)` and never print their text.

## Cache

`<project>/.mk/cache/<op>/<key>.{json,npz}` where `key` is a SHA-256 of the op's inputs (arguments, input-file
fingerprints, solver version). Changing hair settings re-runs only the hair; nothing else is recomputed. Timeline
stems are cached by a hash of the audio span (`<project>/.mk/cache/timeline/stems-<key>.npz`).

Reference clips (`mk ref`) live outside the project: `~/.cache/mk/ref/<project name | default>/<set>/` holds
`clips.json`, the capped downloads, `track/<id>.npz` and the contact sheet; `mk ref clean` deletes everything but
`clips.json`. Only the small measurement JSON is kept with the project (`<project>/ref/<set>.json`).
## Characters (`mk model`)

`mk model build SPEC.toml` builds an original character in code: part builders (numpy and Pillow, no Blender) make
meshes, textures, bones, morphs and rigid bodies; the assembler merges them into one PMX; Blender imports that PMX back
with mmd_tools (the way `mk cast` will), the result is verified against what was assembled, described like `mk inspect`
does and saved as a review `.blend` with a neutral studio. It is the only place mk writes a PMX, and it never edits an
existing one. The spec, builders and assembler live in `mkmmd/model/` (CLI side: numpy, Pillow, scipy), the Blender
side in `mkmmd/blender/model/`; characters are made of parts so several people (or agents) can work on one.

```toml
[model]                       # model.toml; every other table belongs to the part or the file that defines it
name = "rin"                  # PMX model name, output file stem
parts = ["body", "head", "hair", "outfit"]            # build order = spec order
include = ["proportions.toml", "colors.toml", "body.toml"]   # merged first, in order; later wins; `~` works
out = "~/mk-assets/models/rin_mk"      # default output folder
seed = 1                      # ctx.rng per part: the same stream whatever else is built
needs = {hair = ["body", "head"]}      # parts `--only hair` builds first (default: body); builders may declare too
builders = {tails = "mkmmd.model.parts.hair"}       # part -> module when it is not mkmmd/model/parts/<part>.py
```

**Spec** (`mkmmd.model.spec`): tables merge key by key, anything else (lists too) is replaced; `Spec.files` lists what
was read; `--set a.b=1` overrides. **Builders** are `mkmmd/model/parts/<part>.py` with `@builder("hair", needs=(...))
def build(ctx) -> Part` (`mkmmd.model.build`): `ctx.spec`, `ctx.cfg` (the part's own table), `ctx.save_png(name, rgba)`
(into `<out>/tex`, part name prefixed, the returned file name goes into `Material.texture/toon/sphere`), `ctx.parts`
(built so far), `ctx.land` (landmarks: every part's `info["landmarks"]`, semantic name -> np.array(3)), `ctx.rng`,
`ctx.log`, `ctx.need(part)`, `ctx.find_body(bone)`, `ctx.find_bone(name)`. A part returns `Part` (`mkmmd/model/part.py`:
Material, Bone, Mesh, Morph, RigidBody, Joint); after each builder `part.check` and the cross-part checks run (unique
names, parents, weights, materials, bodies, joints).

| Convention | |
|---|---|
| space | metres, Z up, the character faces -Y, her left is +X, feet on z = 0, centred on x = 0 |
| faces, UVs | counter-clockwise from outside; UV per face corner with v UP (the assembler flips v and the winding for PMX) |
| names | PMX names exactly as `mkmmd/core/bonemap.py` expects for standard bones (全ての親 ... 左足ＩＫ); chain bones classify by `core/families.py` (前髪1, 三つ編左1, 猫耳右1, 尻尾1_2, スカート前1, リボン左1, 袖左1, 襟1); morphs by the standard Japanese names (まばたき, あ, 笑い, 照れ ...), panels eye / brow / mouth / other |
| weights | bone name -> (n,) arrays, capped at 4 and normalised by the assembler; an unweighted vertex follows the nearest deforming bone and is reported |
| materials | one PMX material per `Material.name` across parts (faces of several meshes are grouped), list order = draw order: declare translucent ones last |
| rigid bodies | model space; capsule height axis is +Z at rotation 0 (`size = (r, straight length, 0)`), box = half extents along x, y, z, rotation = XYZ Euler (rad); `static` -> PMX type 0, `dynamic` 1, `dynamic_bone` 2; `no_collide` groups become the PMX mask |
| joints | model space; limits and springs in the joint's own frame exactly as Blender's rigid body constraint holds them (the assembler converts to PMX axes) |
| IK | `Bone.ik` limits are PMX-convention degrees (the knee bends about x between -180 and -0.5) |

**Toolkit** (all bpy-free, docstrings are the API): `skeleton.standard_bones(landmarks, opts)` (the full standard and
semi-standard skeleton with IK, twist bones with fixed axes, fingers with local axes, eyes with grants, semantic names;
the module docstring lists the landmarks), `geo` (loft, sweep, tube, ribbon, revolve, spheres, surfaces, merge, weld,
normals, `capsule_between`, `euler_for_axis`), `skin` (chain weights with smooth joint blends, envelope weights,
transfer from body meshes for garments, ramps), `tex` (antialiased UV-space canvas, gradients, toon ramps, sphere maps,
noise, atlas), `subdiv` (Catmull-Clark with creases as sparse operators). `Mesh.subsurf` applies it before export: the
surface equals Blender's Subdivision modifier (boundary "all", linear UVs) to 1e-7 and UVs, weights and morph offsets
are carried through the operators. `mkmmd/model/parts/mannequin.py` uses all of it and is the fixture of the framework.

**Assembly** (`mkmmd.model.assemble`, `pmx_io`): bones go parents first (grant parents too); a PMX vertex per unique
(position, UV, normal); custom `Mesh.normals` as given, else angle-weighted normals split along `Mesh.sharp` edges (a
hard edge between coplanar faces costs nothing); quads split along the shorter diagonal, ngons by ear clipping;
positions `(x, y, z) -> (x, z, y) / 0.08` so `mk cast` (scale 0.08) gives back metres; PMX 2.0 (all mmd_tools reads),
UTF-16, textures as `tex/<file>`, display frames Root, 表情 (every morph, ordered eye, brow, mouth, other), the parts'
frames merged by name, the rest in その他.

**Verification** (`mkmmd.blender.model.verify`, op `model_finish`): the PMX is imported unclean (vertex order = file),
then compared with the assembled arrays: vertex positions, bone heads and parents, weights, UVs, corner normals, winding
against the normals, faces per material, morph offsets (0.1 mm each), bone flags, grants, fixed and local axes, IK,
materials (colours, edge, textures), morph panels and English names, display frames, rigid bodies (pose, size, mode,
groups, masks) and joints (pose, limits, springs). Anything beyond tolerance is listed under `verify.problems` and the
command exits 1.

**CLI.** `mk model build SPEC [--only PARTS] [--out DIR] [--no-export] [--no-blend] [--no-verify] [--set K=V] [--full]`
prints JSON: per-part numbers (meshes, vertices, faces, bones, materials, morphs, bodies, joints), warnings (lint:
unweighted vertices, missing UVs, unused vertices), the assembled model's counts, timings, `verify`, and `rig`: required
semantic bones missing, morph map (semantic -> morph), chain families with bone counts, bodies and measurements. Files in
the output folder: `<name>.pmx`, `tex/*.png`, `<name>.blend` (studio lights; `mk look <name>.blend --view front,3q
--target "bone('head').head" --dist 1.2` works), `<name>.rig.json` (what `mk inspect` writes; pass it as `rig =` to a
`[[cast]]` with `pmx =`), `build.json`. `--only` builds those parts and what they need into `<out>/only_<parts>/`;
`--no-export` only runs and checks the builders; `mk model info SPEC` shows the plan. Exit codes as everywhere: 1 when a
check or verification fails.

Blender note: a Blender session that resets the add-on preferences (a script started with `--factory-startup` that
touches add-ons) can delete mmd_tools' bundled opencc wheel, and every PMX import then fails with "bpy.ops.mmd_tools.
import_model could not be found"; use another config folder (`BLENDER_USER_CONFIG`) for such scripts. `mk doctor --fix`
restores the wheel and `mk model build` repairs and retries on its own.
