# mk-mmd design

mk-mmd is a command-line toolkit for making music videos with MMD models in Blender: understand a model's rig, pose it, put
things in its hands, perform, simulate hair, build sets, frame shots, set type, render and grade. It is built for agents
first (JSON in and out, checks that fail loudly, fast retries) and is pleasant for people too. It imports models and motions
and never edits them: the one PMX it writes is the one of a character it built itself (`mk model`), and nothing is exported
back to MikuMikuDance.

This document is the reference: the contract between the parts, every `mk.toml` section and key, the build stages, the checks,
the shot looks and effects, text, post, the timeline, the cache and the character builder. Change it together with the code.
[../README.md](../README.md) is the short tour and the command table, [AGENTS.md](AGENTS.md) the playbook for making a video,
[modelling.md](modelling.md) how props are modelled as forms. Times in this document are clip seconds, lengths metres, angles
degrees (see [Conventions](#conventions)).

## Contents

- [Layers](#layers)
- [The bridge](#the-bridge)
- [Conventions](#conventions)
- [Configuration](#configuration)
- [Project file](#project-file)
- [Model description (rig.json)](#model-description-rigjson)
- [Prop card](#prop-card)
- [Asset registry](#asset-registry)
- [Colliders](#colliders)
- [Checks](#checks)
- [Looking](#looking)
- [Building](#building)
- [Placement](#placement)
- [PMX props](#pmx-props)
- [Grips](#grips)
- [Playing a worn guitar](#playing-a-worn-guitar)
- [Shots](#shots)
- [Text](#text)
- [Palettes](#palettes)
- [Rendering and post](#rendering-and-post)
- [Timeline](#timeline)
- [Cache](#cache)
- [Characters (mk model)](#characters-mk-model)
- [VRM (mk model vrm)](#vrm-mk-model-vrm)
- [Briefs (mk brief)](#briefs-mk-brief)
- [Reviews (mk review)](#reviews-mk-review)
- [The site (mk site, mkmmd/site)](#the-site-mk-site-mkmmdsite)
- [Exporting the page (mk export web)](#exporting-the-page-mk-export-web)


## Layers

| Layer | Package | Runs in | May import |
|---|---|---|---|
| core | `mkmmd.core` | the CLI and Blender | stdlib, numpy |
| solvers | `mkmmd.solvers` | the CLI only (`solvers.geom` is numpy only and also runs in Blender's sim stage) | numpy, scipy, numba (optional) |
| blender | `mkmmd.blender` (`ops_*.py`, `build/`, `library/`, `model/`) | Blender only | bpy, mathutils, `mkmmd.core`, `mkmmd.solvers.geom` |
| cli | `mkmmd.cli`, `mkmmd.checks`, `mkmmd.bridge`, `mkmmd.project`, `mkmmd.assets`, `mkmmd.cache`, `mkmmd.config` | the CLI only | everything except bpy |
| model | `mkmmd.model` (`mk model`) | the CLI only | numpy, scipy, Pillow, `mkmmd.core` |
| ref | `mkmmd.ref` (`mk ref`) | the CLI only | numpy, opencv, Pillow; MediaPipe with the `ref` extra |
| timeline | `mkmmd.timeline` (`mk timeline`) | the CLI only | numpy; torch, torchaudio, demucs, faster-whisper with the `timeline` extra |
| post | `mkmmd.post`, `mkmmd.cutfx`, `mkmmd.matte` (`mk post`, the cut effects) | the CLI only | numpy, opencv, Pillow, `mkmmd.core` |
| review | `mkmmd.review` (`mk review`) | the CLI and the site's server | stdlib, numpy, Pillow, `mkmmd.model.lab` |
| brief | `mkmmd.brief` (`mk brief`) | the CLI and the site's server | stdlib, Pillow, `mkmmd.project`; ffprobe |
| site | `mkmmd.site` (`mk site`, the review page): the server, its API, `web/` (plain JavaScript) | its own process on 127.0.0.1, the page in a browser or Tern's browser block | stdlib, numpy, `mkmmd.review`, `mkmmd.brief`, `mk model glb`, `mk timeline analyze` and Blender (model files) as subprocesses |
| tern | `tern/` (the Tern plugin: Luau, `plugin.toml`) | Tern's session daemon and windows | Tern's `tern` API; runs `mk` |

Rules:

- Blender imports, samples, keys and renders. Numerical work (posing maths, IK, grips, hair, gaze, placement,
  framing, metrics) lives in `core` or `solvers` and works on sampled arrays, so it is fast, testable and traceable.
- `core` and `blender` run under Blender 4.2's Python 3.11 with numpy 1.24: no Python-3.12-only syntax, no
  numpy-2-only API, no scipy.
- No model, project or prop names inside the library. Bones are addressed by semantic names (`head`, `wrist.R`,
  `index2.L`) resolved through the model's map; props through their cards; projects pass their own numbers in.
- Everything a command produces is reproducible from its inputs; slow results are cached by a hash of their inputs
  (see [Cache](#cache)).

## The bridge

The CLI runs Blender headless for anything that needs a scene:

```
blender -b SCENE.blend -y --python-exit-code 3 --python-expr BOOT -- job.json result.json
```

- `job.json`: `{"op": str, "args": {...}, "project": {...} | null, "config": {...}}`
- `result.json`: `{"ok": bool, "data": ..., "error": str, "trace": str, "seconds": float}`
- Blender-side handlers register with `@runtime.op("name")` in `mkmmd/blender/ops_*.py`: `ping`, `list`, `q`,
  `sample`, `visibility`, `hand_model`, `inspect_model`, `look`, `render_frames`, `build`, `model_finish`, `model_studio`.
- The Blender log of every job goes to `~/.cache/mk/jobs/<time>-<op>-<id>/blender.log`; the folder is kept when a job fails
  (`MK_KEEP_JOBS=1` keeps it on success too). A failing job makes the command print `error`, `log` (the path) and `trace`
  and exit 3.
- Jobs without a scene open `~/.cache/mk/empty.blend` (a factory-default empty scene made once with Blender's user
  folders pointed at a scratch place), never `--factory-startup`: without the user's preferences Blender's extension
  sync removes the wheels of extensions it sees as disabled (mmd_tools' opencc), breaking PMX import in every running
  Blender. `mk doctor --fix` restores the wheels.
- `mk serve SCENE.blend` keeps one Blender alive on a Unix socket. The bridge uses it automatically for the ops that
  leave the scene as they found it (`ping`, `list`, `q`, `sample`, `visibility`, `hand_model`), which matters for big files:
  `mk q`, `mk check` and `mk grip` answer in under a second. Ops that change scene state (`look` hides objects, removes markers,
  adds a camera; `build`; `render_frames`) always get a fresh Blender.
- `sample` is how numbers leave Blender: one pass over a set of frames writes posed bone and object matrices, rest
  matrices, expression values, the active camera, resolved colliders and evaluated meshes (world-space triangles of what
  a render shows, for [`form`](#form)) to an `.npz`; checks and solvers work on that.

## Conventions

- Units are metres, Z is up. A character faces -Y unless a seat or placement says otherwise. MMD models import at
  scale 0.08 (one MMD unit = 8 cm).
- Time: `frame(t) = frame0 + t * fps`, fractional. Clip time 0 is `[project] frame0`; earlier frames are pre-roll (the hair
  settles there). Every time in `mk.toml` is in clip seconds unless a key says song seconds (`[audio] start`).
- Output: JSON on stdout (UTF-8, Japanese names kept). Images are written to files and their paths printed.
- Exit codes: 0 ok, 1 a check, gate or verification failed, 2 usage or configuration error, 3 Blender or runtime error.
- Lyrics: never print lyric text. Timelines reference words as `(line, word)` indices, numbered from 1; the text stays in
  local files and is read only when rendering type. This keeps agent transcripts clean of copyrighted text.
- Frame specs (`--frames`): `181:280` (inclusive), `181:280:5` (step), `100,140,200` (a list; ranges may be mixed in:
  `100,200:210`), `t=1.5:3.0` (clip seconds, inclusive, one frame apart; `t=1.5:3.0:0.5` steps in seconds; `t=2` one time;
  needs a project), `all` (the project's clip: `frame0` for `duration * fps` frames).
- Names: an output is named by its aspect (`16x9`, `9x16`, `1x1`); a cast member `rin` gets the armature `Rin_arm`
  (`[[cast]] armature` overrides); a shot's cameras are `<shot>@<output>`; a text entry is the object of its `name`
  (`<name>@<output>` when it exists once per output). Semantic bone names: see [Model description](#model-description-rigjson).

### Expressions

`mk q`, the `a`, `b` and `points` arguments of checks, `mk look --target` and the `anchor` / `center` of
[transitions and inserts](#transitions-and-inserts) take a Python expression evaluated in the scene at every frame asked for. Names:

| Name | Meaning |
|---|---|
| `bone(NAME[, armature])` | world-space view of a posed bone: `.head` `.tail` `.center` `.dir` `.length` `.matrix` `.quat` `.local_quat`. NAME is a Blender name, a semantic name (`head`, `wrist.R`, `index2.L`) or a PMX name |
| `obj(NAME)` | world-space view of an object: `.loc` `.matrix` `.quat` `.euler` `.dims` `.bbox` `.visible` |
| `morph(NAME[, armature])` | the value of a shape key on the model's meshes |
| `cam()`, `screen(point[, camera])` | the active camera as an `obj` view; a world point to camera-view coordinates `(x, y, depth)` (x, y in 0..1 from the bottom left of the frame, depth in metres) |
| `dist(a, b)`, `angle(a, b)` | distance (m) and angle (degrees) between vectors |
| `arm(NAME)`, `scene`, `bpy`, `np`, `math`, `Vector`, `fps`, `frame0` | the armature object, the Blender scene and modules, the project's `fps` and `frame0` |
| `frame`, `t` | the frame being evaluated and its clip seconds (`None` without a project) |

`mk q SCENE 'bone("head").head' --frames 181:280` prints the values; `--list armatures|bones|semantic|morphs|cameras|markers|
collections|actions|objects` lists names; `--summary` prints min, max and mean.

## Configuration

`~/.config/mk/config.toml` (all optional), overridden by environment variables:

| Key | Env | Default |
|---|---|---|
| `blender` | `MK_BLENDER` | `~/blender-portable/blender-4.2.3-linux-x64/blender` |
| `mmd_addon` | `MK_MMD_ADDON` | `bl_ext.user_default.mmd_tools` |
| `assets` | `MK_ASSETS` | `~/mk-assets` |
| `player` | `MK_PLAYER` | none: the system's opener (`mk play`) |

Environment only: `MK_CONFIG` (the config file, default `~/.config/mk/config.toml`), `MK_CACHE` (the user scratch folder:
job folders, serve sockets, samples, look output outside a project, reference downloads; default `~/.cache/mk`),
`MK_KEEP_JOBS=1` (keep every job folder). `mk doctor` checks all of it.

## Project file

A project is a folder with an `mk.toml`; commands find it by walking up from the working directory (or `--project DIR`).
Paths are relative to the project folder (`~` works). Every section is optional except `[project]` (`fps`, `frame0`, `duration`).

| Section | Read by | Keys |
|---|---|---|
| `[project]` | everything | `name` (default: the folder's name), `fps`, `frame0`, `duration`, `blend` (where `mk build` saves the scene) |
| `[[output]]` | build, look, render, post, framing | `name` (`"16x9"`), `size = [width, height]`; one per delivered aspect: every shot gets a camera per output, the first sets the scene's default render size |
| `[audio]` | post, timeline, the site | `file`, `start` (song seconds at clip time 0), `hits` (a JSON file of named lists of hit times, clip seconds, each a number or `{t, db}`, a gap lane's with its `dur`, a note lane's with its `midi`: `mk timeline hits` writes one; the site's timeline shows them as lanes), `lanes` (which of those lists, in order; default all but `beats` and `downbeats`), `master` (mastering for small speakers: see [Post](#post)) |
| `[scene]` | scene stage | `start`, `end` (frame range; default: 3 s of pre-roll before `frame0` to the last frame of the clip), `settle_frames` (24: the frames the rest pose takes to ease into the base pose) |
| `[look]` | sets, props, lights, post | `palette`, `slots`, and the colour management keys: see [Lights and the look](#lights-and-the-look) and [Palettes](#palettes) |
| `[render]` | `mk render` | `samples`, `shutter`, `engine`: see [Rendering and post](#rendering-and-post) |
| `[post]` | `mk post` | the grade: see [Rendering and post](#rendering-and-post) |
| `[[fx]]` | `mk render`, `mk post`, `mk look` | frame effects on the finished pictures (lens, echo, slitscan, push, scope, zoom, shock, slab, tint, halftone, split, mosh): see [Frame effects](#frame-effects) |
| `[[medium]]` | `mk build` (shots), `mk render`, `mk look` | a window of clip seconds drawn in another look whatever shot is in the cut: see [Medium shifts](#medium-shifts) |
| `[credits]` | `mk assets credits` | `assets` (slugs the credits must name besides the cast's), `lines` (the project's own lines) |
| `[colliders]` | sim, checks | named lists of collision shapes: see [Colliders](#colliders) |
| `[[check]]` | `mk check` | `name`, `metric`, `args`, `frames`, `min`, `max`: see [Checks](#checks) |
| `[[set]]`, `[[prop]]`, `[[scatter]]`, `[[vehicle]]`, `[[cast]]`, `[pose.<cast>]`, `[[motion.<cast>]]`, `[perform.<cast>]`, `[[shot]]`, `[[transition]]`, `[[insert]]`, `[[glitch]]`, `[[light]]`, `[[text]]`, `[[key]]`, `[sim.<cast>]`, `[[stepped]]` | `mk build` | one build stage each: see [Building](#building) |

```toml
[project]
name = "night_drive"
fps = 30
frame0 = 181            # Blender frame of clip time 0
duration = 24.6         # clip length, seconds
blend = "build/night_drive.blend"

[[output]]
name = "9x16"
size = [1080, 1920]

[[output]]
name = "16x9"
size = [1920, 1080]

[audio]
file = "audio/song.flac"
start = 15.59           # song seconds at clip time 0

[[cast]]                # who is in the scene
name = "rin"
asset = "my_model"      # registry slug (its rig.json); or pmx = "path/to/model.pmx" with rig = "path/to/model.rig.json"

[credits]
assets = ["my_model"]   # everything else the credits must name (audio, props, motions)
lines = ["## Music", "A Band - A Song (used as the soundtrack)", "## Palette", "Rose Pine Moon"]   # credits only the project knows

[colliders]             # named sets of scene collision shapes, see Colliders
cafe = [{ type = "box", object = "ChairColBack", rnd = 0.012 }, { type = "floor", z = 0.0 }]

[[check]]               # see Checks
name = "back hair is calm"
metric = "jitter"
args = { family = "back_hair" }
frames = "181:918"      # optional, default: the clip
max = 0.5
```

The conventional layout of a project folder:

```
mk.toml
audio/timeline.json     mk timeline analyze
tracks/<name>.json      reference data only the project can compute (below)
refs/<set>/             reference photos for modelling (mk ref photos)
ref/<set>.json          measurements of reference clips (mk ref measure)
build/<name>.blend      mk build
renders/<preset>/<output>/<frame>.png   mk render (+ plate/ matte/ back/ point/ screen/ keep/ layers)
out/<name>_<output>[_<preset>].mp4      mk post
.mk/                    cache and look output (not kept in git)
```

Project-specific reference data that only the project can compute (where a nib should be, when a hand writes) goes
to `tracks/<name>.json`: `{"frames": [...], "<channel>": [one value per frame (number, vector or null)], ...}`.
Checks refer to channels as `track:<name>.<channel>`. Dense geometry that a one-position-per-frame track cannot hold, such
as the strokes of handwriting, has a shape of its own (`tracks/ink.json`, see [Ink](#ink)).

## Model description (rig.json)

Written by `mk inspect MODEL.pmx` (to `<assets>/rigs/<folder>__<file>.rig.json`, or `--out`) or `mk assets add` (to
`<assets>/rigs/<slug>.rig.json`, recorded in the registry entry), and by `mk model build` for the characters it makes. Bone
names are the Blender names mmd_tools gives on import (left/right prefixes become `.L`/`.R` suffixes); `jp` keeps the original
PMX name. Positions are in armature space (metres at the import scale); body geometry is in its bone's rest frame, so it
follows the bone wherever the model is placed and posed.

It is one data model in layers (`mkmmd/core/rig.py`). The import describes the first five from any PMX; `mk model` adds
the last two, which a PMX cannot hold, from what its part builders declare (`Part.rig`, PMX bone names, translated when
rig.json is written):

| Layer | Keys | What it is | Read by |
|---|---|---|---|
| identity | `schema` `source` `scale` `comment` `kind` `quirks` `stats` | the file, its import scale, what it is | registry, reports |
| skeleton | `map` `missing` `missing_required` `bones` `fingers` `measure` | bones, semantic names; a helper bone's `grant` | cast, pose, motion, perform, checks |
| colliders | `bodies` | rigid bodies, their geometry in the bone's rest frame | sim, checks, cables |
| dynamics | `chains` | the bones dynamic bodies move; a chain's `ties`: the chains it is tied to sideways | sim, checks |
| morphs | `morphs` `morph_list` | shape keys and morph panels | perform, sing, moves |
| folds | `folds` | shapes a joint angle drives | the folds stage |
| cloth | `cloth` | a garment's simulation copy | the cloth stage |

```jsonc
{
  "schema": 2,
  "source": {"path": "...pmx", "sha1": "...", "name_j": "...", "name_e": "..."},
  "scale": 0.08,
  "comment": "the PMX comment (author notes), first 4000 characters",
  "kind": "character",                                                         // "prop" when the head, arms and legs are not all found
  "map": {"head": "頭", "wrist.R": "手首.R", "index2.L": "人指２.L", ...},   // semantic -> Blender bone
  "missing": ["upper_body3", ...], "missing_required": [],                  // standard bones not found
  "bones": {"頭": {"jp": "頭", "parent": "首", "head": [x, y, z], "tail": [x, y, z], "deform": true,
                  "helper": false},                                          // helper: mmd_tools _dummy_/_shadow_
            "腕捩1.L": {..., "grant": {"parent": "腕捩.L", "ratio": 0.25, "rotate": true, "move": false}}, ...},
  "fingers": {"index.L": ["人指１.L", "人指２.L", "人指３.L", "人差指先.L"], ...},
  "bodies": [{"name": "...", "bone": "頭", "shape": "sphere", "size": [..], "type": 0, "group": 0,
              "no_collide": [1, 2], "loc": [..], "quat": [..],
              "geom": {"kind": "sphere", "c": [..], "R": 0.1}}, ...],       // capsule: a b R; box: M half
  "chains": [{"family": "back_hair", "root": "後髪1", "anchor": "頭", "bones": ["後髪1", ...],
              "parents": [-1, 0, ...], "ends": [[x, y, z], ...], "body_radius": [0.072, ...],
              "branching": false, "locked_joints": 14, "length": 1.009,
              "ties": ["後髪2"]}],                                             // only when joints tie it to others
  "morphs": {"blink": "まばたき", "a": "あ", "i": "い", "smile_eyes": "笑い", ...},   // semantic -> morph
  "morph_list": [{"name": "まばたき", "panel": "eye", "kind": "vertex", "shape_key": true}, ...],
  "measure": {"top": 1.71, "eye_height": 1.32, "upper_arm": 0.21, "forearm": 0.17, "hand": 0.17,
              "thigh": 0.34, "shin": 0.34, "hip_height": 0.78, "shoulder_width": 0.18, ...},
  "quirks": ["...: 2 vertex groups are not bones (mmd_edge_scale, mmd_vertex_order); skip them ...", ...],
  "stats": {"bones": 370, "bodies": 212, "dynamic_bodies": 188, "joints": 355, "chains": 25, ...}
}
```

A bone with a `grant` is a helper bone: it takes `ratio` of its grant parent's rotation (`rotate`) or move (`move`) on top
of its own, as PMX does it (twist bones spreading a forearm's roll, D bones, the helper bones a garment puts at a joint).

A chain is every bone driven by a dynamic rigid body, grouped from each root that hangs from a non-simulated bone
(the anchor). `ends` are the rest segment ends: the head of the simulated child, or for a leaf its tail when the
chain is connected (mmd_tools gives unconnected bones a default 0.08 m tail, so then the leaf continues its chain's
last direction). `length` runs along the bone heads. `ties` lists the chains a joint between their bodies ties this one
to (a skirt's or a sleeve's sideways joints: the chains make one sheet). Families come from bone names (`bangs`,
`side_hair`, `back_hair`, `twintail`, `braid`, `hair`, `ears`, `tail`, `skirt`, `ribbon`, `breasts`, `sleeve`, `coat`,
`accessory`, `other`); projects tune the secondary-motion solver per family (see [Sim](#sim)).

`folds` (mk-built models whose parts sculpt them) lists shapes a joint angle drives: `{"morph": "セーター肘左",
"piece": "sweater", "driver": {...}}`, the morph a vertex morph of the model, the driver one of `mkmmd.core.folds`'
kinds: `{"kind": "angle", "bone", "ref", "dir": [x, y, z], "from", "to"}` (the angle in degrees between the bone's
direction and `dir`, armature space at rest, carried by `ref`: an elbow's bend, an arm raised from hanging) or
`{"kind": "twist", "bone", "ref", "axis", "from", "to"}` (the bone's signed turn relative to `ref` about `axis`: a waist
twisting). The shape's weight runs from 0 at `from` to 1 at `to` (smoothstep). The [folds stage](#folds) keys them from the
final pose of every frame; in MMD they are plain sliders.

`cloth` (mk-built models whose parts make a garment's simulation copy, `mkmmd.model.cloth`) lists one entry per garment:
`{"name": "sweater", "materials": ["セーター"], "bones": [...], "settings": {...}, "file": "carly.cloth.npz", "sha1": "...",
"arrays": ["sweater.sim", ...]}`. Its arrays are in `<model>.cloth.npz` beside the PMX (`mk model build` writes it; a file
whose sha1 is not the entry's is refused): `<name>.sim` the simulated cloth's vertices (model space at rest) and
`.sim_faces` its quads, `.pins` (0..1, where the garment is attached: 1 holds a vertex where its weights carry it, the
cloth under a band, and the faces there keep their skin's motion; 0 leaves it free), `.holds` (the least the free cloth
is held toward where its weights carry it: a sleeve rides an arm faster than cloth can follow), `.collides` (0 or 1:
whether it collides with the body; a sleeve does not, or an arm crossing her chest would push it off), `.sim_weights`
(vertex x `bones`: where the armature carries it), and `.collide`, `.collide_faces`,
`.collide_weights` (the skin under the garment, set in a little). `materials` are the render materials whose faces follow
the simulated cloth, `settings` the garment's cloth (the keys of [Cloth](#cloth)). The [cloth stage](#cloth) simulates
it; MMD has the PMX's own chains and fold shapes.

**Semantic names** (`mkmmd/core/bonemap.py`, matched on the PMX name after NFKC normalisation): single bones `root`,
`view_center`, `center`, `groove`, `waist`, `lower_body`, `upper_body`, `upper_body2`, `upper_body3`, `neck`, `head`,
`eyes`; and, with the suffix `.L` / `.R`, `eye`, `shoulder_p`, `shoulder`, `shoulder_c`, `arm`, `arm_twist`, `arm_twist1..3`,
`elbow`, `wrist_twist`, `wrist_twist1..3`, `wrist`, the finger joints `thumb0..2`, `index1..3`, `middle1..3`, `ring1..3`,
`little1..3` and their tips `thumb_tip`, `index_tip`, `middle_tip`, `ring_tip`, `little_tip`, `leg`, `knee`, `ankle`, `toe`,
`leg_ik`, `toe_ik`, `leg_ik_parent`, `leg_d`, `knee_d`, `ankle_d`, `toe_ex`, `waist_cancel`. `required` ones (head, arms, legs ...)
are listed under `missing_required` when a character lacks them. **Semantic morphs** (`morphs`): `blink`, `wink_l`, `wink_r`,
`wink2_l`, `wink2_r`, `smile_eyes`, `a`, `i`, `u`, `e`, `o`, `a2`, `n`, `cheerful`, `serious`, `troubled`, `angry`, `sad`,
`surprised`, `brow_up`, `brow_down`, `jito`, `calm`, `hau`, `star_eyes`, `heart_eyes`, `pupils_small`, `mouth_smile`,
`mouth_down`, `mouth_wide`, `omega`, `tongue`, `blush`, `tears`, `pale` (each found by the PMX names a model commonly uses).

## Prop card

Every prop has a card. Procedural props ship their cards in their builders (`mkmmd/blender/library/props/`); a card file
(`card = "cards/mug.json"`) or an asset of kind `prop` keeps its own next to the files in the asset library. `schema`, `name`,
`kind` and `license` are metadata; the rest is read by the build, the placement rules, the poses and the checks.

```jsonc
{
  "schema": 1,
  "name": "bentwood_chair",
  "kind": "procedural",                    // metadata: procedural | pmx | blend
  "builder": "library:cafe_chair",         // procedural only: the library builder a card file calls
  "source": null,                          // file path for imported props (a .blend to append from)
  "license": {"text": "...", "credit": "...", "url": "..."},
  "size": [0.43, 0.48, 0.85], "origin": "floor_center", "front": "-Y",
  "use": {                                 // use points: places other tools act on, in the prop's local frame
    "sit":  [{"name": "seat", "hip": [0, 0.04, 0.52], "seat_z": 0.45, "facing": [0, -1, 0], "pelvis_deg": 6, "back_deg": 0}],
    "feet": [{"name": "seat", "L": [0.1, -0.3, null], "R": [-0.1, -0.3, null], "floor_z": 0.0}],
    "grip": [{"name": "rim", "type": "ring", "center": [..], "axis": [..], "radius": 0.19, "tube": 0.015}],
    "rest": [{"name": "door_edge", "type": "edge", "a": [..], "b": [..], "normal": [..]}],
    "look": [{"name": "mirror", "point": [..]}],
    "surface": [{"name": "page", "center": [..], "normal": [..], "up": [..], "size": [0.16, 0.22]}],
    "anchor": [{"name": "jack", "point": [..], "dir": [..]}],
    "wear": [{"name": "stand", "bone": "upper_body2", "pivot": [..], "at": [..]}]
  },
  "colliders": [{"type": "box", "object": "bentwood_chair_col_seat", "rnd": 0.01}],
  "slots": {"wood": "#b4637a", "cane": "#ea9d34"}   // palette slots for look development
}
```

The `use` points, by kind (`use.<kind>`, each an entry with a `name`; a target such as `"car:wheel"` names the prop and the entry):

| Kind | Used by | Fields |
|---|---|---|
| `sit` | `[pose.<cast>] sit` | `hip` (the hips' point), `facing` (default `[0, -1, 0]`), `floor_z` (0), `pelvis_deg` (6), `back_deg` (0); `seat_z` records the cushion height (see [Posing](#posing)) |
| `feet` | `[pose.<cast>] feet` | `L`, `R` ankle targets `[x, y, z or null]`, `floor_z` |
| `grip` | `grip = "prop:name"` of a hand, `mk grip` | `type`: `ring` (`wheel` is an alias), `pen`, `pinch`, `neck`, `strum`: see [Grips](#grips) |
| `rest` | `rest =` of a hand, the placement rules | an `edge` (`a`, `b`, `normal`) or a `plane` (`center`, `size` [w, d] or `radius`): see [Placement](#placement) |
| `look` | gaze, camera and light targets | `point` |
| `surface` | `[[text]] on`, placement on walls | `center`, `normal`, `up`, `size` [w, h]: a flat panel, normal pointing out of it |
| `anchor` | wear, text, transitions | `point` (and `dir`), optionally `bone` and `object` |
| `wear` | `[[prop]] wear` | `name`, `bone`, `pivot`, `at`, `ref`, `scale`, `neck_deg`, `yaw_deg`, `roll_deg`, `strap`, `cable`: see [Playing a worn guitar](#playing-a-worn-guitar) |

Fields the placement rules and the build add (all optional; see [Placement](#placement) and [PMX props](#pmx-props)): `bounds`
{min, max} (the visible geometry in the prop's frame; the props stage measures it when a card has none), `layers` (the plan
boxes per height slice), `front` (`-Y`: the axis a person faces), `flat` (true: other props may stand on or overlap it, a
rug), `blocks` (false: it does not occupy its footprint), `origin` (`floor_center`, `center`, `keep`; `wall_center` for
something hung on a wall: its origin lies on the wall plane), `use.rest` planes with `size` [w, d] or `radius`,
`use.surface` panels, `form_max` and `form_exempt` (the form guard, see [Form](#form)), and for imported props `parts`,
`armature` and `stats`. A **set** card may carry `use.rest` (its `floor`) and `obstacles`, `[{name, min, max}]` axis-aligned
boxes in the set's frame that placed props keep clear of (windows, doors, built-ins), plus `paths` (centre lines with lanes,
see [Sets](#sets)) and `lights`. A project can merge its own entries over any card with `card_extra` (see [Props](#props)). A vehicle's card also carries `wheels` (`[{object, radius, axis}]`) and
`steering` (`{object, axis, ratio}`), which the [vehicles stage](#vehicles) reads.

## Asset registry

`<assets>/registry.json`: a list of entries, one per model, motion, prop, vehicle, audio file, reference clip, texture or font
(`kind`: model, motion, prop, vehicle, audio, reference, texture, font, other). `mk assets` edits it.

```jsonc
{"slug": "my_model", "kind": "model", "path": "/abs/path/model.pmx", "name": "Rin", "author": "NAME",
 "source_url": "...", "license": "...", "restrictions": "...", "credit": "Model: NAME",
 "rig": "rigs/my_model.rig.json", "tags": ["original"], "readmes": ["/abs/path/readme.txt"], "extra": {...}}
```

mk never guesses a license. New entries are `unreviewed` and keep the paths of the readme files found next to the
asset (`mk assets show SLUG --terms` prints the lines that talk about terms of use); someone who has read them fills in
`license`, `restrictions` and `credit` with `mk assets set` (`source_url` is the field `--url` fills). `mk assets credits`
builds the credits from the project's cast and `[credits] assets` (one entry per asset, grouped by kind) and exits 1 while
any of them is unreviewed or has neither an author nor a credit line; `[credits] lines` are the project's own lines (a song,
a print made for it, the palette, tools), appended as bullets under `## Also` or under the `## Heading` lines among them.
Slugs are how the project refers to registered assets: `[[cast]] asset`, `[[motion.<cast>]] vmd`, `[[prop]] card`,
`[[text]] font`.

## Colliders

Scene collision shapes for the secondary-motion solver and the `penetration` check, as specs (inline or a named
`[colliders]` collider set). Shapes ride on their source; rounded edges (`rnd`, m) keep contact smooth. Any spec may carry
a `tag` (default: the object, bone or type) that names it in reports.

| Spec | Shape |
|---|---|
| `{type = "box", object = O, rnd}` | O's mesh bounds (scaled), riding on O |
| `{type = "cylinder", object = O, R?, half_h?, center?, rnd}` | about O's local Z through its origin (R, half height default to the bounds) |
| `{type = "cylinder", center = [x, y, z], R, half_h, rnd}` | vertical, fixed in the world |
| `{type = "capsule", object = O, a, b, R}` | segment a-b in O's axes (metres) |
| `{type = "sphere", object = O, c?, R?}` | on O |
| `{type = "ring", object = O, radius, tube, segments?}` | a torus about O's local Z (a steering wheel), as `segments` (16) capsules |
| `{type = "capsule", bone = B, to = B2, R, armature?}` | from B's head to B2's head, riding on B |
| `{type = "capsule", bone = B, a, b, R, armature?}` | segment a-b (metres, the model's space at rest: x her left, -y front, z up), riding on B: a shape where her skeleton has no two bones to span (over her chest, under a lapel) |
| `{type = "fingers", radius = {thumb, index, middle, ring, little, palm}?, sides?, armature?}` | every finger segment and three palm capsules (wrist to index1, middle1, little1); radii in metres (0.010 0.009 0.009 0.0085 0.008 and 0.016 for the palm) |
| `{type = "floor", z}` | the ground plane |
| `{type = "prop", prop}` | every collider of a prop's card, as the sim stage uses them (stored on the prop's root when the scene is saved) |

The model's own bodies (rig.json `bodies`) collide too. A chain skips a body it already overlaps in the rest pose
(the author's intended overlaps); within `anchor_free` (0.25 m) of its root it also skips the body it hangs from and
every pair the PMX collision masks exclude.

## Checks

A check is a named metric with a threshold. Metrics compute from sampled data or renders and return numbers; `mk
check` prints every result and exits 1 if any fails (a check that cannot be computed fails with an `error`). All sampled
metrics of one run share a single Blender pass over the union of their frames.

```toml
[[check]]
name = "back hair is calm"
metric = "jitter"
args = { family = "back_hair" }
frames = "181:918"       # a frame spec (Conventions); default: the whole clip
max = 0.5                # min and max: the check passes when min <= value <= max
```

```jsonc
{"name": "back hair is calm", "metric": "jitter", "args": {"family": "back_hair"}, "value": 0.38, "max": 0.5, "ok": true,
 "detail": {"at_frame": 412, ...}}
```

Every metric that measures over frames names its worst moment in `detail.at_frame` (the Blender frame of the value; an
image metric, the frame its worst image is named for). A run of the project's checks (not an ad-hoc one, and not one
with `--frames`) keeps its results in `.mk/checks.json`, merged by name into the last runs' (an `--only` run keeps the
others): the site's Checks tab and the music timeline's marks read it.

`mk check --only NAME`, `--frames SPEC` and `mk check [SCENE.blend] METRIC --args JSON --max X` (an ad-hoc check) are the other
ways in; `mk check --list` prints every metric's arguments and default limits. Metrics that read a character take `cast` (a
`[[cast]]` name; the only member by default), or `armature` and `rig` (a rig.json path) outside a project.

| Metric | Value | Notes |
|---|---|---|
| `jitter` | median jerk / median speed of chain points in the head's frame | calm hair < 0.5; Bullet hair on a seated model ~1.4; the detail has `jerk_p95_mm`, and a `static` note when a chain barely moves (the ratio is then float32 noise: judge by `jerk_p95_mm`) |
| `contact` | largest distance (mm) between two points (expressions or tracks) where `when` holds | `component = "z"` for a signed axis difference |
| `penetration` | deepest chain point inside a body or collider (mm) | measured on the baked bones, i.e. what renders, with each chain's own radius; reports the frame, the bone and the shape (`body:<bone>` for the model's own bodies) |
| `foot_slide` | 95th percentile horizontal speed of planted feet (mm/frame) | |
| `joint_limits` | worst excess over a limit (deg) | elbow fold-through and in-plane hyperextension, knees, wrists, neck, spine; limits calibrated on professional MMD dances; sideways elbow bends are not flagged (MMD rigs treat the elbow as a ball joint) |
| `framing` | smallest margin to each output's safe area | per output through the active camera; matches Blender's projection; fractions of the frame, negative = outside the safe area |
| `occlusion` | largest share of subject points hidden from the camera | ray casts; the subject's own meshes do not count |
| `camera_inside` | frames with the camera inside a closed mesh a render shows | hidden objects (colliders) and `ignore` (glow and haze volumes) are passed through |
| `flicker` | worst frame's 99th percentile of temporal luma noise | on rendered frames; motion makes it large too, so compare shots with themselves |
| `palette` | near-black share (or median distance to a palette) | on rendered frames; `rose-pine-moon`, `rose-pine`, `rose-pine-dawn` built in |
| `form` | boxiness 0..1 of a prop's evaluated geometry: area-weighted over loose parts, `1 - (1 - cuboid)(1 - flat_sharp)` | what a render shows at one frame, in the prop's frame; hero props stay under the default `max` 0.25; `detail.worst_parts` names the objects to fix (see [Form](#form)) |
| `strum` | largest distance (mm) of the pick tip from the nearest string at the strike of a down stroke | replays the project's `[perform.<cast>] strum` on its timeline and measures the scene: the pick object rides the wrist, so the arm's reach is in the numbers; `detail.timing` has the mean, median, p95 and max of when the tip really crosses the first string against the planned strike time (ms), `detail.missed` the strokes it never reached; default `max` 10 (see [Playing a worn guitar](#playing-a-worn-guitar)) |
| `prop_body` | deepest vertex (mm) of a prop's geometry inside the character's collision bodies | the torso, hips, legs, neck and head bodies of rig.json (not the arms that hold it), over the frames; the bodies are capsules a little fatter than the skin, so a few millimetres is a prop resting on the body; default `max` 8 |
| `clearance` | least gap (m) between the world bounds of objects that fly past (`objects`: names or fnmatch patterns, e.g. a [bullets](#lyrics) entry's `bul_l*`) and the character's collision bodies, over the frames each object has geometry | exact for the bodies' spheres, capsules and rounded boxes; negative: into a body (by at least that much: where the cores overlap it is minus the body's radius). The bodies are rig.json's, often ending at the ankles and wrists, so shoes and fingers stand out of them; default `min` 0; detail: `object`, `at_frame`, `body`, `object_frames` |
| `wrist_bend` | largest bend (deg) of a hand from its forearm (elbow to wrist against wrist to the middle knuckle) while a move holds it | the moves library's rule (`MV.WRIST`, default `max` 60); detail: `side`, `move` |
| `hand_mirror` | largest mismatch (mm) between a two-hand move's hands, each wrist and middle knuckle against the other's mirror image | default `max` 5; detail: `move`, `deg` (the hands' directions, mirrored) |
| `mic_palm` | smallest share of a fist's palm toward the body while it holds a mic or a hammer (the library's aimed places) | default `min` 0.3; detail: `side`, `move` |
| `hand_room` | smallest distance (mm) from a hand's wrist and finger joints to the character's collision bodies at places that do not touch the body | the trunk bodies of rig.json, a little fatter than the skin; default `min` 0; detail: `side`, `move`, `nearest` |

The four hand metrics read the cast member's `[[move.<cast>]]` (see [Moves](#moves)): a move holds a hand from its `t` to
`t + dur`, and a frame counts while the hand holds still at its place there (under 1.5 mm a frame), so a hand setting off
early for its next move is travelling, which the rules do not cover. The hands ride the upper body (the pose stage keys
their places on it), so mirror and palm are measured in its frame: the world with the upper body's motion undone, in the
armature's axes.

Arguments:

| Metric | Arguments |
|---|---|
| `jitter` | `family` (a chain family or a list), `bones` (explicit bone names, besides or instead), `ref` (reference bone, `head`), `cast` |
| `contact` | `a`, `b` (an [expression](#expressions) or `track:NAME.channel`), `when` (a mask: `track:NAME.channel` or an expression, truthy frames count), `component` (`x`, `y` or `z`), `unit` (`m`, `cm` or `mm`, default mm) |
| `penetration` | `cast`, `families` (default hair and ears), `radius` (`{family: {scale, max}}` or one `{scale, max}`: the point radius is the body radius x scale, at most max, in metres), `colliders` (specs or the name of a collider set), `model_bodies` (true), `use_masks` (false: honour PMX collision masks everywhere, not only near the root), `anchor_free` (0.25 m) |
| `foot_slide` | `cast`, `points` (default `ankle.L ankle.R toe.L toe.R`), `floor` (0 m), `tolerance` (0.015 m: how far above its rest height a point still counts as planted) |
| `joint_limits` | `cast`, `limits` (override `{name: [lo, hi]}` of `elbow_fold`, `elbow_back`, `knee_bend`, `knee_side`, `wrist`, `neck_swing`, `neck_twist`, `spine_swing`, `spine_twist`), `sides` (`L R`) |
| `framing` | `points` (bones or expressions; default head and neck), `cast`, `outputs` (default all), `safe` (0.05: the safe-area margin as a fraction of each side). Give one per shot with `frames` set to that shot's frames: a close-up of a page is not supposed to contain the head |
| `occlusion` | `points` (expressions; default the cast's head and chest), `cast` (its meshes are the subject) or `ignore_models` (armature names), `ignore` (object names that never block: glass, hair cards) |
| `camera_inside` | `ignore` (object names a camera may sit inside) |
| `flicker` | `images` (a glob of consecutive frames), `width` (320) |
| `palette` | `images` (a glob), `palette` (a name or a list of hex colours), `black` (0.035: the luma below which a pixel is near-black), `measure` (`black` or `distance`), `width` (240) |
| `form` | `prop` (a prop root name or a list), `objects`, `exclude`, `exempt`, `exempt_weight`, `frame`, `hard_edge_deg` (65), `worst` (6): see [Form](#form) |
| `strum` | `cast`, `hand` (`R`), `pick` (the pick object, default `<prop>_pick`), `tip` (metres from the pick object's origin to its tip along x; default the card's `pick.tip`, 0.008), `strokes` (`down`: the worst down stroke, or `all`) |
| `prop_body` | `cast`, `prop`, `bones` (semantic bones whose bodies count; default `upper_body upper_body2 lower_body neck head leg.L leg.R knee.L knee.R`), `exclude` (a cable, a pick), `samples` (4000 vertices), `frame` |
| `wrist_bend`, `mic_palm`, `hand_room` | `cast`, `sides` (`L R`) |
| `hand_mirror` | `cast` |

Read results before acting: `penetration` is measured on the baked bones with body capsules fatter than the mesh, so a few
millimetres rarely show: look before fixing. A check passes or fails a number; whether a shot looks right is a question for
[`mk look`](#looking).

### Form

`form` measures how blocky a prop is (the rules and the toolkit that make props pass are in [modelling.md](modelling.md)).
`prop = "car"` takes every object a render shows under that prop root (colliders, objects hidden from render and characters
are left out), `objects = [...]` names objects as they are, `exclude = [...]` drops names or fnmatch patterns, `frame` picks
the frame (default the project's `frame0`). The `sample` op evaluates modifiers as a render does (Subdivision Surface and
Multires at render levels) and returns world-space triangles (`meshes` in its job); the maths is `mkmmd.core.form` (numpy
only, tested without Blender). Triangles are welded per object, grouped into loose parts and grown into planar patches
(neighbouring normals within 3 degrees of the patch's mean); a patch is *large* from (0.12 x the prop's diameter)^2 up, a flat
panel at the size the prop is looked at. Per part:

- `cuboid`: how much of it is a box. Flat faces (within 3 degrees of an axis, fading out by 10) on three orthogonal axes
  must make up 60-85 % of its surface and the second axis must carry 10-22 % (a thin plate is no box). A bevelled or
  filleted cuboid stays a box until its fillets reach a sixth to a fifth of its smallest side; a crowned panel is not flat.
- `flat_sharp`: its surface in large flat patches, each weighed by the share of its own rim that turns more than 65
  degrees (a chamfer turns 45 per edge, a cube corner 90): flat panels with unbevelled edges, such as a table top without
  a rounded rim.

The value is the area-weighted mean over parts. `detail` also gives `flat` (all large flat area), `hard_edges` (sharp share
of the edge length around large patches), `worst_parts` (`object`, `part`, `area_m2`, `share`, `boxiness`, `why`, `size_m`,
`at` in the prop's frame), `worst_objects`, `sharp_panels` (the biggest flat patches with unbevelled rims) and `exempt`.
Hidden or internal surfaces count (no occlusion test): `exclude` what nobody sees. Read `detail.worst_parts` first and fix
the top entry; a prop usually needs three or four rounds.

Limits: `max` defaults to 0.25, the limit for hero props (anything the camera sees large); a project overrides it per check.
Furniture and gadgets that are boxes by nature (a boombox, a nightstand) read 0.2-0.7 even when well made: judge them by
eye and give them their own `max`. Architecture (walls, gantries, barriers) is legitimately boxy: `exempt = [names or
patterns]` measures and lists those objects but weighs them at `exempt_weight` (default 0), as does an object tagged
`mk_form_exempt` (a custom property; `shell.exempt(obj)` in the shell toolkit) or named in `[[prop]] card_extra = {
form_exempt = [...] }`; `exclude` removes objects altogether. Graphic layers (a label, a display segment) are tagged, never
used to hide a form.

The props stage runs the same maths on every library prop it places (`mkmmd/blender/build/form_warn.py`; pmx and appended
.blend props are someone else's modelling and are skipped, and so is a prop over 600 000 triangles) and logs `WARNING prop
'car': form 0.51 > 0.25 (car_body part 25: a box: ...)` when one is over its limit. The prop's card says what is fair: `form_max`
(default 0.25; a thing that is a box by nature says 0.7; 1 or more switches the guard off) and `form_exempt` (object names or
fnmatch patterns counted at weight 0), in the builder's card or the project's `[[prop]] card_extra`. Results are cached in
`<project>/.mk/cache/form/` by a hash of the evaluated geometry in the prop's own frame and of `core/form.py`, so an unchanged
prop costs a mesh evaluation and no analysis.

Calibration on the library props and test scenes:

| Scene (`mk check ... form`) | value |
|---|---|
| a car modelled from stacked cuboids | 0.5-0.9 (`car_mockup`: 0.90) |
| `convertible_80s` (one lofted body shell, swept and framed trim) | 0.14 |
| a car body lofted and subdivided with the shell toolkit (the modelling.md example: long flat flanks, creased shoulder) | 0.08 |
| the most boxy of the café props that pass (the iPod) | 0.09 |
| café mug, chair, table, vase, saucer, plants, page, poster, lights, pen; `electric_guitar`; MMD characters | 0.00-0.02 |
| `hearts` (18 inflated hearts, one shared mesh) | 0.00 |

## Looking

`mk look` renders views without touching the file: the cut (scene camera with its timeline markers) per output, named cameras
(`--cam`), or orbit views around any target [expression](#expressions) relative to a cast member's facing (`--view`). It writes
one JPEG per view, output and frame (`<view>_<output>_<frame>.jpg`), plus optional contact sheets (`--sheet`), strips (`--strip`),
A/B pairs against another scene (`--ab`), side-by-side pairs with reference photos (`--ref`, see
[AGENTS.md](AGENTS.md#modelling-props)) and framing guides (`--guides`). Orbit presets are `(yaw, elevation)` pairs in degrees,
yaw 0 in front of the cast member and 90 to her left:

| Preset | front | 3q | left | back | right | 3q_right | 3q_back | top | low |
|---|---|---|---|---|---|---|---|---|---|
| yaw, elevation | 0, 8 | 35, 12 | 90, 8 | 180, 10 | -90, 8 | -35, 12 | 145, 15 | 0, 80 | 0, -20 |

or `yaw:elev` pairs such as `20:15`. The cut shows a shot's render-time look (silhouette, reflection; see
[Shots](#shots)) as `mk render` will draw it, and previews a frame inside a [`[[transition]]` or `[[insert]]`](#transitions-and-inserts) and frames
with [screen type](#screen-type) composited as `mk post` makes them; `--no-styles` draws every shot as it is lit and
`--no-transitions` leaves the effects out. Output goes to `<project>/.mk/look/<time>/` (outside a project
`~/.cache/mk/look/<time>/`) unless `--out` says otherwise.

### Measuring and marking a render

`mk look --map` also writes `<image>.map.npz` beside each image (`mkmmd/pixmap.py`): the Blender op hands over the image's
camera and its frame's geometry (every mesh that renders, as it renders and its mesh's own rest coordinates, in triangles with
their object, material and polygon), and the CLI rasterises it through that camera, as Blender places it (sensor fit, lens
shift; perspective or orthographic): at each pixel's centre ray the nearest triangle (both faces, transparent materials too),
its exact barycentric weights and its depth. So a pixel tells its material and object, where it is in the world and on the
model at rest, and how many centimetres it spans. `mk measure MAP` reads it: `--edges MATS --x A:B[:STEP]` (the first and last
row of the materials per column: pixels, the size as seen and between the two surface points, in the world and at rest, and
the runs), `--extent` (per row), `--gaps` (per row, the runs showing no surface, or none of MATS, between covered pixels: the
space under an arm), `--at X,Y`, `--dist X,Y:X,Y` and `--where X,Y,Z` (a world point in the picture, hidden or not); a
material it lacks is an error that lists those it has. `mk mark IMAGE` draws on any image (a render, a photo, a drawing):
`--grid [STEP]` the image's own pixel coordinates, `--crop X0,Y0,X1,Y1 --zoom K` a close look that keeps them, `--marks
FILE.toml` a plan (`mkmmd/marks.py`: `[[line]]` points, dashed, smooth Catmull-Rom curves, named; `[[fill]]` points or
`between` two named lines; `[[arrow]]`, `[[circle]]`, `[[box]]`, `[[text]]`; a point is `[x, y]` in pixels or `[x, y, z]` in
the world's metres through the image's map; a misspelt key is an error) and `--ref PHOTO` the photo beside it at the same
height.

## Building

`mk build` assembles the scene from `mk.toml` in one Blender session and saves it to `[project] blend` (`--out` saves elsewhere,
`--until STAGE` stops after a stage and saves, `--skip STAGE` leaves one out). It prints the per-stage report and the build log as
JSON; `WARNING` lines in the log are findings (a hand short of its goal, a prop over its form limit, a pattern that matches
nothing). Solvers (secondary motion, grips, the pen) run outside Blender as `python -m <module> IN.npz OUT.npz` on the CLI's
Python and are cached in `<project>/.mk/cache/` by a hash of their inputs.

### Stages

Stages run in this order and each reads its own sections; a stage sees what the earlier ones made (a target that names a cast
member or the camera cannot be used by a stage that runs before them):

| Stage | Sections | Does | Reference |
|---|---|---|---|
| scene | `[scene]`, `[[output]]` | empty scene, fps, frame range including the pre-roll before `frame0`, default render size | [Scene](#scene) |
| sets | `[[set]]` | library set builders with their paths, surfaces and lights | [Sets](#sets) |
| props | `[[prop]]`, `[[scatter]]` | library props (`library:<name>`), card files, MMD accessory models; use points, colliders, `bounds`; placement rules and clutter with a per-prop report | [Props](#props), [Placement](#placement), [PMX props](#pmx-props) |
| vehicles | `[[vehicle]]` | a prop drives a set path's lane: position and heading per frame, roll and pitch, spinning wheels, a steering wheel that follows the curves | [Vehicles](#vehicles) |
| cast | `[[cast]]` | models imported without Bullet, named, placed; morph sliders bound when the model has group, bone or material morphs | [Cast](#cast) |
| pose | `[pose.<cast>]`, `[[move.<cast>]]`, `[[prop]] wear / attach / anchor_to` | the base pose: sit on a seat or stand, feet, spine, head, arm IK, finger presets, grips, drape, worn props; named moves on the clock | [Posing](#posing), [Moves](#moves), [Grips](#grips), [Playing a worn guitar](#playing-a-worn-guitar) |
| motion | `[[motion.<cast>]]` | VMD motions on NLA strips | [Motion](#motion) |
| perform | `[perform.<cast>]` | gaze, blinks, breathing, sway, nods, beat bob, startles, expressions, lip sync, twitches, strumming | [Perform](#perform) |
| shots | `[[shot]]`, `[[transition]]`, `[[insert]]`, `[[glitch]]`, `[vector]`, `[[ring]]`, `[[freeze]]`, `[[medium]]` | the cut: a camera per shot and output, markers, render-time looks, cut effects, medium windows | [Shots](#shots) |
| lights | `[[light]]`, `[look]` | lights in palette colours (mounted, aimed, keyed); view transform, contrast look, exposure | [Lights and the look](#lights-and-the-look) |
| text | `[[text]]` | type on set and prop surfaces, lyric type, handwriting, screen type | [Text](#text) |
| keys | `[[key]]` | keys on set, prop and object properties | [Keys](#keys) |
| sim | `[sim.<cast>]` | secondary motion solved outside Blender and baked to keys | [Sim](#sim) |
| folds | `[folds.<cast>]` | the fold shapes a joint angle drives (rig.json `folds`), keyed from the final pose | [Folds](#folds) |
| cloth | `[cloth.<cast>]` | garments' cloth copies (rig.json `cloth`) simulated with Blender's cloth, the garment's faces following it | [Cloth](#cloth) |
| stepped | `[[stepped]]` | the cast on twos: each member's whole motion held 1 / fps s at a time over a window, the cameras on every frame | [Stepped motion](#stepped-motion) |
| save | | the `.blend` | |
| bake | | `<blend>.bake/`: the clip for the site's scene viewer and mk's own renderer, every frame's bones, morphs, objects, visibility and shot cameras, the meshes, what the glb cannot carry as Blender evaluates it on each frame, what the vector look needs (each object's record, the materials' viewport alpha and textures, the rings' centres, the studio light) and the [[medium]] windows (`mk build --bake` bakes a saved scene again) | [The site](#the-site-mk-site-mkmmdsite), [Rendering](#rendering) |

The text stage runs after `lights` and before `keys`, so a `[[key]]` can toggle a text's `hide_render` or move it. A stage
that has nothing to read for a cast member (no `[pose.rin]`, no `[perform.rin]`) leaves it alone, and a `[pose.x]`, `[perform.x]` or `[sim.x]` named after no cast member is ignored
without a word (see [Cast](#cast)).

### Targets

A target is anything a pose, a gaze, a camera, a light or a text can point at. Targets resolve at the frame the stage is on, so
a target that rides a moving prop or a posed bone moves with it.

| Form | Meaning |
|---|---|
| `[x, y, z]` | a world point |
| `{prop = "car", point = [..]}` | a point in a prop's local frame (follows the prop) |
| `{cast = "rin", point = [..]}` | a point in a cast member's own frame: its model root, at `[[cast]] at` and turned by `yaw`; x its left, y behind it, z up. A pose, a gaze or a camera written this way keeps working when the character is moved to another place in the world (a hand's `dir` and `palm` stay world directions) |
| `{path = "road:road", s = 640, offset = -6.0, z = 1.2}` | a point beside a set's path: arc length `s` (m), `offset` (m left of the centre line, or a lane name such as `"fwd1"`), `z` above the road |
| `"car:road"` | a prop use point by name (the first match in the order look, rest, grip, sit, anchor, surface, pose); a use point that rides an object follows it |
| `"cast:rin"`, `"cast:rin.head"` | another character's eyes, or one of their bones (a semantic name) |
| `"camera"` | the active camera at that frame; it exists only from the lights stage on (after shots, and only when the project has a `[[shot]]`), so only a `[[light]] look` can use it; poses, performance, placement and shots stop with a BuildError that says so |

A `[[shot]] at` takes a list or a dict target only (no strings); a list of three numbers as `frame.subject` reads as three targets (write `[[x, y, z]]`): see [Shots](#shots).

### Scene

`[scene]` (all optional) sets the frame range: `start` (default `max(1, frame0 - 3 * fps)`: three seconds of pre-roll), `end`
(default the clip's last frame, `frame0 + round(duration * fps) - 1`) and `settle_frames` (24: the character eases from its rest
pose into the base pose over these pre-roll frames, the hair settles in the same time). The stage clears the scene, sets fps and
the frame range, takes the render size from the first `[[output]]`, selects EEVEE Next and metric units and removes timeline
markers. The scene's custom property `mk_frame0` is the frame of clip second 0: library props whose drivers run on the song's
clock (`heart`) and that key events at clip seconds (`meteor`, `airplane`) read it.

### Sets

`[[set]]` places a library set builder: `name` (the set root's name, also the prefix of its objects), `kind` (the builder),
`at` ([x, y, z], default the origin), `yaw` (degrees about Z) and the builder's own keys (the highway rejects keys it does not know).
Colours are palette slot names or `#hex` (and `"slot:slot:t"` mixes two); builders take the project palette from
`[look]`. A set returns a card: `paths` (centre lines with lanes, for vehicles and targets), `use` points (`look`, `surface`,
`rest`), `colliders`, `lights`, and for rooms `obstacles`; the stage report lists them. Library sets (`kind`):

| `kind` | What it is |
|---|---|
| `test_road` | a flat two-lane S-curve on a ground plane, for vehicle and camera tests: `length` (600 m), `lane_width` (3.6), `curve` (the lateral swing of the S, 40 m); path `road` with lanes `R1` (with the path) and `L1` |
| `night_sky` | the scene's world: a gradient dome, a city glow, thin lit clouds, stars, the moon with its halo and an optional moonlight, EEVEE ray tracing; nothing spatial (the sky is in world angles, `at` and `yaw` do not turn it) |
| `highway` | a divided night highway along a centre line, built to scale (3 km and more) from a few merged meshes: lanes as card paths, lamps with baked spill, gantries and billboards as `use.surface`, tunnels, trees, wet asphalt |
| `skyline` | a distant night city (an arc or a band of buildings with shader-lit windows, glow, beacons) around the set root |
| `hills` | layered ridges on the horizon: curtains round `center` (x, y) at each layer's `distance`, rolling ridge lines up to `height` (`seed`, `rough` 0..1), an `arc` = [from, to] degrees (azimuth from +X toward +Y) that can leave a city's direction open; flat colours (`color`, "base") pulled toward the horizon haze by `haze` (rising with distance), so they read as silhouettes whatever lights the scene |
| `cafe_room` | the rainy corner café: a window with rain and fog, the street behind it, lightning rigs, a pendant lamp, the room lights; its animatable state is custom properties on the set root |
| `bedroom_80s` | an 80s bedroom at night: striped wallpaper, parquet, trim, a door, a window with a half-raised venetian blind and a lit city behind it, a neon tube; a shell only, the furniture is props |
| `sweep` | a photo studio's seamless paper: the floor from y = -`depth` to the curve, a cove of `radius` up into a wall at y = `radius` that stands to `height`, `width` along x (8, 6, 4 and 1.2 m), `color` ("base"), `rough` (0.8), `segments` of the cove (24); every face turned to the subject, who stands in front of the cove (toward -y, the camera further); `use.surface` `wall`, `use.look` `centre`. An unknown key is an error that lists the known ones |

**`night_sky`** keys (all optional): `zenith` ("base"), `dome` ("surface"), `horizon` (the haze colour the other night sets fade
distant things into), `band` (4 degrees); `stars` (`{density 4 per 100 square degrees, size 0.07, brightness 1.0, fade [3, 18],
color "text", seed 1}` or `false`); `moon` (`{az -74, el 13, size 4.0, color "text", strength 3.0, halo 1.0, light 0.3, shadow true,
seed 1}` or `false`; `az` 0 = +X, counter-clockwise from above); `glow` (`{az -90, width 60, height 2.8, core "gold", rim "rose",
strength 1.0, haze 0.1}` or `false`); `clouds` (`{cover 0.4, softness 0.5, opacity 0.35, drift 0.3, wind 20, seed 1}` or `false`);
`eevee` (`{raytracing true, trace_scale 1, quality 0.5, probe 2048}` or `false` to leave the renderer settings alone). Card:
`use.look` `moon` and `glow`.

**`highway`** keys (defaults in parentheses; unknown keys raise an error):

| Group | Keys |
|---|---|
| Centre line | `points` (control points of a Catmull-Rom spline) or a gently winding road from `seed` (0), `length` (2000), `turn_deg` (12), `leg` (250), `z` (0) |
| Cross-section | `lanes` (2, with the path), `oncoming` (= lanes; 0 = one-way), `lane_width` (3.6), `shoulder` (2.5), `inner_shoulder` (0.9), `divided` (true), `median` (3.0), `drive` ("right" or "left"), `dash` ([3.0, 12.0]: paint, period). Lanes are named `fwd1..` (with the path) and `opp1..`, 1 being next to the median |
| Roadside | `drop` (0.6), `guardrails` (true), `barrier` (true), `ds` (3: station spacing), `trees` (`{density 9 per 100 m per side, seed, near 11, far 70, height [6, 14], pine 0.6}` or `false`); `studs` (`true` or `{spacing 12, size 0.11, strength 2.4}`: retroreflective studs on the lane lines, white, and on the median-side edges, gold); `overpasses` (`[{s, clearance 5.5, width 11, deck 1.0, overhang 14, parapet 1.0, lamp 450, name}]`: road bridges with gold strip lights along both fascias and a real warm lamp under the deck over each carriageway; lamp poles near a deck are dropped); `signs` (`[{s, side "right", size [3.4, 1.8], height 2.2, offset, name}]`: boards on two posts, each a `use.surface` `<name>_panel`); `markers` (`true` or `{every 160, first 40, side "right"}`: mile-marker plates); `pylons` (`true` or `{side "left", offset 60, spacing 200, first 40, height 34, light 40}`: a power line, lattice towers with three sagging cables and a blinking red light on each); `masts` (`[{s, offset -120, height 95, lights 4, light 60, name}]`: radio masts with blinking red lights). Built by `mkmmd/blender/library/sets/roadside.py` |
| Lamps | `lamps` (`{spacing 40, first 20, height 9, arm 3.2, layout "auto" / "outer" / "left" / "right" / "stagger", power 5000, strength 1500, halo 4, halo_strength 2, haze 10, haze_strength 0.035, core 0.42, core_strength 40, soft 1.1, soft_strength 7}` or `false`) |
| Real lights | `lights` (`{every 4, range [s0, s1], max 16, power, reach 90, shadow false, specular 0.5}`): every n-th lamp gets a real spot, the others are lit by baked light (`spill` vertex attribute) |
| Signs | `gantries` (`[{s, panels 2, panel [4.2, 2.4], clearance 5.6, span "forward" or "full", name}]`), `billboards` (`[{s, side, offset, size [12, 5], height 6, angle 12, name}]`); each panel is an object `<set>_<name>_panel<j>` and a card `use.surface` (`gantry1_panel1`, `billboard1_panel`) |
| Tunnels | `tunnels` (`[{from, to}]` arc lengths), `tunnel` (`{wall 3.9, crown 6.4, strip_z 3.3, strength 1.4, power 55, floor 0.25, hill 0.9, ceiling false, lights 0}`) |
| Look | `wet` (0.7), `gloss` (1.3), `floor` (0.32: unlit surfaces show this share of their palette colour), `haze` (`{distance 450, cap 1}`), `slots` (role to slot: `asphalt`, `asphalt_hi`, `paint`, `yellow`, `steel`, `concrete`, `ground`, `sign`, `sign_edge`, `lamp`, `spill`, `tree`, `billboard`, `billboard_edge`, `portal_ring`, ...) |

The card has `paths.road` (`points`, `width`, `lanes`, `length`, `tunnels`), `use.look` (points ahead on the first forward lane),
`use.surface` (every sign panel), `lights` and a floor collider when the road is level.

**`skyline`** keys (all optional): `shape` ("arc" or "band"), `az` (-90: heading of the centre, 0 = +X), `arc` (110 degrees), `width` (1600
m of a band), `distance` (800, keep under about 900), `depth` (260), `count` (160) or `density` (buildings per km2), `height` ([20, 170]),
`footprint` ([16, 48]), `core` (0.6), `seed` (1), `lit` (0.35: share of lit windows), `window` (`{gold 6, rose 1, foam 1.2, text 0.8,
strength 1.5}`: relative weights of the lit colours), `aviation` (12 red lights), `haze` (`{colour, distance, cap 0.55}` or `false`),
`backdrop` (true or `{strength}`), `ground` (true). Objects `<name>_city`, `_glow`, `_land`, `_beacons`, `_halos`; card
`use.look` `downtown` and `skyline`.

**`cafe_room`** keys (all optional): `frame0` and `duration` (default the project's), `ticks` (clip seconds at which a rain splat lands
on the glass; default every 1.5 s) or `timeline` (a file whose `ticks` or `tempo.ticks` are read, else its `beats`, the only tempo marks
`mk timeline analyze` writes: a splat on every beat), `seed` (4242), `pendant` ([x, y], default [0, -0.48]), `render` (true: configure EEVEE Next and AgX
"Base Contrast"), `rain` and `lightning` (true), `colors` (role to slot or `#hex`: `plaster`, `wainscot`, `window_paint`, `oak`,
`oak_dark`, `cream`, `fog_tint`, ...). Room frame: a character sits at the origin facing -Y, +X is her left, the window wall is
x = -0.80, the wall behind her y = 1.05; the big window spans y -1.75..0.65, z 0.82..2.85 and a small round table is expected at
(0, -0.48). Animatable custom properties on the set root (key them with `[[key]]`; the initial values are also spec keys):
`steam` (0..1, 0.6: the `cafe_mug` prop follows it), `fog` (0.12), `outside_sat` (1), `sun` (0), `bolt` (0), `bolt_variant` (0 or 1),
`bolt_rig` (0 front, 1 behind, 2 side), `flash` (0), `refl` (0: her reflection in the pane), `clip_t` (seconds since clip start, driven by the
frame). Card: `use.look` `window`, `street`, `pendant`, `sky`; `use.surface` the 16 panes (`pane_r1c3`, ...) and `back_wall`.

**`bedroom_80s`** keys (all optional): `size` ([3.6, 3.2, 2.55]: interior width, depth, height), `wall` (0.2), `open` (walls and parts
to leave out: "front", "back", "left", "right", "ceiling", "floor"), `window` (`{x 0.4, width 1.3, height 1.2, sill 0.95, cols 2, rows
2}`), `door` (`{y -1.0, width 0.9, height 2.05}`), `floor` ("herringbone" or "boards"), `pendant` (true), `neon` (true, or a level 0..1), `neon_wall`
("left"), `fill` (1.0), `seed` (80), `render` (true), `city` (true, false, or a table of `skyline` keys plus `drop` 10), `near` (the near
layer of low buildings; follows `city`), `sky` (true, false, or `night_sky` keys), `colors` (role to slot: `paper`, `trim`, `door`,
`floor_a`, `neon`, ..., and the light tints), and the initial values of the animatable properties. Frame: floor z = 0, root at the middle of
the room, the window wall is y = +1.6, the door wall x = +1.8, the bed's wall x = -1.8. Animatable custom properties on the set root:
`blinds` (0..1, 0.5: 0 lowered, 1 raised), `slat_angle` (-85..85 degrees, 20), `window_glow` (0..3, 1: the light coming in), `city_glow` (0..3, 1),
`neon` (0..1, 0.7). Card: `use.rest` `floor`, `use.surface` `wall_back` `wall_left` `wall_right` `wall_front` (normals into the room) and `window`,
`use.look` `window`, `city`, `desk_zone`, `bed_zone`, `room`, and `obstacles` (the window and the door, which the placement rules keep props
out of). The architecture is tagged `mk_form_exempt`. `mk look` orbit views clip at 100 m and cut the city away: look through a `[[shot]]` camera or
`--cam`.

### Props

`[[prop]]` puts a prop in the world. The `card` says what the prop is:

| `card` | Prop |
|---|---|
| `"library:<name>"` | a procedural prop of the library (below); its builder gets the project palette plus `slots` |
| `"cards/mug.json"` | a card file whose `builder` is `"library:<name>"` (that builder's card wins over keys of the file with the same name) or whose `source` is a `.blend` to append from |
| `"pmx:PATH"` or a `.pmx` / `.pmd` path | an MMD accessory model (see [PMX props](#pmx-props)) |
| a registry slug | an asset of kind `prop`: its path is a model, a card file or a folder holding a `card.json` |

| Key | Meaning |
|---|---|
| `name` | the prop's name (unique); its root empty and the prefix of its objects |
| `card` | as above |
| `at`, `yaw`, `rot` | world position (default the origin); rotation about Z in degrees, or `rot = [rx, ry, rz]` (overrides `yaw`). `at` may be a point beside a set's path, `{path = "road:road", s, offset, z}` (as a target): then `yaw` turns the prop from facing the traffic that comes along the path (0: its front, -Y, looks back down the road at the cars coming) |
| `parent` | an object the root is parented to (it must exist: props are built in order) |
| `slots` | `{slot: "#hex"}` colour overrides, and the builder's non-colour options (below); ignored, with a warning, for a PMX prop |
| `card_extra` | a table merged over the card: tables key by key, lists of entries with a `name` by name (a known name replaces, a new one is appended, an empty list clears), anything else replaces |
| `place` | `{on, at, facing, align, clear, avoid, distance, bearing, seed}` instead of `at` / `rot`: where the prop lands on a surface (see [Placement](#placement)) |
| `scale` | a library or card prop: a uniform scale of its root, so its use points, surfaces and object colliders grow with it (a collider's own numbers, such as a box's rounding, stay as given); give `at`, not `place`, which fits a prop by its unscaled footprint. A PMX prop: the import's scale (see [PMX props](#pmx-props)) |
| `origin` | PMX props only (see [PMX props](#pmx-props)) |
| `wear`, `attach`, `offset`, `attach_rot`, `anchor_to` | put the prop on a cast member: applied by the pose stage once the cast exists (see [Posing](#posing) and [Playing a worn guitar](#playing-a-worn-guitar)) |

`[[scatter]]` fills a surface with clutter (see [Placement](#placement)). The stage measures `bounds`, `size` and the height `layers` of
every prop that has none (the visible geometry in its own frame), runs the [form guard](#form) on every library prop it places,
stores each prop's use points and colliders on its root for the checks, and reports per prop its size, use points, colliders and,
for placed props, where they landed. Use points stay in the prop's local frame; IK targets parented to the root follow a moving
prop. Library props (`library:<key>`); colour options are palette slots, the non-colour keys of `slots` are listed per prop:

| `library:` | Prop |
|---|---|
| `chair` | a bentwood café chair (round seat, top 0.45 m) |
| `car_mockup` | an open two-seat cabin of boxes for tests (seats two, steers, rests an arm on a door, tests wind): it trips the form guard on purpose |
| `convertible_80s` | an 80s convertible (Dodge 600 proportions, top folded), left-hand drive with the driver on +X, forward -Y: two seats, wheels, steering, a dash with `SPEED_GRID` / `BAR_GRAPH` surfaces and a cassette. Roles in `slots`: `body`, `lower`, `stripe`, `trim`, `rubber`, `interior`, `display`, `headlamp`, `taillamp`, `glass`, `wheel`, `boot`. Animatable root properties (key with `[[key]]`): `lamps` (0..1, with a warm-up flicker), `brake`, `tails` (0.35), `dash_on`, `tape` (0 held out, 1 pushed in), `bars`, `visors` (0 flipped up, the default, 1 down). Card `use`: `sit` driver / passenger, `feet`, `grip` `wheel`, `rest` `sill_L` / `sill_R`, `look` (`road`, `mirror`, `dash`, `cassette`, `headlamp_L` / `_R`, `tail`), `surface` (`speed`, `bars`, `cluster`, `radio`, `cassette_label`, `plate_front`, `plate_rear`), `anchor`; `wheels` and `steering` for the [vehicles stage](#vehicles) |
| `electric_guitar` | an 80s Strat-type electric, upright on its tail (Z up along the strings, face toward -Y, origin at the saddle line). Roles in `slots`: `body`, `pickguard`, `neck`, `fretboard`, `hardware`, `strings`, `knobs`, `strap`, `cable`, `pick`, `inlay`, `dark`. Card: `use.grip` `neck` and `strum`, `use.anchor`, `use.look`, `use.wear`, hidden colliders; see [Playing a worn guitar](#playing-a-worn-guitar) |
| `hearts` | a field of puffy hearts that rise and sway around a point, animated by drivers on the frame (see below) |
| `heart` | one big puffy heart that pounds on the song's beat (a bump on the beat and a smaller one after it) and bursts into a ring of little hearts every few beats; options `size` (0.2), `bpm`, `start` (a clip second on a beat), `beat` (0.25: the swell), `burst` (8 hearts), `every` (4 beats), `reach` (0.45 m), `burst_size`, `turn`, colour `heart` ("love"), `glow`. Drivers on the frame and the scene's `mk_frame0`; objects `<name>_heart`, `<name>_burst<i>` tagged `mk_heart` (silhouette accents: `accent = ["prop:mk_heart"]`); with `attach` it rides a chest |
| `traffic_car` | an 80s car or truck for night traffic: `body` "sedan", "wagon" or "semi" (a cab-over tractor and a box trailer whose sides are `use.surface` `left` / `right`), colours `paint`, `trailer`, `glass`, `lamp`, `tail`, `marker`, emissions `lamps` (30), `tails` (9), `markers` (3.5), `beams` (real headlight spots: `beam_power` 900 W, `beam_angle` 55, `beam_tilt` 5). Faces -Y, origin under the middle; the card lists the wheels and beams, so a `[[vehicle]]` drives it |
| `neon_sign` | a roadside motel / diner sign: a tall pole, a rounded board ringed by a neon tube and a smaller board under it, both `use.surface` (`face`, `sub`) for glowing [[text]]; options `width` (3.2), `tall` (1.5), `height` (6.5), `sub` ([2.0, 0.55], [] none), colours `board`, `neon`, `sub_neon`, `pole`, `glow` (7) |
| `meteor`, `airplane` | sky events at clip seconds, seen from the root: a shooting star (`t`, `dur`, `from` / `to` [azimuth, elevation] of its head, `length`, `color`, `strength`, `distance` 2500) and an airliner's lights crossing (`t0`, `t1`, `from`, `to`, `lights`, colours `red`, `green`, `white` (a strobe), `strength`, `distance` 3000). Keyed at build from `mk_frame0` |
| `line` | the space kit's line (`mkmmd/core/space.py`): a thin glowing line along +X from the root, `length` (40 m) long, `radius` (0.012), made of `seg` (0.25 m) segments `<name>_seg<k>`, with tick marks `<name>_tick<j>` every `ticks` m (0: none) across it, every `bar`-th (4) long (`tick` [short, long] 0.05, 0.12 m). Root property `extent` (0..1, key it): how much of the line there is from its `anchor` end (`"end"`, its +X end, or `"start"`): it retracts into that end, each segment shrinking toward its anchor-side end and each tick vanishing as the line passes it (drivers). `ripples = [[t, x, amp], ...]`: at clip second t a pulse runs out both ways from x metres along the line, a Ricker wavelet `amp` m high (negative: a dip with a lift either side), `ripple = {speed = 7, decay = 0.3, width = 0.5, life = 1.0}`; each segment is tilted to meet its neighbours (keyed). Colour `color` ("text") |
| `hole` | a black hole's shadow: a sphere of `radius` (1.5 m) in `color` (black, no highlight) as `<name>_shadow` (tagged `mk_hole` = its radius: the [lens effect](#effects) bends the light round it). Root property `breathe` scales it (key it on the kicks). `use.look` `centre` |
| `round_board` | a round chessboard round the origin (Byzantine chess is 4 rings of 16): `rings` (6) rings `<name>_r<i>` of `sectors` (16, even) cells from `inner` (2.2 m) out, each `width` (0.8) deep and `thick` (0.05), `gap` (0.03) between cells, light (`light`, "text") and dark (`dark`, black) alternating round a ring and from ring to ring. Root property `phase` (key it): ring i turns `omega_i * phase` radians about the centre, `omega_i = omega * (r_i / r_0) ** kepler` (`omega` 1, `kepler` -1.5: inner rings orbit faster and the board winds into a spiral). Every cell has an overlay `<name>_hot_r<i>s<j>` in `hot` ("love"), hidden: a `[[key]]` on its `hide_render` lights an attacked square. A prop `parent = "<name>_r<i>"` rides a ring. `use.rest` `top` (a plane), `use.look` `centre` |
| `chess` | a chess piece standing on its origin, `height` (1 m) tall: `piece` `pawn`, `rook`, `bishop`, `queen` or `king` turned on a lathe (Staunton-like), or `knight` (a lathed base under a horse's head, a thick plate domed on both faces, facing +X), glossy (`rough` 0.12, `metal` 0, a clear coat) in `color` ("text"); `<name>_piece` |
| `stars` | flat stars in the picture's plane, facing -Y: `points = [[x, z, size, t], ...]` (metres in the root's x-z plane, a size factor, and optionally the clip second it flashes: up to `gain` (1.8) times its size and back over `flash` (0.35) s), `kind` points (4), `inner` radius (0.3 of the outer), `size` (0.12 m), `depth` (0.004), glowing `color` ("text"); with `lines` (true) each star is joined to the next by a rod `<name>_line<i>` (`line_width` 0.01) that draws itself over `draw` (0.12) s as the next star flashes: a constellation, a melody as a piano roll |
| `floor` | a floor to the horizon at z = 0 under the root, `size` (400 m) each way from it (`<name>_floor`): `kind` `mirror` (default: dark glass that shows what stands on it; the floor switches on EEVEE's screen-space ray tracing, which mirrors what the camera sees), `chess` (light and dark squares `cell` (2 m) across) or `grid` (glowing lines every `cell` m, `line` (0.04 m) wide, at `glow` (3) in `color`). Colours `color` (the light squares or the lines; "text") and `dark` (black); `rough` (mirror 0.03, chess 0.12, grid 0.08), `metal` (0). The pattern blends its edges over a few pixels' footprint at the camera's distance, so far away a checker becomes its own grey and a grid dims instead of shimmering (`mkmmd/core/space.py` `checker`, `grid_lines`), and it fades into `dark` between the two `fade` distances from the camera (`[40, 220]` m). `use.rest` `top`. Floors that change with the song are three floors with `hide_render` keys on their `<name>_floor` objects |
| `starfield` | stars streaming past the root along -X at `speed` (12 m/s), plus `surge = [[t, metres], ...]` of extra way (eased between the keys: a jump to light speed). `count` (900) stars per tile, `size` (0.035 m) across and `streak` (0) m long along X, in `color` ("text") glowing at `glow` (12), spread over `length` (160 m) along X, `width` (90) across and `height` (`[-1, 45]`), none within `clear` (5 m) of the X axis, where a character walks; `seed` (1). Two tiles `<name>_tile0` / `<name>_tile1` of the same stars leapfrog round the root (their X keyed per frame from the clock), so the field never runs out: together they always cover `-length` .. `length` |
| `disc` | an accretion disc round a black hole: a flat ring `<name>_disc` from `inner` (2 m) to `outer` (7) about the root, white-hot (`hot`) inside going to `color` ("love") outward and dimming, at `glow` (5), its streaks (a noise stretched along the orbit: `streaks` (3) round it, `rings` (22) across) orbiting `spin` (0.3) turns a second at the inner rim, faster inside (angular speed as `(r / inner) ** kepler`, `kepler` -1.5): a driver on the frame runs the clock. Additive (it lights what is behind it and hides nothing); tilt it with `rot`. `use.look` `centre` |
| `bedroom_desk` | an 80s writing desk, 1.20 x 0.60 x 0.74 m, three drawers; options `edge` (slot of the band round the top, "pine"), `drawers` (three slots, "foam,love,gold"); `use.rest` `top` and `front` |
| `desk_chair` | a chrome cantilever chair with padded seat and back; seats a character exactly as `cafe_chair` does; options `upholstery` ("iris"), `accent` ("love") |
| `desk_lamp` | a spring-arm desk lamp that owns a warm spot light; options `paint` (slot or `#hex`, "love"), `power`; root properties `power` (60 W) and `on` (0..1: key it to switch the lamp); yaw aims it (0 looks toward -Y) |
| `cassette_player` | a compact radio-cassette player (a boombox), 0.34 x 0.115 x 0.16 m; option `label` (the cassette's label colour); root properties `glow` (0..1) and `pump` (0..1: both speaker grilles, `<name>_grille_L` and `_R`, swell 12 % about their centres and come 4 mm out of the front; key it on the kicks); `use.surface` `dial` |
| `cassette_tape` | a compact cassette lying on the desk; option `label` (a slot or `#hex`; by default picked from the instance name) |
| `handheld_mic` | a handheld vocal microphone, 162 mm long with a 51 mm ball grille, standing on its tail along +Z, and its cord `<name>_cable`; colour roles `body`, `grille`, `hardware`, `cable` (`slots`: a palette slot or `#hex`); `use.anchor` `jack` (the cord's exit under the tail), `use.look` `grille` and `grip`. Put it in a hand with `attach` and `cable` (see [Posing](#posing)) |
| `fuse` | a cartoon fuse: a rope (`<name>_cord`, 14 mm thick) that leaves a socket at the root, up and over in a curl, with a spiky spark (`<name>_spark`) that is lit at its tip and burns it down. Root properties for `[[key]]`, 0..1: `grow` (how much of the rope has sprouted, along its length; default 1), `burn` (how much has burned down from the tip; 0), `lit` (the spark's size, flickering; 0). Leave `grow` at 1 in the slots and key it from 0: the props stage measures the form as built. Options `points` (`[[x, y, z], ...]` from the root, the first `[0, 0, 0]`), `radius`; colour roles `cord` (text), `spark` (gold), `cap` (base); `use.look` `base` and `tip`. The spark has no parent (it rides the rope's end on a Follow Path constraint and copies the root's world scale), so `scale` and `parent` carry it too |
| `burst` | a blast of pieces flung out from the root at a clip second: puffy hearts, five-point stars, cloud puffs and shards of glass (pillows facing -Y, and thin flat slivers with sharp edges, mirror-bright metal tinted their colour with a glow of their own: `<name>_heart<i>`, `<name>_star<i>`, `<name>_puff<i>`, `<name>_shard<i>`, the hearts tagged `mk_heart`), each easing out to its own reach while it pops in and spins in the picture's plane, then shrinking away; the directions are flattened toward the picture (x-z), so pieces fly across it more than at the lens. Drivers on the frame (the maths: `mkmmd/core/burst.py`), so a [freeze](#freezes) holds them mid-air. Options `count` (24), `seed` (1), `mix` (`{heart = 0.45, star = 0.35, puff = 0.2}`; `{shard = 1}` for broken glass), colours `heart` ("love"), `star` ("gold"), `puff` ("text"), `shard` ("text"), `glow` (0.3); root properties for `[[key]]`: `start` (the clip second it goes off; 0), `reach` (how far the farthest piece flies; 1.2 m), `life` (1.4 s), `spin` (360 degrees over a life), `size` (the largest piece; 0.16 m), `amount` (1; 0 hides every piece). A `[[key]]` on `start` with a `lane` sets it off again on every kick |
| `squeaky_hammer` | a squeaky toy hammer, 320 mm long: a pleated bellows head (180 mm, its axis along X) with a cap at each end, on a handle standing along +Z from its end at the origin; colour roles `head` ("love"), `caps` ("gold"), `handle` ("gold"); `use.look` `head` and `grip`. Root property `squash` (0..1) for `[[key]]`: the head 40 % shorter and 18 % rounder about its middle (key it up on each hit and back down). Put it in a fist with `attach`, `offset` and `attach_rot`, and play `bonk` with that hand (see [Moves](#moves)) |
| `bed_single` | a single bed with a Memphis quilt and pillows, head end at +Y (push it against a wall); options `pattern` ("memphis", "zigzag", "triangles", "plain"), `seed` (7), `quilt_colors`, `quilt_ground`, `pillowcase`, `frame` ("wood" or "tube"), `frame_color`, `headboard` ("slats" or "padded"), `headboard_color` |
| `rug_80s` | a Memphis wool rug, 1.8 x 1.2 m (`flat`: things stand on it); options `seed` (5), `rug_ground`, `rug_colors` |
| `poster_80s` | a printed paper poster on a wall (origin `wall_center`, five designs); options `style` ("sunset_grid", "memphis", "trio", "car", "sunburst"), `width`, `height` (0.50 x 0.70), `margin`, `mount` ("pins", "tape", "none"), `folds` ("cross", "thirds", "none"), `torn` ("TR", "TL,BR"), `variant` (0..2), `seed`, `accent`; `use.surface` `print` |
| `alarm_clock` | a bedside digital alarm clock with a seven-segment display; options `time` ("02:47"), `led` (slot of the lit segments, "love"), `light` (0.2 W); root properties `glow`, `colon` (key it to blink), `alarm`; `use.surface` `display` |
| `bedroom_nightstand` | a small Memphis bedside cabinet, 0.40 x 0.35 x 0.55 m; `use.rest` `top` and `shelf` |
| `wall_shelf` | a floating wall shelf with books, cassettes and an ornament (origin `wall_center`); option `seed`; `use.rest` `top` and `free` (the stretch for a small prop) |
| `cafe_chair`, `cafe_table` | the café's bentwood cane chair (seat top 0.45 m) and terrazzo bistro table (top 0.74 m) |
| `cafe_page`, `cafe_pen` | a 160 x 220 mm letter page on the table (`use.surface` `page`, a `PaperTone` mix node for ink, an empty `<name>_pen_rest`) and a fountain pen (`use.grip` `barrel`, origin at the nib) |
| `cafe_mug`, `cafe_saucer` | a speckled mug with tea, tag and steam (root property `steam` 0..1, driven by a `cafe_room`'s) and its saucer (put the mug at the saucer's `at` + (0, 0, 0.0058)); `use.grip` `handle` |
| `cafe_ipod`, `cafe_earbuds`, `cafe_vase` | an iPod classic lying screen up, white wired earbuds whose cord runs into its socket (built in the world-aligned café frame: root at the origin), a bud vase with a eucalyptus sprig |
| `cafe_fairy_lights`, `cafe_poster` | two runs of string lights (root property `bulb_gain`) and a paper print taped to the back wall (option `image`: an absolute path of the printed picture) |
| `cafe_pothos`, `cafe_monstera`, `cafe_haworthia` | plants (static) |

**`hearts`**: `card = "library:hearts"`, `at` the middle of the column the hearts climb (the field faces -Y like a character). Build options (the
`slots` of the `[[prop]]`; colours are palette slots): `heart` ("love"), `glow` (0.3; 0 = lit by the scene), `count` (16), `seed` (1),
`size` (`[smallest, largest]` width in m), `puff` (0.42: the pillow's thickness as a share of the width) and any live parameter below as its
starting value. The live parameters are custom properties of the prop root that drivers read every frame (no handler, no Python at render time);
`[[key]]` keys them (`target = "hearts", prop = "amount", keys = [[t, v], ...]` swells the field in or out; `spread`, `sway` and `spin` are safe to
key too; `rise` and `height` are a clock rate, so changing them mid-clip moves the hearts):

| Parameter | Default | Meaning |
|---|---|---|
| `rise` | 0.25 m/s | climb speed |
| `height` | 1.6 m | the column the hearts climb, centred on the root |
| `spread` | 0.62 m | farthest \|x\| of a heart's centre |
| `fan` | 0.5 | 0: parallel climb; 1: the plumes open out from the clear column's edge to `spread` at the top |
| `clear` | 0 m | half-width of the column kept empty (a figure stands there) |
| `clear_top` | 10 m | height above the root where the clear column ends; it closes over 0.3 m, so the hearts arch over a head |
| `depth` | 0.25 m | half the depth (y) the hearts are spread over; keep it small: in perspective a heart behind the column's plane looks nearer the axis |
| `sway` | 0.06 m | amplitude of the slow sideways drift |
| `spin`, `tilt` | 25, 10 degrees | amplitude of the yaw swing and of the roll (the nod is half of it) |
| `pop`, `fade` | 0.4, 0.6 s | the birth ease (it overshoots 10 %) and the shrink to nothing at the top |
| `amount` | 1 | overall scale of the field (0 hides it) |
| `size_min`, `size_max` | 0.06, 0.16 m | the smallest and the largest heart width |

Every heart is an object `<name>_heart<i>` (one shared mesh: a classic two-lobed heart, rounded and inflated, its face along -Y; [`form`](#form) 0.00) with the
custom property `mk_heart`, so a silhouette shot makes them its accents: `keep = [..., "hearts_heart*"]`, `accent = ["hearts_heart*"]`. The hearts
rise in two plumes, one each side of the axis, born at even intervals and climbing at one speed, so neighbours keep their distance and a flat
silhouette never merges two of them. A heart is born at the bottom with the pop, climbs, sways and turns, shrinks away at the top and is reborn at scale 0,
so nothing jumps but the pop; no part of it enters the column \|x\| < `clear` below `clear_top`, through any sway, turn or pop, so in a front view no heart crosses
a figure standing in it. The position, turn and scale of every heart are drivers; each is a Blender simple expression of at most 255 characters (checked at
build; see [AGENTS.md](AGENTS.md#pitfalls)). The maths and the parameter table are `mkmmd/core/hearts.py`.

### Vehicles

`[[vehicle]]` drives a prop along a set's path: its position and heading on every frame, body roll and pitch from the
curvature and the speed, wheels turning with the distance travelled, a steering wheel turning with the curvature. The stage
runs after props (it needs the prop and the set) and before cast and pose, so a character seated in the prop is placed
against the vehicle as it stands at the first frame and then rides it.

| Key | Meaning |
|---|---|
| `prop` | the driving `[[prop]]` (required). Its root is keyed on every frame, so its own `at`, `yaw` and `rot` are replaced; give it no `parent` |
| `path` | `"<set>:<path>"` (required): a `[[set]]` name and a path of that set's card (the `highway` and `test_road` sets have `road`) |
| `lane` | a lane name of the path (`fwd1`, `opp1` ... on a highway, 1 = next to the median; `R1`, `L1` on `test_road`) or metres to the left of the centreline, negative to the right (default 0). A lane whose card `dir` is -1 (an oncoming lane) is driven against the path: the vehicle faces the other way, its roll and steering mirrored |
| `dir` | with a numeric `lane`: 1 drives with the path, -1 against it (default 1) |
| `speed` | m/s, the vehicle's own speed (positive whichever way it drives): a number, or `[[t, v], ...]` on clip seconds, linear between the keys and held before the first and after the last (default 20) |
| `at` | arc length in metres at clip time 0 (default 0) |
| `meet` | `{vehicle, t, ahead = 0}` instead of `at`: alongside that vehicle (an earlier `[[vehicle]]`) at clip time `t`, plus `ahead` metres further along the path. Traffic timed to the cut: an oncoming car passing in a close-up, a truck overtaken in a side shot |
| `leave` | `true`: the vehicle may run off the path's ends (traffic coming and going). There it waits at the end, hidden from the render (`hide_render` keys) with everything under its root, headlight beams included. Without it a vehicle that leaves the path is a build error |
| `height` | metres above the path (default 0) |
| `roll` | degrees of body roll per g of lateral acceleration (speed² × curvature), leaning out of the turn (default 1.2) |
| `pitch` | degrees of body pitch per g of longitudinal acceleration: the nose rises under acceleration and dives under braking (default 0.8) |
| `wheelbase` | metres, for the steering angle (default 2.6) |
| `steer_ratio` | angle of the steering wheel per angle of the road wheels; replaces the card's `steering.ratio` (default the card's, else 14) |

The prop's card says what spins and steers, with two keys of its own:

| Key | Meaning |
|---|---|
| `wheels` | `[{object, radius, axis}]`: each object turns by distance / `radius` about its local `axis`, which points to the car's left (default `[1, 0, 0]`), so the tyre's top moves forward |
| `steering` | `{object, axis, ratio}`: the object turns about its local `axis` (default `[0, 0, 1]`) by `atan(wheelbase × curvature) × ratio` (`ratio` default 14) |

```toml
[[vehicle]]
prop = "car"
path = "road:road"
lane = "fwd1"
speed = [[0, 24], [6, 24], [9, 14]]      # m/s on clip seconds: slows down from 6 s
at = 120.0
```

The prop's -Y axis faces the direction of travel (the cards' `front = "-Y"`; the stage assumes it). Heading and curvature come
from the smooth curve through the path's control points; the road is level across (no banking). The vehicle also drives
through the pre-roll, so `at` must leave `speed` × the pre-roll (3 s unless `[scene] start` says otherwise) of path behind it:
the stage stops with `vehicle 'car' leaves the path` when the arc length falls off either end. A hand that grips the steering
wheel follows it when its goal rides the wheel object (`ride`, see [Posing](#posing)); `[sim.<cast>] wind = {carrier =
"<prop>"}` takes the air's motion from the vehicle and `[[shot]] mount` puts a camera on it.

The report has, per vehicle, `path`, `lane_offset`, `from_m` and `to_m` (the arc length over the whole frame range),
`max_lat_g` and `steer_deg_max` (with a `steering` object), and the log repeats it.

### Cast

`[[cast]]` imports every model with mmd_tools (scale 0.08: one MMD unit is 8 cm), names it and puts it in the world. The tables
of the later stages are read by the member's name: `[pose.<name>]`, `[perform.<name>]`, `[sim.<name>]`, `[[motion.<name>]]`,
the targets `cast:<name>` and `{cast = "<name>", point}`, and a check's `cast = "<name>"`.

| Key | Meaning |
|---|---|
| `name` | the member's name (required) |
| `asset` | registry slug of a model ([Asset registry](#asset-registry)): its PMX path and its `rig.json` |
| `pmx` | path of a .pmx or .pmd (relative to the project, `~` expands); replaces the registry's path when `asset` is given too. One of `asset` and `pmx` is required |
| `rig` | path of the model's `rig.json`; default the registry entry's. Without one a member can be posed, moved and performed (expressions by the model's own morph names), but `sim` and `wear` need it |
| `armature` | name given to the armature object (default `<Name>_arm`, the name with its first letter upper-cased); checks and `mk q` find the armature by it |
| `at` | `[x, y, z]` metres of the model root (default the origin; in the frame of `parent` when there is one) |
| `yaw` | degrees about Z; 0 faces -Y, positive turns toward +X (default 0) |
| `parent` | name of an object the model rides on; `at` and `yaw` are then in its frame |
| `physics` | `"mk"` (default): import without Bullet, so the chains move only where a `[sim.<name>]` table solves them. `"none"`: the same import, and the member is left out of `[sim]` even when it has a table (no secondary motion; the sim report says `skipped`). `"bullet"`: keep the author's rigid bodies and joints (Blender's rigid-body simulation then moves them; give the member no `[sim.<name>]`). Any other word is an error |
| `materials` | `{<PMX material> = {color \| tint \| hide}}`: changes the model's own materials in the scene (the file stays as it is). `color` (a palette slot or `#hex`) makes the material that flat colour: its texture off, its toon and sphere kept, the diffuse the colour and the ambient 0.4 of it (MMD lights it as ambient + 0.6 x diffuse); `tint` multiplies its diffuse and ambient colours (the texture's detail stays); `hide = true` draws it no more (alpha 0). A name the model lacks, an unknown key or both `color` and `tint` is an error that lists what there is; the cast report says per member what was done |

```toml
[[cast]]
name = "rin"
asset = "my_model"
at = [0.0, -0.4, 0.0]
yaw = 90                       # faces +X
```

The stage names the root `<Name>` (the name with its first letter upper-cased), the armature `<Name>_arm` or `armature`, the
meshes `<Name>_mesh<i>`, links everything into the collection `Cast` and hides rigid-body, joint and temporary objects from
render. Images whose files are missing are looked up by name (any case) under the model's folder and its subfolders. The report
has, per member, `armature`, `meshes`, `physics`, `rig` (whether a `rig.json` was found), `textures_relinked`,
`textures_missing` (image names to fix before rendering) and `bound_morphs`; the log notes when the `rig.json` was made from
another file than the model.

Many models build their mouth shapes and some faces from several morphs: a vowel is a group morph (a mouth shape and a tongue
bone), a blush a material morph. When the model has group, bone, material or UV morphs (`bound_morphs` counts them; 0 leaves the
model as imported) the stage binds mmd_tools' morph sliders: every morph becomes a shape key of a hidden `.placeholder` under the
root, whose drivers sum each morph into the meshes' shape keys, the bone morphs' constraints and the material morphs' nodes. The
perform stage's expressions and lip sync, a motion's facial keys and `mk q`'s `morph()` all use the placeholder then, so a group
morph is keyed like any other and two morphs sharing a shape add up. Every mk Blender job opens the scene before it enables
mmd_tools, and drivers that read `mmd_root` fail while the file loads; the job recompiles the drivers Blender marked invalid as
soon as mmd_tools is on (`mkmmd.blender.runtime.revalidate_drivers`).

The per-member tables of the later stages are checked before their stage runs: `[pose.<name>]`, `[perform.<name>]` and `[sim.<name>]`
must be named after a cast member, and a key the stage does not read is an error that lists the cast or the known keys
(`[perform.rin]: unknown key 'blinks' (known: ...)`). The check goes into the sub-tables the stages read (`hands.L`, `head`, `hips`,
`feet`, `gaze[]`, `blink`, `bob`, `sing`, `strum` ...), not into targets, nor into `params` and `wind` of `[sim]` (the solver refuses
its own). A table with no keys does nothing, and the build log carries a WARNING. A `sit` pose places its member itself and replaces
`at` and `yaw` (see [Posing](#posing)).

### Posing

`[pose.<cast>]` is a member's base pose: where it sits or stands, its spine and head, feet, arms and hands, and what it wears or
holds. It eases in from the rest pose over `[scene] settle_frames` (24) at the start of the pre-roll (bone keys at `start` and at
`start + settle_frames`, Bezier; the arm IKs' influence goes from 0 to 1) and then holds: the base pose that `[[motion]]` plays
under, `[perform]` moves and `[sim]` reacts to. The stage reads the scene at frame `start`, so props that
[vehicles](#vehicles) moved are read where they stand there. A member without a `[pose.<cast>]` table, or with an empty one (a WARNING
in the log), is not posed, but a prop that `wear` puts on it is still worn; `[perform]` and `[sim]` still work on it. The stage needs the cast and the props its keys name; grips need the CLI's
Python with scipy (the solver runs there).

Rotations are composed in the armature's axes (the model faces -Y: x its left, y behind it, z up), so characters riding a
vehicle keep correct keys: a bone's posed rotation relative to rest is `D_bone = D_parent . q`, the bone-local key is
`R^-1 q R` with `R` the bone's armature-space rest rotation, and a chain such as `upper_body` → `upper_body2` → `neck` → `head` is
keyed from the wanted deltas with `q_child = D_parent^-1 D_want` (`mkmmd/blender/keys.py`). Angles are degrees: positive `lean`
bends forward, positive `turn`, `hips.yaw` and `head.yaw` turn toward the character's left, positive `head.pitch` looks down
and positive `head.roll` tips the head toward its left shoulder.

| Key | Meaning |
|---|---|
| `sit` | `"prop:seat"`: a `use.sit` point of a prop (`"prop"` alone when it has one), or a table `{hip, facing, floor_z, pelvis_deg, back_deg}` in world coordinates. The root goes to the seat's floor point (`floor_z`; in the prop's frame for a card) facing `facing` (default `[0, -1, 0]`, flattened to the horizontal) and is parented to the prop, so the character rides it and `[[cast]] at` and `yaw` are replaced; the midpoint of the hip joints goes to `hip`. `pelvis_deg` (default 6) tips the pelvis back, which brings the thighs forward; `back_deg` (default 0) is the backrest's recline, the upper body leans back by it and `lean` adds to it. A card's `seat_z` and `back_tilt_deg` are not read |
| `sit_offset` | `[x, y, z]` metres: slides the hip point on the seat, in the seat prop's frame (the world's for a `sit` table): sit further forward |
| `feet` | ankle targets for the leg IK (a model without `leg_ik` bones ignores them). `"seat"` (default when seated): the seat prop's `use.feet` point of the seat's own name, else its only one (`L` and `R` as `[x, y, z]` in the prop's frame); a seat without one, and `"floor"`, put the feet in front of the knees (0.55 of the leg's length ahead of the hip, 0.1 m either side). Standing (no `sit`), `"floor"` keeps the feet where the model stands, on the floor under the hips (what no `feet` does too), and `"seat"` is an error that says so. `"prop:feet"`: another prop's `use.feet` point. A table `{L, R}` gives both feet as world `[x, y]` or any [target](#targets): `{cast = "rin", point = [x, y, 0]}` writes a standing pose in the character's own frame, so it moves with the character. Only x and y count: the ankle keeps the model's own height above the floor |
| `lift` | `{L = m, R = m}`: each ankle's goal that far above its rest height, eased in over the settle like the feet (with or without `feet`): a foot kicked up, or both off the ground with `hips.shift` raised as far (a jump caught at its top). Any other key raises |
| `hips` | `{shift = [x, y, z], roll, yaw}`: moves and turns the pelvis of a standing or seated body, eased in over the settle and on top of what a seat asks for. `shift` is in metres in the character's own frame (x its left, y behind, z up): a drop bends the knees and the leg IK keeps the feet on `feet`. `roll` degrees drops the left hip (+). `yaw` degrees turns the pelvis toward the left about the vertical. Keyed on `center` (shift) and `lower_body` (roll, yaw) |
| `toes` | `{L = deg, R = deg}`: foot yaw about the vertical through each ankle (the leg IK bone), + toward the left: toes in are negative on the left foot and positive on the right. Any other key raises |
| `lean`, `turn` | degrees: upper-body forward lean and turn toward the left, on top of the seat's back angle (default 0) |
| `lean_share`, `turn_share` | the share of `lean` and of `turn` that `upper_body` takes; `upper_body2` takes the rest, and a model without one takes all (default 0.6 each) |
| `head` | `{pitch, yaw, roll, neck = 0.4}`: base head rotation, degrees, absolute: relative to the armature, so a leaning chest does not tip the head (default 0). The neck carries the share `neck` of the way from the chest's rotation to the head's. [Perform](#perform)'s gaze adds on top |
| `hands` | `[pose.<cast>.hands.L]` and `.R`: arm IK to points, edges, moving keys and grips, below |
| `fingers` | `{L = preset, R = preset}`: finger curls (below) for a hand without arm IK. It is applied after the arms and replaces the finger keys of a `hands` table of the same side, a grip's included; a hand table has its own `fingers` |
| `drape` | `[[pose.<cast>.drape]]`: bone chains pointed along chosen directions, below |

Finger curls are a preset: `flat`, `relaxed`, `curled`, `fist`, or `point` (the index stays straight, the others as `fist`, the
thumb tucked); or a table `{index = [8, 10, 0], middle = [10, 12], thumb = [0, 8]}` of degrees toward the palm for each finger's
first three joints (`thumb0`, `thumb1`, `thumb2` for the thumb; a number alone is the first joint; a finger or joint left out
stays straight; `thumb = "tuck"` tucks it). The presets bend (fingers / thumb, degrees per joint) `relaxed` 14, 22, 14 / 8, 10,
8; `curled` 35, 50, 35 / 14, 18, 14; `fist` 80, 95, 65 / 25, 35, 40, and `point` its three curled fingers as `fist`
(`mkmmd/core/fingers.py`). The rotations turn about each finger's own flexion axis. A tucked thumb is solved per hand from its
rest bones (`tuck_thumb`): it folds across the curled middle finger, its tip resting on that finger's middle and end segments
(22 % of the finger's length off its bones) on the middle finger's side of the index, never standing up beside it, the rest of
it clear of both fingers, with the least turning (its base joint costing most).

#### Hands

`[pose.<cast>.hands.L]` and `.R` put an arm on a target. The goal empty `<cast>_hand.<L|R>` (collection `Rig`) is the wrist bone's
tail and orientation: a position-only IK on the forearm (`mk_arm_ik`, chain up to the upper arm, its target a child of the goal
at the wrist's head) bends the elbow toward the pole empty `<cast>_elbow.<L|R>`, the wrist copies the goal's rotation
(`mk_hand_rot`) and the forearm twist bone, when the rig has one, rolls all the way with the hand (`mk_forearm_twist`: the rig
spreads it down the forearm, so a sleeve turns with the hand instead of the hand turning inside the cuff); the wrist's head is
held on the forearm's end (`mk_wrist_on_arm`: rigs put it a fraction of a millimetre off the twist bone's axis). The IK's pole
angle is solved per arm from its rest pose, with the forearm pre-folded 20 degrees toward the front (the solver starts from that
fold), so the elbow points at the pole on either side of any rig. These empties are the names checks and `[[key]]` address. A hand needs one of
`at`, `rest`, `grip` or `keys`. The goals of a seated character ride with its seat prop; a standing character's stay in the world
unless `ride` says otherwise. The wrist goal is keyed at the end of the settle (and at the keys' times), the influence eases
in over the settle.

| Key | Meaning |
|---|---|
| `at` | where the wrist joint goes: a [target](#targets). The hand points along `dir` with the palm toward `palm` |
| `dir` | `[x, y, z]` world direction the hand points along, wrist to fingertips (default straight ahead of the character) |
| `palm` | `[x, y, z]` world direction the palm faces: the hand rolls about `dir` to match (default the least rotation from rest; `[0, 0, -1]`, palm down, for `rest`) |
| `pole` | a target the elbow points at (on either side of any rig: the IK's pole angle is solved per arm from its rest pose). Default, from the shoulder: 0.45 m outward, 0.25 m behind and 0.30 m below it; for a hand on a guitar's `strum` point, 0.45 m outward, 0.12 m behind and 0.08 m below (a strumming forearm drapes over the body's edge); on its `neck`, 0.15 m outward, 0.08 m behind and 0.40 m below |
| `rest` | `"prop:use"`: a `use.rest` point of a prop the hand lies on, an edge (`a`, `b`) or a plane (`center`); the palm faces `palm`, the heading is `dir` (default forward and slightly outward) |
| `along` | `rest` on an edge: 0..1 along it (default 0.5) |
| `offset` | `rest` on a plane: `[x, y, z]` metres from its centre, in the prop's frame (default 0) |
| `lift` | `rest`: metres above the surface along its normal (default 0.03; 0 for a `grip = "rest"`) |
| `keys` | `[{t, at, dir, palm, fingers}]`: moving targets, `t` in clip seconds, Bezier between the keys; `dir` and `palm` default to the hand's own; a key's `fingers` (as the hand's `fingers`) are reached at its time and eased between the keys that have them, from the hand's own at the end of the settle. With `at` that pose holds until the first key; without it the first key's target is the hand's from the start. A neck grip's keys are `{t, fret, chord, move}` instead |
| `ride` | the goal follows something. `"<object>"` (a steering wheel, `"car_wheel"`): the goal is parented to it. `"cast:<name>.<bone>"` (this character's own name; the chest is `cast:rin.upper_body2`): goal and pole are bone-parented to that bone, their keys computed from the world goal with the bone as it stands in the settled pose, so clasped hands follow every `lean`, `turn`, `tilt` and sway and never part; a bone of an arm is refused. Targets and `keys` are read as world points of the first frame (object) or of the settled pose (bone). A hand on a guitar's `neck` or `strum` point rides the prop's root by default |
| `ik` | `[[t, share], ...]`: clip seconds and shares 0..1 of the arm IK (with the hand's copied rotation and the forearm's pre-bend) after the settle, eased between the keys and held outside them (default all of it); where it is under 1 the arm follows its [motion](#motion) strips. The moves stage writes it for a hand a `free` move lets go ([Moves](#moves)) |
| `fingers` | a finger preset (`flat`, `relaxed`, `curled`, `fist`, `point`) or a table `{index = [a, b, c], middle, ring, little, thumb, spread}` of curls in degrees per joint (a finger left out stays straight) and `spread`, degrees the fingers fan apart in the plane of the palm (the index by `spread`, the ring 0.6 and the little finger 1.2 times as far, the thumb 0.8 times away from the index; negative closes them; or `{finger: deg}`) for this hand (replaced by a grip's solved rotations) |
| `grip` | the hand holds something: `"prop:use"` (a `use.grip` point of a prop; `"prop"` alone when it has one) or `"rest"` (with `rest`). A solver finds the finger rotations and the hand's frame on the prop ([Grips](#grips)); the wrist goal and the fingers are keyed from it, replacing `at`, `rest` and `fingers`. Not combinable with `keys`, except the `keys` of a `neck` grip |

The keys of a grip (see [Grips](#grips) for what each style needs and does):

| Key | Meaning |
|---|---|
| `clock` | ring grips: hours on a clock face as the character sees the wheel, 12 top and 3 its right (default 10 for `L`, 2 for `R`), read at the first frame |
| `approach` | ring: degrees round the tube's section where the palm lies, 0 the outer side, 90 the side facing the character. Default: chosen for the arm (below) |
| `wrap` | ring: +1 or -1, the way the fingers go round the tube (default -1: round the outside of the rim, then its front) |
| `edge` | pinch: metres the pads sit inside the held part's edge (default 0.004) |
| `face` | `grip = "rest"`: `"palm"` (default) or `"back"` lies on the surface |
| `seeds` | parallel starts of the solver, the best wins (solver defaults: pen 12, wheel 8, pinch 8, rest 6, neck 6) |
| `skin_radius` | metres: skin vertices within this of the wrist head are solved against (default 0.16) |
| `posture` | pen: the writing posture table (needs at least `nib` without a `track`); with a `track` your keys win over what the stage completes |
| `track` | pen: the nib's path, `"nib"` for `tracks/nib.json` or a `.json` path: the nib follows it on every frame |
| `channel` | the track's channel of positions (default `"target"`) |
| `wobble` | pen with a `track`: degrees, or `{deg, tau = 18, seed = 3}` (frames), a slow random tilt of the pen about the world X and Y (default none) |
| `fret`, `chord` | neck: the position fret under the index finger (1 or more) and the chord shape (a name or `{finger = [string, frets]}`) |
| `press` | neck: the share of the gap between two wires, behind the wire it plays, where a pad presses (default 0.3) |
| `move` | neck: seconds a keyed change takes to land (default 0.12) |
| `thumb`, `tip` | strum: `"neck"` (default) or `"bridge"`, the side the thumb is on; metres the pick's tip sticks out of the pads (default the card's `pick.tip`, 0.008) |

```toml
[pose.rin]
sit = "car:driver"
sit_offset = [0.0, 0.03, 0.0]
lean = 12
head = { pitch = -4, neck = 0.4 }

[pose.rin.hands.L]
grip = "car:wheel"
clock = 10
ride = "car_wheel"

[pose.rin.hands.R]
rest = "car:sill_R"            # the right hand lies on the door sill
fingers = "relaxed"
```

```toml
[pose.rin]                     # standing, written in the character's own frame
feet = { L = { cast = "rin", point = [0.12, 0.0, 0.0] }, R = { cast = "rin", point = [-0.10, 0.06, 0.0] } }
hips = { shift = [0.0, 0.0, -0.03], roll = 3, yaw = -4 }
toes = { L = -8, R = 10 }
turn = -6

[pose.rin.hands.L]
at = { cast = "rin", point = [0.25, -0.15, 1.05] }
dir = [0, -1, 0]               # a world direction: the character faces -Y

[pose.rin.hands.R]
keys = [{ t = 0.0, at = [0.30, -0.60, 1.0] }, { t = 2.0, at = [0.30, -0.35, 1.30], palm = [0, 0, -1] }]
```

#### Drape

`[[pose.<cast>.drape]]` points a bone chain along directions of your choosing, static, as cloth over a seat (a skirt, a scarf):

| Key | Meaning |
|---|---|
| `chain` | bone names root to tip (Blender, PMX or semantic names) |
| `dirs` | `[[x, y, z], ...]`, one per bone: the direction the bone points along in the character's axes (x its left, y behind it, z up), by the least rotation from its rest direction whatever the bones above do |
| `scale` | optional length scale per bone, `[1, 0.75, ...]`: bunched cloth, the bone is shortened along its length |

The keys ease in over the settle like the rest of the pose and the bones stay keyed, not simulated, so the model's collision
bodies on them move and hair collides with them. A draped family listed in `[sim.<cast>] families` loses its drape: the sim
stage keys the same bones afterwards.

```toml
[[pose.rin.drape]]
chain = ["skirt_f1", "skirt_f2", "skirt_f3"]
dirs = [[0.0, -0.7, -0.7], [0.0, -0.9, -0.4], [0.0, -1.0, -0.1]]
scale = [1.0, 0.85, 0.7]
```

#### Props on the body

These keys sit on a `[[prop]]` ([Props](#props)) and are applied by the pose stage:

| Key | Meaning |
|---|---|
| `attach` | `"<cast>:<bone>"` (semantic or Blender bone name): the prop's root is bone-parented, replacing its `at` and `yaw` |
| `offset`, `attach_rot` | with `attach`: `[x, y, z]` metres and `[x, y, z]` degrees (XYZ Euler) in the bone's head frame (default 0) |
| `cable` | with `attach`: `true` or `{object, anchor, radius, trail, out, sway, reach, tail_len, follow, sim}` as a worn prop's: the card's cord is hung from its `jack` anchor to the floor under the cast member at the settled pose and swung by the sim (a mic in a hand) |
| `anchor_to` | `"<cast>"`: the card's `use.anchor` points that name a semantic `bone` and an `object` are bone-parented to that member, keeping their world placement at the settled pose (earbuds in the ears, a cord on the chest) |
| `wear` | `"<cast>"` or a table `{cast, use, at, pivot, scale, neck_deg, yaw_deg, roll_deg, strap, cable}`: a worn prop on the bone the card names (a guitar on the chest) with its strap and cord, see [Playing a worn guitar](#playing-a-worn-guitar) |

Worn props are placed before the arms are solved, so hands can grip them; `attach` and `anchor_to` run after the arms, so a grip
on a prop attached to a bone reads it where the props stage left it. A worn prop does not need a `[pose.<cast>]` table for its wearer: the pose stage puts it on even when the member has none.

#### What the stage reports

Per member the report has `hip_offset` (the pelvis move onto the seat, metres) or `hips`, `toes`, `drape_bones` (bones keyed),
`ride` (side → bone), `wear`, and a `grip` digest per hand: `style`, `contacts_mm` (gap per finger or region), `penetration_mm`,
`finger_clash_mm`, `seconds` and the style's own numbers (`clock` and `axis_flipped` of a ring, `fret`, `chord` and `states` of a
neck, `thumb` and `tip_mm` of a strum, `point` of a rest, `writing` and `track` of a pen path), with `warnings` when it misses.
`attached` lists the props placed by `attach` and `anchor_to`, `cables` the worn props' cords, and `ik_error_mm` the miss of
every arm IK in millimetres, keyed `<cast>.<L|R>`.

The log has a line per grip (and `mkmmd.solvers.grip: cached ...` or `solved in ...s` for each solve) and `WARNING` lines when
something is off: a grip past its gates (a contact gap over 3 mm, penetration or finger clash over 1 mm: the prop does not fit
that hand) and a wrist that ends more than 5 mm short of its goal (out of reach: lean the character, move the seat or bring the
prop closer; an arm is about 0.38 m long for a 1.7 m model). The miss is measured at
the end of the settle, at every landing of a fretting hand's keyed change and on every sixth frame of a pen track; the moving
targets of plain `keys` are not measured. Pin a grip with a `contact` check between a fingertip bone's tail and the prop, and a
reach with one between `bone("wrist.R").tail` and `obj("<cast>_hand.R").loc` (the IK reached its goal).

Interplay with the stages around it: `[perform]` keys `upper_body`, `upper_body2`, `neck` and `head` on every frame, replacing the
pose's keys with per-frame keys that contain the base pose, so a hand that does not `ride` the chest holds its world goal while
the chest leans, turns and tilts. A `[[motion]]` plays under the pose: bones the pose or perform keys are not driven by it.
`sit` re-places the root, so a target written as `{cast = "rin", point}` is read from where the seat put it. Props that
[vehicles](#vehicles) drive carry a seated character and its hand goals with them.

### Moves

`[[move.<cast>]]` places named moves of a small library on the clock (`mkmmd/core/moves.py`); the pose stage compiles them before
it reads its tables into hand keys (`[pose.<cast>.hands.L/R] keys`, the goals riding the chest, `ride = "cast:<cast>.upper_body2"`,
so leans, tilts, sways and the bounce carry the hands: a mic at the mouth stays there), lean, tilt, turn, head tilt and head
turn keys, twitches and expressions (`[perform.<cast>]`), so a move is keys like any other. A hand a move plays may have no
`at`, `keys`, `grip` or `rest` of its own in `[pose.<cast>.hands]`, and `lean`, `tilt`, `turn`, `head_tilt` and `head_turn`
cannot be keyed both in `[perform]` and by a move (each is a build error); other keys of the same tables stay the project's (a
`ride` of the project's own wins).

| Key | Meaning |
|---|---|
| `name` | the move (below, required) |
| `t` | clip seconds the move starts (required) |
| `dur` | seconds it lasts (default 1) |
| `hand` | `"R"` (default) or `"L"`: the hand of a one-hand move |
| `morph`, `value` | a `face` move's morph (semantic or the model's own name) and its value (default 1); on any other move, a face held over it on top of the move's own |
| `hits` | a beat move's own clip seconds to play on instead of the beats inside it (the snares, say); each must fall inside the move |

`{name = "rest", hand, place}` says where a hand waits when no move plays it (and gives that hand its arm IK, so a walk's `arms`
swing it even when it plays nothing): `"rest"` (default: hanging by the hip; over a wide
skirt the arm swings out until the hand clears it), `"dainty"` (resting lightly on the front of the skirt or the thighs, elbows
in, the two hands together when both rest there: a girl's polite stand) or `"mic"` (a fist under the mouth, holding an
attached mic that leans back up to the lips, so the forearm rises to it and the elbow hangs by the side). A hand
travels at a human pace: a trip takes 0.1 s plus its wrist's way at 1.6 m/s (eased: it peaks near twice that), or as long
as turning its palm or its fingers' way at 600 degrees a second takes, if that is longer, and it reaches a move's first
place on the move's time, so it sets off before it. A reaction is the other way round: `flung` leaves where the hand is on
its own time (a hand does not see a blast coming) and is thrown there in 0.06 s plus its wrist's way at 4.5 m/s. A hand
holds its last place to the end of the move, then goes home to its rest place if it can stay there 0.15 s before its next
move, else straight on to that move, leaving its place early enough to get there on time; a next move too close for that
(the hand would have to jump) is a build error that says how long the trip takes. One hand cannot play two moves at once.
A hand that rests at the mic keeps the mic in its fist through every move it plays (the mic moves aim it); a two-hand move
leaves it at the mic and plays with the other hand only, except `hands_up` and `flung` (the mic goes up too), and
`heart_push`, which needs both hands free, is a build error for a member holding a mic.

| Move | What it does |
|---|---|
| `bounce_hand` | the hand out in front at the waist, dipping on each beat inside the move |
| `chest_pat` | the flat hand pats the chest on each beat; blush |
| `paws` | both hands, cat paws under the chin, dipping on the beats; `omega` mouth |
| `point` | the arm out, the index pointing at the camera |
| `heart_wink` | a finger heart by the cheek, its thumb side to the camera; `wink_r`, `mouth_smile` |
| `peace_eye` | a peace sign by the eye; `wink_l` |
| `sparkle` | both hands open beside the face; `smile_eyes` |
| `drip_check` | the hand by the chest turns over halfway; `jito` eyes |
| `hands_up` | both arms up |
| `flung` | both arms thrown up by a blast, the hands leaving on its time; they come home when it ends, so end it as the body lands for the hands to come down with the landing |
| `mic_lens` | the (mic) hand held out at the camera, the mic upright and leaning to the lens |
| `mic_up` | the (mic) hand raised forward and up, the mic upright over it, the palm toward the head (a fist holds a mic across its palm, so a mic pointing forward from an upright forearm would twist the palm outward) |
| `mic_across` | the (mic) hand held out across the body to the other side, the mic upright (offered to a partner on that side) |
| `ears` | both hands over the ears, elbows wide; `hau` (">_<") face: too loud |
| `heart_push` | both hands make one heart in front of the chest: the fingertips meet over its two lobes, the thumbs fold down to its point; `cheerful` face. Needs both hands free |
| `bunny_paws` | both paws up over the head like bunny ears; `wink_r` |
| `bonk` | a toy hammer (held in the fist, see the library's `squeaky_hammer`) cocked up and back over the shoulder, then brought forward and down on each beat or hit, its head's face meeting what is in front at the height of the chest; it starts cocked (its wind-up), so start it a little before the first hit |
| `into_lens`, `lean_back` | the upper body leans 14 degrees forward; 11 back and 7 to the left |
| `stank` | the head flicks back; `hau` face |
| `blown` | blown back by a blast: the upper body leans 16 degrees back, the head snaps back; `surprised` face (the hands keep what they play) |
| `hair_flip` | a pose: the hand runs into the hair by the ear, then tosses it back and out (at 40 % of the move), the head flicking the other way (a twitch for the hair to fly on) and tilted from the hand; `calm` eyes, a small smile |
| `over_shoulder` | a pose: a look back over the shoulder of the move's `hand` side, the upper body turned 25 degrees and the head 45 more that way; `calm`, a small smile. Shoot it from behind her |
| `hand_hip` | a pose: the hand on the hip, the elbow out, the head tilted toward it and the body leaning a little away; `smile_eyes`, `mouth_smile` |
| `finger_gun` | a pose: the arm out ahead at the height of the eyes, the hand a finger gun (index out, the thumb up, the rest curled), leaning in 6 degrees, the head tilted toward it, winking the other eye; `mouth_smile`. Read from a three-quarter view: straight on, the hand stands in front of her face |
| `free` | lets the hand go for the span to whatever else drives the arm (a walk VMD's swing; below) |
| `face` | only its `morph` |

The four poses (`hair_flip`, `over_shoulder`, `hand_hip`, `finger_gun`) are written for the left hand and mirrored for the right:
their turns, tilts and the head's twitch go the other way, their lean does not. A move's body keys ease in over 0.12 s and back
over the last 0.12 s; two moves that move the body at once are a build error.

A beat move needs a beat inside it (the timeline's `beats`, `audio/timeline.json`) or its own `hits`. The hand places are made from the member's
rest pose (its arm joints, chest, mouth, eyes, the arm's two lengths and the hand's), so the library fits any body. Places by
the face and the chest are on the hand's own side, clear of a mic at the mouth and of its cord. Each place is then made
natural:

- **The wrist.** The elbow the arm IK will give (the default pole, `mkmmd.core.armreach`) fixes the forearm, and a hand that
  would bend more than 60 degrees from it is turned toward it as a whole (the IK's own lands within about 10 degrees of that).
  A place with something in the fist (a mic, a hammer) aims it instead: its hand continues the forearm as far as a fist round
  a handle pointing that way allows (the mic at the mouth: the forearm rises to a fist under the lips).
- **The skin.** The build measures the distance to every face of the member's meshes in the rest pose that belongs to neither
  arm: its body and clothes, but not the hair, ear, tail, skirt and ribbon chains its `[sim.<cast>]` table simulates (those
  make way for a hand, colliding with its fingers and palm, so long hair or a skirt at her sides does not push a hanging hand
  out from the body). The hand (a box the size of the model's hand and of its finger shape) comes in from outside and stops
  where it first comes within its margin: an arm held out swings in round the shoulder (so it stays in reach; a hanging hand
  ends against a wide skirt the sim does not move), a hand by the body comes in straight. `chest_pat` and `ears` end on the
  surface (3 mm), every other place stays where it was written unless that is closer than 12 mm to the body or inside it.
  Coming from outside, a hand never ends inside clothes or hair that does not move.

A `free` move lets its hand go for the span: the arm IK (with the hand's copied rotation and the forearm's pre-bend) eases off
over 0.25 s and back on, and in between the arm follows whatever else drives it, a [motion](#motion) strip of the arms (a
walk's swing, below); the hand's goal waits at its rest place meanwhile, so the hand comes back from there. The moves stage
writes the hand's `ik` keys ([Posing](#posing)), and a member with a hand let go has its own keys composed with its strips
(its action blends with Combine, so the pose stage's forearm key adds to the strip's elbow instead of replacing it). A hand
that rests at the mic is never let go.

Faces of the same morph closer than 0.12 s are held through. The pose report has `moves` per member (`moves`, `hand_keys` per
side, `free` spans per side, `expressions`, `twitches`, and `places`: per side and place, `moved_mm`, how far the approach
kept the hand out, and `wrist_deg`, the wrist's bend).

```toml
[[move.len]]
name = "rest"                  # Len's right hand holds the mic at his mouth between moves
hand = "R"
place = "mic"

[[move.len]]
name = "chest_pat"             # pats on the beats from 3.6 s to 5.0 s
t = 3.6
dur = 1.4
hand = "L"
```

### Motion

`[[motion.<cast>]]` plays VMD motions on NLA strips of the member's armature, retimed to the song's beats if asked and limited
to some bones (`[motion.<cast>]` with a single table works too). Each entry makes a strip in a track of its own, `motion0`,
`motion1` ... (the facial keys go to `morph0` ...), under the active action. The stage needs the member from `cast` and, for
`retime`, a timeline.

| Key | Meaning |
|---|---|
| `vmd` | a .vmd path (relative to the project) or the registry slug of a motion (required) |
| `start` | clip seconds at which source frame `from` plays (default 0) |
| `from`, `to` | source frame range, in the motion's own 30 fps frames (default the whole motion) |
| `scale` | time scale of the strip: 2 plays at half speed (default 1) |
| `retime` | `"beats"`: the motion's own beat (`core.vmd.tempo_phase`, a comb filter over 70-180 bpm of the bone motion) is scaled to the song's beat period and the first motion beat at or after `from` lands on the song beat nearest to where it would fall with `from` at `start`. Needs `timeline`; replaces `scale` |
| `timeline` | with `retime`: the timeline JSON whose `beats` are used (default `audio/timeline.json`; a missing file stops the build naming `motion.<cast>[i]`) |
| `bones` | which bones the motion drives: `"all"` (default), `"upper"` (spine, neck, head, eyes, shoulders, arms with their twist bones, wrists, fingers), `"lower"` (center, groove, waist, lower body, legs, knees, ankles, toes and the leg and toe IK bones), `"arms"` (shoulders, arms, elbows, wrists and their twist bones, not the fingers) or a list of these groups and semantic or Blender names. Hair, skirt and the root bone belong to no group |
| `morphs` | also play the VMD's facial keys (default true) |
| `blend_in`, `blend_out` | seconds the strip's influence ramps in and out (default 0) |
| `extend` | `"hold"` (default): the strip's first and last poses stay before and after it; `"forward"`: only its last pose stays, after it; `"none"`: it acts on its own frames only. A later entry's track is above an earlier one's, so where both play the later one wins |

A walk VMD gives a walking character natural arms where mk's own walk (`[perform.<cast>] walk`, its `arms` an IK swing) reads
stiff: let both hands go with `free` moves over each run of steps, and play the walk's `bones = ["arms", "upper_body",
"upper_body2"]` there with `extend = "none"`, its cycle scaled to two of the character's steps and its left heel strike on
her own left steps (a VMD walk in place keeps the legs, the feet and the travel mk's).

A dance that stops with the music and picks up again: play stretches of one choreography as entries whose source ranges follow
on (each `from` the `to` of the entry before), each with `extend = "forward"`. Between two of them she holds the pose the dance
stopped in, and the next starts from that very pose, so nothing jumps or slides; let a hand go with `free` over the stretches
(and over the stops, for a hand that stays in the frozen pose) and give the other a pose move on a stop.

```toml
[[motion.rin]]
vmd = "motions/walk.vmd"        # a path in the project, or a registry slug
start = 2.0
from = 0
to = 300
bones = "lower"
blend_in = 0.2
```

By default (`extend = "hold"`) the motion's first pose holds before `start` and its last after it ends. Bones that `[pose]` or
`[perform]` key win over the motion (the active action plays on top of the strips): `pose` keys the spine, neck and head
always, the pelvis with `sit` or `hips`, the arms with a `hands` table, the legs with `feet`; `perform` keys the spine, neck, head
and eyes on every frame. Give a member that has both a motion and those tables `bones = "lower"` or a list of the bones the
others leave alone. The report has, per entry, `vmd`, `source`, `strip` (first and last frame), `morphs` and, with `retime`,
`motion_bpm`, `song_bpm` and `scale`; `retime` raises when no beat can be found in the motion (a very short or motionless VMD)
or the timeline has fewer than two beats.

### Perform

`[perform.<cast>]` is a character's life on top of its pose: gaze, breathing, sway, nods, beat bob, startles, keyed lean / turn /
tilt, blinks, expressions, lip sync, twitches and strumming. Times are clip seconds, angles degrees, targets are
[Targets](#targets), with one exception: `"camera"` does not resolve here, the cameras are built by the shots stage after this
one. The stage keys `upper_body`, `upper_body2`, `neck`, `head` and both eyes on every frame of the build (LINEAR), composed in
the armature's frame, so characters riding vehicles stay correct. Those keys contain the pose's base (eased in over the settle)
and replace the pose stage's own on the same bones. A model without eye bones gets head motion only; one without `upper_body2`
takes the breathing on `upper_body`. The stage needs from earlier stages the pose's base and the pick hand's goal for `strum`,
the props and members the targets name, `rig.json` for morph names (`blink`, the vowels `a i u e o`, the semantic names of
`expressions`) and a timeline file for `bob`, `sing` and `strum`. `mk ref measure` recommends values for these keys from clips of
real people (`blink.per_min`, `lids`, the rate, `rise` and `hold` of gaze events, `head_share`, `eye_max`, `nod`, `bob`, `sway`,
`breath`, `sing.mouth`), each with a reason.

#### Gaze

Gaze blends unit directions from the eyes, so targets at any distance (a mirror 0.5 m away, a road 30 m ahead) mix evenly.

| Key | Meaning |
|---|---|
| `look` | the idle gaze target (default 3 m straight ahead of the eyes along the model's forward axis) |
| `gaze` | `[{t, at, hold = 1.0, back = 0.4, rise = 0.3, blink = true}]`: look-at events. The look reaches `at` (a target) over `rise` seconds from `t`, starts to return at `t + hold` and is back at the idle look by `t + hold + back`; overlapping events share the weight. A blink follows at `t + 0.03` unless `blink = false` |
| `glance` | `[{t, at, dur = 0.8, pitch = 2.75, rise = 0.12, fall = 0.25, blink = true}]`: eye-only glances. The eyes lead to `at`, the head lifts `pitch` degrees, the lids open (see `lids`) and a blink follows at `t + 0.02` unless `blink = false`; head and neck do not turn. A glance takes the eyes only where no gaze event has them |
| `head_share` | the share of a gaze turn that head and neck take; the eyes do the rest, up to `eye_max` (default 0.7) |
| `head_limits` | `{yaw = 75, up = 35, down = 45}`: the head and neck's range in degrees, always applied with these defaults; a table overrides entries. A target overhead or behind is met by the eyes, not by an impossible head turn |
| `neck_share` | the share of the head's turn that the neck carries (default 0.35) |
| `eye_max` | the eyes' range in degrees (default 24) |

The head turns `head_share` of the way from where the pose holds it to the target, the neck carrying `neck_share` of that, the
eyes the rest. So with a `[perform]` table the head is pulled toward `look` and the gaze events: aim it with those, and a pose
`head` rotation survives only in the remaining share. The eyes lead the head by 70 ms (50 ms for a glance); the head's aim
wanders by a fraction of a degree (a slow noise seeded by the member's name).

#### Body

| Key | Meaning |
|---|---|
| `breath` | `{per_min = 16.5, deg = 0.6}`: chest pitch on `upper_body2` |
| `sway` | `{deg = 0.37, period = 2.5}`: a slow turn of the upper body about the vertical |
| `nod` | `{deg = 0.48, period = 2.3}`: a slow nod of the head, fading while a gaze event holds the head away |
| `bob` | `{deg = 1.5, beats = [t, ...], timeline = "audio/timeline.json", downbeat_accent = 1.6}`: the head dips after each beat (60 ms attack, 220 ms decay). Needs `beats` (clip seconds) or a timeline file (`timeline`, default `audio/timeline.json`: its beats; its downbeats dip `downbeat_accent` times deeper; a missing file stops the build naming `perform.<cast>.bob`); `beats` wins |
| `startle` | `[t, ...]`: at each time the upper body jolts backward (4.5 degrees, settling with a 0.35 s time constant) |
| `lean`, `turn`, `tilt` | `[[t, deg], ...]`: extra upper-body forward lean, turn toward the left and sideways tilt toward the left over time, smoothstep-eased between the keys, held before the first and after the last, on top of the pose's base: a reach that leans in and settles back, a head on a shoulder. Hand targets still hold, except hands whose pose `ride` is a chest bone, which go with it. The first key's value holds from the start of the pre-roll: begin with a `[0, 0]` key to start from the base |
| `head_tilt` | `[[t, deg], ...]`: the head rolls toward the left on top of the gaze, keyed the same way |
| `head_turn` | `[[t, deg], ...]`: the head turns toward the left on top of the gaze, keyed the same way: a look back over the shoulder |
| `rock`, `head_rock` | `{deg, period = 4.0, phase = 0.0, axis = "side", hips = 0}`: a periodic sway of the upper body and of the head: `deg × sin(2π t / period + phase)` of clip time (`phase` in radians), added to the keyed ones; `axis` `"side"` leans the body (like `tilt`) and rolls the head (like `head_tilt`) toward the left, `"front"` leans the body forward (like `lean`) and nods the head, `"turn"` twists the body (like `turn`) and shakes the head. Unlike the keys it goes on at any clip time, so a dreamy sway survives a re-cut; a squirm seen side-on is a fast `turn` (a swing a beat). `rock.hips` (metres) shifts the hips toward the left with the same swing, the feet planted, so the weight goes from foot to foot: a groove without steps |
| `bounce` | `{depth = 0.03, beats = [t, ...], timeline, downbeat_accent = 1.6, from, to, attack = 0.05, decay = 0.16}`: the hips (the `center` bone) dip `depth` m on each beat (downbeats weighted), the feet planted, so the knees pump; `from`, `to` (clip seconds) limit it |
| `rise` | `[[t, metres], ...]`: the whole body, feet too, lifts by that much (eased between keys): she floats |
| `crouch` | `[[t, metres], ...]`: the hips drop by that much, the feet planted, so the knees bend (eased between keys, on top of `bounce`): a landing, a squat, a brace before a jump |
| `kick` | `{foot = "L", height = 0.12, back = 0.08, hold = 0, beats, timeline, downbeat_accent, from, to, attack = 0.06, decay = 0.18}`: that foot flicks up and back on the beats; `hold` (0..1) keeps it that share of the way up between them (a foot kicked up behind a lovestruck girl) |
| `air` | `{from, to, tuck = 0.24, back = 0.10, first = "L", share = 0.7, lag = 0.06, rise = 0.14, extend = 0.22, point = 30}` (or a list): the legs of a body off the ground (a jump, a blast; its height is the root's, keyed by `[[key]]`) from take-off `from` to landing `to`: over `rise` s both feet come up toward the hips by `tuck` m and back by `back` m (the knees bend forward), the foot `first` ahead and the other `lag` s later by `share` of it, the toes pointing `point` degrees down; they hold still (a [freeze](#freezes) in the air shows no jump when the world starts again) and over the last `extend` s reach down again, so the feet are under the hips at the landing. Added to `kick`. Mind the knees: a model's legs straight while its root is lifted look hung, not jumped |
| `step_turn` | `[{t, deg, dur, first, lift = 0.045, dip = 0.02, move}]`: the body turns round on the spot like a person, not a figure on a turntable. From `t` the cast root turns `deg` degrees toward the left (negative: right) over `dur` s (default 0.2 s and 0.11 s per 100 degrees: 0.36 s for 145), and travels `move` `[x, y]` m meanwhile (in the character's frame before its first turn: x its left, y behind; a step toward what it turns to), the chest leading the hips (by at most 30 degrees) and unwinding as they arrive; the foot `first` steps round, then the other (by default the order that keeps the feet further apart: about a point in front of the ankles or behind them, the other order sets a foot down where the other still stands), each lifted `lift` m between rolling off and setting down, one foot always down and none sliding; a foot is never more than 35 degrees from the hips (it swivels on its ankle), and the hips sink `dip` m meanwhile. The report has `feet_gap_mm`, the least distance between the ankles. Give the head a `gaze` at what the body turns to, a moment before, and it leads the chest. The root's yaw, and the axes `move` travels along, are keyed here, so a `[[key]]` on them is an error |
| `walk` | `{steps = [t, ...], hops = [{from, to, move, height}], stride = 0.55, first = "L", lift = 0.07, dip = 0.015, slide = 0, roll = 0.15, hop = 0, arms = 0, cross = 0, sway = 0}`: the body walks (`mkmmd.core.perform.walk_plan`). Each step is a foot coming down (alternating from `first`) a `stride` past the other foot, on its own side of the body, or `cross` of the way in to the body's centre line (1: on it, a catwalk; up to 1.5: across it); it rolls off `roll` of the step's time after the other came down, swings through `lift` m and lands on the step, never sliding while down. A step more than 1.2 s after the one before (or after a hop's landing) starts from a stand, as the first does: the body waits where it stood and the foot swings over the last 0.35 s, so she can stop, hold a pose and walk on in one walk. The cast root travels with the body (straight ahead, at a steady pace, midway between the feet each time one lands) and the hips dip `dip` m at each landing; `sway` (m) carries the hips over the foot that is down, out and back while the other swings (a strut; the feet stay planted). Over a hop both feet leave the ground from `from` to `to`, the root flies a parabola `height` m high (default what the flight time gives, g T² / 8; `hop` sets one for every hop), carried `move` `[x, y]` m (its left, behind; default a stride ahead: a knight's L is `[1.2, -0.6]`), and the legs tuck as `air` does; the feet land a quarter stride either side of the body and the foot behind steps next. `slide` 1 puts the walk on a treadmill (the body stays, the feet down slide back under it: running on the spot), `arms` (m) swings each hand's IK goal forward as the other side's foot goes forward, while the feet move: it dies away over 0.25 s once they stop, so a hand posed between two walks is where its pose put it (the hands need a pose: a `rest` move gives one, and a hand riding the chest swings with it). The root's location is keyed: a `[[key]]` on it must be `relative` (a push on top of the walk). A walk and a `step_turn` in one table are refused: face the walk with `[[cast]] yaw`. The report has `walk` and `sway_mm` |
| `tumble` | `[{from, to, deg, axis = "flip"}]`: the whole body turns in the air about its hips (the root turned and moved so the hips stay put), `deg` over `from`..`to` eased in and out, about its left axis (`"flip"`, positive: head forward) or its forward axis (`"cartwheel"`, positive: toward its left); turns add up and hold (360 lands upright). Give it `air` (or a walk's hop) so the legs tuck |
| `drum` | `{hand = "R", fingers = ["little", "ring", "middle", "index"], deg = 25, lift = 0.16, roll = 0.02, beats, timeline, downbeat_accent, from, to}`: the fingers of one hand tap on the beats (a driver drumming on the wheel): each lifts at its base joint by `deg` (the middle joint by a third of it) about its own flexion axis, from wherever its keys hold it (a grip's solved curl), over the first half of the `lift` seconds before the beat and falls, fastest at the end, back onto it on the beat; the little finger leads the index by `roll` s per finger. The palm stays where the grip put it, so a hand wrapped round a rim never moves through it. The report has `drum` (`hand`, `fingers`, `taps`, `deg`, `frames`) |

#### Face and mouth

| Key | Meaning |
|---|---|
| `blink` | `{per_min = 15, seed = 0, extra = [[t, dur], ...]}`: natural blinks at a mean interval of 60 / `per_min` seconds (±45 %, at least 1.2 s apart, drawn from `seed`), each 0.157 s (no key sets the duration of the natural ones); `extra` adds blinks of the given duration at chosen times |
| `lids` | 0..1, the lids' base lowering: a floor under the `blink` morph, eased in over the settle and open while a gaze event holds the head away and during a glance (default 0) |
| `expressions` | `[{morph, keys = [[t, value], ...]}]`: a morph's value over time, Bezier between the keys. `morph` is a semantic name (rig.json `morphs`) or the model's own, of any kind (a group or material morph through the bound sliders, see [Cast](#cast)); the stage raises when the model has no such morph. An expression on the `blink` morph replaces the blinks, and lip sync replaces an expression on a vowel morph |
| `sing` | lip sync, below |
| `twitch` | `[{bones, family, t, deg = 14, axis = [1, 0, 0], dur = 0.22}]`: a quick flick (out to `deg`, back past rest by a quarter of it, home) about an axis in the armature's frame. `bones` lists semantic or Blender names, `family` (`"ears"`, `"tail"`) takes the top bone of every chain of that family by its name; `t` is required |

`sing` is `{timeline, lines, words, mouth = 0.8, lead = -0.03, voice = "en-gb"}`. The words of the timeline (`lines[].words[]` with
`text`, `start` and `voiced_end` in clip seconds; the text is read and never printed) become IPA phonemes with `espeak-ng` and
then the five vowel morphs `a i u e o`: vowels share the word's sung span (the last one takes the held note), consonants take a
short slice, the mouth closes for m, b, p and rests between words further than 0.14 s apart, and its size follows the vocal
loudness when the timeline has `vocal_db`. `timeline` defaults to `audio/timeline.json` (a missing file stops the build naming `perform.<cast>.sing`). `lines` is `[a, b]`, lines a to b (1-based, inclusive), or
any other list of line numbers (default all lines); `words` instead picks single words, `[[line, word], ...]` (1-based; a negative
word counts from the end of its line, so `[[1, -1], [2, -1]]` are the line endings: a hype partner shouting them), and a choice the
timeline does not have is a build error. `mouth` is the peak weight at full voice, `lead` shifts every key (a small
negative lead reads better on screen), `voice` is the `espeak-ng` voice, the language of the words. `espeak-ng` must be
installed: `mk doctor` reports whether it is found, and without it the stage raises.

A twitch is keyed on the bone itself on top of its other keys, so twitch the bone a chain hangs from (an `ear_root`): the sim
stage keys the chain's own bones and replaces any twitch keys on them.

#### Strum

`strum` is a table or a list of tables: the pick hand strokes across the strings of a worn guitar-like prop on the song's
rhythm, the pick tip meeting the first string at each strike time, down strokes going down and up strokes up, and rests
between windows (`mkmmd/core/strum.py`). The hand's pose table needs `grip = "<prop>:strum"` ([Playing a worn
guitar](#playing-a-worn-guitar)).

| Key | Meaning |
|---|---|
| `hand` | `"R"` (default) or `"L"`: the pick hand; its `[pose.<cast>.hands.<side>]` needs the strum grip, whose goal rides the prop |
| `prop` | the name of the guitar the hand grips (required) |
| `grip` | the prop's `use.grip` point of the strum zone (default `"strum"`) |
| `rhythm` | `"onsets:<stem>"` (the timeline's `onsets`, see [Timeline](#timeline): `mk timeline onsets`), `"beats:N"` (N strokes per bar of four beats, a multiple of 4: 8 is eighths) or a list of strike times in clip seconds (default `"beats:8"`) |
| `from`, `to` | clip seconds: only the strikes inside are played (default the whole clip) |
| `accent` | `"downbeats"`, `"beats"` or a list of times: strikes within `accent_window` of them are accented (wider and deeper by `accent_gain`) |
| `accent_gain` | an accented stroke's amplitude and depth (default 1.25) |
| `accent_window` | seconds (default 0.06) |
| `up_scale` | an up stroke's amplitude and depth relative to a down stroke (default 0.85) |
| `min_gap` | strikes closer than this many seconds are one stroke, the earlier (default 0.09) |
| `beat_tol` | in beats: how close to a beat or an offbeat a strike must be to take its direction from it (default 0.22) |
| `span` | metres: the full sweep of a normal stroke, centred on the strum centre (default 0.09) |
| `depth` | metres the pick tip is pressed into the strings at the strike (default 0.003) |
| `lift` | metres the pick is clear of the strings at the ends of the sweep (default 0.014) |
| `return_lift` | metres the pick rises over the strings when it comes back for a stroke in the same direction (default 0.022) |
| `attack` | seconds from the first string to the last, the audible spread of a strum (default 0.075) |
| `lead`, `follow` | seconds of run-up before the first string and of follow-through after the last (default 0.09 each) |
| `approach`, `retreat` | seconds the hand takes from rest to the first stroke and from the last stroke back to rest (default 0.3 each) |
| `share` | the share of the sideways travel that the hand makes by turning about the wrist, about an axis along the strings; the rest is made by moving the hand (default 0.6) |
| `kick` | degrees: on every accented stroke the guitar's neck kicks up that far (the prop turns about its face normal through its origin and eases back; the hands ride it). In the first table of a hand |
| `windmill` | `[t, ...]`: before the down stroke nearest each time (within 0.12 s) the pick hand swings a full circle, over `windmill_dur` seconds (0.5): its goal turns once about an axis through the shoulder along the guitar's face normal and comes down onto the strings at the strike, which then plays as planned. In the first table of a hand; read it in a front or three-quarter view (side-on the circle is edge-on) |
| `sigma` | half the width of the six strings at the strum centre, metres (default taken from the card's strings, 0.0225 without a `neck` point) |
| `timeline` | the timeline JSON with `onsets`, `beats` and `downbeats` (default `audio/timeline.json`) |

A strike on a beat goes down, one on the offbeat up, an off-grid one alternates (`beats:N` takes its directions from its grid:
beats down, the subdivisions between them alternate). Alternating strokes share their turn-round, so eighths are one continuous
swing; a stroke in the same direction as the one before first returns over the strings, lifted. When the time between one stroke's
follow-through and the next run-up is longer than `approach + retreat + 0.1` s the hand goes back to rest (the tip on the string
plane at the strum centre: the grip) and comes in again. The path is turned into IK goals relative to the grip and keyed every
frame (LINEAR) from the first approach to the last retreat: the hand moves as a rigid body and turns about the wrist for `share`
of the sideways travel (not at all when turning would hardly move the tip sideways, never more than 0.6 rad). Tables that
share a hand take the first one's `prop`, `grip`, `sigma`, `timeline` and path numbers (`span`, `lift`, `return_lift`, `attack`,
`lead`, `follow`, `approach`, `retreat`, `share`); `rhythm`, `from`, `to`, `accent`, `accent_gain`, `up_scale`, `accent_window`,
`min_gap`, `beat_tol` and `depth` are each table's own.

The report has `strokes`, `down`, `up`, `frames` (first and last frame keyed), `sigma_mm`, `turn_share`, `first` (the first
strike times) and `max_u_mm` (the widest sweep). With no strike in the window the stage logs a WARNING and keys nothing; it raises
when the hand has no strum grip or its goal does not ride the prop. The `strum` check replays the plan on the timeline against
the scene.

```toml
[perform.rin]
look = "car:road"
gaze = [{ t = 4.2, at = "car:mirror", hold = 1.2 }, { t = 9.0, at = "cast:ann.head" }]
glance = [{ t = 6.5, at = [3.0, -2.0, 1.2], dur = 0.6 }]
bob = { deg = 1.5, timeline = "audio/timeline.json" }
lean = [[0, 0], [4.0, 6], [5.5, 0]]
blink = { per_min = 15, seed = 2 }
lids = 0.15
sing = { timeline = "audio/timeline.json", lines = [1, 4], mouth = 0.8 }
expressions = [{ morph = "smile_eyes", keys = [[2.0, 0], [2.4, 0.8], [6.0, 0.8], [6.5, 0]] }]
twitch = [{ bones = ["ear_root.L", "ear_root.R"], t = 3.2, deg = 14 }]
```

The report has, per member, `gaze_events`, `glances`, `frames` (the number of frames keyed), `blinks` (how many were scheduled),
`blink_morph` (the morph keyed; `null`, with a log line, when the model has none), `sing_morphs` (how many of the five vowel
morphs the model has), `sing_keys` and `strum` (per hand).

### Lights and the look

`[[light]]` makes lights in palette colours, mounted, aimed and keyed; `[look]` is the colour pipeline: palette, view transform,
contrast look, exposure. The stage runs after shots (a `look = "camera"` aims at the scene camera as it stands at `frame0`) and
before text and keys. Sets and props bring their own lights (a moon, lamps, a desk lamp); these are the project's.

| Key | Meaning |
|---|---|
| `name` | names the light (required) |
| `kind` | `"point"` (default), `"spot"`, `"area"` or `"sun"` |
| `color` | a palette slot or `"#rrggbb"` (default `text`) |
| `power` | watts for point, spot and area, strength for a sun (default 40, so give a sun its own) |
| `mount` | the prop, set or object the light rides on (default the world) |
| `at` | `[x, y, z]` in the mount's frame (default `[0, 0, 3]`) |
| `look` | a [target](#targets) the light is aimed at, read at `frame0` and not tracked afterwards (default straight down; spot, area and sun use the aim) |
| `size` | metres: the radius of a point or spot, the edge of an area (`[w, h]` for a rectangle); a sun ignores it (default 0.1) |
| `spot` | `{angle = 45, blend = 0.3}`: a spot's cone in degrees and the softness of its edge |
| `angle` | a sun's angular diameter in degrees (default 1) |
| `shadow` | cast shadows (default true) |
| `specular`, `diffuse` | factors (default 1) |
| `volume` | the volume-scattering factor (default 0) |
| `keys` | `[[t, multiplier], ...]`: intensity over clip time, multiplying `power`, Bezier between the keys, held outside them |

The colour is not keyed here: `[[key]]` with `prop = "data.color"` and an `index` keys it (linear RGB).

The `[look]` table (`[look.slots]` sits beside it):

| Key | Meaning |
|---|---|
| `palette` | the palette every builder colours by: `rose-pine-moon` (default), `rose-pine` or `rose-pine-dawn` ([Palettes](#palettes)) |
| `slots` | `{slot = "#rrggbb"}` overriding palette slots (`[look.slots]`) |
| `view` | Blender's view transform: `"AgX"` (default), `"Standard"`, `"Filmic"` |
| `contrast` | the name of a Blender look, `"Medium High Contrast"`; with the AgX transform the `AgX - ` prefix is optional (default: the look as the set builders left it) |
| `exposure` | stops (default 0) |
| `gamma` | default 1 |

The stage always writes `view`, `exposure` and `gamma` and writes `contrast` only when given, so the look a set builder chose
(the café and bedroom sets pick `AgX - Base Contrast`) stays otherwise. The settings live in the scene's view settings, so
`mk render` and `mk look` draw with them (the grade is `[post]`).

```toml
[[light]]
name = "dash"
kind = "point"
color = "gold"
power = 12
mount = "car"
at = [0.38, 0.1, 0.9]
size = 0.05
keys = [[0, 0], [1.0, 1.0], [8.0, 1.0]]

[[light]]
name = "moon"
kind = "sun"
color = "foam"
power = 2
look = [0, -50, 0]

[look]
palette = "rose-pine-moon"
view = "AgX"
contrast = "Medium High Contrast"
exposure = 0.3

[look.slots]
love = "#e5546f"
```

The report has the `look` settings as applied (`view`, `look`, `exposure`) and, per light, `kind`, `color`, `power` and `mount`.

### Keys

`[[key]]` keys any property of a set, prop or object on clip seconds: a set's storm and fog, a car's lamps, a light's energy, any
RNA path (`location`, `hide_render`, `data.energy`). The stage runs after text, so a text object can be hidden or moved, and
before sim, so the keys count as motion the chains and colliders react to.

| Key | Meaning |
|---|---|
| `target` | a set or prop name (its root) or any object name (required) |
| `prop` | a custom property of the target (set and prop cards document theirs: `fog`, `lamps`, `blinds`) or an RNA path on the object: `location`, `rotation_euler`, `hide_render`, `pose.bones["bone"].rotation_quaternion`; with a `data.` prefix the path is read on the object's data: `data.energy`, `data.color` (required) |
| `index` | the array index of a vector path (default 0): one `[[key]]` per component |
| `keys` | `[[t, value], ...]`: clip seconds and numbers, sorted by time (required, unless `lane`) |
| `lane` | instead of `keys`: a lane of the `[audio] hits` file ([Timeline](#timeline), `mk timeline hits`). On every hit the property takes the hit's own time, CONSTANT unless `interp` says otherwise, so a prop's `start` (a `burst`) goes off again on each kick; `min_db` keeps the hits at or above that level (0 the lane's loudest), `every` every n-th of them (default 1), `from` / `to` those inside that span of clip seconds (`mkmmd/core/lanes.py`). A lane with no hit left is an error |
| `pulse` | with `lane`: `[[dt, value], ...]`, keys that many seconds after every hit instead (LINEAR by default), the next hit cutting a pulse short: a lamp that flashes on every kick |
| `interp` | `"BEZIER"` (default), `"LINEAR"` or `"CONSTANT"` (a visibility switch) |
| `relative` | `true`: the values are offsets added to what the property already does at each key, its fcurve or its static value (a value another stage solved or keyed, such as an IK goal's location), in the property's own space, and the existing animation outside the first and last key stays. Default false: the keys replace the property's animation |

```toml
[[key]]
target = "cafe"                # a set root: its custom property `storm`
prop = "storm"
keys = [[0, 0.0], [4, 0.0], [4.2, 1.0], [5, 0.0]]
interp = "LINEAR"

[[key]]
target = "rin_hand.R"          # an IK goal empty: offsets on its location, Y
prop = "location"
index = 1
relative = true
keys = [[2.0, 0.0], [2.5, 0.05], [3.0, 0.0]]
```

```toml
[[key]]
target = "glass"               # a burst of shards: off again on every loud kick of the first 12 s
prop = "start"
lane = "kick"
min_db = -1.5
to = 12.0

[[key]]
target = "rim"                 # a light: a flash on every snare, back by 0.12 s
prop = "data.energy"
lane = "snare"
pulse = [[0.0, 900.0], [0.12, 150.0]]
```

The report maps `<target>.<prop>` to the number of keys; a target or property that does not exist, or an empty `keys`, raises.

### Sim

`[sim.<cast>]` simulates secondary motion (hair, ears, tails, skirts, ribbons) outside Blender, with `mkmmd.solvers.strands`
(numpy, with optional numba kernels), and bakes it to keys. It runs last, after everything that moves, because the solver reads
the final animation of the chains' anchors, the model's own collision bodies, the props (and their keyed motion) and the fingers.
It needs the member's `rig.json` (the chains and bodies come from it; a member without one raises). `mk build --skip sim` leaves
it out for a quick look at poses and performance.

| Key | Meaning |
|---|---|
| `families` | the chain families to simulate: `bangs`, `side_hair`, `back_hair`, `twintail`, `braid`, `hair`, `ears`, `tail`, `skirt`, `ribbon`, `breasts`, `sleeve`, `coat`, `accessory`, `other` (default the hair families, `ears` and `tail`: those the rig has). With none of them in the rig the stage logs it and skips the member |
| `params` | `{family: {...}}`: solver parameters per family, below |
| `colliders` | the scene's collision shapes: the name of a `[colliders]` set or a list of specs ([Colliders](#colliders)) (default none) |
| `props` | add every prop card's colliders: seats, doors, dash, wheel ... (default true) |
| `fingers` | add finger and palm capsules of every cast member (default true) |
| `floor` | the ground: `true` (default) is the ground at z 0, a number that height in metres, `false` no floor |
| `wind` | the air, below |
| `gravity` | `[[t, scale], ...]`: the world's gravity over clip time, 1 as it is and 0 none (the hair floats, held only by its pull toward its modelled shape), linear between keys and held outside them; the first frame settles under the first key's. A zero-g float, the music cutting out, a fall: key it with the moment (default 1 throughout) |
| `use_masks` | honour the PMX collision masks everywhere, not only within `anchor_free` of a chain's root (default false) |
| `anchor_free` | metres near a chain's root where it does not collide with the body it hangs from and the PMX masks apply (default 0.25) |
| `substeps` | solver steps per frame (default 10) |
| `settle_s` | seconds the first frame is settled under gravity before the run (default 1.5) |
| `engine` | `"auto"` (default: numba when installed, `pip install 'mk-mmd[fast]'`), `"numpy"` or `"numba"` (an error without numba); the same dynamics and the same results to rounding |

The `penetration` check takes the same `use_masks` and `anchor_free` and measures like the solver; the model's own bodies collide
too, except those of the simulated bones (they move with the hair), and a chain skips a body it already overlaps at rest
([Colliders](#colliders)).

`params.<family>` overrides the family's defaults (an unknown family or key makes the solver refuse, exit 2):

| Key | Meaning |
|---|---|
| `sag` | `[root, tip]` degrees (a number is both): the angle a horizontal segment settles at under gravity; small keeps the modelled shape, large hangs |
| `drag` | 1/s: the air drag, velocity relaxes toward the air's (still air: toward 0) |
| `zeta` | damping ratio of the motion relative to the rest shape (1: a jolt dies without ringing) |
| `radius` | times the PMX body radius of the chain's bones, capped at `radius_max`: a particle's collision radius |
| `radius_max` | metres, the cap of `radius` |
| `friction` | 1/s: velocity damping while touching something |
| `wind_drag` | 1/s: how strongly the air pushes the strand, it reaches `wind_drag / drag` of the air speed (default `drag`) |
| `lateral` | 0..1, the tie stiffness of neighbouring chains (default 0.5 for `skirt` and for chains the rig ties, 0 elsewhere): chains whose rig.json `ties` join them (the joints between their bodies: a row under a sleeve as well as a ring) are tied to those neighbours; a skirt's other chains are tied to their neighbours in a ring found from their roots' rest positions (a slit breaks the ring); the cloth between them is kept out of seats and thighs; 0 = independent chains |
| `lateral_collide` | 0 or 1: push the cloth between tied chains out of the shapes (default 1) |
| `anchor_free` | metres: the family's own zone near the roots (default the stage's; 0 for a tied sheet, which then also collides with the body it hangs from) |
| `inertia` | 0..1, the share of its anchor's motion the strand feels (default 1, all of it; a `skirt` 0.5): each substep first carries the strand along by the rest, as a game engine's inertia scale does. For moves faster than cloth could follow: a 145 degree spin in four frames flares a skirt over her waist at 1, half as far at 0.5; 0 carries the strand rigidly and only gravity, the air and contacts move it |
| `bend_max` | degrees: the most a segment turns from its rest shape on its parent, a joint's limit (default none; a `skirt` 30). A skirt pinched between a resting hand (`place = "dainty"` rests one on it) and a thigh drapes around the hand instead of folding over on itself |

Defaults per family (`sag` root / tip in degrees, `radius_max` in millimetres). `ears`, `bangs`, `side_hair` and `back_hair` are
calibrated on a seated close-up and `skirt` on a dancer's spins, falls and landings (Hello Brooklyn): stiff, almost no air, its
motion damped hard toward hers, with `inertia` 0.5 and `bend_max` 30 too, so a skirt goes with her instead of flying up or
crumpling round her waist; the other rows are plausible starting points to tune:

| Family | sag | drag | zeta | radius | radius_max | friction |
|---|---|---|---|---|---|---|
| `ears` | 0.6 / 1.2 | 5 | 1 | 0.45 | 20 | 10 |
| `bangs` | 3 / 8 | 5 | 1 | 0.5 | 12 | 10 |
| `side_hair` | 12 / 60 | 5 | 0.7 | 0.35 | 10 | 20 |
| `back_hair` | 22 / 90 | 8 | 0.8 | 0.5 | 22 | 25 |
| `twintail` | 18 / 85 | 7 | 0.75 | 0.5 | 22 | 22 |
| `braid` | 15 / 60 | 8 | 0.9 | 0.5 | 22 | 25 |
| `hair` | 15 / 70 | 6 | 0.8 | 0.5 | 20 | 20 |
| `tail` | 10 / 45 | 4 | 0.6 | 0.5 | 40 | 15 |
| `skirt` | 1.5 / 5 | 0.2 | 3 | 0.5 | 30 | 12 |
| `ribbon` | 20 / 80 | 3.5 | 0.5 | 0.5 | 10 | 10 |
| `breasts` | 1 / 2 | 8 | 1 | 0.5 | 50 | 10 |
| `sleeve` | 6 / 25 | 5 | 0.8 | 0.5 | 30 | 15 |
| `coat` | 8 / 30 | 5 | 0.9 | 0.5 | 40 | 15 |
| `accessory` | 15 / 45 | 5 | 0.8 | 0.5 | 20 | 15 |
| `other` | 10 / 45 | 5 | 0.8 | 0.5 | 20 | 15 |

`wind` is the air's velocity at a point: a constant wind + the carrier's velocity × (1 − `exposure`) + gusts + turbulence. All keys
are optional:

| Key | Meaning |
|---|---|
| `direction` | `[x, y, z]` the way the air blows to, any length (default against the carrier's motion, else +X) |
| `speed` | m/s of the constant wind along `direction` (default 0) |
| `carrier` | the `prop` of a `[[vehicle]]` the character rides in: its velocity enters the air |
| `exposure` | 0..1: the share of the carrier's motion the air does not follow (default 0: the air rides along, a closed cabin; about 0.3 for a convertible; 1: still air, the character moves through it) |
| `gust` | m/s amplitude of a slow irregular swell of the speed along `direction` (default 0) |
| `gust_period` | seconds (default 4) |
| `turbulence` | m/s rms of a smooth divergence-free fluctuation, carried along with the mean flow (default 0) |
| `scale` | metres: the turbulence's eddy size (default 0.5) |
| `seed` | the random seed of gusts and turbulence (default 0) |

A carrier at constant velocity with `exposure = 0` changes nothing (the strands move on with it); `exposure` above 0 blows the
hair back. The carrier's velocity comes from the vehicle's position on every frame, so the stage needs `vehicles` (a carrier that
is no vehicle raises).

Branching chains (a strand that splits) are solved as a trunk and branches: the trunk follows the deepest subtree, every other
subtree becomes a chain anchored on the bone it grows from. Trunks and plain chains are solved first and keyed, then the branches
are solved riding the keyed trunks, level by level (one `passes` entry each). Every simulated bone gets LINEAR rotation keys on
every frame of the build, the pre-roll included, so the hair settles while the character eases into its pose; the first frame is
settled under gravity for `settle_s` first. Strands are inextensible (exact segment lengths every substep), so the baked bones
render the solved particles and the penetration reported is the penetration that renders. Results are cached in
`<project>/.mk/cache/strands/` by a hash of the inputs and of the solver's source, one entry per member and level.

```toml
[sim.rin]
families = ["back_hair", "bangs", "side_hair", "ears", "skirt"]
colliders = "car"
floor = false
wind = { carrier = "car", exposure = 0.3, gust = 1.5 }

[sim.rin.params.back_hair]
sag = [20, 85]
drag = 9

[sim.rin.params.skirt]
lateral = 0.6
```

The report has, per member, `families`, `colliders` (resolved shapes), `floor_z` (the ground's height, null without one), `bones`, `penetration_mm_max` (the deepest chain point in
any shape from `frame0` on) and `passes`: per level `chains`, `bones`, `engine`, `seconds` and `report`, which gives per family
`jerk_mm`, `speed_mm`, `ratio` (the `jitter` check's jerk over speed: calm hair is about 0.4, Bullet hair on a seated model about
1.4) and `jerk_p95_mm`, and `penetration_mm` (`max`, `at_frame`, `bone`, `into`, `median`, `frames_over_2mm_by_shape`). A
`params` or `wind` the solver does not accept stops the build with its message.

The stage also swings the cord of every worn prop ([Playing a worn guitar](#playing-a-worn-guitar)), with or without a
`[sim.<cast>]` table: `mkmmd.solvers.cable`, a rope of particles 3 cm apart from the jack to the floor, inextensible, with
gravity, air drag (a swing dies in about a second), a little bending stiffness, static and sliding friction on the floor under
the wearer, and collisions with the wearer's bodies from `rig.json` (not the hair's, ears' or tail's; none within 12 cm of the
plug). The plug follows the jack on every frame and the cord leaves it along the anchor's `dir`; the far end lies where the
pose stage hung it. The first frame is held for 2 s so the cord settles into its own shape, and the curve's control points are
keyed on every frame (LINEAR), replacing the hook. The report has `cables`: per prop `points`, `length_m`, `segment_mm`,
`stretch_pct_max`, `body_pen_mm_max`, `floor_pen_mm_max`, `seconds`.

### Folds

The folds stage keys the fold shapes a cast member's rig declares (rig.json `folds`, [Model description](#model-description-rigjson)):
for every frame of the build it reads the driving bones' rotations in the armature's own space (a model spinning in the
world turns no fold), works out each driver's angle (`mkmmd.core.folds`) and keys the shape's weight LINEAR on the morph
sliders. It runs after sim, when every bone has its final keys. A member whose rig has folds gets them without a table;
`[folds.<cast>]` takes `enabled = false` (none of them) and `depth` (scales every weight, default 1). The report has per
member `folds` and `keyed`: the most of each shape any frame shows. A fold on a piece the cloth stage simulates for the
member (its `piece` the name of a cloth copy) is left to the cloth, and the report lists those (`left_to_cloth`).

### Cloth

The cloth stage simulates the cloth copies a cast member's rig declares (rig.json `cloth`,
[Model description](#model-description-rigjson)) with Blender's cloth, and the garment's faces follow the simulated
cloth. It runs after sim and folds, when every bone has its final keys, for a member with a `[cloth.<cast>]` table (an
empty one will do):

| Key | Meaning |
|---|---|
| `enabled` | false: no cloth for this member (default true) |
| `pieces` | the copies to simulate, by name (default every one the rig declares) |
| `quality` | Blender's cloth steps a frame (8; the sweater's 10) |
| `mass` | kg a vertex of the copy (0.3) |
| `tension`, `shear`, `bending` | stiffness against stretching and squashing, shearing and bending (10, 5, 0.5) |
| `air` | air damping (1) |
| `distance` | metres the cloth keeps off the skin it collides with (0.003) |
| `hold` | 0..1: how much the free cloth is held where the armature carries it, each vertex at least its copy's `holds` (the sweater's 0.3, its sleeves' 0.9: it keeps its shape and swings and lags on it; near 0 the cloth goes where gravity and the body put it, as a skirt that swings) |
| `grip` | how hard what is held is pulled back there (Blender's pin stiffness, 1) |
| `cache` | false: simulate again though a simulation of the same inputs is cached (default true) |

`[[cloth.<cast>.make]]` makes a cloth copy from the member's own faces, for a model that brings none (an imported
character's blazer; the maths is `mkmmd/core/garment.py`, numpy only). It is a copy like the rig's, by its name (`pieces`
picks among both; a rig copy of the same name is an error):

| Key | Meaning |
|---|---|
| `name` | the copy's name (required) |
| `materials` | the PMX names of the garment's materials (required) |
| `collide` | the PMX names of the materials of the body under it (required): give all it covers, a skirt over a shirt the shirt too, or the cloth falls in through the hole. The skin within `reach` of the garment (0.15 m) is its collider, each sheet wound outward from the bones and set in by `inset` (0.003 m) |
| `pin` | `{above, fade = 0.06}`: carried by the armature above `above` metres (rest height), free below `above - fade` (default: nothing pinned) |
| `sleeves` | `{hold = 0.9, collides = false}`: what the arm bones carry most of is held that much toward where the arm carries it (a sleeve on an arm that moves faster than cloth) and collides with the body or not |
| `open` | degrees, or `{deg, below, fade = 0.06}`: an unbuttoned front, eased in over the settle: each front panel slides round the torso about its side, toward its own side, as a jacket hangs open; with `below` only under that height (the lapels above the top button stay) |
| `trims` | the PMX names of hard pieces on the garment (buttons, a brooch): every loose piece of them within `gap` (0.01 m) of the garment, whole, rides the cloth's triangle nearest its centre, never stretched |
| `thick` | metres: a modelled garment's lining is what lies within it under the outside (0.003) |

`make` keeps the garment's outside (no lining), collapses the slivers a modeller left (edges under 3 mm), keeps two faces
to an edge (Surface Deform binds to nothing else) and winds every face outward. A lapped closure (a jacket's front laps over
itself) is found: each side of the lap is a part of the cloth, and the garment's faces of each side bind to their own part,
not to the one lying on them (the report's `laps` and `lapped`). The report adds `made`: `trims`, `pinned`, `sleeves`.

The garment brings its own values (rig.json `settings`, e.g. `[sweater.cloth]`); the table's win. For each copy the stage
makes `<root>_<name>_cloth`: the copy skinned to the armature, its pinned vertices following their skin and the others
held by the larger of the copy's `holds` (the sweater's sleeves', `[sweater.cloth] sleeve_hold`) and `hold`, colliding
with the copy's skin (skinned too; single sided, so cloth found inside goes back out along the skin's normal), Blender's
cloth baked from the build's first frame to its last with its rest shape taken from her skin on every frame (Dynamic
Mesh: the cloth keeps the garment's ease in any pose and adds its own motion to it). The frames go to
`<cache>/cloth/<key>.pc2` (the key hashes the copy, its settings, the frames and, on every frame, where each bone that
carries it is in the world, so a rebuild with the same motion reuses them) and beside the .blend to
`<blend>.cloth/<cast>.<name>.pc2`, which a Mesh Cache modifier plays on the cloth: the saved scene holds no physics. The
model's faces of the copy's `materials` move to a mesh of their own, `<root>_<name>`, bound to the cloth (Surface Deform,
the cloth triangulated) on the first frame; each of its vertices follows the cloth as much as the copy near it is free
(the vertex group `mk_cloth`), so what a band holds keeps its skin's motion exactly. Fold shapes on a simulated piece are
left to the cloth; leave the chains it replaces (the sweater's `sleeve`) out of `[sim.<cast>] families`.
`mk build --skip cloth` is the fast loop.

The report has per member and copy `seconds` and `per_frame_ms` (what the simulation costs), `cached`, `pen_mm` and
`pen_frame` (the deepest the free cloth goes into the skin it collides with over the clip, and on which frame),
`pen_over_2mm` (the share of the free cloth's vertex-frames deeper than 2 mm), `lag_mm` (how far the free cloth trails
where the armature alone would carry it: most of it, p99 and the most), `jerk_mm` (the largest change of a free vertex's
frame-to-frame motion over the clip) beside `jerk_pinned_mm` (the same for the pinned ones, which move as her skin does,
so more than theirs is the cloth's own jump), `stretch_pct` (the most an edge with a free end stretches beyond where the
armature carries it),
`verts`, `faces`, `collide_faces`, `settings`, `key`, `pc2` and `follows` (the mesh that follows the cloth and its vertex
count). That mesh is too big for the bake to hold frame by frame in all but short clips: the site's scene viewer then
goes without it and `mk render --engine mk` draws those shots in Blender (the bake report's `not_by_frame` says so).

### Stepped motion

`[[stepped]]` puts cast members on twos for a window of clip seconds, the way hand-drawn animation is shot: a new pose `fps`
times a second, each held until the next, while the cameras keep every frame (a figure on 12 against a camera on 60 is the
look of an animated film's character on its background). The stepped stage runs after everything that keys the cast (pose,
motion, perform, keys, sim, folds), so the hair steps with her; `mkmmd/core/stepped.py` (bpy-free) has the timing and
`mkmmd/blender/build/stepped.py` the keys.

| Key | Meaning |
|---|---|
| `from`, `to` | clip seconds (required): the window, `from` .. `to` - 1 in frames, inside the clip |
| `cast` | a member's name or a list of them (default: every member). A member is in one window at a time |
| `fps` | poses a second (default 12: on twos at 24), at most the clip's. A pose is drawn on frame `from + round(k x clip fps / fps)` |

Every animated datablock under the member's root (the root, the armature and its bones, the hands' goals, the morphs'
placeholder, the meshes' shape keys) holds its value at each step from half a frame before it to half a frame before the next,
keyed CONSTANT so no frame's shutter spans a step (motion blur leaves a stepped figure sharp), and takes up its own curve
again half a frame before the window ends; keys outside the window keep their values, interpolation and handles. A VMD on an
NLA strip ([Motion](#motion)) is held through the strip's own clock (Animated Strip Time, keyed to play as it did). A
garment's cloth copy that Blender simulates does not step, and a curve with modifiers is left alone and counted. The report
has per member `windows` (Blender frames), `poses` and `curves`.

```toml
[[stepped]]           # the cartoon flash on the snare: she moves on twos, the camera glides
from = 20.39
to = 21.15
cast = "reisen"
```

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
line per placement (`place desk on room:floor: (0.400, 1.280, 0.000) yaw 0.0 deg, 20 mm from the edge, nearest room:door 1615 mm`).
When a rule cannot be met the build fails with the numbers and the prop in the way: `prop 'lamp': no place on 'desk:top' for its
0.160 x 0.512 m footprint at u = 0.500, v = 0.024: overlaps 'room:window' by 60 mm (clear 20 mm)`. Cards read by the rules:
`bounds`, `layers`, `front`, `flat`, `blocks`, `origin`, `use.rest` planes with `size` or `radius`, `use.surface` panels, and on
set cards `obstacles` ([{name, min, max}] boxes in the set's frame).

A worked example (the `bedroom_80s` set and its props; it builds in about 8 seconds): a desk on the floor against the window
wall, the chair tucked under its edge facing it, the lamp and the player pushed against the desk's ends (the window's obstacle box
keeps them 70 mm off the wall, which is why they are not pushed into the back corners), tapes scattered on its free front, a poster on a
wall, the bed in a corner.

```toml
[[set]]
name = "room"
kind = "bedroom_80s"
open = ["front"]                    # no front wall: the camera stands where it was

[[prop]]
name = "desk"                       # on the floor against the window wall, centred under the window
card = "library:bedroom_desk"
place = { on = "room:floor", at = [0.4, 0.0], align = "edge:back" }

[[prop]]
name = "chair"                      # tucked 20 cm under the desk's front edge, facing it
card = "library:desk_chair"
place = { on = "room:floor", at = "near:desk", distance = -0.2, bearing = 0, facing = "desk" }

[[prop]]
name = "lamp"                       # on the desk, against its right end, aimed along the desk
card = "library:desk_lamp"
place = { on = "desk:top", at = [0.0, 0.08], align = "edge:right", facing = -90 }

[[prop]]
name = "player"                     # against the left end, as far back as the window leaves room
card = "library:cassette_player"
place = { on = "desk:top", at = [0.0, 0.08], align = "edge:left" }

[[scatter]]                         # three tapes at random spots of the free front of the desk
name = "tape"
props = ["library:cassette_tape"]
on = "desk:top"
count = 3
region = [[-0.3, -0.25], [0.3, 0.0]]
min_dist = 0.03
seed = 4

[[prop]]
name = "poster"
card = "library:poster_80s"
place = { on = "room:wall_left", at = [0.0, 0.25] }

[[prop]]
name = "bed"                        # head end into the left back corner
card = "library:bed_single"
place = { on = "room:floor", align = ["edge:left", "edge:back"] }
```

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

A grip is how a character's hand holds a prop: the 15 finger rotations and the frame the hand takes on the prop, solved on the
model's own skin (`mkmmd/solvers/grip.py`, numpy + scipy, no Blender) until the pads touch the prop's surface, nothing of the
hand is inside it and no finger is inside another. The hand is exported from the scene by the Blender op `hand_model` (rest
bones, rest skin near the wrist with its weights, and for a missing `*_tip` bone a virtual tip). `mk grip STYLE ...` runs both
and prints or writes JSON; the build's pose stage calls the same solver through `python -m mkmmd.solvers.grip IN.npz OUT.npz`
(cached in `<project>/.mk/cache/grip/` by a hash of the hand, the spec and the solver source).

| Style | Prop (`use.grip` point of the card, metres) | Grip frame (`target_in_wrist` is this frame) | Options |
|---|---|---|---|
| `pen` | `type = "pen"`: `length`, `radius` (a number or `[[distance from nib, radius], ...]`), `tip`, `nib_offset` | the prop's own: origin at the nib (minus `nib_offset`), +Z nib to cap, +X the barrel side facing the back of the hand | `--length`, `--radius`, `--tip`, `--nib-offset X Y Z`; lateral tripod, and `--posture` (nib, shoulder, pole, table, ...) adds the writing orientation |
| `wheel` | `type = "ring"` (`"wheel"` too): `center`, `axis` (prop frame), `radius`, `tube` | origin on the tube's centreline where the palm sits, x radially outward, y tangent (counter-clockwise seen from +z), z the ring axis | `--radius`, `--tube`, `--approach DEG` (palm side in the section plane, 0 = +x, 90 = +z), `--wrap` (+1 or -1) |
| `pinch` | `type = "pinch"`: `width` (thickness between the pads), `span` (depth, becomes `depth`), `length`, `center`, `axis` (along the strap), `normal` (outward) | midway between the pads, z from the index pad to the thumb pad, x away from the wrist, y = z cross x | `--width`, `--edge M` (the pads' distance inside the edge, default 0.004) |
| `rest` | a `use.rest` point: an `edge` with `a`, `b`, `normal`, or a plane with `center` (the solver takes `surface = "plane"`) | origin on the plane below the palm centre, z the normal, x the hand's heading | `--surface`, `--face palm` or `back` |
| `neck` | `type = "neck"`: `frame`, `frets`, `strings`, `section`, `fret_height` (see [Playing a worn guitar](#playing-a-worn-guitar)) | on the board's centre line under the position fret's wire, x toward the nut, z out of the board, y = z cross x | `--card` (a prop card or its neck grip point), `--chord` (a name or a table), `--fret`; or `--prop` with the solver's own neck problem |

A guitar's `strum` point (a pick pinched between thumb and index) has no style of its own: the build solves it as a `pinch` of
the pick, with the grip frame x into the pick's face and z along the strings.

`mk grip STYLE [SCENE.blend]` takes these options:

| Option | Meaning |
|---|---|
| `--cast NAME`, `--armature NAME` | whose hand: a cast member of the project (default its only one) or an armature object (default the scene's only MMD armature) |
| `--side {L,R,l,r}` | which hand (default R) |
| `--skin-radius M` | skin vertices within this of the wrist head are solved against (default 0.16) |
| `--frame FRAME` | the Blender frame the armature's world matrix is read at |
| `--prop JSON\|@FILE` | the prop's keys as JSON or a file; the style's flags above override them. A flag of another style is a usage error |
| `--params JSON\|@FILE` | solver keyword arguments |
| `--seeds N`, `--workers N` | parallel starts, the best wins; processes for them (default all cores, 1 = this process) |
| `--max-gap MM`, `--max-penetration MM`, `--max-clash MM` | the gates: contact gaps (default 3), penetration (1), finger inside finger (1) |
| `--out FILE.json` | write the whole result there (else it is printed) |
| `--no-cache` | solve again even when the same inputs were solved before |
| `--project DIR` | the project folder (default the nearest `mk.toml` above the working directory) |

Exit codes: 0 ok, 1 a gate missed (the file is still written, `problems` lists them), 2 usage error, 3 Blender or runtime error.
`pen` was tuned on one hand shape; another hand can miss the gates (exit 1).

`--params` also reaches the solvers' tuning arguments, which a build never sets: `pen` `slots` (`{index, thumb, middle: [clock
degrees, from m, to m]}`, the contact slots on the barrel), `orient_starts` (32) and `nfev` (iteration limits); `wheel` `prior`
(`{joint: degrees}`), `thumb_section` and `thumb_tangent` (`[min, max]`: degrees behind the palm, millimetres along the tube
toward the index) and `nfev` (reach, polish); `pinch` `heading` (degrees, 25: how far the wrist-to-pinch direction may stray
from the frame's x), `clear` (metres, 0.010: how far the middle, ring and little fingers stay from the object) and `nfev`;
`rest` `angles` (`{joint: degrees}` replacing the relaxed priors), `pitch` and `roll` (`[degrees, tolerance]`), `gap_mm` (0.3)
and `nfev`; `neck` `nfev`. The joints are `<finger>_mcp`, `_pip`, `_dip` and `_spr` for `index`, `middle`, `ring` and `little`,
and `t_palmar`, `t_radial`, `t_roll`, `t_mcp`, `t_ip` for the thumb. An unknown name is a usage error.

**Result.** `{"style", "side", "bones": {blender_bone: [w, x, y, z]}, "target_in_wrist": 4x4, "report", "solver"}`. `bones` are
`pose_bone.rotation_quaternion` values (bone-local, relative to rest): the 15 finger joint bones, identity where the grip
leaves one alone; the wrist and the tip bones are not driven. With the wrist posed, `frame_world = wrist_bone_world @
target_in_wrist`, so the wrist goes where `grip_frame_world @ inv(target_in_wrist)` says. `report` has
`contacts.<name>.gap_mm` (0 = touching), `penetration_mm`, `finger_clash_mm`, `angles_deg` and style numbers; a pen with a
posture adds `frame_world_quat` (the grip frame's world rotation while writing) and `report.writing`. The command prints a
summary (`ok`, `style`, `side`, `armature`, `scene`, `seconds`, `cached`, `bones` as a count, `target_in_wrist`, `report`,
`frame_world_quat`, `problems` when a gate is missed) and either `out` (the file written) or `result` (the whole result). The
solver's `python -m` interface takes the hand's arrays and `spec` (JSON `{style, prop, params}`) and writes `bones` (names),
`quats`, `target_in_wrist` and `report` (JSON of the rest of the result) to the output `.npz`; exit code 0 ok, 2 bad input.

**In the build** `grip` in `[pose.<cast>.hands.<L|R>]` names a `use.grip` point of a prop (`"car:wheel"`; `"prop"` alone when it
has one) or `"rest"`; the keys of the table, the grip's included, are listed in [Posing](#hands). The point's `type` picks the
style and the solver's finger rotations replace `fingers`. The build warns past the same gates as the command (a contact gap
over 3 mm, penetration or finger clash over 1 mm) and reports each hand's digest (Posing: What the stage reports).

- **Ring** (`type = "ring"`): a power grip round the rim. The grip frame sits at `clock` on the ring, hours as the character
  sees the wheel (12 top, 3 its right; default 10 for `L`, 2 for `R`), read at the first frame; the ring's axis is flipped to
  point at the character when the card gives it the other way. Without `approach` the grip is chosen for the arm: the palm on
  the rim's outer or driver side (0, 30, 60, 90) with either `wrap` is solved, and the one whose wrist bends least on the forearm
  the arm can make wins; without a `pole` the elbow is chosen with it (out from the shoulder and at least 5 cm below it, the
  pole put where that elbow points; `mkmmd.core.armreach`), with one the elbow is the pole's. With a chosen elbow the solver
  is given the arm too (shoulder, upper arm and forearm, up, outward) and keeps the wrist within 15 degrees of the forearm the
  best allowed elbow gives where the contacts allow: a grip found for the rim alone tilted a short-fingered, long-palmed hand
  and bent its wrist 44 degrees. The skin is searched on one vertex per 2 mm. The digest gives `approach`, `wrap`,
  `wrist_bend_deg`, `elbow` (`chosen` or `pole`), `reach` (arm lengths to the wrist) and every grip `tried`; a bend over 35
  degrees, or a wheel out of reach, is a WARNING. `approach` and `wrap` given turn the grip by hand (`wrap` default -1: the
  fingers go round the outside of the rim, then its front). A hand that follows a turning wheel also says
  `ride = "car_wheel"`, the wheel's object.
- **Pinch** (`type = "pinch"`): a thumb-index pad pinch of a strap or handle; the card needs `axis` and `normal`. The pads sit
  `edge` inside its edge.
- **Rest** (`grip = "rest"` with `rest = "prop:edge"`, a `use.rest` point): the relaxed hand lies on the surface, the palm
  centre at the rest point (`along` on an edge, `offset` on a plane, `lift` default 0.0, `face` `palm` or `back`), `dir` its
  heading. A hand with only `rest` is posed by IK without solving the fingers.
- **Pen** (`type = "pen"`): a lateral tripod. The pen's own frame is the grip frame, so the wrist goes where the prop's object
  stands. It needs a `posture` (a table with at least `nib`) or a `track`.
- **A pen whose nib follows a path** (`grip = "pen:barrel"` with a `track`): the writing hand of a character who writes while
  the camera watches. The track is project reference data (`tracks/nib.json` for `track = "nib"`, or any `.json` path):
  `{"frames": [Blender frames], "target": [[x, y, z] world metres per frame]}`, positions interpolated linearly, held before
  the first and after the last frame; `channel` names another channel of positions. The stage solves ONE grip for the pen on
  the hand, with its writing orientation (the whole hand's rotation about the nib, solved against the arm and the desk), then
  keys the arm IK on every frame with the wrist goal `pen_frame @ inv(target_in_wrist)`, the pen frame's origin on the track
  and its rotation the solved one (LINEAR keys), and bone-parents the pen prop to the wrist in the solved grip: the pen
  object's origin, the nib, lands on the track within the IK's accuracy (a `contact` check of `obj("pen").loc` against
  `track:nib.target` pins it). `wobble` tilts the pen slowly about the world X and Y, as a hand is never perfectly still. The
  solve takes minutes on a pen (cached by a hash of the hand, the track-derived posture and the solver source), so keep the
  pose (`lean`, `head`, `pole`) steady while you tune the rest. The grip digest carries the `writing` numbers and the track's
  extent; the stage reports the worst IK miss over the track (`ik_error_mm`) and logs a WARNING past 5 mm.
- **Neck** and **strum**: [Playing a worn guitar](#playing-a-worn-guitar).

The writing `posture` completes itself from the scene (`mkmmd/core/pentrack.py`): your keys win over what the stage fills in.

| Key | Meaning |
|---|---|
| `nib` | `[x, y, z]` where the nib writes (a list of points is averaged). With a track: its mean x, y at `paper_z` |
| `paper_z` | metres, the height of the paper (default the nib's height: with a track, its lowest z) |
| `shoulder` | the arm's shoulder joint (default the shoulder in the seated pose at the end of the settle; with `mk grip` the model's arm bone) |
| `facing` | the character's horizontal heading, so that "outward" is known (default the character's own; `[0, -1, 0]` with `mk grip`) |
| `pole` | the elbow pole target (default the IK's own, the hand's `pole`; with `mk grip` outward, back and down of the shoulder) |
| `upper`, `fore` | the arm's segment lengths, metres (default the model's) |
| `table` | `"table:top"` (a prop's `use.rest` plane, in a build) or `{z, center, radius}`: the forearm stays `forearm_lift` above it wherever it is over the table (all of it when `center` and `radius` are omitted) |
| `forearm_lift`, `wrist_lift` | metres the forearm clears the table (default 0.028) and the wrist joint clears the paper (0.032) |
| `reach` | the share of the arm's length the wrist may reach (default 0.97) |
| `target` | `{elevation, azimuth, tilt, extension, ulnar: [degrees, tolerance]}`: the writing posture wanted, the pen's angle above the paper, where it points against the shoulder, the back of the hand rolled outward, the wrist's extension and ulnar deviation (default 52 ± 7, 12 ± 15, 35 ± 6, 15 ± 12, 10 ± 10) |
| `up` | `[x, y, z]` the vertical (default +Z) |

Frames: `mkmmd/core/gripframe.py` builds the grip frame in the world (`ring_frame`, `surface_frame`, `pinch_frame`,
`strum_frame`) and the wrist goal (`wrist_goal`); it is numpy only, so the maths is tested without Blender. Check a grip with a
`contact` check between a fingertip bone's tail and the prop (see [AGENTS.md](AGENTS.md)).

## Playing a worn guitar

A standing character can wear a prop on a bone (a guitar on its strap), fret its neck and strum its strings in time with the
song. The mechanism is generic: the prop's card says where it is worn and where the hands go (`use.wear`, `use.grip` points of
type `neck` and `strum`, `use.anchor`), the project gives the numbers (a chord and fret, the times, the rhythm). The library prop
`electric_guitar` (frame and card: its module docstring) is a card that does.

```toml
[[prop]]
name = "guitar"
card = "library:electric_guitar"
wear = "rin"                         # or {cast = "rin", neck_deg = 30, yaw_deg = 5}: the card's numbers, overridden

[pose.rin]                           # standing: no `sit`; the elbows hang under the shoulders by default
[pose.rin.hands.L]
grip = "guitar:neck"                 # fretting hand: thumb behind the neck, pressing fingers arched on their strings
fret = 3                             # the position: the fret under the index finger (1 or more)
chord = "power"                      # a name below, or a table {index = [6, 0], ring = [5, 2]}: finger -> [string, frets above `fret`]
keys = [{ t = 1.92, fret = 5 }, { t = 3.86, fret = 1, chord = "E" }]   # the shape in place at t, reached over `move` s (0.12) before it
[pose.rin.hands.R]
grip = "guitar:strum"                # the pick pinched between thumb and index, its tip on the strings; `thumb`, `tip`

[perform.rin]
strum = { hand = "R", prop = "guitar", rhythm = "onsets:other", from = 0.96, to = 15.0, accent = "downbeats" }
```

**Wearing** (`mkmmd/core/wear.py`, `mkmmd/blender/build/wear.py`, run by the pose stage before the arms). `[[prop]] wear` is the
cast member's name (the card's only `use.wear` point) or a table; the prop's own `at` and `yaw` are then replaced, its root is
bone-parented where the point says. The table's keys:

| Key | Meaning |
|---|---|
| `cast` | who wears it (required in a table) |
| `use` | the name of the card's `use.wear` point (default its only one) |
| `at`, `pivot`, `scale`, `neck_deg`, `yaw_deg`, `roll_deg` | the point's numbers of the same names, overridden. Any other key raises (`wear: unknown key`) |
| `strap`, `cable` | tables laid over the card point's own table of that name, key by key (the keys are the card point's, below); the card's point must have the table, and a key it does not have raises (`wear cable: unknown key`) |

The card's `use.wear` point (the library guitar's is called `stand`) has these keys:

| Key | Meaning |
|---|---|
| `name`, `bone` | the point's name and the semantic bone the prop rides on (the chest, `upper_body2`) |
| `pivot` | a point of the prop's own frame (metres) that is put at `at`: the back of a guitar at its saddle line |
| `ref` | the reference body the numbers were made for, `{measure name: value}` of rig.json `measure`; the wearer's rig.json needs a `measure` too (the stage raises without) |
| `at` | `[x, y, z]` metres from the bone's head in the character's axes (x its left, y behind it, z up), for the `ref` body |
| `scale` | one measure name per coordinate of `at`: each coordinate is multiplied by the wearer's `measure` of that name over the `ref`'s |
| `neck_deg` | the prop's +z axis (a guitar's neck) is brought to the character's left and raised that many degrees above horizontal |
| `yaw_deg`, `roll_deg` | the neck swung toward the character's front; the prop rolled about the neck, positive turning its -y face up (default 0) |
| `strap` | `{top, bottom, over, width = 0.05, thickness = 0.004, material, shoulder_radius}`: builds the band `<prop>_strap` |
| `cable` | `{object, anchor = "jack", radius = 0.0032, trail, out = 0.05, sway = 0.035, reach = 0.12, tail_len = 0.9, follow = 0.35, sim = true}`: re-hangs the cord; `sim = false` keeps it hung and hooked instead of swinging it |

The prop's frame needs the guitar's convention: z along the neck, the face looking along -y, x across it. `strap` runs from the
card's anchor `top` over the wearer's shoulder joint (`over`, a semantic bone: default `arm.L`, `shoulder.L` reads as `arm.L`;
the band's arc over the shoulder has radius `shoulder_radius`, default 0.22 of the wearer's `upper_arm`) round the back to the
anchor `bottom`, riding the torso at the radius of the chest's collision body (rig.json; else 0.55 of the shoulder width) plus a
7 mm standoff (`core.wear.strap_path`); a `material` that does not exist logs a WARNING and leaves the strap without one.

`cable` rebuilds the curve `<prop>_cable` (a bevelled NURBS curve the prop's builder made) from the `anchor`'s jack
(`core.cable.hang`): `out` metres straight out of the plug, a lazy S of `sway` metres, down to the floor under the wearer
(the cast root's height), `reach` metres from under the jack along `trail`, then `tail_len` metres of cord lying on the floor
along `trail`: a horizontal direction `[dx, dy]`, default behind the wearer and to the wearer's right. The curve is unparented, so its
lower end stays, and hooked at the jack with a smooth falloff of `follow` metres, so its top follows the guitar. The sim stage
then swings it from that shape ([Sim](#sim)): the whole cord answers the guitar's motion, falls against the wearer's legs and
lies on the floor it was hung on (a wearer whose root is keyed off it, falling or jumping, leaves the floor where it is), keyed
on every frame; with `sim = false` (or `mk build --skip sim`) the hooked cord stays.

The `wear` report has `bone`, `entry`, `at_mm` and the strap's numbers (`object`, `points`, `length_m`, `radius_m`), `cables` the
cord's (`points`, `top`, `floor_z`, `length_m`, `follow_m`). The numbers decide whether both arms reach: the build logs a WARNING
for a wrist that ends more than 5 mm short of a goal, at the settle and at every keyed chord.

**Hands.** The neck point feeds the `neck` solver style (`mkmmd/solvers/grip_neck.py`). Its keys:

| Key | Meaning |
|---|---|
| `frame` | `{along, across, normal}`: toward the nut, across the strings from the low E to the high e, out of the board |
| `frets` | the board-surface point under every fret wire, the nut first (index 0); the library guitar has 22 |
| `strings` | low E first: `{nut, bridge, radius = 0.0006}`, the points on the string's axis at the nut and the saddle |
| `section` | the neck solid under the board: `{width: [nut, last fret], depth: [nut, last fret], p = 2.6}` with the superellipse exponent `p` of its back |
| `fret_height` | metres a wire stands out of the board (default 0.0012) |

The solver works in the neck frame N on the board under the position wire (x toward the nut, z out of the board, y = z cross x):
each pressing finger's pad on its string a share (`press`, 0.3) of the way behind its wire, within a few millimetres across it,
the distal phalanx steep onto the board, the thumb pad on the neck's back near the fingers, no skin in the neck, the other
fingers hovering over the strings, no finger in another (`mkmmd/core/fretting.py`: chord tables, targets). `fret` is the
position: the fret under the index finger in the shape's own numbering, a power chord at `fret = 5` has its index on fret 5 and
its ring and little fingers on fret 7; the open chords are written for position 1, so open E at `fret = 1` presses frets 1, 2
and 2. A shape is a name below or a table `{finger = [string, frets above the position]}` of `index`, `middle`, `ring`,
`little` (string 6 = low E ... 1 = high e; two fingers on one string, a fret past the neck or `fret` below 1 raise). Strings no
finger presses are left alone, fingers the shape does not use hover over the board.

| Chord | Fingers as `finger: string, frets above the position` |
|---|---|
| `power` | `index: 6, 0`, `ring: 5, 2`, `little: 4, 2` (root on the low E) |
| `power5` | `index: 5, 0`, `ring: 4, 2`, `little: 3, 2` (root on the A string) |
| `E` | `index: 3, 0`, `middle: 5, 1`, `ring: 4, 1` |
| `A` | `index: 4, 1`, `middle: 3, 1`, `ring: 2, 1` |
| `D` | `index: 3, 1`, `middle: 1, 1`, `ring: 2, 2` |
| `G` | `middle: 6, 2`, `index: 5, 1`, `ring: 1, 2` |
| `C` | `ring: 5, 2`, `middle: 4, 1`, `index: 2, 0` |
| `Em` | `middle: 5, 1`, `ring: 4, 1` |
| `Am` | `middle: 4, 1`, `ring: 3, 1`, `index: 2, 0` |

`fret` and `chord` come with the hand table; `keys = [{t, fret, chord, move}]` change the shape over time: a key says the shape
in place at clip time `t`, reached over `move` seconds (default 0.12, or the hand's own `move`) before it, and a key without
`fret` or `chord` keeps the previous one. Without `fret` and `chord` in the hand table the first key is the start and needs
both. Every distinct (fret, chord) is solved once and cached (a few seconds each); a keyed change moves the wrist goal and
every finger bone from one solved shape to the next.

The strum point (`center`, `along`, `across`, `normal`, `pick = {object, thickness, length, width, tip}`) is solved as a
`pinch` of the pick: the grip frame has x into the face and z along the strings (`core.gripframe.strum_frame`; `thumb = "neck"`
toward the nut, `"bridge"` away), its origin `tip` above `center` (the hand's `tip`, else the card's `pick.tip`, else 0.008);
the pick object, modelled in that frame, is bone-parented to the wrist. Hands on a `neck` or `strum` point ride the prop
(`ride` defaults to its root). `mk grip neck --side L --card @card.json --chord power --fret 3` solves one without a build.

**Strumming** (`[perform.<cast>] strum`; the keys and the stroke model are under [Perform](#perform), `strum`, and in
`mkmmd/core/strum.py`). The strokes come from the song: `rhythm = "onsets:other"` plays the plucks the timeline's detector
found in that stem (`mk timeline onsets`, see [Timeline](#timeline)), `"beats:8"` a grid of eighths from the beat list, or a
list of times; `from` and `to` pick the window, `accent = "downbeats"` accents the first beat of each bar, and several tables
on one hand give different rhythms in different parts of the song. The build keys the pick hand's goal on every frame of the
window, relative to the grip, so the arm's reach is part of the result. `plan` and `path` are numpy only and tested without
Blender; the `strum` check replays the plan on the timeline against the scene.

Checks to keep: `joint_limits` (the default elbow pole is under the shoulder; give `pole` per hand for another), `contact`
between `bone("wrist.R").tail` and `obj("<cast>_hand.R").loc` (the IK reached its goal at every frame, `max` 5 mm), `strum`
(distance at the down strokes, timing), `prop_body` (the guitar against the character's body), `form` on the prop.

## Shots

`[[shot]]` entries make the cut. Each shot gets one camera per output, named `<shot>@<output>` (`<shot>@main` when the project has
no `[[output]]`), keyed linearly on every frame (position, rotation, lens, focus distance, and the lens shift when `keys` move it):
no constraints, no drivers. One timeline marker per shot, on its `from` frame, switches the cameras; `mk look` and `mk render` point
the markers at the cameras of the output they draw and switch the shot's look (below) frame by frame. `mk build` keeps the shot table
in `scene["mk_shots"]` (JSON, per shot `name`, `from` / `to` in Blender frames, `cameras` {output: camera object}, `keyed` (the frames
the cameras are keyed over), `plate`, and per output the normalised `styles` with their colours resolved). Times are clip seconds; a
shot starts on the frame `round(frame0 + from * fps)`.

| Key | Meaning |
|---|---|
| `name` | unique; the cameras are `<name>@<output>`. `[[transition]]` and `[[insert]]` windows, `lyrics.zones` and `mount = "<name>@<output>"` (a plane that rides a camera) refer to it |
| `from`, `to` | clip seconds, required unless `plate = true`. The markers use `from` only: set `to` to the next shot's `from` (it bounds the frames the camera is keyed over, ±2, and is what `[[transition]]`, `[[insert]]` and `lyrics.zones` read) |
| `plate` | `true`: the shot is not in the cut (no marker; `from` / `to` optional). It exists for a `[[transition]]` or `[[insert]]` to take frames from, drawn through its own camera and look at the frames wanted; a plate that no effect uses is skipped with a WARNING. A shot in the cut that an effect takes frames from before its `from` is keyed over those frames as well |
| `mount` | prop, set or object the camera rides: `at` is in its frame and its motion carries the camera. An unknown name is an error (default: the world) |
| `at` | camera position: `[x, y, z]` in the mount's frame (the world without a mount) or a dict target, `{path = "road:road", s = 640, offset = 7, z = 1.2}` (beside a set's path), `{prop = "car", point = [x, y, z]}`, `{cast = "rin", point = [x, y, z]}`, resolved in the world on every frame (the mount does not apply to it); `keys[].at` takes either form too, and a camera between two keys moves between the world points they give on every frame (default `[0, 0, 0]`) |
| `look` | what the camera aims at: any [target](#targets), `[x, y, z]` (world), `"cast:rin.head"`, `"cast:rin"` (the eyes), `"car:road"` (a card's use point), `{path = ...}`; read on every frame, so a moving target is followed (default: straight ahead, along the mount's -Y) |
| `lens` | focal length in mm on a 36 mm sensor that spans the larger image side; every camera clips from 0.02 m to 5000 m (default 35) |
| `roll` | degrees about the viewing axis; positive turns the picture clockwise; keyed in `keys[].roll` (default 0) |
| `lag` | operator lag in seconds: the aim follows its target through a critically damped filter of that time constant, applied in the mount's frame, so a camera in a car lags the subject, not the road (default 0) |
| `follow` | `{target, axes = "x", lag = 0, from}`: the side-scroller lock. On every frame the camera's fixed position and aim point (`at` and `look` given as `[x, y, z]`, the keys' too) move by the world displacement of `target` (any [target](#targets)) since the frame of clip second `from` (default the shot's own `from`), along the world axes in `axes` only (distinct letters from `"xyz"`: `"x"` runs with her and leaves her hops to rise in the frame, `"xyz"` holds on her), eased through the critically damped filter of `lag` seconds that the `lag` key aims through. A target given as `at` or `look` already moves and is left alone; the move is added after the aim's own lag, so the camera never turns by it. It reads the target where the stages before shots put it (`walk`, `rise`, `tumble`, vehicles, motions), not a `[[key]]` or the sim. A target that names nothing, axes that are not letters from `"xyz"` and a negative lag are errors, and so is `follow` with `mount`. The report's `moves.<output>.follow_m` is the farthest it went |
| `shake` | handheld shake in degrees: smooth random turns of the camera about its three axes (noise low-passed with a 6-frame time constant, seeded by the camera's name so a rebuild is identical); about `shake / 2` rms in pitch and yaw and a quarter of `shake` in roll (default 0) |
| `keys` | `[{t, at, look, lens, roll, shift}, ...]`, a move inside the shot; `t` is in clip seconds (not relative to the shot). Between two keys `at`, `lens`, `roll` and `shift` move with a smoothstep ease and `look` switches to the next key's target halfway, so aim at something that moves rather than between two fixed points. A field a key leaves out is the shot's own; the first and last key hold outside their span |
| `frame` | `{subject, fill, solve}`: solves the lens, or the distance, for each output so the subject fills a share of the frame height. `subject` is a target or a list of targets (one world point is written `[[x, y, z]]`); its height is the vertical extent of those points (at least 0.25 m) plus 0.15 m of headroom, taken at the median of about twelve frames of the shot. `fill` is the share of the frame height (default 0.45). `solve = "lens"` (default) sets one lens for the whole shot (a keyed `lens` is replaced), `"distance"` keeps the lens and moves the camera along the aim line by one factor, `"dolly"` keeps the keyed lens and moves the camera along the aim line on every frame so the subject keeps `fill` (a dolly zoom: a lens keyed from 28 to 75 mm sends the camera 2.7 times as far, she keeps her size and the room behind her swells; exact for a subject on the aim point; the report's `moves.<output>.dolly_m` is the range of distances); any other solve is an error |
| `shift` | `[x, y]`, Blender's lens shift: fractions of the larger image side, the picture moving the other way (a positive `y` moves it down; probed in Blender 4.2.3); constant, or keyed in `keys[].shift` (default `[0, 0]`). See the crop below |
| `dof` | `{focus, fstop, offset}`: depth of field; `focus` is any target (required), keyed as the distance from the camera on every frame, less `offset` metres (default 0): the focus plane sits that much nearer the camera than the target. A character's eye bones (`"cast:rin"`) are inside the head, 4-8 cm behind the face's surface (more for `.head`, the base of the skull), while a close-up at f/2.8 keeps only a few centimetres sharp: give a face target `offset = 0.05`. `fstop` (default 2.8). Without `dof` nothing blurs |
| `aspect.<output>` | `{...}`, `[shot.aspect.<output>]`: that output's own version of the shot. It may hold any key above except `name`, `from`, `to`, `plate` and `aspect`. The tables `frame`, `dof`, `follow`, `colors`, `tones`, `knockout` and `reflection` merge key by key with the shot's own (`frame = { fill = 0.6 }` keeps the `subject`, `follow = { lag = 0.4 }` the `target`); every other key, targets and lists included, is replaced whole; `reflection = false` and `follow = false` switch an inherited reflection or follow off for that output. A table for an output the project does not have is an error |
| `style`, `colors`, `hide`, `keep`, `accent`, `tint`, `knockout`, `grow`, `samples` | the silhouette look, below |
| `style`, `colors`, `tones`, `hide`, `keep` | the vector look, below (with the project's `[vector]` table) |
| `reflection` | the window reflection, below |

```toml
[[shot]]
name = "hood"
from = 0.0
to = 4.0
mount = "car"                          # `at` is in the car's frame; the car's motion carries the camera
at = [0.0, -1.6, 1.3]
look = "cast:rin.head"                 # any target, followed on every frame
lens = 35
lag = 0.25                             # the aim trails its target by a quarter of a second, in the car's frame
shake = 0.3
dof = { focus = "cast:rin", fstop = 2.8, offset = 0.05 }    # her eyes are bones inside the head: focus on her face
keys = [{ t = 0.0 }, { t = 4.0, at = [0.0, -1.1, 1.25], lens = 50 }]     # an eased push-in; the first key is the shot's own

[shot.aspect.9x16]                     # this output frames the figure instead: the lens is solved per output
frame = { subject = [{ cast = "rin", point = [0, 0, 0] }, "cast:rin.head"], fill = 0.6 }
```

`shift` is how a 1:1 output becomes an exact crop of a 9:16 master: the square camera keeps the master's position, aim, roll and
keys and changes three numbers (`crop_camera` in `mkmmd/core/shotstyle.py`). The lens is multiplied by `k = max(master) / max(crop)`
so a pixel keeps its angular size (1920 / 1080 for a 1080 x 1920 master and a 1080 x 1080 crop), the `dof` f-stop by the same `k`
so the blur keeps its size in pixels, and `shift = [0, (420 - top) / 1080]` puts the master's axis `960 - top` px below the crop's
top row for a crop whose top row lies `top` px below the master's (`top = 420` is centred: no shift). Pitching the camera to re-aim
at the crop would change the perspective instead (a keystone that grows toward the corners): the shifted camera is the crop. `roll`
turns the picture around the shifted axis, so keep rolled shots centred.

```toml
[[shot]]
name = "close"
from = 8.0
to = 12.0
at = [0.0, -2.0, 1.5]
look = "cast:rin.head"
lens = 50
dof = { focus = "cast:rin", fstop = 2.0, offset = 0.05 }

[shot.aspect.1x1]                      # a square crop of the 9x16 master whose top row is 300 px below the master's top
lens = 88.89                           # 50 x 1920 / 1080
dof = { focus = "cast:rin", fstop = 3.56, offset = 0.05 }    # 2.0 x 1920 / 1080
shift = [0.0, 0.1111]                  # (420 - 300) / 1080
```

`mk build` refuses a `mount` that names nothing; a shot without `from` and `to` that is not a plate, and a plate with only one of them;
a `shift` that is not `[x, y]`; a key that `[[shot]]`, `frame`, `dof`, `keys[]` or an `aspect` table does not have (the message lists
the known ones: `shot 'storm': unknown key 'lenss' (known: ...)`); an `aspect` table for an output the project does not have; a `frame`
without `subject`; a window of an effect that needs frames from a shot before `[scene] start` (`lower [scene] start`); and bad look
tables (the message names the shot and the output: `shot 'storm' (16x9): ...`). It logs a WARNING for a plate no effect uses and for
an object pattern that matches nothing.

### Looks: silhouette, vector and reflection

Three looks need more than the lit scene. They belong to the shot and are switched on and off frame by frame by `mk render` and
`mk look` (`mkmmd/blender/styles.py`; the normalised specs per output are in the shot table), put back everything they change, and
leave `mk post` to grade the finished frames as usual. `mk render --no-styles` and `mk look --no-styles` draw every shot as lit. A
shot has one look per output: `style` and `reflection` together are an error. A silhouette shot can show a reflection in one output
if that output's `aspect.<output>` table says `style = "none"` and gives the `reflection`; a `reflection` set on the shot itself
can be switched off for one output with `reflection = false` in that output's `aspect` table.

Object patterns (`hide`, `keep`, `accent`, `only`, `knockout.objects`; a single string is a list of one): `name*` (fnmatch on the
object name, case-sensitive), `@collection` (the object is in that collection or below it; fnmatch on the collection name) and
`prop:key` (that custom property is set, truthy, on the object). The stages file objects in the collections `Cast`, `Props`, `Sets`
(a child collection per set, named after it), `Text`, `Cameras`, `Lights` and `Rig`. Only geometry objects (mesh, curve, surface,
text, meta, curves, point cloud) that render at that frame are matched. The build logs a WARNING for a `hide`, `keep`, `accent` or
`only` pattern that matches no object in the finished scene (checked just before it is saved, so the words the text stage makes
count).

| Key | Meaning |
|---|---|
| `style` | `"silhouette"`: the flat look, a flat background, the scene as one colour, accents in another. `"vector"`: the flat-vector look, below. `"none"` (in `[shot.aspect.<output>]`) switches an inherited look off; nothing else is accepted |
| `colors` | `{background, subject, accent}`, each a palette slot, `#hex`, `"a:b:t"` mix (`t` weighs the second) or `[r, g, b]` in 0..1 (defaults `base`, `text`, `surface`) |
| `hide`, `keep` | objects not rendered (walls, outside, rain), minus `keep` |
| `accent` | objects in the accent colour. One with a transparent material (a lightning bolt) keeps its softness; the others (earbud cords) are painted over everything and grown by `grow` so a thin wire reads |
| `grow` | pixels the hard accents are grown by, counted at 1080 wide (default 1, not negative) |
| `tint` | `[{object, prop, color, gain, glow}]` (or one table): the background moves toward `color` by `min(1, gain * value)`, `value` the custom property `prop` of the object named `object` on that frame (a missing object or property reads 0): a lightning flash, a sun. `object`, `prop` and `color` are required, `gain` defaults to 1. `glow = {at, size}` (defaults `[0.5, 0.5]`, `[1, 1]`) makes the move a gaussian bloom `exp(-(dx / sx)^2 - (dy / sy)^2)` about `at`, fractions of the frame from the top left |
| `knockout` | `{objects, color}`: type reversed over the silhouette: `color` (default `text`) on the background, the background colour where it overlaps the silhouette. These objects are drawn over everything; other type is hidden behind what stands in front of it. [Screen type](#screen-type) asks for it on the text itself (`knockout = true`), no table needed |
| `samples` | EEVEE samples of the passes below (default 16, at least 1) |

```toml
[[shot]]
name = "storm"
from = 0.0
to = 1.3
at = [0.0, -2.2, 1.4]
look = "cast:rin.head"
style = "silhouette"                   # the names below are your own objects, collections and custom properties
colors = { background = "base", subject = "text", accent = "gold" }
hide = ["wall*", "@Outside"]
keep = ["wall_front"]
accent = ["bolt*"]
tint = [{ object = "weather", prop = "flash", color = "text", gain = 1.0, glow = { at = [0.7, 0.2], size = [0.5, 0.6] } }]
knockout = { objects = ["title"], color = "text" }
```

Silhouette. Blender 4.2 ignores `view_layer.material_override` in EEVEE Next and Workbench (only Cycles honours it), so the frame is
composed inside the render call from passes that work (`compose_silhouette`, numpy):

- An object is hidden (`hide` minus `keep`), a knock-out object, type (a modifier whose node group starts with `mk_text_`), an
  accent, or subject (everything else). One whose `hide_render` is keyed is sorted as if shown, so it is drawn in its colour on
  the frames its keys show it, and one the look hides has those keys muted for the render, so they cannot show it.
- Workbench with flat light and `Object.color` draws the subject and the hard accents in their flat colours: exact to one level of 255
  (Workbench's colour transform), antialiased by Workbench.
- An accent whose material contains a Transparent shader is soft: EEVEE with an AOV on the Mix Shader factor of its materials and
  the compositor (R = alpha, B = AOV, Raw) gives its opaque share, painted in the accent colour; what its soft shells have beyond
  that (`alpha - AOV`) stays a glow in the subject colour. The subject and the hard accents hide it. The pass needs a scene with no
  compositor tree of its own.
- Type is rendered by EEVEE with the lights off and a black world, so its emission shows it, keyed opacity included, and drawn over the
  silhouette; the subject and the hard accents in front of it hide it. Knock-out objects are rendered the same way as a mask and drawn
  last.
- Silhouettes are always 8-bit.

Vector. `style = "vector"`: a flat-vector drawing of the scene. Every material is filled with a tone of the project's `[vector]`
table: one colour, two (lit and in shadow under one hard light), a few picked by the texture's brightness, so what a model draws
on itself (irises, lash lines, a mouth) carries over into the palette, or the model's own texture colours in two tones (cel).
Lines run round the figures, where two parts meet and where one part passes in front of another. The shot itself gives `style`,
and may give `colors`, `tones` and `lines` (laid over the project's, key by key: a shot inside an inverted palette swaps a few),
`materials` (its own rules, checked before the project's: `[{match = "*", tone = "cel"}]` draws everything cel), `sketch`
(below) and `hide` / `keep`; the silhouette's other keys are an error in a vector shot, and the vector style's own keys
(`tones`, `materials`, `lines`, `sketch`) in a shot of another look (an aspect's `style = "none"` drops them with the look).

| `[vector]` key | Meaning |
|---|---|
| `colors` | `{background, line, inner}`: the background, the outline round the figures and the inner lines (palette slot, `#hex`, `"a:b:t"` or `[r, g, b]`; defaults `base`, `text`, and `inner` the line's colour) |
| `tones` | `{name = tone}`. A tone is a colour (flat), `{lit, shade}` (the second where the look's light does not reach: the far side and cast shadows; no `shade` is flat), `{colors = [n colours], at = [n - 1 brightness steps], grey}`: drawn, each texel takes the colour of its brightness band (sRGB luma), and the last colour only texels greyer than `grey` (saturation, default 1: all), so a bright tinted iris stays in the band below the white highlights; or `{texture = true, shade}`: cel, the texture's own colour (a material without an image: its viewport colour) and that colour multiplied by `shade` where the light does not reach (no `shade`: the texture under flat light). Default `{fill = {lit = "text", shade = "subtle"}}` |
| `materials` | `[{match, tone, group}]`, in order: a material takes the first rule whose `match` (material name patterns, fnmatch, Blender's `.001` suffix ignored) fits it, else the tone `fill` (or the first tone). Lines are drawn where two `group`s meet; a material without one is its own group, so skin and the shirt get a line between them, and a face and its eyes that share a group do not. `mk q BLEND --list materials` names a scene's materials with the name a rule matches; the build logs a WARNING for a rule that names no material in the finished scene |
| `lines` | `{outline, inner}`: line widths in px at 1080 on the frame's short side (default 5 and 2) |
| `shadow` | the share of the light's full strength below which a two-tone surface takes its `shade` colour (default 0.3); cast shadows always do |
| `light` | `[x, y, z]` toward the light that casts the shadows, in the world (default `[0.45, -0.35, 0.82]`: above, in front, from the figure's left) |
| `opposite` | `{colors, tones}`: the palette inside a [ring](#rings), laid over the look's own (`colors` key by key, `inner` kept unless given; `tones` by name). A tone keeps its kind; a drawn tone gives only its `colors`, as many, and keeps the look's bands; a cel tone keeps its `shade` or its lack of one |

```toml
[vector]
colors = { background = "#FFD21F", line = "#1B2A6B" }

[vector.tones]
white = { lit = "#FFFFFF", shade = "#8FB4F0" }
dark = "#1B2A6B"
iris = { colors = ["#1B2A6B", "#8FB4F0", "#FFFFFF"], at = [0.55, 0.88], grey = 0.15 }

[[vector.materials]]
match = ["skin", "face"]                 # your model's own material names
tone = "white"
group = "face"

[[vector.materials]]
match = "eye*"
tone = "iris"
group = "face"

[[shot]]
name = "flip"
style = "vector"
colors = { background = "#1B2A6B", line = "#FFD21F" }   # the inverted palette for this shot
tones = { dark = "#FFD21F" }
```

A shot's (or a [medium window](#medium-shifts)'s) `sketch = true | {paper, ink, boil, fps, strokes, grain, hatch, seed}` draws
the vector look as line art instead of fills: paper everywhere and the outline and inner lines in ink, drawn `strokes` times by a
hand that wobbles up to `boil` px (at 1080; it bends once every 48 px) and draws the picture again `fps` times a second of the clip,
so the lines boil; with two strokes or more each pass is a little lighter than ink (a pencil), its pressure drifting along the
line and broken by the paper's `grain` (0..1). `hatch` (px at 1080 between strokes, 0: none) shades what the look's light does not
reach with diagonal strokes, crossed by the second pass. Defaults: paper `text`, ink `base`, boil 1.2, fps 12, strokes 2, grain
0.35, hatch 0, seed 1. Rings do not flip a sketch (it needs no `opposite`), and it reads no texture.

The frame is composed with numpy (`compose_vector`) from Workbench passes rendered at twice the frame's size and boxed down, which
antialiases the fills and the lines:

- `id`: flat light, every material's viewport colour set to its number (Raw, no antialiasing, a float EXR: exact; a material under
  half opaque is left out, as the other passes leave it out). The compositor writes the Z pass of the same render, so the scene must
  have no compositor tree of its own.
- `tex`: flat light and the textures (only when a drawn or cel tone is used, and not for a sketch).
- `shade`: the studio light, fixed in the world, with shadows cast from `light`, on plain white. A two-tone surface is in shadow under
  `shadow` times the light on a white surface facing it (0.40 in Raw); the shadow is opened and closed by one pass pixel, so no
  sliver or pinhole is left.
- Lines: the outline runs round everything; inner lines run where two groups meet and where the depth breaks (its second
  difference beyond 1 cm plus 0.4 % of the distance), so a surface turning away draws no line but a chin over the neck, a hand over
  the face or one hair strand over another does. Both are drawn as round pens, then the frame is boxed down.
- Type is rendered as for the silhouette and laid over the drawing.
- A vector shot's figure can be a transition's matte, as a silhouette's can.

Reflection. `reflection = {object = "<glass>"}` shows what stands in front of a window pane; screen-space tracing cannot see what is
behind the camera. EEVEE Next draws it through a plane light probe on the glass (a probe the set already has at the pane, else one is
made) and a mirror layer mixed in front of the glass shader: the shader becomes
`mix(mix(glass, black, dim), glossy(roughness, tint), strength * (1 - backfacing))`.

| Key | Meaning |
|---|---|
| `object` | the glass: a mesh object in the scene (required), flat to 10 mm, with faces, a node material and a connected Material Output; its first face's normal points to the side the glass is seen from |
| `strength` | how much of the mirror over the glass, 0..1 (default 0.5) |
| `dim` | how much the glass shader darkens where the mirror shows, as glass does against a bright street, 0..1 (default 0) |
| `roughness`, `tint` | the mirror's roughness, 0..1 (default 0), and colour (palette slot, `#hex`, `"a:b:t"` or `[r, g, b]`; default white) |
| `hide`, `only` | what the pane reflects: the probe skips the objects `hide` matches, and with `only` every object it does not match |
| `bend` | `true`: the mirror takes the glass material's own Normal Map node, so rain on the glass bends the reflection (default false) |
| `world` | `true`: the sky shows in the pane; by default the world is black to the probe, so a pane that reflects only a cast member lets the dimmed street through at `1 - strength` (default false) |
| `probe` | `{pad, influence, clip}` of a probe the render makes: its size as a factor of the glass (default 1.02), its influence distance (0.1 m) and clip start (0.001 m); it stands 4 mm in front of the glass |

One mirror weighs everything the pane reflects the same: leave out what should not show (a room) with `hide` or `only`. The glass
object has to exist when the shots stage runs (`mk q BLEND --list objects` names them).

```toml
[[shot]]
name = "window"
from = 6.0
to = 9.0
at = [1.2, -0.55, 1.5]
look = [-0.91, -0.55, 1.6]                  # the glass, seen from the room; the cast member behind the camera shows in it
reflection = { object = "cafe_WindowGlass", strength = 0.35, only = ["@Cast"], bend = true }   # the glass of a library cafe_room set named "cafe"
```

#### Rings

`[[ring]]` entries flip a vector shot to its look's `opposite` palette inside a disc or a band that sweeps out from a point:
the 808 ring that turns the world inside out for a beat. A pixel inside an odd number of rings is flipped and inside an even
number it is not, so two rings that overlap cut back into each other; a thin edge runs on each wavefront. Rings are drawn in
vector shots only: the shots stage warns about a ring live over another shot, and refuses one over a vector shot whose look
has no `opposite`.

| Key | Meaning |
|---|---|
| `at` | clip seconds: the ring starts there, a point (required) |
| `center` | where it sweeps out from (required): `[x, y]` frame fractions from the top left, `[x, y, z]` a world point, or an expression of the `mk q` language (`'obj("boombox_cone_L").matrix_world.translation'`), seen through the shot's camera on every frame of each output; a point outside the frame sends the ring in from that side, one behind the camera draws none |
| `dur` | seconds the ring takes to sweep (default 0.6): a band until its inner rim has passed the frame's farthest corner from the centre, a held disc until it covers the frame |
| `width` | the band's width, frame heights (default 0.25) |
| `hold` | `true`: a disc whose inside stays flipped once it has covered the frame, until the next ring starts (default false: a band that passes) |
| `ease` | `out` (default, `1 - (1 - u)²`: fast, then slowing), `linear` or `inout` (smoothstep) |
| `edge` | `{color, width}`: the wavefront (default white, 3 px at 1080 on the short side); `false`: none |

```toml
[vector.opposite]                       # inside a ring: the background and the outline swap, the inner lines stay
colors = { background = "#1B2A6B", line = "#FFD21F", inner = "#1B2A6B" }
tones = { dark = "#FFD21F" }

[[ring]]
at = 3.622                              # an 808 (`mk timeline onsets --stem drums --stem bass --lo 30 --hi 120`)
center = 'obj("boombox_cone_L").matrix_world.translation'

[[ring]]
at = 7.576
center = [0.5, 0.5]
hold = true                             # the biggest hit: the world stays flipped until the next ring
```

### Freezes

`[[freeze]] {from, to}` (clip seconds) holds the world on the frame of `from` until the frame of `to` while the cameras go on
moving: the hit-stop, the world stopped on a hit and the camera swinging round it. On a frame inside the window `mk render`
and `mk look` set the scene to the window's first frame (poses, props, cords and the sim hold) and every camera to where its own
keys put it on the real frame, the timeline markers of the real frame choosing the camera (they are set aside while the frame
renders, else the render would bind the camera of the world's frame); the looks go on with the clock (a ring keeps sweeping, a
cut inside the window still cuts). At `to` the world is where the timeline has got to, so moves keyed on the beat stay on it:
give the keys inside the window to what should be there when it ends. Windows may not overlap; the shots stage keeps them in
`scene["mk_freeze"]` (`mkmmd.core.freeze`, `mkmmd.blender.freeze`).

```toml
[[freeze]]                      # the biggest 808: the twins stop mid-move and the camera of the shot orbits them
from = 7.576
to = 8.30
```

### Medium shifts

`[[medium]]` draws the whole picture in another look for a window of clip seconds, whatever shot is in the cut: for a beat or
two on a hit the picture snaps into another medium and back, reality flickering between universes (her own colours as cel, a
flat cartoon, a boiling sketch). A window is a shot's look over a span of time: `from`, `to` (clip seconds, required; frames
`from` .. `to` - 1, inside the clip) and the look keys of a [[shot]] (`style` silhouette or vector, required, `colors`,
`tones`, `materials`, `lines`, `sketch`, `hide`, `keep`, a silhouette's `accent`, `tint`, `knockout`, `grow`, `samples`;
[Looks](#looks-silhouette-vector-and-reflection)), with `[medium.aspect.<output>]` tables laid over it as over a shot's
(`style = "none"` leaves an output to its shots). No reflection: a window is a flat look. The cameras, the cut and the cut
effects go on as they are; `mk render` and `mk look` draw the window's look on the cut's frames and on the plates of the
transitions inside it alike, and mk's own renderer leaves its frames to Blender. Windows may not overlap; the shots stage
keeps them in `scene["mk_medium"]` (`mkmmd.core.medium`), checks their object patterns and material rules on the finished
scene like a shot's, and reports them (`_medium`: each window's frames and look per output); a ring live over a vector window
needs its `opposite` palette there too.

The comic print, the broken signal and the datamosh are post work, `[[fx]]` kinds `halftone`, `split` and `mosh`
([Frame effects](#frame-effects)), and the figure on twos is a [[stepped]] window ([Stepped motion](#stepped-motion)): a
medium shift is usually a window of each over the same span.

```toml
[[medium]]                      # the snare: a flat cartoon, red sky, white ground, bold ink
from = 26.46
to = 26.94
style = "vector"
colors = { background = "love", line = "base", inner = "base" }
tones = { paper = "text", ink = "base" }
materials = [{ match = ["*_floor"], tone = "paper" }, { match = ["*_piece"], tone = "ink" }]
lines = { outline = 7, inner = 3 }

[[medium]]                      # the stop: her own colours as cel, thick ink
from = 23.43
to = 24.19
style = "vector"
materials = [{ match = "*", tone = "cel" }]
tones = { cel = { texture = true, shade = "#5F5A86" } }
lines = { outline = 7, inner = 3 }

[[medium]]                      # the next stop: a pencil drawing that boils on twos
from = 29.51
to = 30.27
style = "vector"
hide = ["*_floor"]
lines = { outline = 3.2, inner = 1.6 }
sketch = { paper = "text", ink = "base", strokes = 2, grain = 0.35, hatch = 7 }
```

### Transitions and inserts

Effects that need two shots at once are composited by `mk post` from layers `mk render` draws next to the cut's frames. `mk build`,
`mk render`, `mk post` and `mk look` all read one plan made from mk.toml alone (`mkmmd.core.transition.plan`, numpy-free maths, tested
without Blender). A **window** is the run of frames an effect changes: the `dur` seconds that END at the cut `at` for a transition
(`round(dur * fps)` frames, at least 2, so the incoming shot lands on the beat), the whole `from`..`to` of an insert (its frames
`from` .. `to` - 1). **Plates** are the other shot's frames through its own camera and look. The layers sit in the output's frame
folder, named by Blender frame (`<frame>.png`, five digits):

- `plate/<shot>/<frame>.png`: a shot's frame; a shot that is in the cut at that frame is its own plate: the cut's frame.
- `matte/<shot>/<frame>.png` and `back/<shot>/<frame>.png`, for an expand or collapse: the flat shot's figure alone as coverage,
  drawn at twice the frame's size, and its frame without the figure.
- `point/<key>/<frame>.json`: a projected anchor or centre, `p` (frame fractions from the top left), `depth` and `m` (frame heights
  per metre at that depth, which sizes things given in metres); the key is `t<i>` for transition `i`'s `center`, `i<j>` for insert
  `j`'s `anchor`.

Layers are claimed with an empty file like frames (the finishing file is written last), so a stopped render resumes, `--jobs N`
shares them and the disk check counts them. `mk build` keys the camera of a shot that lends frames (`keyed` in the shot table).

A transition has these keys; the kind decides the rest, and a key of another kind is an error that lists the known ones.

| Key | Meaning |
|---|---|
| `at` | clip seconds of the cut (required): exactly one shot of the cut must start on that frame (use that shot's `from`); the window is the frames before it, inside the outgoing shot |
| `kind` | `"expand"`, `"collapse"`, `"slash"` or `"whip"` (required) |
| `dur` | window length in seconds, ending at the cut (default 0.4; 0.18 for `slash`, 0.15 for `whip`) |
| `ease` | `in` (u², slow start), `out` (1 - (1 - u)²) or `inout` (smoothstep) (default `in`; `inout` for `slash` and `whip`) |

`expand`: the figure of the OUTGOING flat shot grows and turns; inside it plays the next shot. `collapse`: the INCOMING flat shot's
figure starts huge with the outgoing shot inside it and shrinks onto its place, landing on the cut (an expand played backwards).
The figure's shot must have a flat look (`style = "silhouette"` or `"vector"`) in every output.

| Key | Meaning |
|---|---|
| `scale` | `[lo, hi]`, ratio-interpolated from the figure itself: `lo` is its own size (default 1), `hi` a number or `"fill"` (default): exactly what fills the frame (corners included, 2 % to spare) at the last window frame of an expand, the first of a collapse. A number too small for that is raised to it; `0 < lo < hi` |
| `turn` | `[a, b]` degrees, clockwise on screen, linear over the window (default `[0, 90]`) |
| `center` | the point the figure scales and turns about: `"subject"` (default), the figure's centroid, moved to the nearest point at least 35 % as deep inside the figure as its deepest point; `[x, y]` frame fractions from the top left; `[x, y, z]` a world point; or a `mk q` expression string (`'bone("spine", "Model_arm").head'`), projected through the figure shot's camera on every window frame. A given centre outside the figure is moved inside it |
| `edge` | `{color, width}`: a rim round the moving figure; `color` (default `text`), `width` in px at 1080 on the short side (default 4). No rim without the key |

`slash`: a diagonal band sweeps across; behind it the next shot, ahead of it this one, the switch under it. The window is
`round(dur * fps)` frames with the band on screen in each.

| Key | Meaning |
|---|---|
| `angle` | degrees off vertical, clockwise positive (default -20) |
| `width` | of the frame diagonal, in (0, 1.5] (default 0.3) |
| `color` | palette slot, `#hex`, `"a:b:t"` or `[r, g, b]` (default `love`) |
| `dir` | `"right"` or `"left"`: the way the band sweeps (default `right`) |
| `second` | an optional thin band, `{color, width, offset}`: `color` (default `text`), `width` of the diagonal (default 0.04), `offset` the gap to the main band as a fraction of the diagonal (default -0.03; negative: behind it, positive: ahead of it) |

`whip`: a whip pan. The outgoing frame is flung off one side and the next shot follows it in from the other, side by side as
if the camera had swung between them, both smeared along the way by the shutter: each window frame is the two pictures laid
edge to edge, moved on by the eased progress (`mkmmd.core.transition.whip_progress`) and box-blurred along the way by as many
pixels as they travel in `blur` frames. Like a slash it needs only the next shot's plates for its window.

| Key | Meaning |
|---|---|
| `dir` | `"left"`, `"right"`, `"up"` or `"down"`: the way the pictures travel (a pan to the right sends them left; default `left`) |
| `blur` | the shutter in frames, 0 .. 4: how long a smear each frame's speed draws (default 1; 0 is sharp) |

An insert is a thought bubble over the host shot holding another shot (picture in picture).

| Key | Meaning |
|---|---|
| `from`, `to` | clip seconds (required, `to` after `from`): the window. It must lie inside one shot of the cut, the host (the shot in force at `from`) |
| `shot` | the shot inside the bubble (required): any `[[shot]]`, one in the cut or `plate = true`, but not the host |
| `shape` | `"thought"` (the only shape) |
| `anchor` | a `mk q` expression (required), projected through the host shot's camera on every frame: the middle of a head for a thought bubble, `'bone("head", "Model_arm").center'` (with a head bone that runs from the neck to the crown `.center` is about the middle of the head; `.head` is the neck's end and puts the trail on the face) |
| `radius` | metres: the anchor stands for a head of this radius, seen at the anchor's depth; the trail of circles starts at its edge, on the side of the bubble, never over the head (default 0.11; 0 starts them at the anchor) |
| `size`, `ratio` | the bubble's height as a fraction of the frame height, in (0, 1] (default 0.34), and its width / height (default 1.35) |
| `offset` | `[dx, dy]` from the anchor to the bubble's centre in frame heights, x right and y down, kept inside the frame (default `[0.12, -0.30]`: up and to the right) |
| `outline` | `{color, width}` of the ring and the trail: `color` (default `text`), `width` in px at 1080 on the short side (default 5); always drawn |
| `pop` | `{dur, overshoot}`: the three trailing circles pop one after another, then the bubble, springy: `dur` seconds (default 0.3), `overshoot` the peak above rest size (default 0.12). `out = "pop"` plays it backwards, ending on `to` |
| `out` | `"pop"` (default) or `"expand"`: the bubble grows until its picture is the frame, ending on `to`, where the cut goes to `shot`: `shot` must be the shot that starts at `to` |
| `expand` | `{dur, turn, ease}`, only with `out = "expand"`: `dur` seconds (default 0.4), `turn` degrees in total (default 0), `ease` (default `in`) |
| `aspect.<output>` | `{size, offset, ratio}`: that output's own geometry; nothing else may be overridden |

```toml
[[transition]]               # the figure of the OUTGOING flat shot grows and turns; inside it plays the next shot
at = 3.86                    # clip seconds of the cut: a shot starts there; the window is the 0.4 s before it
kind = "expand"              # "collapse": the INCOMING flat shot's figure starts huge and shrinks onto its place
dur = 0.4
scale = [1, "fill"]          # "fill": exactly what fills the frame at the last frame (first of a collapse)
turn = [0, 90]               # degrees, clockwise on screen, linear
ease = "in"
center = "subject"           # or [0.5, 0.4] frame fractions, [x, y, z] a world point, 'bone("spine", "Model_arm").head'
edge = { color = "text", width = 5 }   # a rim round the moving figure; width in px at 1080 on the short side

[[transition]]               # a diagonal band sweeps across; behind it the next shot, ahead of it this one
at = 20.76
kind = "slash"
dur = 0.18
angle = -20                  # degrees off vertical, clockwise positive
width = 0.3                  # of the frame diagonal
color = "love"
second = { color = "text", width = 0.04, offset = -0.03 }   # an optional thin band: gap to the main one (negative: behind it)
dir = "right"                # or "left"

[[insert]]                   # a thought bubble over the host shot, holding another shot (picture in picture)
from = 12.40
to = 13.50
shot = "close"               # any [[shot]]: one in the cut, or `plate = true`; not the host
shape = "thought"
anchor = 'bone("head", "Model_arm").center'   # projected through the host shot's camera on every frame
radius = 0.11                # metres: the anchor stands for a head of this radius
size = 0.34                  # the bubble's height, a fraction of the frame height; ratio = 1.35 is its width / height
offset = [0.12, -0.30]       # from the anchor to the bubble's centre, frame heights, x right, y down
outline = { color = "text", width = 5 }
pop = { dur = 0.3, overshoot = 0.12 }
out = "expand"               # the bubble grows until its picture is the frame, ending on `to`: `shot` must start at `to`
expand = { dur = 0.4, turn = 0, ease = "in" }
aspect.9x16 = { size = 0.2, offset = [0, -0.22], ratio = 1.0 }   # per-output size, offset and ratio
```

The plan refuses what cannot be drawn, with a message that names the effect: a transition `at` where no shot (or several) starts;
a window that starts before the outgoing shot does (`shorten dur`); an expand or collapse whose figure shot is not a silhouette in
every output; an insert that crosses a cut, whose `shot` is the host or does not exist, whose `out = "expand"` shot is not the one
that starts at `to`, or whose window is too short for its pop-in and exit (each at least 2 frames); and any two windows, transitions
and inserts together, that overlap in frames. `mk build` also refuses a window that needs frames from a shot before `[scene] start`
(`lower [scene] start`), and keys the cameras of the shots that lend frames: after editing an effect run `mk build` again (`mk render`
says `run mk build again` when a camera is not keyed over the frames an effect needs). `mk post` reports `matte_scale` per expand or
collapse (`scale`, whether it was `raised`, and `needed`, the scale that fills the frame): a number far above `needed` spends the
window with the new shot already full. A figure that is not in the frame at the covering frame, or a `center` that cannot lie inside
it, is an error that says so (`center = "subject"` always can).

In an expand the first window frame is the figure turned into a window at its own size (the dark shape fills with the next shot at
once; the rim, if any, marks it) and the last is the incoming shot at full frame; a collapse ends on the figure landing on its place
with the outgoing shot still inside it, which turns dark on the cut. The composite is graded as one frame: the grade, grain and
vignette cover plates, figure and rim alike. An insert's picture is its shot's whole frame scaled to cover the bubble
(`warp_scaled`), growing with it; the bubble is an ellipse scalloped by lumps (`CloudField`).

The trail is three circles in the outline colour between the head and the bubble, the smallest first. They begin at the edge of the
head (a sphere of `radius` metres about the anchor, seen at its depth, so one number serves both outputs and a moving camera) on the
side of the bubble, shrink to the room between that edge and the bubble's rim (none under 3 px: move or shrink the bubble with `size`
and `offset`), and pop in turn: the circles start at 0, 14 % and 28 % of the pop and take 30 % each, the bubble starts at 45 %.
`radius = 0` starts them at the anchor itself. Give the anchor the head's middle, not a joint.

**Matte maths** (`mkmmd/matte.py`, numpy and OpenCV): a figure is kept as a signed distance field (negative inside), so turning and
scaling it moves the contour exactly and the edge stays one pixel wide and true to the shape at any zoom: a coverage image scaled
n-fold would be n pixels soft. `signed_distance` takes each partly covered pixel's offset from the exact inverse of a straight edge's
area (from the coverage and the gradient's direction), pixels within two of them the best neighbouring edge line, the rest the nearest
edge pixel plus its offset; the field is sampled bilinearly, Catmull-Rom within 3 matte pixels of the contour (OpenCV's cubic ripples
on a ramp). The tests hold it to these bounds on antialiased half-planes: partial pixels within 0.03 px, within 2.5 px of the edge
0.06 px on average; a zoomed 16-fold edge deviates at most 0.5 px from its line with a ramp of at most 3 px; an L-shape turned and
scaled overlaps its analytic image by 98.5 %. `cover_scale` bisects the scale that fills the frame, corners included.

Blender draws the layers in `render_frames` and `look` (`mkmmd/blender/transition.py`): the cameras are looked through with the
timeline markers set aside, `Looks.prepare(frame, output, shot=...)` enters that shot's look, the matte is a Workbench pass of the
subject objects alone (`Looks.matte`) and `back` the silhouette composed without the subject (`compose_silhouette(...,
subject=False)`). `mk post` lays the effects over the cut's frames before the screen type and the grade; a missing layer is an error
unless `--allow-gaps` (the plain cut shows there). `mk look` previews a frame inside a window composited, from layers it draws into
its own folder. `--no-transitions` leaves the effects out of `mk render`, `mk post` and `mk look` (`--no-styles` leaves them out of
`mk render` and `mk look` too: a figure needs its silhouette look).

### Glitches

`[[glitch]]` makes objects vanish in a digital glitch over a window inside one shot: the figure they make breaks up in slices
that jump sideways, two ghosts of its shape in their own colours trail it to either side, a block or two of ghost colour cuts
through it, and more and more of its slices drop out, with a frame now and then where it blinks out whole or holds still,
until on the window's last frame it is gone. From the frame the window ends on, the build keeps the objects hidden (a CONSTANT
`hide_render` key added to what they already have), so they stay gone.

| Key | Meaning |
|---|---|
| `from`, `to` | clip seconds (required, `to` after `from`): the window, `from` .. `to` - 1 in frames (at least 2). It must lie inside one shot of the cut and overlap no other effect |
| `objects` | the objects that vanish (required): a pattern or a list of them, as a look's `hide` takes (`name*`, `@collection*`, `prop:key`): a character is its meshes, `"Rin_*"`; add what it holds, `"mic*"` |
| `seed` | the random stream's seed (default 1): another seed, another glitch; the same, the same frames |
| `shift` | the farthest a slice jumps, as a share of the frame width, in 0..0.5 (default 0.06); it jumps a third as far on the first frame and the whole way from the middle on |
| `split` | the ghosts' offset in px at 1080 on the short side (default 6), more as the window goes on |
| `colors` | the two ghosts' colours, `[left, right]`: palette slots, `#hex`, `"a:b:t"` or `[r, g, b]` (default `["love", "foam"]`) |

```toml
[[glitch]]                   # the twins glitch out on the snare and are gone on the next one
from = 10.803
to = 11.163
objects = ["Len_*", "Rin_*", "mic*"]
colors = ["#FFFFFF", "#8FB4F0"]
```

`mk render` draws the shot again with the objects hidden on every window frame (`bare/g<i>/<frame>.png`, a full frame, counted in
the disk check) and `mk post` and `mk look` composite (`mkmmd/core/glitch.py`, numpy only): the figure is where the frame
differs from that bare frame (by more than 0.03 in any channel, fully from 0.12), its outline, its own colours and the shadows
it casts included, so any look works. The window is cut into slices 1.2 to 7 % of the frame high; on a glitching frame 45 % of
them jump, and the share dropped out grows as `u^1.3` over the window (`u` 0 on its first frame, 1 on its last); a frame after
the first blinks out with a chance of 12 % rising to 42 %, and an early one holds still with a chance of 15 % falling to 0.
Every frame draws from its own stream (the seed and the frame's place in the window), so a render is repeatable and a frame
does not depend on the ones before it. Away from the figure, rows it does not reach and columns beyond its farthest jump and
ghost, the picture is never touched. The plan, `mk build`'s shots report (`_cut_effects.glitches`) and its keys report
(`_glitch_hidden`: how many objects each glitch hides after it) list it; a pattern that matches no object is a WARNING.

### Frame effects

`[[fx]]` entries work on the finished pictures of a span of the cut: `mk post` applies them to each frame after the transitions,
inserts and glitches and before the lens streaks, the screen type and the grade, and `mk look` previews them the same way
(`mkmmd/core/framefx.py` checks and times them, `mkmmd/framefx.py`, numpy and OpenCV, does the picture work). Unlike the cut
effects they may overlap one another and any window, and on a frame inside several they are applied in the order of the table.
They read the rendered frames as they are, without their effects: an echo of a frame inside a glitch echoes the unglitched frame.

| Key | Meaning |
|---|---|
| `kind` | `lens`, `echo`, `slitscan`, `push`, `scope`, `zoom`, `shock`, `slab`, `tint`, `halftone`, `split` or `mosh` (required) |
| `from`, `to` | clip seconds (required, `to` after `from`): the span, `from` .. `to` - 1 in frames, inside the clip |
| `strength` | a number, or `[[t, v], ...]` keys over clip seconds, linear between them and held outside (default 1): the lens's mass (its Einstein radius goes as the square root), an echo's opacity, a slitscan's depth, a scope's opacity, a zoom's and a shock's size, a slab's opacity, a tint's, a halftone's and a mosh's mix, how far a split tears. A push has none |
| `center` | lens and push (required), zoom and shock (default the middle): `[x, y]` fractions of the frame (0, 0 the top left), or `{point = "<expression>"}`, a point of the scene ([Expressions](#expressions)) that `mk render` projects through the camera of the shot in the cut on every frame of the span (`point/x<i>/<frame>.json`) |
| `radius`, `radius_m` | lens and push: the lens's Einstein radius, or the push's disc at the start; `radius` a share of the frame height (default 0.1 for a lens, 0.02 for a push), `radius_m` metres at the depth of a scene point (a hole of a known size, a pupil), one of the two |
| `shadow` | lens: the share of the Einstein radius painted black about the centre (default 0: the scene draws its own). The lens bends the whole picture, what passes in front of its centre too (a figure there wraps into a ring, as if it were behind): key `strength` down while something passes in front that should stay whole |
| `count`, `every`, `fade`, `blend`, `tint` | echo: `count` copies (4) of the frames `every` (3), 2 x `every` ... back, the first at opacity `fade` (0.55) and each next at `fade` times the one before; `blend` `lighten` (default: keeps the brighter, for a light figure on a dark ground) or `darken`; `tint` a colour (slot, `#hex`, ...) for the copies where they differ from the ground (default their own colours). A copy of the ground changes nothing, so only what moves leaves echoes |
| `depth`, `now` | slitscan: each column (`now` `left` / `right`, default `right`) or row (`top` / `bottom`) shows the frame of its own age, from the present at the `now` edge to `depth` frames back (default 24) at the far one; before the clip's first frame the oldest there is stands in |
| `inside`, `play`, `ease` | push: the picture zooms into the disc about `center`, which shows the clip's frame at `inside` (clip seconds, default 0: its first frame), a still or, with `play = true`, playing on from there; the disc's centre glides to the frame's middle as it grows and at the span's last frame it covers the frame with the inside frame at its own size: the seam of a loop back to the start. `ease` how the zoom runs: `log` (default, a steady push), `smooth` or `linear` |
| `stem`, `timeline` | scope: the stem drawn, `drums`, `bass` (default), `other`, `vocals` or `mix`, from the stems cached when `timeline` (default `audio/timeline.json`) was analysed: `mk timeline analyze` makes them, and a missing cache is an error before anything is encoded |
| `at`, `size`, `color`, `width`, `glow` | scope: the trace's centre (`[x, y]` frame fractions, default the middle), its radius as a share of the frame height (0.3), colour (`text`), line width in px at 1080 on the short side (2) and glow (0..1, 0.6); it is added to the picture |
| `window`, `lag`, `gain` | scope: the seconds of sound drawn on a frame (0.04, the latest ending on the frame), the delay of the y axis behind the x axis in seconds (0.0025: a note draws a loop whose shape is its harmonics; 0 draws the waveform against time instead) and a gain on the stem's 99.5th percentile over the span, which fills the radius (1) |
| `times`, `lane`, `min_db`, `every` | zoom and shock: what they fire on, one of `times` (clip seconds) or `lane`, a lane of the `[audio] hits` file (`mk timeline hits`), its hits inside the span at or above `min_db` (0 the lane's loudest), every `every`-th of them (`mkmmd/core/lanes.py`; a lane without the file is an error that says to write it) |
| `amount`, `attack`, `decay` | zoom: a punch on every hit, the picture jumping in by `amount` (0.06: 6 %) about `center` over `attack` s (0.02, eased out) and coming back as exp(-t / `decay`) (0.16 s); the strongest hit counts, they do not stack |
| `speed`, `width`, `amount`, `life`, `color`, `glow` | shock: a ring runs out from `center` on every hit at `speed` frame heights a second (1.6), its front `width` (0.035 frame heights) wide pushing the picture out ahead of it and pulling it in behind by up to `amount` (0.012 frame heights), lit in `color` (`text`) at `glow` (0.6); it fades as (1 - t / `life`)² and is gone after `life` (0.55 s) |
| `at`, `size`, `slant`, `angle`, `color`, `alpha`, `dir`, `in`, `out`, `shutter` | slab: a bar of `color` (`love`) at `alpha` (1) resting with its centre at `at` (frame fractions), `size` `[w, h]` frame heights (`[1.2, 0.18]`), its sides leaning `slant` degrees (18: to the right going up, as italics) and the whole turned `angle` degrees counter-clockwise (-8). It travels `dir` (`right`, `left`, `up`, `down`): in from the other side over `in` s (0.1, slamming: an ease-out) from fully off the frame, rests, and leaves the way it travels over the last `out` s (0.12, flung: an ease-in), drawn at 8 places over `shutter` frames (1) of its motion so it streaks. Persona's panels; stack a few, a thin one behind a thick one |
| `color`, `dark`, `light` | tint: the picture through a gradient by its brightness, `dark` (black) at black, `color` (`#46E07A`, the Dark Hour's green) at mid grey, `light` (white) at white, mixed in by `strength`: key it up over a stop and back down |
| `cell`, `angle`, `level`, `ink`, `lift`, `offset` | halftone: comic print. Where the luma is under `level` (0.5) the picture is lifted toward white by `lift` (0.35) x its darkness and screened with round dots of `ink` (black) on a grid `cell` px at 1080 apart (9) turned `angle` degrees (45), a dot's area growing with the darkness until the dots merge into solid ink at black; the lights stay clean. Then the red plate is printed `offset` `[x, y]` px at 1080 off (`[3, 2]`) and the blue as far the other way: a press out of register |
| `bands`, `shift`, `share`, `rgb`, `hold`, `seed` | split: a broken digital signal. The frame is cut into `bands` (14) bands of random heights, `share` (0.35) of them torn sideways by up to `shift` (0.08 of the frame width; what leaves one side comes in at the other), the red and blue planes pulled `rgb` px at 1080 apart (10) in a torn band and a third of that everywhere else. A tear holds `hold` seconds (0.05), then the signal breaks another way; the same `seed` (1) breaks the same frames the same way |
| `block` | mosh: a datamosh. The cut's frame before the span is held and carried along the cut's own motion from each frame to the next since the span's first frame (Farneback optical flow on the frames at a quarter size, averaged over `block` px at 1080 squares, 16, so it moves in chunks as a codec's motion vectors do), and mixed over the real frame by `strength`. Started on a cut it is the classic one: the outgoing picture smeared by the incoming shot's motion, as when the key frame of a cut is lost. Frames asked out of order are worked out again from the span's start |

```toml
[[fx]]                        # the hole bends the stars behind it, its mass pumping with the kicks
kind = "lens"
from = 12.0
to = 20.0
center = { point = "obj('bh').loc" }
radius_m = 0.8
strength = [[12.0, 0.6], [12.1, 1.0], [12.4, 0.6]]

[[fx]]                        # her hops leave Muybridge echoes
kind = "echo"
from = 20.0
to = 24.0
every = 2
tint = "foam"

[[fx]]                        # into her eye, and back to the first frame
kind = "push"
from = 40.0
to = 41.0
center = { point = "bone('eye.L').head" }
radius_m = 0.006

[[fx]]                        # the kicks punch the picture in
kind = "zoom"
from = 0.0
to = 41.6
lane = "kick"
min_db = -1.5
amount = 0.05

[[fx]]                        # a Persona panel slams in from the right and is flung off the left
kind = "slab"
from = 2.1
to = 2.9
at = [0.74, 0.2]
size = [0.9, 0.12]
color = "#3FE6FF"
dir = "left"

[[fx]]                        # the Dark Hour: green over a dead stop
kind = "tint"
from = 17.3
to = 18.0
strength = [[17.3, 0.0], [17.36, 1.0], [17.8, 1.0], [18.0, 0.0]]

[[fx]]                        # the finger gun as a comic panel: Ben-Day dots, the plates off register
kind = "halftone"
from = 35.58
to = 36.34
cell = 13
level = 0.5
offset = [4, 3]

[[fx]]                        # the drop after the stop: the old picture dragged by the new shot's motion
kind = "mosh"
from = 18.21
to = 18.97
```

`mk render` draws the point layers of a scene centre with the other layers; `mk post` reads the frames an echo, a slitscan, a
push or a mosh needs from the cut it rendered, and `mk look`, which draws only the frames asked for, renders those beside them
as plates of the shot in the cut there (`TR.fx_frames`). The plan, `mk build`'s shots report (`_cut_effects.fx`) and `mk post`'s
report (`effects.fx`) list them.

## Text

`[[text]]` puts type on a surface: a sign panel of a set (the `highway` gantries and billboards), a prop's screen, label or page
(any `use.surface` card entry), or a panel placed freely on anything. Each entry is one object named `name`: an empty mesh with a
geometry-nodes modifier (String to Curves, Fill Curve, Extrude Mesh for `depth`) and a palette-coloured emissive material, parented to
the owner's root (or to the surface's `object`), so it rides a moving car and sits `lift` (2 mm) in front of the surface. Whatever
changes over time is a key on a node input (the number, the typewriter's ramp, the emission gain), never a frame handler, so
`mk render` needs nothing but the saved `.blend`; the fonts are packed into it. The text frame is the card's: x right (`up x normal`),
y up, z out of the surface. The stage runs after `lights` and before `keys`, so `[[key]]` can toggle a text's `hide_render` or move it.
The same entry also makes lyric type (`lyrics`), handwriting (`ink`) and type of the picture (`screen`); the rows below say where.

| Key | Meaning |
|---|---|
| `name` | object name (required, unique, at most 50 characters: a lyric word's object is `<name>_l<line>w<word>[_<shot>][@<output>]` and its node group adds a prefix); `[[key]]` can target it |
| `on` | `"<set or prop>:<surface>"`, a card `use.surface` (center, normal, up, size); the surface name may go when the owner has just one. A list of surfaces is for `lyrics` only |
| `mount`, `at`, `facing`, `up`, `box` | free placement instead of `on` (`facing` is then required): panel centre `at` (default `[0, 0, 0]`), the direction it `facing`, and `up` (default +Z) for its top edge, in the frame of `mount` (a set, prop or object; a camera `<shot>@<output>` carries the text with the shot; default the world). `box = [w, h]` (m) is the panel to fit and align in; without it text is placed around `at` and needs a `size`. `box` also narrows a surface's panel |
| `text`, `value` | exactly one of them. `text` is the string (`\n`, or a list of lines, for several lines). `value = {keys = [[t, v], ...], format = "{:.0f}", interp = "BEZIER"}` is a number keyed over clip seconds, `interp` `CONSTANT`, `LINEAR` or `BEZIER`, formatted with literal text around one `{:[0][width][.decimals][f\|d]}` field (`"{:03d}"`, `"{:.1f} MPH"`; `d` takes no decimals); zero padding is for non-negative numbers |
| `font` | asset registry slug (kind `font`) or a font file; default Blender's built-in font ([Fonts](#fonts)) |
| `size`, `fit` | `size` is the cap height (m). `fit`, in (0, 1], is the share of the panel the ink of the widest string the text will ever show may fill (a keyed number is measured at every string it can show), also the margin text aligns in. With neither, `fit` is 0.9; with both, `size` is the largest cap height `fit` allows; `size` alone does not shrink to fit |
| `align`, `valign` | `left` / `center` / `right` and `top` / `middle` / `bottom` (`align = "left top"` works too; default `center`, `middle`): the ink block of the widest string meets that edge of the margin, or the centre; lines align inside the block by `align` |
| `offset` | `[u, v]` metres along the panel's right and up |
| `color`, `glow` | `color` a palette slot, `#hex` or `"slot:slot:t"` (a mix, `t` weighs the second) (default `text`); `glow` the emission strength (default 1.0; 0 = lit only by the scene) |
| `lit` | share of the scene's light the letters also reflect, 0..1 (default 1.0); 0 is flat ink: only the emission shows |
| `depth`, `lift` | extrusion toward the viewer in m (default 0 = flat); distance in front of the surface in m (default 0.002; screen type 0) |
| `tracking`, `word_spacing`, `leading` | character spacing and word gap (factors, default 1.0); line pitch in cap heights (default 1.5) |
| `reveal` | `{from, to}` clip seconds, both required, `to` after `from`: typewriter, the first character appears at `from`, the last at `to`; spaces cost no time |
| `blink`, `flicker`, `fade` | multipliers on the emission, all needing `glow` > 0. `blink = {period, duty, low, phase, from, to}`: on for `duty` of each `period` seconds, `low` otherwise (defaults 1.0, 0.5, 0.0, 0.0). `flicker = {amount, rate, dips, seed, from, to}`: every 1/`rate` s a new level, mostly near 1, with random dips up to `amount` deep (0..1) in a share `dips` of the steps (defaults 0.5, 12 Hz, 0.25, 1). Both step, only between `from` and `to` (default the whole build range; the gain is 1 outside). `fade = [[t, gain], ...]` (or `{keys, interp}`, default `LINEAR`) is a keyed ramp, e.g. a dash waking up |
| `ghost` | `true`, a number or `{strength, text, color}`: for display fonts, the unlit segments behind the lit ones (the widest string with every letter and digit as an 8, or `text`), added as light so they stay a faint hint. `strength` is their brightness as a share of the lit segments' as displayed (default 0.10; the emission is that share to the 2.2), `color` defaults to the text's pulled halfway to the palette's `muted` |
| `halo` | `true` or `{strength, size}`: a slight glow past the lit edges (copies of the lit shapes on three rings, additive, just behind them); `strength` the share of the lit emission it adds (default 0.3), `size` its reach in cap heights (default 0.04). EEVEE has no bloom; `mk post` halation comes on top |
| `haze` | `false`, `true` or `{distance, cap}`: aerial perspective, a mix toward the horizon colour of `1 - exp(-d / distance)` at most `cap` (defaults 450 m, 1.0; `true` is those). Text on a `highway` set fades into the road's haze like its signs (with that set's `haze`; `false` turns it off); text anywhere else has none unless given |
| `back` | `true`: the text goes on the other side of a card surface (its normal reversed, `up` kept: a pane read from outside); with `on` only |
| `backing` | `true` or `{color, pattern, pattern_color, pad, height, torn, dz, glow, scale, seed, slant}`: a strip of tape behind the text, the ink box plus `pad` `[x, y]` (em; default 0.55, 0.3) or a fixed `height` em, `color` (default `surface`), ends torn by `torn` x its height (default 0.1, below 0.5), `pattern` `plain` (default) / `stripe` / `dots` / `check` in `pattern_color` (default `surface`) drawn from the strip's own UV, `scale` its pitch factor (1.0), `glow` the strip's own emission (default 0.3; 1.0 in screen type, which has no lights), `dz` metres behind the text (default 0.0006), `seed` of the tearing (1), `slant` degrees its ends lean, to the right going up (default 0; -60 .. 60: with `torn = 0` a clean parallelogram, Persona's slab under a heavy italic); it moves, rotates and fades with the word |
| `outline` | `true` or `{color, width, alpha, dz}`: a ring of another colour behind the letters (the filled shapes grown by `width` em, default 0.05, `color` default `text`, at `alpha` in (0, 1], default 0.5, `dz` metres behind, default 0.0003) so they read on any background; it moves and fades with the word |
| `kinetic` | `{show, scale, sx, sy, dx, dy, dxp, dyp, rot, pivot, tracking, weight, opacity, tint, drip, headroom}`: motion keyed on node inputs, written by `lyrics` or by hand ([Lyrics](#lyrics), the kinetic keys) |
| `lyrics` | `{timeline, line, words, style, ...}`: one text per sung word, read from the timeline ([Lyrics](#lyrics)) |
| `ink` | `{strokes \| track, on, width, lift, color, dry, glossy, from, to}`: handwriting instead of type, a pen's strokes that appear behind the nib; the entry carries only `name` and `ink` ([Ink](#ink)) |
| `screen` | `true` or `{anchor, side, margin, height, width}`: type of the picture, laid over the finished frame of whichever shot is cutting; `at`, `box`, `size`, `offset` are in frame heights, and no `on`, `mount`, `facing` or `up` ([Screen type](#screen-type)) |
| `knockout` | `true`: screen type reversed out of a silhouette shot, in any other shot ordinary screen type ([Screen type](#screen-type)) |
| `aspect.<output>` | `{...}`: an output's own version of the entry, its keys laid over the entry's (tables merge key by key, everything else is replaced). An entry with `screen` or `aspect` is built once per output; with several outputs the objects end `@<output>` and exist for that output only. An output the project does not have is an error |
| `extends`, `abstract` | `extends = "<name>"`: the entry is that entry (resolved first, so chains work; a loop or an unknown name is an error; the entry needs a name of its own) with its own keys laid over it, tables merged; `abstract = true` makes an entry a base that is never built (and is not inherited), so a look that eight lyric lines share is written once |

```toml
[[text]]                              # a highway sign panel: two lines, fitted to 80 % of the panel
name = "exit_sign"
on = "road:gantry1_panel2"
text = "NORTH\nEXIT 12"
font = "sign_bold"
fit = 0.8

[[text]]                              # typed out on the lower strip of a billboard
name = "tagline"
on = "road:billboard1_panel"
box = [10, 1.1]
offset = [0, -1.55]
text = "open all night"
font = "hand_marker"
reveal = { from = 3.0, to = 5.0 }

[[text]]                              # a speedometer: a keyed number in a 7-segment font, unlit segments behind it
name = "speedo"
on = "car:speed"                      # or free: mount = "car", at = [..], facing = [0, 1, 0], box = [0.3, 0.09]
value = { keys = [[6.0, 58], [7.9, 71]], format = "{:.0f}" }
font = "seven_segment"
align = "right"
ghost = true
halo = true
fade = [[5.4, 0.0], [5.9, 1.0]]
```

Blender only measures: the ink box of every string the text can show is taken from the same String to Curves node at em size 1, and
`mkmmd/core/typeset.py` (numpy only, tested) does the rest: size and fit, alignment, number formats, the typewriter's keys and
character count, blink and flicker keys, the surface frame. The build reports each text's cap height (`cap_mm`), ink size and whether
it fits, and logs a WARNING when a given `size` overflows its panel. Read a keyed number with `mk q
'bpy.data.node_groups["mk_text_<name>"].nodes["Value"].outputs[0].default_value' --frames ...`.

`backing` and `outline` are built by the kinetic tree (below), so a text that has either takes none of `value`, `blink`, `flicker`,
`fade`, `ghost` or `halo`, like a `kinetic` text (they need a material of their own). The stage refuses, with a message that starts
`text '<name>':`: an unknown key (it lists the known ones), a missing `name`, a name that is too long or that another object has,
both or neither of `text` and `value`, an empty text, an `on` that names no prop or set, no such surface, or a set or prop with
several surfaces when none is named, no `on` and no `facing`, a `reveal` without `from` and `to`, `blink`, `flicker` or `fade` with
`glow` 0, a `lit` outside 0..1, a font that is neither a file nor a registry font (or has no `E` or `H` to take a cap height from,
or draws nothing for a string), a number format the node tree cannot rebuild, and any bad table (`kinetic`, `backing`, `outline`,
`screen`, `lyrics`, `ink`: the message names the key).

### Fonts

`font = "<slug>"` names an entry of the [asset registry](#asset-registry) (kind `font`); a value with a path separator, `~` or a font
extension (`.ttf` `.otf` `.ttc` `.pfb` `.pfm`) is a file, project-relative, absolute or `~/...`. Without `font` Blender's built-in font
is used. Fonts are packed into the `.blend`, so `mk render` needs the file no more. Which fonts a project can use is local state, not
a fact of this repository: it is whatever the project registered (`mk assets list --kind font`).

```sh
mk assets add ~/fonts/Sign-Bold.ttf --kind font --slug sign_bold --author NAME
mk assets set sign_bold license=LICENCE credit="Sign Bold: NAME"       # read the font's terms first; mk never guesses a licence
```

List the slug in `[credits] assets` so `mk assets credits` prints its credit. Use static fonts: Blender reads a variable font's
default instance. `python -m mkmmd.fontinst SRC.ttf OUT.ttf wght=420 opsz=72` pins a variable font's axes into a static font (every
other axis at its default; the family name carries the axes, so instances register side by side) and `--axes` lists them with their
minimum, default and maximum; it needs fontTools (`pip install fonttools`). Register the output like any font. Blender measures the
font for fit and alignment, so size and fit are exact for any font.

### Lyrics

A `[[text]]` with `lyrics = {...}` is lyric type: one text per sung word, read from the timeline ([Timeline](#timeline)) and keyed so
that every word lands on its onset. `mkmmd/core/wordtype.py` (numpy only, tested on synthetic timelines) selects the words and decides
when each lands, arrives, spreads, weighs, leaves and where it stands; `mkmmd/blender/build/wordtype.py` reads the file and hands one
ordinary text per word to the stage, each with a `kinetic` table of keys. The entry's other keys (`on`, `mount` / `at` / `facing` /
`box`, `font`, `size` / `fit`, `color`, `glow`, `lit`, `depth`, `lift`, `offset`, `outline`, `backing`, `screen`, ...) go to every
word; the words take their text from the timeline, so `text`, `value`, `kinetic`, `reveal` and `lyric` are errors, and so are the
looks of a kinetic text that need a material of their own (`ghost`, `halo`, `blink`, `flicker`, `fade`). `on` may be a list of
surfaces (word k goes to surface k mod len: a diagonal chain over window panes in lyric order is such a list, each word fitted into its
own pane). Words that must stay where they are on screen while the camera moves are type of the picture (`screen`, below), or, to
be part of the scene (lit, hidden by what stands in front, seen through glass), on a plane that rides the shot's camera:
`mount = "<shot>@<output>"`, `at = [0, 0, -distance]`, `facing = [0, 0, 1]`, `up = [0, 1, 0]`.

The words never leave the local files: the log, the stage report, errors, object names (`<name>_l<line>w<word>`) and custom
properties (`mk_lyric = [line, word]`) carry numbers only; the string lives in the object's String to Curves input and nowhere
else. Lines and words are numbered from 1 as everywhere.

**Timing.** A word lands on the Blender frame `frame0 + floor(start * fps + 0.4)` (`start` in clip seconds), never before the frame of
`from`, and is on screen from that frame (its arrival plays after it). It is gone from `round(frame0 + to * fps)`, the frame the shots
stage cuts to the next shot, or earlier when its line leaves (`leave`) or another word takes its slot (`recycle`); a word that would
never be on screen is left out (the report lists it by number). Tempo ticks are the timeline's `ticks` (`tempo.ticks`), else its
`beats`.

Keys inside `lyrics = {...}`:

| Key | Meaning |
|---|---|
| `timeline` | the timeline JSON (required; project-relative, or `~/...`) |
| `line`, `lines`, `words` | which words: exactly one of `line = N` and `lines = [a, b]`, 1-based and inclusive; `words = [i, j]` applies to every selected line (one number: one word; default all). Numbers out of range are errors that name the numbers |
| `case`, `clean` | `case` is `keep` (default) / `lower` / `upper` / `title`; `clean = true` keeps letters, digits and apostrophes (a curly one turns straight; a word with nothing else left stays as it was) |
| `style`, `arrive` | how a word arrives: `pop` (default), `slap`, `drop`, `rise`, `slide`, `slam`, `type` or `none`; `arrive = {...}` overrides that style's numbers (below) |
| `from`, `to` | clip seconds. `from`: no word lands before the frame of `from` (default: each lands on its onset). `to`: where the shot cuts and the text is gone (default: the end of the last selected word's note) |
| `leave` | how the line leaves: `cut` (default, hold to `to`), `drip` or `rewind` (below) |
| `drip`, `rewind` | the numbers of `leave = "drip"` and `leave = "rewind"` (below) |
| `spread` | letters spread (tracking grows) while a note is held, by `0.07 + 0.27 smoothstep(0.12, 0.85, hold)` of the word's width (`hold` the note's seconds, at least 0.06) with an ease-out cubic, and relax after it with a 0.13 s time constant; the fit leaves room for the widest it gets. On by default; `false`, or `{scale}` a factor of the amount (default 1) |
| `weight` | `true` or `{lo, hi, breath}`: strokes grow by `lo` .. `hi` em (default 0 .. 0.03) with the word's own `vocal_db` level (the mean over its note, -34 dB the light end, -5 dB the bold one), breathing with the voice (smoothed per frame, `breath` the share of it, default 0.65) while the note is held and 0.1 s after. Off by default |
| `colors`, `color_by` | accent colours (a slot or a list) cycled by `line` (default; the line's number) or by `word` (its place in the selection); without `colors` the entry's `color` |
| `recolor` | `{color, from, over, ease, words}`: a keyed mix toward another palette colour (frost, ash), `color` required; from clip seconds `from` (default `"last"`: when the last word lands) over `over` seconds (default 0.5), `ease` `smooth` (default) or `linear`, for `words = [first, last]` of the selection only (1-based; default all); a word that is gone by then never changes |
| `tilt`, `sizes`, `offsets`, `scales` | per word, a number or a list cycled over the words: `tilt` degrees about the word's pivot, counter-clockwise from the front (`tilt = {random = deg}`: ± that, fixed by the word, the same every build; a stack row's or a slot's own `tilt` wins); `sizes` cap heights in m (for a layout that does not fit one common size: a slot's own `size` wins); `offsets` `[u, v]` nudges in m on top of where the word stands; `scales` cap heights as shares of the line's common one (what `flow` and `stack` use; times the `scale` of a `punch` or `write` look) |
| `backing` | a `backing` table, or a list of them cycled over the words |
| `recycle` | `true` or `{gap, fade, assign, slide}`: a word is gone `gap` frames before the next word of its slot lands (default 0) and fades over the `fade` frames before (default 0: it cuts), and with `slide` (em, + right) is flung off sideways over those frames, faster and faster. A word's slot is its surface (`same`), row (`stack`) or slot (`slots`); `assign` is a list of group numbers cycled over the words instead (the way to share a slot in `flow`, where each word stands alone) |
| `wow` | `true` or a table: tape wow and flutter while a note is held (below) |
| `punch`, `write` | the words that stand out, and the words whose note is held long, take a look of their own (below) |
| `carry` | what a word does that is already up when its block starts (a `from` after its onset, a new shot with `zones`): `land` (default) arrives again, `still` is simply there in its new place |
| `zones` | `{<shot> = {overlay}}`: the line is set again in every shot it crosses (below) |
| `layout` | where the words stand: `same` (default), `flow`, `stack` or `slots`, or a table `{kind, ...}` (below) |
| `bullets` | `{at, offsets, box, facing, up, come, go, lead, after}`: each word flies past a target in the world, an obstacle for a character to dodge (below) |

**Arrival.** Every style also takes `alpha0` (opacity at the landing, default 1) and `alpha_frames` (frames it takes to reach 1,
default 2); an `arrive` key the style does not have is an error. Times are seconds after the landing. `drop`, `rise` and `slide` move
in shares of the panel, so the text needs a surface or a `box`.

| Key | Meaning |
|---|---|
| `style = "pop"` | a damped spring of its scale about its centre, `1 + a e^(-t/td) cos(2 pi t/tp)`. `arrive`: `a` 0.20, `td` 0.06, `tp` 0.18, `rise` 0 (em: it starts that far below and rises over `rise_s`), `rise_s` 0.2, `jitter` 0 (em: a sideways kick of random sign that dies in 50 ms) |
| `style = "slap"` | `pop` plus a decaying wobble of its angle (70 ms, period 0.16 s, the sign alternating with the word). `arrive`: `a` 0.17, `td` 0.045, `tp` 0.13, `wobble` 2.8 degrees, and `rise`, `rise_s`, `jitter` as `pop` |
| `style = "drop"` | falls in from half a panel above and squashes on landing, about its bottom. `arrive`: `drop` 1.0 (the fall height, a factor) |
| `style = "rise"` | steam: from `base` to its slot, swaying, arriving when the last word lands; needs a layout other than `same` and a panel. `arrive`: `base` `[u, v]` shares of the panel (default the last word's slot), `sway` 0.13 em, `sway_rot` 7 degrees, `a` 0.22, `td` 0.06, `tp` 0.20 |
| `style = "slide"` | a strip fed into the deck: in from one side, decelerating, tilted so it straightens as it arrives, and a damped squash of its length (50 ms) as it clunks home; opaque from the first frame (a word at 40 % opacity is hashed grain). `arrive`: `from` `left`, `right` or `alt` (default: alternating, the first from the right), `dist` 0.5 (shares of the panel), `dur` 0.16 s, `tilt` 4 degrees, `clunk` 0.07 |
| `style = "slam"` | slammed onto the picture: it starts `big` times its size and comes down to it over `dur` s, faster and faster, then rings, a damped spring of its scale (`a`, `td`, `tp` as `pop`; `a` negative: it squashes on the impact) and a shake of `shake` em gone in 60 ms; opaque from the first frame. `arrive`: `big` 2.6, `dur` 0.05, `a` -0.10, `td` 0.05, `tp` 0.16, `shake` 0.05. With `layout = "same"`, `recycle = {fade, slide}` and a slanted red `backing` it is Persona's kinetic type |
| `style = "type"` | a typewriter over its note: the first letter on the landing, the last when the note ends (`reveal`) |
| `style = "none"` | no arrival: it is simply there |

**Leaving.** `leave = "drip"`: when a tempo tick falls between the end of the last word's note (minus 30 ms) and `to` minus `min_run`,
every letter stretches downward from its top and falls like a drop, the first one leaving on the tick's frame; with no such tick the
line holds to the cut. `drip = {g, stretch, life, gap, order, min_run}`: gravity in px/s² at a 150 px em (2600), the extra height
(3.2), seconds to fade (0.55), seconds between words (0), `order` `none` (default) / `bottom_first` (the lowest row first) /
`last_first`, and seconds of the shot that must be left (0.4). `leave = "rewind"` on the same tick rule: every word shoots back like a
tape being rewound, `rewind = {life, dist, stretch, gap, order, min_run}`: `dist` shares of the panel with an ease-in (0.6), stretched
to `1 + stretch` times its length (stretch 3) and fading over the second half of its `life` (0.2 s), `gap` seconds between words
(0.02), `order` `first_first` (default) / `last_first` / `none`, `min_run` 0.4 s.

**Tape.** `wow = {amount, rate, flutter, flutter_rate, pitch, tilt, min_hold, attack, tail}` (`true` or `{}` is all defaults): while a
note is held for `min_hold` seconds (0.3) or more the strip drifts like a tape with wow and flutter, fading in over `attack` s (0.12)
and out over `tail` s (0.15) after the note: a slow sway of position (`amount` 0.12 em at `rate` 1.4 Hz), length (`pitch` 5 %) and
angle (`tilt` 2.5 degrees) and a fast flutter (`flutter` 0.03 em at `flutter_rate` 9 Hz), added to the arrival. With `style = "slide"`
and `backing` this is the cassette look ([Cassette type](#cassette-type)).

**Emphasis.** `punch = {top, above, words, ...look}` picks the words that stand out: the `top` loudest of the selection by `vocal_db`,
those at or `above` a level in dB (against the peak of the vocal), and `words` by number within the selection (1-based); at least one
of the three. `write = {min_hold, ...look}` takes the words that are not punch words and whose note is held for `min_hold` seconds
(default 0.6) or more. The look is any of `font`, `color`, `glow`, `outline`, `backing` (`false`: no strip), `scale` (cap height
relative to the line's, positive), `style` / `arrive` (not `rise`), `tilt`, `weight`, `reveal` (`true` writes the word out letter by
letter over its note: the typewriter's ramp, first letter on the landing, last when the note ends). A word in its own font is
measured in it, so the layout fits. The report lists the punch and write words by number.

**Layouts.** `layout = "same"` (default) sets every word in the entry's panel, each fitted on its own and centred: meant for one word
at a time (`recycle`). `flow` (`gap` 0.28 em, times the entry's `word_spacing`): words side by side in reading order, one row per
lyric line, the block centred on the panel and fitted to it (one cap height for all); `rows = N` sets the words in N rows of nearly
equal width instead (`"auto"`: the fewest rows that let them be set at `size`, or the number that sets them biggest), rows are
`leading` cap heights apart and never closer than the tallest strip (a row holding a bigger word, a punch word or a `scales` share
above 1, stands off its neighbours by what that word rises above or hangs below the rest, so rows never overlap), and `align` (`left` / `center` / `right`; default the `screen`
band's side) sets each row against a margin of the panel. `stack` (`rows`, `pitch` 1.3 em, `dir` `down` | `up`, `align` and `shift`
cycled by row, `tilt` per row): word k on row `k mod rows`, the first on top (`down`) or at the bottom (`up`). `slots` (`slots =
[{at, box, align, tilt, size}]`, `assign`): `at = [u, v]` and `box = [w, h]` are shares of the panel (u right, v up from its centre),
the word is fitted into its box and aligned there, `size` is a cap height in m; word k takes slot `assign[k]` (default k mod len).
`flow`, `stack` and `slots` need a panel (`on`, or `box`; `flow` and `stack` also work with a `size`). The words are measured in
Blender with the node the text is drawn with; the rest is `core/wordtype.py`.

**Zones.** `zones = {<shot> = {overlay}}` sets one `line` again in every shot of the cut it is on screen in (at least 2 frames): each
shot's block is the entry with that shot's overlay laid over it (tables merged: any `[[text]]` key, `lyrics = {...}` included; a shot
without an overlay uses the entry as it is), and its objects are named `<name>_l<line>w<word>_<shot>`. The layout of a block is
always the whole line's, so a word that carries across a cut keeps its place when the block does not change; only the last block takes
the line's `leave`, the others hold to the end of their shot. The words that were up at a cut land again on the first frame of the
next shot as `carry` says. See [Cassette type](#cassette-type).

**Bullets.** `bullets` puts every word on a free panel of its own in the world and flies it past a [target](#targets), `at`
(required, e.g. `"cast:rin.center"`), read in the scene on the frame the word lands (its onset, `floor(onset × fps + 0.4)`): the word
is at that point plus `offsets[k]` (`[x, y, z]` m in the world, a list cycled over the words, default `[[0, 0, 0]]`: under the hips
to be jumped, above the head to be ducked) on that frame, came from `come` m away (default `[6, 0, 0]`) `lead` s before (0.6) and
goes on to `go` (`[-6, 0, 0]`) `after` s after (0.6), keyed linearly; it is on screen over that span only. The panel is `box`
`[w, h]` m (default `[1, 0.3]`) facing `facing` (`[0, -1, 0]`) with `up` (`[0, 0, 1]`), so the entry itself gives no placement
(`on`, `mount`, `at`, `facing`, `up`, `box`, `screen` are errors), nor `zones` or a `leave` (a bullet leaves by flying on). The style
still plays inside the flying panel (a `pop` pops as it passes). The summary's `cut_frame` is when the last word has flown, and it
counts the `bullets`. The [`clearance`](#checks) check says how near they came.

Kinetic keys (`kinetic = {...}`; clip seconds; `[[t, v], ...]` lists are linear keys, one per frame where a value changes; a list
whose values are all one number just sets the node):

| Key | Meaning |
|---|---|
| `show` | `[on]` or `[on, off]`: on screen from the frame of `on` until the frame of `off` (node `Show`) |
| `scale`, `sx`, `sy` | uniform scale and per-axis squash / stretch about the pivot, 1 = rest (`Pop`, `SquashX`, `SquashY`) |
| `rot` | degrees about the pivot, counter-clockwise seen from the front (`Rot`) |
| `dx`, `dy` | offset along the panel's right / up in em, the String to Curves size (`Dx`, `Dy`) |
| `dxp`, `dyp` | the same in shares of the panel's width / height (`DxPanel`, `DyPanel`); the text needs a surface or a `box` |
| `pivot` | `center` (default), `bottom` or `top` of the word's ink box |
| `tracking` | Character Spacing multiplier over the text's own `tracking` (`Tracking`) |
| `weight` | radius in em the filled letters are grown by: copies of the shape on a ring of 12 and in the middle (`Weight`) |
| `opacity` | 0..1 alpha of the whole word (`Opacity`) |
| `tint` | `{color, keys}`, both required: a mix toward another palette colour, 0 = the text's own (`Tint`) |
| `drip` | `{t, g, stretch, life, seed}` (`t` required): the exit, from clip second `t` every letter stretches downward from its top and falls, the first letter on `t`, the rest within 0.1 s; `g` px/s² at a 150 px em (2600), `stretch` the extra height (3.2), `life` seconds to fade (0.55), `seed` the random order the letters go in (1) (`Drip`) |
| `headroom` | the share of the word's width the fit leaves for the letters' spread (`lyrics` sets it from `spread`) |

They become keys on Value nodes of the text's node group, so a render needs no Python (read one with `mk q
'bpy.data.node_groups["mk_text_<name>"].nodes["Pop"].outputs[0].default_value' --frames ...`). Letters stay filled shapes of the
String to Curves instances: the drip scales and moves each character about its own top, opacity and tint travel as the attributes
`mk_alpha` and `mk_tint` that the shared material reads (hashed alpha: a word fades without sorting against the glass), the weight is
the shape repeated on a ring.

```toml
[[text]]                              # steam words over a mug: they land on their onsets, rise to their slots and frost over
name = "mug"
mount = "mug"
at = [0, 0, 0.14]                     # the plane through the focus point, facing the camera
facing = [0.5, -0.87, 0.0]
box = [0.13, 0.14]
font = "serif_italic"
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
from = 1.3                            # clip seconds: no word lands before it; the cut hides them at `to`
to = 2.6
arrive = { alpha0 = 0.55, base = [0.0, -0.21] }
recolor = { color = "foam", from = "last", over = 0.2 }
[text.lyrics.layout]
kind = "slots"
slots = [{ at = [-0.2, 0.28] }, { at = [0.2, 0.15] }, { at = [-0.2, -0.07] }, { at = [0.0, -0.2] }]

[[text]]                              # one word per window pane, falling in and dripping away on a tempo tick
name = "front"
on = ["cafe:pane_r1c3", "cafe:pane_r1c2", "cafe:pane_r0c2", "cafe:pane_r0c3"]
font = "serif_roman"
fit = 0.8
lyrics = { timeline = "audio/timeline.json", line = 1, words = [5, 8], style = "drop", from = 2.6, to = 5.0, leave = "drip", sizes = [0.13, 0.055, 0.12, 0.21] }
```

The stage report has one entry per `lyrics` text, `{words, lines, style, layout, skipped, first_frame, last_frame, cut_frame,
drip_from, rewind_from, carried, punch, write}` (Blender frames; seconds for `drip_from` and `rewind_from`; `skipped`, `carried`,
`punch` and `write` as `[line, word]` pairs) and, with `zones`, `blocks` (`{shot, words, style, layout, first_frame, last_frame}` per
shot); and per word the usual text numbers with `lyric` `[line, word]`, `chars`, `kinetic_keys` and `frames` [on, off). The string
itself (`widest`) is not reported.

### Screen type

A `[[text]]` with `screen = ...` is type of the picture, not of the scene. The text stage builds it as an ordinary text object on the
plane z = 0 of the world, one unit per FRAME HEIGHT (1 is the height of the picture, x right and y up from its centre; a 16:9 frame is
1.78 wide, a 9:16 one 0.56), hidden from every scene render. A frame that has any of it on screen gets a second pass from `mk render`
and `mk look` (`Looks.render` in `mkmmd/blender/styles.py`): the type alone, as its emission shows it (lights off, black world, the
plain view transform: a palette colour is that colour), through an orthographic camera one frame height tall. The pass is kept off
the frame, in a file of its own beside it, `<frames>/screen/<frame>.png` (straight RGBA, written first, so a frame on disk always has
its layer); `mk post` lays it over the cut's frame AFTER the cut effects and before the grade (`mkmmd.core.screentype.composite`,
tested without Blender), so a slash, an expanding figure or a growing bubble passes under the words and the grade covers them like the
rest, and `mk look` does the same for the images it shows. It stays where it is put whichever camera is cutting, whatever stands in
front of it and however the shot is lit.

It is a layer rather than a plane that rides the shot cameras because a plane a hand from the lens is hidden by anything nearer,
blurred by depth of field, tone mapped to dusty colours and, the world being single precision, jitters by about a pixel with a long
lens far from the origin: a graphic layer has none of that. Text that belongs to the world and must be seen at an angle keeps
`mount` and `on` (`mount = "<shot>@<output>"` puts a plane on a camera).

`screen = true` is the whole frame inside a margin of 0.04. A band is `screen = {anchor, side, margin, height, width}`:

| Key | Meaning |
|---|---|
| `anchor` | `top`, `bottom` or `center`: the band lies against that edge (default: none, the whole frame inside the margin) |
| `side` | `left`, `center` or `right`: against that margin of the frame, or centred (default `center`) |
| `margin` | frame heights kept clear at the edges, in [0, 0.5) (default 0.04) |
| `height` | the band's height in frame heights, in (0, 1] (default 0.25 with an `anchor`; without one, the frame less its margins) |
| `width` | the share of what the margins leave across, in (0, 1] (default 1.0) |

`at` and `box` (frame heights from the picture's centre) win over the band; `size` (the cap height) and `offset` are in frame heights
too, and `fit` is a share of the panel as always. Because the unit is the picture, a size, a margin or a band means the same in every
shot; the outputs differ only in the frame's width, so `aspect.<output>` (`mkmmd.core.screentype.deep_merge` of its table over the
entry's) gives each its version. An entry with `screen` or `aspect` is built once per output; with several outputs the objects are
named `<name>@<output>` and carry `mk_aspect`, and `mk render` and `mk look` hide the other outputs' objects (screen type is left to
the screen layer, which asks for the output it is drawing). A screen text is on screen on every frame unless `kinetic = {show = [on,
off]}` limits it (lyric words always have it); the layer pass runs only on frames that have some.

`knockout = true` goes into the silhouette's own passes instead: the type is the figure's ink (the shot's `subject` colour, or the
`color` of its own `knockout` table) where it lies on the background and the background colour where it crosses the figure, drawn
after everything else in the shot, so an expanding or collapsing figure takes it along. Lyric words drop their strip and ring for it
(on a plain text leave `backing` and `outline` off: they would be alpha of the same kind). In a shot that is not a silhouette it is
ordinary screen type. A shot's own `knockout = {objects, color}` still serves type that belongs to the scene.

```toml
[[text]]                              # a title in a band at the bottom left, up from 2 to 5 s; 9x16 centres it
name = "title"
screen = { anchor = "bottom", side = "left", height = 0.2 }
text = "HELLO"
font = "sign_bold"
size = 0.07
kinetic = { show = [2.0, 5.0] }
aspect.9x16 = { size = 0.05, screen = { side = "center" } }
```

Limits. The words are one more layer over the whole frame: a transition or an insert never hides them (a `slash` band passes under
them). What belongs to a figure is the exception: a silhouette's `knockout` type is part of that shot's picture. `mk render
--no-styles` and `mk look --no-styles` leave screen type out (it is part of the looks). The layer files are small (the type is mostly
transparent) and a frame with nothing on screen has none. The pass is one more EEVEE render of a frame that has any type on it. A
stale `screen/` folder of an earlier build stays on disk with the frames it belongs to: delete `renders/<preset>` when the type
changes, as for any frame.

### Cassette type

A mixtape look: lyrics hand-lettered on strips of label tape, the loud words in a tape-brand face. Nothing is special to it; each part
is a general key of [Lyrics](#lyrics) (`mkmmd/core/tapefx.py` is the motion and the row breaks, `mkmmd/core/typezones.py` decides who
stands in which shot, `mkmmd/core/wordtype.py` assembles them), and the example below uses them like this:

- Strips: a `backing` list cycled over the words (plain paper in palette colours; `pattern = "stripe"` stripes it), ink in `base`, a
  hand-lettering font. Screen type is unlit, so the paper is its colour exactly.
- Loud words: `punch` (the two loudest: a display font, 1.25 times as tall, a ring and no strip, a `slap` arrival); the others `slide`
  in from alternating sides.
- Held notes: `wow` for a note held 0.3 s or more, `write` with `reveal` for one held 0.6 s or more.
- Leaving: `leave = "rewind"`, on the first tempo tick that leaves `rewind.min_run` (0.4 s) of the shot.
- Across a cut: `zones` sets the line again in every shot it crosses, each with its own `screen` band where the faces are not and
  `knockout = true` over a silhouette; `carry = "still"` keeps the words that were up from arriving twice. The choice is made per
  shot, not per line, so the table lives once, in an `abstract` entry that every line `extends`:

```toml
[[text]]
name = "ly"
abstract = true
screen = { anchor = "top", side = "left", width = 0.58, height = 0.28 }     # 16:9: a block in the top left
font = "hand_marker"
size = 0.06
color = "base"
aspect.9x16 = { screen = { anchor = "bottom", side = "center", width = 1.0, height = 0.3 }, size = 0.05 }

[text.lyrics]
timeline = "audio/timeline.json"
style = "slide"
carry = "still"
wow = {}
leave = "rewind"
backing = [{ color = "text" }, { color = "rose" }, { color = "gold" }, { color = "foam" }]
layout = { kind = "flow", rows = "auto" }
punch = { top = 2, font = "tape_display", color = "text", scale = 1.25, backing = false, style = "slap", outline = { color = "base" } }
write = { min_hold = 0.6, reveal = true }
zones = { rin_cu = { screen = { anchor = "bottom", side = "center", width = 1.0 } }, g1_hero = { knockout = true } }

[[text]]
name = "ly3"
extends = "ly"
lyrics = { line = 3, to = 11.57 }          # the line, and where it is let go: only what differs
```

Preview a word landing with `mk look --frames F --output 9x16` (frames are Blender frames: the report's `first_frame` and the words'
`frames`), a stretch with its motion with `mk render --preset draft --frames t=8:18.7` then `mk post --preset draft --allow-gaps`.

### Ink

A `[[text]]` with `ink = {...}` is handwriting: the strokes of a pen, drawn on a surface, that appear exactly behind the nib with no
Python at render time. `mkmmd/core/ink.py` (numpy only, tested) makes ONE ribbon mesh of all strokes in the surface frame (x right,
y up, z out): per pen-down stroke a strip of quads `width` wide, `lift` above the paper, with a square cap half a width long at both
ends, bends mitred (up to two half widths: sharp turns thin out, they never spike), repeated points merged and a one-point stroke made
a dot. Every vertex carries the time the nib is on its cross-section as the point attribute `tw`. The material (dithered alpha,
exactly 0 or 1) shows a fragment once the ink object's clock has passed its `tw`; the clock is the object's custom property `clip_t`,
KEYED linearly over the frames to clip seconds (no driver, no handler: `mk q 'bpy.data.objects["ink"]["clip_t"]' --frames ...`).
Attributes interpolate along every quad, so the front sits at the nib between points and between frames, and a nib that moves 3-10 mm
per frame still leaves legible letters. Fresh ink is glossy (coat, low roughness) and 18 % deeper in colour, and dries matte over
`dry` seconds. Like any text the object is parented to the owner's root, so it follows the page; it casts no shadow and takes `[[key]]`
like any object. The entry carries only `name` and `ink`; every other key goes inside `ink = {...}`.

| Key | Meaning |
|---|---|
| `strokes` | the strokes file (project-relative): `{"unit": "mm", "page": [w, h], "strokes": [{"t": [...], "p": [[x, y], ...]}]}`: page millimetres from the top-left corner as read (x right, y down) and the clip seconds the nib is on each point (never decreasing); `unit` may be left out; `page` (optional) must match the surface to 1 mm. Points ~0.15 mm apart make letters. Exactly one of `strokes` and `track` |
| `track` | instead of `strokes`, the coarse fallback: `"nib"` (the project's `tracks/nib.json`) or a `.json` path of a pen track (`frames`, the nib's world positions `target` and a `down` flag per frame, 1 while the pen is down; see [Grips](#grips)): each run of `down` becomes a stroke through the `target` positions, one point per frame (shapes, not letters), counted only while the nib is within 1 mm of the surface's plane (a pen lifted away for a pause draws nothing) |
| `on` | `"<prop or set>:<surface>"`, a card `use.surface` (required): page millimetres map onto its panel from the top-left corner (the library `cafe_page` prop, `[[prop]] name = "page"`, `card = "library:cafe_page"`, is such a surface, 160 x 220 mm) |
| `width`, `lift` | ribbon width (default 0.00042 m) and distance above the surface (default 0.00018 m) |
| `color` | palette slot, `#hex` or `"slot:slot:t"` (default `pine`) |
| `dry`, `glossy` | seconds fresh ink stays glossy (default 0.55); `glossy = false` (or `dry = 0`): matte from the first moment |
| `from`, `to` | clip seconds: only ink written inside the window is built, a stroke that crosses a bound is cut exactly there (an empty window is an error) |

```toml
[[text]]                              # handwriting on the page prop, in pine, from the project's own strokes file
name = "notes"
ink = { strokes = "tracks/ink.json", on = "page:page", from = 2.0, to = 9.5 }
```

```jsonc
// tracks/ink.json: page millimetres from the top-left corner as read (x right, y down), and the clip second of every point
{ "unit": "mm", "page": [160, 220], "strokes": [
    { "t": [2.00, 2.03, 2.06], "p": [[20.0, 25.0], [20.2, 25.1], [20.4, 25.3]] },
    { "t": [2.40, 2.43], "p": [[22.0, 25.0], [22.1, 26.4]] } ] }
```

Only the project can compute where the pen is at every moment, so the strokes are reference data like the pen track: one nib
position per frame (`tracks/nib.json`, see [Grips](#grips)) cannot draw letters of 3 mm that the nib crosses in a frame; a dense
strokes file can. The build reports `strokes`, `points`, `verts`, `faces`, the length of ink (`ink_mm`) and its time range `t`.

## Palettes

`[look] palette` picks a named palette (`rose-pine-moon`, the default, `rose-pine`, `rose-pine-dawn`: the Rosé Pine colours),
`[look.slots]` overrides slots (`slot = "#hex"`). Slots: base, surface, overlay, muted, subtle, text, love, gold, rose, pine,
foam, iris, hl_low, hl_med, hl_high. Sets, props, lights and the grade colour by slot, never by hard-coded values, so a
scene is re-coloured by switching palettes (the café set is made for a light palette, the bedroom and the highway for a dark one). A colour
written in `mk.toml` is a slot name, a `#hex`, or `"slot:slot:t"` (a mix of two in sRGB; `"love:text:0.3"` is 30 % text).

```toml
[look]
palette = "rose-pine-moon"
slots = { gold = "#f2c46d" }        # one slot overridden for this project
```

## Rendering and post

### Rendering

`mk render --preset draft|preview|final` renders each output's cut to `<project>/renders/<preset>/<output>/<frame>.png`
(`<frame>` is the Blender frame, five digits):

| Preset | Resolution | Samples | Motion blur |
|---|---|---|---|
| `draft` | 50 % | 16 | off |
| `preview` | 50 % | 32 | on |
| `final` | 100 % | 64 | on |

`[render]` (all optional) overrides the preset: `samples`, `shutter` (the motion blur shutter, 0.35) and `engine` (`eevee`, the
default (EEVEE Next), `cycles`, `workbench` or `mk`, below). `--samples`, `--percent` and `--engine` override both for one run; `--output NAME` renders some
outputs, `--frames SPEC` some frames (default the whole clip), `--jobs N` runs N Blender processes per output that share the frames.
Frames are claimed with an empty file first, so a stopped render resumes; a disk check refuses to start when the frames would not fit
(it keeps 1.5 GB spare). Shots with a render-time look ([Shots](#looks-silhouette-and-reflection): silhouette, reflection) are finished
flat frames on disk, composed inside the render job; `--no-styles` renders them as lit, and leaves out the cut effects and the screen type too (they are part of the looks). `[[transition]]` and `[[insert]]`
([Shots](#transitions-and-inserts)) add the layers they need next to the frames (`plate/`, `matte/`, `back/`, `point/`; `--no-transitions`
leaves them out). Screen type ([Text](#screen-type)) is `screen/<frame>.png`, straight RGBA, for the frames that have any. With
`[post] keep` ([Post](#post)) each frame gets `keep/<frame>.png` too, the mask of where the kept materials are seen, drawn before
the frame (a frame already on disk without one gets it).

#### mk's own renderer

`engine = "mk"` draws the vector look ([Looks](#looks-silhouette-and-reflection)) without Blender: headless Chromium
(`chromium` on the PATH, or `$MK_CHROMIUM`; it reports the GPU it ran on, and SwiftShader means none was found) draws the
passes the look is composed from on the GPU, from the scene's bake (the bake stage of [Building](#building): `mk build`
bakes it, `mk build --bake` again after the .blend was saved by hand), and worker processes finish them with the same
`compose_vector` and rings as Blender's frames, while the page draws the next (`mkmmd/render`, the page in
`mkmmd/site/web/js/render/`). The passes follow Workbench's own rules, read from its shaders: the material ids without
antialiasing (a material under half opaque only hides what is behind it; of two surfaces at one depth the one drawn first
stays) and the depth of the same draw; the studio light of the scene's shading on white, times Workbench's shadow factor,
and nothing inside the stencil shadow volumes the look's `light` casts, extruded from every mesh's outline; the texture
colour where a tone is drawn or cel (a material without an image: its viewport colour, as Workbench draws it); Workbench's
16 jittered samples, spread with its Blackman-Harris filter and summed in log2, on the light and texture passes. A sketch
boils on the clip second of each frame, as Blender's does.

A shot it cannot draw renders in Blender, whole, with the scene's own engine: another look (silhouette, reflection, lit by
the scene), type over the picture (the text stage's in-scene type), an object the bake cannot hold (a curves or point-cloud
object, particles, a collection instance, a volume, a mesh too big to bake frame by frame), a drawn or cel tone whose texture
the bake lacks, a ring whose centre failed; and so does every frame of a [[medium]] window with a look of its own for the
output (the bake carries the windows; a bake from before them, version 2, is refused: `mk build --bake`). Blender also draws
every frame's layers beside mk's (screen type, the cut effects' plates and mattes), all outputs at once. The report adds per
output `engines` (frames per engine), `shots` (each shot's, or `medium <i>`'s, engine and, for Blender's, why) and
`screen_layers`, and `mk` (`frames`, `seconds`, `gpu_seconds`, `renderer`). A page that
sends no frame for a minute (five while it loads the bake) has stopped: the render fails naming the frame.

What still differs from Blender's frames: an MMD model's custom normals ride its bones where Blender recomputes them on the
deformed mesh, so on a bent limb the line where light turns to shadow can move a pixel or two. On Hello Brooklyn's draft
(690 frames of two characters, a boombox, a cable, rings, a freeze) 0.2 % of the pixels of a 9:16 frame differ by more than
8 levels on average (1.2 % at worst, a square frame), and the screen layers are the same files. That draft took 2 min 31 s
with `mk` (its own frames in under two of them, about 0.1 s of GPU each; Blender's share, the lit orbit, the screen type and
the effects' layers, alongside) against 14 min 7 s in Blender, one job per output.

### Post

`mk post` composites the cut effects over the cut's frames, draws the lens streaks (`streaks`, below) from the result, lays the screen
type over it, then grades the frames and encodes
`<project>/out/<name>_<output>[_<preset>].mp4` (the preset is left out of the name for `final`) with `[audio]` (`file`, `start` = song
seconds at clip time 0): H.264 (x264, `--crf` 15, yuv420p, even size) and AAC 320k with a 60 ms fade-out. A clip of one frame
(`[project] duration` of one frame) is a still: each output's graded frame is written as `<name>_<output>[_<preset>].png`, at
its own size, with no sound (the sim and cloth reports then measure its frame and the two before). A missing frame or layer is an error
unless `--allow-gaps` (the previous frame repeats; the plain cut shows where a layer is missing); `--to DIR` also copies the videos;
`mk look` composes the same way for the images it shows. The report has each video's frames, size, `screen_type_frames`, and the darkest luma
after the grade (`min_luma`).

`[audio.master]` (or `master = true` in `[audio]` for the defaults) masters the clip's audio for small speakers before it is
encoded (`mkmmd/master.py`, numpy only). A phone's speakers play little below about 250 Hz, so a mix whose kick or 808 lives
at 50-150 Hz loses its booms there (Hello Brooklyn's 808s: 94 % of their energy under 150 Hz) while big speakers thump. The
booms are a lane of `[audio] hits` (clip seconds), and everything happens on them: the low band under `split` around each
boom is saturated into overtones a phone can play (the ear hears the missing fundamental from its harmonics) and mixed in,
the boom's attack (1.5-5 kHz) is lifted, and the rest of the mix ducks under it; then the sub is cut and the whole is
brought to a loudness under a look-ahead limiter. The mastered clip is a WAV in `.mk/cache/post/`; the song file is never
touched. The report's `audio_master` has the loudness in and out (ITU-R BS.1770-4, LUFS), the peaks, how far the limiter
turned down, and `boom_phone_db`: how far the booms stand out through a phone's speakers (the level just after each against
just before it), before and after.

| Key | Meaning |
|---|---|
| `lane` | the hits lane of the booms (default `"808"`) |
| `split` | Hz: the booms' low band (160) |
| `harmonics`, `drive` | dB of the overtones against that band (-3) and how hard it is saturated (4) |
| `tail` | s: how long after each hit the overtones last (0.4) |
| `knock` | dB: the attack's lift on each boom, fading over 25 ms (6) |
| `duck`, `release` | dB the rest of the mix dips on each boom (4), held 40 ms and back over `release` s (0.15) |
| `highpass` | Hz: everything under it is cut (35) |
| `loudness`, `ceiling` | the integrated loudness in LUFS (-12) and the limiter's ceiling in dBFS (-1) |

Hits with a `db` weigh the attack and the duck by it. The defaults are a punchy master (Hello Brooklyn's booms: 0.4 dB over
their surroundings on a phone, 7.7 after); `harmonics = 0, drive = 6, knock = 9, duck = 7, release = 0.18, loudness = -10` a
hard one (10.1).

`streaks = {strength 0.35, threshold 0.88, length 0.15, core 0.02, tint "foam", mix 0.75, cap 1.0}` (or `true` for those; off
without it) is the flare of an anamorphic lens: luminance above `threshold` (eased in over the rest of the range: lamp cores,
headlights, neon) is smeared sideways only, a tight core of Gaussian width `core` and a long tail of width `length` (fractions of
the frame's longer side, so every output streaks alike), each scaled to its peak so a lone lamp draws a thin line about as bright
as itself; the sum is capped softly at `cap`, tinted `mix` of the way toward the palette slot `tint` and screened over the picture
at `strength`. It is drawn before the screen type, so lyrics never streak. An unknown key is an error.

`[post] screen = "over"` lays the screen type over the finished, graded picture instead of under the grade (`"graded"`, the
default): its colours exactly as written, no grain or vignette on it, as type set on a photograph. Any other value is an error.

`[post]` (all optional) is otherwise the grade, applied after the screen type in this order:

| Key | Meaning |
|---|---|
| `halation` | `{strength 0.22, radius 0.012, threshold 0.72, tint "rose"}`: glow around highlights (radius as a fraction of the frame height), tinted like film halation; off unless the table has a key |
| `saturation` | 1.06 |
| `keep` | `{cast, materials, saturation 1.0}`: those materials of a cast member (PMX names; her irises: `["瞳"]`) keep their colour where they are seen, at `saturation` of the frame's own instead of the grade's (a black-and-white picture with only her eyes in colour: `saturation = 0.0` beside it). `mk render` draws the mask beside each frame, `keep/<frame>.png`: an EEVEE pass with every material a flat emission, white for the kept ones and black for the rest, so what stands in front hides them as in the picture (MMD materials see-through by their own alpha, texture included; anything else opaque), linear, antialiased. Screen type laid on under the grade is graded as type; inside a cut effect's window the frame is graded whole; a missing mask is an error unless `--allow-gaps`; the report counts `keep_frames`. An unknown member, key or material is an error that lists what there is |
| `contrast`, `pivot` | 1.10 around the pivot 0.60: midtone contrast |
| `split` | `{shadows "iris", highlights "gold", amount 0.06}`: split toning toward palette slots; off unless the table has a key |
| `vignette` | 0.12: corner darkening (a fraction) |
| `floor`, `toe` | `"base"`, 0.12: the darkest colour allowed (a palette slot or `#hex`; `false` for none): luminance below the floor plus the toe is lifted smoothly into that band and tinted toward the floor colour, so nothing in the frame is black |
| `grain` | `{amount 0.012, size 1.5, seed 7}`: film grain, eight cycled noise frames |

```toml
[post]
contrast = 1.12
split = { shadows = "iris", highlights = "gold", amount = 0.05 }
halation = { strength = 0.2, tint = "rose" }
grain = { amount = 0.01 }
```

### Play

`mk play [--preset P] [--output NAME]...` opens the encoded videos (`out/<name>_<output>[_<preset>].mp4`; default preset
`final`, else `preview`, else `draft`; every output unless `--output` says) in the player of your choice: the `player`
setting (Configuration), a command line run once with the videos. Without it, the system's opener (`xdg-open`, `open`)
gets each video. The command line may hold placeholders: `{files}` (the videos, one argument each; without it they go
last), `{chapters}` (an FFMETADATA file of the shots in the cut, plates left out, as chapters, written to
`.mk/play/shots.ffmeta`; mpv's `--chapters-file` and tern-video-block's `--chapters` read it), `{frame0}` (the project's
number of the clip's first frame, so a player can show the frame numbers `mk look --frames` takes) and `{fps}`.

```toml
# ~/.config/mk/config.toml
player = "tern-video-block --split right --chapters {chapters} --first-frame {frame0}"
```

[tern-video-block](https://github.com/verticalrectangle/tern-video-block) plays them in a Tern block beside the pane
you are in, focused: every output side by side, in sync, with the song, the shots as chapters (PgUp / PgDn) and the
project's frame numbers in its status line. `player = "mpv --chapters-file={chapters}"` plays them one after another.

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
numbered from 1; tools refer to words as `(line, word)` and never print their text. The aligner (wav2vec2 trained on
English, words reduced to a-z) times English words; for a song in another language `--no-words` gives the beats and
bars only. While it runs, `analyze` names its stages on stderr (`timeline: stems`, `whisper`, `align`): the Brief tab
shows them.

`onsets` (`{"other": [t, ...]}`, clip seconds, sorted) are the moments notes are plucked or strummed in a Demucs stem,
`onsets_meta` its `band`, `latency_ms` and per-onset `strength` (dB). `analyze` stores the `other` stem (guitars and keys).
The detector (`signal.band_onsets`, numpy only) is the spectral flux of the stem in 800-6000 Hz (pick noise and string
harmonics; bass, kick and hi-hat fall outside): a 1024-sample Hann STFT every 2.5 ms, the power in 8 log-spaced slices,
per frame the mean rise in dB over the last 10 ms (half-wave rectified; levels relative to the stem's loud parts, floored
60 dB below; frames under -70 dBFS are silence). A flux peak is an onset when it exceeds the local median + 2.5 MAD (1 s
window) and 2 dB, and the strongest within 0.1 s wins (a strum is one onset). The time is where the strum STARTS: strums
climb in steps and the biggest is often the last, so from the winner the detector walks back through the peaks that are at
least 0.3 of it and within 40 ms of each other and takes the earliest, then refines it to a fraction of a frame and
subtracts the latency: centred frames put the flux peak about 6 ms before a pluck's first sample, measured by running
the same flux on synthetic plucks (`signal.onset_latency`), not set by hand. `mk timeline onsets [--stem S] [--lo HZ
--hi HZ] [--timeline FILE] [--context 20]` recomputes a stem into an existing timeline and changes nothing else in the
file: the song span is read again the way `analyze` read it (the `audio` block), so the stem cache hits (a `--context`
that does not reproduce the recorded window is refused). It prints numbers only: the count, onsets per beat, and the
offset to the 8th-note grid of the beats (median, MAD, p95 in ms; 8ths without an onset; onsets between 8ths).

`mk timeline hits [--timeline FILE] [--out FILE] [--lane NAME=STEM:LO-HI] [--gap NAME=STEM:DROP_DB:SECONDS] [--notes
NAME=STEM:LO-HI] [--context 20]` writes the hit lanes of a timeline's span to a hits file (default `audio/hits.json`
beside the timeline: `[audio] hits` takes it), from the same cached stems (`mkmmd.timeline.hits`), in clip seconds:

| Lane (default) | Kind | Entries |
|---|---|---|
| `kick` (drums, 35-150 Hz), `snare` (drums, 180-1000 Hz), `hat` (drums, 7-16 kHz) | onsets | `{t, db}`: when the band is struck (the onset detector above with a 60 ms separation, so a 16th-note roll at 158 bpm stays four hits, and no walking back through a strum), `db` the band's level over the 50 ms after it against the lane's loudest. A hit must lift the band 4 dB over the 35-12 ms before it (a wobble in a decaying hit is none), hits more than 24 dB under the loudest are dropped, and a hit that a lane of the same stem has within 15 ms at least 12 dB more prominently (each against its own loudest) is that lane's: a kick's thump in the snare's band. A kick and a snare struck together stay in both |
| `stop` (the mix: 10 dB, 80 ms), `bass_out` (bass: 14 dB, 120 ms) | gaps | `{t, dur, db}`: where the level (30 ms RMS) stays more than the drop under its 3 s running median for at least that long: the start, the length (cut to the clip) and the depth |
| `notes` (other, 200-4000 Hz) | notes | `{t, db, midi}`: onsets with the pitch they start on (the harmonic product spectrum of 10-80 ms after the onset, fundamentals 55-2000 Hz, a fractional MIDI number; an octave off on a weak fundamental happens) |

`--lane`, `--gap` and `--notes` add a lane or replace the default of the same name (stems `drums`, `bass`, `other`,
`vocals`, `mix`). It prints numbers only: per lane its count, hits per beat, a gap lane's total seconds, a note lane's
MIDI range, and its settings. The lanes are what things fire on: a `[[key]]` with `lane` ([Keys](#keys)), a `zoom` or a
`shock` with `lane` ([Frame effects](#frame-effects)); both pick from a lane the same way (`mkmmd/core/lanes.py`: inside a
span, at or above a level, every n-th).

A timeline file has: `fps`, `clip` (`start`, `end`, `fps`), `audio` (`file`, `start`, `duration`, `context`: seconds of song read before and
after the clip), `bpm`, `beat_s`, `beats` and `downbeats` (clip seconds), `grid` (`period`, `phase`, `resid_ms` of the constant-tempo fit, or null),
`downbeat_contrast`, `vocal_db`, `energy_db` and `drums_db` (one value per frame, relative to the clip's loudest moment), `lines` (each with
`words`: `text`, `start`, `end`, `voiced_end`, `ctc_start`, `whisper`, and the line's `start` and `end`), `words_source` ("whisper" or "lyrics file"),
`word_stats`, `onsets` and `onsets_meta`. A timeline may also carry `ticks` (or `tempo.ticks`), a list of tempo ticks in clip seconds that lyric drip and rewind prefer; without it they use the beats. Lyric type reads `lines[].words[]` (`text`, `start`, `end`, `voiced_end`) and `vocal_db`.
`mk timeline show` prints the numbers without any text.

## Cache

`<project>/.mk/cache/<kind>/` holds what is slow to compute, keyed by a SHA-256 of the inputs (arguments, input-file fingerprints,
the solver's source): `grip/` (hand grips, one `.npz` per hand and spec), `strands/` (secondary motion per chain group),
`form/` (prop boxiness, by the evaluated geometry), `timeline/` (Demucs stems, by a hash of the audio span:
`stems-<key>.npz`). Changing hair settings re-runs only the hair; nothing else is recomputed; editing a solver invalidates its results.
`<project>/.mk/look/<time>/` is where `mk look` writes by default. The user cache `~/.cache/mk/` (`MK_CACHE`) holds the job folders, serve
sockets, `empty.blend`, `samples/` and `look/` outside a project, and the reference sets.

Reference clips (`mk ref`) live outside the project: `~/.cache/mk/ref/<project name | default>/<set>/` holds
`clips.json`, the capped downloads, `track/<id>.npz` and the contact sheet; `mk ref clean` deletes everything but
`clips.json`. Only the small measurement JSON is kept with the project (`<project>/ref/<set>.json`). Reference photos for modelling
(`mk ref photos`) go to `<project>/refs/<set>/` (outside a project `~/.cache/mk/ref/photos/<set>/`) with `SOURCES.json` and a contact sheet.

## Characters (`mk model`)

`mk model build SPEC.toml` builds an original character in code: part builders (numpy and Pillow, no Blender) make
meshes, textures, bones, morphs and rigid bodies; the assembler merges them into one PMX; Blender imports that PMX back
with mmd_tools (the way a `[[cast]]` does), the result is verified against what was assembled, described like `mk inspect`
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
height = 1.60                 # optional: the built parts are scaled about the floor so her skull top stands at 1.60 m
needs = {hair = ["body", "head"]}      # parts `--only hair` builds first (default: body); builders may declare too
builders = {tails = "mkmmd.model.parts.hair"}       # part -> module when it is not mkmmd/model/parts/<part>.py
```

**Spec** (`mkmmd.model.spec`): tables merge key by key, anything else (lists too) is replaced; `Spec.files` lists what
was read, `Spec.origin` the file that set each key last; `--set a.b=1` overrides. The builders see the spec watched
(`spec.watch`): after a build, every key of the author's that no part read is a warning, one line per top-level table
(`[spec] WARNING hair.back.x: never read by any part ...`; a `--only` build checks the tables of the parts it built). Keys a
model base set are left out (they serve features a character may switch on), and `tests/test_model_bases.py` keeps the
bases free of dead ones. A builder merges its defaults under its table with `spec.merge(DEFAULTS, cfg)` and reads keys
by name; listing or copying a table counts as reading all of it. **Model bases** are complete characters shipped in
`mkmmd/model/bases/<name>/` (`girl`, the neutral base; `rin`, the worked example on it): `include = ["base:girl"]` merges
one under the file's own tables, `"base:girl/hand.npz"` names a file inside it, and `mk model build base:girl` builds it
alone; [model_base.md](model_base.md) is the playbook for making a character from one. `[proportions] leg_extra = 0.03`
(`mkmmd.model.proportions`, made on the builders' spec before any part reads it) lengthens the legs: every height from the
hip joint up rises, the thigh and shin stretch alike, the feet stay. The **body** part draws its skin one of three ways
(`[body] source`; `mkmmd/model/parts/body.py` lists the keys): `procedural`, lofted from
`[proportions]`; `mesh`, an artist's whole body fitted to the skeleton (`body_donor.py`; the girl base wears one, Blender
Studio's stylized body, CC0, which `bases/girl/make_body.py` rebuilds from its source); `pmx`, taken from an existing PMX
(`body_pmx.py`). **Builders** are `mkmmd/model/parts/<part>.py` with `@builder("hair", needs=(...))
def build(ctx) -> Part` (`mkmmd.model.build`): `ctx.spec`, `ctx.cfg` (the part's own table), `ctx.save_png(name, rgba)`
(into `<out>/tex`, part name prefixed, the returned file name goes into `Material.texture/toon/sphere`), `ctx.parts`
(built so far), `ctx.land` (landmarks: every part's `info["landmarks"]`, semantic name -> np.array(3)), `ctx.rng`,
`ctx.log`, `ctx.need(part)`, `ctx.find_body(bone)`, `ctx.find_bone(name)`. A part returns `Part` (`mkmmd/model/part.py`:
Material, Bone, Mesh, Morph, RigidBody, Joint); after each builder `part.check` and the cross-part checks run (unique
names, parents, weights, materials, bodies, joints).

`[model] height` (metres, floor to the top of her skull) scales what `build.run` returns about the floor, by height over
`[proportions.head] crown_skull` (after leg_extra): vertices, morph offsets, bones, rigid bodies and joints
(`part.scaled`); the builders, `ctx.parts` and `Part.info` stay at the spec's own size, so every proportion the base
tunes holds at any height. **Garments** other than the dress (`outfit`, Rin's) are parts of their own, each picked by
listing it in `[model] parts`: `sweater` (off-shoulder knitwear, below), `pants` (skinny), `socks` (ribbed crew),
`sneakers` (high-tops: the Mary-Jane's foot fitting with a sneaker's settings and a painted face), `choker` (band and
flower), and `hair_bow` (a bow on the side of the head with two tails on chains). Pants and socks are cut out of the
body's own skin (`outfit_shell.Shell`: fields on the skin such as a height or a limb's length coordinate, clipped exactly
along their level lines, moved out along the normals, smoothed, kept off the skin at the vertices and the triangles'
centres; loose cloth is `outfit_shell.drape`d round an axis, coming in under the widest part above it no faster than a
taper), so they carry the skin's weights and bend with it. The sweater is made as knitwear is made, from separate pieces
of even rows of stitches (lofted rings) with the thickness of knit at their edges: a folded rib band round the top of
her body (its top line level round her chest and back, a ring across each upper arm were the arms hanging down, open
only in a slit under each arm; its rows pulled taut round her as an elastic rib is, spanning the cleavage and the fold
where the arm meets the chest; its columns, four to a rib, straight down her chest and back and fanning round her arms;
rolled over at the top, a rounded lip at the bottom, ribbed in its shape), a knit body (rings round her waist's axis:
the convex section of her torso plus ease, hanging from the bust no faster than a taper but curving in under the bust
first, gathered under the band and into the waistband) and sleeves (the convex section of each arm plus ease, sagging,
puffed and gathered into the cuffs, folds at the elbow), ribbed waistband and cuffs. Every piece takes the weights of
the skin nearest to it, smoothed round its rings; round her chest and back the band and the knit near it leave out the
shoulder girdle's share, so they stay with her body when an arm moves, and a helper bone at each elbow and shoulder
(`[sweater.helpers]`, `mkmmd.model.helpers`) takes what the joint's two sides share, so a bent elbow keeps the sleeve
round and a raised arm spreads the cloth at the armpit instead of parting it. Under each sleeve rows of short chains,
tied sideways to their neighbours (`[sweater.chains]`, family `sleeve`), carry the bag of cloth hanging under the arm:
MMD's physics or mk's solver (`[sim.<cast>] families` with `sleeve`) makes it hang and swing. Fold shapes
(`[sweater.folds]`) bunch the knit as her joints turn (the crook of a bending elbow, the shoulder as an arm rises, her
back as an arm reaches forward, above the waistband as she leans, her sides as she twists): mk's [folds stage](#folds)
keys them from her pose, MMD has them as sliders. A cloth copy of the knit body and sleeves (`[sweater.cloth]`,
`mkmmd.model.cloth`: every other ring and column, held under the band, inside the waistband and the cuffs, the sleeves
held toward their shape, with the skin of her torso to collide with) lets mk's [cloth stage](#cloth) simulate the knit
instead. Sizes are metres x S like the dress's. Colours come from `[colors.<part>]`, materials from `outfit_wear` (a
toon ramp, a dark outline; the knit's texture a
stockinette, the bands' a rib). Pants and socks are **drawn** like the design: `outfit_ink` lays pen strokes on the
cloth (thin dark ribbons a fraction of a millimetre above it, with its weights: the socks' ribs, the creases of the
pants; `[<part>.ink]`, materials 靴下線, ズボン線), and the sneakers' painted face outlines its rubber, stripes and
laces. The hair keeps its hanging locks off every garment built before it (`[model] parts` order), so long hair rests
on a sweater instead of inside it.
`[body] bust` (metres, the mesh body) moves each breast's front forward as it is, keeping the donor's round shape; the
flesh it grows follows the chest (`body_donor.chest_flesh`: the shoulder girdle's weights go to 上半身2 there), so the
breasts keep their shape when the arms come down. `[body] bust_tip` moves each breast's tip that much more over a smaller
round (`body_donor.BUST_TIP_RADII`, 3.5 to 4.5 cm): negative flattens their curve, the fullest part back while the
chest round them stays where `bust` put it, which reads as a smaller cup in profile (about a centimetre at most: much
more hollows the front).
Over the top of each shoulder the skin inboard of the shoulder joint keeps only half of the arm's share, the rest going
to the shoulder girdle (肩) (`body_weights.spine_weights`): an arm swinging down lifts whatever turns with it inboard of
its joint, like the far end of a lever, so with all of that share the top of the shoulder rose into a peak when her arms
hung down; with none of it, the skin between what turns with the arm and what stays sank into a groove at the front of
the shoulder that the light missed (a dark patch).
`[body] upper_arm` (the mesh body) scales the upper arms' girth over the donor's, `[at the shoulder, at the elbow]`
(linear between, one number for the shoulder with the elbow left at 1): the arm keeps its joints and length and the
forearm stays; where the skin shares the torso's fit, near the shoulder, it grows less. `[body.skin] toon_edge` and
`toon_soft` set where on the skin's toon ramp the step into `toon_shadow_multiplier` starts (0.5) and how wide it is
(0.10): a wider step and a lighter multiplier keep the shade the same skin in softer light instead of a patch.

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
transfer from body meshes for garments, ramps), `helpers` (a helper bone at a joint that turns a share of the joint's
rotation by a grant, and the weights the joint's two sides share given to it, so a bent elbow or a raised arm keeps a
garment round instead of pinching it: the sweater has one at each elbow and shoulder), `cloth` (a garment's simulation
copy for the cloth stage: a coarse grid off lofted rings, pins, the skin under it to collide with), `tex` (antialiased
UV-space canvas, gradients, toon ramps, sphere maps, noise, atlas), `subdiv` (Catmull-Clark with creases as sparse
operators).
`Mesh.subsurf` applies it before export: the surface equals Blender's Subdivision modifier (boundary "all", linear UVs) to
1e-7 and UVs, weights and morph offsets are carried through the operators. `mkmmd/model/parts/mannequin.py` uses all of
it and is the fixture of the framework. A part's rig layers a PMX cannot hold (fold shapes, cloth copies) go in
`Part.rig` and reach rig.json through the build ([Model description](#model-description-rigjson)).

**Assembly** (`mkmmd.model.assemble`, `pmx_io`): bones go parents first (grant parents too); a PMX vertex per unique
(position, UV, normal); custom `Mesh.normals` as given, else angle-weighted normals split along `Mesh.sharp` edges (a
hard edge between coplanar faces costs nothing); quads split along the shorter diagonal, ngons by ear clipping;
positions `(x, y, z) -> (x, z, y) / 0.08` so a `[[cast]]` (import scale 0.08) gives back metres; PMX 2.0 (all mmd_tools reads),
UTF-16, textures as `tex/<file>`, display frames Root, 表情 (every morph, ordered eye, brow, mouth, other), the parts'
frames merged by name, the rest in その他.

**Verification** (`mkmmd.blender.model.verify`, op `model_finish`): the PMX is imported unclean (vertex order = file),
then compared with the assembled arrays: vertex positions, bone heads and parents, weights, UVs, corner normals, winding
against the normals, faces per material, morph offsets (0.1 mm each), bone flags, grants, fixed and local axes, IK,
materials (colours, edge, textures), morph panels and English names, display frames, rigid bodies (pose, size, mode,
groups, masks) and joints (pose, limits, springs). Anything beyond tolerance is listed under `verify.problems` and the
command exits 1.

**CLI.** `mk model build SPEC [--only PARTS] [--out DIR] [--no-export] [--no-blend] [--no-verify] [--set K=V] [--full] [--no-cache]`
prints JSON: per-part numbers (meshes, vertices, faces, bones, materials, morphs, bodies, joints), `warnings` (lint:
unweighted vertices, missing UVs, unused vertices; the parts' findings, `[hair] WARNING ...`; spec keys no part read,
`[spec] WARNING ...`), the assembled model's counts, timings, `verify`, and `rig`: required
semantic bones missing, morph map (semantic -> morph), chain families with bone counts, bodies and measurements. Files in
the output folder: `<name>.pmx`, `tex/*.png`, `<name>.vrm` (mk's own model file, VRM 1.0: [VRM](#vrm-mk-model-vrm); written
right after the PMX, without Blender; `[vrm] enabled = false` leaves it out, `--only` builds write none),
`<name>.blend` (studio lights; `mk look <name>.blend --view front,3q --target "bone('head').head" --dist 1.2` works),
`<name>.rig.json` (what `mk inspect` writes; pass it as `rig =` to a `[[cast]]` with `pmx =`), `<name>.cloth.npz` (the
arrays of rig.json's `cloth`, when a part makes a cloth copy), `build.json`. `--only` builds those parts and what they
need into `<out>/only_<parts>/`;
`--no-export` only runs and checks the builders, into `<out>/no_export/` (an exported model's files stay as they are).
**Part cache** (`mkmmd.model.partcache`, in `~/.cache/mk/model_parts`): a part is reused, with its textures and its log
lines, while all it read is unchanged: the code (the package's .py files and the bases' files, by size and time), its
builder (module, name and file: a project's own builder too), the seed, the parts built before it (chained, so a part
built again builds every part after it again) and what it read from the spec (the values of its keys, a file a value
names by its size and time, the keys it looked up that were not set, the key sets of the tables it listed). The report
marks each part `cached`; `--no-cache` builds all of them again.
`mk model info SPEC` shows the plan; `mk model new NAME [--from BASE] [--dir DIR] [--out DIR]` writes DIR/model.toml, a
character that includes the base (default `girl`), and never overwrites one; `mk model studio SCENE.blend --out OUT.blend
[--floor X,Y ...] [--lights]` copies a built scene with the neutral review studio (grey world, soft floor discs,
optionally key/fill/rim suns) so sheets of several models compare side by side.
`mk model lab MODEL... [--region body|head|hand|foot|arm|leg] [--side L|R] [--views V,..] [--poses P,..] [--parts PARTS]
[--set K=V] [--label NAME] [--out SHEET.png]` (`mkmmd.model.lab`, no Blender) looks at specs (built and assembled
in-process, so it sees the PMX a build writes) and .pmx files: a sheet with one row per (pose, model), every cell at one
scale (bodies on one floor line), posed with each model's own weights through the bone tree (rotation grants followed:
D bones, twists); poses are rest, relaxed, curled, fist, spread, arms_down, tpose (VRM's T-pose: `lab.tpose_rotations`,
[VRM](#vrm-mk-model-vrm)), sit,
or any morph name (`--no-cache`: build every part of a spec again; see the part cache). Under the cells, the
rest pose's numbers in mm (heights, widths, lengths, and girths cut across the skin: the innermost closed loop round a
point on the skeleton, hidden skin included, so clothes never count; an open cut through a skirt or a frill encloses
nothing). A spec builds the body only unless `--parts` says more, so its region boxes (the head's height, width and
depth) lack the hair a .pmx carries. Beside the PNG: `.json` (each cell's camera, the numbers)
and `.mask.png` (coverage). `mk model trace MARKED.png --sheet SHEET.png` reads a pure red line drawn on the sheet, or on
a screenshot of part of it (any zoom, window borders), back in model space: the cell, and how far inside (+) or outside
(-) the outline the line runs every 2 mm, with its place along the region (from the wrist, from the floor).
`mk model glb MODEL [--pose tpose] [--morph NAME[=W]] [--region R --side S] [--parts PARTS] [--set K=V] [--out F.glb]`
(`mkmmd.model.glb`, no Blender) writes a .pmx or a spec, posed with its own weights (T-pose by default), as one binary
glTF with its textures inside: what Tern's 3D block turns, pans and zooms (it opens OBJ, PLY, STL, glTF, FBX, USD and
3DS, not PMX). Model space becomes glTF's (+Y up, +Z forward); PMX's clockwise faces are written reversed; each drawn
material is a primitive with its texture (PNG), BLEND when its texture or diffuse alpha is see-through, two-sided when
the PMX says so; `--region head` writes the head's subtree alone, so a viewer frames the face. It prints the model's
numbers as JSON (names, height, counts, morphs by panel, the poses its bones allow; `--info` writes nothing). Exit
codes as everywhere: 1 when a check or verification fails. A build never deletes the previous files first, because other
projects may be casting the PMX at that moment: the PMX, the VRM and rig.json are replaced atomically when ready, textures
are overwritten in place (stale ones removed after the PMX is written), `.mk/build.lock` makes two builds into one folder
take turns, and `build.json` reads `{"ok": false, "stage": "running"}` until the build ends (failures are kept there too).

Blender note: a Blender session that resets the add-on preferences (a script started with `--factory-startup` that
touches add-ons) can delete mmd_tools' bundled opencc wheel, and every PMX import then fails with "bpy.ops.mmd_tools.
import_model could not be found"; use another config folder (`BLENDER_USER_CONFIG`) for such scripts. `mk doctor --fix`
restores the wheel and `mk model build` repairs and retries on its own.

## VRM (`mk model vrm`)

mk's own model file is VRM 1.0 (glTF 2.0 with the VRM consortium's extensions), written beside the PMX: `mk model build`
writes `<name>.vrm` right after the PMX, and `mk model vrm MODEL [--out F.vrm] [--set K=V] [--unit U]` writes one for any
.pmx (with the rig.json and .cloth.npz beside it, when there are, for the folds and the cloth copy) or a spec, built
in-process. `mkmmd.model.vrm` (numpy, scipy, Pillow; no Blender) reads the PMX as the lab does and writes one file,
textures inside. Any VRM app reads it; the review page shows a .vrm as it is (its own viewer: the mesh in its rest pose).

- **Rest pose**: VRM's T-pose (`lab.tpose_rotations`, VRM 1.0 tpose.md): each upper arm, forearm and hand straight out
  along X and level, the palm down (a roll the hand needs goes to the forearm's twist bone, so the wrist does not wring),
  the four fingers straight along the hand, the thumb straight and level 45 degrees forward with its nail turned forward
  and in, the feet turned to point straight forward. The model is posed with its own weights and grants and that pose is
  the rest: no node keeps a rotation, the vertices, normals and morph offsets are the posed ones (an offset turned by its
  vertex's blended rotation), an IK bone sits where its target went. glTF's axes: +Y up, facing +Z, her left +X.
- **Humanoid**: the semantic bones (core.bonemap) under VRM's names; the hips are the lowest bone both the spine and the
  legs hang from (MMD's 腰), the legs' D bones in place of the legs' when a model has them; a bone that breaks VRM's
  order (it must hang from the humanoid bone VRM puts above it) is left out with a `[vrm]` warning.
- **Expressions**: VRM's presets from MMD's standard morphs by semantic name (`vrm.PRESETS`, core.bonemap.MORPHS: happy
  笑い + にこり + 口角上げ, angry, sad, relaxed, surprised, aa..oh あいうえお, blink まばたき, blinkLeft ウィンク and
  blinkRight ウィンク右); every other vertex and group morph a custom expression of its PMX name; a material morph's
  colour and edge as material colour binds. Fold shapes are in none (MK_rig drives them). Every vertex morph is a morph
  target (sparse), named in `mesh.extras.targetNames`.
- **Look-at**: the eye bones, from between the eyes, `[vrm.look]` degrees at most each way; **first person**: auto.
- **Springs** (VRMC_springBone): a spring per physics chain (the bones dynamic rigid bodies move, from a root hanging
  from one they do not; a branch is its own spring), ended by a node at its last bone's tail so the last bone swings
  too; stiffness, gravity and drag by the chain's family (core.families, `[vrm.springs.<family>]`), each joint's hit
  radius its rigid body's. Colliders: the static rigid bodies on their bones (a box becomes the capsule along its longest
  side), one collider group per PMX collision group; a spring collides with the groups its bodies do. VRM springs cannot
  tie chains sideways (MK_rig keeps the ties).
- **Constraints** (VRMC_node_constraint): a rotation constraint per PMX rotation grant onto a bone the humanoid does not
  drive (a helper bone at the elbow, an eye highlight), weight its ratio; a grant from both eyes (両目) comes from the
  eye on the bone's side, which VRM's look-at turns. Grants onto humanoid bones, moving grants and ratios outside 0..1
  are MK_rig's.
- **Materials** (VRMC_materials_mtoon): MMD's lit colour saturate(ambient + 0.6 x diffuse) (its default light) times
  the texture; the shade that times the toon ramp's darkest row, its step where the ramp's is (MMD reads row
  0.5 - 0.5 N.L of the ramp: a step v up from its bottom, w wide between 10 and 90 %, is shadingShift 1 - 2v and
  shadingToony 1 - 1.25w); flat without a toon ramp; an additive sphere map the matcap; the edge the outline (screen
  ratio `[vrm] outline` per unit of edge size, its colour, unlit); BLEND with depth writes where the texture or alpha is
  see-through, as MMD draws. A shared toon (toon01..10) is approximated.
- **MK_rig** (mk's extension, `extensions.MK_rig`): what VRM cannot hold, nodes by index, positions in the T-pose in
  glTF's axes: `source` (the PMX's names and comment), `rest.pmx_rest` (each turned bone's local rotation, as a
  quaternion, that takes the VRM's rest back to the PMX's rest pose), `bones` (PMX names, English names, tails, flags,
  fixed and local axes), `ik`, `grants` (every PMX grant, why one is not a constraint), `morphs` (every PMX morph: its
  target, or a bone, UV, flip or impulse morph's offsets), `materials` (the PMX material numbers), `physics` (every rigid
  body and joint), `chains` (each spring's family and the springs it is tied to sideways), `folds` (rig.json's fold
  shapes with their morph targets, the drivers' directions turned into the T-pose with their reference bone, so they
  measure the same angles), `cloth` (each cloth copy, its points skinned into the T-pose by their own weights, its
  arrays as buffer views in the binary chunk), `frames` (the display frames).

`[vrm]` (all optional; an unknown key is an error, and mk model build reads the table itself, so the parts' unread-key
report leaves it alone):

| Key | Meaning |
|---|---|
| `enabled` | false: no .vrm |
| `[vrm.meta]` | VRM's meta: `name` (default the model's), `version`, `authors` (default `["unknown"]`), `copyrightInformation`, `contactInformation`, `references`, `licenseUrl` (the VRM Public License 1.0), and the permissions, most restrictive by default: `avatarPermission` onlyAuthor, `commercialUsage` personalNonProfit, `creditNotation` required, `modification` prohibited, `allowRedistribution` and the `allow*Usage` keys false |
| `outline` | MToon outline width per unit of PMX edge size, a ratio of the screen height (0.0016) |
| `[vrm.look]` | `inner`, `outer`, `up`, `down`: degrees the eyes turn at most (10) |
| `[vrm.springs.<family>]` | `stiffness`, `gravity`, `drag` of a chain family's springs (`vrm.SPRINGS`) |
| `[vrm.expressions]` | `<preset> = [[morph, weight], ...]`: a PMX or semantic morph name, in place of `vrm.PRESETS`' |

Checked with the Khronos glTF validator (no errors, no warnings) and the VRM schemas, and loaded by three-vrm (the
reference runtime) in Chromium: humanoid, expressions, springs swinging and settling, constraints, MToon.

## Briefs (`mk brief`)

A video starts from a brief: what the person wants from it, written before anything is built, which the agent answers
with a treatment (a review, below; docs/AGENTS.md: Brief). `mk brief new DIR --song FILE [--lyrics FILE] [--name NAME]
[--where right|down|tab] [--no-open]` makes the project: the song copied into `audio/`, an mk.toml whose clip is the
whole song (30 fps, `frame0` 91 after 3 s of pre-roll), an empty `brief/brief.json`, and `audio/lyrics.txt` when lyrics
are given; a folder with an mk.toml is refused. Then it opens the project's page (below) on its Brief tab beside the
agent's pane; `mk site --tab brief` opens it again. `mkmmd/brief.py` has the format and the checks.

The Brief tab (`web/js/brief.js`) has three cards:

- **The song**, on one canvas where every lane has the same scale, in song seconds: the part lane, where the part of
  the song is a bar with a handle at each end (dragging a handle resizes it, snapping to lines and bars; dragging the
  bar moves it; a drag on the empty lane draws a new one; a click listens there; each change says what the part is
  now, with Undo, and "use the whole song" takes it all); under it the time ruler and the waveform, which only move the
  playhead (a click or a drag scrubs), dimmed outside the part; then once the words are timed the bars and the lines by
  number (the words as ticks, never their text), and the parts lane. The part is the project's clip: `[audio] start`
  and `[project] duration` are rewritten in mk.toml at once, in place, every other line as it was (`PUT clip`; refused
  while the words are being timed, and on a project with shots unless the person moves the music under them). Nothing
  zooms by itself: **Song** shows the whole song, **Part** the part (the page opens on it), Ctrl+wheel zooms; a drag
  held at the edge of the view pans it, and a handle let go past the edge stays in sight. The lyrics are pasted once,
  one sung line per line, and kept as `audio/lyrics.txt` for `mk timeline analyze --lyrics`: the page says how many
  lines it has and never shows them again (`PUT lyrics`). Sung in English (the words are timed) or another language
  (the bars only). **Time the words** runs `mk timeline analyze` on the clip in the background (`POST words`: with the
  lyrics when there are, `--no-words` for another language; its output in `.mk/words.json` and `.mk/words.log`) and
  shows its stage and time until the lines and bars appear; a timeline of another stretch than the clip is called
  stale.
- **Part by part**: a click on a line writes about it (the part that holds it, else a new part over it; Shift reaches
  the selected part to it); a drag across lines, or along the parts lane over any stretch (an intro, a solo), makes a
  part; a part's edges drag (a drag held at the edge of the view pans it). Parts never overlap. Each has a card:
  its lines and times, ▶ to play it, a note (what the person wants there, as loose or as exact as they like), pictures
  (added, dropped or pasted: `POST picture`, kept as `brief/pictures/<hash>.<ext>`, PNG, JPEG, WebP or GIF), and after
  the first, the earlier part it comes back from (a chorus), the same or bigger. A click on a word names it, and a
  button puts `(line, word)` into the selected part's note. Deleting a part offers Undo (its note, pictures and the
  parts that came back from it come back with it). A part the clip no longer holds (the part of the song changed after
  it was written) is kept and marked.
- **The video as a whole**: the idea, who is in it, where, the look, where it is posted, the sizes (9:16, 1:1, 16:9),
  pictures.

It is saved as it changes (`PUT brief`, checked whole like a review's answers: parts inside the song that do not
overlap, a part coming back from one before it, known sizes, pictures in `brief/pictures/`). **Send** (in the dock)
gives it the time and types it into the pane of the agent that opened the page, as a review's Send does: the part of
the song, the video, each part with its lines, times and note, on one line that ends with `mk brief show` when it is
long. `brief/brief.json`: `version`, `language` (`en` or `other`), `video` (`story`, `cast`, `place`, `look`,
`posting`, `sizes`, `pictures`), `parts` (`id`, `from` and `to` in song seconds, `note`, `pictures`, `echo` {`of`,
`how`: `same` or `bigger`}), `saved`, `sent`.

`mk brief show` prints what the agent reads: the song and its length; the clip (song seconds, whether it is the whole
song); the lyrics' line count; the language; the timeline's numbers (bpm, beats, bars, lines, words, `words_source`,
`matches_clip`); the video (pictures as paths); each part with its song and clip seconds, `in_clip`, the lines, the
first and last words `(line, word)` and the bars it covers (bar 1 starts at the first downbeat, 0 is a pickup), its
note, pictures and `echo` (the part's number and how); and `open`, the stretches of the clip (half a second or more)
no part covers. Numbers and the person's words, never lyric text.

## Reviews (`mk review`)

A review puts decisions to a person with pictures, choices and models, and gets the answers back to the agent that asked.
`NAME.review.toml` (`mkmmd/review.py` has the format): `[review]` title, text (Markdown) and preselect; `[[question]]` id,
ask, text, images, models, music, recommended, tag; `[[question.option]]` id, label, text, images, models. With
`preselect = true` a question the answers file has nothing for starts on its recommended option, so Send alone takes
the agent's picks (a treatment); an answer the person gave stays theirs. A tag is a short label beside the question
("my idea": the agent's own ideas in a treatment), read back with the answers. Images are paths beside the
review (PNG, JPEG, WebP, GIF; others become previews); a model is `{label, file}` (a .glb or a .vrm, shown as it is; a
model file Blender imports; or a .pmx) or `{label, spec, parts, set}`, with pose (default tpose), morph and region
applied by `mk model glb`; music is a project folder or `{project, from, to}`: the project's music timeline (below)
under the question, over that span of the clip. `mk review check` names every problem at once; `mk review open
NAME.review.toml [--where right|down|tab]
[--wait [--timeout S]]` checks it, starts (or keeps) `NAME.answers.json` and opens it as a page of the site (below), with
`--wait` returning when the person presses Send, with the answers; `mk review answers` prints the answers with their
options' labels, what is still open, the marks and the views kept from the 3D viewer.

The page shows each question as a card: its text, its music, its pictures (one fits the width; several sit whole in a
strip that scrolls sideways), its models (a button opens the viewer), its options with a radio and a notes field. Keys: j/k
move between questions, 1-9 choose, n writes a note, Esc leaves a field, ? shows them. Choices and notes are saved as
they change (`PUT answers`, checked: an answer that does not fit the review is refused whole). A picture's pen button
opens it to draw on, filling the pane: pen, arrow, ellipse, box, text and eraser in eight colours, undo and redo, zoom
(wheel, pinch, + and -, 0 fits) and pan (space-drag, right-drag, two fingers). **Done** saves the marks in the picture's
own pixels (`NAME.marks.json`: line and arrow by their ends, box and ellipse by their box, ink by its points, text with
its words) and draws each marked picture for the agent (`.NAME.marks/`). On a lab sheet (its `.json` beside it) `mk
review answers` turns them into model space with the lab's `trace_points`: a line's or a stroke's length and how far
inside or outside the outline it runs, a box's size in mm. **Send** writes the message the agent reads into the answers
file (with the time) and gets it to the agent: a waiting `mk review open --wait` returns it (it tells the server it
waits, `mkmmd/site/show.py`); else, in Tern, the server types the message into the pane of the agent that opened the
page, as one line and Enter, as if the person had written it there. The agent is the last one to open the review's page
or its project's, known by its shell's environment (`AGENT`, `CLAUDECODE` or `OMPCODE`); a page a person opened from
their own shell has none, and there Send only saves and says so (`mk review answers` reads the answers).

## The site (`mk site`, `mkmmd/site`)

One page mk serves on this machine for reviews and a project's page. `mkmmd/site/server.py` is Python's standard
library on 127.0.0.1 at a free port; every URL starts with a random token (`/<token>/...`), a request must name the
server in its Host header (no DNS rebinding), and files come only from the folders a page was opened for (a review's
folder and the folders of its pictures and models, a project's root), the page's own files (`web/`) and the site cache
(`<cache>/site/`), with byte ranges for media. One server serves every page; `server.json` in the site cache (pid, port,
token) lets the next command find it, and it stops after three hours without a request (`mk site --stop` stops it).
`mk render` with `engine = "mk"` runs a server of its own on the same code for its page (`render`: the bake, the frames,
the passes coming back; [mk's own renderer](#mks-own-renderer)).
`api.py` holds the routes (`open`, `review`, `answers`, `marks`, `send`, `model`, `section`, `measures`, `snapshot`,
`project`, `music`, `scene`, `draft`, `checks`, and the brief's `brief`, `clip`, `lyrics`, `picture`, `words`,
`send_brief`). `show.py` opens a page: in Tern (when `$TERN_PANE` is set) as a browser block docked beside the agent's
pane (`--where right|down|tab`), else in the system's browser. `mk site [--tab brief|reviews|music|scene|checks]
[--where ...] [--url] [--stop]` opens the project's page: its brief ([Briefs](#briefs-mk-brief)), its reviews (each with
its title, how many questions are answered and whether it was sent), its music when it has an [audio] file or a
timeline, its scene when `mk build` has baked it, and its checks when it has `[[check]]` entries. It opens on the brief
while one is written and not sent, else on the reviews. A review opens inside that page,
under its tabs (‹ Reviews or the Reviews tab goes back to the list, any other tab leaves it, its answers saved), and so
does a review `mk review open` opens when it lies in a project (an mk.toml above it); one in no project opens alone.

The music timeline (`mkmmd/site/music.py` builds it, `web/js/music.js` draws it) puts a project's clip on one canvas, in
clip seconds: the clip's audio (the [audio] file cut from `start` for `duration` by ffmpeg, a WAV in the site cache;
without ffmpeg the whole file plays from `start`) played through Web Audio over its waveform, then lanes for the beats
(bars numbered at the downbeats), each list of the project's `[audio] hits` and each stem of the timeline's `onsets`,
the words by `(line, word)` (the page never gets their text), the shots of the cut, and the effects packed into rows
(transitions, inserts, glitches, frame effects, medium shifts, freezes, rings, and texts while they are up). Space plays, a click or a drag moves the
playhead (a click on a hit, word, shot or effect jumps to its start and names it, a shot with its frames), a drag along
the ruler loops that span, Ctrl+wheel or a pinch zooms, a sideways wheel pans, a double-click fits; ← and → step a beat,
l loops, f fits, Home goes back. The clock reads clip seconds, the Blender frame and bar.beat. What the project gets
wrong (a timeline that is missing, a cut that does not plan) is listed under it and the rest is still drawn.

The scene (`web/js/scene/`) plays the bake `mk build` writes beside the .blend (`<name>.bake/`, the bake stage in
[Building](#building); `mkmmd/blender/build/bake.py` has its format) to the music timeline under it, which is its clock.
3D draws it with WebGL2: each mesh placed by its node matrix on the frame, skinned on the GPU from a texture of its
joints' matrices, its morph targets added on the CPU when their weights change, and what the glb cannot carry (curves
such as a mic's cable, meshes with modifiers besides the armature) as Blender evaluated it on that frame; shaded (base
colour and texture) or grey; Shot looks through the cut's camera for the chosen output as Blender frames it (the
picture's aspect, the lens on the sensor fitted as the camera says, the shift), Free orbits. Draft shows mk post's draft of that output
(`out/<name>_<output>_draft.mp4`) on the same clock, decoded with WebCodecs (Tern's browser block plays no `<video>`):
the server's `draft` route (`mkmmd/site/draft.py`) gives the decoder its avcC record and every sample's offset, size,
time and keyframe flag in decode order (ffprobe), and the page decodes from the keyframe before a jump. The label names
the shot, the Blender frame and, in a freeze, the frame the world holds; the page says when the .blend is newer than the
bake (`mk build --bake` bakes a saved scene again).

The checks (`web/js/checks.js`) are cards, failing ones first: the check's name, its metric with what it measures in
plain words, its value against its rule, and its worst moment (who, what, when: the cast member, the side, the move or
the bone the detail names, the frame and the clip second) with a picture of that moment drawn from the bake through the
cut's camera of the first output; Look opens the scene there. They read the last results `mk check` keeps in
`.mk/checks.json` (a check never run says so); Run checks runs `mk check` in the background (`POST checks`) and the
page follows it until it ends. The music timeline marks each check's worst moment on a lane of its own, red when it
fails.

The page (`web/`) is plain JavaScript modules and one stylesheet, no build step and no library, laid out for a narrow
pane (about 560 px) and following the host's light or dark theme. Its dropdowns are its own (`dropdown` in `ui.js`):
Tern's browser block shows no native popup, so a `<select>`'s list never opens there. Its 3D viewer (`web/js/viewer/`)
is a WebGL2 renderer of the .glb files `mk model glb` writes: a model file that is not a .glb is turned into one by
Blender (`mkmmd/blender/ops_site.py`), a .pmx or a spec is posed by `mk model glb`, each cached in the site cache by its
command and its sources (a spec's .toml files and those of the bases it includes). Views: Shaded (its textures), Grey, Lines
(edges over grey; open edges and flipped faces in colour), Silhouette, X-ray and Parts (a colour per material; tap one to
hide or solo it). The camera turns, pans and zooms (double-click zooms to a spot), with front, side, back and top
buttons and Blender's numpad keys (1, 3, 7, Ctrl for the opposite side, 5 ortho); f fits, r turns it on a turntable, v
steps through the views. **Cut** drags a plane through the model and reads each loop's girth (the lab's
`sections`; on .pmx and spec models it also jumps to the lab's named girths); **Measure** reads the distance between two
points in mm; **Compare** shows a second model of the review beside the first, or over it (Tab), in the same pose and
cut; **Pose** switches a .pmx or spec model between T-pose, rest, arms down and sit; **Mark** keeps the view as a picture
to draw on (`.NAME.views/`, with its model and camera) under its question.

In Tern, the mk plugin (`tern/`: `tern plugin link tern/`) opens a `NAME.review.toml` from Tern's Files pane as the
page, in place of the pane, and opens any `.pmx` as a PMX block: its names and numbers, chips for the poses its bones
allow and for its morphs by panel, the model posed in Tern's 3D preview (T-pose first), and a lab sheet of the pose. A
page in Tern's browser block cannot reach the plugin (its title and URL changes raise no event, and `mk://` links do not
navigate), so Send goes through the server.


## Exporting the page (`mk export web`)

`mk export web [--out DIR] [--outputs 9x16,1x1] [--video final|draft|none] [--side PX] [--crf N] [--texture-side PX]`
(`mkmmd/site/export.py`) writes the project's scene and music as a folder of static files any web server can host (default
`out/web`): the site's scene view and its music timeline from the site's own modules, with `web/js/api_static.js` as
`js/api.js` so every route reads a file under `data/` instead of the server. `export.html` and `export.css` lay the view
out as a document a host page can embed: `?theme=light|dark|mono` (Rosé Pine Dawn, Rosé Pine Moon, black and white;
without it the system's), `?view=3d|draft` and `?maxh=PX` for the stage's height; in an iframe it posts its height and its
clock (`{source: "mk-export", type: "height"|"clock"}`) to its parent and takes `{source: "mk-host", type: "theme"}`.

What it writes: `scene.json` and `music.json` (what the `scene` and `music` routes answer, with the files' URLs and no
local path; the music carries the checks' worst moments), `bake.json`, `bake.mkz`, `scene.glb.gz` (the textures as WebP
no larger than `--texture-side`), each output's video (mk post's final, else its draft; H.264 at most `--side` px, its
sound) with `draft_<output>.json` for the Video view, `audio.m4a`, `CREDITS.md` (as `mk assets credits` writes it),
`licences/<asset>/` (each asset's readme files) and `manifest.json`. The page loads `bake.mkz` and `scene.glb.gz` only
when 3D is first shown, behind a button that names their size. Nothing is written while any asset the project uses is
unreviewed or its restrictions forbid redistribution (exit 1), since its meshes and textures leave this machine; a
folder `mk export web` did not write is never written over.

`bake.mkz` is `bake.bin` packed: `MKZ1`, a JSON header's length and the header (`floats` and one entry per stream: `at`,
`rows`, `cols`, `mode`), then the gzipped body. A stream is a block the bake's description names (an armature's joints,
a node's matrices, a camera, morph weights, a cable's vertices, its triangles, a ring's points). Each number is rounded
to the stream's step (`STEP`, `CABLE_STEP`: 5e-5 m for joints and nodes, 1e-4 for weights, 1/256 for normals, 1 for
indices and visibility, which come back exact), stored as its change from the frame before (`delta`) or from the
geometry it shares a triangle list with (`ref`), zigzagged and split into four byte planes. A block with a NaN or a
number past the grid, and the floats between the named blocks, stay float32 (`raw`).
