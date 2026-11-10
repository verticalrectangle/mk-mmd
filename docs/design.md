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
- [The GPU lock](#the-gpu-lock)
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
- Every Blender the CLI starts (a job, `mk serve`, and a build's breathe probe) runs in a transient systemd scope of its
  own (`systemd-run --user --scope`, `mkmmd/core/jobscope.py`) with its memory capped at `blender_memory` of the RAM
  (0.6). Otherwise a Blender runs in the cgroup of whatever started mk, often a terminal app's systemd service, and when
  the memory runs out the kernel kills the biggest process, a Blender, and the service's default `OOMPolicy=stop` stops
  the whole service: the terminal closes with the job. In a scope of its own an overflow kills that job alone, and the
  command says so (`Blender 'build' was killed: most likely the memory ran out ...`). Without systemd-run or a user
  manager, or with `blender_memory = 0`, Blender runs as it is.
- `mk serve SCENE.blend` keeps one Blender alive on a Unix socket. The bridge uses it automatically for the ops that
  leave the scene as they found it (`ping`, `list`, `q`, `sample`, `visibility`, `hand_model`), which matters for big files:
  `mk q`, `mk check` and `mk grip` answer in under a second. Ops that change scene state (`look` hides objects, removes markers,
  adds a camera; `build`; `render_frames`) always get a fresh Blender.
- `sample` is how numbers leave Blender: one pass over a set of frames writes posed bone and object matrices, rest
  matrices, expression values, the active camera, resolved colliders, evaluated meshes (world-space triangles of what
  a render shows, for [`form`](#form)) and a character's skinned limbs on every frame, with their rest shape (for
  `skin_collapse`) to an `.npz`; checks and solvers work on that.

## The GPU lock

One GPU job at a time (`mkmmd/gpulock.py`; in Rust the crate `mkr/mkr-gpulock`). The card's memory is shared by every job on
it, and what does not fit spills into system memory, which cannot be swapped: two or three renders at once filled the 12 GB
card, spilled 10 GB into RAM and got processes killed. So every job that draws on the GPU holds one lock while it runs and the
others wait their turn, saying once on stderr who has it (`waiting for the GPU, held by <pid> <command>`):

- `mkr` and `mkr-post` take it when they open the GPU (`Gpu::new`) and keep it until they exit;
- the bridge's render jobs (`render_frames`, `look`, `lab_build`, `lab_variant`) hold it while Blender runs;
- `mk render`, and `mk post` when it uses the GPU (`--gpu` or the vaapi encoder), hold it for the whole command, so what
  they start (mkr, mkr-post, Blender jobs, the encoder) runs inside their turn;
- anything else that uses the GPU (a Blender script run by hand, GPU tests such as `cargo test` in `mkr/`, a scratch mkr
  build) goes through `mk gpu run [--what TEXT] -- CMD [ARG...]`: the command and everything it starts hold the lock until
  the last of them exits.

The lock is an advisory `flock` on `$MK_GPU_LOCK`, else `<cache>/gpu.lock` (`$MK_CACHE`, default `~/.cache/mk`); the
kernel releases it however its holder ends, so a crashed or killed job never leaves it held. The holder writes its pid and
command into the file. Jobs are served first come, first served: one that finds the GPU busy, or others already waiting,
puts a ticket `<ns>-<pid>` into `gpu.lock.queue/` beside the lock file and waits until its ticket is the oldest of those
whose process still runs (`waiting for the GPU, held by <pid> <command> (N in line)`); a finished or killed job's ticket is
dropped by whoever sees it. A holder sets `MK_GPU_LOCK_HELD=1` for what it starts, and anything that finds it set skips
the lock (a job inside a job does not wait for itself). Hold it only while the GPU works and for minutes, not hours: a
loop of renders takes it per render (`mk parity tune` per evaluation). `mk gpu status` prints who holds it, the card's
memory in use and spilled into system memory (`spilled_mb`, the GTT pool), and the processes using GPU memory, largest
first.

## Conventions

- Units are metres, Z is up. A character faces -Y unless a seat or placement says otherwise. MMD models import at
  scale 0.08 (one MMD unit = 8 cm).
- Time: `frame(t) = frame0 + t * fps`, fractional. Clip time 0 is `[project] frame0`; earlier frames are pre-roll (the hair
  settles there). Every time in `mk.toml` is in clip seconds unless a key says song seconds (`[audio] start`).
- Output: JSON on stdout (UTF-8, Japanese names kept). Images are written to files and their paths printed.
- Exit codes: 0 ok, 1 a check, gate or verification failed, 2 usage or configuration error, 3 Blender or runtime error.
- Lyrics: the text lives in a project's local files (lyrics, timelines, mk.toml); the tools read it freely to align,
  render and check type, and logs may show it. The repository (code, tests, docs examples, commits) refers to words as
  `(line, word)` indices, numbered from 1, and carries no lyric text.
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
| `cam()`, `screen(point[, camera])` | the active camera as an `obj` view; a world point to camera-view coordinates `(x, y, depth)` (x, y in 0..1 from the bottom left of the frame, depth in metres), a shot camera seen at its own output's size; through a fisheye ([Fisheye shots](#fisheye-shots)) the place in its picture and, as depth, the distance from the camera, negative past its field |
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
| `vaapi_device` | `MK_VAAPI_DEVICE` | `/dev/dri/renderD128`: the GPU `mk post --encoder vaapi` encodes on |
| `blender_memory` | `MK_BLENDER_MEMORY` | `0.6`: the share of the RAM one Blender job may hold, in a systemd scope of its own ([The bridge](#the-bridge)); `0` or `"off"`: no scope |

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
| `[[fx]]` | `mk render`, `mk post`, `mk look` | frame effects on the finished pictures (lens, echo, slitscan, push, scope, zoom, shock, slab, tint, halftone, split, mosh, the collages, the picture effects pixelsort, film, camcorder, doodle, booth, the haze: mist, fog, bokeh, and mitosis): see [Frame effects](#frame-effects) |
| `[[medium]]` | `mk build` (shots), `mk render`, `mk look` | a window of clip seconds drawn in another look whatever shot is in the cut: see [Medium shifts](#medium-shifts) |
| `[credits]` | `mk assets credits` | `assets` (slugs the credits must name besides the cast's), `lines` (the project's own lines) |
| `[colliders]` | sim, checks | named lists of collision shapes: see [Colliders](#colliders) |
| `[[check]]` | `mk check` | `name`, `metric`, `args`, `frames`, `min`, `max`: see [Checks](#checks) |
| `[[set]]`, `[[prop]]`, `[[scatter]]`, `[[vehicle]]`, `[[cast]]`, `[pose.<cast>]`, `[[motion.<cast>]]`, `[perform.<cast>]`, `[[ribbon]]`, `[[shot]]`, `[[transition]]`, `[[insert]]`, `[[glitch]]`, `[[light]]`, `[[text]]`, `[[key]]`, `[sim.<cast>]`, `[[stepped]]` | `mk build` | one build stage each: see [Building](#building) |

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
footage/pexels_<id>.mp4 stock footage a footage frame effect shows (mk ref footage)
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
`steering` (`{object, axis, ratio, wheelbase}`), which the [vehicles stage](#vehicles) reads. A card's `rates` (`[{rate, into, per}]`, the
`turntable`'s) are root properties the [keys stage](#keys) integrates, and its `deals` (`[{from, into}]`, the `confetti`'s)
root properties whose keyframes the keys stage deals out in turn: see there.

## Asset registry

`<assets>/registry.json`: a list of entries, one per model, motion, prop, vehicle, audio file, reference clip, texture or font
(`kind`: model, motion, prop, vehicle, audio, reference, texture, font, other). `mk assets` edits it.

```jsonc
{"slug": "my_model", "kind": "model", "path": "/abs/path/model.pmx", "name": "Rin", "author": "NAME",
 "author_url": "...", "source_url": "...", "license": "...", "restrictions": "...", "credit": "Model: NAME",
 "rig": "rigs/my_model.rig.json", "tags": ["original"], "readmes": ["/abs/path/readme.txt"], "extra": {...}}
```

mk never guesses a license. New entries are `unreviewed` and keep the paths of the readme files found next to the
asset (`mk assets show SLUG --terms` prints the lines that talk about terms of use); someone who has read them fills in
`license`, `restrictions` and `credit` with `mk assets set` (`source_url` is the field `--url` fills; `extra='{...}'` merges a
JSON object over the entry's `extra`, a `null` removing its key: a sprite's googly-eye anchors, see Googly eyes). `mk assets credits`
builds the credits from the project's cast, the videos its footage frame effects and split and iris inserts show, the sprites
its frame effects and inserts show (every `sprites` of an `[[fx]]` or `[[insert]]` entry, at any depth: a mandala ring's, a
stack's band, a paint app's stamps), the photographs its `prints` show (every `photos` and `base` photo, aspect tables too) and
the photos its `photo_world` sets are made of (a slug, or a file the registry has) and
`[credits] assets` (one entry per asset, grouped by kind, a credit line two of them share printed once) and exits 1 while any
of them is unreviewed or has neither an author nor a credit line; `[credits] lines` are the project's own lines (a song, a print
made for it, the palette, tools), appended as bullets under
`## Also` or under the `## Heading` lines among them. Slugs are how the project refers to registered assets: `[[cast]] asset`,
`[[motion.<cast>]] vmd`, `[[prop]] card`, `[[text]] font`, `[[fx]] clip`, `[[insert]] clip`, `[[set]] image` (a `photo_world`'s
photo) (`mk ref footage` registers Pexels videos with their author, pages and licence).

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
every pair the PMX collision masks exclude. Any spec may name the chain `families` it acts on (a list; default all of
them): a worn guitar's strap is capsules for the hair families alone, since hair lies over a strap and a skirt is under it.

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
| `foot_slide` | 95th percentile horizontal speed of planted feet (mm/frame) | between consecutive frames only: `frames` with gaps (a dance's stops) are measured run by run. An ankle is judged by what of its foot is down: the point of the sole under the ankle, or the ball of the foot (the toes' joint, where the model has that bone), so a heel lifting about planted toes is not a slide |
| `ground` | how deep (mm) the skin of a character's feet goes under the surface of what it stands on, the deepest vertex over the frames | the skin its ankle bones carry (the toes with them), read straight down at every vertex against a set's or a prop's geometry as a render shows it: a studio sweep's floor and the cove it curves up into, a stage, a slope. A step back into a cove counts as what it is; a surface more than `reach` over a vertex (a table top, a ceiling) is not its ground. A few millimetres is a sole pressed on the floor: default `max` 10; detail: `at_frame`, `foot`, `p95_mm`, `frames_over_10mm`, `worst_frames`, `ground` (the objects read) |
| `joint_limits` | worst excess over a limit (deg) | elbow fold-through and in-plane hyperextension, knees, wrists, neck, spine (limits calibrated on professional MMD dances; sideways elbow bends are not flagged: MMD rigs treat the elbow as a ball joint), and how far each arm turns about itself, by anatomy: `forearm_twist` and `arm_twist` (below); each series over its limit names its worst frame (`at_frame`) and up to ten `spans` of frames over it (`[first, last, worst, degrees]`) |
| `skin_collapse` | largest change of a limb's cross-section against its rest size, as a share | the skinned meshes of a character's upper arms, forearms, thighs and shins cut square to each segment from 30 to 80 % of its length, on every frame: a twist the rig's weights cannot follow pulls a sleeve in (below); default `max` 0.15; detail: `at_frame`, `kind` (`collapse` or `swell`), `region` (`forearm.L` ...), `bone` (the heaviest bone at the cut), `object`, `at` (the cut, as a fraction of the segment), `regions` (each one's worst `loss` and `swell`) |
| `framing` | smallest margin to each output's safe area | per output through the active camera; matches Blender's projection; fractions of the frame, negative = outside the safe area |
| `occlusion` | largest share of subject points hidden from the camera | ray casts; the subject's own meshes do not count |
| `camera_inside` | frames with the camera inside a closed mesh a render shows | hidden objects (colliders) and `ignore` (glow and haze volumes) are passed through |
| `flicker` | worst frame's 99th percentile of temporal luma noise | on rendered frames; motion makes it large too, so compare shots with themselves |
| `palette` | near-black share (or median distance to a palette) | on rendered frames; `rose-pine-moon`, `rose-pine`, `rose-pine-dawn` built in |
| `form` | boxiness 0..1 of a prop's evaluated geometry: area-weighted over loose parts, `1 - (1 - cuboid)(1 - flat_sharp)` | what a render shows at one frame, in the prop's frame; hero props stay under the default `max` 0.25; `detail.worst_parts` names the objects to fix (see [Form](#form)) |
| `strum` | largest distance (mm) of the pick tip from the nearest string at the strike of a down stroke | replays the project's `[perform.<cast>] strum` on its timeline and measures the scene: the pick object rides the wrist, so the arm's reach is in the numbers; `detail.timing` has the mean, median, p95 and max of when the tip really crosses the first string against the planned strike time (ms), `detail.missed` the strokes it never reached; default `max` 10 (see [Playing a worn guitar](#playing-a-worn-guitar)) |
| `string_play` | how many notes of a worn guitar-like prop are not played as planned (`max` 0) | replays the project's strum spec and fretting hand ([Strings that ring](#strings-that-ring)): per note the note's string rings from the note's fret on the first frame after it (`mk_amp` up, `mk_u0` the fret's place; in a line played one note at a time no other string starts with it), a fretted note's finger (the skin of its distal phalanx) lies on the string within `finger_mm` across and of the press window behind the wire and touches it (`touch_mm` from its surface), and the pick tip (or a fingerstyle stroke's fingertip skin) crosses the string within `time_ms` of the note, passing within `pick_mm` of its axis; detail: `notes`, `missed`, `worst` (each number's worst), `misses` (the first eight, `miss` naming silent, wrong_fret, other_string, no_finger, finger, no_strike or strike) |
| `prop_body` | deepest vertex (mm) of a prop's geometry inside the character's collision bodies | the torso, hips, legs, neck and head bodies of rig.json (not the arms that hold it), over the frames; the bodies are capsules a little fatter than the skin, so a few millimetres is a prop resting on the body; default `max` 8 |
| `clearance` | least gap (m) between the world bounds of objects that fly past (`objects`: names or fnmatch patterns, e.g. a [bullets](#lyrics) entry's `bul_l*`) and the character's collision bodies, over the frames each object has geometry | exact for the bodies' spheres, capsules and rounded boxes; negative: into a body (by at least that much: where the cores overlap it is minus the body's radius). The bodies are rig.json's, often ending at the ankles and wrists, so shoes and fingers stand out of them; default `min` 0; detail: `object`, `at_frame`, `body`, `object_frames` |
| `ribbon_clear` | least gap (m) between the [[ribbon]]s as they show on each frame (the drawn part of the centre line the stage stores on each root, `mk_ribbon`, with the radius that holds its surface, pulses included) and the characters' collision bodies as the scene poses them (after the sim) | exact for a round ribbon, a flat one counted as thick as it is wide; negative: into a body; default `min` 0; detail: `ribbon`, `at_frame`, `body`, `at_m` (how far along the ribbon), `ribbons` (each one's least gap); see [Ribbons](#ribbons) |
| `wrist_bend` | largest bend (deg) of a hand from its forearm (elbow to wrist against wrist to the middle knuckle) while a move holds it | the moves library's rule (`MV.WRIST`, default `max` 60); detail: `side`, `move` |
| `hand_mirror` | largest mismatch (mm) between a two-hand move's hands, each wrist and middle knuckle against the other's mirror image | default `max` 5; detail: `move`, `deg` (the hands' directions, mirrored) |
| `mic_palm` | smallest share of a fist's palm toward the body while it holds a mic or a hammer (the library's aimed places) | default `min` 0.3; detail: `side`, `move` |
| `hand_room` | smallest distance (mm) from a hand's wrist and finger joints to the character's collision bodies at places that do not touch the body | the trunk bodies of rig.json, a little fatter than the skin; default `min` 0; detail: `side`, `move`, `nearest` |

The four hand metrics read the cast member's `[[move.<cast>]]` (see [Moves](#moves)): a move holds a hand from its `t` to
`t + dur`, and a frame counts while the hand holds still at its place there (under 1.5 mm a frame), so a hand setting off
early for its next move is travelling, which the rules do not cover. The hands ride the upper body (the pose stage keys
their places on it), so mirror and palm are measured in its frame: the world with the upper body's motion undone, in the
armature's axes.

**Arm twists.** `joint_limits` measures how far each arm turns about itself as `mkmmd.core.armtwist` defines it, the pose
stage's arm rig holding the same ([Posing](#posing)): `forearm_twist` is the swing-twist of the hand (the wrist bone)
relative to the elbow about the rest forearm (elbow to wrist), so a twist bone's roll and any the wrist adds both count;
`arm_twist` is the upper arm's distal bone (its twist bone, else the arm bone) relative to the chest about the rest upper
arm, so a rig or a motion that splits the turn between the arm and its twist bone reads the same. Both are against the
rest pose, + supination and external rotation on either side. An MMD arm rests out, palm down, the thumb forward, and the
elbow folds forward from there: that is the forearm's mid-prone neutral and the humerus's, so the limits are anatomy's
from neutral with a little room (AAOS and Boone & Azen 1979: a forearm turns about 80-90 degrees each way, an upper arm
70-90): `forearm_twist` -90 (pronation) to +100 (supination), `arm_twist` -95 to +95, its external bound widened by as
much as the arm is raised past level (`elevation_deg` in detail: an arm raised overhead out to the side turns about 90
out on its way up, and one raised forward overhead reads near 180 against the rest). A twist near half a turn reads
either side of 180 (the same pose). Professional dance motions turn the arms' twist bones much further at moments
(up to 160-170 degrees on 8 calibration dances, 24,900 frames), where an MMD rig's ball-joint elbow hides it: a project
that plays one widens the two series in `limits` and leaves the skin to `skin_collapse`.

**Skin.** `skin_collapse` takes the meshes a character's armature deforms (a mesh a cloth or soft-body modifier moves is
left out and named in `detail.skipped`), and per limb segment the vertices whose heaviest bone is the segment's own or a
twist or D bone under it, with every triangle that has one; the chains a sim moves (hair, a skirt, ears, a sleeve's
chain) belong to no segment. It cuts each segment with planes square to it (the bones' heads, as posed) at 11 places from
30 to 80 % of its length and compares the cut curve's mean distance from the segment with the same cut at rest, mesh
object by mesh object. Linear blend skinning pulls a ring of skin weighted to bones turned far apart about the limb toward
its axis, and the faces between rings turned against each other neck in: a forearm turned past what its twist bones can
spread goes flat and jagged. The joints' own pinch is outside the cuts: an MMD rig's shoulder, elbow and knee lose volume
at every deep bend (no corrective shapes), up to a quarter of a segment from the joint. On a jacketed model with three
twist bones per arm, ordinary moves change a cut by at most 7 %, a forearm turned 155-180 degrees by 22-25 %, and
professional dances at their most twisted by 15-16 %. A 30 s clip (900 frames, one character) takes about 9 s: 5 for the
sample, 4 for the cuts.

Arguments:

| Metric | Arguments |
|---|---|
| `jitter` | `family` (a chain family or a list), `bones` (explicit bone names, besides or instead), `ref` (reference bone, `head`), `cast` |
| `contact` | `a`, `b` (an [expression](#expressions) or `track:NAME.channel`), `when` (a mask: `track:NAME.channel` or an expression, truthy frames count), `component` (`x`, `y` or `z`), `unit` (`m`, `cm` or `mm`, default mm) |
| `penetration` | `cast`, `families` (default hair and ears), `radius` (`{family: {scale, max}}` or one `{scale, max}`: the point radius is the body radius x scale, at most max, in metres), `colliders` (specs or the name of a collider set), `model_bodies` (true), `use_masks` (false: honour PMX collision masks everywhere, not only near the root), `anchor_free` (0.25 m) |
| `foot_slide` | `cast`, `points` (default `ankle.L ankle.R toe.L toe.R`), `floor` (0 m), `tolerance` (0.015 m: how far above its rest height a point still counts as planted) |
| `ground` | `cast`, `ground` (the set or prop stood on: its root's name, or a list; required), `reach` (0.25 m: a surface further above a foot's vertex is not its ground) |
| `joint_limits` | `cast`, `limits` (override `{name: [lo, hi]}` of `elbow_fold`, `elbow_back`, `knee_bend`, `knee_side`, `wrist`, `neck_swing`, `neck_twist`, `spine_swing`, `spine_twist`, `forearm_twist`, `arm_twist`; the upper arm's external bound still widens as the arm rises past level), `sides` (`L R`) |
| `skin_collapse` | `cast`, `sides` (`L R`), `limbs` (default `upper_arm forearm thigh shin`), `span` (`[0.3, 0.8]`: where the cuts go, as fractions of a segment from its joint) |
| `framing` | `points` (bones or expressions; default head and neck), `cast`, `outputs` (default all), `safe` (0.05: the safe-area margin as a fraction of each side). Give one per shot with `frames` set to that shot's frames: a close-up of a page is not supposed to contain the head |
| `occlusion` | `points` (expressions; default the cast's head and chest), `cast` (its meshes are the subject) or `ignore_models` (armature names), `ignore` (object names that never block: glass, hair cards) |
| `camera_inside` | `ignore` (object names a camera may sit inside) |
| `flicker` | `images` (a glob of consecutive frames), `width` (320) |
| `palette` | `images` (a glob), `palette` (a name or a list of hex colours), `black` (0.035: the luma below which a pixel is near-black), `measure` (`black` or `distance`), `width` (240) |
| `form` | `prop` (a prop root name or a list), `objects`, `exclude`, `exempt`, `exempt_weight`, `frame`, `hard_edge_deg` (65), `worst` (6): see [Form](#form) |
| `strum` | `cast`, `hand` (`R`), `pick` (the pick object, default `<prop>_pick`), `tip` (metres from the pick object's origin to its tip along x; default the card's `pick.tip`, 0.008), `strokes` (`down`: the worst down stroke, or `all`) |
| `string_play` | `cast`, `hand` (`R`: the pick hand), `pick` (default `<prop>_pick`), `finger_mm` (3), `touch_mm` (2.5), `time_ms` (8), `pick_mm` (5) |
| `prop_body` | `cast`, `prop`, `bones` (semantic bones whose bodies count; default `upper_body upper_body2 lower_body neck head leg.L leg.R knee.L knee.R`), `exclude` (a cable, a pick), `samples` (4000 vertices), `frame` |
| `ribbon_clear` | `ribbons` (names or fnmatch patterns; default every [[ribbon]]), `cast` (a member or a list; default every member with a rig.json) |
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

By default mkr draws the images from the scene's bake (`--engine mkr`; `mk build` writes the bake): the cut and the shots'
cameras (`--cam <shot>@<output>`, that shot's plates), each output at `--size`'s share of its size, with the layers the
cut's effects read and the screen type. It refuses, before drawing anything, a bake older than the scene and anything mkr
cannot draw (as `mkr info` sorts it), so nothing renders in Blender. Orbit views (`--view`), `--hide`, `--ab`, `--map` and
another scene file need a Blender engine (`--engine eevee`, `workbench` or `cycles`).

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
Python and are cached in `<project>/.mk/cache/` by a hash of their inputs. A stage that samples the scene on every frame (the
gaze's targets, the shots' mounts and targets, the secondary motion's sources, the folds' bones, the cloth's motion key)
evaluates only what it reads and what that depends on (`mkmmd.blender.scope`: parents, constraint and driver targets, the
objects modifiers name), every other object left out of the view layer's evaluation for the pass and put back after: on a
set dressed with a thousand animated props a frame change went from 13 to 4 ms, the samples the same.

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
| ribbons | `[[ribbon]]` | 3D paint strokes in the scene (crayon, striped toothpaste, neon marker) round a bone, up a body, through points or out of a prop's nozzle, drawn on, retracted, pulsing on a lane, kept clear of the cast's bodies | [Ribbons](#ribbons) |
| shots | `[[shot]]`, `[[transition]]`, `[[insert]]`, `[[glitch]]`, `[vector]`, `[[ring]]`, `[[freeze]]`, `[[medium]]` | the cut: a camera per shot and output, markers, render-time looks, cut effects, medium windows | [Shots](#shots) |
| lights | `[[light]]`, `[look]` | lights in palette colours (mounted, aimed, keyed); view transform, contrast look, exposure | [Lights and the look](#lights-and-the-look) |
| text | `[[text]]` | type on set and prop surfaces, lyric type, handwriting, screen type | [Text](#text) |
| keys | `[[key]]` | keys on set, prop and object properties | [Keys](#keys) |
| sim | `[sim.<cast>]` | secondary motion solved outside Blender and baked to keys | [Sim](#sim) |
| folds | `[folds.<cast>]` | the fold shapes a joint angle drives (rig.json `folds`), keyed from the final pose | [Folds](#folds) |
| cloth | `[cloth.<cast>]` | garments' cloth copies (rig.json `cloth`) simulated with Blender's cloth, the garment's faces following it | [Cloth](#cloth) |
| stepped | `[[stepped]]` | the cast on twos: each member's whole motion held 1 / fps s at a time over a window, the cameras on every frame | [Stepped motion](#stepped-motion) |
| clip | | every shot camera starts at half the depth of the nearest geometry it sees, keyed on every frame (follows `--until` as `save` and `bake` do) | [Clip ranges](#clip-ranges) |
| save | | the `.blend` | |
| bake | | `<blend>.bake/`: the clip for the site's scene viewer and mk's own renderers, every frame's bones, morphs, objects, visibility and shot cameras (with their clip ranges), the meshes, what the glb cannot carry as Blender evaluates it on each frame, what the vector look needs (each object's record, the materials' viewport alpha and textures, the rings' centres, the studio light), the [[medium]] windows, and the lit scene mkr draws from: the view transform, world, lights, material node graphs, images and every mesh as Blender renders it, checked against Blender's ([The bake](#the-bake); `mk build --bake` bakes a saved scene again) | [The bake](#the-bake), [The site](#the-site-mk-site-mkmmdsite), [Rendering](#rendering) |

The text stage runs after `lights` and before `keys`, so a `[[key]]` can toggle a text's `hide_render` or move it. A stage
that has nothing to read for a cast member (no `[pose.rin]`, no `[perform.rin]`) leaves it alone, and a `[pose.x]`, `[perform.x]` or `[sim.x]` named after no cast member is ignored
without a word (see [Cast](#cast)).

### The bake

The bake stage writes `<blend>.bake/` (version 5): `scene.glb`, `bake.json` and `bake.bin` (float32 values per frame) for the
site's scene viewer and the vector look's renderer, and the `lit` section of bake.json with `lit.bin` (vertex attributes,
triangles and shape-key deltas, little-endian), `view_lut.bin` and `images/` for mkr, mk's GPU renderer of the lit frames.
`mkmmd/blender/build/bake.py` has the format of the first and `bake_lit.py` of the second. Positions and directions are in
glTF's axes (+Y up); what node graphs read (Generated coordinates, attributes) stays in Blender's.

- View: Blender's display transform for the scene's view settings (transform, look, exposure, gamma, curves, display), captured
  by saving a float image of a 65³ grid of scene-linear values (2^-12.5 to 2^12.5 on a log2 scale) through them as a 16-bit
  PNG with no dither and reading it back: the display-referred value before 8-bit quantisation (`zero` for black), and the
  render's dither. Film filter and transparency, every render and EEVEE setting.
- World, materials and node groups: every node of their trees with every value of the node (colour ramps with a 257-sample
  table, curve mappings sampled where Blender builds its tables), its sockets, the links and what a muted node passes through;
  node groups once each with their interface. Images as files: absolute paths, packed and generated ones written into
  `images/`, and one Blender cannot load marked `missing` (Blender draws it magenta).
- On every frame: each light (position, direction, colour, energy, size, spot cone, whether it shows), each shot camera's
  depth of field, every node value that is keyed or driven (the mmd_bind inputs mmd_tools drives with a material morph), and
  per mesh what its materials' Attribute nodes of type Object, Instancer or View Layer read off its object (`object_attrs`:
  a custom property, else an RNA path, of the object, else of its data, as Blender looks it up; a ringing string's swing, the
  ink's clock), which mkr fills into each draw's own parameters.
- Meshes: every mesh the clip shows, as Blender renders it, its vertices split where Blender's corner data differs, with what
  its materials read (UV maps, colour attributes, named attributes, tangents, Generated coordinates), its triangles grouped by
  material slot, and its world matrix, visibility and shape-key values on every frame (a key that stays 0 over the clip is
  left out). A mesh is `static`, `skinned` (one armature modifier: up to 8 bones a vertex, weighted as the modifier normalises
  them, on the joints the bake already has) or `by_frame`: anything else Blender evaluates (other modifiers, curves, text,
  metaballs, mmd_tools' SDEF, preserve volume, absolute shape keys) as its evaluated mesh in its own space, with its world
  matrix, on every frame it shows: an unchanged geometry or array is not stored again (a sign's type riding a car, 38 000
  vertices, is stored once and moves with its matrix), up to 256 MB an object, evaluated with the render's modifier settings
  (shown as in renders, Subdivision Surface and Multires at their render levels, curves and metaballs at their render
  resolution: the scene is put back after the bake; the report's `lit.render_settings` lists what that changed, and the
  version-4 `by_frame` geometry of those objects is evaluated the same way). Instances (geometry nodes' object instances,
  collection instances, particles) are meshes of their own that share their source's vertices. A mesh that bends (skinned or
  shape-keyed) carries its corner fans, so a renderer rebuilds its normals on the deformed mesh as Blender does, custom
  normals included; the bake picks each fan's reference edge against Blender's own deformed normals on a few frames (turning
  the rest normals with the joints instead misses Blender's by 14 to 23° on 1 % of an MMD model's corners).
- Every lit mesh names its look record (`record`: the object's own entry in bake.json's `objects`, an instance its
  instancer's), and `lit.records` holds the records of those no look sorts (screen type, objects drawn for one output only),
  so the figure, bare and keep patterns resolve from the bake alone.
- What Blender renders that the bake cannot hold (a volume, hair, a point cloud, a mesh over its cap, instances of a deforming
  object) is listed in `left_out` with the frames it shows on, so `mk render` leaves those frames to Blender.
- Screen type (`lit.screen`): every word of it on the frames it shows (`show` [on, off) frames, `aspect` its output,
  `knockout`, `negative`: a negative word is drawn alone into `negative/<frame>.png` before the rest goes into `screen/<frame>.png`,
  and the PNG writer puts the first file on disk before the second), the screen layer's orthographic camera per output (one frame height tall, at z = 10 looking down, +Y up,
  in glTF's axes), its pass (EEVEE, film transparent, lights off, black world, no motion blur or dither, the shot look's
  samples or 16), its view transform (Standard, no look) captured as a LUT like the scene's, and its file format
  (straight RGBA, 8 bits). A word is held as fitted geometry (`mkmmd/core/fitgeo.py`, numpy, tested; mkr decodes it in
  `mkr/mkr/src/screen_geo.rs`): the prototype shapes of its islands (a letter, each outline copy of it, its parts), and
  per frame one map of the whole word or a map per island (affine; with quadratic terms for letters that wobble before
  the word bends round a ring; projective for a tilt seen in perspective) and the islands' attribute values, every map
  measured as decoded against Blender's evaluation (1e-5 frame heights, normals within 1.1 degrees); a frame no map
  holds starts a new geometry. Crimson and Clover's 297 words, 21.8 GB as meshes of every frame, fit in a few tens of
  MB. The layer may add 512 MB to lit.bin in all: what is not baked when that runs out is listed in `lit.screen.left_out`
  with its output and frames, and those frames' screen layers are Blender's. The self-check decodes every word on the
  first, middle and last frame of its window from the files and measures it against Blender's drawing (`check.screen`).
- Points (`lit.points`): every world point the cut's layers ask for (a transition's or insert's anchor, an effect's
  centre: `point` items of mkmmd.core.transition.demands), evaluated on the frames that ask for it as `mk render` evaluates
  it (the `mk q` namespace; NaN elsewhere, a failure kept with its frames) with the shot per run of frames. The point layer
  file is `{"p": [x, 1 - y], "depth": z, "m": hypot((qx - x) w / h, qy - y)}`, (x, y, z) the point through the shot
  camera of the output (0..1 from the bottom left, z its depth), (qx, qy) the point one metre along the camera's own +Y
  axis seen the same way, w and h the output's size (a fisheye's: mkmmd.core.fisheye, which mkr leaves to Blender).
- Self-check: the bake rebuilds every static and skinned mesh from lit.bin and bake.bin alone on the export frame, the middle
  frame and the last, and measures it against Blender's evaluated mesh. The report's `lit` has the worst distance per mesh
  (`check_mm`; over 0.1 mm is a WARNING), bake.json's `lit.check` the normals' worst, 99th and 99.9th-percentile angle too.

### Targets

A target is anything a pose, a gaze, a camera, a light or a text can point at. Targets resolve at the frame the stage is on, so
a target that rides a moving prop or a posed bone moves with it.

| Form | Meaning |
|---|---|
| `[x, y, z]` | a world point |
| `{prop = "car", point = [..]}` | a point in a prop's local frame (follows the prop) |
| `{cast = "rin", point = [..]}` | a point in a cast member's own frame: its model root, at `[[cast]] at` and turned by `yaw`; x its left, y behind it, z up. A pose, a gaze or a camera written this way keeps working when the character is moved to another place in the world (a hand's `dir` and `palm` stay world directions) |
| `{cast = "rin", bone = "head", point = [..]}` | the same point given where it is at rest, riding that bone (semantic, Blender or PMX name): carried by the bone's move from its rest pose, so a close-up camera in front of her eye stays in front of it however she dances |
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
| `photo_world` | the scene's world made of ONE photo projected round the scene in world directions, so any camera (a Cycles fisheye too) sees it bent as through its lens, it lights the cast and glossy things reflect it; nothing spatial (`at` and `yaw` do not turn it) |
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

**`photo_world`** keys (the maths is `mkmmd/core/photoworld.py`, tested; the world's nodes follow it step by step, checked by
rendering a numbered grid in Cycles, EEVEE and a Cycles fisheye, about one level in 255 from the prediction): `image` (required:
the photo, a file, project-relative, absolute or `~/...`, or the slug of a registry entry whose path is an image: jpg, jpeg, png,
webp, exr or hdr; a missing file or one that is not an image is an error; its alpha, if it has one, lets `fill` through), `toward`
(`[0, 1, 0]`: behind a cast member who faces -y, what a camera in front of her sees; the world direction of the photo's centre),
`roll` (0: degrees the photo turns clockwise about `toward`, as a camera looking along it sees it; unrolled, its up is the world's
up made level, +Y when `toward` is vertical), `fov` (required: degrees across the photo's long side, its true field of view;
2 atan(18 / f) for a 35 mm-equivalent focal length f), `spread` (= `fov`: degrees its long side spans in the world), `fill`
(`"#080808"`: palette slot or `#hex`, the world where the photo does not reach), `strength` (1: what the camera sees, the photo's
own pixels at 1, through a floor faded to clear too), `light` (= `strength`: every other ray, a Light Path split: the light it
casts on the cast and what glossy things and puddles reflect; daylight on the cast may want 3 or more while the background stays
the photo) and `saturation` (1). A ray at the angle θ from `toward` and the azimuth φ about it shows the photo at the true angle
α = θ · fov / spread, at tan(α) / tan(fov / 2) · (cos φ, sin φ) half long sides from its centre, aspect-correct, sampled smoothly
(Smart: cubic when magnified): `spread = fov` is the perspective of the camera that took it, a larger one stretches the photo
angularly and uniformly about its centre, so a modest photo fills a fisheye's 180 degrees and its straight lines still bend as
through a lens; where α reaches 90 degrees or the point falls outside the photo, the world is `fill`. The rest of the sphere is
`fill` too and lights the cast as such: a sky colour there is a soft fill from behind the camera, near black none. Under AgX the
photo's whites roll off and its colours soften; `[look] view = "Standard"` shows it as shot. EEVEE: the world probe is 2048 px and
its sun extraction is off. Card: `use.look` `centre` (100 km out along `toward`). `mk assets credits` counts the photo (see
[Asset registry](#asset-registry)).

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
| `electric_bass` | a 60s-70s short-scale bass of the Mustang Bass kind (30 inch scale, 19 frets, an offset body, a split single coil, a string-through bridge, a control plate with two knobs and the jack, four in-line clover keys), upright like the guitar. Roles in `slots`: the guitar's, `knobs` (the black plastic of the knobs and the pickup covers) and `stripe`: naming it adds the racing stripes. Card: the guitar's contract with four strings and their `tuning` (E1 A1 D2 G2); see [Playing a worn guitar](#playing-a-worn-guitar) |
| `offset_guitar` | an offset-waist electric of the late 1950s kind (the electric guitar's neck on a long body with a skewed waist, a big tortoiseshell pickguard, two wide flat pickups, a rocking bridge and a floating tremolo with its long arm, a toggle, a slide switch and two thumb wheels, the jack on the guard), upright like the guitar. Roles in `slots`: the guitar's, with `pickguard` and `guard_dark` (the tortoiseshell's flames and ground) and `knobs` (white plastic: knobs, pickup covers, the toggle's and the arm's tips); options `guard = "plain"` (the `pickguard` role alone) and `arm = false` or `{swing = 8, lift = 0.016, length = 0.165}` (degrees the arm lies off the strings toward the treble side, metres its tip stands further off the face than its bend, its length). Card: the guitar's contract, its strum point between the two pickups, a small pick held across (`hold`, `clear` in the strum entry), and the anchor `trem`, the arm's tip: `[[prop]] anchor_to = "<cast>"` puts the arm in that member's picking hand; see [Playing a worn guitar](#playing-a-worn-guitar) |
| `hearts` | a field of puffy hearts that rise and sway around a point, animated by drivers on the frame (see below) |
| `heart` | one big puffy heart that pounds on the song's beat (a bump on the beat and a smaller one after it) and bursts into a ring of little hearts every few beats; options `size` (0.2), `bpm`, `start` (a clip second on a beat), `beat` (0.25: the swell), `burst` (8 hearts), `every` (4 beats), `reach` (0.45 m), `burst_size`, `turn`, colour `heart` ("love"), `glow`. Drivers on the frame and the scene's `mk_frame0`; objects `<name>_heart`, `<name>_burst<i>` tagged `mk_heart` (silhouette accents: `accent = ["prop:mk_heart"]`); with `attach` it rides a chest |
| `butterfly` | one butterfly about 9 cm across, flapping (`mkmmd/core/butterfly.py`: outlines from landmarks read off photographs, the meshes, the flap maths; `butterfly_pattern.py`: the wing patterns): `style` ("monarch"; "swallowtail" with tails, "morpho", "blue": a small blue), `size` (the span, 0.09 m), `way` (a colourway of the style, 0..2), `sheen` (the iridescence at a glancing view: morpho 1, blue 0.6, else 0), `glow` (0.1: the wings' own light, an underside in shade never goes black), `tile` (pattern pixels per wing, 1024). Fore and hind wings are thin membranes with a rolled edge, the hindwing tucked under the forewing; the patterns (veins fanning from a discal cell, borders, edge dots, eyespots, stripes, the sheen) are masks in a packed atlas (cached in `~/.cache/mk/butterfly`) coloured by the roles `ground`, `under` (the underside), `ink`, `spot`, `accent`, `sheen`, `body`, `eye` (palette slots, hexes or blends `"gold:2,light:1"`; `dark` and `light` name the palette's darkest slot at 42 % and its lightest). Live root properties for `[[key]]`: `flap` (3 beats/s; the keys stage integrates it into `beats`, so a keyed rate never jumps the wings), `flap_amp` (100: the full stroke, degrees), `flap_mid` (20: its middle above flat), `phase` (cycles; with `flap = 0` the pose the wings rest in: 0 up, 0.25 mid going down, 0.5 down), `hit` and `hit_time` (0.3 s): `[[key]] prop = "hit", lane = "kick"` gives one wing beat on every kick (and with `flap = 0` none in between). Objects `<name>_body` (the span's scale; it bobs with the beat) and `<name>_wing_L` / `_R` (its children, turning about their hinges): drivers only, so a freeze holds it mid-beat. Front -Y, origin the middle of the thorax; card `use.look` `head`, `body`, `tip_L`, `tip_R`, `use.anchor` `perch` (under the feet: `attach` it to a finger with that offset), `rates` |
| `butterflies` | a swarm of butterflies round the root (40 to a few hundred), each with its own wing-beat phase and rate, span, style and colourway, flapping (the `butterfly`'s parts and flap), in formations blended by keyable weights: `rings` (a mandala facing -Y), `orbit` (tilted orbits round `orbit_at`: her head with `attach`), `drift` (a wandering flock); `burst` flings them out and away, `bloom` grows the rings in from the centre. One geometry-nodes object `<name>_swarm` (see below) |
| `traffic_car` | an 80s car or truck for night traffic: `body` "sedan", "wagon" or "semi" (a cab-over tractor and a box trailer whose sides are `use.surface` `left` / `right`), colours `paint`, `trailer`, `glass`, `lamp`, `tail`, `marker`, emissions `lamps` (30), `tails` (9), `markers` (3.5), `beams` (real headlight spots: `beam_power` 900 W, `beam_angle` 55, `beam_tilt` 5). Faces -Y, origin under the middle; the card lists the wheels and beams, so a `[[vehicle]]` drives it |
| `semi_truck` | an American long-nose conventional tractor and a 53 ft dry van on its fifth wheel, built for close shots (`semi_truck_layout.py`: proportions of a long-hood tractor's body-builder drawing, generic and unbadged): hood with crowned fenders and a tall grille, cab and 63 in mid-roof sleeper, a steer axle and two drive axles with duals and the van's tandem (11R22.5 tyres on polished ten-stud wheels, every wheel on its hub, axle and suspension), the van's landing gear, doors, lamps and tape, and a cab inside (seats, a 20 in wheel, a dash with lit gauges, a CB radio and its mic, a shifter, a dome light). Faces -Y, origin on the ground midway between the drive axles, the driver on +X; the bumper at y -7.14, the van's rear at +14.94. Roles in `slots`: `body`, `accent`, `trim`, `chassis`, `rubber`, `glass`, `headlamp`, `marker`, `taillamp`, `van`, `van_trim`, `tape`, `interior`, `dash`, `gauges`, `glow`, `display`, `wood`, `curtain`, `hose_red`, `hose_blue` (slots, '#hex', `slot:w,slot:w` blends whose parts may be `a:b:t` mixes). Options `trailer` (true; false: the tractor alone), `beams` (true: headlamp spots driven by `lamps`, `beam_power` 1100 W, `beam_angle` 50, `beam_tilt` 4), `mural` (an image file, project-relative / absolute / `~`, or a registry slug whose path is an image, painted on both van sides; `mural_left`, `mural_right`, `mural_back` per panel; `mural_fit` "cover" (default) / "fill" / "contain"; the image's alpha shows the van's colour, the joint strips stay on top). Root properties for `[[key]]` (also starting values in `slots`): `lamps` (1), `markers` (1), `brake` (0), `dash_on` (1), `dome` (0). Card: `wheels` (ten wheel ends, a dual pair one object) and `steering` (`<name>_wheel`, `ratio` 20, `wheelbase` 6.35), so a `[[vehicle]]` drives it; `lights` (the beams, `<name>_dome`, `<name>_dash_glow`); `use.sit` `driver` / `passenger`, `feet`, `grip` `wheel` (a ring riding `<name>_wheel`), `rest` `sill_L` / `sill_R`, `look`, `surface` (`left`, `right`, `back`: the van's panels; `door_L` / `_R`, `sleeper_L` / `_R`), `anchor`; box colliders and a ring; `trailer` {`object`: `<name>_trailer`, the empty every van part rides, `pivot`: the kingpin}. `semi_truck_card.py` builds the card (tested without Blender) |
| `rail_track` | an 1860s standard-gauge railroad track (1435 mm) that lays itself as `laid` moves (see below): hewn ties 8.5 ft long every 2 ft (six variants with wane and adze marks), 56 lb/yd iron T rails in 30 ft lengths joined by fish plates and bolts, cut spikes hooked over the rail foot, optional tie plates (`plates`), on a ballast bed with sloped shoulders; from the root along +X, straight or round a gentle curve. Options `length` (30 m, rounded to whole tie spacings), `radius` (0: straight; m, > 0 bends left toward +Y, < 0 right, at least 50), `plates` (false), `seed` (1: the ties' variety), `laid` (its starting value; default the card's `track.done`: all of it). Roles in `slots`: `ballast` ("subtle:gold:0.25") and `ballast_dark` ("muted:ink:0.45") the stones, `tie` ("gold:muted:0.6") and `grain` ("gold:ink:0.7") the wood, `rail` ("muted:ink:0.35"), `railhead` ("subtle:paper:0.35": the running surface), `rust` ("love:gold:0.45"), `spike` ("ink:muted:0.35"), `plate` ("muted:ink:0.5": fish plates, bolts, tie plates). Card: `use.anchor` `front` (the empty `<name>_front`, on the centre line where `laid` is, at the rails' top: a camera target that rides the laying) and `last_tie` (the last tie's top: stand a spike in it), `use.look` `start`, `end`, `front`; `params` {laid}; `track` {length, radius, gauge, spacing, rail, done, ties, rails, plates, tie_top, rail_top, objects} |
| `mile_post` | a mile post, its number in numerals built in code (`mkmmd/core/numerals.py`: a monoline skeleton per digit 0-9 in the proportions of the FHWA highway alphabets, Series E, made into meshes; no font file), standing on the floor at its origin (floor_center), its number facing -Y. `style` "railroad" (default): a 19th-century railroad mile post, a 6 x 6 in timber 1.30 m to its shoulder under a hipped top 55 mm high, its arrises chamfered 1 1/8 in and a season check down each face, painted white and weathered (rain stains from a greyed top, flaking paint, grime, a tarred foot), and on its front face 35 mm under the shoulder a cast-iron plate with a raised bead, four domed bolts and the numerals cast in relief (a condensed gothic 115 mm tall). `style` "interstate": a US reference location sign (MUTCD D10, freeway size, without the word MILE), a 12 in panel with 1.5 in corners and a white border, 10 in Series E numerals stacked one per row, bolted to a galvanised 3 lb/ft U-channel post with its punched holes, the panel's bottom 4 ft up; the sheeting glows a little (retroreflective), its border and numerals are flat graphic layers. Options `number` (a whole number 0 .. 999 or a string of 1 to 3 digits, "007" kept; "" (default) leaves the plate blank), `digits` (1 .. 3: the numerals the plate is sized for; default the number's count, 2 when blank: the railroad plate is 0.11 / 0.20 / 0.29 m wide, the panel 17 / 30 / 43 in tall); a bad style, number or digits is refused saying what to give. Roles in `slots`: railroad `post` ("paper"), `plate` ("ink"), `numerals` ("paper"), `wood` ("muted:gold:0.25": the bare timber where the paint is gone), `grime` ("ink:gold:0.3": dirt, rust specks), `foot` ("ink"); interstate `post` ("muted:paper:0.35": the galvanised post, the panel's back, the bolts), `plate` ("pine": the green signs' slot, as the `highway` set's), `numerals` ("paper"), `border` ("paper"). Objects `<name>_post`, `<name>_plate` (railroad) or `<name>_panel` and `<name>_border` (interstate), `<name>_numerals` (none on a blank plate). Card: `use.surface` `plate` (the numeral area on the plate's face: `[[text]] on = "<name>:plate"` writes on a blank plate; an interstate column is `text = "1\n0"` with `leading = 1.3`; give the text a colour that reads on the plate in your palette), `use.look` `plate` and `top`. Static |
| `neon_sign` | a roadside motel / diner sign: a tall pole, a rounded board ringed by a neon tube and a smaller board under it, both `use.surface` (`face`, `sub`) for glowing [[text]]; options `width` (3.2), `tall` (1.5), `height` (6.5), `sub` ([2.0, 0.55], [] none), colours `board`, `neon`, `sub_neon`, `pole`, `glow` (7) |
| `meteor`, `airplane` | sky events at clip seconds, seen from the root: a shooting star (`t`, `dur`, `from` / `to` [azimuth, elevation] of its head, `length`, `color`, `strength`, `distance` 2500) and an airliner's lights crossing (`t0`, `t1`, `from`, `to`, `lights`, colours `red`, `green`, `white` (a strobe), `strength`, `distance` 3000). Keyed at build from `mk_frame0` |
| `line` | the space kit's line (`mkmmd/core/space.py`): a thin glowing line along +X from the root, `length` (40 m) long, `radius` (0.012), made of `seg` (0.25 m) segments `<name>_seg<k>`, with tick marks `<name>_tick<j>` every `ticks` m (0: none) across it, every `bar`-th (4) long (`tick` [short, long] 0.05, 0.12 m). Root property `extent` (0..1, key it): how much of the line there is from its `anchor` end (`"end"`, its +X end, or `"start"`): it retracts into that end, each segment shrinking toward its anchor-side end and each tick vanishing as the line passes it (drivers). `ripples = [[t, x, amp], ...]`: at clip second t a pulse runs out both ways from x metres along the line, a Ricker wavelet `amp` m high (negative: a dip with a lift either side), `ripple = {speed = 7, decay = 0.3, width = 0.5, life = 1.0}`; each segment is tilted to meet its neighbours (keyed). Colour `color` ("text") |
| `hole` | a black hole's shadow: a sphere of `radius` (1.5 m) in `color` (black, no highlight) as `<name>_shadow` (tagged `mk_hole` = its radius: the [lens effect](#effects) bends the light round it). Root property `breathe` scales it (key it on the kicks). `use.look` `centre` |
| `round_board` | a round chessboard round the origin (Byzantine chess is 4 rings of 16): `rings` (6) rings `<name>_r<i>` of `sectors` (16, even) cells from `inner` (2.2 m) out, each `width` (0.8) deep and `thick` (0.05), `gap` (0.03) between cells, light (`light`, "text") and dark (`dark`, black) alternating round a ring and from ring to ring. Root property `phase` (key it): ring i turns `omega_i * phase` radians about the centre, `omega_i = omega * (r_i / r_0) ** kepler` (`omega` 1, `kepler` -1.5: inner rings orbit faster and the board winds into a spiral). Every cell has an overlay `<name>_hot_r<i>s<j>` in `hot` ("love"), hidden: a `[[key]]` on its `hide_render` lights an attacked square. A prop `parent = "<name>_r<i>"` rides a ring. `use.rest` `top` (a plane), `use.look` `centre` |
| `chess` | a chess piece standing on its origin, `height` (1 m) tall: `piece` `pawn`, `rook`, `bishop`, `queen` or `king` turned on a lathe (Staunton-like), or `knight` (a lathed base under a horse's head, a thick plate domed on both faces, facing +X), glossy (`rough` 0.12, `metal` 0, a clear coat) in `color` ("text"); `<name>_piece` |
| `stars` | flat stars in the picture's plane, facing -Y: `points = [[x, z, size, t], ...]` (metres in the root's x-z plane, a size factor, and optionally the clip second it flashes: up to `gain` (1.8) times its size and back over `flash` (0.35) s), `kind` points (4), `inner` radius (0.3 of the outer), `size` (0.12 m), `depth` (0.004), glowing `color` ("text"); with `lines` (true) each star is joined to the next by a rod `<name>_line<i>` (`line_width` 0.01) that draws itself over `draw` (0.12) s as the next star flashes: a constellation, a melody as a piano roll |
| `floor` | a floor to the horizon at z = 0 under the root, `size` (400 m) each way from it (`<name>_floor`): `kind` `mirror` (default: dark glass that shows what stands on it; the floor switches on EEVEE's screen-space ray tracing, which mirrors what the camera sees), `chess` (light and dark squares `cell` (2 m) across), `grid` (glowing lines every `cell` m, `line` (0.04 m) wide, at `glow` (3) in `color`) or `asphalt` (a street: angular stones of the aggregate, coarse and fine, standing out of the binder with polished tops, mottled by wear and patches; `cracks` (0.3: the share of a net of thin tar-sealed cracks round cells about 1.8 m across that shows, 0 none), `wet` (0 dry, 1 rain-wet: darker and as rough as 0.3), `puddles` (0: the share of the floor under standing water, flat pools as rough as 0.03 with a damp rim) about `puddle_size` (1.5 m) across, `seed` (1: other puddles, cracks and stones)). Colours `color` (the light squares, the lines, the asphalt's lightest stone; "text", asphalt "#a3a19c") and `dark` (black, asphalt's binder "#4a4946": a sun-bleached grey street); `rough` (mirror 0.03, chess 0.12, grid 0.08, asphalt 0.8 where dry), `metal` (0). The pattern blends its edges over a few pixels' footprint at the camera's distance, so far away a checker becomes its own grey and a grid dims instead of shimmering (`mkmmd/core/space.py` `checker`, `grid_lines`; the asphalt's stones fade to their mean tone, measured, once a pixel is wider than a stone), and it fades into `dark` between the two `fade` distances from the camera (`[40, 220]` m), or with `fade_to = "clear"` into nothing (any kind: a mix with transparency, dithered in EEVEE, so the world shows beyond: a [photo_world](#sets)'s own street). `use.rest` `top`. Floors that change with the song are three floors with `hide_render` keys on their `<name>_floor` objects |
| `starfield` | stars streaming past the root along -X at `speed` (12 m/s), plus `surge = [[t, metres], ...]` of extra way (eased between the keys: a jump to light speed). `count` (900) stars per tile, `size` (0.035 m) across and `streak` (0) m long along X, in `color` ("text") glowing at `glow` (12), spread over `length` (160 m) along X, `width` (90) across and `height` (`[-1, 45]`), none within `clear` (5 m) of the X axis, where a character walks; `seed` (1). Two tiles `<name>_tile0` / `<name>_tile1` of the same stars leapfrog round the root (their X keyed per frame from the clock), so the field never runs out: together they always cover `-length` .. `length` |
| `disc` | an accretion disc round a black hole: a flat ring `<name>_disc` from `inner` (2 m) to `outer` (7) about the root, white-hot (`hot`) inside going to `color` ("love") outward and dimming, at `glow` (5), its streaks (a noise stretched along the orbit: `streaks` (3) round it, `rings` (22) across) orbiting `spin` (0.3) turns a second at the inner rim, faster inside (angular speed as `(r / inner) ** kepler`, `kepler` -1.5): a driver on the frame runs the clock. Additive (it lights what is behind it and hides nothing); tilt it with `rot`. `use.look` `centre` |
| `bedroom_desk` | an 80s writing desk, 1.20 x 0.60 x 0.74 m, three drawers; options `edge` (slot of the band round the top, "pine"), `drawers` (three slots, "foam,love,gold"); `use.rest` `top` and `front` |
| `desk_chair` | a chrome cantilever chair with padded seat and back; seats a character exactly as `cafe_chair` does; options `upholstery` ("iris"), `accent` ("love") |
| `desk_lamp` | a spring-arm desk lamp that owns a warm spot light; options `paint` (slot or `#hex`, "love"), `power`; root properties `power` (60 W) and `on` (0..1: key it to switch the lamp); yaw aims it (0 looks toward -Y) |
| `cassette_player` | a compact radio-cassette player (a boombox), 0.34 x 0.115 x 0.16 m; option `label` (the cassette's label colour); root properties `glow` (0..1) and `pump` (0..1: both speaker grilles, `<name>_grille_L` and `_R`, swell 12 % about their centres and come 4 mm out of the front; key it on the kicks); `use.surface` `dial` |
| `cassette_tape` | a compact cassette lying on the desk; option `label` (a slot or `#hex`; by default picked from the instance name) |
| `turntable` | a direct-drive DJ deck (SL-1200 proportions, 453 x 360 mm, unbranded) with a 7-inch single on its felt mat: grooves with their sheen, a paper label, the 45 adapter in the 38.2 mm hole, an S-shaped arm (headshell, cartridge, counterweight) with its rest, cue and lever, strobe dots, power knob, start / stop, 33 / 45 lamps, target light, pitch slider. Front -Y, origin on the floor under the plinth's middle. Roles in `slots`: `plate`, `body`, `metal`, `mat`, `vinyl`, `label`, `print` (the label's rings), `cartridge`, `lamp` (`ink` and `paper` name the palette's darkest and lightest); option `strobe` (W, 0.15). Root properties for `[[key]]`, also starting values in `slots`: `rpm` (0; 33.3, 45, negative turns it back; the keys stage integrates it into `turns`, the platter's revolutions since clip second 0, so a spin-up never jumps; the deck needs 0.7 s to reach speed), `arm` (0 on its rest, 1 over the lead-in groove, 1..2 the side played in to the last groove: key 1 + seconds / 210 while it plays), `cue` (0 down, the stylus on the record; 1 lifted 10 mm: lower it over 1-1.5 s for a needle drop). Card: `use.surface` `label` and `label_lower` (above and below the hole, riding the platter: `[[text]] on = "deck:label"` turns with the record; give it `lift = 0.0003`), `use.look` `record` and `stylus`, `use.anchor` `spindle` and `stylus` (the empty `<name>_stylus` at the tip), `rates`. Objects: `<name>_platter` (turns) carrying `_mat`, `_record`, `_label`, `_adapter`; `_arm`, `_yoke`, `_cue_bar`, `_cue_lever`; `_plinth`, `_feet`, `_controls`, `_arm_base`; the light `_strobe` |
| `handheld_mic` | a handheld vocal microphone, 162 mm long with a 51 mm ball grille, standing on its tail along +Z, and its cord `<name>_cable`; colour roles `body`, `grille`, `hardware`, `cable` (`slots`: a palette slot or `#hex`); `use.anchor` `jack` (the cord's exit under the tail), `use.look` `grille` and `grip`. Put it in a hand with `attach` and `cable` (see [Posing](#posing)) |
| `mic_stand` | a boom microphone stand with its microphone: a round base on the floor (origin at its middle), a two-part pole with its clutch, a boom on a swivel rising toward +Y with a counterweight behind, a clip at its end holding a `handheld_mic`-shaped microphone over it, the cord run back under the boom, down the pole and away along the floor. The singer stands on the +Y side: turn it with `yaw`. Options in `slots`: `height` (m, 1.30: the grille's centre, where a mouth goes), `angle` (degrees the boom rises, 28), `boom` (m, 0.36); the pole's length follows, and a height the pole cannot reach above its clutch is refused. Roles: `stand` (the tubes), `base` (the cast parts and the clip), `body`, `grille`, `hardware`, `cable` (the microphone's). Card: look points `grille` and `pole`, and `reach`, the grille's level distance from the pole: stand it so the grille is 4 to 6 cm before her lips |
| `speaker_stack` | a festival speaker stack: `stack` (`["sub", "sub", "top"]`) lists the cabinets from the floor up, each on four rubber feet on the one below, fronts flush toward -Y, origin on the floor under the middle; a `sub` is 1.20 x 0.75 x 0.56 m (two 18 inch woofers side by side, a slot port with a flared lip at each end of the baffle), a `top` 0.80 x 0.52 x 0.40 m (two 10 inch mids either side of a constant-directivity horn). Textured shells with rolled edges and a rim round the recessed baffle (cut with the drivers' holes, the ports' tunnels and the horn's mouth), lathed woofers (bolted frame, half-roll surround, flared cone, domed dust cap), a welded wire grille (`grille`: "mesh", or "none"), rigging plates with eyelets and quick-release pins, bar handles, a connector plate on the back: objects `<name>_cab<k>_<part>` (shell, drivers, hardware, ports, horn, grille; cabinets of one kind share their meshes) and hidden collider boxes `<name>_cab<k>_col`. Root property `pump` (0..1, for `[[key]]`: a `pulse` on a lane's hits): every cone and dust cap pushes out (18 mm on a sub, 7 on a top: the shape key `pump` of each `drivers`), the surround stretching. Roles in `slots`: `cabinet` ("base"), `baffle` ("ink"), `grille` ("hl_med"), `cone` ("ink"), `surround` ("overlay:ink:0.5"), `cap` ("love": the dust caps), `horn` ("ink"), `hardware` ("subtle"), `rubber` ("ink"), `port` ("ink"); `ink` and `paper` are the palette's darkest and lightest slot. Card: `use.look` `front`, `top` and `cone<i>` (bottom up, left to right as you face it), `use.anchor` `cone<i>` (`dir` -Y), `use.rest` `top`, box colliders, `form_max` 0.8 (cabinets are boxes by nature: the stack measures about 0.7) |
| `penlight` | an idol fan penlight, 265 mm: a 31 mm handle with a soft ribbed grip, a pill button on its front (-Y) above the fist, a strap ring at its end, a clear collar and a frosted 24 mm tube with a domed tip that glows; it stands on the end of its handle along +Z (`<name>_body`, `<name>_tube`). The light is one of `colors` (palette slots, hex or `a:b:t`; default love, gold, foam, iris, rose, pine, text): root property `color` is an index into them (rounded, wrapping round the list), keyed CONSTANT to switch on a hit (or a lane's `values`, see [Keys](#keys)), `glow` (0..1) its level (0: frosted plastic, the tube tinted by the light as it glows); option `strength` (2: the emission at glow 1). Roles in `slots`: `handle` ("base"), `grip` ("overlay"), `button` ("text"), `collar` ("hl_high"), `ring` ("subtle"). Card: `use.grip` `hold` (type "hold": the 75 mm of handle a fist closes on, `axis` +Z toward the tube, `radius` 0.0155), `use.anchor` `tip` (`dir` +Z: where a light trail starts) and `light` (the tube's middle), `use.look` `tube` and `grip`. A move's fist holds it by its `hold`, or `attach` puts it in a hand |
| `idol_stage` | a pastel idol stage: a riser on a rounded plan (`size` [6.0, 3.6], a superellipse) with a padded skirt bulging between a rolled foot and a rolled bullnose deck edge (gold piping round it), the flat deck at `height` (0.6 m); soft cushion steps (`steps` "front" (default, centred), "sides" (a flight on each side by the front corners, climbing toward the middle) or "none"; rise = height / n, n = ceil(height / 0.18)); LED panels at the back of the deck, each a rounded housing (rolled rim, piping bead) on slim posts with an LED dot-matrix face lit by a chase (default: a 2.2 m heart in the middle, a 1.3 m star either side, a 0.9 m heart outside each; `panels` = `[{shape = "heart" or "star", at = [x, z], size, pattern, color, alt}, ...]` replaces them, bigger panels standing further back); pixel bars on the front fascia under the bullnose, chased by the same field (see below). Root properties for `[[key]]`, also starting values in `slots`: `speed` (0.5 cycles a second: a card `rates` entry the keys stage integrates into `chase`; unkeyed `chase` runs at the starting speed from clip second 0), `chase` (the phase in cycles, keyable directly), `on` (1: the panels' level), `bars` (1: the light bars'). Options `pitch` (0.05 m: the dot pitch), `wave` (0.47 m: the bands' spacing), `glow` (2). Roles in `slots`: `deck` ("rose"), `skirt` ("iris"), `trim` ("gold"), `treads` ("rose:paper:0.4": the steps), `housing` ("overlay"), `heart` ("love"), `star` ("gold"), `heart_small` ("rose") (the default panels' lit dots), `alt` ("foam": the dim dots and pixels), `pixels` ("love": the bars' lit pixels), `metal` ("subtle": posts), `ground` ("ink": the LED faces). Origin floor_center, front -Y; form 0.00. Card: `use.rest` `deck` (a plane on the flat top in front of the panels' posts), `use.feet` `centre`, `left`, `right` (marks on the deck, her left at +X, `floor_z` = height: stand the cast at `[x, y, height]` with `feet = "<name>:centre"`), `use.surface` `heart` (the biggest heart's face, for [[text]]), `use.look` `centre` and `heart`, `rates`, `params`, `idol_stage` (the resolved panels, the flight, the bars). Objects `<name>_riser`, `_trim`, `_steps`, `_bars`, `_panel<i>`, all at identity under the root; materials `<name>_led<i>` and `<name>_pixels` carry the driven Value nodes `mk_chase`, `mk_on` / `mk_bars` |
| `fuse` | a cartoon fuse: a rope (`<name>_cord`, 14 mm thick) that leaves a socket at the root, up and over in a curl, with a spiky spark (`<name>_spark`) that is lit at its tip and burns it down. Root properties for `[[key]]`, 0..1: `grow` (how much of the rope has sprouted, along its length; default 1), `burn` (how much has burned down from the tip; 0), `lit` (the spark's size, flickering; 0). Leave `grow` at 1 in the slots and key it from 0: the props stage measures the form as built. Options `points` (`[[x, y, z], ...]` from the root, the first `[0, 0, 0]`), `radius`; colour roles `cord` (text), `spark` (gold), `cap` (base); `use.look` `base` and `tip`. The spark has no parent (it rides the rope's end on a Follow Path constraint and copies the root's world scale), so `scale` and `parent` carry it too |
| `burst` | a blast of pieces flung out from the root at a clip second: puffy hearts, five-point stars, cloud puffs and shards of glass (pillows facing -Y, and thin flat slivers with sharp edges, mirror-bright metal tinted their colour with a glow of their own: `<name>_heart<i>`, `<name>_star<i>`, `<name>_puff<i>`, `<name>_shard<i>`, the hearts tagged `mk_heart`), each easing out to its own reach while it pops in and spins in the picture's plane, then shrinking away; the directions are flattened toward the picture (x-z), so pieces fly across it more than at the lens. Drivers on the frame (the maths: `mkmmd/core/burst.py`), so a [freeze](#freezes) holds them mid-air. Options `count` (24), `seed` (1), `mix` (`{heart = 0.45, star = 0.35, puff = 0.2}`; `{shard = 1}` for broken glass), colours `heart` ("love"), `star` ("gold"), `puff` ("text"), `shard` ("text"), `glow` (0.3); root properties for `[[key]]`: `start` (the clip second it goes off; 0), `reach` (how far the farthest piece flies; 1.2 m), `life` (1.4 s), `spin` (360 degrees over a life), `size` (the largest piece; 0.16 m), `amount` (1; 0 hides every piece). A `[[key]]` on `start` with a `lane` sets it off again on every kick |
| `confetti` | paper confetti and streamers fired from the root in volleys on the hits: gently curled squares, hearts and five-point stars (flat paper die-cuts: `<name>_square<i>`, `<name>_heart<i>`, `<name>_star<i>`) and curly streamers (a paper strip curled into a helix, `<name>_ribbon<i>`, shot out along its flight and unrolling as it goes), matte two-sided paper in the `colors` with a small `glow` (tiny glints, never a flash), all tagged form-exempt (paper is flat by nature). Each piece flies out in a cone of half-angle `cone` about `aim`, slowed by linear air drag, tumbling and fluttering as it rains down, and shrinks away at the end of its `life` or as it comes down to `floor` (no part of it ever passes it). Drivers on the frame (the maths: `mkmmd/core/confetti.py`), so a [freeze](#freezes) holds them mid-air. Options `count` (60 per volley, up to 200), `volleys` (3, 1 .. 4), `seed` (1), `mix` (`{square = 0.55, ribbon = 0.15, heart = 0.15, star = 0.15}`), `colors` (`["love", "gold", "foam", "iris", "rose", "pine", "text"]`: palette slots, `#hex` or `a:b:t` mixes, shared out evenly over each volley), `aim` (`[0, -0.25, 1]`: up, a little toward the camera), `cone` (24 degrees), `ribbon` (0.5 m: a streamer's length), `glow` (0.25); root properties for `[[key]]`, also starting values in `slots`: `start` (0: the clip second it goes off), `reach` (3.8 m: how far the push carries the fastest piece, gravity aside), `drag` (0.5 s: how soon the air slows it), `fall` (0.7 m/s), `flutter` (0.14 m: the sway as it falls), `sway` (0.8 Hz), `spin` (1.2 rev/s: the tumble), `size` (0.05 m: the largest piece), `life` (6.5 s), `floor` (its height in the prop's frame; -1e4: none), `amount` (1; 0 hides every piece). With the defaults a volley shot from 1 m off the floor rises 2 to 3 m and rains down over about 5 s. Volley k reads `start<k>`: a `[[key]]` on `start` with a `lane` (`lane = "accent"`) is dealt round the volleys by the [keys stage](#keys) (the card's `deals`), so each volley falls its whole life while the next goes off (keep `life` under the time `volleys` hits take, or a volley that goes off again leaves the air at once); unkeyed, `start0` is `start` and the others never go off. Keep the prop upright: its pieces fall along its -Z. Card: `size`, `bounds` (the whole flight), origin `center`, front `-Y`, `blocks` false, `params`, `deals`, `confetti` {objects, volleys, count, fps} |
| `squeaky_hammer` | a squeaky toy hammer, 320 mm long: a pleated bellows head (180 mm, its axis along X) with a cap at each end, on a handle standing along +Z from its end at the origin; colour roles `head` ("love"), `caps` ("gold"), `handle` ("gold"); `use.look` `head` and `grip`. Root property `squash` (0..1) for `[[key]]`: the head 40 % shorter and 18 % rounder about its middle (key it up on each hit and back down). Put it in a fist with `attach`, `offset` and `attach_rot`, and play `bonk` with that hand (see [Moves](#moves)) |
| `spike_mallet` | a giant wooden mallet, 1.24 m, that reads both as the moon rabbit's mochi pestle (a yokogine) and as a spike maul: a turned barrel head 0.46 m long and 0.24 m across (fuller at its middle, the faces crowned and their rims rolled, end grain on the faces) with an iron hoop near each face, its axis along X and its middle at z 1.12, the +X end the striking face, on a straight turned handle standing along +Z from its end at the origin (a swell at the end the lower fist rests on, a flare into the head, the tenon's end grain and wedge on the head's top); front -Y. Colour roles `wood` ("gold:ink:0.45"), `grain` ("gold:ink:0.6": the end grain), `band` ("ink": the hoops), `handle` ("gold:ink:0.3"). Static. Card: `use.grip` `hold` (type "hold": `center` [0, 0, 0.4], `axis` [0, 0, 1] toward the head, `length` 0.64, `radius` 0.0208, `profile` [[s, r], ...] from the head end), `use.anchor` `face` (the middle of the striking face [0.23, 0, 1.12], `dir` [1, 0, 0]), `use.look` `face`, `head` and `grip`. Swing it with `pound` (see [Moves](#moves)) (spike_mallet_layout.py) |
| `golden_spike` | the ceremonial last spike of 1869 at its real size, 5 5/8 in (0.1429 m), standing on its TIP at the origin along +Z (so `at` a tie's top less how deep it is driven stands it in the tie), front -Y: a cut spike's square shank (17/32 in, rounded arrises, tapering a little) with a chisel point whose edge runs across X, a flared neck and a 25 x 19.5 mm head with rolled edges and a crowned top; round the shank's middle an engraved band between two raised beads (interlaced waves and dots cut in by the material: no lettering). Option `length` (m, 0.1429; 0.09 .. 0.40: the shank stretches, the head, point and section keep their size); `[[prop]] scale` scales it all. Colour roles `metal` ("gold"), `engraving` ("gold:ink:0.5"). Static. Card: `use.grip` `hold` (type "hold": the shank from the point to the neck, `axis` [0, 0, 1] toward the head, `radius` the section's, as long round), `use.anchor` `tip` ([0, 0, 0], `dir` [0, 0, -1]), `use.look` `head` (the crown) and `band` (golden_spike_layout.py) |
| `toothbrush` | a giant manual toothbrush, 0.74 m (about 3.9 times a 190 mm brush, the handle kept fist-thick: 45 x 32 mm), standing on the end of its handle along +Z, the bristles facing -Y: ONE cage of superellipse rings (handle, neck angled forward, head) and 36 lathed tufts; colour roles `handle` ("iris"), `grip` ("rose": the soft panels), `bristles` ("surface"), `tufts` ("foam": the middle columns). Card: `use.grip` `hold` (type "hold": `center`, `axis` toward the head, `length` of the handle a fist may hold, `radius`), `use.anchor` `brush` (the middle of the bristles' tips, `dir` -Y), `use.look` `head` and `grip` (toothbrush_layout.py) |
| `toothpaste_tube` | a giant toothpaste tube, 0.60 m (a 100 ml tube at about 3.4 times), standing on its crimp along +Z, the open threaded nozzle at the top: ONE cage from the ridged crimp up the body (each ring an ellipse keeping the tube's perimeter as it flattens toward the seal), over the shoulder, the helical thread and the lip, down the bore; colour roles `body` ("surface"), `shoulder` ("love"), `paste` ("foam", in the mouth), five body bands `stripes` ("love,gold,foam,iris,rose"), `cap` ("gold", with `capped = true`: a ribbed screw cap). Root property `squeeze` (0..1, a card param; a [[ribbon]] `from` its nozzle keys it) drives the shape key "squeeze": the body pressed flat round the hold. Card: `use.anchor` `nozzle` (the mouth, `dir` +Z, `radius` of the paste it lets out, 0.02), `use.grip` `hold`, `use.look` `nozzle` and `body` (toothpaste_tube_layout.py) |
| `kaleidoscope` | a cardboard kaleidoscope toy, 230 mm long and 49 mm across, lying on its side on its two turned rims (the axis along +Y, 26.8 mm up, the eyepiece toward -Y; origin on the floor under its middle, so `at` on a table top or `place` lays it there). Printed paper with a coated sheen, the step of its seam underneath and wear where the rims rub: `pattern` ("roses": rose and clover motifs, "memphis" or "marbled"), `seed` (1), `wear` (0.6: how far the print is rubbed through there), `print` (`{role = slot}`: the print's own colour roles, `mkmmd/core/kaleidoscope_print.py`). The eyepiece: a turned cap with a painted dish, a brass eyelet round a 7 mm eyehole and a domed lens (its glint), the mirrors' dark tunnel and the view glowing at its end (`glow`, 1.2). The far end: a frosted window with beads and glass bits settled behind it (`beads`: their slots, "love,gold,rose,pine,foam,iris"). Colour roles `rim` ("love"), `face` ("gold"), `ferrule` ("gold"), `frost` ("surface"), `inside` ("text"). Root property `turn` (degrees, for `[[key]]`): the chamber and the view turn about the axis. Card: `use.look` `eyepiece` (the eyehole's centre: aim a camera there), `chamber`, `grip`; `use.anchor` `eyehole` (`radius` 3.6 mm, `object` `<name>_eyehole`: an empty there, so an insert's `anchor` is `'obj("<name>_eyehole").loc'`); `use.rest` `side` (an edge along the top of the tube: a hand rests on it with `grip = "rest"`); `use.grip` `tube` (a cylinder wrap round the paper, as the mug's `body`; to hold it up, `attach` it to a hand as the mic). The maths and the card: `mkmmd/core/kaleidoscope.py` |
| `pillow_cloud` | a plump pillow (0.56 x 0.17 x 0.38 m: two stuffed panels sewn round a piped seam, the sides pulled in, folds at the ears, a polka-dot cotton case) that puffs into a cartoon cumulus (about 1.27 x 0.61 x 0.53 m, a flat soft underside): `<name>_pillow` and the cord in its seam `<name>_piping`, each with the shape key `cloud` (the same points on the cloud: `pillow_cloud_layout`). Held upright: broad faces +-Y, front -Y, origin at its centre. Root property `puff` for `[[key]]` (0 the pillow, 1 the cloud, up to 1.25 overshoots; its starting value in `slots`): the shape and the materials follow it (the print fades into a softly glowing white fluff, its underside shaded) and the cord sinks into the cloud. Roles in `slots`: `case` ("foam"), `dots` ("surface"), `piping` ("rose"), `cloud`, `shade`. Card: `use.anchor` `hug` (the middle of its back, `dir` [0, -1, 0]: the way its front faces), `use.rest` `hold_L` / `hold_R` (planes on its +X / -X ends, normals out), `use.look` `centre`, `cloud` {size, bounds} |
| `giant_carrot` | a giant carrot, 0.60 m tip to shoulder (0.10 across it), standing on its TIP at the origin along +Z, a green-tinged domed crown and seven leafy fronds to ~0.33 m above the shoulder, faint rings and root-hair dimples; front -Y. Root properties `bite1` .. `bite4`: the clip second of each bite (default 1e4 = never; key CONSTANT, in order); bite k eats the end up to a scalloped, tooth-notched face centred 0.06 k above the tip (non-additive shape keys `bite<k>` on `<name>_body`, carved in over 0.05 s; the face shows the paler core rings) and throws its own chunks `<name>_chunk<k>_<j>` that fly, tumble and shrink away within 0.6 s, falling toward the world's floor however the carrot is held. `f0`, `fps` set at build. Roles in `slots`: `skin`, `flesh`, `core`, `crown`, `stem`, `leaf` (true colours). Card: `use.grip` `hold` (type "hold": `center`, `axis` [0, 0, -1] toward the tip, `length` of the grippable shaft, `radius` at its shoulder end, `radius_tip`, `profile` [[s, r], ...]), `use.look` `grip`, `tip`, `use.bite` `bite1` .. `bite4` (`point`, `dir` [0, 0, -1], `prop`) |
| `carrot_plush` | a soft toy carrot, 0.34 m tip to top plus three stuffed felt leaves ~0.12 m above it, five minky gores with seam valleys, an embroidered face on its front (eyes with highlights, a smile, cheeks: graphic layers); origin at the body's middle, +Z tip to leaves, face -Y. Roles in `slots`: `body`, `leaf` (true colours), `face` ("text"), `shine` ("surface"), `cheek` ("love"), `stitch` ("base"). Card: `use.anchor` `hold` (the middle of its back, `dir` [0, -1, 0]: the way its face looks), `use.look` `face` |
| `dinner_plate` | a glazed ceramic dinner plate 0.27 m across (foot ring, flat well, the rim flaring up to a rolled lip, a coloured band on the rim) with stylised veggies in true colours (two broccoli florets, carrot coins with core rings, a heap of peas, a cherry tomato with its calyx); origin floor_center, front -Y. Options `veggies` (true; false: empty), `seed` (1). Roles in `slots`: `plate` ("surface"), `band` ("foam"), `floret`, `stalk`, `skin`, `flesh`, `core`, `pea`, `tomato`, `calyx`. Card: `use.rest` `well` (plane, `radius` 0.08), `use.look` `centre`, `use.grip` `rim` (type "pinch" at the back edge: `width` the rim's thickness, `span`, `length`, `center`, `axis` along the edge, `normal` out along the rim) |
| `record_treadmill` | a giant turntable to run in place on: a 1.68 m platter with strobe dots on its bevel on a low rounded-square plinth, a 1.60 m record (grooves and their sheen, a raised bead and label area), an original printed label (sunburst, rings, three beets; no lettering) and the spindle. Origin on the floor under the spindle, front -Y; the record's top at 0.16 m. It turns clockwise seen from above, so at the run spot the top moves +Y (1.57 m/s at 33.3 rpm). Roles in `slots`: `plinth` ("foam"), `deck`, `feet`, `plate`, `rim`, `dots`, `vinyl`, `label` ("gold"), `rays`, `print` ("love"), `leaf`. Root property `rpm` (33.3; also a starting value in `slots`; the keys stage integrates it into `turns`, so a spin-up never jumps; unkeyed it turns at the starting rpm from clip second 0). Card: `use.feet` `run` (L [-0.35, 0], R [-0.55, 0], floor_z 0.16: stand the cast at [-0.45, 0, 0.16] with `feet = "<name>:run"`), `use.rest` `top` (plane, radius 0.78), `use.surface` `label` (rides `<name>_platter`), `use.look` `centre`, `rates`. Objects `<name>_plinth`, `_feet`, `_platter` (turns) carrying `_record` and `_label` |
| `sandbox_piano` | a children's wooden sandbox (2.2 x 2.4 m, stacked boards 0.30 m tall with rounded edges, triangular corner seats, bumpy sand at 0.22 m) holding a toy grand piano (two octaves of chunky keys between cheek blocks, a rim on a grand's plan with a gold plate and strings inside, the lid propped open 38 degrees on a slanting stick, a music desk, three turned legs and the pedal lyre going into the sand) and its padded bench. Keys 0.70 m and seat 0.45 m above the sand. Origin floor_center, front -Y; the pianist sits facing -Y with the bass at her left (+X), the lid opening to her right. Roles in `slots`: `wood`, `seats`, `sand`, `piano` ("love"), `lid` (= piano), `inside`, `plate` ("gold"), `trim`, `keys_white`, `keys_black`, `bench` ("iris"), `bench_wood`. Card: `use.sit` `bench` and `use.feet` `bench` (as the café chair's: `sit = "<name>:bench"`), `use.rest` `keys` (an edge on the white keys) and `sand` (a plane), `use.look` `keys`. Static |
| `chalkboard` | a school chalkboard on an A-frame easel (about 1.67 m tall): rounded wooden legs, the back leg hinged at the top under a turned knob, a crossbar, the ledge with a chalk tray (white and pink sticks, a stub, a felt eraser), and a green-black slate (0.90 x 0.65 m) in a rounded wooden frame leaning back 12 degrees, with faint chalk dust. Origin floor_center, front -Y. Option `drawing` ("none"; "noughts": a game of noughts and crosses in chalk, a graphic layer). Roles in `slots`: `slate` ("#26332d"), `wood`, `chalk` ("paper"), `chalk_pink`, `felt`, `eraser` ("pine"). Card: `use.surface` `board` (the slate's face: `[[text]] on = "<name>:board"` writes on it), `use.rest` `tray` (an edge on the tray's lip), `use.look` `board`. Static |
| `parent_cutout` | a giant picture-book parent cut out of construction paper, standing on its foot line (origin floor_center, faces -Y, +X the figure's left): `figure` "father" (default: narrow-lapel suit, slim tie, side-parted hair, his left forearm raised with the scolding index finger up, 3.29 m) or "mother" (A-line shift, Peter Pan collar, hem band, bouffant with a flip and a bow, hands on hips, 3.16 m); scissor-cut paper with layered details on a thick white die-cut card with rolled edges and a slight bow, a soft drop shadow (`<name>_shadow`, a graphic layer about 6 mm behind: it shows against a backdrop), brass paper-fastener brads. Roles in `slots`: `paper` (pine / rose), `shade`, `hair`, `detail`, `trim`, `accent`, `rim`, `shadow`, `brad`. Father: `<name>_hand` turns in the card's plane about the elbow's brad by angle = wag + wagging sin(2 pi wag_hz t) (drivers); root properties `wag` (degrees, counter-clockwise seen from the front; key it), `wagging` (0), `wag_hz` (2.5). Card: `use.hand` `wag` {object, pivot, axis [0, -1, 0], tip (the index fingertip at angle 0), prop "wag"} (father), `use.look` `head` |
| `colorbar_card` | a rounded 4:3 TV-screen card printed with colour bars (`mkmmd.core.colorbars`: the face `texture` at 1024 x 768 as an sRGB byte image, the back `back()`), edges rolled, its cut edge cream paper (role `edge`); origin center, the bars face -Y; options `size` ([0.60, 0.45]), `thick` (0.012), `style` ("both", "dawn", "true"); the bars take the palette's slots (rose-pine-dawn without one). No drivers: key the root (the `cards` frame effect tumbles its own). Card: `use.surface` `face` (the printed rectangle), `use.look` `centre` |
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

**`butterflies`**: `card = "library:butterflies"`, `at` the mandala's centre (it faces -Y like a character); with `attach = "<cast>:head"`, `offset`
and `level = true` a swarm rides her head and stays level while she nods. Build options (the `slots` of the `[[prop]]`; the colour roles of
`butterfly`, `ground`, `under`, `ink` ..., recolour every style):

| Option | Default | Meaning |
|---|---|---|
| `ring_counts` | `[8, 16, 24]` | butterflies per ring, the inner one first; or `count` (1 .. 500): rings made for that many (K rings with 4 K (K + 1) >= count, ring k's share as k + 1: 100 is `[8, 12, 20, 28, 32]`) |
| `inner`, `ring_gap` | 0.5, 0.28 m | ring k's radius is `inner + k * ring_gap` |
| `cup` | 0.3 m | the outer ring that much further back (+y) than the inner, a bowl opening toward the camera (seen from below it shows its depth); negative: a dome |
| `heading` | 0 degrees | where the heads point, from outward (0: petals) toward each ring's direction of motion (90: flying along it) |
| `formation` | `"rings"` | the starting weights: that formation 1, the others 0 |
| `styles` | all four, equal | the mix `{style = share}` of `swallowtail`, `monarch`, `morpho`, `blue`. The mandala is n-fold symmetric (n the greatest common divisor of the ring counts): one sector takes the styles in proportion and a seeded colourway each, and repeats round every ring |
| `seed` | 1 | phases, rates, spans, orbits, drift paths, burst ways |
| `size_var` | 0.1 | each span within `size_var` of the mean (times its style's: a morpho is the biggest, a blue the smallest) |
| `phase_spread` | 1 | 0 .. 1: wing beats all over the cycle with rates 1 +- 15 % (times the style's); 0: all flap in unison |
| `orbit_at`, `orbit_tilt` | the root, 30 degrees | the orbits' centre and the inclinations' range (+-) |
| `drift_r` | 0.9 m | the drift ellipsoid's radius (half of it in y, 0.8 of it in z) |
| `level` | false | an empty `<name>_level` carries the swarm with a Limit Rotation (world space, x and y at 0): yaw follows, tilt does not |

Live root properties for `[[key]]` (also starting values in `slots`):

| Property | Default | Meaning |
|---|---|---|
| `rings`, `orbit`, `drift` | 1, 0, 0 | the formations' weights: positions, heads and backs blended by them (normalised; all 0 means rings) |
| `burst` | 0 | 0 .. 1: each flies out along its way o_i = normalize(normalize(p + eps_i) + jitter_i + 0.35 Z) by `burst_reach` v_i s^2 (v_i 0.7 .. 1.3), its head turning to o_i over the first 15 %, beating `burst_flaps` s extra beats, shrinking to nothing from 0.85 to 1. Keyed back down they fly back in on the same paths |
| `bloom` | 1 | 0 .. 1: ring k grows in over its window (0.4 of the bloom, inner rings first, the last ending at 1) out from the root on a short spiral; orbit and drift butterflies by their ring's window; 0 hides them all |
| `spin` | 0.05 turns/s | the inner ring's speed (a rate into `turns`); ring k turns `(-1)^k` at `(inner / r_k)^0.5` of it, an orbit at 3 times it (faster when tighter) |
| `radial` | 0 | 0 .. 1: every butterfly turned about its head axis by `radial` x 90 degrees (1: wings radial, spokes from the front) |
| `size` | 0.16 m | the mean wingspan |
| `flap`, `flap_amp`, `flap_mid`, `phase`, `hit`, `hit_time` | 3 beats/s, 100, 20 degrees, 0, -1000, 0.3 s | the `butterfly`'s: `flap` is a rate into `beats`; `hit` the clip second of the last hit (a `lane` keys it), after which every butterfly beats once more over `hit_time` |
| `ripple` | 0.25 s | a hit's beat starts `ripple * k / (K - 1)` later on ring k: it runs out through the mandala |
| `orbit_r` | 0.35 m | the orbits' radius (each within 15 %) |
| `burst_reach`, `burst_flaps` | 3 m, 6 | how far a whole burst carries them and its extra wing beats |

Clocks: the card's `rates` are `flap` into `beats` and `spin` into `turns` ([Keys](#keys)): unkeyed they run at the starting rates from
clip second 0; keyed, the keys stage writes their integrals, so a speed change never jumps a wing or a ring. The drift wanders on the clip
clock itself (slow sines, 0.06 .. 0.25 Hz). A butterfly's wing-beat phase is `beats * k_i + phase + offset_i + hit beat + burst_flaps * s`.

One object `<name>_swarm` is a mesh of one point per butterfly holding its seeded constants (and an `id`, so motion blur matches every
instance from frame to frame) under the geometry-nodes modifier "Swarm": the swarm's maths (`mkmmd/core/butterflies.py`, written once
against an ops interface: numpy there, one node per op here) runs once per frame on the points, and every butterfly's body and wings are
instances of hidden sources `<name>_<style><way>_body`, `_wing_L`, `_wing_R`, picked by its style and colourway; drivers carry the root's
properties onto the modifier's inputs and the scene frame onto its clip second (no Python at render time; a freeze holds them mid-beat). A
frame of 100 butterflies evaluates in about 5 ms. Card: `size` and `bounds` (the rings, orbits and drift; a burst flies out of them),
origin `center`, front `-Y`, `blocks` false, `rates`, `params`, `use.look` `centre`, `butterflies` {count, rings, styles, combos, objects,
sources, level, inputs}.

```toml
[[prop]]
name = "swarm"
card = "library:butterflies"
at = [0.0, 0.25, 1.3]                     # round her head and shoulders, behind her
slots = { ring_counts = [8, 16, 24], heading = 20, styles = { monarch = 2, morpho = 1, swallowtail = 1 } }

[[key]]                                   # out on every kick, drifting back in until the next
target = "swarm"
prop = "burst"
lane = "kick"
pulse = [[0.0, 0.0], [0.8, 1.0]]

[[key]]                                   # one more wing beat on every snare, rippling out ring by ring
target = "swarm"
prop = "hit"
lane = "snare"

[[key]]                                   # the mandala opens into orbits over a bar
target = "swarm"
prop = "orbit"
interp = "LINEAR"
keys = [[12.0, 0.0], [15.0, 1.0]]
```

**`rail_track`**: the live root property `laid` (metres along the centre line from the start, keyable) is how far the crews
have come, and nothing exists past it. As it passes (`mkmmd/core/rail_track.py`, numpy, tested; the meshes and the card in
`rail_track_layout.py`):

- a bed piece (one tie spacing long, `<name>_bed<k>`) grows from its start, the toe of its front ramp at `laid`;
- a tie (`<name>_tie<i>`) appears when `laid` reaches its near edge, 0.45 m over its seat and tipped a few degrees, and
  drops onto the bed with a small bounce within the next 0.4 m;
- a rail span (`<name>_rail<j>`: both rails, with the fish plates and bolts of the joint at its start) appears once the
  last one has landed, riding a rail's height up on it, and slides along the track with its nose 0.45 m behind `laid`
  (behind every tie still dropping) until its nose reaches its end, then lands with a small bounce within the next tie
  spacing (the first rail drops in at its place; round a curve a span turns about the circle's centre, so it slides
  along the arc);
- the spikes of the ties under it (`<name>_spikes<i>`) pop up out of the ties a tie spacing later, a wave along the rail.

Every whole number of tie spacings is a clean rest (no tie in the air, no rail landing, no spike half out; a rail may wait
lifted, sliding), so key `laid` in steps of a tie spacing (the card's `track.spacing`, 0.6096: a tie per hit) or of a rail
(`track.rail`, 9.144: a rail per hit) from 0, each step gliding over part of the time between hits: a counting key does it
(`step`, [Keys](#keys)). The track is whole from `track.done` (its length plus two spacings) on, the default starting value:
leave it so in `slots` (the props stage measures the form as built) and give the counting key `start = 0`. Every part is
moved by drivers on `laid` alone (Blender simple expressions; no Python at render time, so a [freeze](#freezes) holds the
laying mid-drop); the meshes are shared (six ties, one set of spikes, a bed piece, a mesh per rail length). The empty
`<name>_front` rides the front at the rails' top (target `"track:front"`). Keep `length` to what the camera sees: every
tie spacing is three objects.

```toml
[[prop]]
name = "track"
card = "library:rail_track"
slots = { length = 60, radius = 400 }        # whole as built; the key lays it from 0

[[key]]                                      # a tie per hit of the chug, each step gliding 0.3 s
target = "track"
prop = "laid"
lane = "chug"
step = 0.6096                                # track.spacing; 9.144 (track.rail) lays a rail per hit
start = 0
pulse = [[0.0, 0.0], [0.3, 1.0]]
interp = "BEZIER"
```

**`idol_stage`**: the panels' faces and the bars' lens are LED dot matrices drawn by their materials. `idol_stage_layout.py`
writes the field once against an ops interface: numpy is the tested reference, each material the same maths as one shader
node per op. The dots sit on a grid of `pitch` in the stage's x-z plane, round on the dark `ground` (every object stands at
identity under the root, so a material's Object coordinates are the stage's frame: mkr draws no Texture Coordinate node
with an object set). Each dot cell has a coordinate u by its panel's `pattern`: `sweep` x / `wave` (left to right in the
stage's frame, so one sweep runs on across panels and bars), `rise` (bands up the panel, each holding the same share of
its dots), `ring` (scaled copies of its outline running out from its middle), `spin` (spiral arms, 2 on a heart and 5 on
a star, turning counter-clockwise), `blink` (alternate dots); its lit level is a band travelling with `chase`, level =
band(fract(chase - u)), and a dot shows `color` x level + `alt` x 0.1 (1 - level), times `glow` x `on` (`bars` for the
pixels, which run the sweep). Flash-safe: ring, spin, rise and blink spread u evenly over a panel's dots, so as the bands
move its light holds within a few per cent; a sweep holds several bands across a panel (keep `wave` under half the width
of a panel that sweeps). Key `speed` in cycles a second (a band per beat is bpm / 60), or `chase` itself to step it on
hits.

```toml
[[prop]]
name = "stage"
card = "library:idol_stage"

[prop.slots]
steps = "sides"
speed = 1.2                                  # a band per beat at 72 bpm
panels = [
  { shape = "heart", at = [0.0, 2.25], size = 2.2, pattern = "sweep" },
  { shape = "star", at = [-1.8, 2.6], size = 1.3, pattern = "sweep", color = "foam" },
  { shape = "star", at = [1.8, 2.6], size = 1.3, pattern = "sweep", color = "foam" },
]

[[cast]]
name = "rin"
asset = "my_model"
at = [0.0, -0.288, 0.6]                      # the deck's centre mark (card use.feet `centre`)

[pose.rin]
feet = "stage:centre"

[[key]]                                      # the panels dark until 2.4 s, up over 0.2 s
target = "stage"
prop = "on"
keys = [[0.0, 0.0], [2.4, 0.0], [2.6, 1.0]]
```

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
| `wheelbase` | metres, for the steering angle; replaces the card's `steering.wheelbase` (default the card's, else 2.6) |
| `steer_ratio` | angle of the steering wheel per angle of the road wheels; replaces the card's `steering.ratio` (default the card's, else 14) |

The prop's card says what spins and steers, with two keys of its own:

| Key | Meaning |
|---|---|
| `wheels` | `[{object, radius, axis}]`: each object turns by distance / `radius` about its local `axis`, which points to the car's left (default `[1, 0, 0]`), so the tyre's top moves forward |
| `steering` | `{object, axis, ratio, wheelbase}`: the object turns about its local `axis` (default `[0, 0, 1]`) by `atan(wheelbase × curvature) × ratio` (`ratio` default 14; `wheelbase` the vehicle's, metres, default 2.6) |

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
| `hips` | `{shift = [x, y, z], roll, yaw, pitch}`: moves and turns the pelvis of a standing or seated body, eased in over the settle and on top of what a seat asks for. `shift` is in metres in the character's own frame (x its left, y behind, z up): a drop bends the knees and the leg IK keeps the feet on `feet`. `roll` degrees drops the left hip (+). `yaw` degrees turns the pelvis toward the left about the vertical. `pitch` degrees tips it forward (+): the thighs fold at the hip joints with the feet planted, so with `lean` and a `shift` back the body hinges at the hips (bent over a table, hands on the knees) instead of curling at the waist. Keyed on `center` (shift) and `lower_body` (roll, yaw, pitch) |
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
stays straight; `thumb = "tuck"`, `"ring"` or `"cross"` solves it, below). The presets bend (fingers / thumb, degrees per joint)
`relaxed` 14, 22, 14 / 8, 10, 8; `curled` 35, 50, 35 / 14, 18, 14; `fist` 80, 95, 65 / 25, 35, 40, and `point` its three curled
fingers as `fist` (`mkmmd/core/fingers.py`). The rotations turn about each finger's own flexion axis (`joints`). Curled that way
a thumb only swings out of the palm's plane: it can neither cross the palm nor turn its pad to a fingertip, so a thumb that
must touch something is solved per hand from its rest bones and skin:

- `tuck` (`tuck_thumb`): it folds across the curled middle finger, its tip resting on that finger's middle and end segments
  (22 % of the finger's length off its bones) on the middle finger's side of the index, never standing up beside it, the rest
  of it clear of both fingers, with the least turning (its base joint costing most).
- `ring` and `cross` move it as a thumb moves (`opposed`): its base swings out of the palm's plane and across it and rolls
  about its length (opposition), its other two joints bend toward its pad (the side its rest chain bends to); `oppose_thumb`
  solves those five angles within a thumb's range, its joints clear of the index and middle fingers, with the least turning.
  `ring`: its fingertip meets the index's, the two last segments rounding a ring (an OK sign). `cross`: its last segment lies
  skin on skin across the index's last one on the thumb's side, both tips just past the crossing (the finger heart).

The fingers end where their skin does, not at their tip bones (`finger_end`: the farthest skin the last bone moves, and that
skin's radius): a model's tip bone may stand a centimetre past the fingertip, so tips that meet as bones are apart as skin.

#### Hands

`[pose.<cast>.hands.L]` and `.R` put an arm on a target. The goal empty `<cast>_hand.<L|R>` (collection `Rig`) is the wrist bone's
tail and orientation: a position-only IK on the forearm (`mk_arm_ik`, chain up to the upper arm, its target a child of the goal
at the wrist's head) bends the elbow toward the pole empty `<cast>_elbow.<L|R>`, the wrist copies the goal's rotation
(`mk_hand_rot`) and the forearm twist bone, when the rig has one, rolls with the hand (`mk_forearm_twist`: the rig spreads it
down the forearm, so a sleeve turns with the hand instead of the hand turning inside the cuff), and the twist bones are held in
range whatever drives the arm, the IK, a motion strip while the hand is let go, or both at a hand-off (a clamp, so the turn
levels off rather than jumps): the forearm's within -75 to +85 degrees of MMD's rest roll (`mk_forearm_limit`: the range of
`mkmmd.core.armtwist.FOREARM`, -90 to +100, + supination, less the 15 degrees the wrist may turn the hand on past it,
`mk_wrist_roll`), the upper arm's within 85 either way (`mk_arm_twist_range`), and within 75 while the IK holds the arm
(`mk_arm_twist_limit`; the twist bone is left out of the IK). A goal that asks more leaves the hand short of its roll instead of
wringing the sleeve into a flat pillar, and the moves library turns the elbow so that none does ([Moves](#moves)); a dance's
own shrug past the range is levelled off the same way.
The wrist's head is held on the forearm's end (`mk_wrist_on_arm`: rigs put it a fraction of a millimetre off the twist bone's
axis). The IK's pole angle is solved per arm from its rest pose, with the forearm pre-folded 20 degrees toward the front (the
solver starts from that fold), so the elbow points at the pole on either side of any rig. That fold makes the IK's elbow a
hinge whose crease faces forward at rest: MMD's rest roll (palms down, thumbs forward) is the forearm's mid-prone neutral, and
`mkmmd.core.armtwist.twists` predicts both twists of an IK-posed arm from its joints and hand within about a degree (the same
definitions `joint_limits` measures). These empties are the names checks and `[[key]]` address. A hand needs one of
`at`, `rest`, `grip` or `keys`. The goals of a seated character ride with its seat prop; a standing character's stay in the world
unless `ride` says otherwise. The wrist goal is keyed at the end of the settle (and at the keys' times), the influence eases
in over the settle.

| Key | Meaning |
|---|---|
| `at` | where the wrist joint goes: a [target](#targets). The hand points along `dir` with the palm toward `palm` |
| `dir` | `[x, y, z]` world direction the hand points along, wrist to fingertips (default straight ahead of the character) |
| `palm` | `[x, y, z]` world direction the palm faces: the hand rolls about `dir` to match (default the least rotation from rest; `[0, 0, -1]`, palm down, for `rest`) |
| `pole` | a target the elbow points at (on either side of any rig: the IK's pole angle is solved per arm from its rest pose). Default, from the shoulder: 0.45 m outward, 0.25 m behind and 0.30 m below it; on a guitar's `neck`, 0.15 m outward, 0.08 m behind and 0.40 m below; for a hand on its `strum` point the elbow is chosen with the grip (out from the shoulder, below it, the forearm clear of the card's `avoid` objects: over the body's edge) and the pole put where it points (0.45 m outward, 0.12 m behind and 0.08 m below when no elbow is allowed). The pole of a `neck` or `strum` hand is then turned round the shoulder-wrist line the least way (5 degree steps, at most 60) that keeps the arm's twists 10 degrees inside the human range ([Moves](#moves), the twist) at every grip of the hand, the fret changes and the two ends of a pick hand's widest stroke (`[perform]` strum's `span` and `accent_gain`), without bending the wrist past 55 degrees: a fretting hand down by the nut turned its forearm over past what it can from an elbow hanging under the shoulder, and a pick hand's upper arm turned in past its range at the end of a down stroke (the grip's digest gives `pole_turn_deg`); a `pole` given is kept |
| `rest` | `"prop:use"`: a `use.rest` point of a prop the hand lies on, an edge (`a`, `b`) or a plane (`center`); the palm faces `palm`, the heading is `dir` (default forward and slightly outward) |
| `along` | `rest` on an edge: 0..1 along it (default 0.5) |
| `offset` | `rest` on a plane: `[x, y, z]` metres from its centre, in the prop's frame (default 0) |
| `lift` | `rest`: metres above the surface along its normal (default 0.03; 0 for a `grip = "rest"`) |
| `keys` | `[{t, at, dir, palm, fingers, pole, sight}]`: moving targets, `t` in clip seconds, Bezier between the keys; `dir` and `palm` default to the hand's own; a key's `fingers` (as the hand's `fingers`) are reached at its time and eased between the keys that have them, from the hand's own at the end of the settle; a key's `pole` (a target) is where the elbow's pole is at its time, eased between the keys like the goal, the keys without one going back to the hand's own `pole` (the moves stage gives it to a place that holds the elbow its own way: a dainty hand alone, [Moves](#moves)). `sight = true` on consecutive keys holds the hand with the head from the first of them to the last (a second goal, `<name>_sight.<side>`, rides the head bone with their goals, and the hand's goal copies it, handed over in 0.1 s either side): a ring held at the eye stays on it as the head leans, tilts, nods and rocks (the moves stage marks a `peek`'s keys so). With `at` that pose holds until the first key; without it the first key's target is the hand's from the start. A neck grip's keys are `{t, fret, chord, move}` instead |
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
| `press` | neck: the share of the gap between two wires, behind the wire it plays, where a fingertip presses (default 0.3: the `finger_targets` point the solver puts the fingertip on) |
| `move` | neck: seconds a keyed change takes to land (default 0.12): the fingers ease off the strings over its first quarter, the hand moves (slides along the neck to a new position) and they land over its last quarter |
| `thumb`, `tip` | strum: the thumb lies on top of the pick, on the bass side, leaning 45 degrees toward the neck (`"neck"`, default) or the bridge (`"bridge"`); metres the pick's tip sticks out of the pads (default the card's `pick.tip`, 0.008) |

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
`finger_clash_mm`, `seconds` and the style's own numbers (`clock` and `axis_flipped` of a ring, `fret`, `chord`, `states`,
`knuckles_mm`, `pads_mm` (per target finger, its fingertip from its target: across the string, along the neck) and `from_ref`
(how far the hand moved from the shape it was solved from) of a neck, `thumb`, `tip_mm`, `forearm_clear_mm` and `pole` of a
strum, `wrist_bend_deg`, `reach` and `elbow` (`pole` or `chosen`) of both, `point` of a rest, `writing` and `track` of a pen
path), with `warnings` when it misses.
`attached` lists the props placed by `attach` and `anchor_to`, `cables` the worn props' cords, and `ik_error_mm` the miss of
every arm IK in millimetres, keyed `<cast>.<L|R>`.

The log has a line per grip (and `mkmmd.solvers.grip: cached ...` or `solved in ...s` for each solve) and `WARNING` lines when
something is off: a grip past its gates (a contact gap over 3 mm, penetration or finger clash over 1 mm: the prop does not fit
that hand) and a wrist that ends more than 5 mm short of its goal (out of reach: lean the character, move the seat or bring the
prop closer; an arm is about 0.38 m long for a 1.7 m model). The miss is measured, each hand on its own frames only, at
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
so leans, tilts, sways and the bounce carry the hands: a mic at the mouth stays there), lean, tilt, turn, head tilt, head
turn and head nod keys, twitches and expressions (`[perform.<cast>]`), so a move is keys like any other. A hand a move plays may
have no `at`, `keys`, `grip` or `rest` of its own in `[pose.<cast>.hands]`, and `lean`, `tilt`, `turn`, `head_tilt`, `head_turn`
and `head_nod` cannot be keyed both in `[perform]` and by a move (each is a build error); other keys of the same tables stay
the project's (a `ride` of the project's own wins).

| Key | Meaning |
|---|---|
| `name` | the move (below, required) |
| `t` | clip seconds the move starts (required) |
| `dur` | seconds it lasts (default 1) |
| `hand` | `"R"` (default) or `"L"`: the hand of a one-hand move |
| `morph`, `value` | a `face` move's morph (semantic or the model's own name) and its value (default 1); on any other move, a face held over it on top of the move's own |
| `hits` | a beat move's own clip seconds to play on instead of the beats inside it (the snares, say); each must fall inside the move. A gag or mosh move's may be a lane of the [audio] hits file instead, `{lane, min_db, every, from, to}` (`from` / `to` default to the move's span), read by the build |
| `prop` | a gag move's held prop (`chomp`, `brush`; `yawn` hugs one), or `pound`'s mallet: a `[[prop]]` name. The move places it itself, so the prop has no `attach`, `wear`, `anchor_to` or `place` of its own: key its appearance (its `scale`, a pop) for the moments it should show |
| `chore` | `chores`' loop: `"sweep"` (default), `"stir"` or `"water"` |
| `accents` | `bounce`'s fists-up-and-hop moments: clip seconds or a lane table as `hits` (default: the timeline's downbeats inside the move) |
| `target` | `pound`: where the mallet's face meets, a [target](#targets) (`{cast = "rin", point = [0, -1.0, 0.43]}`: the top of a stake 1 m in front of her; a prop's use point), read where it stands when the moves are compiled; default: wherever her natural blow lands (about 1.2 m ahead at knee height) |
| `depth` | `headbang`: a share of a full headbang (default 1: the upper body 20 degrees down and the head 34 more on each hit; more than 0, at most 2) |
| `hands` | `headbang`: `"free"` (default: the move plays no hand, so they hang and swing with her, or play other moves) or `"hips"` (both on the hips) |
| `props` | `wotagei`: `{L = "<prop>", R = "<prop>"}`, the penlight in each fist (each a `[[prop]]` with `use.grip` `hold`, placed by the move like `prop`) |
| `routine` | `wotagei`'s: `"arcs"` (default), `"eights"`, `"punch"` or `"circle"` |
| `style` | `bass_face`'s: `"scowl"` (default) or `"pout"` |

`{name = "rest", hand, place, t}` says where a hand waits when no move plays it (and gives that hand its arm IK, so a walk's `arms`
swing it even when it plays nothing): `"rest"` (default: hanging by the hip; over a wide
skirt the arm swings out until the hand clears it), `"dainty"` (resting lightly on the front of the skirt or the thighs, elbows
in, the two hands together when both rest there: a girl's polite stand. It is a pose for two: a dainty hand alone, its
partner away (playing a move, a reaction or let go) or resting elsewhere, rests on the front of its own thigh instead, a few
cm out from the two-hand spot: the hand continues its forearm (its wrist bent 12 degrees at most), the fingers down the
thigh and the palm on the skirt, and the elbow is held out to the side (the hand's keys there carry the elbow's pole), so
the forearm and sleeve clear the hem of a jacket. It glides there from the spot as its partner leaves, arriving as the
partner gets to its move, and is back as the partner is, unless that stretch is shorter than 0.15 s; the partner's
absences closer than its round trip plus that count as one, so it does not go back for a moment), `"behind"` (the hands
clasped behind the back, the elbows back; alone, the back of the hand rests on the small of the back), `"clasp"` (the
hands clasped together at the chest, fingers laced, the elbows down; alone, the hand rests on the chest) — both poses for
two as `dainty` is —, `"skirt"` (each hand pinching the skirt at its side, a little curtsy; `skirt` and `behind` rest on
the skirt even where the sim moves it, so the hand is on the cloth, not under it) or `"mic"` (a fist under
the mouth, holding an attached mic that leans back up to the lips, so the forearm rises to it and the elbow hangs by the
side). Without `t` the entry is where the hand waits from the start; with `t` (clip seconds) it changes where it waits from
then on: the hand is at the new place on `t`, setting off before it as for a move, and a move it plays across `t` or that
leaves it too little time to get there, or to go on from there to its next move, is a build error saying how long the trip
takes (a move starting on `t` takes it there afterwards). A hand left at a pose for two when its partner changes place goes
to its lone place as the partner gets to its new one, and back as it returns, when it has the time (busy, it goes there
after its move). A hand at the mic holds it from the start to the end: its rest place does not change. A hand
travels at a human pace: a trip takes 0.1 s plus its wrist's way at 1.6 m/s (eased: it peaks near twice that), or as long
as turning its palm or its fingers' way at 600 degrees a second takes, if that is longer, and it reaches a move's first
place on the move's time, so it sets off before it. A reaction is the other way round: `flung` leaves where the hand is on
its own time (a hand does not see a blast coming) and is thrown there in 0.06 s plus its wrist's way at 4.5 m/s. A hand
holds its last place to the end of the move, then goes home to its rest place if it can stay there 0.15 s before its next
move, else straight on to that move, leaving its place early enough to get there on time; a next move too close for that
(the hand would have to jump) is a build error that says how long the trip takes. One hand cannot play two moves at once.
A hand that rests at the mic keeps the mic in its fist through every move it plays (the mic moves aim it); a two-hand move
leaves it at the mic and plays with the other hand only, except `hands_up` and `flung` (the mic goes up too), and
`heart_push`, `butterfly` and `bloom`, which need both hands free, are a build error for a member holding a mic.

| Move | What it does |
|---|---|
| `bounce_hand` | the hand out in front at the waist, dipping on each beat inside the move |
| `chest_pat` | the flat hand pats the chest on each beat; blush |
| `paws` | both hands, cat paws under the chin, dipping on the beats; `omega` mouth |
| `point` | the arm out, the index pointing at the camera |
| `heart_wink` | a finger heart by the cheek: the thumb laid across the index's last segment (a solved `cross` thumb), the two tips up from their crossing and its face (the thumb's side) to the lens; `wink_r`, `mouth_smile` |
| `peace_eye` | a peace sign by the eye; `wink_l` |
| `sparkle` | both hands open beside the face; `smile_eyes` |
| `drip_check` | the hand by the chest turns over halfway; `jito` eyes |
| `hands_up` | both arms up |
| `flung` | both arms thrown up by a blast, the hands leaving on its time; they come home when it ends, so end it as the body lands for the hands to come down with the landing |
| `mic_lens` | the (mic) hand held out at the camera, the mic upright and leaning to the lens |
| `mic_up` | the (mic) hand raised forward and up, the mic upright over it, the palm toward the head (a fist holds a mic across its palm, so a mic pointing forward from an upright forearm would twist the palm outward) |
| `mic_across` | the (mic) hand held out across the body to the other side, the mic upright (offered to a partner on that side) |
| `ears` | both hands over the ears, elbows wide; `hau` (">_<") face: too loud |
| `heart_push` | both hands make one heart in front of the upper chest: the fingers curl over its two lobes to meet at the top, the thumbs point down to its point, each fingertip half a finger's radius off the body's midline so the two hands' tips touch; the heart upright with its face (the hands' little-finger sides) to the lens; `cheerful` face. Needs both hands free |
| `bunny_paws` | both paws up over the head like bunny ears; `wink_r` |
| `bonk` | a toy hammer (held in the fist, see the library's `squeaky_hammer`) cocked up and back over the shoulder, then brought forward and down on each beat or hit, its head's face meeting what is in front at the height of the chest; it starts cocked (its wind-up), so start it a little before the first hit |
| `into_lens`, `lean_back` | the upper body leans 14 degrees forward; 11 back and 7 to the left |
| `stank` | the head flicks back; `hau` face |
| `blown` | blown back by a blast: the upper body leans 16 degrees back, the head snaps back; `surprised` face (the hands keep what they play) |
| `hair_flip` | a pose: the hand runs into the hair behind the ear, then flicks it out (at 40 % of the move), the fingers up and back, the palm up, the elbow kept out to the side so the upper arm keeps its turn out (a toss behind the head turned it past what it can), the head flicking the other way (a twitch for the hair to fly on) and tilted from the hand; `calm` eyes, a small smile |
| `over_shoulder` | a pose: a look back over the shoulder of the move's `hand` side, the upper body turned 25 degrees and the head 45 more that way; `calm`, a small smile. Shoot it from behind her |
| `hand_hip` | a pose: the hand on the hip, the elbow out, the head tilted toward it and the body leaning a little away; `smile_eyes`, `mouth_smile` |
| `finger_gun` | a pose: the arm out ahead at the height of the eyes, the hand a finger gun (index out, the thumb up, the rest curled), leaning in 6 degrees, the head tilted toward it, winking the other eye; `mouth_smile`. Read from a three-quarter view: straight on, the hand stands in front of her face |
| `free` | lets the hand go for the span to whatever else drives the arm (a walk VMD's swing; below) |
| `float` | the arm drifts out and down, then up and open over three quarters of the move, weightless (the hand relaxed, palm down); played on both hands a beat apart, with `air`, `rise` and a slow `tumble`, a body floats off the ground in zero gravity |
| `peek` | a pose: the hand up at its own eye, peeking through the ring of its thumb and index (a solved `ring` thumb: the tips meet), the ring's opening to the lens (a circle round the eye from the front) and the middle of its hole (the circle through the index's curl and the thumb's pad) on the line from the lens to the eye's visible pupil, 6.5 cm in front of it or as near as the face lets it come, the thumb toward the face and the other three fingers fanned up, 15 degrees in from upright, the elbow out to the side at about the shoulder's height so the forearm comes in from the side and the mouth stays clear (thumb to the lens, or the elbow low in front, the forearm would have to turn over past what a forearm turns); from its arrival to the end of its hold the hand goes with the head (keys marked `sight`), so the ring stays on the eye however the head moves; the other eye winked (whichever wink morph shuts it on the model: the build reads the morphs' shape keys), leaning in 4 degrees, the head tilted toward it; `mouth_smile` |
| `butterfly` | the shadow-puppet butterfly before the collarbone: the wrists crossed (the right hand 5 cm in front of the left, so the forearms pass), the backs of the hands to the lens, the thumbs reaching across the midline (overlapping there as the butterfly's body, a hand's depth apart) and the spread fingers leaning up and out as wings, below the chin so the face stays clear; on each beat (or `hits`) the wings fold 30 degrees forward about the thumbs, opening between; `smile_eyes`, `mouth_smile`. Needs both hands free; `hand_mirror` takes the crossing's 5 cm out |
| `bloom` | both hands cupped round a bud before the chest, then opening up and out at 40 % of the move, palms up, the fingers fanned like petals; `smile_eyes`, `mouth_smile`. Needs both hands free |
| `reach` | a pose: the arm reaching to the lens, the open hand's palm to it, fingers spread, leaning in 8 degrees, the head tilted toward it; `calm`, `mouth_smile` |
| `wings` | both arms out to the sides, the hands trailing, coming down on every second beat inside the move (or on its `hits`) and up between, like slow wings: for a floating body; `calm` |
| `blow` | the palm up in front of the mouth, then pushed on out and open at half the move, blowing something off it (a kiss, a butterfly); `u` lips (not over sung words), `calm` |
| `giggle` | a pose: the fingers over the lips, hiding a giggle, the head tilted toward the hand; `smile_eyes`, `blush` |
| `face` | only its `morph` |
| `yawn` | both fists stretched up high, then the arms opening out wide (at 55 % of the move), the back arched 9 degrees and the head tilted; the mouth opens slowly into a yawn (`a`, `o`) and closes by 90 %, the eyes shut (`calm`). With `prop` (a pillow: `use.anchor` `hug`, the middle of its back and the way its front faces, `use.rest` `hold_L` / `hold_R` planes on its ends, normals out, card `bounds`) she hugs it instead: it rides her chest upright, its back 12 mm off her body, its top 6 cm under her mouth, a palm flat on each end, leaning 6 degrees onto it with her head tilted |
| `chomp` | a held prop (`prop`, a giant carrot: `use.grip` `hold` and `use.bite` `bite1`, `bite2` ...) brought to the mouth and bitten on each hit (`hits`, else the beats; at least 0.25 s apart): bite k's point 8 mm behind the lips on the prop's k-th hit, counted on across every chomp of it in time order, the mouth wide open as it comes and shut on the hit, chewing after; between bites the prop tips 20 degrees down about the fist. The prop's root property the bite names (`prop`, default `bite<k>`) is set to that clip second (its bite carves in, its chunks fly then); more hits than bites is a build error. `smile_eyes` |
| `brush` | a held toothbrush (`prop`: `use.grip` `hold`, `use.anchor` `brush`, the middle of the bristles' tips and the way they face) across the mouth, the bristles on the teeth facing them, scrubbing 2 cm each way on the beats (or `hits`); held down its handle, 0.72 of the arm's reach from the bristles (a giant one too); `i` (the teeth bared), `smile_eyes` |
| `run_in_place` | a cartoon run: the knees come up in turn on each beat (or hit) and halfway between (perform's `walk` on the spot: stride 0, the feet coming down where they rose, half the arm's reach high), the fists pumping opposite them (the arm on the raised knee's side back), leaning 7 degrees in; even steps, so each run starts on the left knee. On a treadmill (the record) stand her on it: `[[cast]] at` its spot, `[pose] feet = "<prop>:run"` |
| `wag` | a scolding wag: the move's hand by the face, the forearm upright and the index up, tipped out on each beat (or hit) and in between, the palm to the lens; the other hand on the hip; `jito`, `mouth_down`, `angry` |
| `wink_point` | a pose: the arm out pointing at the lens (`point`), a wink of the other eye, leaning in 7 degrees with the head tilted toward the hand and a little nod on its start; `mouth_smile` |
| `bounce` | an EDM bounce: the knees pump on each hit (perform's `bounce` on them, 45 mm), both fists before the chest dipping with them, and on each accent (`accents`) both fists punch up and she hops: 7 cm over 0.3 s (`rise`), the legs tucking (`air`), the knees giving on the landing (`crouch`); an accent too soon after the last hop only punches |
| `chores` | simple mimed loops on the beats (or `hits`), the move's hand the main one: `sweep`, two fists up a broom handle sweeping to the low (other) hand's side; `stir`, a fist circling once a beat over the other hand cupped as a bowl; `water`, a fist round a watering can's handle held out low, tipped forward on each beat; leaning 8 degrees, `calm` |
| `pound` | a mallet (`prop`: `use.grip` `hold`, its handle, and `use.anchor` `face`, the middle of the striking face and the way it faces; the library's `spike_mallet`) in both fists side by side, the move's hand toward the head: raised overhead (its first place is the wind-up, so start it a little before the first hit) and swung down on each hit (or beat), faster and faster, the face meeting `target` on the hit while the body leans into the blow (`lean`, up to 45 degrees, and the hips drop, `crouch`, for a low target) and the knees give after it with the face held on the target; up again between, as far as the time to the next blow allows (all the way from 0.85 s, at least a third). The mallet rides the move's hand; `serious` |
| `headbang` | deep nods of the upper body (`lean`) and the head (`head_nod`) down on each hit (or beat) and up past level between, `depth` a share of a full one; `hands = "hips"` puts both hands on the hips, else they hang and swing with her (or play other moves). Start it a little before the first hit; give `[sim.<cast>]` the hair families and the hair whips by itself |
| `pogo` | a straight-up jump taking off on each hit (or beat), as high as a body flies in the time to the next one (a parabola, `rise`: 0.2 m at most, 8 cm on eighths of 73 BPM), the legs tucking (`air`), the knees giving on each landing and pushing off into the next (`crouch`), both fists up (`bounce`'s `pump_up`) dipping as she lands; a hit too soon after the last take-off is left out. Unlike `bounce`, where the knees pump on every hit and only the accents hop, every hit is a jump; mirrored hands; `cheerful`, `mouth_smile` |
| `wotagei` | a penlight in each fist (`props = {L, R}`) waved in a routine (`routine`) on the beats (or `hits`): `arcs`, both lights swept overhead from her left to her right and back, a side a beat, crossing at the top halfway; `eights`, each light drawing a figure eight before its shoulder, mirrored, through its middle on each beat (an eight every two beats); `punch`, the fists punched up and forward in turns on each hit, else on the beats and halfway between (the eighths), the other pulled in by the shoulder; `circle`, both arms circling in front, a turn every two beats, the right half a turn behind (a windmill). Each light rides its fist, out of the thumb's side; `cheerful`, `smile_eyes` |
| `whistle` | a train conductor pulling the whistle's cord: the move's hand up beside the head pulls it down on each hit (or beat) and lets it up between, the other hand on the hip, leaning back 4 degrees with the head tilted toward the cord (mirrored for the left hand); `cheerful`, `smile_eyes` |
| `bass_face` | a face on each hit (or beat), `style` a `scowl` (`angry`, `jito`, `mouth_down`) or a `pout` (`u`, `jito`, `troubled`; it has the mouth, so lip sync gives way), on in 0.05 s, held 0.1 s and gone in 0.22 s (between hits closer than that only down to half), with a short nod of the head (`head_nod`, 9 degrees) |

The poses (`hair_flip`, `over_shoulder`, `hand_hip`, `finger_gun`, `peek`, `reach`, `giggle`, `wink_point`) and `wag` and `yawn`'s tilt are written for the left hand and mirrored for the right:
their turns, tilts and the head's twitch go the other way, their lean does not. A move's body keys ease in over 0.12 s and back
over the last 0.12 s (the mosh moves key their own curves); two moves that key the same one of `lean`, `tilt`, `turn`,
`head_tilt`, `head_turn` and `head_nod` at once are a build error (a `bass_face` nods the head over a `pound`'s lean).

The gag moves (`yawn`, `chomp`, `brush`, `run_in_place`, `wag`, `bounce`, `chores`; `mkmmd/core/gags.py`) are played like the
others (their hands travel, rest, stay natural and keep off the body the same way) and also key the legs and the mouth: their
perform keys (`bounce`, `rise`, `crouch`, `walk`) cannot be the project's own as well (a build error), their `air` spans join the
project's, and `mouth` gets the spans where `yawn`, `chomp` and `brush` have it, so lip sync gives way to their vowels there. A held
prop rides the wrist of its hand (bone-parented at the end of the pose stage): the middle of its grip in the fist's hole (the
member's own fingers closed round a handle of the grip's `radius`, or of its `profile` there: [[s, r], ...] from the business end,
or `radius_tip` at that end) and its `use.grip` `hold` `axis` (toward the business end: a carrot's tip, a brush's head) out of the
fist's thumb side, as a fist holds a mic. The fist closes on the span `center` +- `length` / 2 where the move needs it (a carrot as
far from its tip as still brings its first bite to the lips with the wrist 0.88 of the reach from the shoulder, and 7 cm from the
deepest bite; a brush down its handle), and the first move that holds a prop sets how it sits in the fist (its front to the lens,
the bristles to the teeth): every place of it is then solved with that grip, the wrist moved until the prop's part is where the
place wants it. A prop is held by one hand (or hugged) for the whole build; the pose report's `held` names the bone it rides and
the moves report its `perform` keys, `held` props and the root `props` it set.

The mosh moves (`pound`, `headbang`, `pogo`, `wotagei`, `whistle`, `bass_face`; `mkmmd/core/mosh.py`) are played the same way:
their places join the library's (the cord and the penlights' places are fists round a handle, aimed as a mic is), their legs
(`rise`, `crouch`, `air`) the gag moves', and their own body curves (`lean`, `head_nod`) go to perform as any move's. A
penlight rides its fist held by the middle of its grip span, its light out of the thumb's side. A pound's mallet rides the
move's hand; the other fist closes round the handle beside it at every key, and the move keys its swing itself (at most 30
degrees of the handle a key, each elbow from the key before's), so no key is put halfway between two of them. Its blow is
solved in the frame the hands ride: the fists low before the belly, the handle tilted up 12 to 40 degrees there, and the
body's lean (5 to 45 degrees about the waist, the landmark `waist`: the upper body's joint), the hips' drop (to 0.12 m), the
tilt and where along the handle the fists hold it chosen as near their preferred ones (30 degrees, none, 26, the handle's
end) as bring the face within 15 mm of the target with the fists off the thighs; the chest then carries the face onto it,
its own `dir` down onto it. A target out of reach is a build error saying where the face can come down. The first pound
of a mallet sets how it sits in the fists; a later one holds it there and finds its own lean and tilt.

A beat move needs a beat inside it (the timeline's `beats`, `audio/timeline.json`) or its own `hits`. The hand places are made from the member's
rest pose (its arm joints, chest, mouth, eyes, the arm's two lengths and the hand's), so the library fits any body. Places by
the face and the chest are on the hand's own side, clear of a mic at the mouth and of its cord. A gesture (`peek`'s ring, the
finger heart, half of the two-hand heart, a butterfly wing: `mkmmd.core.moves.GESTURES`) is placed by the member's own hand:
the build reads its rest bones and skin (the landmark `fingers`: both hands averaged as the left), the core poses it in the
gesture's shape (`mkmmd.core.fingers.pose`) and turns it so the part that must read faces the lens the right way up, setting
the wrist so that part lands where it belongs (the ring's hole on the pupil's line to the lens, the heart's tips on the
midline, the thumbs hooked across it); a gesture comes in to the body straight from the front, so it stays on its line of
sight, and the two-hand ones keep the orientation they were given. The pupil is the landmark `pupil`: an MMD eye's iris is a
mesh of its own about a centimetre out from the eye bone and 4 cm in front of it, so the build takes the middle, by area, of
the faces whose every vertex the eye bone moves most (both eyes, averaged as the left; a model whose eyes no face follows
falls back to the eye bone). A place seen through (`mkmmd.core.moves.SIGHTED`: the ring) is moved back onto that line after
the wrist is turned toward its forearm and the approach has kept it off the face, and its keys are marked `sight`: the pose
stage hands the hand's goal to a second goal riding the head bone from the first such key to the last (`_sight_ride`, handed
over in 0.1 s either side), so the move's lean and head tilt and perform's nods and head rocks carry the ring with the eye.
The moved hands ride the chest otherwise. Each place is then made natural:

- **The wrist.** The elbow the arm IK will give (the default pole, `mkmmd.core.armreach`) fixes the forearm, and a hand that
  would bend more than 56 degrees from it is turned toward it as a whole (`HELD`: the `wrist_bend` rule of 60, less the few
  degrees the IK poses the arm off the plan; a ring at the eye 48, as the head that carries it tilts and nods). A place
  with something in the fist (a mic, a hammer) aims it instead: its hand continues the forearm as far as a fist round a
  handle pointing that way allows (the mic at the mouth: the forearm rises to a fist under the lips).
- **The twist.** Where that elbow would leave the forearm or the upper arm turned past what an arm turns (less a 10 degree
  margin; `mkmmd.core.armtwist`: the forearm -90 to +100 degrees from MMD's palms-down rest, the upper arm 95 either way,
  more outward as it rises above the shoulder), the elbow is turned round the shoulder-wrist line the least way to where
  they are not, rather than the forearm twisting (`mkmmd.core.moves.arm_for`): up and out for a hand turned palm out, down
  and in under the hand for one turned palm up (`blow`), preferring an elbow the hand needs no turning toward (a gesture
  still reads), no higher than the shoulder unless the hand is, on its own side of the body and out of the torso. The key
  carries that elbow's pole. A trip between two places whose elbows sit far apart round the shoulder (from behind the back
  up to the eye), or whose twists are 120 degrees or more apart, passes keys halfway with the pole round the shoulder, the
  elbow turned toward the middle of the two ends' twists and the hand rolled with its forearm, so the forearm turns over
  the short way, through the middle. The wrist of a key halfway goes round the shoulder at the mean of the two ends'
  distances from it, as an arm swings: on the straight line from the hip up behind the head it passed 8 cm from the
  shoulder, an arm folded so far that its upper arm turned past its range whichever way the elbow pointed. So a trip whose
  straight line passes the shoulder closer than 0.6 of its nearer end's distance from it (from a hand on the hip to fists
  over the head) gets the key round the shoulder even when the hand is fine there.
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
walk's swing, a singing motion's gestures, below). The moves stage writes the hand's `ik` keys ([Posing](#posing)), and a
member with a hand let go has its own keys composed with its strips (its action blends with Combine, so the pose stage's
forearm key adds to the strip's elbow instead of replacing it). Once the motion stage has laid the strips, it reads where they
have each let-go arm over its spans (the wrist, the hand's direction and palm, the elbow; in the frame of the chest the goal
rides, so a strip that turns the chest under the arm carries the goal with it) and keys the hand again
(`mkmmd.blender.build.pose.free_hands`): the goal follows the strip's hand over the span and the elbow's pole its elbow (turned
at the span's ends by what the IK's elbow misses by; for an arm too straight to show its bend, the way its upper arm turns), so
the IK lets go and takes over on the strip's own arm and nothing moves at the hand-offs; the hand travels to where the strip
has it at the span's start and away from it at its end at the human pace, as between any two places, and leaves the strip as
late as still gets it to its next move on time (at the span's end or before: the report's `free_hands` has the spans as
played). It joins and leaves the strip only where the strip keeps the arm's twists in range (5 degrees inside): a dance that
shrugs past what an upper arm turns where the span starts is joined after the shrug, and the IK lets go from there. Where the
strip itself twists the arm past the range while the hand is let go to it, the build warns (`free_hands` `twisted`: let the
hand go over another stretch). Over each span the forearm's pre-bend is off (the IK starts from
the strip's own bent arm), and the hand's finger keys give way by its IK share to a strip that plays the fingers (the strip's
curl alone, not composed with the hand's). Let a strip of the arm cover the span, eased in before it and out after it, and play
the fingers in a strip of their own over the span itself, eased over 0.25 s: the build warns when no strip plays the arm fully
over a span (the IK would let go of a half-played arm). Without a strip of that arm the goal waits at the hand's rest place
over the span, so the hand comes back from there. A hand that rests at the mic is never let go.

Faces of the same morph closer than 0.12 s are held through. The pose report has `moves` per member (`moves`, `hand_keys` per
side, `free` spans per side, `expressions`, `twitches`, and `places`: per side and place, `moved_mm`, how far the approach
kept the hand out, `wrist_deg`, the wrist's bend, and `twist_deg`, the forearm's and the upper arm's twists).

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
| `retime` | `"beats"`: the motion's own beat (`bpm`, else `core.vmd.tempo_phase`: a comb filter over 70-180 bpm of the bone motion) is scaled to the song's beat period over `per_beat` and the first motion beat at or after `from` lands on the song beat (or the 8th, ... with `per_beat`) nearest to where it would fall with `from` at `start` (`core.vmd.beat_fit`). Needs `timeline`; replaces `scale` |
| `bpm` | with `retime`: the motion's own tempo, the bpm of the song it was made for. The comb filter can settle on half or twice the beat, or on a pattern of the moves that is not the beat (a 204 bpm routine is out of its range); the report's `motion_bpm` says what it found. The phase is still searched, at this tempo |
| `per_beat` | with `retime`: motion beats to each song beat (default 1): 2 plays the motion's beats on the song's 8ths (a 130 bpm dance on a 73 bpm song runs at 146 a minute, not at 73), 0.5 one on every other beat. The first motion beat then lands on the song's beats cut into `round(per_beat)` parts |
| `timeline` | with `retime`: the timeline JSON whose `beats` are used (default `audio/timeline.json`; a missing file stops the build naming `motion.<cast>[i]`) |
| `bones` | which bones the motion drives: `"all"` (default), `"upper"` (spine, neck, head, eyes, shoulders, arms with their twist bones, wrists, fingers), `"lower"` (center, groove, waist, lower body, legs, knees, ankles, toes and the leg and toe IK bones), `"arms"` (shoulders, arms, elbows, wrists and their twist bones, not the fingers) or a list of these groups and semantic or Blender names. Hair, skirt and the root bone belong to no group |
| `legs` | `"ik"` (default): the member's leg IK holds the feet where the leg IK bones are, so a motion with FK legs (a BVH conversion's) dances with its feet planted and its hips turning over them. `"fk"`: the leg IK bones are keyed where the motion's own legs put the ankles (`motion.follow_legs`), so it steps, kicks, turns and jumps as it was made: a motion keying both legs moves the whole body down by how far its lower foot floats above the floor (a body with other legs than the motion's), and a resting foot that slides is held where it landed (`core.footplant`), eased back onto its path as it lifts. A motion that keys no leg rotation is an error. The report's `legs` has the sides, `lowered_m` and the `rests` held |
| `morphs` | also play the VMD's facial keys (default true) |
| `blend_in`, `blend_out` | seconds the strip's influence ramps in and out (default 0) |
| `influence` | the share of the motion the strip plays, above 0 and at most 1 (default 1), eased in and out over the blends (the strip's influence is keyed): on a member whose action blends with Combine (a hand let go), a strip of `["upper_body", "upper_body2"]` at 0.35 turns and leans the chest a third as far as the motion does, under `[perform]`'s sway, so the chest and shoulders go with the gestures of the arms' strips |
| `mirror` | `true`: the motion mirrored left for right, as MMD mirrors one (`core.vmd.mirror`: the bones of the two sides trade keys, positions flip and rotations reflect through the body's middle plane): the other hand plays the gestures. The facial keys are played as they are (default false) |
| `extend` | `"hold"` (default): the strip's first and last poses stay before and after it; `"forward"`: only its last pose stays, after it; `"none"`: it acts on its own frames only. A later entry's track is above an earlier one's, so where both play the later one wins |

A walk VMD gives a walking character natural arms where mk's own walk (`[perform.<cast>] walk`, its `arms` an IK swing) reads
stiff: let both hands go with `free` moves over each run of steps, and play the walk's `bones = ["arms", "upper_body",
"upper_body2"]` there with `extend = "none"`, its cycle scaled to two of the character's steps and its left heel strike on
her own left steps (a VMD walk in place keeps the legs, the feet and the travel mk's).

Singing arms (a singing motion's gestures): let each hand go with `free` over its stretches and play the motion's arm in a
strip that covers each stretch, eased in before it and out after it, and its fingers in a strip of their own over the
stretch itself, eased over 0.25 s ([Moves](#moves)); a motion whose other hand holds a mic all through gives that hand
gestures `mirror`ed from other moments of it (or its relaxed hanging arm from a quiet stretch), and a strip of
`["upper_body", "upper_body2"]` at an `influence` of about a third, on the gesturing arm's clock, turns and leans the chest
with the gestures under the sway.

A dance that stops with the music and picks up again: play stretches of one choreography as entries whose source ranges follow
on (each `from` the `to` of the entry before), each with `extend = "forward"`. Between two of them she holds the pose the dance
stopped in, and the next starts from that very pose, so nothing jumps or slides; let a hand go with `free` over the stretches
(and over the stops, for a hand that stays in the frozen pose) and give the other a pose move on a stop.

A dance made for a faster song on a slow one: `retime = "beats"` with the song's own tempo (`bpm`: the comb filter reads
half or twice the beat or none) and `per_beat = 2` plays the dance's beats on the song's 8ths (a 130 bpm dance on a 73 bpm
song at 146 a minute, 112 % of its speed, rather than at 56 % on the beats), and `legs = "fk"` for a BVH conversion's
steps: `vmd = "my_dance"`, `retime = "beats"`, `bpm = 130`, `per_beat = 2`, `legs = "fk"`.

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
others leave alone. A motion that switches the leg IK off (FK legs, as a BVH conversion's are) plays under the member's IK all
the same: mmd_tools would switch it off for the whole scene at the import, so the stage keeps it on (and drops the switches the
VMD keys); its leg keys give way to the IK, the feet stay where the leg IK bones are, and the rest of the body plays as keyed
(`legs = "fk"` keys the IK bones where its legs put the feet instead).
A bone track keyed on the far side of the rest (quaternions with w below 0 throughout, as BVH conversions write them) is turned
to the near side at the import (`near_rest`: the same rotations), so blends with the rest pose and other strips turn the short
way instead of half a turn round.
The report has, per entry, `vmd`, `source`, `strip` (first and last frame), `morphs`, `mirror` and
`influence` when given and, with `retime`, `motion_bpm`, `song_bpm`, `scale` and `per_beat` (when not 1), and `free_hands`
the hands let go to the strips (per member and side: the `spans` as played, the `samples` read and the `pole` stretches
keyed); `retime` raises when no beat can be found in the motion (a very short or motionless VMD) or the timeline has fewer
than two beats, and `bpm` or `per_beat` without `retime` is an error.

### Perform

`[perform.<cast>]` is a character's life on top of its pose: gaze, breathing, sway, nods, beat bob, startles, keyed lean / turn /
tilt, turns and spins of the whole body, blinks, expressions, lip sync, twitches and strumming. Times are clip seconds, angles
degrees, targets are [Targets](#targets), with one exception: `"camera"` does not resolve here, the cameras are built by the
shots stage after this one. The stage keys `upper_body`, `upper_body2`, `neck`, `head` and both eyes on every frame of the
build (LINEAR), composed in the armature's frame, so characters riding vehicles stay correct. Those keys contain the pose's
base (eased in over the settle) and replace the pose stage's own on the same bones. A model without eye bones gets head motion
only; one without `upper_body2` takes the breathing on `upper_body`. The stage needs from earlier stages the pose's base and the
pick hand's goal for `strum`, the props and members the targets name, `rig.json` for morph names (`blink`, the vowels
`a i u e o`, the semantic names of `expressions`) and a timeline file for `bob`, `sing` and `strum`. `mk ref measure`
recommends values for these keys from clips of real people (`blink.per_min`, `lids`, the rate, `rise` and `hold` of gaze
events, `head_share`, `eye_max`, `nod`, `bob`, `sway`, `breath`, `sing.mouth`), each with a reason.

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
wanders by a fraction of a degree (a slow noise seeded by the member's name). Over a `spin` the spin has the head and the eyes,
all of the turn (they spot where she faced, [Body](#body)).

#### Body

| Key | Meaning |
|---|---|
| `body` | `false`: the face alone (`blink`, `lids`, `expressions`, `sing`, `mouth`, `twitch` and a `squash`): the spine, neck, head and eyes are not keyed, so a member's motion plays them as it was made (a dancer's face alive under her dance: perform keys those bones on every frame otherwise, over the motion's). Any other key beside it is an error; the report says `body` |
| `breath` | `{per_min = 16.5, deg = 0.6}`: chest pitch on `upper_body2` |
| `sway` | `{deg = 0.37, period = 2.5}`: a slow turn of the upper body about the vertical |
| `nod` | `{deg = 0.48, period = 2.3}`: a slow nod of the head, fading while a gaze event holds the head away |
| `bob` | `{deg = 1.5, beats = [t, ...], timeline = "audio/timeline.json", downbeat_accent = 1.6, from, to}`, or a list of such tables (each over its own `from`..`to`, summed: a small bob in a verse, a deeper one in a chorus): the head dips after each beat (60 ms attack, 220 ms decay). Needs `beats` (clip seconds) or a timeline file (`timeline`, default `audio/timeline.json`: its beats; its downbeats dip `downbeat_accent` times deeper; a missing file stops the build naming `perform.<cast>.bob`); `beats` wins |
| `startle` | `[t, ...]`: at each time the upper body jolts backward (4.5 degrees, settling with a 0.35 s time constant) |
| `lean`, `turn`, `tilt` | `[[t, deg], ...]`: extra upper-body forward lean, turn toward the left and sideways tilt toward the left over time, smoothstep-eased between the keys, held before the first and after the last, on top of the pose's base: a reach that leans in and settles back, a head on a shoulder. Hand targets still hold, except hands whose pose `ride` is a chest bone, which go with it. The first key's value holds from the start of the pre-roll: begin with a `[0, 0]` key to start from the base |
| `head_tilt` | `[[t, deg], ...]`: the head rolls toward the left on top of the gaze, keyed the same way |
| `head_turn` | `[[t, deg], ...]`: the head turns toward the left on top of the gaze, keyed the same way: a look back over the shoulder |
| `head_nod` | `[[t, deg], ...]`: the head nods forward and down on top of the gaze, keyed the same way: a headbang (the gaze still pulls the head toward its target by `head_share` of the body's lean, so a nod over a deep lean takes more) |
| `groove` | `{every = 2, shift = 0.03, dip = 0.018, light = 0.6, soft = 8, roll = 4, yaw = 3, still = 0.8, heel = 7, lean = 1, nod = 1.5, vary = 0.25, first = "L", seed = 1, from, to, ease = 0.4, beats = [t, ...], timeline, downbeat_accent = 1.3}`, or a list of such tables (each over its `from`..`to`, summed): she stands in time with the music as a person does. ONE motion drives her: her weight goes from one foot to the other every `every` beats (2: on 1 and 3; the accented beats where the beat list has any), `first` the foot it goes to first. It is a sprung mass: it sets off before the beat, arrives on it, runs a little past the leg and settles, over a time that goes with the sway's own (a slow sway changes leg slowly), each change a little different in reach and timing (`vary`, drawn from `seed`). On that one curve hang, each a little later than the part below it: the hips going `shift` m over the standing leg, the pelvis tilting (`roll` degrees: the standing hip up, the free one dropped) and turning (`yaw`: the free hip forward), the chest leaning back the other way by the share `still` of what keeps her head where it was (a singer at a mic: her hips move under a head that hardly does; 0.5 lets her head sway), the shoulders turning a little against the pelvis, the head righting itself, and the free foot's heel coming off the floor (`heel` degrees; the foot bends at the ball where the toes have their own bone and skin, so they stay flat on the floor, else it rises on its tip). On every beat her knees let her down `dip` m onto the beat and bring her back up before the next (a wave under where she stands, without a corner: not a pulse), in full on the beats her weight lands on and the share `light` of it on the others, the hip going a little further out with it, the chest curling `lean` degrees and the head nodding `nod` degrees after it. Her knees are never locked while she grooves: the straighter one keeps `soft` degrees of bend from the pose the model was made in (which is the longest an IK leg gets), and her hips come down for that; a pelvis tilted over the standing leg hangs from that hip, it does not push the foot off the floor. Under it all her posture wanders slowly. The report gives `shift_mm`, `dip_mm`, `roll_deg`, `heel_deg`, `sink_mm` (how far her hips come down for her knees) and `knee_deg` (the least and most her knees bend). Use it for a body that stands and plays or sings: a `rock` with a `bounce` over it is a pendulum and a pulse, and she bobs like a buoy |
| `rock`, `head_rock` | `{deg, period = 4.0, phase = 0.0, axis = "side", hips = 0, from, to, ease = 0.4}`: a periodic sway of the upper body and of the head: `deg × sin(2π t / period + phase)` of clip time (`phase` in radians), added to the keyed ones; `axis` `"side"` leans the body (like `tilt`) and rolls the head (like `head_tilt`) toward the left, `"front"` leans the body forward (like `lean`) and nods the head, `"turn"` twists the body (like `turn`) and shakes the head. Unlike the keys it goes on at any clip time, so a dreamy sway survives a re-cut; a squirm seen side-on is a fast `turn` (a swing a beat). `rock.hips` (metres) shifts the hips toward the left with the same swing, the feet planted, so the weight goes from foot to foot: a groove without steps. `from`, `to` (clip seconds) keep a sway to a span, eased in over `ease` seconds before `from` and out over them after `to` (as `bounce`'s window); a list of tables is several sways, summed: a sway in each span where she only stands (still for the poses and dances between them), or the hips in one table and the body leaning against them in another |
| `bounce` | `{depth = 0.03, beats = [t, ...], timeline, downbeat_accent = 1.6, from, to, attack = 0.05, decay = 0.16}`, or a list of such tables (summed): the hips (the `center` bone) dip `depth` m on each beat (downbeats weighted), the feet planted, so the knees pump; `from`, `to` (clip seconds) limit it |
| `rise` | `[[t, metres], ...]`: the whole body, feet too, lifts by that much (eased between keys): she floats |
| `crouch` | `[[t, metres], ...]`: the hips drop by that much, the feet planted, so the knees bend (eased between keys, on top of `bounce`): a landing, a squat, a brace before a jump |
| `kick` | `{foot = "L", height = 0.12, back = 0.08, hold = 0, beats, timeline, downbeat_accent, from, to, attack = 0.06, decay = 0.18}`: that foot flicks up and back on the beats; `hold` (0..1) keeps it that share of the way up between them (a foot kicked up behind a lovestruck girl) |
| `air` | `{from, to, tuck = 0.24, back = 0.10, first = "L", share = 0.7, lag = 0.06, rise = 0.14, extend = 0.22, point = 30, kicks}` (or a list): the legs of a body off the ground (a jump, a blast, a float; its height is the root's, keyed by `[[key]]`, or `rise`) from take-off `from` to landing `to`: over `rise` s both feet come up toward the hips by `tuck` m and back by `back` m (the knees bend forward), the foot `first` ahead and the other `lag` s later by `share` of it, the toes pointing `point` degrees down; they hold still (a [freeze](#freezes) in the air shows no jump when the world starts again; `kicks` give that up) and over the last `extend` s reach down again, so the feet are under the hips at the landing. `kicks` (`true` or `{knee = 10, hip = 2, foot = 15, period = 3.0, every, beats, timeline, wander = 0.3, seed = 0, ease = 1.0}`) makes the hold a float (weightless, treading water): from when both legs are tucked until they reach down, eased in and out over `ease` s, the legs take turns in small lazy kicks, each knee straightening and bending `knee` degrees either side of its tuck, the thigh swinging `hip` degrees a little ahead of it and the foot pointing as the knee straightens and flexing as it bends (`foot` degrees against the shin, a little after it). A leg kicks every `period` s; with `every` its knee is straightest on every n-th beat of the timeline (`timeline`, default `audio/timeline.json`) counted from a downbeat (`every = 2`: one leg on beat 1, the other on beat 3), with `beats` on those clip seconds. `wander` (at most 0.6) varies each kick's size and, off the beats, its pace by up to that share, smoothly (`seed`): no metronome. The bends are the knees' and hips' own, turned into the feet's places through the rig's legs (`mkmmd.core.perform.air_kicks`), so they look alike on any character with leg IK; a knee never straightens past 5 degrees (from a straight tuck the kicks bend it further). Side-on a tucked shin slopes back (about 40 degrees for a 0.12 m tuck) and a foot looks pointed only when `point` passes that slope: a float wants about 55. The report's `air.kicks` has, per span, each leg's landings (its knee straightest), its knee's range and how far its foot moves (`foot_mm`). Added to `kick`. Mind the knees: a model's legs straight while its root is lifted look hung, not jumped |
| `step_turn` | `[{t, deg, dur, first, lift = 0.045, dip = 0.02, move}]`: the body turns round on the spot like a person, not a figure on a turntable. From `t` the cast root turns `deg` degrees toward the left (negative: right) over `dur` s (default 0.2 s and 0.11 s per 100 degrees: 0.36 s for 145), and travels `move` `[x, y]` m meanwhile (in the character's frame before its first turn: x its left, y behind; a step toward what it turns to), the chest leading the hips (by at most 30 degrees) and unwinding as they arrive; the foot `first` steps round, then the other (by default the order that keeps the feet further apart: about a point in front of the ankles or behind them, the other order sets a foot down where the other still stands), each lifted `lift` m between rolling off and setting down, one foot always down and none sliding; a foot is never more than 35 degrees from the hips (it swivels on its ankle), and the hips sink `dip` m meanwhile. The report has `feet_gap_mm`, the least distance between the ankles. Give the head a `gaze` at what the body turns to, a moment before, and it leads the chest. The root's yaw, and the axes `move` travels along, are keyed here, so a `[[key]]` on them is an error |
| `spin` | `[{from, to, deg, spot = true, windup = 20, overshoot = 8, twist = 35, neck = 70, whip = 0.3, foot, lift = 0.1}]` (or one table): the body spins round like a dancer, not a figure on a turntable (`mkmmd.core.perform.spin_curves`). From `from` she winds up `windup` degrees the other way, turns `deg` toward her left (negative: right) and `overshoot` degrees past it, and has settled back by `to`, each eased; the chest leads and the hips trail it by 0.12 s (at most 8 % of the spin), so the spine twists with the speed, by `twist` degrees at most (50 at most: `joint_limits` keeps a spine to 60). With `spot` the head holds where she faced before the spin (a dancer's spot: the camera in front of her) while the neck lets it, `neck` degrees from the chest at most (80 at most: `joint_limits` keeps a neck to 85, and `head_limits` clips it at its `yaw`), then whips round in `whip` s ahead of the chest, as far ahead as the neck lets it, onto the next spot (every full turn, then where she ends facing) and holds it as the chest and hips come round under it; without `spot` it leads the chest by the speed. The eyes lead the head by 0.07 s. A spin has the head: the idle look and gaze events give way to it from 0.25 s before it to 0.25 s after. All inside an `air` span or a `rise` of more than 2 cm, the legs go round with the hips (their tuck and kicks go on); on the ground she turns on the ball of `foot` (default the side she turns toward; on the tip of its toes where the model's toes have no bone and skin of their own): the weight goes over that foot as she winds up, both feet flat, then as the hips set off its heel rises 20 degrees about the toes' joint, the toes staying flat on the floor (a demi-pointe; the body rises with it and comes over the ball of the foot), and the other foot lifts `lift` m, drawn halfway in toward it, and both come down in her stance over the last 0.2 s, after she has settled; the root goes round that point (its location is keyed). A spin partly in the air is an error. The root's yaw is keyed here, so the hair and skirt simulate from the real turn, and a `[[key]]` on it is an error; step turns and spins follow one another in time order, each starting after the one before has ended, and a walk or a tumble with them is refused. The report's `spin` has, per spin, `whips` (when the head whipped round), `twist_deg`, `neck_deg`, `air` and `foot` |
| `walk` | `{steps = [t, ...], hops = [{from, to, move, height}], stride = 0.55, first = "L", lift = 0.07, dip = 0.015, slide = 0, roll = 0.15, hop = 0, arms = 0, cross = 0, sway = 0}`: the body walks (`mkmmd.core.perform.walk_plan`). Each step is a foot coming down (alternating from `first`) a `stride` past the other foot, on its own side of the body, or `cross` of the way in to the body's centre line (1: on it, a catwalk; up to 1.5: across it); it rolls off `roll` of the step's time after the other came down, swings through `lift` m and lands on the step, never sliding while down. A step more than 1.2 s after the one before (or after a hop's landing) starts from a stand, as the first does: the body waits where it stood and the foot swings over the last 0.35 s, so she can stop, hold a pose and walk on in one walk. The cast root travels with the body (straight ahead, at a steady pace, midway between the feet each time one lands) and the hips dip `dip` m at each landing; `sway` (m) carries the hips over the foot that is down, out and back while the other swings (a strut; the feet stay planted). Over a hop both feet leave the ground from `from` to `to`, the root flies a parabola `height` m high (default what the flight time gives, g T² / 8; `hop` sets one for every hop), carried `move` `[x, y]` m (its left, behind; default a stride ahead: a knight's L is `[1.2, -0.6]`), and the legs tuck as `air` does; the feet land a quarter stride either side of the body and the foot behind steps next. `slide` 1 puts the walk on a treadmill (the body stays, the feet down slide back under it: running on the spot), `arms` (m) swings each hand's IK goal forward as the other side's foot goes forward, while the feet move: it dies away over 0.25 s once they stop, so a hand posed between two walks is where its pose put it (the hands need a pose: a `rest` move gives one, and a hand riding the chest swings with it). The root's location is keyed: a `[[key]]` on it must be `relative` (a push on top of the walk). A walk and a `step_turn` or a `spin` in one table are refused: face the walk with `[[cast]] yaw`. The report has `walk` and `sway_mm` |
| `tumble` | `[{from, to, deg, axis = "flip"}]`: the whole body turns in the air about its hips (the root turned and moved so the hips stay put), `deg` over `from`..`to` eased in and out, about its left axis (`"flip"`, positive: head forward) or its forward axis (`"cartwheel"`, positive: toward its left); turns add up and hold (360 lands upright). Give it `air` (or a walk's hop) so the legs tuck |
| `drum` | `{hand = "R", fingers = ["little", "ring", "middle", "index"], deg = 25, lift = 0.16, roll = 0.02, beats, timeline, downbeat_accent, from, to}`: the fingers of one hand tap on the beats (a driver drumming on the wheel): each lifts at its base joint by `deg` (the middle joint by a third of it) about its own flexion axis, from wherever its keys hold it (a grip's solved curl), over the first half of the `lift` seconds before the beat and falls, fastest at the end, back onto it on the beat; the little finger leads the index by `roll` s per finger. The palm stays where the grip put it, so a hand wrapped round a rim never moves through it. The report has `drum` (`hand`, `fingers`, `taps`, `deg`, `frames`) |
| `squash` | `{lane, min_db, every, from, to, depth = 0.1, stretch = 0.05, attack = 0.04, rebound = 0.14}` or `hits = [t, ...]` in place of the lane: a cartoon squash and stretch of the whole body on each hit of a `lane` of the [audio] hits file (or on `hits`, clip seconds), keeping its volume and anchored at the feet: the cast root's scale keyed on every frame about its origin (where she stands), the height squashing over `attack` s to 1 - `depth` on the hit, then springing back up past 1 to 1 + `stretch` `rebound` s after it and settling (a damped spring, `mkmmd.core.perform.squash`; overlapping hits add up), the width and depth by 1 / its square root. The report has `squash` (`hits`, `squash`, `stretch`) |

#### Face and mouth

| Key | Meaning |
|---|---|
| `blink` | `{per_min = 15, seed = 0, extra = [[t, dur], ...]}`: natural blinks at a mean interval of 60 / `per_min` seconds (±45 %, at least 1.2 s apart, drawn from `seed`), each 0.157 s (no key sets the duration of the natural ones); `extra` adds blinks of the given duration at chosen times |
| `lids` | 0..1, the lids' base lowering: a floor under the `blink` morph, eased in over the settle and open while a gaze event holds the head away and during a glance (default 0) |
| `expressions` | `[{morph, keys = [[t, value], ...]}]`: a morph's value over time, Bezier between the keys. `morph` is a semantic name (rig.json `morphs`) or the model's own, of any kind (a group or material morph through the bound sliders, see [Cast](#cast)); the stage raises when the model has no such morph. An expression on the `blink` morph replaces the blinks, and lip sync replaces an expression on a vowel morph |
| `sing` | lip sync, below |
| `twitch` | `[{bones, family, t, deg = 14, axis = [1, 0, 0], dur = 0.22}]`: a quick flick (out to `deg`, back past rest by a quarter of it, home) about an axis in the armature's frame. `bones` lists semantic or Blender names, `family` (`"ears"`, `"tail"`) takes the top bone of every chain of that family by its name; `t` is required |
| `mouth` | `[[t0, t1], ...]`: clip seconds where the expressions on the vowels win over `sing` (eased in and out over 0.08 s): the moves write them for a yawn, a chomp and a brush, so their mouths are not lip-synced away |

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

A twitch is keyed on the bone itself on top of its other keys. On a bone the sim stage simulates (a model whose ear's top
bone is its chain's first: there is no `ear_root` above it to twitch) the flick rides the simulation: the sim keys the
chain's bones and turns a twitched one by its twitch on top, the rest of the chain going with it.

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
| `rhythm` | `"onsets:<stem>"` (the timeline's `onsets`, see [Timeline](#timeline): `mk timeline onsets`), `"beats:N"` (N strokes per bar of four beats, a multiple of 4: 8 is eighths), `"notes"` (pick the other hand's bass line one string at a time: [Playing a worn guitar](#playing-a-worn-guitar), Picking) or a list of strike times in clip seconds (default `"beats:8"`) |
| `technique` | `"pick"` or `"fingers"` (rest strokes with the index and middle fingers; needs `rhythm = "notes"`): default the card strum entry's `technique` (the library bass's is `"fingers"`, as most bassists play, the guitar's `"pick"`), else `"pick"`. The first table's decides for every table of the hand ([Bass lines and picking](#bass-lines-and-picking), Fingerstyle) |
| `hold`, `recover`, `shift`, `pause` | fingers: seconds a finger rests on the next string after its stroke (0.25), lifts back to hovering (0.12), the hand moves between stations (0.12); a gap longer than `pause` (0.8) starts the alternation again with the index. `attack` is the whole swing hover -> rest (0.11), `min_gap` 0.05 |
| `from`, `to` | clip seconds: only the strikes inside are played (default the whole clip) |
| `accent` | `"downbeats"`, `"beats"` or a list of times: strikes within `accent_window` of them are accented (wider and deeper by `accent_gain`) |
| `hits` | `"downbeats"`, `"beats"` or a list of times: only the strikes within `accent_window` of them are played. The rhythm's other strokes are ghost strokes: the hand keeps its swing going on every eighth and passes `ghost_lift` m (0.007) over the strings on `ghost_scale` (0.8) of a stroke's travel; they ring no string and the `strum` check does not judge them. Default: every stroke is played |
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
| `share` | the share of the sideways travel that the hand makes by turning about the wrist, about an axis along the strings; the rest is made by moving the hand (default 0.6). Faded out where the turn would move the tip into the face more than across it (a hand hanging from a wrist above the strings: the arm carries the stroke). Picking a line (`rhythm = "notes"`, default 0.85) turns about the axis square to the sweep and to the wrist-to-tip lever (`core.strum.wrist_turn`: the wrist deviating, the forearm rolling), so the stroke is the hand turning and the arm hardly moves; the tip still follows the path exactly |
| `swing`, `loose` | a strum is swung from the elbow, and the hand turns with the forearm in the plane of the strum instead of sliding along the strings square to them: `swing` is the share of that turn it takes (default 1). The wrist makes part of every stroke itself, the more the faster the strokes come (a fifth of eighths at an easy tempo, half of fast sixteenths), and a `loose` wrist (default 1; 0: stiff) lets the hand fall behind the forearm on the way and flick through at the turn, by 14 degrees at most (`core.strum.hand_turn`). Each part acts only where its lever lies across the stroke: a forearm that comes down onto the strings along the stroke does not swing it, and a hand hanging from a wrist above the strings does not flick it (the whole arm carries the stroke then, and the hand keeps square: the report's `swing_deg` and `wrist_deg` are 0). The tip follows the path exactly either way: it is the wrist and the arm that go elsewhere. Picking a line turns nothing but `share` (both 0) |
| `human`, `seed` | a player's own unevenness: how far each stroke goes varies from stroke to stroke (5 %) and drifts over a few strokes (6 %), drawn from `seed` (default 1), so no two bars are alike; `human = 0` is a machine's strokes (default 1; 0 when picking a line) |
| `kick` | degrees: on every accented stroke the guitar's neck kicks up that far (the prop turns about its face normal through its origin and eases back; the hands ride it). In the first table of a hand |
| `windmill` | `[t, ...]`: before the down stroke nearest each time (within 0.12 s) the pick hand swings a full circle, over `windmill_dur` seconds (0.5): its goal turns once about an axis through the shoulder along the guitar's face normal, reaching out over the top (the arm straightens most of the way) and coming down onto the strings at the strike, which then plays as planned. The elbow's pole goes round with the goal, a little behind it over the top (a flung arm's elbow follows its hand), so the whole arm turns from the shoulder and the hand keeps its turn on the forearm (with the pole left standing the goal asked the wrist for a roll no wrist has, and its limit flipped the hand for a few frames at the top). The strokes struck while it goes round are not played (ghost strokes: no string rings, the `strum` check leaves them out). Know what it is before using it: the hand stays in the plane of the strings all the way round, so the circle lies in front of her, not beside her as a real windmill's does (the arm back past the hip, up behind and over): on a tall player with a low guitar it passes over her shoulder, on a small one or a guitar worn high it crosses her chest and her face. The report's `windmills` gives each one's `head_cm` (how near the line from her shoulder to her hand comes to the middle of her head) and the build warns when that is inside two head radii. Half a second is nine frames for most of the circle: without motion blur (a draft) it strobes. Nothing else of her takes part (no wind-up, no shoulders), and a cut on the strike hides the landing. In the first table of a hand; read it frame by frame in a front or three-quarter view (side-on the circle is edge-on) and keep `joint_limits` at 0 |
| `ring` | the strings the strokes play vibrate on screen (default `true`; `false`: still), or `{amp, decay, damp, shimmer}`: see [Strings that ring](#strings-that-ring). In the first table of a hand |
| `sigma` | half the width of the six strings at the strum centre, metres (default taken from the card's strings, 0.0225 without a `neck` point) |
| `timeline` | the timeline JSON with `onsets`, `beats` and `downbeats` (default `audio/timeline.json`) |

A strike on a beat goes down, one on the offbeat up, an off-grid one alternates (`beats:N` takes its directions from its grid:
beats down, the subdivisions between them alternate). Alternating strokes share their turn-round, so eighths are one continuous
swing; a stroke in the same direction as the one before first returns over the strings, lifted. When the time between one stroke's
follow-through and the next run-up is longer than `approach + retreat + 0.1` s the hand goes back to rest (the tip on the string
plane at the strum centre: the grip) and comes in again. The path is turned into IK goals relative to the grip and keyed every
frame (LINEAR) from the first approach to the last retreat: the hand moves as a rigid body and turns about the wrist for `share`
of the sideways travel (`core.strum.turn_share`: not at all when turning would hardly move the tip sideways or would move it
more into the face than across, never more than 0.6 rad). Tables that
share a hand take the first one's `prop`, `grip`, `sigma`, `timeline` and path numbers (`span`, `lift`, `return_lift`, `attack`,
`lead`, `follow`, `approach`, `retreat`, `share`); `rhythm`, `from`, `to`, `accent`, `accent_gain`, `up_scale`, `accent_window`,
`min_gap`, `beat_tol` and `depth` are each table's own.

The report has `technique`, `strokes`, `down`, `up`, `frames` (first and last frame keyed), `sigma_mm`, `turn_share`, `first`
(the first strike times) and `max_u_mm` (the widest sweep); played with the fingers `fingers` (strokes per finger) and
`stations` (the strings) instead of the pick's numbers. With no strike in the window the stage logs a WARNING and keys nothing;
it raises when the hand has no strum grip or its goal does not ride the prop. The `strum` check replays the plan on the
timeline against the scene.

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

### Ribbons

`[[ribbon]]` draws a 3D paint stroke in the world, so the camera moves through it: a crayon line round her, a striped
toothpaste rope squeezed out of a tube, a neon marker loop the lens flies through. The stage runs after perform (it reads the
cast as posed and moving) and before shots (`mkmmd/core/ribbon.py` checks the entries and makes the centre lines, numpy,
tested; `mkmmd/blender/build/ribbons.py` and `ribbon_nodes.py` build them). Each ribbon is an empty `<name>` (its frame:
the world's or, with `ride`, what it rides, followed without scale) carrying the keyed properties `grow` and `tail` (it
shows from `tail` to `grow` of its length) and `thick` (its radius scale), and `<name>_ribbon`: a mesh of its centre line
(a point a centimetre or so apart) under geometry nodes that trim it to that window, give it rounded ends (a dome one
radius long at each), its profile and its materials on every frame, with no Python at render time, so every look, `mk
look --map` and the bake see the drawn part only. A neon also gets `<name>_glow`, a round shell of additive light about
it (left out of the looks, viewport alpha 0). A long ribbon that pulses changes on many frames and may be too big for the
bake to hold frame by frame: the bake then says so and `mk render --engine mk` draws those shots in Blender.

| Key | Meaning |
|---|---|
| `name` | the root's name and its objects' prefix (required, new) |
| `style` | `"crayon"` (default: wax, matte with a soft sheen, streaked along the stroke and pitted where the paper's tooth took no wax, slightly flat, its path and width wobbling as a hand draws), `"toothpaste"` (round, glossy, `stripes` running its length), `"neon"` (a flat marker ribbon of its own light, hotter across its middle, with a glow) |
| `orbit` | `{around, radius = 0.45, height = 0, rise = 0, turns = 1, start, tilt = 0}`: round a target (`"cast:<name>.<bone>"`, a bone's head; any [target](#targets)), in its member's frame (angle 0 in front of her, 90 at her left; negative `turns` go clockwise seen from above), `radius` a number or `[from, to]`, `height` above the target, climbing `rise` over the turns, the plane leaning `tilt` degrees (front up, or `[front_up, left_up]`); `start` defaults to 0, or toward the `from` nozzle |
| `spiral` | `{around, radius = 0.4, bottom = 0.1, top, turns = 2.5, start = 0}`: up her body round the vertical through her root (`"cast:<name>"`) or a bone's head, from `bottom` to `top` metres above her feet (default her eyes + 0.25) |
| `swoop` | `{points, cast, closed = false}`: a centripetal Catmull-Rom through targets; with `cast` plain `[x, y, z]` points are in her frame (x her left, y behind her, z up) |
| `fall` | `{reach = 0.25, floor = 0}`: with `from` alone (the default path then): out of the nozzle along its direction, bending down onto the floor `reach` metres on, then a pile |
| `from` | `"prop:<prop>.<use point>"` (a tube's `nozzle`): the ribbon starts 1.5 widths inside it and leads out along its direction (a use point without one: toward the path) into the path; its `radius` sets the default width; a prop with a `squeeze` param gets it keyed 0 -> 1 over the draw (back over a retract; `squeeze = false` or `[lo, hi]`) |
| `pile` | `true` or `{turns = 2.5, radius = 1.6 widths, shrink = 0.6, step = 0.8 widths}`: the end coils up on itself, each turn resting on the one below |
| `draw` | `[from, to]` clip seconds: the front runs from the start to the end (required), eased by `ease` (`linear`, `smooth`, `in`, `out`) |
| `retract`, `chase` | `[from, to]` after the draw, one of them: `retract` rewinds the front into the start (the paste back into the tube), `chase` runs the tail after the front (it leaves along its path) |
| `width`, `thickness` | metres (crayon 0.05, toothpaste 0.056 or twice the nozzle's `radius`, neon 0.03); the flat styles' thickness (0.45 and 0.15 of the width); the flat side faces `face`: `"out"` (default round an orbit or a spiral: away from its axis), `"up"` (default for a swoop or a fall), or a direction |
| `color`, `stripes`, `glow`, `halo`, `wobble`, `seed` | the crayon's and neon's colour (palette slot or `#hex`; `love`, `iris`); a toothpaste's 2 to 8 `stripes` round its profile (`["love", "gold", "foam", "iris", "rose"]`, the first centred on the side it faces); a neon's emission (6) and its glow `{size = 4 half widths, strength = 0.35}` (`false`: none); a crayon's wobble (0.3 widths) and its seed |
| `pulse` | `{lane or hits, gain = 0.35, attack = 0.04, decay = 0.3, min_db, every, from, to}`: on every hit of a lane of the [audio] hits file (or at the clip seconds `hits`) the radius swells to 1 + `gain`, rising over `attack` up to the hit and back over `decay` (`thick`, keyed linearly) |
| `ride` | `false` (default: it stays where it was drawn), `true` (what the path goes round, the swoop's `cast`, the `from` prop), `"cast:<name>[.<bone>]"` or `"prop:<name>"` |
| `clear` | `true` (default: 0.03 m), a margin in metres, `{margin, cast}` or `false`: kept that far clear of the cast's collision bodies (rig.json, every member by default) |
| `at`, `spacing`, `shadow` | the clip second its targets are read at (default the draw's start); the centre line's point spacing (default min(0.01, width / 4)); casting shadows (not a neon) |

Keeping clear: the stage samples the bodies as posed (before the sim) on the frames the ribbon shows and moves what of it
would come within `clear` of one, every second frame, the moves smoothed over 0.12 m along it so it bends round a body
(`push_clear`); a nozzle's lead-in stays put. The report gives per ribbon its `style`, `path`, `width`, `points`,
`length_m`, the draw frames, `pulses`, `squeeze`, `ride` and `clear` {`margin`, `pushed_points`, `moved_mm`,
`least_gap_mm` (every frame, pulses in), `at_frame`, `body`}; a WARNING when the least gap is under zero (bodies on both sides
closer than the ribbon is fat). The `ribbon_clear` check measures the saved scene the same way, hair and skirt as simulated.

```toml
[[ribbon]]                    # striped paste squeezed out of the tube, wrapping her; it swells on the snares
name = "paste"
style = "toothpaste"
from = "prop:tube.nozzle"
orbit = { around = "cast:reisen.lower_body", radius = 0.58, turns = 1.6, rise = 0.3, tilt = 5 }
draw = [0.3, 2.6]
retract = [6.0, 6.8]
pulse = { lane = "snare", gain = 0.2 }

[[ribbon]]                    # a neon hoop in front of her for the camera to fly through, leaving along its path
name = "hoop"
style = "neon"
color = "foam"
swoop = { cast = "reisen", points = [[-1.6, -2.2, 0.6], [0.0, -1.5, 0.65], [0.5, -1.45, 1.05], [0.25, -1.5, 1.55], [-0.5, -1.5, 1.05], [0.8, -2.0, 0.9]] }
draw = [0.0, 1.2]
chase = [3.2, 3.9]
```

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
| `breathe` | `{palette, keys, ease, slots, lights}`: the palette breathes into another one and back over clip time (below) |

The stage always writes `view`, `exposure` and `gamma` and writes `contrast` only when given, so the look a set builder chose
(the café and bedroom sets pick `AgX - Base Contrast`) stays otherwise. The settings live in the scene's view settings, so
`mk render` and `mk look` draw with them (the grade is `[post]`).

**Breathing palette.** `[look.breathe]` mixes the project's palette (mix 0) with another one of the same slot names (mix 1)
over clip seconds (`mkmmd/core/breathe.py`, numpy-free, tested): `palette` (required: the other one), `keys` (required,
`[[t, mix], ...]` or `[t, mix, ease]`, in time order; the mix is held before the first and after the last; two keys at one
time jump), `ease` from each key to the next (`"smooth"`, the default, a smoothstep; `"linear"`; `"hold"`), `slots`
(`{slot = "#rrggbb"}` over the other palette: `[look.slots]` stay the project's palette's own, except slot names no palette
has, which carry over) and `lights` (a strength factor at mix 1, `lerp(1, factor, mix)` times a light's own: a number for
every light, or `{pattern = factor}` matched on the light object's name, the first that matches; default 1: a night scene
may want its lamps stronger and its sun weaker). The `breathe` stage (after `keys`) makes the build follow it:

- A probe Blender builds the stages before it (`scene` .. `keys`) with the other palette, beside the build from its first
  stage (solver results and the form cache are written whole, so the two share the project's cache), and saves them
  (`<cache>/breathe/`, removed after; `MK_KEEP_JOBS=1` keeps it). Builders are deterministic, so whatever differs between
  the two builds is what the palette made: nothing in a builder has to say what is a palette colour. Two Blenders at once
  is twice the memory: `mk build` runs the probe beside the build only when the memory available holds two Blenders at
  the last build's peak (`<cache>/build_memory.json`) with 2 GB to spare; otherwise, and on a project's first build, it
  runs the probe first and then the build, which takes it as it is. The report's `memory` has the run's `peak_gb` and
  how the `probe` ran (`beside`, `first`).
- Every difference is driven by one scene property, `mk_breathe` (the mix, keyed on every frame): socket values of
  materials (props, sets, text, ribbons, emission), worlds and shader node groups and the values of RGB and Value nodes, a
  driver per channel (colours mixed as `c ** (1 / 2.2)`, close to the sRGB mix `"a:b:t"` makes, numbers linearly); a
  colour ramp, an image or a mesh attribute a material reads (baked lamp light) that differs gets the probe's beside it (a
  ramp node, a packed image, an attribute `<name>~breathe`) and a Gamma / Mix / Gamma crossfade by the mix; lights get
  their colour and their strength (a keyed strength keyed again on every frame, a driven one's expression multiplied by
  the factor); materials' viewport colours and the world's colour too. At mix 0 the scene is the plain build with the
  project's palette and at mix 1 the plain build with the other (the `lights` factors on top), value for value.
- The cast's own materials are left alone (a `[[cast]] materials` paint keeps the project's palette); photographs and
  footage keep their colours; the render-time looks (silhouette, vector, rings, a `[[medium]]`) keep the project's palette.
  What cannot breathe (a value animated already, a node or a link only one build has, a setting that is not a number, an
  object or material named after its colour that no object slot pairs) keeps the project's value and is listed in the
  report's `fixed`, logged as a WARNING. Keep builders deterministic and their node trees the same whatever the palette.

The report has `palette`, `keys`, `probe` (`stages`, `seconds`, `waited`: how long the build waited for it), counts of
`drivers`, `sockets`, `values`, `ramps`, `images`, `attributes`, `lights`, `world`, `viewport`, and `fixed`. `mk build
--skip breathe` builds the project's palette alone (quick looks). Renderers need nothing of their own: Blender evaluates
the drivers on every frame and the bake records every driven node value and every light per frame, which mkr draws
([mkr](#mkr)); post mixes its own colours by the same mix ([Post](#post)).

```toml
[look]
palette = "rose-pine-moon"

[look.breathe]                         # night at the drop, melting into Dawn, Dawn in the silence, back at the wrap
palette = "rose-pine-dawn"
keys = [[0.0, 0.0], [2.5, 0.0], [24.7, 1.0], [29.6, 1.0]]
lights = { "*" = 0.7, "sky_moonlight" = 0.0 }
```

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
| `step`, `start` | with `lane`: a counting key. The property holds `start` (default: its value as built) before the first hit and `step` more after each hit (`mkmmd.core.lanes.steps`): a jump on the hit (CONSTANT), or, with `pulse = [[dt, share], ...]`, each step shaped by shares of it (0 .. 1 `dt` seconds after its hit, LINEAR by default; `interp = "BEZIER"` eases them), the next hit cutting a step short and going on from there: a track laid a tie per hit (`rail_track`'s `laid`, `step` = its `track.spacing`). `step` or `start` without `lane`, or a pulse key before its hit (dt < 0), is an error |
| `values` | with `lane`: `[v0, v1, ...]`, the value the property takes on each hit in turn instead of the hit's time (hit i takes `values[i % len(values)]`, cycling; before the first hit the first value), CONSTANT unless `interp` says otherwise: a penlight's `color` index stepping through its colours on every accent. Not with `pulse` or `step`; a list that is empty or holds anything but numbers is an error, and so is `values` without `lane` |
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

A driver can read a position but cannot add up a speed, so a prop whose card has `rates = [{rate, into, per}]` (the `turntable`:
`rpm` into `turns`, per 60) gets its root property `into` keyed by the stage on every frame of the build, after the keys,
with the running integral of `rate / per` over clip seconds (Simpson's rule per frame, 0 at clip second 0, linear past the
ends), once `rate` is animated: key the speed and the platter turns by what it adds up to, a spin-up never jumping the
record. Unkeyed, the builder's own `into` already runs at the starting speed. The report counts its keys as `<prop>.<into>`.

A prop whose volleys each read a start of their own has `deals = [{from, into}]` on its card (the `confetti`: `start` into
`start0`, `start1`, ..., one per volley): once the keys have animated the root's property `from`, the stage deals its
keyframes out after the keys, in time order and in turn, into the properties `into` as CONSTANT keys (keyframe i, at its own
frame and value, into `into[i % len(into)]`), so a `[[key]]` on `start` with a `lane` sets the volleys off one after the
other and each falls its whole life while the next goes off. Unkeyed, the builder's own `into` values stand (the confetti's
first volley at `start`, the others never). The report counts the keys dealt as `<prop>.<into>`.

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
render the solved particles and the penetration reported is the penetration that renders. A body pushes a strand along
without launching it, and a lock pinched near its root (a dancer's arm sweeping it against her back) is pushed, not flung:
the rest-shape targets the damping measures against are held within 6 m/s of riding rigidly with the anchor
(`TARGET_SPEED`), as the pinched segments' frames flap and the targets below would otherwise swing at tens of m/s and whip
the lock out. Results are cached in `<project>/.mk/cache/strands/` by a hash of the inputs and of the solver's source, one
entry per member and level.

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

And it lets the strap of every worn prop move as a band does (`mkmmd.solvers.strap`): the pose stage laid it on the wearer at
rest and sorted its points; here the points that lie on her (from the band's first touch to its last) are carried by her
skin (a helper mesh skinned like her, sampled on every frame, says where), the buttons and a run across a pin are held
by the prop, and the runs between hang: `slack` longer than the straight way (a shallow curve), with gravity, their motion
relative to their ends dying in a fraction of a second, a little bending stiffness, never farther from an end than the
band between is long, and her surface under them (a slack run lies on her chest or her hip). A run's facing goes from
one end's to the other's and leans into its bow, so a run that sags edge on rolls over. The first frame is held for
1 s. The band's mesh on every frame is written beside the scene (`<blend>.wear/<strap>.pc2`) and played by a Mesh Cache
modifier (the bake holds it frame by frame). `--skip sim`, or `sim = false` in the wear's `strap`, leaves the one shape
that rides the prop: a bar. The hair collides with the strap as it was laid (capsules riding the prop). The report has
`straps`: per prop `points`, `free`, `runs`, `bow_mm_max` (how far a run hangs from the straight way), `stretch_pct_max`,
`seconds`.

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
| `pluck` | a `strum` point played with the fingers: its `fingers` table and the neck's strings, per string played (`core.strum.pluck_prop`; the build only) | a station: on the played string's axis in the middle of the plucking zone, x across toward the higher strings, y along toward the nut, z out of the face | `--params` `nfev`, `arm`; the result adds `poses` (each plucking finger's hover, cross, rest) and the report `travel` |

A guitar's `strum` point (a pick pinched between thumb and index) has no style of its own: the build solves it as a `pinch` of
the pick, with the grip frame x into the pick's face and z the squeeze axis, the thumb on the bass side leaning toward the
neck, and with the arm (below).

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
from the frame's x), `clear` (metres, 0.010: how far the middle, ring and little fingers stay from the object), `stop` (metres:
no skin of the hand past the plane x = stop) and `nfev`; `rest` `angles` (`{joint: degrees}` replacing the relaxed priors),
`pitch` and `roll` (`[degrees, tolerance]`), `gap_mm` (0.3) and `nfev`; `neck` `nfev` and `ref` (the solved shape the hand
comes from: Playing a worn guitar). `pinch` and `neck` also take `arm`, the
arm that carries the hand in the grip frame (`{shoulder, a, b, pole}`: the elbow the IK makes; or `{shoulder, a, b, up, out,
avoid, fore_r}`: an elbow chosen out from the shoulder and below it, its forearm `fore_r` (0.028 m) clear of the `avoid` points,
bending the wrist least past the style's free angle and hanging lowest): the wrist stays within 0.96 of the arm's length and
its bend against that forearm within 20 (pinch) or 35 (neck, a fretting wrist is flexed) degrees where the contacts allow;
the report adds `wrist_bend_deg`, `reach`, `elbow`, `elbow_from` and `forearm_clear_mm` (`mkmmd.solvers.grip._Arm`). The
joints are `<finger>_mcp`, `_pip`, `_dip` and `_spr` for `index`, `middle`, `ring` and `little`,
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
type `neck` and `strum`, `use.anchor`), the project gives the numbers (a chord and fret, the times, the rhythm). The library props
`electric_guitar`, `offset_guitar` and `electric_bass` (frame and card: their module docstrings) are cards that do; a bass
plays a bass line one note at a time ([Bass lines and picking](#bass-lines-and-picking), below).

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
| `strap` | `{top, bottom, over, width = 0.05, thickness = 0.004, material, standoff = 0.007, slack = 0.006, sim = true}`: builds the band `<prop>_strap`, and capsules as wide as the band along it (about 4 cm each, their outside on its outer face) that join the prop's colliders: the wearer's simulated hair lies over the strap instead of passing through it (the report's `strap.colliders` counts them); the sim stage then lets it move ([Sim](#sim)) |
| `cable` | `{object, anchor = "jack", radius = 0.0032, trail, out = 0.05, sway = 0.035, reach = 0.12, tail_len = 0.9, slack = 0.15, follow = 0.35, sim = true}`: re-hangs the cord; `sim = false` keeps it hung and hooked instead of swinging it |

The prop's frame needs the guitar's convention: z along the neck, the face looking along -y, x across it. `strap` runs from the
card's anchor `top` over the wearer's shoulder (`over`, a semantic bone: default `arm.L`, `shoulder.L` reads as `arm.L`), across
her back and round her other side to the anchor `bottom`, pulled taut over her own surface as she stands at rest
(`core.wear.strap_path` on `blender.build.wear.wearer_surface`: her meshes without what hangs or swings, her arms and her
head): straight from the upper button to her shoulder, on the shoulder beside the neck's foot and on her back, `standoff`
off her under its whole width (7 mm: the prop sways a little on its bone) and rolled to the slope of what it lies on,
bridging her hollows, passing beside what stands out of her (a tail), and straight again from her side to the lower
button: the diagonal of a real strap, not a belt round the hips. Nothing is tuned by eye: no radius, no arc. That is
the band at rest; the sim stage makes it a soft one ([Sim](#sim)): carried by her skin where it lies on her, hanging
`slack` (a share of their length) loose and swinging between her and the buttons; without it the band rides the prop as
one stiff shape, her hips moving under its lower run. Its ends are tabs: the band narrows toward each button and ends
in a round tip past it. An anchor's direction is
the way its button's pin points (out of the body's edge); a band that would turn back over the prop from a button (the
tail of a body worn across her) first runs 3 cm across the pin, so it comes to the button from behind the prop. A
`material` that does not exist logs a WARNING and leaves the strap without one.

`cable` rebuilds the curve `<prop>_cable` (a bevelled NURBS curve the prop's builder made) from the `anchor`'s jack
(`core.cable.hang`): `out` metres straight out of the plug, a lazy S of `sway` metres, down to the floor under the wearer
(the cast root's height), `reach` metres from under the jack along `trail`, then the cord lying on the floor `tail_len` metres
along `trail` (a horizontal direction `[dx, dy]`, default behind the wearer and to the wearer's right) in loose curves, `slack`
more cord than that run (0.15: 15 %; what the wearer steps and turns in before the cord pulls). The curve is unparented and
hooked at the jack with a smooth falloff of `follow` metres, so its top follows the guitar. The sim stage then swings it from
that shape ([Sim](#sim)): the whole cord answers the guitar's motion, falls against the wearer's legs and lies on the floor it
was hung on (a wearer whose root is keyed off it, falling or jumping, leaves the floor where it is), keyed on every frame;
with `sim = false` (or `mk build --skip sim`) the hooked cord stays. The cord never stretches along what hangs, and what
lies on the floor is held by the floor's friction alone, a dragged weight's (it loses 0.5 g of sliding speed and holds
still under a small slide): lying still it stays where it lies; a wearer who steps away pulls it after her, its curves
straightening from her end first, and it comes to rest draped from the jack to the floor about half a metre off; one who
walks past its slack drags its far end too, and a fast twirl yanks a metre or two of it off the floor for a moment, as it
would (give a wearer who travels a longer `tail_len`, more `slack` or a `trail` toward where she goes). The swung curve
has three steps a 3 cm particle: a 3.4 m lead is 2 700
vertices a frame.

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
each target finger's fingertip (all the skin of its distal phalanx, what is seen to touch: on some models the distal bone runs
centimetres past the skin's end) with its lowest skin on its string and that skin's centroid within 1 mm across the string and
1.5 mm along the neck of its target point, a share (`press`, 0.3) of the way behind its wire (two fingers on one fret of
neighbouring strings stagger up to 5 mm along it), touching with the pad, the distal phalanx steep onto the board, the thumb
pad on the neck's back behind the position's index and middle fingers (the same place for every shape at a position), no skin
in the neck, the knuckles at most 12 mm out of the board (the hand under the neck: contacts alone are also met by a hand
standing in front of the board, its knuckles centimetres out and its fingers curling back onto the strings, a claw over the
fretboard from the front), the other fingers curled just over the strings (their pads over the board, 7 to 12 mm above it,
each over its own fret, one finger per fret), no finger in another, and the wrist within the arm's reach and its bend against
the forearm the IK makes for the hand's `pole` within 35 degrees (the build passes the arm) (`mkmmd/core/fretting.py`: chord
tables, targets). Each start first lets the fingertips reach their targets with the neck seen softly (a reach that ignored it
left the hand inside the neck, and the full solve then pushed it out to the front of the board), and the best start wins by
the gates it misses, then its cost; when even it misses one, twice as many starts run again. `fret` is the
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
both. Every distinct (fret, chord) is solved once and cached (a few seconds each), in time order and each from the shape it
follows (the solver's `ref`): the first shape solved at its position, so every shape there keeps that hand (its palm within
millimetres, a few more along the neck for a stretch, the thumb where it was, the free fingers' joints) and moves only the
fingers that change; at a new position the shape before it, so the hand slides along the neck from where it was. A keyed
change passes through both shapes eased off the strings (`fretting.lifted`: every finger 3 mm over its place, solved from the
shape): the fingers lift over the first quarter of `move`, the hand moves, they land over the last quarter
(`fretting.change_keys`); a pressing finger presses from its shape's landing until the next move starts.

The strum point (`center`, `along`, `across`, `normal`, `pick = {object, thickness, length, width, tip}`, `avoid`, and optionally
`palm = {center, normal, size: [across, along], lift}`: a patch over the strings by the bridge the heel of a picking hand rests
over, at most `lift` above it, as a pick player's hand rests on the bridge; the pinch solve pulls the heel there) is solved as
a `pinch` of the pick: the grip frame has x into the face and z the squeeze axis from the index pad to the thumb pad, on the
bass side (against `across`) leaning 45 degrees toward the nut (`core.gripframe.strum_frame`; `thumb = "neck"`, `"bridge"`
toward the bridge), its origin `tip` above `center` (the hand's `tip`, else the card's `pick.tip`, else 0.008); no skin of the
hand comes nearer than `clear` to the strings' plane (the hand's `clear`, else the card entry's, else 4 mm: with a pick held
far back, a `clear` near its `tip` keeps the fingertips from reaching down to the strings beside it), held as the hand's `hold` says, else the card entry's, else
`"along"`: the thumb and the index finger point down the pick at the strings and meet it with their tips, its wide end in the open
between them (a card held by its edge); `"across"` is how a plectrum is held: the hand's own frame is the pick's turned a quarter
about the squeeze axis (`core.gripframe.hand_frame`), so the thumb lies across the pick along the strings, its pad on the wide end,
and only the point shows under the thumb's side (the fingers within 35 degrees of the pick's width; the stop plane is then the
frame's y). A small pick suits a small hand: the `offset_guitar`'s is 24 x 20 mm, held across; and the solver is given the arm: the elbow is chosen out from the shoulder
and below it with the forearm clear of the card's `avoid` objects (the library cards name their body: the forearm comes over
the body's edge, never through it) and the wrist straight on it (20 degrees free), and the build puts the pole where that
elbow points. The pose stage warns when the forearm still passes more than 10 mm into the prop (the arm is too short to come
over the edge there: wear the prop lower or further toward that arm), when a wrist bends more than 55 degrees against its
forearm or reaches past 0.98 of the arm's length;
the pick object, modelled in that frame, is bone-parented to the wrist. Hands on a `neck` or `strum` point ride the prop
(`ride` defaults to its root), and on a prop the character wears the elbow's pole rides it too: the arms keep their shape
when she steps, turns or spins (a pole left standing in the world twists them as soon as she leaves the spot).
`mk grip neck --side L --card @card.json --chord power --fret 3` solves one without a build.

**Strumming** (`[perform.<cast>] strum`; the keys and the stroke model are under [Perform](#perform), `strum`, and in
`mkmmd/core/strum.py`). The strokes come from the song: `rhythm = "onsets:other"` plays the plucks the timeline's detector
found in that stem (`mk timeline onsets`, see [Timeline](#timeline)), `"beats:8"` a grid of eighths from the beat list, or a
list of times; `from` and `to` pick the window, `accent = "downbeats"` accents the first beat of each bar, and several tables
on one hand give different rhythms in different parts of the song. The build keys the pick hand's goal on every frame of the
window, relative to the grip, so the arm's reach is part of the result. `plan` and `path` are numpy only and tested without
Blender; the `strum` check replays the plan on the timeline against the scene.

**The tremolo arm in the hand.** An `offset_guitar`'s arm is an object of its own (`<name>_trem_arm`) that turns about its bend
above the collet to aim at the empty `<name>_trem_aim`, which stands at its tip. The card's anchor `trem` names that empty and the
picking hand's bone (`wrist.R`), so `[[prop]] anchor_to = "<cast>"` beside `wear` parents the empty to her hand at the settled pose
([Props on the body](#props-on-the-body)): the arm's tip then stays with the hand through every stroke of a `strum`, swinging and
dipping as she plays, the way these guitars are strummed in shoegaze. Lay the arm into the hand with the prop's `arm` option (`swing`
toward the strings, negative past them; `lift`; `length`): look at a close-up of the picking hand, the tip should end in the curled
ring and little fingers. Without `anchor_to` the arm stays where it lies.

### Bass lines and picking

A fretting hand plays a bass line instead of chords when its table has `notes`, and the pick hand picks it one string at a
time with `rhythm = "notes"` (`mkmmd/core/fretting.py`: `line`; `mkmmd/core/strum.py`: `pick_notes`, `plan`):

```toml
[[prop]]
name = "bass"
card = "library:electric_bass"
wear = "rin"

[pose.rin.hands.L]
grip = "bass:neck"
notes = { lane = "bass_notes", from = 24.42, to = 28.84, position = 7 }    # a note lane of [audio] hits (mk timeline hits --notes bass_notes=bass:30-400)
[pose.rin.hands.R]
grip = "bass:strum"

[perform.rin]
strum = { hand = "R", prop = "bass", rhythm = "notes", from = 24.42, to = 28.84 }
```

| `notes` key | Meaning |
|---|---|
| `lane` | a lane of the `[audio] hits` file whose hits carry a `midi` pitch (`mk timeline hits --notes NAME=STEM:LO-HI`), or |
| `times` | `[[t, midi], ...]`: clip seconds and MIDI note numbers (one of `lane` and `times`) |
| `from`, `to` | clip seconds: the notes inside are played (inclusive; default all) |
| `min_db` | a lane's hits under this level are left out (0 is the lane's loudest) |
| `octave` | whole octaves added to every pitch (default 0): a detector an octave off, or the line an octave lower |
| `tuning` | the MIDI pitch of each open string, the lowest first (default the card's neck `tuning`; the library bass says `[28, 33, 38, 43]`, E1 A1 D2 G2) |
| `position` | the lowest the hand may go: the index finger never below that fret (default 1), so every fretted note is on it or higher and the least travel is found within it; open strings stay allowed (the hand does not move for them), and a note with no string and fret from there up raises. A bassist keeps a line near the body this way: a 30 inch neck's first frets are out of a standing character's reach with the bass at the hip, its 7th position is not (the library bass is worn for it) |

Every pitch is rounded to a semitone and placed on a string and fret over the whole line at once (dynamic programming): one
finger per fret (a note on fret f at position p is played by finger f - p, index to little, so p is f - 3 .. f), open
strings allowed, minimising the hand's travel along the neck (metres between the position wires) plus a cost per shift, a
small one per open string (a fretted note shows the line; an open one saves a shift), per string crossed and per fret (the
lower of two equal places). A note no string and fret can play raises with its time and pitch (`octave` or `from` / `to` mend
it). Each change of note becomes a key of the neck grip (as `keys` do; `notes` replaces `fret`, `chord` and `keys`): the
one-finger shape `{finger = [string, frets above the position]}` in place at the note's time, reached over the hand's `move`
(0.12 s) cut to 0.75 of the time since the note before, so the hand arrives just before each note; a repeated note keys
nothing. On an open string the finger of the note before lets go and hovers 3 mm over its string where it was (a shape's
finger written `[string, frets, "hover"]` does that anywhere; strings are numbered from the highest, 1, to the lowest, 4 on
a bass). Every shape at the line's position is solved from its first one, so between two notes only the playing fingers move
(a stretch to the little finger slides the palm a centimetre or two along the neck). The grip digest has `notes`: how many,
the shapes, the strings and positions used, the open notes, the fret range.

Picking (`rhythm = "notes"`): the strikes are the notes of the other hand's `notes` (the fretting hand must grip this prop's
neck), each a stroke on the string its note is played on: it meets that string's near side at the note's time (u = c - d w:
c is the string's place on the strum axis at the strum centre, w its radius and half a millimetre) and leaves its far side
after `attack`, the tip `depth` past the strings' axis plane so it catches the string, over the other strings `lift` above
them and to and from rest lifted by `return_lift`. The strokes alternate strictly, down first and down again after a pause
long enough for the hand to rest (lead, attack, follow, approach, retreat and 0.1 s). A picking table's defaults are
`span` 0.026, `depth` 0.002, `lift` 0.006, `attack` 0.012, `lead` and `follow` 0.05, `approach` and `retreat` 0.25,
`return_lift` 0.008, `accent_gain` 1.15, `up_scale` 1 and `min_gap` 0.05 (`strum.PICK`; the table's own keys win). The strum
report adds `picked` {strokes, strings}; the `strum` check judges every picked stroke (its `strokes` default is `all` for a
picking spec) by the tip's distance from the string of its note when it crosses it, with the timing of the crossings.

Fingerstyle (`technique = "fingers"`, the library bass's default: most bassists play with their fingers): the same notes are
played with REST STROKES. The index and middle fingers alternate strictly (the index first, and again after a gap longer than
`pause`); each stroke is one steady swing of the fingertip from hovering over its string (on the side toward the higher
strings) through it toward the lower strings to rest on the next lower string (on the lowest string toward the thumb), and
`Stroke.t` (`finger` set, `dir` -1) is when the fingertip crosses the string's axis, the moment the string is let go. The
thumb is anchored: on the pickup cover's top beyond the lowest string while that string rings, on the lowest string while a
higher one does; the hand moves between these STATIONS (one per string played) just before the notes that need it, the other
finger hovers ready, the ring and little fingers stay curled off the strings, and the pick is not held (it stays where the
builder laid it). The pose stage solves a station per string with the grip style `pluck` (`mkmmd/solvers/grip_pluck.py`: the
frame, the thumb on its anchor, both fingers' crossings, the arm as for a pick: the forearm over the body's edge; then each
finger's hover and rest with the hand held) and perform keys the wrist goal and every finger joint per frame from them
(`core.strum.pluck_keys`: the swing is timed from where each solved pose has the fingertip across the string, `travel`, and
`core.strum.retime` keeps its speed steady along each blend, from the fingertip's places the solve measured at nine blend
weights of each segment, `curves`, so the crossing falls on the note's time on every frame rate). The
card's strum entry says where: `technique` and `fingers = {zone: [along0, along1] (where the fingers pluck), anchor: {center,
normal, size: [across, along]} (the thumb's rest on the cover), face: {point, normal} (the surface under the strings), blocks:
[{center, size}] (boxes standing on it: the pickup covers)}` (`core.strum.pluck_prop`); a card without `fingers` is played
with a pick. Fingers with another rhythm than `"notes"` raise. The `strum` check judges every rest stroke by the fingertip's
skin (the distal phalanx's skin nearest the note's string): its gap to the string when it crosses the string's axis toward
the lower strings, and the time it does.

Checks to keep: `joint_limits` (the default elbow pole is under the shoulder; give `pole` per hand for another), `contact`
between `bone("wrist.R").tail` and `obj("<cast>_hand.R").loc` (the IK reached its goal at every frame, `max` 5 mm), `strum`
(distance at the down strokes, or every picked or plucked stroke; timing), `prop_body` (the guitar against the character's body), `form`
on the prop. A bass is worn lower than a guitar: the library bass's card hangs it at the hip with the neck 25 degrees up (`at`
[-0.23, -0.07, -0.13], `neck_deg` 25, `yaw_deg` 8) and its pick zone between the bridge and the pickups (the heel of a picking
hand over the strings by the saddles, its `palm`), where a pick
player plays a Mustang and a hanging right arm reaches; with that, play the line from a `position` near the body (7 reaches on
a small anime body; the first positions would need the bass hung high and far to the right). A longer-armed character can take
a lower position or wear it differently with `wear = {cast, at = [...], neck_deg = ...}`.

### Strings that ring

Every string the pick hand plays vibrates on screen, and only those (`mkmmd/core/strings.py`, numpy, tested;
`mkmmd/blender/build/strings.py` keys it at the end of the perform stage's strum). It needs a card whose neck `strings` name
their objects (`object`: one mesh per string with a ring of vertices at every wire and every press point; the library guitar's
`<name>_string_1` .. `_6` and bass's `_1` .. `_4`, 1 the highest, are built so); a card without them logs a WARNING and nothing
rings.

- **Presses.** A fretting finger holds its string down from its shape's landing until the hand leaves for the next shape (the
  pose stage's neck states, `fretting.hand_states`), coming down over the last 0.05 s of the move and letting go over 0.04 s
  (never more than a quarter of the move: the hand lifts its fingers in the first quarter of a change and lands them in the
  last); a "hover" finger holds nothing. A held string runs straight from the nut down to the finger
  (the `press` point behind its wire, at the wire's crown plus the string's radius: where the pad holds it), along the crown
  and straight up to the saddle: the shape key `press_<fret>`.
- **Hits.** A picked note sounds when the pick tip crosses its string's axis (the stroke's `t` plus half its `attack`), a
  plucked one (fingerstyle) at its stroke's `t`; a strum crosses its strings one after another across its attack at the
  contact speed of the stroke (down: the lowest first). A strum rings only the strings the shape at that moment holds and the
  open strings its named chord lets ring (`strings.OPEN`: power chords none, open E all six; a chord table rings only the
  strings it presses; no fretting hand: every string open).
- **Rings.** A hit string vibrates from the fret held on it then (the highest, or the nut) to the saddle, its blur's half-width
  at the antinode `amp` times the stroke's amplitude, decaying with time constant `decay`, until the next hit on it, the finger
  that frets it letting go or, picking a line one note at a time, the next note on any string; then it is damped over `damp`.
  Each frame flickers by `shimmer`.
- **What renders.** The vibrating length becomes a spindle (the shape key `ring_<fret>`, its value the amplitude over 4 mm):
  each ring of the string's surface an ellipse of half-axes r + a p(u) across the board and r + 0.45 a p(u) out of it, p(u) =
  sin(pi (u - u0) / (1 - u0)) from the fret's place u0 to the saddle, so it is widest midway along the vibrating length. Its
  material (any material of a string object, given a coverage mix once) draws it as a filmed string reads at 30 fps: a
  translucent band, its middle as covered as a string swept over its width is (2 r / (pi a)), denser toward its edges where
  the string turns, a faint shimmer along it, opaque again as it dies down; a still string is opaque. The object carries
  `mk_amp` (m), `mk_u0`, `mk_phase` and `mk_r` (custom properties the material reads, keyed with the shape keys) and the
  vertex attribute `mk_u` (its place from the nut, 0, to the saddle, 1). The flat looks (silhouette, vector, mk's renderer)
  and `mk look --map` see the spindle as geometry, opaque; the shape keys bake as morph targets, so a frame costs a few shape
  keys on four or six small meshes.

| `strum` key `ring` | Meaning |
|---|---|
| `true` (default) / `false` | the strings ring / stay still |
| `{amp, decay, damp, shimmer}` | metres at the antinode (default 0.0031 per metre of the card's `scale`: 2.4 mm on the bass, 2.0 on the guitar), seconds to 1/e (0.9), seconds a damped string takes to stop (0.06), the frame-to-frame flicker (0.06) |

The strum report adds `strings`: `played` (per string its rings, the frets it rang from and the frets held on it), `hits`,
`keys` and `amp_mm`. The `string_play` check measures every note against the plan: the right string ringing from the right
fret, the fretting fingertip on it behind its wire, the pick or fingertip crossing it on time ([Checks](#checks)).

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
| `plate` | `true`: the shot is not in the cut (no marker; `from` / `to` optional). It exists for a `[[transition]]` or `[[insert]]` to take frames from, or a collage's figure (`[[fx]] shot`), drawn through its own camera and look at the frames wanted; a plate that no effect uses is skipped with a WARNING. A shot in the cut that an effect takes frames from before its `from` is keyed over those frames as well |
| `mount` | prop, set or object the camera rides: `at` is in its frame and its motion carries the camera. An unknown name is an error (default: the world) |
| `at` | camera position: `[x, y, z]` in the mount's frame (the world without a mount) or a dict target, `{path = "road:road", s = 640, offset = 7, z = 1.2}` (beside a set's path), `{prop = "car", point = [x, y, z]}`, `{cast = "rin", point = [x, y, z]}`, resolved in the world on every frame (the mount does not apply to it); `keys[].at` takes either form too, and a camera between two keys moves between the world points they give on every frame (default `[0, 0, 0]`) |
| `look` | what the camera aims at: any [target](#targets), `[x, y, z]` (world), `"cast:rin.head"`, `"cast:rin"` (the eyes), `"car:road"` (a card's use point), `{path = ...}`; read on every frame, so a moving target is followed (default: straight ahead, along the mount's -Y) |
| `lens` | focal length in mm on a 36 mm sensor that spans the larger image side; every camera clips at 5000 m and starts where its view begins, frame by frame (see Clip ranges, below) (default 35) |
| `roll` | degrees about the viewing axis; positive turns the picture clockwise; keyed in `keys[].roll` (default 0) |
| `lag` | operator lag in seconds: the aim follows its target through a critically damped filter of that time constant, applied in the mount's frame, so a camera in a car lags the subject, not the road (default 0) |
| `follow` | `{target, axes = "x", lag = 0, from}`: the side-scroller lock. On every frame the camera's fixed position and aim point (`at` and `look` given as `[x, y, z]`, the keys' too) move by the world displacement of `target` (any [target](#targets)) since the frame of clip second `from` (default the shot's own `from`), along the world axes in `axes` only (distinct letters from `"xyz"`: `"x"` runs with her and leaves her hops to rise in the frame, `"xyz"` holds on her), eased through the critically damped filter of `lag` seconds that the `lag` key aims through. A target given as `at` or `look` already moves and is left alone; the move is added after the aim's own lag, so the camera never turns by it. It reads the target where the stages before shots put it (`walk`, `rise`, `tumble`, vehicles, motions), not a `[[key]]` or the sim. A target that names nothing, axes that are not letters from `"xyz"` and a negative lag are errors, and so is `follow` with `mount`. The report's `moves.<output>.follow_m` is the farthest it went |
| `shake` | handheld shake in degrees: smooth random turns of the camera about its three axes (noise low-passed with a 6-frame time constant, seeded by the camera's name so a rebuild is identical); about `shake / 2` rms in pitch and yaw and a quarter of `shake` in roll (default 0) |
| `keys` | `[{t, at, look, lens, roll, shift}, ...]`, a move inside the shot; `t` is in clip seconds (not relative to the shot). Between two keys `at`, `lens`, `roll` and `shift` move with a smoothstep ease and `look` switches to the next key's target halfway, so aim at something that moves rather than between two fixed points. A field a key leaves out is the shot's own; the first and last key hold outside their span |
| `frame` | `{subject, fill, solve}`: solves the lens, or the distance, for each output so the subject fills a share of the frame height. `subject` is a target or a list of targets (one world point is written `[[x, y, z]]`); its height is the vertical extent of those points (at least 0.25 m) plus 0.15 m of headroom, taken at the median of about twelve frames of the shot. `fill` is the share of the frame height (default 0.45). `solve = "lens"` (default) sets one lens for the whole shot (a keyed `lens` is replaced), `"distance"` keeps the lens and moves the camera along the aim line by one factor, `"dolly"` keeps the keyed lens and moves the camera along the aim line on every frame so the subject keeps `fill` (a dolly zoom: a lens keyed from 28 to 75 mm sends the camera 2.7 times as far, she keeps her size and the room behind her swells; exact for a subject on the aim point; the report's `moves.<output>.dolly_m` is the range of distances); any other solve is an error |
| `shift` | `[x, y]`, Blender's lens shift: fractions of the larger image side, the picture moving the other way (a positive `y` moves it down; probed in Blender 4.2.3); constant, or keyed in `keys[].shift` (default `[0, 0]`). See the crop below |
| `fisheye` | `{fov = 180, projection = "equisolid", fill = "frame"}`: a real fisheye camera, drawn by Cycles alone ([Fisheye shots](#fisheye-shots)), sized for each output. `fov` (degrees, 30 to 360) is the field across the frame's diagonal (`fill = "frame"`, the default: a full-frame fisheye, its image circle round the frame) or across the circle (`fill = "circle"`: the whole circle inside the frame's short side, black outside); `projection` is `"equisolid"` (default, as most real fisheyes) or `"equidistant"`. `at`, `look`, `roll`, `keys`, `lag`, `follow`, `shake`, `mount` and `dof` work as for any shot; `lens`, `frame`, `shift` (and a key's `lens` and `shift`) and the looks (`style`, `reflection`) are refused, saying why. `fisheye = false` in an aspect table drops it for that output |
| `dof` | `{focus, fstop, offset}`: depth of field; `focus` is any target (required), keyed as the distance from the camera on every frame, less `offset` metres (default 0): the focus plane sits that much nearer the camera than the target. A character's eye bones (`"cast:rin"`) are inside the head, 4-8 cm behind the face's surface (more for `.head`, the base of the skull), while a close-up at f/2.8 keeps only a few centimetres sharp: give a face target `offset = 0.05`. `fstop` (default 2.8). Without `dof` nothing blurs |
| `aspect.<output>` | `{...}`, `[shot.aspect.<output>]`: that output's own version of the shot. It may hold any key above except `name`, `from`, `to`, `plate` and `aspect`. The tables `frame`, `dof`, `follow`, `fisheye`, `colors`, `tones`, `knockout` and `reflection` merge key by key with the shot's own (`frame = { fill = 0.6 }` keeps the `subject`, `follow = { lag = 0.4 }` the `target`, `fisheye = { fill = "circle" }` the `fov`); every other key, targets and lists included, is replaced whole; `reflection = false`, `follow = false` and `fisheye = false` switch an inherited reflection, follow or fisheye off for that output. A table for an output the project does not have is an error |
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
without `subject`; a window of an effect that needs frames from a shot before `[scene] start` (`lower [scene] start`); bad look
tables (the message names the shot and the output: `shot 'storm' (16x9): ...`); a `fisheye` with a bad `fov`, `projection` or `fill`,
or with `lens`, `frame`, `shift` or a look beside it, as any output sees the shot (`shot 'low' (1x1): `lens` with `fisheye`: a
fisheye's focal length comes from its `fov` and `fill` ...`); and a `[[medium]]` window whose look covers a fisheye shot's frames.
It logs a WARNING for a plate no effect uses and for an object pattern that matches nothing.

### Fisheye shots

A shot with `fisheye = {...}` gets, per output, a camera of type `PANO` (Blender 4.2's own camera properties, no longer under
`camera.data.cycles`): `projection = "equisolid"` is `panorama_type = "FISHEYE_EQUISOLID"`, which images the ray `theta` off
the axis at r = 2 f sin(theta / 2) sensor mm from the centre with `f = fisheye_lens`, black past `fisheye_fov`;
`"equidistant"` is `FISHEYE_LENS_POLYNOMIAL` with `fisheye_polynomial_k1 = -1 / f` and k0, k2, k3, k4 = 0 (theta = r / f),
because Blender's own `FISHEYE_EQUIDISTANT` ignores the sensor and stretches its circle to both sides of the picture (an
ellipse in a 16:9 frame). The sensor is 36 mm across the picture's longer side (`sensor_fit = "AUTO"`), and `f` is solved for
each output's size so the frame's half-diagonal (`fill = "frame"`) or half its short side (`"circle"`) lies at `fov / 2`;
`fisheye_fov` is the field for a circle and a hair (1 %) past the corners for a full frame, so no corner pixel's filter
reaches black (`mkmmd/core/fisheye.py`: `lens`, `cut`, `blender`; the report's `fisheye` gives each output's `lens_mm`).
Rendering markers at known directions in Blender 4.2.3 (1:1, 16:9, 9:16, both projections, sensor fits, lens shift) put every
one within half a pixel of these formulas; Cycles clips a panoramic camera's rays at `clip_start` along each ray.

Only Cycles draws a panoramic camera: EEVEE and Workbench draw it as a perspective one. `mk render` (and so `mk post`, which
composes what it drew) refuses, before it draws anything, a frame or a plate seen through a fisheye under another engine,
naming the fix (`[render] engine = "cycles"` or `--engine cycles`; with `engine = "mk"` the shots Blender draws use the
scene's own engine, so set `cycles`). `mk look` draws the images seen through a fisheye with Cycles whatever `--engine`
says (with its `--samples`). The looks are refused on a fisheye shot; the passes other
layers take through its camera are drawn by Cycles (each surface a flat emission through its MMD alpha): the figure
layer (`figure/<key>/<frame>.png`, the collages' and a masked pixel sort's or mosh's) and `[post] keep`'s mask, so they line
up with the frame.

Everything that places scene points in the picture projects through a fisheye with the same maths (`Lens`): the `framing`
check (each output through its own camera of the shot in the cut; a circle's edge bounds the frame too), `mk q`'s
`screen()`, the cut effects' scene points (`{point = ...}`: `p` its place, `m` the scale of a patch facing the camera),
the rings, `mk look --map`'s pixel maps (rays bent as the lens bends them; `depth` along each ray) and the clip stage
(below: a fisheye's nearest geometry is a distance in its cone). The bake records a fisheye's lens: mk's own renderer
leaves its shots to Blender and the site's 3D view shows the free view for them. `occlusion` and `camera_inside` cast
rays and need no projection.

### Clip ranges

A render tells two surfaces on one line of sight apart only down to a depth step that grows as the square of their distance
over the camera's `clip_start`: from a 2 cm start, about 0.05 mm at 4 m. MMD models keep their drawn expressions (star and
heart eyes, a blush, tears, `><`) a few hundredths of a millimetre behind the skin until a morph brings them out, so a camera
that starts at 2 cm shows them all at once through her face from a couple of metres away. The clip stage (after everything
that moves geometry, before `save`) keys every shot camera's `clip_start` (CONSTANT) on every frame it is keyed over to half
the depth of the nearest geometry inside its view, the view widened by a tenth on each side, and never under 1 mm (1 m where
it sees nothing): as late as it can start without cutting anything off (`mkmmd/core/clip.py`, `mkmmd/blender/depth.py`). The
depth is exact, read coarse to fine (each drawn object's box, the boxes of chunks of 256 of its triangles, those triangles;
geometry instanced by another object counts as its box), on the world a render sees (a freeze's held frame with the camera
on its own). A fisheye starts at half the distance of the nearest geometry whose bounding sphere reaches into its cone of
sight (its widest angle off the axis the picture shows, widened by a tenth, at most all round), as Cycles clips each of its
rays along the ray (`mkmmd/core/clip.py`: `least_distance`); what lies just outside the cone may bring it nearer, never
farther. `mk look`'s orbit cameras start the same way on each frame and size, and the bake keeps every camera's clip
range per frame for mk's own renderer. Its report: `cameras`, `frames`, `near_m` (least, median, most) and `touching`
(frames where something was within 2 mm of a lens). `mk build --skip clip` leaves every camera at 2 cm.

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

`pen = "crayon"` draws it in waxy crayon instead (`mkmmd/core/crayon.py`, numpy only, shared with the [activity
book](#activity-book-and-labels)): each line pressed hardest along its middle, its edges broken, the paper's tooth (a grain
fixed to the frame by `seed`) showing through the wax by `grain` (a crayon's default 0.8), drawn again `fps` times a second.
`inks` colours the lines part by part: a colour or a list of them cycled over the line groups, or `"tones"`, each
material's lines in its tone's colour pulled 40 % toward `ink` (an outline takes the part inside it, a line where one part
passes in front of another the nearer part's); without it every line is `ink`. `fill` (0..1, default 0) colours every part
in first with back-and-forth strokes in its tone's colour, each part's strokes leaning their own way, the paper between
them (a crayon with a `fill` reads the textures for a drawn or cel tone); `inks` and `fill` are a crayon's only. Her in
crayon line art on a Dawn page: `sketch = {pen = "crayon", paper = "base", ink = "text", inks = "tones", boil = 1.4}`
with `lines = {outline = 7, inner = 4}`.

`cutout = true | {rim, shadow, rough, grain, lift, fps, seed}` (a shot's or a medium window's, not with a `sketch`) cuts
the look's fills out of construction paper (`mkmmd/core/cutout.py`, numpy only): no lines; every line group a piece of
coloured paper with a mottle and fibres of its own (`grain` 0..1, 0.6), its edge cut by a wandering pair of scissors
(`rough` px at 1080, 1.6) again `fps` times a second (8; 0: one cut, a stop-motion boil otherwise), grown a little and set
off its place by up to `lift` px (3), laid far to near (by depth), each casting a soft shadow on the pieces under it. The
figure stands on a white paper `rim` (`{color, width}`, white, 10 px at 1080; `false`: none) and casts a drop `shadow`
(`{color, offset, blur, alpha}`: black, `[0.005, 0.008]` frame heights, 5 px, 0.35; `false`: none) on the background,
the look's `background` colour as paper. Flat or two-tone tones read as paper colours; a drawn or cel tone keeps its
picture on the piece. Rings flip a cut-out to its `opposite` palette as they flip a vector look. A construction-paper flash:
`[[medium]] {from, to, style = "vector", colors = { background = "gold" }, cutout = true}`.

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
`tones`, `materials`, `lines`, `sketch`, `cutout`, `hide`, `keep`, a silhouette's `accent`, `tint`, `knockout`, `grow`, `samples`;
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
(`round(dur * fps)` frames, at least 2, so the incoming shot lands on the beat; a spin's runs on `after` seconds past the cut),
the whole `from`..`to` of an insert (its frames `from` .. `to` - 1). **Plates** are the other shot's frames through its own
camera and look. The layers sit in the output's frame folder, named by Blender frame (`<frame>.png`, five digits):

- `plate/<shot>/<frame>.png`: a shot's frame; a shot that is in the cut at that frame is its own plate: the cut's frame. A
  collage's own `shot` ([Footage, stickers and figure moves](#footage-stickers-and-figure-moves)) lends its figure's picture so.
- `matte/<shot>/<frame>.png` and `back/<shot>/<frame>.png`, for an expand or collapse (and a slice through the figure, on its first
  window frame): the flat shot's figure alone as coverage, drawn at twice the frame's size, and its frame without the figure.
- `point/<key>/<frame>.json`: a projected anchor or centre, `p` (frame fractions from the top left), `depth` and `m` (frame heights
  per metre at that depth, which sizes things given in metres); the key is `t<i>` for transition `i`'s `center`, `i<j>` for insert
  `j`'s `anchor`, `x<k>` for frame effect `k`'s.
- `figure/<key>/<frame>.png`: a collage's figure, the coverage of its objects alone through the camera of the shot in the cut, or
  of the collage's own `shot`, as grey ([Frame effects](#frame-effects)).

Layers are claimed with an empty file like frames (the finishing file is written last), so a stopped render resumes, `--jobs N`
shares them and the disk check counts them. `mk build` keys the camera of a shot that lends frames (`keyed` in the shot table).

A transition has these keys; the kind decides the rest, and a key of another kind is an error that lists the known ones.

| Key | Meaning |
|---|---|
| `at` | clip seconds of the cut (required): exactly one shot of the cut must start on that frame (use that shot's `from`); the window is the frames before it, inside the outgoing shot (a spin's goes on past it, inside the incoming shot). A `page` may also stand at the clip's end, `[project] duration`: the loop's wrap (below) |
| `kind` | `"expand"`, `"collapse"`, `"slash"`, `"whip"`, `"slice"`, `"spin"` or `"page"` (required) |
| `dur` | window length in seconds, ending at the cut (default 0.4; 0.18 for `slash`, 0.15 for `whip`, 0.3 for `slice` and `spin`, 0.6 for `page`) |
| `ease` | `in` (u², slow start), `out` (1 - (1 - u)²) or `inout` (smoothstep) (default `in`; `inout` for `slash`, `whip` and `page`; a spin has none: its turn is set, below) |

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

`slice`: the outgoing frame is cut in two along a straight line and the halves slide apart across it, each its own way, eased,
off the frame by the cut, the incoming shot in the widening gap: the first window frame opens a crack along the cut, the last
leaves slivers at the frame's edges. Each half travels from the cut to the frame's farthest corner on its side
(`mkmmd.core.transition.slice_reach`), so a cut off the middle sends its bigger half faster. Like a slash it needs only the
next shot's plates (and, for `center = "figure"`, the outgoing shot's figure on the window's first frame).

| Key | Meaning |
|---|---|
| `center` | the point the cut runs through: `[x, y]` frame fractions from the top left (default `[0.5, 0.5]`), or `"figure"`: the centroid of the outgoing shot's figure on the window's first frame, held over the window (that shot needs a flat look in every output; a first frame without the figure is an error) |
| `angle` | degrees off vertical, clockwise positive (default -20, as a slash's) |
| `edge` | `{color, width}`: a line along each half's cut side, inside it, as a sticker's white border: `color` (default `text`), `width` in px at 1080 on the short side (default 4). No line without the key |

`spin`: a record spin. The outgoing frame turns about `center`, faster and faster up to the cut; the incoming one takes over under
the blur on the cut, still turning, and slows to rest upright `after` seconds later. The window straddles the cut: `round(dur *
fps)` frames before it, in the outgoing shot, and `round(after * fps)` from it, in the incoming one (each at least 2; the spin
may not run past the incoming shot). Both sides are the cut's own frames turned, so it needs no layers. The turn is a steady
acceleration from rest (on the frame before the window) to a top speed between the outgoing shot's last frame and the incoming
one's first, then a steady slowing to rest (on the frame after the window), `turns` x 360 degrees in all, each side turning its
share in proportion to its time so the speed is the same on both (`mkmmd.core.transition.spin_angle`). Each frame is the mean
of copies turned across what the shutter (`blur` frames) sweeps, at most 1.5 px apart at the far corner, and its scale dips
smoothly to `scale` at the cut and back.

| Key | Meaning |
|---|---|
| `after` | seconds it spins on after the cut, in the incoming shot (default `dur`) |
| `turns` | turns in all, both sides of the cut together (default 1.5); clockwise, negative for counter-clockwise |
| `blur` | the shutter in frames, 0 .. 4: how far each frame's smear reaches (default 1; 0 is sharp) |
| `scale` | the picture's scale at the cut, in (0, 2] (default 0.9: a slight dip; 1 for none) |
| `center` | `[x, y]` frame fractions it turns about (default `[0.5, 0.5]`) |
| `back` | what shows where the turned picture leaves the frame: `"mirror"` (default: the frame reflected at its edges, as editors' spin transitions do) or a colour (palette slot, `#hex`, `"a:b:t"` or `[r, g, b]`) |

`page`: a picture-book page turn. The outgoing frame is the page: it lifts at its edge (a corner first, by `angle`), curls over a
roll of `radius` that travels across the frame and lies turned over on itself, its paper back showing, until all of it has left
the frame by the cut; under it is the next shot. Every pixel is worked out from the roll (`mkmmd.core.pageturn`): the page's
front flat and then up the inside of the roll, dimmer the deeper in; over it the page's back, round the top of the roll and laid
turned over beyond it: the `back` paper with the picture showing `through` it from behind (the same point of the page seen from
the other side, so mirrored), a fine paper grain, lit as a cylinder from the upper left (brightest on the roll's top, dim on its
far side, a soft sheen along it); the lifted page casts a soft `shadow` to the lower right on the next shot and the flat page.
The roll's line travels from the page's farthest corner (nothing lifted) to past its nearest one by the roll (all of it gone),
eased, never at either end inside the window: the first window frame lifts the corner, and the page is gone on the cut. Like a
slash it needs only the next shot's plates. At the clip's end (`at` = `[project] duration`, where no shot starts) it is the
loop's wrap: the clip's last frames turn over onto its first frame, so the video loops through the turn; the outgoing shot is
the last shot of the cut (a clip of one shot turns onto itself). The wrap is not one of the cut effects, which work on the
cut's frames: it turns finished pictures. The page is each frame of its window as finished, its `[[fx]]`, lens streaks and
screen type on it; under the page is frame 0 as finished by the same steps (its own cut effects, `[[fx]]`, streaks and type),
made once per output and held still. The grade runs on the result as on any other frame; screen type laid over the graded
picture (`[post] screen = "over"`) turns with the page instead of lying over it, each frame's own on the page's front and frame
0's under it, and so do the `[post] keep` masks the grade reads. So where the page has gone the wrap's frames show frame 0
exactly as the loop starts again on it (but for the page's shadow and the grade's grain, which is each frame's own), whatever
effects or type either end carries: frame 0 may have `[[fx]]` like any frame. `mk post` keeps frame 0 as it finishes it;
`mk look` draws it beside the wrap's frames it is asked for and finishes it as it does them. Only a page turn wraps: any other
kind at the clip's end is the error any cut without a shot gets. `mkmmd/core/pageturn.py` checks it and does the geometry,
`mkmmd/pageturn.py` draws it.

| Key | Meaning |
|---|---|
| `dir` | the way the page turns over: `"left"` (default: lifted at the right edge and turned toward the left, as a book's page), `"right"`, `"up"` or `"down"` |
| `angle` | degrees the curl's line is turned off square to `dir`, clockwise positive, -60 .. 60: the page lifts at one corner first (default 20: with `left` the bottom right corner, -20 the top right) |
| `radius` | the roll's radius, a share of the frame's short side, 0.01 .. 0.5 (default 0.12) |
| `back` | the paper of its back (palette slot, `#hex`, `"a:b:t"` or `[r, g, b]`; default the palette's lightest: Dawn's `surface`) |
| `through` | how much of the picture shows through the paper from behind, 0..1 (default 0.18) |
| `shadow` | how dark the shadow the lifted page casts, 0..1 (default 0.45) |

An insert puts another picture over a window of the host shot: a thought bubble holding another shot (picture in picture, the
keys below), a split screen (`shape = "split"`, its own keys further down) or an iris (`shape = "iris"`, a round window opening
from a point of the scene, after the split's).

| Key | Meaning |
|---|---|
| `from`, `to` | clip seconds (required, `to` after `from`): the window. It must lie inside one shot of the cut, the host (the shot in force at `from`) |
| `shot` | the shot inside the bubble (required): any `[[shot]]`, one in the cut or `plate = true`, but not the host |
| `shape` | `"thought"` (default), `"split"` or `"iris"` (below) |
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

[[transition]]               # the outgoing frame cut along a line, the halves sliding apart onto the next shot
at = 8.42
kind = "slice"
dur = 0.3
center = "figure"            # through the outgoing flat shot's figure; or [x, y] frame fractions (default the middle)
angle = -20                  # degrees off vertical, clockwise positive
edge = { color = "#ffffff", width = 14 }   # a line along each half's cut side

[[transition]]               # a record spin: up to speed into the cut, the next shot spinning down to rest after it
at = 10.20
kind = "spin"
dur = 0.3                    # the spin up, before the cut
after = 0.4                  # the spin down, after it, in the incoming shot
turns = 1.5
blur = 1.0                   # the shutter in frames
scale = 0.9                  # at the cut
back = "mirror"              # or a colour

[[transition]]               # a picture-book page turn onto the next shot, its paper back showing
at = 14.72
kind = "page"
dur = 0.6
dir = "left"                 # lifted at the right, turned toward the left; or "right", "up", "down"
angle = 20                   # the bottom right corner first (-20: the top right)
radius = 0.12                # the roll, a share of the frame's short side
back = "surface"             # the paper (default the palette's lightest)
through = 0.18               # the picture showing through it from behind
shadow = 0.45

[[transition]]               # the loop's wrap: at the clip's end the last frames turn over onto frame 0, held
at = 24.25                   # [project] duration
kind = "page"

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

`shape = "split"`: a split screen. Over the window the frame is divided across its width or its height: the host on one side
and the inner half on `side`, each its own picture at its own scale, cropped about its `focus` (that point of the picture kept
mid-panel as far as the picture still covers the panel), never squeezed. The inner half slides in from its side over `in`
seconds (eased out: it settles into place) and back out over `out` seconds, gone on `to` (0: a cut); on each `swap` the halves
trade sides, the row of panels running on the way the inner half came in, like a carousel (the host leaves on one side as it
comes back on the other), as long as the slide in, landing on the swap's frame (with `in = 0`, a cut on it). A half is its
picture moved by whole pixels, so at rest it is as sharp as the frame. It needs the inner shot's plates on its frames, or
nothing for a video.

| Key | Meaning |
|---|---|
| `from`, `to` | clip seconds (required, `to` after `from`): the window, inside one shot of the cut, the host |
| `shot` | the inner half's shot: any `[[shot]]`, one in the cut or `plate = true`, but not the host. Exactly one of `shot` and `clip` |
| `clip` | a video in the inner half instead: a file (project-relative, absolute or `~/...`) or the slug of an asset whose path is a video (`mk ref footage` registers Pexels clips so), read through `mkmmd.footage` as the footage frame effect's; or `{src, in, rate, loop, fit, mirror, back}`: `src` that video, `in` seconds into it on the window's first frame (default 0), `rate` its speed (default 1; 0 a still), `loop` (default false: past its end the last frame holds; a window that starts past its end is an error), `fit` `"cover"` (default: it fills the half, cropped about `focus.inner`) or `"contain"` (whole, on `back`, default `base`), `mirror` (flipped left to right, default false) |
| `side` | where the inner half sits: `"left"`, `"right"`, `"top"` or `"bottom"` (default `"right"`, `"bottom"` in an output taller than wide) |
| `share` | the inner half's share of the frame across the split, in (0, 1) (default 0.5) |
| `focus` | `{host = [x, y], inner = [x, y]}`: the point of each picture kept mid-panel, frame fractions (default the middle of each); a left / right split crops across, a top / bottom one up and down |
| `divider` | `{color, width}`: a line along each seam between the halves: `color` (default `text`), `width` in px at 1080 on the short side (default 6). No line without the key |
| `in`, `out` | seconds the inner half slides in from its side from `from`, and back out by `to` (default 0.3 each; 0 a cut) |
| `swap` | clip seconds, one or a list: the frames by which the halves have traded sides. Each swap slides as long as `in`, after the slide in has landed, and lands no later than the slide out begins |
| `aspect.<output>` | `{side, share, focus}`: that output's own (`focus` entry by entry); nothing else may be overridden |

```toml
[[insert]]                   # her head beside the wide shot: in from the right, trading sides on the hit, out to the left
from = 12.60
to = 15.20
shot = "head"                # or clip = "pexels_123" / { src = "pexels_123", in = 3.5, rate = 1, loop = false, fit = "cover" }
shape = "split"
side = "right"               # an output taller than wide takes "bottom" when no side is given
share = 0.5
focus = { host = [0.5, 0.5], inner = [0.5, 0.4] }
divider = { color = "#ffffff", width = 10 }
in = 0.3                     # seconds it slides in from its side (0: a cut); `out` the same, back out
out = 0.3
swap = [13.90]               # the halves have traded sides by then
aspect.9x16 = { side = "bottom", focus = { host = [0.5, 0.42] } }
```

`shape = "iris"`: a round window opening from a point of the scene onto another shot or a video, seen as through a lens at the
end of a tube: no bubble, no trail. Its centre is the `anchor` projected through the host shot's camera on every window frame (the
`point` layer `i<j>`, as a thought's). From `from` it opens from that point to its rest size over `open.dur` seconds, eased, every
frame of the opening bigger (the first small, not 0), uncovering the inner picture, which holds still at the scale that just covers
the rest circle (the inner shot's whole frame, or the video fitted to the frame, scaled about its centre and centred on the circle).
`out = "close"` closes it back onto its point the same way, a small circle on the window's last frame, gone on `to`. `out =
"expand"` grows it over `expand.dur` seconds until its picture is the whole frame on the window's last frame: the centre moves to
the frame's middle, the radius grows geometrically to the one whose vignette has left the frame (the half diagonal / (1 -
`rim.width`), 2 % to spare), the picture geometrically to the frame's own size (never smaller than what covers the circle's part
inside the frame, and slid off the circle's centre just enough to cover it: no edge of the picture ever shows in the circle), and
the glint fades. The last frame is the inner picture itself, so the cut lands on it: with a `shot` that shot must start at `to`;
with a `clip` the shot after it should show the same video (a footage collage). A rest size in metres (`radius`) stays the size of
a disc in the scene at the anchor, so a camera pushing in makes the circle grow; `size` is its diameter as a share of the frame's
short side, alike in every output unless `aspect.<output>` gives one its own. The rim is the lens's: over the outer `width` of the
radius the picture darkens toward `color` in a smoothstep, `dark` at the edge; there its colour channels are pulled apart along
the radius (`fringe` px at 1080 on the short side at the edge, red outward, blue inward); and a glint, a thin crescent of the
palette's lightest colour about 70° long in that band at the upper left, tapering at both ends in a soft bloom, has a fainter,
shorter twin opposite. Inside the vignette's inner edge the picture is untouched, and outside the circle and its `edge` ring the
host frame is; the edge is antialiased over a pixel, and a circle under half a pixel draws nothing. It needs the inner shot's
plates and the anchor's point on its frames (a video: the point alone). `mkmmd/core/iris.py` times it and does the circle's maths,
`mkmmd/iris.py` draws it on the circle's box, the rim only where it runs (at most about 0.08 s a 1080p frame, late in an expand).

| Key | Meaning |
|---|---|
| `from`, `to` | clip seconds (required, `to` after `from`): the window, inside one shot of the cut, the host |
| `shot` | the shot inside: any `[[shot]]`, one in the cut or `plate = true`, but not the host. Exactly one of `shot` and `clip` |
| `clip` | a video inside instead, in the split's forms (a file or an asset slug, or `{src, in, rate, loop, fit, mirror, back}`), fitted to the frame |
| `anchor` | a `mk q` expression (required), projected through the host shot's camera on every window frame: the circle's centre (`'obj("toy_eyehole").loc'`) |
| `radius` | metres: the rest radius at the anchor's depth (the point layer's `m` turns it into pixels), so the circle stays the size of a disc in the scene |
| `size` | the rest diameter as a share of the frame's short side, in (0, 1] (default 0.3 when there is no `radius`); `radius` and `size` together are an error |
| `open` | `{dur, ease}`: it opens from the anchor to its rest radius over `dur` seconds (default 0.25; 0: open on the first frame), eased `ease` (default `out`) |
| `out` | `"close"` (default): back onto its point over `open.dur`, gone on `to`; `"expand"`: grown until its picture is the whole frame on the window's last frame |
| `expand` | `{dur, ease}`, only with `out = "expand"`: `dur` seconds (default 0.6), `ease` (default `in`) |
| `rim` | `{width, dark, color, glint, fringe}`, the inner rim: a vignette over the outer `width` of the radius (default 0.3, in (0, 1)) darkening toward `color` (default the palette's darkest: Dawn's `text`) by `dark` at the edge (0..1, default 0.8); `glint` 0..1 (default 0.85); `fringe` px at 1080 on the short side (default 2; 0 none). `rim = false`: none |
| `edge` | `{color, width}`: a ring round the outside of the circle, `color` (default `text`), `width` in px at 1080 on the short side (default 4). No ring without the key |
| `ring` | a ring portal round the window: `"vinyl"`, `"plate"` or `"lens"`, or `{kind, size, ...}` with the kind's keys (below); none by default |
| `aspect.<output>` | `{radius}` or `{size}`: that output's own rest size (either replaces both); nothing else may be overridden |

```toml
[[insert]]                   # the toy's eyehole opens onto the next shot and grows into it: the cut lands inside
from = 3.10
to = 4.20
shot = "inside"              # or clip = "pexels_123" / { src = "pexels_123", in = 2.0 }
shape = "iris"
anchor = 'obj("toy_eyehole").loc'   # projected through the host shot's camera on every frame
radius = 0.004               # metres: the eyehole's own size, growing as the camera pushes in (or size = 0.3)
open = { dur = 0.25, ease = "out" }
out = "expand"               # "close": back onto the eyehole, gone on `to`
expand = { dur = 0.6, ease = "in" }
rim = { width = 0.3, dark = 0.8, glint = 0.85, fringe = 2.0 }   # color: the palette's darkest; rim = false for none
aspect.9x16 = { size = 0.4 }
```

**Ring portals.** `ring` puts a thing round the window that opens, holds and closes or grows with it: every part of it is
sized in window radii (the window's radius is 1, `size` the ring's outer radius), so it scales with the window from the first
frame of the opening to the last of an expand, where it has left the frame with the vignette and the cut lands on the inner
picture as before. It is laid over the host frame before the window (`mkmmd/portal.py`, in passes each worked out over the
annulus it covers, antialiased over a pixel; `mkmmd/core/portal.py` checks and lays it out): nothing past its reach (its size
and its shadow, glow or sprites) changes, and the window is the iris's own. A ring sets the rim's defaults (`rim` keys given
are laid over them; `rim = false` still turns it off): `vinyl` a dark lip, the hole's wall (width 0.12, dark 0.6, no glint or
fringe), `plate` a soft shade in the well (0.16, 0.3, none), `lens` a lens's rim (0.25, 0.6, glint 0.85, fringe 3).

- `vinyl`: a record whose centre hole is the window. Round it the label, an original design (its ground, an arch of the band
  colour over the top with a dot under it, two lines of small print, a ring at its rim), the dead wax, the grooves (their
  unevenness in fine rings, smooth gaps between five tracks, faded where finer than a pixel) and the lead-in edge catching the
  light; label and grooves turn clockwise at `rpm` (33 1/3: 200 degrees a second). The sheen, two bright wedges toward and away
  from the light at the upper left, stands in the light, swinging a few degrees and shimmering as the grooves turn through it.
  A soft shadow falls to the lower right.
- `plate`: a dinner plate whose well is the window: the slope up from the well, the broad glazed rim with two thin lines of
  `band` colour, the rounded lip, lit from the upper left (a crisp highlight on the lip, a soft sheen, a broad glaze highlight
  across the rim at the upper left and a fainter one opposite), a soft shadow. `sprites` stand round the rim, `count` of them
  from the top, dealt shuffled by `seed`, every one before any comes again, each on a soft shadow, `upright` or `radial`,
  travelling `turn` degrees a second; `eyes` and `mouth` give them googly eyes and a smile ([Googly eyes](#googly-eyes)),
  swinging as they ride the rim and grow with the window (an iris has no hits of its own: the eyes' own `times` or `lane`
  kick them). A plate's sprites count for `mk assets credits`; one that finds nothing is an error before anything is encoded.
- `lens`: concentric chrome rings round the window, `count` of them, the first round its edge, their widths, gaps and breaks
  (some rings broken into arcs) seeded by `seed`. Each is a glass tube showing the window's picture refracted (across the tube
  turned over and magnified, swirled with the ring's turn, red and blue pulled `chroma` px apart toward its edges) under a
  chrome reflection of a two-tone studio (the palette's lightest above a bright horizon line, its darkest below) and a
  highlight, darker toward its edges; each turns its own way, `turn` degrees a second, every other one the other way and each
  35 % faster than the one inside it; a glow of `tint` lies round and between them.

| Key | Meaning |
|---|---|
| `kind`, `size` | the ring (required in a table); its outer radius in window radii, 1.05 .. 8 (default 2.8 vinyl, 1.46 plate, 2.1 lens) |
| `rpm`, `color`, `label`, `sheen` | vinyl: turns a minute (33.333; negative the other way, -300 .. 300); the vinyl (default the palette's darkest at 22 % of its value: near black); `[ground, band, print]`, one to three colours (default `["gold", "love", "text"]`); the sheen's strength 0..1 (0.8) |
| `color`, `band` | plate: the glaze (default the palette's lightest) and its two lines (`pine`; `false` none) |
| `sprites`, `count`, `scale`, `orient`, `turn`, `seed`, `eyes`, `mouth` | plate: the sprites, a mandala ring's forms (registry slugs, files, globs or `{tag = "veg_green"}`, one or a list); 8 of them (1 .. 64); each `scale` window radii at its longer side (0.3); `"upright"` (default) or `"radial"`; degrees a second they travel round, counter-clockwise (0); the deal (1); googly eyes and a smile. The keys after `sprites` need it |
| `count`, `chroma`, `turn`, `tint`, `seed` | lens: rings (4, 1 .. 8); the colour split, px at 1080 on the short side (5, 0 .. 64); degrees a second (30); the glow's and the chrome's cast (`iris`); the rings' layout (1) |

```toml
[[insert]]                   # a record at her chest opens onto the next shot and grows into it
from = 0.20
to = 1.60
shot = "close"
shape = "iris"
anchor = 'bone("upper_body").center'
size = 0.3
out = "expand"
ring = "vinyl"               # or { kind = "vinyl", rpm = 45, label = ["pine", "gold", "text"] }

[[insert]]                   # a dinner plate with googly-eyed veggies round its rim
from = 1.80
to = 3.20
shot = "wide"
shape = "iris"
anchor = 'obj("table_plate").loc'
radius = 0.12
out = "expand"
ring = { kind = "plate", band = "love", sprites = { tag = "veg_green" }, count = 6, eyes = true, turn = 20 }

[[insert]]                   # chrome lens rings, the TikTok portal
from = 3.40
to = 4.80
shot = "close"
shape = "iris"
anchor = '(0.0, -0.15, 1.05)'
out = "expand"
ring = { kind = "lens", count = 5, chroma = 8, turn = 40, tint = "foam" }
```

The plan refuses what cannot be drawn, with a message that names the effect: a transition `at` where no shot (or several) starts;
a window that starts before the outgoing shot does (`shorten dur`), or a spin that runs past the incoming shot (`shorten after`);
an expand or collapse whose figure shot is not a silhouette in every output, or a slice through the figure whose outgoing shot is
not flat in every output; an insert that crosses a cut, whose `shot` is the host or does not exist, whose `out = "expand"` shot is
not the one that starts at `to`, or whose window is too short for its pop-in and exit (each at least 2 frames); a split with both
or neither of `shot` and `clip`, too few frames for its slides in and out, or a swap that slides before the inner half is in,
overlaps the swap before it or lands after the slide out has begun; an iris with both or neither of `shot` and `clip`, both
`radius` and `size`, or too few frames for its opening and its close or expand (an opening of `open.dur` above 0 and an expand
take at least 2 frames each), or a `ring` of no known kind, a key of another kind, a plate's sprite keys without `sprites`; a
page turn at the clip's end without `[project] duration`; and any two windows, transitions
and inserts together, that overlap in frames. `mk build` also refuses a window that needs frames from a shot before `[scene] start`
(`lower [scene] start`), and keys the cameras of the shots that lend frames: after editing an effect run `mk build` again (`mk render`
says `run mk build again` when a camera is not keyed over the frames an effect needs). `mk post` reports `matte_scale` per expand or
collapse (`scale`, whether it was `raised`, and `needed`, the scale that fills the frame): a number far above `needed` spends the
window with the new shot already full. A figure that is not in the frame at the covering frame, or a `center` that cannot lie inside
it, is an error that says so (`center = "subject"` always can), and so is a slice's figure missing on its first window frame, or a
split's or an iris's video that cannot be read or whose window starts past its end (`mk post` checks them before encoding anything).

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
subject=False)`). `mk post` lays the effects over the cut's frames before the screen type and the grade (the loop's wrap turns the
finished frames, above); a missing layer is an error unless `--allow-gaps` (the plain cut shows there). `mk look` previews a frame
inside a window composited, from layers it draws into its own folder (a spin turns the frame it drew and needs none; a frame of the
wrap turns over frame 0, which it draws beside it). `--no-transitions` leaves the effects out of `mk render`, `mk post`
and `mk look` (`--no-styles` leaves them out of `mk render` and `mk look` too: a figure needs its silhouette look).

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
inserts and glitches and before the lens streaks, the screen type, the loop's wrap and the grade, and `mk look` previews them the
same way (`mkmmd/core/framefx.py` checks and times them, `mkmmd/framefx.py`, numpy and OpenCV, does the picture work). Unlike the
cut effects they may overlap one another and any window, and on a frame inside several they are applied in the order of the table.
They read the rendered frames as they are, without their effects: an echo of a frame inside a glitch echoes the unglitched frame.
With a breathing palette ([Lights and the look](#lights-and-the-look)) their colours follow the breath: each frame draws an
entry in its palette colours mixed by that frame's mix, in `mk post` and `mk look` alike ([Post](#post)).

| Key | Meaning |
|---|---|
| `kind` | `lens`, `echo`, `slitscan`, `push`, `scope`, `zoom`, `shock`, `slab`, `tint`, `halftone`, `split` or `mosh`, or a collage: `grid`, `orbit`, `mirror`, `panels`, `stack` or `footage` ([Footage, stickers and figure moves](#footage-stickers-and-figure-moves)), or a video sticker `clip` ([Video stickers](#video-stickers-clip)), or a picture effect: `pixelsort`, `film`, `camcorder`, `doodle` or `booth` ([Picture effects](#picture-effects-pixel-sort-film-camcorder-doodles-photo-booth)), or `mandala` or `trails` ([Mandalas and trails](#mandalas-and-trails)), or a colour field: `liquid`, `spiral` or `flood` ([Colour fields](#colour-fields-liquid-light-op-art-floods)), or a remap: `kaleido`, `tunnel` or `melt` ([Remaps](#remaps-kaleidoscope-tunnel-melt)), or the paper world's `activity` or `label` ([Activity book and labels](#activity-book-and-labels)), or the toy paint app `paintapp` ([Toy paint app](#toy-paint-app)), or an era look: `digicam`, `chrome`, `gradmap`, `invert`, `testcard` or `cards` ([Era looks](#era-looks-digicam-chrome-gradmap-invert-test-card-cards)), or `lasers` ([Lasers](#lasers)), or `prints` ([Prints](#prints-photographs-laid-over-the-picture)), or `key` ([Keys pushed too far](#keys-pushed-too-far)), or `shapes` ([Shapes](#shapes)), or the haze: `mist`, `fog` or `bokeh` ([Haze](#haze-mist-fog-bokeh)), or `mitosis` ([Cell division](#cell-division-mitosis)) (required) |
| `from`, `to` | clip seconds (required, `to` after `from`): the span, `from` .. `to` - 1 in frames, inside the clip |
| `aspect` | any entry: `[fx.aspect.<output>]`, its own keys for one output ([Each output its own: aspect](#each-output-its-own-aspect)) |
| `strength` | a number, or `[[t, v], ...]` keys over clip seconds, linear between them and held outside (default 1): the lens's mass (its Einstein radius goes as the square root), an echo's opacity, a slitscan's depth, a scope's opacity, a zoom's and a shock's size, a slab's opacity, a tint's, a halftone's, a mosh's and a collage's mix, how far a split tears, the lasers' light, the prints' mix. A push has none |
| `center` | lens and push (required), zoom, shock, orbit, mirror and stack (default the middle): `[x, y]` fractions of the frame (0, 0 the top left), or `{point = "<expression>"}`, a point of the scene ([Expressions](#expressions)) that `mk render` projects through the camera of the shot in the cut on every frame of the span (`point/x<i>/<frame>.json`) |
| `radius`, `radius_m` | lens and push: the lens's Einstein radius, or the push's disc at the start; `radius` a share of the frame height (default 0.1 for a lens, 0.02 for a push), `radius_m` metres at the depth of a scene point (a hole of a known size, a pupil), one of the two |
| `shadow` | lens: the share of the Einstein radius painted black about the centre (default 0: the scene draws its own). The lens bends the whole picture, what passes in front of its centre too (a figure there wraps into a ring, as if it were behind): key `strength` down while something passes in front that should stay whole |
| `count`, `every`, `fade`, `blend`, `tint` | echo: `count` copies (4) of the frames `every` (3), 2 x `every` ... back, the first at opacity `fade` (0.55) and each next at `fade` times the one before; `blend` `lighten` (default: keeps the brighter, for a light figure on a dark ground) or `darken`; `tint` a colour (slot, `#hex`, ...) for the copies where they differ from the ground (default their own colours). A copy of the ground changes nothing, so only what moves leaves echoes |
| `depth`, `now` | slitscan: each column (`now` `left` / `right`, default `right`) or row (`top` / `bottom`) shows the frame of its own age, from the present at the `now` edge to `depth` frames back (default 24) at the far one; before the clip's first frame the oldest there is stands in |
| `inside`, `play`, `ease` | push: the picture zooms into the disc about `center`, which shows the clip's frame at `inside` (clip seconds, default 0: its first frame), a still or, with `play = true`, playing on from there; the disc's centre glides to the frame's middle as it grows and at the span's last frame it covers the frame with the inside frame at its own size: the seam of a loop back to the start. `ease` how the zoom runs: `log` (default, a steady push), `smooth` or `linear` |
| `stem`, `timeline` | scope: the stem drawn, `drums`, `bass` (default), `other`, `vocals` or `mix`, from the stems cached when `timeline` (default `audio/timeline.json`) was analysed: `mk timeline analyze` makes them, and a missing cache is an error before anything is encoded |
| `at`, `size`, `color`, `width`, `glow` | scope: the trace's centre (`[x, y]` frame fractions, default the middle), its radius as a share of the frame height (0.3), colour (`text`), line width in px at 1080 on the short side (2) and glow (0..1, 0.6); it is added to the picture |
| `window`, `lag`, `gain` | scope: the seconds of sound drawn on a frame (0.04, the latest ending on the frame), the delay of the y axis behind the x axis in seconds (0.0025: a note draws a loop whose shape is its harmonics; 0 draws the waveform against time instead) and a gain on the stem's 99.5th percentile over the span, which fills the radius (1) |
| `times`, `lane`, `min_db`, `every` | zoom and shock: what they fire on, one of `times` (clip seconds) or `lane`, a lane of the `[audio] hits` file (`mk timeline hits`), its hits inside the span at or above `min_db` (0 the lane's loudest), every `every`-th of them (`mkmmd/core/lanes.py`; a lane without the file is an error that says to write it). A collage steps on them (and without either never steps) |
| `amount`, `attack`, `decay` | zoom: a punch on every hit, the picture jumping in by `amount` (0.06: 6 %) about `center` over `attack` s (0.02, eased out) and coming back as exp(-t / `decay`) (0.16 s); the strongest hit counts, they do not stack |
| `speed`, `width`, `amount`, `life`, `color`, `glow` | shock: a ring runs out from `center` on every hit at `speed` frame heights a second (1.6), its front `width` (0.035 frame heights) wide pushing the picture out ahead of it and pulling it in behind by up to `amount` (0.012 frame heights), lit in `color` (`text`) at `glow` (0.6); it fades as (1 - t / `life`)² and is gone after `life` (0.55 s) |
| `at`, `size`, `slant`, `angle`, `color`, `alpha`, `dir`, `in`, `out`, `shutter` | slab: a bar of `color` (`love`) at `alpha` (1) resting with its centre at `at` (frame fractions), `size` `[w, h]` frame heights (`[1.2, 0.18]`), its sides leaning `slant` degrees (18: to the right going up, as italics) and the whole turned `angle` degrees counter-clockwise (-8). It travels `dir` (`right`, `left`, `up`, `down`): in from the other side over `in` s (0.1, slamming: an ease-out) from fully off the frame, rests, and leaves the way it travels over the last `out` s (0.12, flung: an ease-in), drawn at 8 places over `shutter` frames (1) of its motion so it streaks. Persona's panels; stack a few, a thin one behind a thick one |
| `color`, `dark`, `light` | tint: the picture through a gradient by its brightness, `dark` (black) at black, `color` (`#46E07A`, the Dark Hour's green) at mid grey, `light` (white) at white, mixed in by `strength`: key it up over a stop and back down |
| `cell`, `angle`, `level`, `ink`, `lift`, `offset` | halftone: comic print. Where the luma is under `level` (0.5) the picture is lifted toward white by `lift` (0.35) x its darkness and screened with round dots of `ink` (black) on a grid `cell` px at 1080 apart (9) turned `angle` degrees (45), a dot's area growing with the darkness until the dots merge into solid ink at black; the lights stay clean. Then the red plate is printed `offset` `[x, y]` px at 1080 off (`[3, 2]`) and the blue as far the other way: a press out of register |
| `bands`, `shift`, `share`, `rgb`, `hold`, `seed` | split: a broken digital signal. The frame is cut into `bands` (14) bands of random heights, `share` (0.35) of them torn sideways by up to `shift` (0.08 of the frame width; what leaves one side comes in at the other), the red and blue planes pulled `rgb` px at 1080 apart (10) in a torn band and a third of that everywhere else. A tear holds `hold` seconds (0.05), then the signal breaks another way; the same `seed` (1) breaks the same frames the same way |
| `block` | mosh: a datamosh. The cut's frame before the span is held and carried along the cut's own motion from each frame to the next since the span's first frame (Farneback optical flow on the frames at a quarter size, averaged over `block` px at 1080 squares, 16, so it moves in chunks as a codec's motion vectors do), and mixed over the real frame by `strength`. Started on a cut it is the classic one: the outgoing picture smeared by the incoming shot's motion, as when the key frame of a cut is lost. Frames asked out of order are worked out again from the span's start |
| `flow` | mosh: a made-up motion instead of the cut's own, so a still moshes too (a clip of one frame has no frames before it): `{kind, amount, steps, jitter, stuck, seed}` and the kind's own key. The span's first frame as it reached the mosh (the effects before it in the table on it; `mk look` renders it beside a later frame asked for, as a film's held frame) is carried `steps` (12) times along the flow on that frame and as many times more on each frame after: a P-frame played again and again. The frame is cut into `block` squares and each moves whole, by whole pixels, so the picture moves in sharp chunks: where a moving block pulls from one that stays, that one's edge is smeared out in a trail, and colours are dragged across the picture. `kind`: `drift` (every block the same way, `angle` degrees counter-clockwise from the right, 0), `zoom` (the picture pushed out from `center`, `[x, y]` frame fractions, default the middle, as a camera pushing in does; drawn in toward it with a negative `amount`), `swirl` (round `center`, counter-clockwise on screen; clockwise with a negative `amount`) or `noise` (a smooth random field whose swirls are `size` frame heights across, 0.3); `amount` (6) px at 1080 the farthest block moves per step (a zoom's and a swirl's at the frame's farthest corner from `center`); each block's move is off by up to `jitter` (0.25) x `amount` at random, and a share `stuck` (0.3) of the blocks, in patches, stays where it was (the frame itself shows there). The same `seed` (1) moves the same blocks: the same picture again; a larger `stuck` holds more of the same patches, and a frame of another size (`mk look`'s) is cut into the same blocks. Trails run about `amount` x `steps` px at 1080: a small amount over many steps stretches them smooth, a large one over a few stamps them in repeats |
| `mask`, `objects` | mosh: `figure` moshes only inside the coverage of `objects` (the figure layer `mk render` draws beside the frames, as for a collage or a pixel sort), `ground` only outside it; both or neither. The held picture is mixed in by the coverage: the outline stays antialiased as the figure's own, and a figure a ground mask keeps is left exactly as it was (across a `rect`'s feathered edge too). With a `flow`, a block whose source holds anything of the rest on the span's first frame (her outline included) stays put, so her colours never smear into the ground, nor the ground's into her |
| `objects` | a collage (required): its figure, object patterns as a look's `hide` takes (`"Rin_*"`, `"@collection"`, `"prop:key"`; one or a list). `mk render` draws their coverage alone through the camera of the shot in the cut beside each frame of the span (`figure/<key>/<frame>.png`, the key named from the patterns, so collages of the same figure share it) and the collage cuts the figure out of the picture with it: the figure as the frame shows it, look and lights and all. With `shot` the figure comes from that shot's own render instead ([Footage, stickers and figure moves](#footage-stickers-and-figure-moves)) |
| `back`, `fill` | a collage: the flat colour behind (default `base`), a pattern painted there instead (`{pattern = "spiral", colors = [...], ...}`: [Colour fields](#colour-fields-liquid-light-op-art-floods)) or `"picture"`, the frame as it reached the effect (the effects before it in the table on it: her world behind the collage, as behind a mandala's rings, and the figure in it wherever the frame shows her; a grid's cell shows it shrunk and laid as the cell's figure is, reflected past its edges where the cell is wider or taller than the frame; a mirror folds it with the figure; a contained footage clip sits on it), and how the figure is drawn, `"picture"` (default: as the frame shows it), a flat colour (a silhouette) or `"silkscreen"` (printed in flat inks: [Dividing grids and silkscreen prints](#dividing-grids-and-silkscreen-prints)). Either may be a list, read at the step: the number of hits so far (a hit on the span's first frame is where it starts) plus the number of the cell, copy, panel or region drawn, round and round. So `back = ["love", "base"]` is a chessboard in a grid that flips on every hit, and one colour on every hit of an orbit |
| `edge` | a collage: `{color, width}`, an outline round every figure and copy, `width` px at 1080 on the short side (default 3); default none. With `shadow` (a drop shadow) a thick white one makes a sticker; `keys` and `pivot` move the figure ([Footage, stickers and figure moves](#footage-stickers-and-figure-moves)) |
| `cells`, `piece`, `seed` | grid: the frame shrunk into cells, `n` x `n` (3) or `[columns, rows]`, or a list of them read at the step (`divide` makes each step cell division: [Dividing grids and silkscreen prints](#dividing-grids-and-silkscreen-prints)), the figure in each on its colour; a `piece` colour draws one cell's figure flat in it, the cell hopping a knight's move on every hit (from the middle of the top row; `seed` 1 picks among the moves, never straight back while another is open) |
| `count`, `radius`, `tilt`, `scale`, `size`, `turn`, `spin` | orbit: the figure moved to `center`, `scale` times its size (1), and `count` (6) copies of it, `size` times as large again (0.6), on a ring of `radius` (0.36 of the frame's short side) round it tilted `tilt` degrees from face-on (60): an ellipse whose near half is drawn over the figure and a little larger, its far half under it and smaller. The ring turns `spin` degrees a second (0) and `turn` more on every hit (30); copy j is drawn `fill` entry step + j |
| `axes`, `place` | mirror: the frame folded about the vertical line through `center` (`axes` `"x"`, default), the horizontal one (`"y"`) or both (`"xy"`; a list is read at the step): the side or quarter with the figure's centroid is shown as it is, the others are its reflections, region j (0 that side, 1 across the vertical line, 2 across the horizontal, 3 both) drawn in its own `back` and `fill` entry step + j. `place` `[x, y]` (frame fractions) moves the figure's centroid there first: beside the fold, two of her face each other |
| `count`, `slant`, `size`, `delay` | panels: the frame cut into `count` panels (3, 2 .. 12) across its long side, columns of a wide or square frame and rows of a tall one, by lines leaning `slant` degrees (12: to the right going up, as italics), panel j its own `back` and `fill` entry step + j with the figure moved onto its middle line, `size` times its size (1; and `count` times smaller in a row, so a figure framed for the whole frame fits one). With a `delay` (frames, default 0) panel j shows the figure j x `delay` frames before |
| `count`, `step`, `scale`, `bands`, `delay`, `mirror` | stack: a banded echo. The figure in front (`fill`, on `back`) and `count` (5) copies of it behind, copy j `step` `[dx, dy]` further per copy (`[-0.02, -0.006]` of the frame's short side), `scale` ** j as large about `center` (1; under 1 the stack shrinks toward it, a vanishing point) and drawn `bands` entry step + j - 1 (`["text", "base"]`, colours, `"picture"`, `"silkscreen"` or tables of sprites); with a `delay` (frames, default 0; or a list, each copy's own) copy j is the figure j x `delay` frames before, a trail of the poses before. `mirror = true` (default false) draws the stack again over all that, reflected left for right about the vertical line through `center` (put on the nearest half pixel): its figure the same pixels flipped, its copies stepping the mirrored way (both echoes trail outward) and its edges and shadows flipped with it (a shadow falling down and right falls down and left on the reflection). Only the stack is reflected, never the frame: on `back = "picture"` with `center` on her, a figure on one side of her is also on the other. The reflection is drawn over whatever is where it lands, and the figure's layer is drawn alone and stops at the frame's edge: what stands in front of the figure comes along into its reflection, and what the frame cuts off it is missing there too. Keep the figure clear of both, or take it from a `shot` of its own, set beside her with `at` and `height`. Bands of sprites, copies in eras' `looks`, a tunnel that `recede`s forever, busy clones that `peel` off and a `squash` on the hits: [Rainbow stacks](#rainbow-stacks-sprite-bands-era-looks-tunnels-busy-clones-squash) |

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

[[fx]]                        # a still: the street dragged in blocks out from the middle, on the left edge, her left as she is
kind = "mosh"
from = 0.0
to = 0.04
flow = { kind = "zoom", amount = 8, steps = 14, seed = 3 }
mask = "ground"
objects = "Rin_*"
rect = [0.0, 0.0, 0.3, 1.0]
feather = 40

[[fx]]                        # a chessboard of her on the snare roll: it flips on every hit, a white knight hops across it
kind = "grid"
from = 28.83
to = 29.51
objects = "Reisen*"
lane = "snare"
back = ["love", "base"]
cells = [2, 3, 4]
piece = "text"

[[fx]]                        # her guitarist, alone in her own void, set beside her and mirrored, both echoing outward
kind = "stack"
from = 29.72
to = 35.0
objects = ["Rin_guitar*", "guitar*"]
shot = "guitar_plate"         # a [[shot]] with plate = true framing her whole body, by her own lights
at = [0.2, 0.94]              # her feet (pivot "base") there
height = 0.55                 # her height: 0.55 of the frame's
center = { point = "bone('upper_body2', 'Rin_arm').head" }   # projected through the shot in the cut
back = "picture"
mirror = true
count = 5
step = [-0.03, -0.01]
delay = 2
bands = ["love", "gold", "foam", "iris", "rose"]
edge = { color = "#ffffff", width = 14 }
shadow = { color = "#575279", alpha = 0.35 }

[fx.aspect.9x16]              # no room beside her in the tall frame: smaller, by her legs
at = [0.2, 0.97]
height = 0.3
```

`mk render` draws the point layers of a scene centre and the figure layers of a collage with the other layers; `mk post` reads
the frames an echo, a slitscan, a push, a mosh or a collage's `delay` needs from the cut it rendered, and `mk look`, which draws
only the frames asked for, renders those beside them as plates of the shot in the cut there, with a collage's figure on them
(`TR.fx_frames`). The plan, `mk build`'s shots report (`_cut_effects.fx`) and `mk post`'s report (`effects.fx`) list them. A
collage cuts its figure out of the picture as the effects before it in the table left it, but with the figure's coverage as
it was rendered: put it before an effect that moves the picture (a zoom, a lens, a shock), which then moves the collage.

#### Where an effect applies: rect

Any `[[fx]]` entry may be limited to a part of the frame: `rect = [x0, y0, x1, y1]` (frame fractions, 0, 0 the top left;
`false`, the default, the whole frame) and `feather` (px at 1080 on the short side: a smooth ramp that wide across the rect's
edges, not along the frame's own; default 0, a hard antialiased edge). Outside the rect the frame is left as the effects before
it made it. The picture effects `film`, `camcorder`, `pixelsort` and `booth` and the era looks `digicam` and `testcard` take
the rect as their frame: a film's vignette and weave, a camcorder's or a digicam's stamp corner, a booth's strip and its
layout, a test card's layout are the rect's, and a pixel sort's runs stop at its edge; the
others draw on the whole frame and are cut to the rect (a mosh drags in what lies outside it). A pixel an effect leaves as it
was stays so across a feathered edge: a figure a masked mosh keeps. Give each output its own through `[fx.aspect.<output>]`:
a split screen left / right in 16x9 and 1x1 is top / bottom in 9x16, and the Screen Test covers her half only.

```toml
[[fx]]                        # the home-video look on the left half, on the top half of the vertical output
kind = "camcorder"
from = 11.0
to = 12.0
date = "OCT. 23 1999"
rect = [0.0, 0.0, 0.5, 1.0]

[fx.aspect.9x16]
rect = [0.0, 0.0, 1.0, 0.5]
```

#### Picture effects: pixel sort, film, camcorder, doodles, photo booth

`mkmmd/core/picturefx.py` checks, times and lays them out (numpy, tested), `mkmmd/picturefx.py` draws them (numpy, OpenCV and
Pillow). Their randomness is seeded by `seed` and the entry: the same frames come out the same in `mk look` and `mk post`. A
film's held frame and a booth's capture are the picture as it reached the effect on that frame (the effects before it in the
table on it): `mk post` keeps it as it goes through the frames; `mk look` renders that frame's plate beside the frames asked for
(with the points and figures the effects before it read there) and applies those effects to it again.

| Key | Meaning |
|---|---|
| `by`, `lo`, `hi`, `axis`, `reverse` | pixelsort: every run of pixels whose `by` (`luma`, default, `hue` round the wheel from red, or `sat`, HSV saturation; 0..1) lies in `lo` .. `hi` (0.25 .. 0.8), unbroken along a row (`axis` `x`, default) or a column (`y`), is sorted by it, rising to the right (down), falling with `reverse`. All runs at once in one numpy sort: a 1080 x 1920 frame takes about 0.1 s |
| `mask`, `objects` | pixelsort: `figure` sorts only inside the coverage of `objects` (a collage's figure layer, drawn by `mk render` beside the frames), `ground` only outside it; both or neither |
| `times`, `lane`, `min_db`, `every`, `decay` | pixelsort: without them it is steady at `strength` (a mix over the picture); with them it fires: full on the frame of each hit and dying as exp(-age / `decay`) (0.12 s), the strongest hit counting, times `strength` |
| `era` | film and camcorder: a preset of the other keys for an era of pop, the keys given beside it winning (`mkmmd.core.eralook.ERAS`): a film's `"1967"` is a Technicolor print (24 fps, `dye` 1, `saturation` 1.35, `contrast` 1.3, `halation` 0.35, fine grain, a little weave, dust and a scratch); a camcorder's `"1985"` a VHS tape (its stamp `PM 3:12` ticking over `JUL. 4 1985` with `PLAY`, chroma 9, softness 2, saturation 0.7, noise 0.06, a tracking band and the head-switching tear, `switching` 0.7). A number works (`era = 1967`); an unknown era is an error that lists the known ones |
| `mono`, `contrast` | film: black and white (`true`) or colour (default) through a print's S curve, a soft toe and shoulder about mid grey, steeper with `contrast` (1; 0 none) |
| `saturation`, `dye`, `halation` | film in colour: its saturation times `saturation` (1) before the print's curve; `dye` (0..2, 0) a three-strip dye print's purer primaries (each colour pushed off the other two, reds and oranges richest, the shadows a little cool and the lights warm); `halation` (0..2, 0) a red-orange glow round the lights (white in `mono`), light scattered back through the film |
| `fps` | film: the picture changes `fps` times a second (default every frame): each film frame shows the clip frame at its start, counted from the span's first frame, and the damage below changes with the film frame. 15 is a Warhol Screen Test, 10 hand-made film |
| `grain`, `flicker`, `weave`, `vignette` | film: grain, its amount in luma (0.06, strongest in the mid tones) or `{amount, size}` (clumps 1.5 px at 1080); exposure flicker 0..1 (0.05); gate weave, px at 1080 (1.5: a slow wander of the whole picture and a little shake); the corners darkened by `vignette` (0.3) |
| `dust`, `scratches`, `hair` | film: specks per film frame (3, or `{count, size}`, 3 px at 1080), white and dark, a fifth of them bent fibres; `scratches` thin white vertical lines (1, at most 32), each lasting 4 .. 12 film frames, drifting a little sideways, then another elsewhere; `hair` the chance per second (0) of a dark hair in from a side of the gate for 0.4 .. 1.2 s |
| `date`, `time`, `tick`, `indicator` | camcorder: the stamp's text, `time` over `date` (short texts, either may be left out); `tick = true` shows the time's seconds counting up from `time` at the span's start, in the style it was written (`8:41 PM`, `PM 8:41`, `20:41`); `indicator` `PLAY` (default, with a triangle), `REC` (a red dot blinking once a second) or `none`, at the top left (top right when the stamp is there) |
| `font`, `corner`, `color`, `size` | camcorder: a font slug of the asset registry or a font file (default: the registry's first font tagged `mono`; a missing font is an error before any frame is drawn); the stamp's corner, `bottom-right` (default), `bottom-left`, `top-left`, `top-right`; its colour (white); its capitals' height in short sides (0.05). It is drawn after the tape's damage, crisp, with a dark rim and shadow. The font is credited through `[credits] assets` |
| `tracking`, `chroma`, `scanlines`, `jitter`, `soft`, `saturation`, `noise`, `switching` | camcorder, the tape: a tracking band rolling up the frame, its amount (0.3) or `{amount, speed}` (0.25 frame heights a second: lines torn sideways, snow, washed out); colour bled sideways and to the right, px at 1080 (6); scanlines 0..1 (0.15); each line jittered sideways, px at 1080 (1); luma softness, px at 1080 (1.2); `saturation` (0.8); noise in horizontal streaks (0.04); `switching` (0..1, 0) the head-switching tear, the last 2 % of the lines pulled sideways, up to 30 px at 1080 at the foot |
| `marks` | doodle (required): `[{shape, at, size, color, angle, t}]`, purikura-pen marks: `shape` one of `star`, `heart`, `sparkle`, `x` (an X over an eye), `swirl`, `scribble`, `arrow`, `crown`, `cat_ears`, and the purikura stamps `bow` (a ribbon bow), `flower`, `twinkle` (a cluster of sparkles), `bubble` (a speech bubble), `note` (two quavers), `bunny_ears`, `whiskers`, `blush` (a hatched cheek); `at` `[x, y]` frame fractions or `{point = "<expression>"}`, projected through the camera of the shot in the cut on every frame as a `center` is (`point/x<i>m<j>/`); `size` frame heights (0.08) or, at a point, `size_m` metres at its depth (an eye's width); `color` (`love`); `angle` degrees counter-clockwise (0); `t` the clip second it pops in (the span's start), with a small spring (a 12 % overshoot, settled in 0.3 s) |
| `fps`, `width`, `outline`, `wobble` | doodle: the marks are drawn again `fps` times a second (8) by a hand that wobbles up to `wobble` px at 1080 (1.5), so they boil; a pen line `width` px at 1080 (9) in its colour over a white rim `outline` px wide (5) |
| `count`, `times`, `lane`, `min_db`, `every` | booth (one of `times` or `lane` required): on each of the first `count` (4) hits in the span the frame is captured; a white flash `flash` s long (0.15) over all of it; the capture drops into its cell over `drop` s (0.3), falling from 1.3 times its size with its shadow and a little bounce |
| `layout`, `ratio`, `focus`, `size`, `tilt` | booth: `[columns, rows]` of the photos (default per frame: a column in a tall frame, a row in a wide one, as square a block as fits in a square one: 1 x 4 in 9x16, 4 x 1 in 16x9, 2 x 2 in 1x1; `aspect.<output>` sets one per output); a photo's width over its height (0.8), cut from the frame about `focus` (`[0.5, 0.42]`); the strip as large as fits `size` (0.86) of the frame, turned `tilt` degrees (-4) |
| `back`, `paper`, `border`, `doodles`, `caption`, `color`, `font` | booth: the colour behind the strip (`rose`); the strip's paper (white) round each photo, `border` px at 1080 (16); `doodles` per photo (2), stickers drawn in the doodle shapes and the palette's accents near its edges, popping in after it lands; a short `caption` (no lyrics) in the strip's foot in `color` (`text`) and `font` (as a camcorder's) |
| `look` | booth: `plain` (default, the strip above) or `purikura` (`mkmmd/core/purikura.py`, drawn by `mkmmd/purikura.py`): each capture through a beauty filter in a pastel frame on dotted pastel paper, stickers stamped on it and a date or a word written over it by a gel pen. The look sets the booth keys the entry leaves out to its own: `paper` `rose`, `back` `love`, `doodles` 3 (the stamps a photo), `border` 24, `color` `love`; and its flash goes off at most three times a second (a capture past them is taken without one; the plain booth flashes on every capture) |
| `purikura` | booth with `look = "purikura"`: a table of the look's keys. `frames` the photos' frame colours in turn (`["rose", "iris", "foam", "gold"]`), a band `frame` px at 1080 (16) over the photo's edge with a row of dots along it, the corners rounded; `pastel` (0..1, 0.5) mixes the frames and `glow` toward the palette's light neutral (the lighter of `base` and `text`: pastel with Moon and with Dawn) and the paper 30 % further; polka dots `dots` px at 1080 across (8; 0 none) in that light colour. `stamps` the doodle shapes the booth's `doodles` are drawn in (`heart`, `sparkle`, `twinkle`, `bow`, `star`, `flower`), each a sticker: its closed outlines filled in one of the `inks` (`love`, `iris`, `foam`, `gold`), holes and lines across it in a light tint, a white rim, popping in after the photo lands round its edges (not along its foot when the pen writes there) and boiling; `marks` `[{shape, at, size, color, angle}]` stamps on every photo at `at` (photo fractions; `size` photo heights, 0.2): ears on her head. `write` a short text a photo in turn (`""` none; up to 24 of A-Z, a-z, 0-9, space and `. , : / - ! ? ' + ~ ♡ ☆ ♪`; no lyrics) written by the gel-pen hand (`mkmmd/core/hand.py`: drawn strokes in the photo's ink, white rimmed, boiling with the doodles) along the photo's foot, `letters` photo heights tall (0.13, smaller when it would not fit), a moment after it lands, the pen moving `speed` caps a second (14). The beauty filter: `lift` (0..1, 0.45) brightens it toward its lights, `soft` (0..1, 0.5) soft-focuses it (the lights spread, skin smoothed), `bloom` (0..2, 0.7) blooms its whites in `glow` (`rose`, pastel) |

```toml
[[fx]]                        # Warhol's Screen Test on the head close-up
kind = "film"
from = 3.0
to = 6.0
mono = true
fps = 15
grain = { amount = 0.09, size = 1.8 }
flicker = 0.08
scratches = 0

[[fx]]                        # an X over each eye
kind = "doodle"
from = 6.0
to = 8.0
marks = [{ shape = "x", at = { point = 'bone("eye.L").head' }, size_m = 0.05, color = "pine" },
         { shape = "x", at = { point = 'bone("eye.R").head' }, size_m = 0.05, color = "pine", t = 6.25 }]

[[fx]]                        # the photo-booth strip: four snares, four captures
kind = "booth"
from = 12.0
to = 14.0
lane = "snare"
min_db = -3.0
caption = "print club"

[[fx]]                        # a purikura strip on the chug: dates and words written over the photos, ears on her head
kind = "booth"
from = 3.0
to = 6.0
lane = "chug"
focus = [0.56, 0.4]
look = "purikura"
purikura = { write = ["11.01", "kawaii", "BFF ♡", "best friends"], marks = [{ shape = "bunny_ears", at = [0.55, 0.13], size = 0.3 }] }
```

#### Footage, stickers and figure moves

`kind = "footage"` is a collage over a video instead of flat colour: the figure (`objects`), as the frame shows it (or as
`fill` says), cut out of the picture and laid over the frames of a clip, a sticker on stock footage and never blended into it.
`mkmmd/footage.py` reads the clip with OpenCV as display-space RGB, keeps the last frames it decoded (reading on is fast, a
jump seeks), and fits each frame to each output's size about that output's `focus`; anything else that shows a clip (a split
screen's video half) reads it through the same module. A clip that is missing or unreadable, or a span that starts past its
end without `loop`, is an error before `mk post` encodes anything (in `mk look` when the frame is drawn).

| Key | Meaning |
|---|---|
| `clip` | footage (required): a video file, project-relative, absolute or `~/...` (anything with a path separator, `~` or a video extension), or the slug of an asset whose path is a video (`mk ref footage` registers Pexels videos as `pexels_<id>`, below) |
| `in`, `rate`, `loop` | footage: the clip's second on the span's first frame (0), its speed (1; 0 a still, 0.5 slow motion, at most 16), and `loop = true` to start it over past its end; without it the last frame holds there |
| `fit`, `focus`, `mirror` | footage: `cover` (default: the frame filled, the clip cropped with its point `focus` `[x, y]` (0..1 of the clip as the file has it; default the middle) as near the frame's middle as the crop allows) or `contain` (all of it, centred on the step's `back`); `mirror = true` flips it left to right (the focus with it). Give a vertical output its own `focus` through `[fx.aspect.9x16]` |
| `shadow` | every collage: `{color, offset, blur, alpha}`, a drop shadow under the figure and its edge: `color` (black), `offset` `[dx, dy]` frame heights right and down (`[0.008, 0.012]`), `blur` px at 1080 (10, a Gaussian's sigma), `alpha` (0.4); default none. A smaller copy (a grid cell, an orbit's ring) casts a smaller one. With a thick white `edge` (12-18 px at 1080, grown round the figure and rounded where it bends in, as a die cut is) the figure is a sticker |
| `keys`, `pivot` | every collage: the figure moved before it is laid out, `[{t, turn, dx, dy, scale, ease}, ...]` over clip seconds: `turn` degrees counter-clockwise on screen about `pivot`, `dx`, `dy` frame heights (right, down), `scale` times. From key to key by the `ease` of the key arrived at: `inout` (default), `in`, `out`, `linear` or `back` (a 10 % overshoot that settles); held before the first key and after the last, and what a key leaves out stays as the key before left it (at rest before the first). `pivot` `base` (default: the middle of the bottom of the figure's box on the span's first frame, her feet), `center` (its centroid there) or `[x, y]` fractions of the figure's own frame. A pivot found on the figure reads its layer on the span's first frame, which `mk look` renders beside the frames it is asked for |
| `shot` | every collage: the figure taken from another shot's own render instead of the cut's frame, through its camera, look and lights: any `[[shot]]` but the one in the cut over the span, typically one with `plate = true` looking at a cast member who stands alone, far from the set (out of every other shot's view, and nothing in front of her). `mk render` draws that shot's frames (`plate/<shot>/`) and the figure's coverage through its camera (`figure/<key>/`, the key named from the patterns and the shot) on the span's frames and on those it reads (a `delay`'s poses before), claimed, resumable and counted in the disk check like an insert's plates; `mk build` keys its camera there and `mk look` renders them beside the frames asked for. The figure is laid where its own frame shows it unless `at` and `height` set it elsewhere; a `center` at a scene point is still projected through the shot in the cut. The same entry for every output (an `aspect` table may not change it) |
| `at`, `height` | every collage: the figure set into the frame, its `pivot` (found on the span's first frame) put at `at` `[x, y]` (frame fractions) and its box there scaled to `height` of the frame's height (0.01 .. 10; either left as it is when not given). One map for the whole span, fixed from that first frame: the figure moves in the frame exactly as it moves in its own, and nothing jumps when its box changes (an arm strumming out). `keys` move it on about the pivot where it lands. Each output its own with `[fx.aspect.<output>]`: a tall frame with no room beside her places it smaller, elsewhere |

A cardboard stand-up playing possum: the sticker tips over flat on the downbeat (an ease-in, falling) and pops back up:

```toml
[[fx]]
kind = "footage"
from = 24.42
to = 26.5
objects = "Reisen*"
clip = "pexels_4254119"
in = 6.0
edge = { color = "#ffffff", width = 14 }
shadow = { color = "#1f1d2e", alpha = 0.45 }
keys = [{ t = 24.45 }, { t = 24.8, turn = -90, ease = "in" }, { t = 25.6 }, { t = 25.85, turn = 0, ease = "back" }]

[fx.aspect.9x16]              # lying flat across the tall frame: smaller, and to the left
keys = [{ t = 24.45 }, { t = 24.8, turn = -90, scale = 0.7, dx = -0.22, ease = "in" }, { t = 25.6 },
        { t = 25.85, turn = 0, scale = 1.0, dx = 0.0, ease = "back" }]
```

Footage comes from `mk ref footage ID... [--min-height 1080] [--orientation landscape|portrait|square]` (`mk ref search`
finds the ids): each Pexels video's best rendition, the largest whose short side is at least `--min-height` under the 150 MB
cap, is downloaded into `<project>/footage/pexels_<id>.mp4` (kept when a file of its size is there) and registered as the asset
`pexels_<id>`, kind `reference`: its author and their page (`author_url`), the video's page, the Pexels License and what it
forbids, a credit line, and in `extra` the file's own size, frame rate and length (Pexels sometimes labels a rendition bigger
than the file it serves: a file whose short side is under `--min-height` is not registered and stays in `footage/`);
`--orientation` refuses a video of another shape. The registry has one entry per video: its path is the latest project that
fetched it. A footage `clip` (and a video sticker's) counts for `mk assets credits` and `mk export web` as the cast does, a slug
or a file the registry knows; credit any other file in `[credits] lines`.

#### Video stickers: clip

`kind = "clip"` turns footage round: the video itself, cut out in a shape, laid over the picture as a die-cut sticker (the
picture stays; a `footage` collage lays a figure over a clip that fills the frame). `mkmmd/core/videosticker.py` checks it
and lays it out (tested), `mkmmd/videosticker.py` draws it: the clip's frame (read by `mkmmd/footage.py` as footage's)
fitted to cover the sticker's box at the size it shows at, cut out by the shape (antialiased by each pixel's share), under
its `edge` and `shadow` (as a collage's figure wears them). A missing clip, or a span starting past its end without `loop`,
is an error before `mk post` encodes anything. It reads no layers; mk post draws it on the CPU (mkr-post has no `clip`).
A `negative` sticker shows no clip: its face is the picture under it inverted, so negative type answers whatever is
behind it, light on dark and dark on light, and reads with no edge or shadow (but not over flat mid-grey, its own negative).

| Key | Meaning |
|---|---|
| `clip`, `in`, `rate`, `loop`, `focus`, `mirror` | the video, as footage's: a file or an asset slug (required but by a `negative` sticker, which takes none of these keys), its second on the span's first frame (0), its speed (1), `loop` (false), its point `focus` `[x, y]` (0..1 of the video, the middle) kept as near the box's middle as the crop allows, `mirror` (false). Always fitted to cover the box (no `fit`) |
| `shape`, `radius` | `rect` (default, its corners rounded by `radius` of the box's short side: 0.12; 0 square, 0.5 its short sides half circles), `circle` (an ellipse filling the box: round at `ratio` 1), `heart`, `star` (five puffy points) or `text` (the letters of `text`: the clip shows through them, keyed type); `radius` is a rect's only |
| `text`, `font` | a `text` sticker's words (required; `"\n"` breaks lines, centred) and its font (a font file or a font slug of the asset registry; default the registry's first font tagged `mono`): heavy faces show the most of the clip. The letters are set once (Pillow), cropped to their ink, and are the sticker's shape; a green `edge` round them reads as a chroma key's spill |
| `at`, `size`, `ratio` | the box's middle at `at` (frame fractions, `[0.5, 0.5]`; it may lie off the frame), `size` frame heights tall (0.4, 0.01..10) and `ratio` times as wide (default the shape's own: 1 for a rect and a circle, 1.106 for a heart, 1.044 for a star; a `text` sticker's is its letters', and takes no `ratio`) |
| `keys` | moves `[{t, turn, dx, dy, scale, ease}, ...]` over clip seconds, as a collage's figure's ([Footage, stickers and figure moves](#footage-stickers-and-figure-moves)), about the sticker's middle; its edge and shadow scale with it |
| `edge`, `shadow` | a die-cut `{color, width}` (px at 1080, default 3; a sticker's white paper 12-18) grown round the shape, rounded where it bends in (the heart's cleft, the star's inner corners); `{color, offset, blur, alpha}` a drop shadow, as a collage's; both default none |
| `blend` | `over` (default: an opaque sticker; the shape and its edge cast the shadow), `screen` or `add`: the face is light, the clip screened or added onto the picture inside the shape, so its black vanishes and its lights glow over the scene (night light trails). The edge stays paper, laid over, and only that rim casts the shadow (inside the face too: a face of light casts none, and a `shadow` without an `edge` is an error). `negative`: no clip, the face is the picture under the sticker inverted (before its own edge and shadow), laid as `over` lays a clip |
| `strength` | the whole sticker's mix over the picture (default 1) |

```toml
[[fx]]                        # a heart of night traffic screened over the drive: its black gone, it pops in and bounces
kind = "clip"
from = 0.0
to = 6.0
clip = "pexels_28615172"
in = 11.0
shape = "heart"
at = [0.74, 0.3]
size = 0.42
blend = "screen"
edge = { color = "#ffffff", width = 14 }
shadow = { color = "base", alpha = 0.55 }
keys = [{ t = 0.0, scale = 0.35, turn = -24 }, { t = 0.5, scale = 1.0, turn = 8, ease = "back" }, { t = 2.9 },
        { t = 3.3, turn = 5, scale = 1.12, ease = "back" }]

[fx.aspect.9x16]              # the tall frame: above her, smaller
at = [0.5, 0.2]
size = 0.24
```

#### Dividing grids and silkscreen prints

A grid's `cells` entry is `n` (n x n) or `[columns, rows]`, and a list of them steps on the hits: `cells = [1, [2, 1], [2, 2],
[4, 2], [4, 4]]` is 1 -> 2 -> 4 -> 8 -> 16 (one layout alone is `[[4, 2]]`; a list that walks back down divides back). Each
cell shows the whole frame contain-fitted and centred in it on the cell's `back` (seamless: the figure is cut out).

| Key | Meaning |
|---|---|
| `divide` | grid: seconds (default 0: a cut). On a step to a layout of more cells each new cell grows out of its parent, the cell of the layout before that holds its middle (child (i, j) of parent (i x c0 // c1, j x r0 // r1) when one layout divides the other), its copy sliding and scaling from the parent's place to its own over `divide` s, eased out: cell division. To fewer cells it plays the other way, the old cells sliding into the new cell that holds their middle, in its colours. A step's move ends early when the next hit comes first; the copies on their way overlap, none cut off |
| `fill = "silkscreen"`, `inks`, `offset` | every collage (`fill`, or a stack's `bands` entry): Warhol's silkscreen. The figure's picture cut into flat inks by its luma: `inks` ink sets of 2 to 4 colours each, the darkest band's first (default: the palette's darker of `base` and `text`, then two of its accents; four sets), read at the step plus the cell (copy, panel, region) as `back` is. The bands are cut at the figure's own luma quantiles, so a dark figure prints as well as a light one (the darkest fifth is the key plate, the rest in equal shares), its luma smoothed a little first: no gradient inside a band. The key plate also draws the lines (eyes, lashes, the edges of fingers and folds: where the luma dips under its surroundings) and is printed `offset` `[dx, dy]` px at 1080 off register (`[4, -3]`): where it misses, the band under it shows, and past the figure it prints on the ground |

```toml
[[fx]]                        # her face as a Warhol grid, its colours turning on the snares
kind = "grid"
from = 12.41
to = 15.67
objects = "Reisen*"
cells = [[2, 2]]
fill = "silkscreen"
lane = "snare"
min_db = -3
back = ["#34a8d8", "#f08a24", "#e8508e", "#7cc35a"]
inks = [["#141414", "#f2c14e", "#f6b8c8"], ["#141414", "#e8508e", "#9ad8f0"], ["#141414", "#7cc35a", "#f7d23e"],
        ["#141414", "#f08a24", "#f4f0e6"]]

[[fx]]                        # cell division on the snares, 1 -> 2 -> 4 -> 8 -> 16
kind = "grid"
from = 15.67
to = 17.86
objects = "Reisen*"
cells = [1, [2, 1], [2, 2], [4, 2], [4, 4]]
lane = "snare"
min_db = -5
divide = 0.22
back = ["gold", "love", "pine", "rose"]

[fx.aspect.9x16]              # a tall frame divides along its length first
cells = [1, [1, 2], [2, 2], [2, 4], [4, 4]]
```

#### Rainbow stacks: sprite bands, era looks, tunnels, busy clones, squash

A stack grows into a video's main motif with more keys, all optional, each composing with the others, with `mirror` and with a
figure from a `shot` of its own (`mkmmd/core/stack.py` checks them and lays the copies out in time, numpy-free and tested;
`mkmmd/stack.py` draws them). Every figure and copy is laid by an affine map from the figure's frame within the box it covers,
the same pixels as pasting it over the whole frame: a plain stack draws exactly as it did. The colours of a band, its look and
its sprites ride with a copy wherever it goes.

| Key | Meaning |
|---|---|
| `bands` entry `{color, sprites, ...}` | a band may be a table instead of a colour: that copy is filled with a wallpaper of `sprites` (registry slugs, files, globs or `{tag = ...}`, as a mandala ring's: `{tag = ["veggie", "veg_red"]}` the red veggies) on a ground of `color` (required), each pulled `tint` (0.3) of the way toward it, `size` (0.1 of the frame's short side at the copy's own size) long, in a white die-cut, turned and sized a little at random in a staggered lattice (`seed`, 1), the wallpaper scrolling `scroll` `[dx, dy]` short sides a second (`[0.02, -0.04]`), clipped to the copy's silhouette, a line `line` px at 1080 wide (6) of `color` inside its rim (none where the figure meets the frame's edge). It is laid in the figure's frame, so it shrinks and moves with its copy. `eyes` and `mouth` put googly eyes and a smile on the sprites ([Googly eyes](#googly-eyes)), kicked by the stack's own hits; the whites are not tinted. A list mixes tables, colours, `"picture"` and `"silkscreen"`, read round the copies as ever: eat the rainbow, one family a band |
| `looks` | the copies drawn in eras' media, a name or a list read as `bands` is (copy j its entry step + j - 1): `"halftone"` (60s print: Ben-Day dots of each colour's own ink on newsprint, as dense as the colour is dark, a screen 10 px at 1080 apart turned 15 degrees), `"flat"` (Peter Max: the colours smoothed into flat shapes in twelve vivid hues at three values, each shape and the rim in a thick ink line), `"vhs"` (80s tape: a camcorder's soft luma, the colour bled to the right and past the copy's edge, washed out, noise in streaks, jittered lines, scanlines), `"pixel"` (00s: blocks 12 px at 1080 in four levels a channel, there or not as each is half covered), `"chrome"` (now: iridescent liquid chrome as `kind = "chrome"` draws it, its bands drifting) or `"none"` (as its band draws it). A look works on the copy where it lands, its dots, lines and blocks sized with the copy (down to 0.3 of its size, no smaller) and fixed to it as it moves (a tape's scanlines and noise are the screen's); the copy's `edge` goes round what it draws (a pixel copy's blocks) |
| `recede` | `true` or `{count, scale, fade, speed, snap}`: the infinite tunnel. `count` (16, 2 .. 32) copies, each `scale` (0.8, 0.3 .. 0.98) as large as the one in front about `center`, the vanishing point: a copy u places deep is `scale` ** u as large (put `center` off the figure; copies shrinking toward a point inside her hide behind her). It falls inward forever: a place deeper on each of the stack's hits (`times` or `lane`), eased over `snap` s (0.15) with a little overshoot, and `speed` places a second (0; negative rises out). A new copy comes out from behind the figure as the farthest fade out over the last `fade` (0.3) of the tunnel, so nothing jumps, and a tunnel of n bands at a steady speed comes round again every n places. Each copy keeps its band and look as it falls; with a `delay` a copy u places deep shows her u x `delay` frames behind (it ages as it falls). The plain `count`, `scale` and `step` are unused; a tunnel copy's `edge` thins with its size. Not with `peel` |
| `delay` | frames each copy is behind the one before (the plain key), or a list: copy j's own (entry j - 1, round and round), big ones too: busy clones, each her at a moment of her own. A tunnel reads a list at its depths, between them the share of the way. `mk look` renders the frames a copy shows beside the ones asked for |
| `peel` | `{at, t, dur, gap, scale, ease}`: busy clones leave the stack. Copy j slides from its place in the stack to its own `at` entry (`[x, y]` frame fractions where its pivot goes, her feet with `pivot = "base"`; one point or a list read round the copies), `scale` (1, or a list) times the figure's size, starting at `t` (clip seconds; default the span's start) + (j - 1) x `gap` (0.12 s) and taking `dur` s (0.45), eased `ease` (`back`, default: it overshoots and settles; `inout`, `in`, `out`, `linear`). With different `delay`s each clone is her doing something else. The pivot is found on the span's first frame |
| `squash` | `{times or lane (min_db, every), amount, decay, attack}`: on each hit the figure and every copy are pressed down by `amount` (0.2, at most 0.8) of their height over `attack` s (0.03), then spring back up past it and settle, a damped swing whose first stretch comes 0.9 x `decay` s (0.15) after the squash, e^-0.9 as big; hits close together add up. Each is scaled about its own feet (the middle of its box's bottom edge: it stays on the ground) by h down and 1 / h across, its area kept, and a copy behind in time squashes when its moment comes: the squash runs back through the echo. On `back = "picture"` with the figure cut from the frame itself, her own place in the picture is painted over from round it (an inpaint at a quarter of the size) while she squashes, so no second her shows; a figure from a `shot` of its own never needs it |

All of it composes: `mirror` draws everything again reflected about the vertical line through `center` (the clones peel to
the mirrored places; a tunnel, its vanishing point on that line, falls into it from both sides), a figure from its own `shot`
set in by `at` and `height` takes every key, and `aspect.<output>` tables may give an output its own places and sizes (`at`,
`height`, `peel`, `center`) but not what changes the frames it reads (a `delay`, a tunnel's `recede` with a `delay`: the plan
checks). A 1080p frame of a 20-copy tunnel in looks takes about half a second, bands of sprites a little more.

```toml
[[fx]]                        # drop 2: the echo eats the rainbow, each copy an era, falling into a tunnel on the kicks
kind = "stack"
from = 11.14
to = 18.1
objects = "Reisen_mesh*"
center = [0.8, 0.22]          # the vanishing point, off her
recede = { count = 20, scale = 0.8, speed = 0.7 }
lane = "kick"
bands = [{ color = "love", sprites = { tag = ["veggie", "veg_red"] }, eyes = true },
         { color = "gold", sprites = { tag = ["veggie", "veg_orange"] } },
         { color = "#ead27a", sprites = { tag = ["veggie", "veg_yellow"] }, eyes = true, mouth = true },
         { color = "pine", sprites = { tag = ["veggie", "veg_green"] } },
         { color = "iris", sprites = { tag = ["veggie", "veg_purple"] }, eyes = { look = "viewer" } }]
looks = ["halftone", "flat", "vhs", "pixel", "chrome"]
squash = { lane = "sub", amount = 0.2 }
back = "picture"
edge = { color = "#ffffff", width = 12 }

[[fx]]                        # line 2: busy clones peel off, each her a third of a second further back
kind = "stack"
from = 6.0
to = 9.0
objects = "Reisen_mesh*"
count = 4
delay = [10, 20, 30, 40]
bands = ["picture"]
peel = { at = [[0.12, 0.96], [0.3, 0.96], [0.7, 0.96], [0.88, 0.96]], t = 6.3, scale = 0.6 }
back = "picture"
edge = { color = "#ffffff", width = 10 }
```

The guitarist trio of the stack example above takes them all as they are: add `bands` of sprites, `looks` and a `squash` to
it and the plate's guitarist and her mirror image echo in eras and veggies on either side of the dance.

#### Mandalas and trails

`kind = "mandala"` lays concentric rings of copies about a `center` on its `back`: copies of the figure (`objects`, cut out of
the picture as a collage cuts it and drawn as its `fill`), of sprites (pictures with alpha: photos of real butterflies, clovers
and flowers from the asset registry) or of stickers drawn in palette colours (flat ones: a clover, a flower, a heart, a star,
a butterfly; and two kits drawn with a pen's ink line: a moon, a rabbit, a carrot and dango; a daisy, a cassette, a guitar
pedal and cherries).
Every copy is a sticker: with a thick white `edge` (grown round it and rounded where it bends in, as a die cut is) and a drop
`shadow` it is cut out and laid on the page. Each ring turns its own way (every other ring the other way), steps on the hits
(`times` or a `lane`, as a collage steps; a list of `back`, `fill` or shape `colors` is read at the step plus the copy), with a
`bloom` jumps out on each hit and settles, and with a `grow` its copies pop in one by one; its `strength` mixes it over the
picture, as a collage's does. `kind = "trails"` leaves rainbow after-images behind a moving figure. `mkmmd/core/mandala.py`
checks, times and lays them out and draws the stickers' shapes as polygons (numpy, tested), `mkmmd/mandala.py` draws them
(numpy, OpenCV, Pillow): each copy is a tile cut once at its ring's size and laid by an affine map, its shadow cast where it
lands (about 0.15 s for a 1080p frame of 30 stickers and her copies).

| Key | Meaning |
|---|---|
| `rings` | mandala (required): up to 12 tables, ring i `{count, radius, scale, turn, step, phase, mirror, spin, orient, angle, source, fill}`: `count` copies (6 + 4 i, at most 64) on a circle `radius` short sides of the frame (0.15 (i + 1)) round the centre, the first `phase` degrees counter-clockwise from straight up (0), the rest evenly after it, each `scale` short sides large (the figure's height as it stands on the span's first frame, 0.25, so its copies keep their size against the rings in every output; a sprite's or a shape's longer side, 0.12). A ring turns `turn` degrees a second counter-clockwise (12) and `step` more on each hit (0), eased over `snap`; its copies stand `radial` (`orient`, default: each one's top away from the centre) or `upright`, turned `angle` degrees more (0) and spinning `spin` degrees a second about themselves (0); `mirror = true` flips every other copy left to right (a kaleidoscope's reflections). The outer rings are drawn first |
| `source` | ring: `"figure"` (default: the figure in `fill`, the ring's own `fill` list over the entry's, `inks` as a collage's), `{sprites = [...]}` or `{shape = ...}`. Sprites: registry slugs, files or globs (project-relative, absolute or `~/...`; `.png`, `.webp`, `.tif`, `.jpg`) or `{tag = "butterfly"}` (one tag or a list: every picture of the registry that has them all), one or a list; trimmed to their alpha and dealt round the ring shuffled by `seed`, every one before any comes again, each ring its own deal. A slug, tag, file or glob that finds nothing is an error before anything is drawn. Shapes: `shape` one of the flat ones (`clover` (four heart leaves with their pale chevrons, a stem), `flower`, `heart`, `star`, `butterfly`) or of two kits drawn with a pen's ink line round their parts: a moon rabbit's (`moon` (a crescent asleep), `rabbit` (a white one's head: its colour is the inside of its ears, its `accent` its eyes), `carrot`, `dango` (three on a stick: the colour, white and the `accent`)) and a zine's (`daisy` (white petals round the colour), `cassette`, `pedal` (a guitar pedal) and `cherry`), or a list of them round the ring, in `colors` (default the shape's own: the clover `pine` and `foam`, the flower `love`, `rose` and `iris`, the heart `love` and `rose`, the star `gold` and `rose`, the butterfly `iris`, `love`, `foam` and `gold`, the moon and the carrot `gold` and `rose`, the rabbit and the dango `rose`, `love` and `iris`, the daisy `gold`, `rose` and `love`, the cassette `iris`, `foam`, `love` and `gold`, the pedal `love`, `pine`, `iris` and `gold`, the cherries `love` and `rose`), shaded toward `ink` (`text`: the pen's line too) and white, with an `accent` (the flower's middle, the butterfly's spots, the rabbit's eyes, the carrot's leaves; default the shape's own: `gold`, but `rose` for the heart and the moon, `love` for the star and the rabbit, `pine` for the carrot and the cherries, `foam` for the dango). `pen = true` draws the pen's line round the flat ones too (round the whole shape, in `ink`): a star or a heart beside a kit's stickers is drawn as they are |
| `eyes`, `mouth` | a ring's sprite source, `{sprites = [...], eyes = true, mouth = true}`: googly eyes on every sprite's face and a doodled smile under them (see Googly eyes below) |
| `center`, `back` | mandala: the rings' centre (`[x, y]` frame fractions or `{point}`, default the middle); the ground, a colour, a pattern (as a collage's) or `"picture"` (the frame as it reached the effect: the rings over her world), or a list of them read at the step (default `base`) |
| `objects`, `center_figure` | mandala: the figure, required with a ring of it or `center_figure = true` and only then; `center_figure` lays the figure itself over the rings where the frame shows it, drawn `fill` entry step, with its edge and shadow. `fill`, `inks`, `offset`, `keys` and `pivot` are a collage's |
| `edge`, `shadow` | mandala: every copy's, as a collage's: a white `edge` 10-16 px at 1080 and a soft `shadow` make stickers. The shadow's offset and blur are the same under every copy (stickers on one page) |
| `bloom` | mandala: a number (`amount`) or `{amount, grow, attack, decay, lag}`: on each hit the rings jump out by `amount` of their radius (0.15) and their copies grow by `grow` (0.1) over `attack` s (0.05), dying away as exp(-t / `decay`) (0.3 s), each ring `lag` s (0.04) after the one inside it. It needs the mandala's `times` or `lane` |
| `grow` | mandala: `{times or lane (min_db, every), dur, gap, order, per}`: the copies pop in one by one, each a spring from nothing that overshoots 12 % and settles over `dur` s (0.35), `gap` s (0.06) after the one before, in `order`: `rings` (default: the inner ring first, round each from its first copy), `inward`, `around` (round the clock from the top across the rings) or `random` (by `seed`). The copies are dealt over the times, `per` a time (default all of them shared evenly; those left over join the last), each group from its time on: a field of clovers sprouting into a mandala on the beats. Before its time a copy is not drawn |
| `alternate`, `seed`, `snap` | mandala: every other ring turning and stepping the other way (true); the sprites' deal and a `random` grow (1); seconds a hit's step takes, eased out with a 10 % overshoot (0.2; 0 a cut) |
| `objects`, `count`, `delay`, `fade` | trails: the figure (required) on `count` (6) earlier frames `delay` (2) frames apart, copy j (1 the newest) at opacity `fade` ** j (0.85), laid oldest first behind the figure, which stays over them as the frame shows it. `mk render` draws its figure on those frames, and `mk look` renders them beside the frames asked for |
| `colors`, `spread`, `cycle`, `times`, `lane`, `snap` | trails: a ramp of colours (default a Rosé Pine rainbow: `love`, `rose`, `gold`, `foam`, `pine`, `iris`), copy j its colour `spread` x (j - 1) entries along it (1), blended between entries; the colours run out along the trail `cycle` entries a second (1), and one more on each hit (`times` or a `lane`) over `snap` s (0.12) |
| `blend`, `tint`, `step` | trails: `over` (default: for light grounds), `screen` or `add` (light trails on a dark ground); `tint` 1 copies flat in their colour (default), less mixes in the figure's own picture; `step` `[dx, dy]` short sides (`[0, 0]`): copy j moved j x `step` further, a fan of rainbow even while she stands still. `strength` scales the copies' opacity |

A mandala's sprites count for `mk assets credits` and `mk export web` as a footage clip does: a slug or a tag names registry
entries, and a file or a glob's match counts when the registry has an entry with its path; credit any other in `[credits]
lines`. A tall output frames the same rings across its width: give `[fx.aspect.9x16]` its own `rings` (an outer ring more) to
fill its length.

```toml
[[fx]]                        # a mandala of her: three mirrored rings on rose, turning both ways, blooming on the snares
kind = "mandala"
from = 12.0
to = 15.0
objects = "Reisen*"
back = ["rose", "iris"]
lane = "snare"
edge = { color = "#ffffff", width = 7 }
shadow = { color = "#575279", alpha = 0.3, blur = 5 }
bloom = { amount = 0.12, grow = 0.08 }
rings = [{ count = 6, radius = 0.15, scale = 0.24, mirror = true, turn = 14 },
         { count = 10, radius = 0.32, scale = 0.2, mirror = true, step = 18 },
         { count = 14, radius = 0.49, scale = 0.17, mirror = true, turn = 8 }]

[[fx]]                        # clovers sprouting round her, then butterfly stickers and real butterflies from the registry
kind = "mandala"
from = 15.0
to = 18.0
objects = "Reisen*"
center_figure = true
edge = { color = "#ffffff", width = 14 }
shadow = { color = "#575279", alpha = 0.3 }
grow = { times = [15.0], gap = 0.03, order = "around" }
rings = [{ count = 8, radius = 0.3, source = { shape = "clover" } },
         { count = 12, radius = 0.42, source = { shape = "butterfly" }, mirror = true },
         { count = 18, radius = 0.56, scale = 0.1, source = { sprites = { tag = "butterfly" } }, orient = "upright", spin = 20 }]

[[fx]]                        # rainbow after-images while she dances, fanned out a little to her right
kind = "trails"
from = 18.0
to = 21.0
objects = "Reisen*"
delay = 3
cycle = 1.5
step = [0.006, 0.0]
```

##### Googly eyes

Every registry sprite source (a mandala ring's `{sprites = ...}`, a stack's band, a paint app's stamps, an iris's plate
ring) takes `eyes = true` or a table, and `mouth = true`: two white googly eyes with black pupils on each sprite's face, and a
small doodled smile under them. The whites (a dark rim, a dome shading grey toward its lower right) and the smile are
painted into the sprite's tile, which grows to hold eyes that stick out of it (the die-cut `edge` goes round them); the
pupils and the dome's glint over them are drawn on every frame where the tile is laid. The eyes sit where the sprite's
registry entry says (`extra.eyes`), else on the upper middle of the widest part of its alpha, moved up or down that part to
where both lie on it. Each pupil is a damped spring inside its eye: it rests where gravity drops it (or looks out, or at a
point), swings when the sprite moves (a pupil keeps going when its eye starts, stops or turns: a ring's step, a bloom, a
copy popping in) and is kicked on the hits, every sprite its own way and its two eyes a little apart, bouncing off the
rim. It is a function of the clip second and the `seed` (the spring runs from rest over four of its decay times before the
frame, about 1.4 s at the defaults and at most 3 s, on a grid of 60 samples a second aligned to clip second 0), so any frame
comes out the same in any order, in `mk look` and `mk post`.
`mkmmd/core/eyes.py` checks the keys, places the eyes and swings the pupils (numpy, tested); `mkmmd/eyes.py` draws them;
`mkmmd.mandala.sprite_tile(path, size, edge, face=)` and `lay(..., pupils=)` are the shared path every sprite source uses.

| Key | Meaning |
|---|---|
| `eyes` | `true` or `{size, wobble, look, seed, times or lane (min_db, every)}`. `size`: the eyes' size times their anchored or found size (1; 0.1 .. 4). `wobble`: a number (`amount`: 0 still pupils, 1 the default swing, 2 a wilder one) or `{amount, hz, damping}`, the spring's frequency (3 Hz) and damping ratio (0.15: about three swings before it settles). `look`: `"down"` (default: the pupils fall to the bottom of their eyes, a googly eye's way), `"viewer"` (straight out of the frame), `"center"` (toward what shows them: a mandala's `center`, her head with `{point}`) or `[x, y]` (frame fractions). `seed` (1): another take of the kicks. `times` (clip seconds) or a `lane` of the [audio] hits file kick the pupils; default the hits of what shows them (a mandala's `times` or `lane`) |
| `mouth` | `true`: a small doodled smile under the eyes, where the sprite's entry says (`extra.eyes.mouth`; `false` there: no smile on that sprite); with `eyes` or alone |
| registry `extra.eyes` | a sprite entry's own anchors, `{at = [[x, y], [x, y]], r, mouth = [x, y] or false}`: the eyes' and the smile's middles (fractions of the sprite's box trimmed to its alpha, x across, y down; a little past the box is allowed) and an eye's radius (a share of the box's longer side); what is left out is found from the alpha. `mk assets set SLUG 'extra={"eyes": {...}}'` merges it in |

The registry's veggies, `{tag = "veggie"}` (`mk assets list --tag veggie`), are photos of vegetables cut out, each tagged
`["sprite", "veggie", <name>, <family>]` with its family by its real colour (`veg_red`, `veg_orange`, `veg_yellow`,
`veg_green`, `veg_purple`, `veg_white`: `{tag = ["veggie", "veg_green"]}` fills a green band) and anchors in `extra.eyes`.

```toml
[[fx]]                        # eat the rainbow: a ring of each family round her, their eyes on her, jiggling on the kicks
kind = "mandala"
from = 3.0
to = 6.0
objects = "Reisen*"
center = { point = "bone('head', 'Reisen_arm').center" }
center_figure = true
lane = "kick"
edge = { color = "#ffffff", width = 10 }
rings = [{ count = 8, radius = 0.24, scale = 0.13, orient = "upright", source = { sprites = { tag = ["veggie", "veg_red"] }, eyes = { look = "center" } } },
         { count = 16, radius = 0.42, scale = 0.12, orient = "upright", source = { sprites = { tag = ["veggie", "veg_green"] }, eyes = { size = 1.3, wobble = 1.5 }, mouth = true } }]
```

#### Colour fields: liquid light, op-art, floods

Three kinds paint the picture with a field of colour, kept behind her or on her by a `mask` (`objects`, the figure layer `mk
render` draws, as a pixel sort's): `liquid`, a 1960s oil-and-water light show; `spiral`, op-art in 2 to 4 colours; `flood`, a
colour blooming out from a point to fill the frame and drawing back. `mkmmd/core/fields.py` checks and times them (numpy-free),
`mkmmd/fields.py` paints them (numpy, OpenCV). Lengths are short sides of the frame about the `center`, so every output shows
the same field there, cropped to its own frame; a field is a function of the clip second and its `seed` (the noise is hashed
from its lattice point, not drawn from a stream), so any frame comes out the same in any order, in `mk look` and `mk post`.
Each takes about 0.1 to 0.2 s for a 1080p frame (a drift about 0.25 s).

| Key | Meaning |
|---|---|
| `inks`, `light` | liquid: the inks (1 to 6 colours; default `love`, `gold`, `foam`, `iris`) and the lamp's `light` through the clear glass (`base`). Each ink is in the dish twice: clouds of dye swirled through the water (marbled, each place mostly one ink's) and oil, balls of it (metaballs: they merge and part) wandering, breathing and turning with the ink's own dish, the dishes turning alternate ways and the water's swirl (a curl-noise warp) bending everything. An ink's oil pushes the dyed water aside and shows its own dye, deeper in its middle; clear oil lets the light through. Along an oil edge runs a dark meniscus line with the light it bends lit just inside it, and each blob is lit like a drop from the top left |
| `scale`, `speed`, `seed` | liquid: the blobs' size (1; 0.2 .. 5, fewer balls when larger), how fast everything flows, turns and morphs (1; 0 holds it), and which light show (1). The inks are a function of the clip second: two entries with one seed are one show going on |
| `cells`, `bubbles` | liquid: how much of the oil shows oil cells, a foam of droplets each a lens (0.35; 0 none), and bubbles to a square of the frame's short side (6), each a lit rim with a glint, riding the first dish, growing in and shrinking away |
| `times`, `lane`, `min_db`, `every`, `pulse`, `decay` | liquid (optional): on each hit the inks swell out from `center` by `pulse` (0.15) and their oil grows, the edges flaring, rising over 0.05 s and dying away as exp(-t / `decay`) (0.45 s) |
| `blend`, `opacity` | liquid and spiral: how the field is laid on the picture, `over` (default: the field itself), `screen`, `multiply` (the scene lit through it) or `soft-light`, mixed in by `opacity` (1) times `strength` (and the mask) |
| `mask`, `objects` | every colour field: `ground` paints only outside the figure's coverage (her outline as antialiased as the figure layer), `figure` only inside it (a light show projected on her); both or neither. A figure the mask keeps is left exactly as it was |
| `center` | every colour field: `[x, y]` frame fractions (default the middle) or `{point = "<expression>"}`, a scene point projected on every frame of the span as a doodle's marks are: where the liquid's dishes turn about and its hits push from, a pattern's centre, where a flood falls |
| `pattern`, `colors` | spiral: `spiral` (default), `rings`, `moire`, `sunburst`, `checker` or `drift`, in 2 to 4 `colors` (`["rose", "base"]`, a drift's `["base", "pine", "gold", "text"]`; pastels mix as `"gold:base:0.5"`). Every edge is antialiased, and where the stripes run finer than a pixel (a spiral's or a sunburst's middle) it shows their colours' mean |
| `arms`, `twist` | spiral: a spiral of `arms` (5) arms of each colour, each turning `twist` (1; negative the other way) times from the centre out to half the short side; a sunburst's rays, `arms` (12) of each colour |
| `size` | rings, moire, checker and drift: a stripe's width, a square's side or a drift's ring or row width in short sides (rings 0.05, moire 0.018, checker 0.1, drift 0.07) |
| `moire`, `offset`, `angle` | moire: two interfering sets, `rings` (default) about two centres `offset` stripes apart (4) across the `center`, or `lines` crossing at `angle` degrees (6); with 2 colours a pixel in either set's stripe and not both is the second colour, with 3 or 4 each of none, one (the first or the second) and both has its own |
| `warp`, `angle` | checker: a checkerboard turned `angle` degrees (0) and bulged out about the centre by `warp` (0.6; 0 flat), Vasarely's sphere |
| `layout`, `motif`, `angle` | drift: a peripheral drift illusion, still op-art that seems to turn by itself. Its motifs stand on rings about the `center` (`layout = "rings"`, default; a whole number round each ring, a still hub in the middle) or in rows `size` apart through it (`"rows"`, turned `angle` degrees counter-clockwise, 0; every other row offset half a motif); `motif` `tie` (default: a rail tie, a rounded bar across its ring or row, two to a ring width along it) or `wheel` (a disc with a hub, one to a width). Each motif is the dark tone, its back side the darkest and its front side the lightest, on the light tone: darkest, dark, lightest, light the same way round every motif, the luminance steps the eye reads as motion, so each ring (row) seems to creep forward and the next one the other way. The `colors` take the four tones by their luma on every frame (the darkest is the darkest whatever the palette: Dawn's text and Moon's base, so the illusion turns the same way through a palette breath); with 3 colours the middle one is the dark tone or the light one (whichever it is nearer) and the other is mixed, with 2 both middle tones are mixed (0.3 and 0.7 of the way). Strongest in the corner of the eye, with high-contrast tones and motifs a few degrees of the view across |
| `speed`, `step`, `times`, `lane`, `min_db`, `every` | spiral: the stripes flow `speed` stripes a second (0.5; negative the other way): a spiral or a sunburst turns, rings flow out, a moire's second set slides through the first, a checkerboard runs diagonally. On each hit (optional; a hit on the span's first frame is where it starts) it steps on `step` stripes (1: two colours trade places), eased out over 0.12 s. A drift's stripes are its motifs: it stands still (`speed` 0; a small one, 0.1 to 0.3, creeps each ring or row the way it seems to turn), steps `step` (0.25) of a motif on a hit, and at most three times a second (a hit sooner than 1/3 s after the last step is skipped: flash-safe). mkr-post leaves a drift (and a collage on a drift back) to the CPU |
| `color`, `times`, `lane`, `min_db`, `every` | flood: the colour (`love`), or a list read drop by drop; a drop falls on each of `times` (clip seconds) or of a `lane`'s hits, one of them required, a later drop blooming over the one before. A time before the span counts: that drop is already on its way on the span's first frame, so a clip that loops can open full of the colour it closed on and draw it back |
| `grow`, `hold`, `recede`, `fade` | flood: the front runs out from the centre to past the frame's farthest corner in `grow` seconds (0.8; fast, then slowing: the whole frame is the colour at `grow`), the frame stays full for `hold` (0.4), then the front draws back to the centre over `recede` (0.6) and / or the colour fades over `fade` (0); with neither it stays to the span's end |
| `soft`, `wobble`, `seed` | flood: the front's feather, px at 1080 (24), with a paler wash running ahead of it and a darker line of ink along it, as a drop's edge dries; how blotchy and fingered the front runs and how uneven the colour lies (0.5; 0 a circle); `seed` another drop's shape (1) |

A collage's `back` (grid, orbit, mirror, panels, stack, footage round a contained clip, and a mandala's) may be a pattern
instead of a flat colour, `{pattern, colors, ...}` with the spiral kind's own keys (not `mask`, `blend`, nor hits: it steps on
the collage's own hits, and flows from the collage's span start), its `center` `[x, y]` fractions of what it fills. A pattern
fills each grid cell as the cell's own (centred in it, its lengths the cell's short side: a grid of spirals), and the whole
frame behind an orbit, a stack or the panels; a mirror folds it with the figure. A list mixes patterns and colours, read at
the step as colours are.

```toml
[[fx]]                        # the light show behind her, swelling on the kicks
kind = "liquid"
from = 8.0
to = 14.0
mask = "ground"
objects = "Reisen*"
lane = "kick"
inks = ["love", "gold", "foam", "iris"]

[[fx]]                        # projected on her too: the scene lit through the inks
kind = "liquid"
from = 8.0
to = 14.0
mask = "figure"
objects = "Reisen*"
blend = "multiply"
opacity = 0.5

[[fx]]                        # a hypnotic spiral behind her, a step on every snare
kind = "spiral"
from = 14.0
to = 17.0
pattern = "spiral"
colors = ["rose", "base"]
arms = 4
lane = "snare"
mask = "ground"
objects = "Reisen*"

[[fx]]                        # a chessboard of spirals turning both ways, the grid's colours flipping on the hits
kind = "grid"
from = 17.0
to = 19.0
objects = "Reisen*"
cells = [[2, 2]]
lane = "snare"
back = [{ pattern = "spiral", colors = ["rose", "base"], arms = 4 },
        { pattern = "spiral", colors = ["foam", "base"], arms = 4, speed = -0.5 }]

[[fx]]                        # ink dropped from her head fills the frame on the downbeat, then draws back
kind = "flood"
from = 19.0
to = 21.0
times = [19.1]
color = "love"
center = { point = "bone('head').head" }
mask = "ground"
objects = "Reisen*"

[[fx]]                        # still op-art behind her that seems to turn: rings of rail ties, every other ring the other way
kind = "spiral"
from = 21.0
to = 24.0
pattern = "drift"
colors = ["base", "pine", "gold", "text"]    # the darkest, a dark, a light and the lightest, in any order
lane = "accent"                              # a quarter of a tie on each accent, three a second at most
mask = "ground"
objects = "Rin_*"

[[fx]]                        # rows of wheels across a tall frame, creeping slowly the way they seem to go
kind = "spiral"
from = 24.0
to = 26.0
pattern = "drift"
layout = "rows"
motif = "wheel"
size = 0.09
speed = 0.15
```

#### Remaps: kaleidoscope, tunnel, melt

Three kinds move the picture's pixels about instead of painting over it: `kaleido`, a kaleidoscope; `tunnel`, an infinite
Droste tunnel falling inward; `melt`, the picture breathing, rippling, swirling, dripping or wobbling. `mkmmd/core/remap.py`
checks and times them and works out where each pixel reads the picture (numpy), `mkmmd/remap.py` draws them (numpy, OpenCV):
bilinear, the frame's edge mirrored where a remap reaches past it, and from a smaller copy of the picture where it shrinks it
(no shimmer in fine detail as a tunnel falls or a kaleidoscope zooms out). Lengths are shares of the frame's short side about
the `center` (`[x, y]` frame fractions, default the middle, or `{point = "<expression>"}`, projected on every frame of the span
as a zoom's is), so every output shows the same remap cropped to its own frame; angles are degrees counter-clockwise on screen.
Everything is a function of the clip second (and a melt's `seed`): any frame comes out the same in any order, and nothing jumps
from frame to frame but a hit's step. A 1080p frame takes about 0.05 s (a kaleidoscope, a melt) to 0.12 s (a tunnel), up to
0.3 s with a mask, an edge and a shadow.

| Key | Meaning |
|---|---|
| `segments`, `mirror` | kaleido: the frame folded into `segments` wedges (6; 2 .. 64) about `center`, alternately mirrored as in a two-mirror kaleidoscope (`mirror = true`, the default: seamless, an even number of wedges) or plain turned copies (`false`: any number, their seams antialiased) |
| `source` | kaleido: `{center, angle, radius}`, the wedge of the picture every wedge shows: its apex (`[x, y]` frame fractions, default the kaleidoscope's `center`), the way its middle points (90: up) and `radius` (short sides; default none), where it ends and folds back on itself ring after ring. At rest with the source at the centre the wedge pointing `angle` is the picture as it is. Aim it at what should feed the mandala: the ink, the petals, the street beside her rather than her |
| `turn`, `step`, `glide`, `spin` | kaleido: the source wedge turns `turn` degrees a second (8) and `step` more on each hit (30), eased out over `glide` s (0.3): the mirrors stay where they are and the pattern in them blooms and folds; `spin` turns the whole kaleidoscope on screen, degrees a second (0) |
| `zoom`, `pulse`, `decay` | kaleido: the mandala `zoom` times as large about the centre, a number or `[[t, v], ...]` keys as `strength` takes them (1), `pulse` larger on each hit (0: none; 0.25 a heartbeat), rising over 0.04 s and dying away as exp(-t / `decay`) (0.25 s) |
| `scale`, `twist`, `depth`, `frame` | tunnel: the frame holds itself `scale` times as large (0.5; 0.05 .. 0.95) about `center`, each level turned `twist` degrees (0; a spiral), `depth` levels in (10; none under a pixel is drawn). `frame = {color, width}` draws a line round each level inside its edge (a colour, or a list read level by level, each colour falling with its level; width px at 1080, 6) that shrinks with it: on busy footage the nesting hardly reads without |
| `speed`, `step`, `glide` | tunnel: it falls inward `speed` levels a second (0.5; negative rises out of it) and `step` levels more on each hit (1), eased out over `glide` s (0.35). A level fallen through is the picture it started as, the next fading in at the far end: it falls forever without a seam |
| `mode` | melt (required): `breathe` (the middle swells and shrinks back: a slow scale pulse), `ripple` (rings running out from `center`), `swirl` (a twist that grows toward the centre), `drip` (the picture sags and runs down like melting wax: straight down, most in drips) or `noise` (from heat haze, small and fast, to a liquid wobble, large and slow) |
| `amount`, `scale`, `speed` | melt, each mode its own defaults. `amount`: breathe, how much larger the middle grows (0.08; negative shrinks it); ripple, how far the rings push, short sides (0.015); swirl, degrees at the centre (90); drip, how far the bottom of the frame sags, short sides (0.25; up to 0.45 of the frame's height it stretches without folding); noise, the most it wobbles, short sides (0.012). `scale`, short sides: the bulge's radius (0.7), the rings' wavelength (0.12), the twist's radius (0.45), a drip's width (0.1), a wobble's size (0.1; 0.02 is haze, 0.3 liquid). `speed`: breaths a second (1/3: one a bar at 80 BPM), rings a second (0.8), twists a second (0: held), how fast the drips run (0.5), how many times a second the wobble changes (1; haze 3) |
| `times`, `lane`, `min_db`, `every`, `decay`, `seed` | melt (optional): on each hit the amount jumps up (over 0.04 s) and dies away as exp(-t / `decay`) (0.4 s), still between hits; a ripple's swell runs out from the centre at 1.2 short sides a second, a ring of rings on every hit. Without hits the amount holds. `seed` another take of a drip or a noise (1) |
| `strength` | kaleido and tunnel: a mix over the picture; melt: the amount (half strength is half a melt, not a mix). Key a melt in from 0: it does not fade in by itself |
| `mask`, `objects` | every remap (both or neither): `ground` remaps the ground and lays the figure back on top as rendered; `figure` remaps the figure alone, its picture and coverage together (a mandala of her, a queue of her copies falling out of the tunnel's vanishing point, the nearest fading as it passes, her melting), over the ground or `back`. Either way her place in the ground is filled from the ground round it first (each row's run mirrored in from both sides, the ground's colours spread in where a mirror cannot reach), so no copy of her is folded, nested or melted into it |
| `edge`, `shadow`, `back` | with a mask: the figure drawn as a collage's sticker, an `edge = {color, width}` grown round it and a drop `shadow` ([Footage, stickers and figure moves](#footage-stickers-and-figure-moves)). A sticker a footage collage drew before the remap is part of its ground: give the remap the same `edge` and `shadow` and they stay with her (their place is filled too); `back` (a figure mask) a flat colour under the remapped figure instead of the ground |

```toml
[[fx]]                        # the ink beside her in eight mirrored wedges, turning, a step on each snare, her on top
kind = "kaleido"
from = 4.0
to = 7.0
segments = 8
turn = 15
source = { center = [0.8, 0.45], angle = 0 }
lane = "snare"
mask = "ground"
objects = "Reisen*"
edge = { color = "#ffffff", width = 14 }
shadow = { color = "#1f1d2e", alpha = 0.45 }

[[fx]]                        # a mandala of her alone on rose, turning, pulsing on the kicks
kind = "kaleido"
from = 7.0
to = 9.0
spin = 12
lane = "kick"
pulse = 0.25
mask = "figure"
objects = "Reisen*"
back = "rose"
edge = { color = "#ffffff", width = 10 }

[[fx]]                        # the clover falling inward behind her, a level a bar and one more on the downbeat
kind = "tunnel"
from = 9.0
to = 12.0
scale = 0.6
twist = -12
speed = 0.33
times = [9.0]
frame = { color = ["love", "gold", "pine", "iris"], width = 10 }
mask = "ground"
objects = "Reisen*"

[[fx]]                        # she melts away on the stop
kind = "melt"
from = 12.0
to = 13.5
mode = "drip"
amount = 0.4
strength = [[12.0, 0.0], [13.0, 1.0]]
mask = "figure"
objects = "Reisen*"

[[fx]]                        # the picture breathes in and out once a bar
kind = "melt"
from = 13.5
to = 19.5
mode = "breathe"
amount = 0.1
```

#### Activity book and labels

`kind = "activity"` makes the frame a page of a children's workbook (`mkmmd/core/activity.py` checks and times it,
`mkmmd/activity.py` draws it): cream paper with its fibres, faint ruled lines and a red margin, the frame drawn on it
(multiplied by the page: a [crayon sketch](#looks-silhouette-vector-and-reflection) of her on paper of the colour `key`
sits on the page as if drawn there), and exercises printed on the page that a crayon does on the hits (`times`, or a
`lane` of the [audio] hits file, as a zoom's). Printed things (dotted lines, dots, numbers, a colour by number's lines)
are crisp ink; what the crayon does is wax (`mkmmd/core/crayon.py`), drawn again `fps` times a second by a hand that
wobbles `boil` px.

| Key | Meaning |
|---|---|
| `items` | required: the exercises, `[{kind, ...}]`, drawn in order (below) |
| `paper`, `grain`, `rule`, `margin` | the page: its colour (`#F6EEDC`, a cream) and fibres (0..1, 0.6); faint rules `{color, pitch, alpha, width}` (`#9DB6D6`, every 0.075 frame heights, 0.35, 1.6 px at 1080) and a margin line `{color, at, alpha, width}` (`#E68A8A` at 0.09 of the width); `false` leaves either out |
| `picture`, `key` | `multiply` (default: the frame drawn on the page, its colour `key` (default `paper`) the paper) or `none` (a blank page) |
| `ink`, `crayon`, `font` | the printed ink (`text`), the crayon an item takes by default (`love`), the numbers' font (a registry slug or a file; default the registry's first font tagged `mono`) |
| `objects` | the figure a `trace` of `shape = "figure"` follows (her figure layer, drawn by `mk render` as a collage's) |
| `snap`, `fps`, `boil`, `seed` | seconds a step eases in over after its hit (0.22); the crayon drawn again `fps` times a second (12), wobbling `boil` px (1.2); the paper's tooth |

An item (`items[]`) is `{kind, at, size, angle, t, steps, color}` and one source: `shape` (`star`, `heart`, `circle`,
`clover`, `flower`, `butterfly`; a trace may say `figure`, her silhouette grown `offset` px at 1080), `sprites` (a registry
slug, a file or `{tag}` as a mandala's, `pick` choosing among a tag's; `eyes` and `mouth` as any sprite source) or `points`
(`[[x, y], ...]` of the frame, `closed` false). `at` is its middle (frame fractions), `size` its longer side (frame
heights, 0.3), `t` when it appears (the span's start); it takes a step on each hit from `t` on and is done after `steps`
(default every hit it has). Outlines start at their top and run clockwise.

| `kind` | What the crayon does |
|---|---|
| `trace` | a printed dotted line (`dash` px at 1080 apart, 14) traced in crayon (`width` px, 10), a share of it a step, the crayon itself riding the tip (`tip`, true) |
| `dots` | `count` numbered dots (16) round the outline, `per` joined a step (default as many as finish in its steps), `dot` px; when the last meets the first the picture appears in the outline (`reveal`, true): its sprite, googly eyes and all, or the shape coloured in; `numbers` (true) |
| `paint` | colour by number: a sprite cut into regions by its colours (k-means into `colors`, 5; a region's number is its colour's, 1 the commonest) or a shape into `count` wedges (6) in `colors` (a list; the palette's accents), each region's number printed in it, `per` coloured in a step in its colour (a sprite's true colours); `legend` (true) the key of swatches under it |

`kind = "label"` sets tiny editorial type beside the picture (`mkmmd/core/labels.py`, `mkmmd/labels.py`): each label's
own short text (never words of the song) typed out from its `t`, then a dotted leader drawn from it to the point it names
and a dot set there. `labels = [{text, at, place | offset, t, align, color, size}]`: `at` is `[x, y]` of the frame or
`{point = "<expression>"}` projected through the camera of the shot in the cut (as a doodle's marks, `point/x<i>l<j>/`),
`offset` the text's anchor from the point in frame heights (`[0.07, -0.06]`) or `place` it in the frame, `align` `left`
(the text runs right from its anchor) or `right` (default: away from the point); the text slides in to stay 0.025 frame
heights inside the frame in every output. The entry's `font` (a small grotesk or typewriter face of the registry: the
registry's first `mono` by default, as a camcorder's), `size` (cap height, frame heights, 0.02), `color` (`text`),
`tracking` (px at 1080 between letters, 1), `leader` (`{color, dot, gap, end}`: 2.2 px dots 6 px apart, a 3.5 px dot at
the point; `false`: none), `box` (`{color, pad, alpha}`: a paper box behind the text; default none), `type` (s to type a
label out, 0.25) and `draw` (s to draw its leader, 0.2).

```toml
[[fx]]                        # the activity page under her crayon drawing: her outline traced, a veggie's dots, a colour by number
kind = "activity"
from = 0.0
to = 6.0
lane = "notes"
key = "base"                  # the paper of the shot's crayon sketch
font = "overpass_bold"
objects = "Reisen*"
items = [{ kind = "trace", shape = "figure", offset = 34, color = "iris" },
         { kind = "trace", shape = "star", at = [0.13, 0.3], size = 0.26, color = "gold" },
         { kind = "dots", sprites = { tag = "veggie" }, eyes = true, at = [0.86, 0.3], size = 0.3, count = 12, color = "pine" },
         { kind = "paint", sprites = "butterfly_sulphur_orange", at = [0.85, 0.73], size = 0.28, colors = 4 }]

[[fx]]                        # the era of the shot, beside her head
kind = "label"
from = 6.0
to = 8.2
font = "overpass_semibold"
labels = [{ text = "1967", at = { point = 'bone("head").head' }, offset = [0.12, -0.1] },
          { text = "SIDE A", at = [0.78, 0.8], place = [0.84, 0.92], t = 6.6 }]
```

#### Toy paint app

`kind = "paintapp"` lays a children's paint program over the frames, "editing" the video live (`mkmmd/core/paintapp.py`
checks, times and lays it out, `mkmmd/paintapp.py` draws it; all its art is drawn in code from the palette). Its chrome: a
border round the canvas (the picture, its edges under the border), the ink line along the canvas's edge and the shade it
casts in, a chunky toolbar of six buttons (pencil, crayon, stamp, bucket, eraser, undo: a face over a lip, thick ink
outlines, the tool in hand a pastel of its colour, pressed half down) and a palette strip of swatches (the colour picked
larger and ringed in white), laid out per output: the toolbar on the left of a wide or square frame with the strip along
the bottom, at the bottom of a tall one with the strip above it. A cursor (a gloved hand or a pencil) glides from stop to
stop on a gentle arc, eased, arriving on time, and clicks there (pressed down and springing back past rest, a ring
spreading where it clicked). Its stops are the places its tools act on (where a stamp lands, a fill's point, an eraser's
centre or strokes, the undo button), its `keys`, and on its way the buttons and swatches it picks: before an action whose
tool is not in hand it clicks the tool's button, before a fill whose colour is a swatch not picked the swatch, when 0.7 s
since the click before allow it (else the tool or colour just changes as it acts). Everything is a function of the clip
second and the `seed`: frames come out the same in any order, in `mk look` and `mk post`.

| Key | Meaning |
|---|---|
| `toolbar`, `scale` | the side the toolbar stands on: `auto` (default: the bottom of a frame 1.2 times taller than wide, else the left), `left`, `right`, `top` or `bottom` (a panel too long for its side is shrunk to fit); `scale` the chrome's size (1: px at 1080 on the short side, buttons 104, swatches 58) |
| `border`, `panel`, `ink`, `swatches` | the chrome's colours: the border round the canvas (`gold:base:0.45`), the panels and the idle buttons (`surface`), the outlines (`text`); the strip's swatches (`love`, `rose`, `gold`, `pine`, `foam`, `iris`, `text`, `base`; at most 12) |
| `ui` | `false`: no chrome, the cursor and the paint alone (default true) |
| `cursor` | `{kind, size, keys, travel}` or `false` (none): `hand` (default) or `pencil`, its length `size` short sides (0.1), its hot spot the fingertip or the point; `keys` `[{t, at, click}]` more stops (`at` frame fractions; `click = true` clicks there); `travel` seconds it glides to a stop (0.3, at most 0.8 of the time since the last) |
| `stamps` | `{sprites, times or lane (min_db, every), size, at, angle, life, fade, scroll, pop, sparkle, edge, shadow, eyes, mouth}`: on each hit a sprite is stamped where the cursor clicks. `sprites` (required): registry slugs, files, globs or `{tag = ...}`, as a mandala ring's, dealt shuffled by `seed`, every one before any comes again; `eyes`, `mouth`: googly eyes and a smile as any sprite source's. `size` its longer side, short sides (0.16); `at` `[x, y]` frame fractions or a list used in turn (default: spread over the canvas, never bunching); `angle` it lands turned within that either way (12 degrees), settling from a little turn; `pop` s it pops in over (0.3, a 15 % overshoot) with a burst of sparkles (`sparkle`, true); it stays, or fades out over `fade` s (0.25) at the end of its `life` s; `scroll` `[dx, dy]` short sides a second it drifts after it lands (`[0, 0]`); a sticker's `edge` (`{color = "#ffffff", width = 8}`) and `shadow` (`{color = "text", alpha = 0.3, blur = 6}`), `false` for none |
| `bucket` | `{at, times or lane (min_db, every), color, tolerance, grow, life, fade, opacity}`: on each hit the picture's colour region round a point is flooded: every pixel joined to it through pixels within `tolerance` (0.1 on each channel) of its colour, on the canvas as it stands (the stamps and fills before it on it), spreading out from the point over `grow` s (0.3, at least 0.1) with a soft, wandering front. `at` (required): `[x, y]` frame fractions or `{point = "<expression>"}` (projected through the shot in the cut, `point/x<i>b<j>/`), or a list used in turn; `color` one or a list read fill by fill (`gold`, `love`, `foam`, `iris`, `rose`, `pine`); `life`, `fade` as a stamp's; `opacity` (1). The region is found again on every frame from its point, so the paint keeps to it as the picture moves: pin the point to the scene (`{point}` on a wall) where the camera or the figure moves |
| `eraser` | `{at, dur, mode, center, pieces, width}`, or a list of them: a transition across the cut at `at` (clip seconds; the frame before it in the clip): the outgoing picture on the frame before `at`, as it reached the effect with that frame's stamps and fills on it, is held and taken off the incoming frames over `dur` s (0.5, at least 0.2; inside the span), revealing them. `mode` `confetti` (default): the held picture is cut into about `pieces` (14) cells along the short side, jittered quads and triangles, that lift off and fly out from `center` (`[0.5, 0.5]`) nearest first, kicked up, falling, spinning and fluttering (their backs a flat accent), shrinking away by the end; `wipe`: the cursor rubs it out with an eraser block, back and forth strokes `width` short sides wide (0.3) scrubbing up and down, the strokes' trail revealing the shot. The paint on the canvas before `at` goes with the held picture; erasers may not overlap |
| `undo` | `{times or lane (min_db, every), back}`: each takes back the last stamp or fill still on the canvas (none: it only clicks), a stamp's pop or a fill's spread played in reverse over `back` s (0.3); the cursor clicks the undo button |
| `seed` | the stamps' deal, angles and sparkles, the fills' fronts, the confetti (1) |

Flash-safe: no more than three big changes of the picture fall in any second, fills landing or leaving (taken back or
faded) and erasers starting (WCAG's general flash threshold), else the entry is an error before anything is drawn; a fill
spreads over at least 0.1 s and an eraser takes at least 0.2. `strength` mixes everything the app draws over the picture.
The frame an eraser holds is read as a film's held frame is: `mk post` keeps it as it goes through the frames, `mk look`
renders it beside the frames asked for (with the points of the fills on it). A paint app's stamps count for `mk assets
credits` and `mk export web` as a mandala's sprites do. A 1080p frame takes about 0.05 s, up to 0.3 s with fills and the
confetti. `--gpu` draws a paint app that is only stamps (`ui = false`, `cursor = false`, no `bucket`, `eraser` or `undo`)
on mkr-post, each stamp a sticker as a mandala's are ([Post](#post)); a 1080p frame with a dozen stamps is about 20 ms
there for 60 ms on the CPU, the eyes' pupils (the wobble of each worked out per frame by the workers, as the CPU does)
adding 80 ms to either.

```toml
[[fx]]                        # the paint app over the drop: veggies stamped on the kicks, the wall filled, undo, an eraser
kind = "paintapp"
from = 0.0
to = 6.0
stamps = { sprites = { tag = "veggie" }, eyes = true, lane = "kick", min_db = -10, life = 3.0 }
bucket = { times = [2.44], at = { point = "(-0.5, 1.2, 1.5)" }, color = "gold" }   # a point on the wall behind her
undo = { times = [4.6] }
eraser = { at = 5.33, dur = 0.5 }

[[fx]]                        # no chrome: a pencil rubs the shot out across the cut
kind = "paintapp"
from = 10.9
to = 11.6
ui = false
cursor = { kind = "pencil" }
eraser = { at = 11.14, dur = 0.4, mode = "wipe", width = 0.4 }
```

#### Era looks: digicam, chrome, gradmap, invert, test card, cards

The picture hopping through the eras of pop, and the motifs of a TV station. `mkmmd/core/eralook.py` checks, times and lays
them out (numpy, tested), `mkmmd/eralook.py` draws them (numpy, OpenCV and Pillow); their randomness is seeded by `seed` and
the entry, so `mk look` and `mk post` draw the same frames. With the `era` presets of `film` and `camcorder` ([Picture
effects](#picture-effects-pixel-sort-film-camcorder-doodles-photo-booth)) every era is one key: `film` with `era = "1967"` (a
Technicolor print), `camcorder` with `era = "1985"` (a VHS tape), `digicam` (2005) and `chrome` (now).

- `digicam`: a 2005 point-and-shoot. Its lens (barrel distortion, purple fringes round the clipped lights), its sensor's low
  resolution and noise, its white balance off and hunting, its hard tone curve and heavy sharpening, a JPEG's 8 x 8 blocks
  and an orange date stamp in seven-segment digits.
- `chrome`: now, hyperpop. Iridescent liquid metal (on the figure alone with `mask = "figure"`), a glossy bloom spilling
  past its outline, sparkle stars twinkling on it and Y2K bubbles floating up.
- `gradmap`: the picture's luma onto a list of colours, the next map on each cut of the cut list (or each hit): the same
  shot comes back regraded.
- `invert`: the picture inverted (its hues turned back toward the palette with `hue`), eased in and out, over a window or
  on hits, never more than three big flashes a second.
- `testcard`: an original homage to the layout of 1967's Test Card F: the scene in a circle centred on a point, colour bars,
  castellations, a grey scale and gratings round it on a ruled grey ground, small labels (`PLEASE STAND BY`).
- `cards`: TV colour-bar cards flying in on hits, tumbling in 3D.

| Key | Meaning |
|---|---|
| `noise`, `sharpen`, `halo`, `jpeg`, `pixel` | digicam: CCD noise (0.05: grainy, strongest in the shadows, its colour blotchy); the camera's unsharp mask `sharpen` (1.2) of radius `halo` px at 1080 (2): bright and dark halos either side of every edge; the JPEG's quality `jpeg` (1..100, 45); the sensor's pixel, `pixel` px at 1080 across (2): the picture is taken to the sensor's size, its noise, white balance, curve, sharpening and JPEG made there, and back (its blocks 8 x `pixel` px at 1080; a smaller preview frame is taken up to it, so it shows the blocks a full frame does) |
| `fringe`, `barrel`, `wb`, `hunt`, `contrast` | digicam: purple fringes just outside the clipped lights and the red and blue planes apart toward the corners (0..1, 0.6); barrel distortion (0.04; below 0 a pincushion), the picture zoomed so its corners stay filled; the white balance off by `wb` (-1 cool and a little green .. 1 warm amber, -0.3) and hunting about it slowly by up to `hunt` (0.25), as auto white balance does; a hard digital curve `contrast` (1.2), its lights clipping |
| `date`, `corner`, `color`, `size` | digicam: the date imprint, seven-segment digits drawn by mk (no font): digits, `'`, `/`, `.`, `:`, `-` and spaces (`"'05 10 28"`; `false`: none), in its `corner` (`bottom-right`, `bottom-left`, `top-left`, `top-right`), `color` (`#ff7a1a`, glowing a little), its digits `size` short sides tall (0.045), drawn over the JPEG |
| `mask`, `objects` | chrome, gradmap, invert: `figure` works only inside the coverage of `objects` (the figure layer `mk render` draws beside the frames, as a pixel sort's), `ground` only outside it; both or neither |
| `sheen`, `iris`, `bands`, `colors`, `speed` | chrome: liquid metal mixed over the picture by `sheen` (0..1, 1): a sky over a dark horizon and the ground, reflected in `bands` (2.5) that follow the picture's shading (its lights stay bright, its shades run into the bands) and wrap round the figure's outline (a puffy dome in from the mask's outline; without a mask the luma is the relief); tinted round the loop of `colors` (thin-film colours, each at its full brightness; `["rose", "foam", "iris", "gold"]`) by `iris` (0.6), the colours sweeping across it as on a holographic foil; the bands moving on `speed` bands a second (0.1) |
| `bloom`, `sparkles`, `bubbles` | chrome: a glossy bloom of its lights (0.5), spilling past the outline; `sparkles` (8) four-pointed stars twinkling on it (on the figure, with a mask), each from nothing to nothing over about 0.7 s and back somewhere else; `bubbles` (5) glossy Y2K bubbles floating up over the picture and coming in again at the foot, a clear body, an iridescent rim and white highlights |
| `times`, `lane`, `min_db`, `every` | chrome: on each hit the sheen swings 0.15 bands forward and back over 0.4 s and every sparkle flashes at once, fading over 0.45 s. gradmap: it steps on the hits instead of on the cuts. invert: an inversion on each hit (below). cards (one of `times` or `lane` required): each hit sends cards in |
| `maps`, `offset` | gradmap: the maps, each the luma onto 2 to 8 colours, dark to light and evenly spaced, the name of one of mk's maps made from the Dawn palette, its lights going to its cream (`teal_lime`, `pink_purple`, `sepia_orange`, `blue_gold`; the default is the four), or `"none"` (the picture as it is). On each cut of the cut list after the span's first frame (a shot of the cut starting inside the span; plates are not cuts), or on each hit when it has `times` or a `lane`, the next map comes in, round and round, starting on map `offset` (0): the same shot comes back regraded. Two shots ping-ponging with an even number of maps meet only half of them each: give an odd number (add `"none"`) |
| `hue`, `in`, `hold`, `out`, `rate` | invert: the picture inverted, its hues then turned `hue` degrees (0; 180 turns them back: Dawn's pinks stay pink and its cream goes dark), mixed in by `strength`. Over a window it eases in over `in` s (0.05) from `from` and out over `out` s (0.18) to `to`; with hits each one eases in over `in`, holds `hold` s (0.12) and eases out over `out`. Flash-safe (WCAG's general flash threshold): no more than `rate` hits (0.1..3, 3) start an inversion in any second, a hit while `rate` kept ones began in the second before it being skipped, and a window whose `strength` keys would flash more than 3 times in a second is an error |
| `center`, `at`, `radius`, `zoom` | testcard: the scene inside a circle `radius` short sides (0.36) about `at` (frame fractions, `[0.5, 0.5]`), moved so that `center` sits in its middle (`[x, y]` of the frame, the middle by default, or `{point = "<expression>"}`, a scene point projected through the camera of the shot in the cut on every frame, as a lens's), `zoom` times its size (1), with a white rim |
| `grid`, `bars`, `ground`, `line` | testcard: round the circle a grey ground (`ground`, `muted`) ruled in squares, `grid` (9) to the short side, in `line` (`base`); castellation blocks round the edge, half a square long (the palette's accents between white along the top, black and white down the sides, the right's half a block off the left's, and along the foot); colour bars across the top (`bars`: `dawn`, the palette's white and accents brightest first; `true`, 75 % bars; `both`, a row of each); a six-step grey scale and six gratings, coarse to fine, flanking the circle (left and right in a frame as wide as it is tall or wider, above and below in a tall one, as wide as the room there; left out without room); diagonal stripes in its corners and arrowheads in the middle of its edges |
| `labels`, `font`, `open` | testcard: up to 4 short labels (28 characters each; `["PLEASE STAND BY"]`, `[]` none), the first in a box on the circle's foot, the others small in the lower corners and under the bars, in `font` (as a camcorder's: a registry slug or a file, default the registry's first `mono` font; found before any frame is drawn); `open` seconds (0) at its end over which the circle grows, slowly then fast, until it covers the frame, the labels fading: the card blows open into what is behind it |
| `count`, `size`, `life`, `spin`, `path` | cards: on each hit `count` (1) cards 0.06 s apart, each flying for `life` s (1) and tumbling `spin` turns a second (1.25) about an axis of its own; `path` `in` (from far behind the middle of the frame at the lens, growing, veering off past an edge) or `across` (from beyond one side to beyond the other at a middle distance, on a gentle arc); a card stands `size` frame heights tall (0.32) at the middle of its depths, through a lens of about 42 degrees down the frame |
| `style`, `shadow` | cards: a 4:3 card with rounded corners and a little thickness, the colour bars of `style` (`dawn`, `true`, `both`: below) on its front, a card's back on its back, cream edges, lit from the upper left with a glint; a soft drop `shadow` (0..1, 0.3) under each; `strength` is their opacity |

The colour bars are `mkmmd/core/colorbars.py` (numpy only, so a prop builder draws with it inside Blender):
`texture(width, height, palette, style)` is a card's face, (height, width, 3) display-space RGB, row 0 its top: seven bars over
two thirds of it, the castellation strip under them and SMPTE's foot (the -I and +Q patches, a white and the pluge), in true 75 %
colours (`true`), in the palette's (`dawn`: its lightest neutral and its accents by luminance, brightest first, as true bars
run) or the palette's bars over a row of the true ones (`both`); `back(width, height, palette)` a card's back (the palette's
white ruled in squares, a ring, its accents along the foot). The `library:colorbar_card` prop lays them on its faces.

```toml
[[fx]]                        # the same shot comes back regraded on every cut (five maps: each shot meets them all)
kind = "gradmap"
from = 0.0
to = 6.76
maps = ["teal_lime", "pink_purple", "none", "sepia_orange", "blue_gold"]

[[fx]]                        # 2005: the point-and-shoot, its date stamped orange
kind = "digicam"
from = 6.76
to = 8.24

[[fx]]                        # now: chrome on her, the sheen swinging and the sparkles flashing on the kicks
kind = "chrome"
from = 8.24
to = 9.70
mask = "figure"
objects = "Reisen*"
lane = "kick"

[[fx]]                        # 1967, one key
kind = "film"
from = 9.70
to = 11.14
era = "1967"

[[fx]]                        # 1985, one key
kind = "camcorder"
from = 11.14
to = 12.52
era = "1985"

[[fx]]                        # the world inverts on the drop's hits, its hues turned back to Dawn's, three a second at most
kind = "invert"
from = 11.14
to = 12.52
lane = "kick"
hue = 180

[[fx]]                        # please stand by: her face in the circle, the card blowing open into what follows
kind = "testcard"
from = 15.33
to = 16.74
center = { point = "bone('head').head" }
font = "overpass_semibold"
open = 0.25

[[fx]]                        # colour-bar cards tumbling through on the kicks
kind = "cards"
from = 16.74
to = 20.5
lane = "kick"
count = 2
```

#### Lasers

`kind = "lasers"` lays a festival's lasers over the picture: fans of thin bright beams cutting through haze, sweeping and
fanning on the hits. `mkmmd/core/lasers.py` checks and times them (numpy-free, tested), `mkmmd/lasers.py` draws them (numpy,
OpenCV): each beam an antialiased line in its colour, fading along its length, a glow round it and the light a slow smoke
of haze scatters round the fans, all of it light added to the picture (or screened), its brightest channel held to 1 with its
hue kept and what lies past that turned white, so an overexposed beam's middle runs white-hot while its edges keep their
colour. Everything is a function of the clip second, its hits and its `seed`: any frame comes out the same in any order.
About 0.15 s for a 1080p frame.

| Key | Meaning |
|---|---|
| `origin` | where the fans leave from: `[x, y]` frame fractions (default `[0.5, 1.0]`, the bottom middle; it may lie off the frame, a bank of lasers under the stage), `{point = "<expression>"}`, a point of the scene projected through the camera of the shot in the cut on every frame of the span as a `center` is (`point/x<i>o<j>/`: a headlamp, a speaker stack; the fan rides it), or a list of up to 16 of them, a fan from each |
| `count`, `spread`, `aim` | a fan's beams (7), spread evenly over `spread` degrees (50) about its `aim`, degrees counter-clockwise on screen from the right (90: straight up) |
| `sweep`, `fan`, `rate` | the fans swing `sweep` degrees (30) either side of the aim and breathe, closing to 1 - `fan` (0.5) of their spread. Without hits they swing steadily, `rate` swings a second (0.25), widest through the middle of each swing and closing toward its ends |
| `mirror` | `true` (default): every other fan is the one before it reflected left for right, its aim about the vertical and its sweep the other way, so two headlamps' fans cross and part; `false`: all alike |
| `times`, `lane`, `min_db`, `every` | optional hits (clip seconds, or a lane of the `[audio] hits` file, as a zoom's): on each the fans swing over to the other side, are thrown wide open and light up, then close and dim; their colours step on. Flash-safe: a hit sooner than 1/3 s after the last one followed is skipped, so the pulses, swings and colour steps come three times a second at most |
| `snap`, `pulse`, `decay` | on a hit: the swing takes `snap` s (0.18, eased in and out); `pulse` (0.4) is the share of the beams' light that comes with each hit (between hits they glow at 1 - `pulse`), rising in 0.03 s and dying with the fans' opening as exp(-age / `decay`) (0.35 s) |
| `colors`, `by` | a colour or a list (`["love", "foam"]`), read round and round from the hits so far plus the fan's number (`by = "fan"`, default: a fan in one colour, as a laser projector shows one at a time) or the beam's within its fan (`"beam"`: a rainbow fan) |
| `width`, `glow`, `haze` | a beam's core, px at 1080 on the short side (2.5); the glow round it, its strength at the core (0.6, a Gaussian of 8 px at 1080); the haze's scatter (0.35): a wide glow (30 px at 1080) round the fans, thick and thin in a slowly drifting smoke, running softly up to 0.6 where a fan's beams meet |
| `length`, `fade` | how far the beams reach, frame heights (4: past the frame), and how much of their light they lose by the end (0.35) |
| `blend` | `add` (default: light added, the frame's lights add up) or `screen` (softer on a bright picture) |
| `mask`, `objects` | `ground` keeps the beams off the figure's coverage (she stands in front of them), `figure` on it only; both or neither. `strength` mixes the light in (keys fade the show in and out) |
| `seed` | the haze's smoke (1) |

```toml
[[fx]]                        # lasers out of the car's headlamps, swinging over on every hit of the bass, thrown open and lit
kind = "lasers"
from = 0.0
to = 3.0
origin = [{ point = "obj('car_headlamp_L').loc" }, { point = "obj('car_headlamp_R').loc" }]
lane = "chug"
aim = 70                      # up and out: the mirrored fan aims 110
spread = 36
sweep = 28
count = 6
colors = ["love", "foam", "iris"]

[[fx]]                        # a stage's bank under the frame, behind her, one rainbow fan breathing slowly
kind = "lasers"
from = 8.0
to = 16.0
origin = [0.5, 1.15]
count = 12
spread = 80
by = "beam"
colors = ["love", "gold", "foam", "iris"]
rate = 0.2
mask = "ground"
objects = "Rin_*"
```

#### Prints: photographs laid over the picture

`kind = "prints"` deals real photographs over the picture as physical prints, one landing on each hit: it falls from above,
larger and turning, gathering speed, lands on the hit, hops a little and settles, its drop shadow growing and softening while
it is in the air. Each is the photograph whole on a cream card (a mounted print, its border deeper at the foot; or a
stereocard, both halves of a stereograph side by side under arched tops), held by two tape strips across its corners or by
pushpins pushed in as it lands, turned a little. A print never cuts anything out of its photograph and never retouches it:
the photograph is only resized (and toned when `tone` says); give it whole photographs, not cut-outs. `mkmmd/core/prints.py`
checks, times and lays them out (numpy, tested), `mkmmd/prints.py` draws them (numpy, OpenCV, Pillow): each print is a tile
drawn once at its size (the card's grain and aged edges, a hairline where the photograph meets it, the tape) and laid by an
affine map. Everything is a function of the clip second, the hits and the `seed`: any frame comes out the same in any order.
About 0.13 s for a 1080p frame of eight prints. Print k shows photograph k of the deal on place k mod `count`, over what
already lies there; without `times` or a `lane` every place holds a print from the span's start.

| Key | Meaning |
|---|---|
| `mode` | `scatter` (default: spread over `region`, each place the candidate farthest from those before it), `ring` (`count` places evenly round `center`, the ring turning) or `pin` (a `base` print laid large and the prints pinned over its `region`: a scrapbook page) |
| `photos` | required: registry slugs, files, globs or `{tag = ...}` (one tag or a list), one or a list, as a mandala's sprites; a slug, tag, file or glob that finds nothing is an error before anything is drawn |
| `order`, `seed` | `given` (default: as listed, a tag's by slug, round and round) or `shuffle` (anew each time round, every one before any comes again); `seed` (1) also turns, tapes and scatters them |
| `times`, `lane`, `min_db`, `every` | the hits a print lands on, one each (clip seconds, or a lane of the `[audio] hits` file, as a zoom's) |
| `count`, `size` | places (scatter 6, ring 8, pin 5; at most 24) and a card's longer side, short sides of the frame (0.3, 0.24, 0.16) |
| `style`, `card` | `print` (default) or `stereo`; `card = {color, border, foot, arch}`: its colour (`"gold:#f8f2e4:0.85"`, cream warmed by the palette's gold), its border round the photograph (0.06 of its shorter side), the foot under it (1.8 borders) and a stereocard's arches (their corners' radius, 0.3 of a window's width) |
| `fix`, `tape`, `pins` | `tape` (scatter and ring), `pins` (pin) or `none`; `tape = {colors, alpha, width, length}` (`["gold:#fffdf6:0.82"]` round the prints, 0.5, 0.14 and 0.34 of the card's shorter side; the strips cross two corners, square to the diagonal, 0.45 of their length on the card); `pins = {colors, size}` (heads in `["love", "pine", "gold", "iris", "foam"]` round the prints, 0.018 short sides across) |
| `shadow` | a collage's `{color, offset, blur, alpha}` (on by default, `false`: none): the shadow at rest, cast by the card |
| `tilt`, `spin`, `drop`, `settle`, `lift` | each print turned within ±`tilt` degrees (7) at rest; it falls over `drop` s (0.16) from 1 + `lift` (0.4) times its size, turning up to `spin` degrees (14) and sliding in from up to a quarter of its size away, lands with 0.3 of that left, then settles over `settle` s (0.3), easing a little past its rest and back |
| `tone`, `toning` | `own` (default: the photograph as it is), `albumen` (its luma through the tones of Alfred A. Hart's albumen prints as the Library of Congress scanned them: a copy print sits with them), `palette` (the palette's darkest neutral, its `gold`, its lightest neutral: it breathes with the palette) or a list of colours, dark to light; `toning` mixes it in (1) |
| `center`, `radius`, `turn`, `phase` | ring only: its centre (`[x, y]` or `{point}`, the middle), radius (0.32 short sides), degrees a second it turns counter-clockwise (6) and its first place's angle from straight up (0) |
| `region` | scatter: `[x0, y0, x1, y1]` frame fractions the prints lie in (`[0.03, 0.04, 0.97, 0.96]`), half a card inside it; pin: fractions of the base photograph (its foreground, `[0.04, 0.66, 0.96, 0.99]`) |
| `base` | pin only (required): `{photo, at, size, angle, t, tone}`, the photograph (a slug or a file), its card's middle (`[0.5, 0.5]`), longer side (0.95 short sides), turn (-1.2 degrees), the clip second it lands (none: there from the start) and its own tone (the entry's) |

A stereocard shows the two halves of a stereograph when the photograph's registry entry names the other in `extra.stereo`
(`{"right": FILE}` on a left half, `{"left": FILE}` on a right one), else its half twice. The photographs count for `mk
assets credits` and `mk export web` as a mandala's sprites do. An archive's stereograph is prepared as one half of the pair
cropped from the master scan inside the photograph's edges, its levels and colour as scanned; note the source, the crop and
the other half in the entry's `extra`.

```toml
[[fx]]                        # photographs of the crews dealt round her head on every other hit, the ring turning slowly
kind = "prints"
from = 4.0
to = 9.0
mode = "ring"
photos = { tag = ["photo", "crew"] }
center = { point = "bone('head', 'Rin_arm').center" }
lane = "chug"
every = 2

[[fx]]                        # stereocards strewn over the frame on the accents, in the palette's tone
kind = "prints"
from = 9.0
to = 12.0
photos = "photos/*.png"
style = "stereo"
size = 0.42
tone = "palette"
times = [9.1, 9.9, 10.7, 11.5]

[[fx]]                        # a scrapbook page: one photograph laid large, the crews' prints pinned over its foreground
kind = "prints"
from = 24.7
to = 29.6
mode = "pin"
base = { photo = "my_base_photo", tone = "albumen" }
photos = { tag = ["photo", "crew"] }
order = "shuffle"
lane = "chug"
every = 3
```

#### Keys pushed too far

`kind = "key"` keys a colour out of the picture and pushes the key past clean, so what lies behind burns through in
colour fringes and crushed steps, as a video key turned up until it breaks and its planes pushed about. `mkmmd/core/colorkey.py`
checks it and times its tolerance (tested), `mkmmd/colorkey.py` draws it per pixel: how far each pixel's colour lies from
the key colour, a matte for each of the red, green and blue planes at its own tolerance, crushed into steps, the trail of
the frames before, and what shows through laid in plane by plane. What shows through is one of a video (`clip`, read and
fitted to cover the frame by `mkmmd/footage.py` as footage's, credited as footage is), a flat colour (`back`) or the cut's
own frame `delay` frames before. A pixel within `tolerance` of the colour is keyed fully, one past `tolerance` + `soft`
not at all; with `split` the red plane keys wider and the blue narrower (or each plane by its own offset), so the edge of
the key comes apart into fringes of the picture's own planes, and `bands` crushes the matte into hard steps. A `burn`
keeps the key of the cut's frames before (keyed at their own tolerance, as an echo reads them: without the effects
before it in the table), frame j back weighing `fade` ** j: moving edges smear into one another. The tolerance moves
with its keys, with a stem's loudness (`follow`: the world pouring in while the voice sings) and on the hits (`pump`).
About 0.1 s a 1080p frame, 0.25 s with a spill, a mask and a trail on the CPU; `--gpu` draws it on mkr-post
([Post](#post)), a pass a frame of the trail more.

| Key | Meaning |
|---|---|
| `color`, `space` | the colour keyed out (`base`) and how a pixel's distance from it is measured: `rgb` (default: the Euclidean distance over sqrt 3, 0..1), `chroma` (in the BT.709 Cb Cr plane alone, brightness left out, as a chroma keyer: a lit and a shaded green key alike) or `luma` (brightness alone: key the dark and only the lights are left) |
| `tolerance`, `soft` | keyed fully within `tolerance` (0.15; a number or `[[t, v], ...]` keys over clip seconds, as a `strength`), not at all past `soft` (0.1) further, a hard step with 0 |
| `split`, `bands` | each plane its own tolerance: a number s (0) keys red s wider and blue s narrower, `[r, g, b]` an offset each (-1 .. 1); `bands` crushes each plane's matte into that many equal steps (0: smooth, up to 64) |
| `clip`, `in`, `rate`, `loop`, `focus`, `mirror` | what shows through: a video (a file or an asset slug), as footage's keys, always fitted to cover the frame |
| `back`, `delay` | or a flat colour, or the cut's frame `delay` frames before (1 .. 600; the clip's first frame stands in before it): one of `clip`, `back`, `delay` (required) |
| `gain` | `[r, g, b]` multipliers (0 .. 8, `[1, 1, 1]`) on what shows through before it is laid in: push a plane and the burn takes a cast |
| `spill` | a colour, or `{color, amount, reach}`: what is left of the picture near the key colour stained toward it, by `amount` (0.5) where it all but keys, nothing `reach` (0.15) past `tolerance` + `soft` |
| `burn`, `fade` | the key of the `burn` (0 .. 60, 0) cut frames before kept as a trail, frame j back at `fade` ** j (0.7) |
| `follow` | `true` or `{stem, amount, smooth, range, timeline}`: the tolerance grows by `amount` (0.2) times a stem's loudness (`vocals`; `drums`, `bass`, `other`, `mix`, from the stems cached when `timeline`, `audio/timeline.json`, was analysed), 1 at the span's loud passages (their 95th percentile) and 0 `range` dB (24) under them, smoothed over `smooth` s (0.12) |
| `pump`, `decay`, `times`, `lane`, `min_db`, `every` | the tolerance grows by `pump` (-2 .. 2) on each hit, rising over 0.02 s and dying as exp(-age / `decay`) (0.25 s), the strongest counting; a pump needs hits and hits need a pump |
| `mask`, `objects` | `ground` keeps the key off the figure (its coverage: the figure is left exactly as it was, though it wears the key colour), `figure` keys inside it alone |
| `spread`, `center` | the key opens from a point: each pixel's tolerance is `spread` (0 .. 20, 0: none) lower for every frame height it lies from `center` (`[x, y]` frame fractions or `{point = "<expression>"}`, projected on every frame as an effect's centre is; the middle by default; only with a spread), so a flat ground keys first round the point and the burn grows out from it as the tolerance rises, its front coming apart in the `split` and the `bands` |

```toml
[[fx]]                        # the cream fails round her head: the jungle opens out from it, wider on every downbeat
kind = "key"
from = 7.38
to = 14.77
color = "base"
clip = "pexels_29780524"
tolerance = [[7.38, 0.0], [9.23, 0.07], [11.08, 0.14], [12.92, 0.22], [14.5, 0.34]]
spread = 0.3
center = { point = "bone('head', 'Rin_arm').center" }
soft = 0.025
split = 0.012
bands = 4
burn = 5
gain = [1.12, 1.0, 1.18]
follow = { stem = "vocals", amount = 0.02 }
pump = 0.05
times = [7.38, 9.23, 11.08, 12.92]
mask = "ground"
objects = "Rin*"
```

#### Shapes

`kind = "shapes"` redraws the picture as flat shapes, a circle, square, triangle or diamond in each cell of a lattice,
each in the colour of the picture under its cell: a figure comes apart into a mosaic of shapes, and with the cell
shrinking on the hits it assembles out of them. `mkmmd/core/shapes.py` checks it, times the cell and deals the shapes
(tested), `mkmmd/shapes.py` draws it: each pixel finds its cell in closed form and draws that cell's shape as a signed
distance antialiased over a pixel, a shape never past the circle its cell's shorter side holds (whatever its turn, and
while it pops). The cells are `cell` short sides apart in offset rows (`hex`) or a square `grid`, centred on the
frame's middle; each shape's colour is the picture averaged over about a cell at its middle, or one of `inks`. On a
flat `back` a cell of the back's own colour draws nothing (`skip`), so a figure on a plain ground comes apart and the
ground stays empty; over the picture (`back = "picture"`) the picture shows between the shapes. Everything is a
function of the clip second, the hits and the `seed`: any frame comes out the same in any order. About 0.05 to 0.2 s a
1080p frame on the CPU; `--gpu` draws it on mkr-post ([Post](#post)).

| Key | Meaning |
|---|---|
| `cell`, `layout` | the cells' spacing, short sides (0.05, 0.004 .. 1; a number or `[[t, v], ...]` keys over clip seconds); `hex` (default: offset rows sqrt 3 / 2 of a cell apart) or `grid` |
| `times`, `lane`, `min_db`, `every` | with hits the cell changes only on them (its keys read at the last hit, held until the next): the picture refines or coarsens on the beat, each layout a new deal; without them it follows its keys on every frame |
| `shape`, `size`, `turn`, `spin`, `seed` | `circle`, `square`, `triangle` (a point up), `diamond` or a list to deal from (`["circle", "triangle", "square"]`); a shape's size, a share of its cell's inscribed circle (0.92); turned at random within ± `turn` degrees (0) and spinning `spin` degrees a second (0); the deal (1) |
| `inks`, `pick` | colours the shapes take instead of the picture's: `nearest` (default: each cell's colour snapped to the nearest ink), `shape` (shape j of `shape` in ink j, round and round) or `deal` (dealt at random) |
| `back`, `skip` | what lies between the shapes: `"picture"` (default) or a colour, a cell within `skip` (0.06, RGB distance over sqrt 3) of whose colour draws nothing |
| `pop`, `stagger` | each layout's shapes pop in over `pop` s (0.15; 0: there at once), out of nothing and 15 % past their size, cell by cell up to `stagger` s (0.1) apart |
| `mask`, `objects` | `figure` keeps the shapes to the figure, `ground` to what is round it: the rest of the picture stays exactly |

```toml
[[fx]]                        # her silhouette assembles out of shapes, finer on every kick: plum circles, rose triangles, pine squares
kind = "shapes"
from = 0.0
to = 7.38
cell = [[0.0, 0.45], [7.38, 0.012]]
lane = "kick"
back = "base"
inks = ["text", "love", "pine"]
pick = "shape"
turn = 25
```

#### Haze: mist, fog, bokeh

Three kinds put air and light between the lens and the picture: a room full of haze, a diffusion filter on the lens,
fairy lights out of focus. `mkmmd/core/haze.py` checks them and lays out a bokeh's lights (tested), `mkmmd/haze.py` draws
them (numpy and OpenCV). Every frame is a function of the entry, its `seed` and the clip second, so any frame comes out
the same in any order; lengths are short sides of the frame, so every output shows the same haze. `strength` scales the
mist's light, the fog's thickness and the lights' level. About 0.15 s (mist), 0.05 s (fog) and 0.03 s (bokeh) a 1080p
frame on the CPU; `--gpu` leaves them to it.

- **`mist`**: a bloom. What each pixel has over `threshold` is spread round it and screened back on, so lights glow and
  whites breathe; only what is over the threshold spreads (a bright ground does not wash out: raise `threshold` until
  only the lights glow). `diffuse` softens the whole picture as a diffusion filter does.
- **`fog`**: sheets of noise drifting across the frame in one to four colours, from where it is thin to where it is
  thick, each further sheet slower and a little off the fog's way. With `floor` it lies toward the ground. With
  `mask = "ground"` it is behind the figure, and only a `veil` of it in front of her.
- **`bokeh`**: discs of light, most of them small, drifting through the frame, twinkling, a little brighter at the rim
  as a lens draws a light out of focus; a light that leaves the frame comes round again outside it, never popping. On
  hits they swell and brighten. With `mask = "ground"` they pass behind the figure, but for the largest of them (`front`).

| Key | Meaning |
|---|---|
| `radius`, `threshold`, `knee` | mist: how far the light spreads, short sides (0.03, 0.002 .. 0.5; it also spreads fainter at 2.5 times that); the brightness (a pixel's brightest channel) over which light spreads (0.7), coming in softly from `knee` (0.15) under it to `knee` over it |
| `diffuse`, `tint`, `blend` | mist: how much of the softened picture is mixed in (0, 0 .. 1); a colour the spread light is multiplied by; `screen` (default) or `add` |
| `colors`, `density` | fog: one to four colours, the thin fog's first (default `rose`); how much of the picture the thickest fog covers (0.5, 0 .. 1) |
| `scale`, `detail`, `layers`, `seed` | fog: the size of its patches, short sides (0.6); octaves of noise in a sheet (3, 1 .. 5); sheets (2, 1 .. 4); the deal (1) |
| `speed`, `angle` | fog and bokeh: the drift, short sides a second (fog 0.03, bokeh 0.02), along `angle` degrees counter-clockwise from the right (fog 0; bokeh 90: rising) |
| `floor`, `veil`, `blend` | fog: how much it thins toward the top of the frame (0: even; 1: none at the top); the share of it in front of a masked figure (0.35); `over` (default) or `screen` |
| `count`, `size`, `colors` | bokeh: the lights (40, up to 400); their radius, short sides, a number or `[smallest, largest]` (`[0.015, 0.07]`; most are small); their colours (`["gold", "rose", "foam"]`) |
| `wander`, `twinkle`, `rate` | bokeh: how far each light's heading strays from `angle` and how much it sways (0.5, 0 .. 1); how deep it dims as it twinkles (0.5) and how often, times a second (0.5) |
| `soft`, `rim`, `front` | bokeh: the edge of a disc, a share of its radius (0.25); how much brighter its rim is (0.35); the share of the lights, the largest, drawn in front of a masked figure (0.15) |
| `times`, `lane`, `min_db`, `every`, `pulse`, `decay` | bokeh: on each hit the lights are `pulse` (0.5) brighter and a quarter of that larger, dying over `decay` s (0.3) |
| `mask`, `objects` | `ground` keeps a fog and a bokeh behind the figure (but for the veil and the front lights) and a mist off her; `figure` the other way round |

```toml
[[fx]]                        # a pink room full of haze, thick at the floor, drifting left to right behind her
kind = "fog"
from = 0.0
to = 33.5
colors = ["rose", "#F7D7DE"]
density = 0.7
floor = 0.6
mask = "ground"
objects = "Reisen*"

[[fx]]                        # fairy lights out of focus, rising, swelling on the snare; a few pass in front of her
kind = "bokeh"
from = 0.0
to = 33.5
colors = ["gold", "#FFE9F0", "foam", "iris"]
lane = "snare"
mask = "ground"
objects = "Reisen*"

[[fx]]                        # and everything glows a little
kind = "mist"
from = 0.0
to = 33.5
radius = 0.035
diffuse = 0.4
```

#### Cell division: mitosis

`kind = "mitosis"` lets a figure divide as a cell does, made cute: she sits in a soft cell of her own (a membrane round
her outline, a little light on its upper left), and on each hit the cell is pulled long, pinches in the middle and
comes apart into two smaller copies of her, each in its own cell, that wobble for a moment; `max` hits later there are
2 ** max of her. `mkmmd/core/mitosis.py` checks it, paces the divisions and lays the cells out (tested), `mkmmd/mitosis.py`
draws it: the figure cut out of the picture by its layer (`objects`, the figure layer `mk render` draws), a copy in every
cell, and the membrane where the blurred outlines of all the copies sum past a level, so two cells still close share one
skin and it pinches as they part. Odd divisions pull apart sideways, even ones up and down, each off true by up to
`tilt`; daughters end `gap` of the figure's own box further apart than they are wide, each 0.68 the size of its mother;
the figure's box on the span's first frame sizes all of it, and the colony is scaled and moved to stay inside `fit` of
the frame. A hit that falls while a cell divides is passed over. With `back = "picture"` her place in the picture is
closed from the ground round her (seamless on a plain ground, a blur on a busy one); best where the whole figure is in
the frame. A function of the entry, its hits and the clip second: any frame comes out the same in any order. About
0.2 s a 1080p frame on the CPU; `--gpu` leaves it to it.

| Key | Meaning |
|---|---|
| `objects` | the figure that divides (required: object patterns, as a collage's) |
| `times`, `lane`, `min_db`, `every` | the hits the cell divides on (none: one cell, never dividing) |
| `split`, `max` | how long a division takes, seconds (0.6, 0.1 .. 10); how many there are at most (4: sixteen cells; 0 .. 6) |
| `gap`, `tilt`, `seed` | the space between daughters, a share of the figure's box (0.06); how far off true each division pulls, degrees (10); the deal of those (1) |
| `fit` | the share of the frame the colony stays inside (0.92) |
| `body`, `cytoplasm`, `alpha`, `membrane` | how far the cell stands round the figure, short sides (0.03); its colour (`rose`) and how much of it covers the ground (0.5); the colour of its skin (`love`) |
| `wobble` | how much a freed cell wobbles, a share of its size (0.12), dying over a quarter of a second |
| `back` | `"picture"` (default: the picture, her place closed) or a colour |

```toml
[[fx]]                        # she divides on the four snares of the last bar: sixteen of her
kind = "mitosis"
from = 44.7
to = 48.4
objects = "Reisen*"
lane = "snare"
cytoplasm = "#FFF4F7"
membrane = "#E84A7A"
```

#### Each output its own: aspect

Any `[[fx]]` entry may give `[fx.aspect.<output>]` tables of its own keys for one output: a footage `focus`, a grid's `cells`, a
figure's `keys`, a booth's `layout`, a `rect`. Each is checked as the entry is, merged over it (`mkmmd.core.framefx.for_output`),
and `mk look` and `mk post` draw every output with its own. It may not set `kind`, `from`, `to` or `objects` (every output draws
one span of one figure), nor read frames, scene points or a figure the entry does not: the layers `mk render` draws are planned
from the entry, so the keys that choose them (a `delay`, a `center` point, a pivot found on the figure) belong to the entry.

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
| `kinetic` | `{show, scale, sx, sy, dx, dy, dxp, dyp, rot, pivot, tracking, weight, opacity, tint, drip, headroom, orbit, bend, wobble}`: motion keyed on node inputs, written by `lyrics` or by hand ([Lyrics](#lyrics), the kinetic keys) |
| `lyrics` | `{timeline, line, words, style, ...}`: one text per sung word, read from the timeline ([Lyrics](#lyrics)) |
| `blocks` | `true` or `{paint, ink, wood, ground, letter, relief, round, rim, sides, pictures, layout, rows, gap, space, view, fall, hits, next, crash, tumble, land, to}`: the text set in 3D toy alphabet blocks, one per letter or digit: a label (with `text`) or the sung words (with `lyrics.style = "blocks"`) ([Block type](#block-type)) |
| `ink` | `{strokes \| track, on, width, lift, color, dry, glossy, from, to}`: handwriting instead of type, a pen's strokes that appear behind the nib; the entry carries only `name` and `ink` ([Ink](#ink)) |
| `screen` | `true` or `{anchor, side, margin, height, width}`: type of the picture, laid over the finished frame of whichever shot is cutting; `at`, `box`, `size`, `offset` are in frame heights, and no `on`, `mount`, `facing` or `up` ([Screen type](#screen-type)) |
| `knockout` | `true`: screen type reversed out of a silhouette shot, in any other shot ordinary screen type ([Screen type](#screen-type)) |
| `negative` | `true`: screen type as a photographic negative: the picture under its glyphs is inverted by their coverage `a` (`out = bg * (1 - a) + (1 - bg) * a`, the text's `color` unused). Needs `screen`; refused with `knockout`, `backing`, `outline` or `blocks`; lyric entries and `lyrics.zones` take it like any key ([Screen type](#screen-type)) |
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
`box`, `font`, `size` / `fit`, `color`, `glow`, `lit`, `depth`, `lift`, `offset`, `outline`, `backing`, `screen`, `negative`, ...) go to every
word (`negative = true` with `screen` makes each a photographic negative of the picture under it: [Screen type](#screen-type), where it is refused with `backing` and `outline`); the words take their text from the timeline, so `text`, `value`, `kinetic`, `reveal` and `lyric` are errors, and so are the
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
| `style`, `arrive` | how a word arrives: `pop` (default), `slap`, `drop`, `rise`, `slide`, `slam`, `type`, `none` or `blocks` (3D toy blocks that drop in: [Block type](#block-type); their look and motion are in `blocks`, and only `timeline`, `line`, `lines`, `words`, `case`, `clean`, `from` and `to` apply); `arrive = {...}` overrides that style's numbers (below) |
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
| `wobble` | `true` or `{amount, sway, tilt, rate, wave, follow, attack, release, min_hold}`: the letters bob and sway on a wave that travels along the word while it is sung, as big as the voice is loud (below) |
| `punch`, `write` | the words that stand out, and the words whose note is held long, take a look of their own (below) |
| `carry` | what a word does that is already up when its block starts (a `from` after its onset, a new shot with `zones`): `land` (default) arrives again, `still` is simply there in its new place |
| `zones` | `{<shot> = {overlay}}`: the line is set again in every shot it crosses (below) |
| `layout` | where the words stand: `same` (default), `flow`, `stack`, `slots`, `ring` or `bleed` (huge words off the edges, tilted in depth), or a table `{kind, ...}` (below) |
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
| `style = "blocks"` | the word in toy alphabet blocks, one per letter, that drop in from above, land on the onset and bounce ([Block type](#block-type)) |

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

**Wobble.** `wobble = {amount, sway, tilt, rate, wave, follow, attack, release, min_hold}` (`true` or `{}` is all defaults;
`mkmmd/core/typewobble.py`): while a word is sung its letters bob on a wave that travels along it from the first letter to the last,
`wave` letters to a cycle (4.5) at `rate` cycles a second (2.2 Hz), each rising and falling by up to `amount` em (0.08), swaying
sideways `sway` times that (0.35) a quarter cycle apart and turning up to `tilt` degrees (7) with the slope of the wave under it,
about its own middle: jelly letters, not a shake. How far they go follows the voice: the smoothed level of the timeline's
`vocal_db` (as `weight` reads it: -34 dB still, -5 dB re the vocal's peak the full amount) while the note is held, rising over
`attack` s from its start (0.12) and dying away over `release` s after its end (0.35), so a note that swells wobbles more and a
silent voice keeps the letters still. `follow` (0..1, default 1) is the share of the amplitude the level sets; 0 wobbles fully
whenever the voice sounds at all. Notes held less than `min_hold` s (0) stay still. Every word starts its wave at a phase of its
own (fixed by the word), so neighbours never bob in step, and every build is the same. The rate is chosen, not measured: a
level sampled once a frame cannot show a singer's vibrato. A `punch` or `write` look can give its words a `wobble` of their own
(`false`: none). It plays on top of the arrival, the spread and a ring's turn (the letters bob across the ring).

**Emphasis.** `punch = {top, above, words, ...look}` picks the words that stand out: the `top` loudest of the selection by `vocal_db`,
those at or `above` a level in dB (against the peak of the vocal), and `words` by number within the selection (1-based); at least one
of the three. `write = {min_hold, ...look}` takes the words that are not punch words and whose note is held for `min_hold` seconds
(default 0.6) or more. The look is any of `font`, `color`, `glow`, `outline`, `backing` (`false`: no strip), `scale` (cap height
relative to the line's, positive), `style` / `arrive` (not `rise`), `tilt`, `weight`, `wobble`, `reveal` (`true` writes the word out letter by
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
`flow`, `stack`, `slots` and `ring` need a panel (`on`, `box` or `screen`; `flow` and `stack` also work with a `size`). The words are measured in
Blender with the node the text is drawn with; the rest is `core/wordtype.py`.

**Bleed.** `layout = {kind = "bleed", width, height, rows, align, at, yaw, pitch, persp, swing, tilt, inset}` sets huge
words that run off the panel's edges, a motion-design reel's type (`wordtype.layout_bleed`): each word as large as makes its
ink `width` times the panel's width (1.2: the frame crops it), its cap height at most `height` of its band (0.8: a short
word stays readable), word k in band k mod `rows` (1) from the top; `align` (cycled, default `["left", "right"]`) starts a
word `inset` of its own width (0.04) off the panel's left edge and runs it off the right one, or the other way round, or
off both (`center`); `at` `[u, v]` nudges (shares of the panel, cycled). Every word is then turned in depth onto a plane,
`yaw` degrees its right side away and `pitch` its top away (cycled, default `[-24, 18]` and `[8, -6]`), seen in
perspective from `persp` em (3), swinging in `swing` degrees further on its landing over 0.2 s (0); `tilt` (cycled) turns it
in the plane too. It needs a panel (`screen = {margin = 0}` makes it the whole frame), takes a size per word (never shrunk
to a `fit`) and a slot per band, so `recycle = true` makes each new word take its band's place. Heavy grotesk words
slammed in on their onsets: `font = "fira_sans_heavy"`, `lyrics = {style = "slam", recycle = true, layout = {kind =
"bleed", rows = 2, yaw = [-26, 20, -14, 24], pitch = [10, -8, 6, -12], swing = 30}}`.

**Rings.** `layout = {kind = "ring", center, radius, pitch, gap, turn, start, rings, face, even, bend}` (`mkmmd/core/typering.py`)
sets the words round circles about a point, reading round them, the letters following the curve: `face = "out"` (default) reads
clockwise with the tops away from the centre (type round the top of a badge), `"in"` counter-clockwise with the tops toward it
(along its bottom); `bend = false` sets each word straight on its tangent instead. Each word takes an arc as long as its ink (with
the spread's headroom, the strip behind it and its outline: a sticker's die-cut edge) plus `gap` em (0.3, times `word_spacing`) of
clear ring, measured where neighbours come closest. A ring word stays up from its landing to the cut like any other, so an entry
that spans several lines (`lines = [a, b]`: a phrase sung again and again) ACCUMULATES instead of each line replacing the last:
`rings = "line"` (default) gives every lyric line a ring of its own, `pitch` em (default: the tallest word with its edge, plus
`gap`) further out than the one before, the line centred on `start` (`even = true`: its words share the whole ring out between
them); `rings = "fill"` winds the words round one ring, each repeat taking up where the last stopped, until the next word would not
fit, then shares that full ring's free arc out between its words (it closes evenly) and goes on round the next ring out. `start`
is where a ring starts, in degrees clockwise from the top (default 0, the top; 180 for `in`), `center` the point the rings go
round: `[u, v]`, shares of the panel from its centre (default `[0, 0]`; for screen type, the middle of the picture), or a
[target](#targets) (`"cast:rin.head"`, `[x, y, z]`, a table): a point of the scene that the rings follow on every frame, as the
cut's camera of each output sees it for screen type and on the panel's plane for type in the scene (the build keys an empty,
`<name>_ring`, `@<output>` for one output of several, there; a `[[freeze]]` does not hold it). `radius` is the first ring's, a share of half the panel's shorter side
(0.35). The common size is the largest at which every ring fits `fit` of the room about the centre (at most `size`), no line wraps
into itself on its ring and the innermost words keep half their height clear of the centre: a long line on a small first ring
sets the type small. Each ring turns about the centre at `turn` degrees a second (8; clockwise for a positive number), the next
ring the other way; a list (`turn = [6, -10, 4]`) gives each ring its own, cycled. A ring turns rigidly from the moment its first
word lands, so every word lands where its place on the turning ring has got to (the kinetic `orbit` and `bend`, below). An
entry's `tilt` (`{random = 6}`: stickers slapped on a little askew) is added on top of each word's turn to its ring. The report
gives `rings` (`{ring, words, radius, turn, first_frame}` per ring) and, round a point of the scene, `ring_track` (`{object,
frames}`).

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
| `orbit` | `{center, keys, track}` (`keys` required): after everything above, the word turned about `center` (`[x, y]` m in the text frame) by `keys` degrees counter-clockwise: a ring word carried round its turning ring (`Orbit`); `track` names an object whose place, seen from the text on its plane, the centre moves with (a ring round a point of the scene) |
| `bend` | `{radius, side}`: the letters (and the strip behind them) follow a circle of `radius` m through the word's pivot, its centre below the word (`side = 1`: the tops out) or above it (`-1`: the tops in) |
| `wobble` | `{amp, phase, amount, sway, tilt, wave}` (`amp` and `phase` required): every letter bobs by `amount` em x `amp` on a wave that travels along the word, `wave` letters to a cycle at the keyed `phase` (cycles), sways `sway` times that and turns up to `tilt` degrees about its own middle (`WobAmp`, `WobPhase`; `lyrics.wobble` writes them) |
| `yaw`, `pitch`, `persp` | the word turned in depth about its pivot, `yaw` degrees its right side away and `pitch` its top away (keyed, `Yaw`, `Pitch`), seen in perspective from `persp` em in front of it (a number, default 8; nearer is stronger) and laid flat in its plane again: a 2D projective warp of the word, its strip and its outline (a `bleed` layout writes them) |

They become keys on Value nodes of the text's node group, so a render needs no Python (read one with `mk q
'bpy.data.node_groups["mk_text_<name>"].nodes["Pop"].outputs[0].default_value' --frames ...`). Letters stay filled shapes of the
String to Curves instances: the drip scales and moves each character about its own top, opacity and tint travel as the attributes
`mk_alpha` and `mk_tint` that the shared material reads (hashed alpha: a word fades without sorting against the glass), the weight is
the shape repeated on a ring, the wobble moves and turns each character about its own middle, the bend moves every point of the
filled word round its circle.

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

```toml
[[text]]                              # a phrase sung again and again: a ring of die-cut stickers per repeat, turning in turn
name = "chant"
screen = true                         # round the middle of the picture, in every output
font = "sticker_display"
outline = { color = "#ffffff", width = 0.14, alpha = 1.0 }
[text.lyrics]
timeline = "audio/timeline.json"
lines = [9, 14]                       # the six repeats; every word stays up until `to`
to = 33.0
style = "slap"
tilt = { random = 5 }
colors = ["love", "pine", "gold", "iris", "foam"]
color_by = "word"
wobble = { amount = 0.1 }             # the letters bob with the voice while each word is sung
layout = { kind = "ring", rings = "line", radius = 0.3, turn = [6, -9] }
# round her head instead, as each output's camera sees it: center = "cast:rin.head"
# fill each ring before the next instead of a ring per repeat: rings = "fill"
```

The stage report has one entry per `lyrics` text, `{words, lines, style, layout, skipped, first_frame, last_frame, cut_frame,
drip_from, rewind_from, carried, punch, write, wobble}` (Blender frames; seconds for `drip_from` and `rewind_from`; `skipped`, `carried`,
`punch` and `write` as `[line, word]` pairs; `wobble` how many words wobble) and, with `zones`, `blocks` (`{shot, words, style, layout,
first_frame, last_frame}` per shot), with a ring `rings` and `ring_track` (Rings, above); and per word the usual text numbers with
`lyric` `[line, word]`, `chars`, `kinetic_keys` and `frames` [on, off). The string itself (`widest`) is not reported.

### Calls

`calls = {...}` on a `[[text]]` sets words of the entry's own, written in mk.toml, one on each hit: the shouts idol fans
yell at a show (a MIX, "ハイ！"), a hype word, an onomatopoeia; never the song's words (those are `lyrics`). Each call is an
ordinary text made by the lyric machinery (`mkmmd/core/calltype.py`, `mkmmd/blender/build/calltype.py`): it lands on its
hit's frame in a lyric `style`, stands where `layout` puts it and leaves when the next call takes its slot (`recycle`),
when its `hold` is over or at `to`. A call has no sung note, so the styles and keys that follow one (`type`, spreading,
breathing) do not apply. The object of hit k (from 1) is `<name>_c<k>`; the report lists each call's string.

| Key | Meaning |
|---|---|
| `words` | the calls (required): hit k shows `words[k mod len]` |
| `times`, `lane`, `min_db`, `every` | the hits: clip seconds, or a lane of the `[audio] hits` file at or above `min_db`, every `every`-th ([Timeline](#timeline)) |
| `from`, `to` | the span the hits are picked in (default the clip); a call still on screen leaves at `to` |
| `hold` | seconds a call stays at most (default: until the next call takes its slot, or `to`) |
| `style`, `arrive`, `layout`, `recycle`, `backing` | as `lyrics` takes them ([Lyrics](#lyrics)): `slam` lands big and fast with a ring and a shake |
| `colors`, `color_by` | a colour list cycled by `call` (default: the word's place in `words`) or by `hit` |
| `tilt`, `sizes`, `offsets`, `scales` | lists cycled over the hits: degrees, cap heights, nudges, scales |

```toml
[[text]]                              # fan calls on the chug of one bar, in the penlights' colours
name = "mix"
screen = { anchor = "center", height = 0.45 }
font = "dela_gothic_one"
calls = { words = ["ハイ！", "オイ！"], lane = "chug", from = 4.96, to = 7.43, style = "slam", colors = ["love", "gold", "foam"], tilt = [-8, 6] }
```

### Screen type

A `[[text]]` with `screen = ...` is type of the picture, not of the scene. The text stage builds it as an ordinary text object on the
plane z = 0 of the world, one unit per FRAME HEIGHT (1 is the height of the picture, x right and y up from its centre; a 16:9 frame is
1.78 wide, a 9:16 one 0.56), hidden from every scene render. A frame that has any of it on screen gets a second pass from `mk render`
and `mk look` (`Looks.render` in `mkmmd/blender/styles.py`): the type alone, as its emission shows it (lights off, black world, the
plain view transform: a palette colour is that colour), through an orthographic camera one frame height tall. The pass is kept off
the frame, in a file of its own beside it, `<frames>/screen/<frame>.png` (straight RGBA, written first, so a frame on disk always has
its layer); `mk post` lays it over the cut's frame AFTER the cut effects and before the grade (`mkmmd.core.screentype.composite`,
tested without Blender), so a slash, an expanding figure or a growing bubble passes under the words and the grade covers them like the
rest, and `mk look` does the same for the images it shows. The loop's wrap turns the frame with its words: they ride the page, and
frame 0's lie under it ([Transitions and inserts](#transitions-and-inserts), `page`). It stays where it is put whichever camera is
cutting, whatever stands in front of it and however the shot is lit.

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

`negative = true` is screen type as a photographic negative: where a glyph covers a pixel the picture is inverted instead of painted,
per channel `out = bg * (1 - a) + (1 - bg) * a` with `a` the glyph's antialiased coverage (a solid pixel is exactly `1 - bg`, a
rim pixel part way to it; the text's `color` is not used, its keyed opacity, if any, scales the coverage). The coverage is a
layer of its own, `<frames>/negative/<frame>.png` (RGBA, only its alpha is read), written before the frame's `screen/<frame>.png`
(which a frame with negative type has too, fully transparent when it has no ordinary type there), so a screen layer on disk says its
negative one is there as well. `mk post` and `mk look` lay the ordinary layer first and invert the picture with the negative one
last, at the same place: after the cut effects and before the grade, or after the grade with `[post] screen = "over"`
(`mkmmd.core.screentype.composite` on the CPU, the `negate` kernel of mkr-post on the GPU; the two agree within a level, tested).
Negative and ordinary type therefore live in one frame without touching each other, and negative type inverts whatever ordinary type
it lies over too. Negative type that overlaps (two words, or the overlapping letters of one) is one coverage, `1 - (1 - a1)(1 - a2)`,
and inverts once: a second inversion would hand the picture back and punch the type full of holes. Over a kept eye (`[post] keep`)
the mask is cut where it covers, as for ordinary type, and at the loop's wrap the negative layer turns with the page like the screen
one. The key needs `screen`, and is refused (`text '<name>': negative and knockout both reverse type ...`) with `knockout` (the other
reversal: out of a silhouette's figure; the two do not combine, pick one), `backing` and `outline` (a strip or ring is a plate of its
own: it would be inverted as a block with the word cut out of it) and `blocks`; a value that is not true or false is an error too.
Lyric entries (`lyrics = {...}` with `screen`) take it as any entry key does, one negative word each, and `lyrics.zones` can turn it
on or off per shot (`zones = { <shot> = { negative = true } }`). `mk render` draws the negative layer in Blender and, where the bake
holds the words, in mkr; frames with none have no file.

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
them), but for the loop's wrap, whose page carries them. What belongs to a figure is the exception: a silhouette's `knockout` type
is part of that shot's picture. `mk render --no-styles` and `mk look --no-styles` leave screen type out (it is part of the looks).
The layer files are small (the type is mostly transparent) and a frame with nothing on screen has none. The pass is one more EEVEE
render of a frame that has any type on it (two, with negative type: one for each layer). A stale `screen/` or `negative/` folder of an
earlier build stays on disk with the frames it belongs to: delete `renders/<preset>` when the type changes, as for any frame.

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

### Block type

A `[[text]]` set in toy alphabet blocks: one wooden cube per letter or digit, its edges and corners rounded, its six faces painted in
palette colours, the letter raised (or printed) on its front inside a rim, and another letter, a digit or a little picture on each of
the other faces. The sung words are lyric type with `lyrics = {style = "blocks", ...}` (`timeline`, `line` / `lines`, `words`, `case`,
`clean`, `from` and `to` as in [Lyrics](#lyrics); its other keys are errors); a label is `text = "..."` (`\n` or a list of strings for
rows) with `blocks = {...}`. The look, the layout and the motion are in `blocks` (`true` is all defaults). `mkmmd/core/blocktype.py`
(numpy only, tested on synthetic timelines) decides the blocks, what their faces show, where they stand, how big they are and how each
moves, and makes each block's mesh; `mkmmd/blender/build/blocktype.py` builds them. A block gets no character but letters and digits:
an apostrophe or a hyphen leaves no gap.

The entry's own keys that apply: the placement (`on`, `mount`, `at`, `facing`, `up`, `box`, `back`, `lift`, `screen`, `aspect`,
`offset`), `font`, `size` (the EDGE of a block: m, frame heights for screen type), `fit`, `align` / `valign`, `glow` and `lit`; any other
text key (`color`, `depth`, `kinetic`, `backing`, ...) is an error that names it. The blocks are fitted as text is: the largest edge at
which their outline, as the viewer sees it after `view`, fills `fit` of the panel (default 0.9; at most `size`; `size` alone does not
shrink), aligned in that margin. They stand in front of the panel, their back faces on it.

**Building.** Per entry (per output for screen type) an empty `<name>` (`<name>@<output>` for one output of several) stands at the
panel, turned by `view` and scaled by the edge, so `[[key]]` can move a whole entry; under it one mesh object per block,
`<name>_l<line>w<word>c<letter>` (a label's `<name>_c<k>`, numbered in reading order), keyed on whole frames: location, rotation
(quaternion) and scale where it moves (LINEAR; straight runs dropped, so linear keys still give every frame exactly), its scale 0 until
its fall begins and from the frame it is gone (CONSTANT steps): `mk render` needs no Python. Blocks with the same letter share a mesh,
each object links its own colours (OBJECT material slots). Each glyph is read once per font off a throwaway String to Curves / Fill
Curve object and becomes geometry, so no string is left in the .blend; the objects carry `mk_lyric = [line, word]` and `mk_block` (the
letter's place in its word), screen type `mk_screen` and `mk_show` as any screen text. The paint shades itself from the camera's side
(a fixed light over the viewer's left shoulder in camera space, with a gloss highlight) as emission times `glow`, so screen type, drawn
with the lights off, still shows solid blocks; in the scene `lit` adds that share of a Principled surface the scene lights. Defaults:
screen type `glow = 1` (and no `lit`), in the scene `glow = 0.3`, `lit = 1`.

**Layout.** The blocks of a word stand `gap` edges apart and words `space` apart, in rows that build up from the bottom as real blocks
do: a line's first row is its lowest and every row stands on the one under it. `layout = "flow"` (default) sets each lyric line in
`rows` rows (`"auto"`: the count that sets the blocks biggest; a number: at most that many), `layout = "stack"` puts every word on a
row of its own, a tower with the first word at the bottom. With `next = "stack"` (default) each later line stands on top of the one
before; with `next = "tumble"` every line takes the same place. A label reads top to bottom: its last row stands on the floor and lands
first. `view = [pitch, yaw]` turns the blocks to show their tops (pitch) and right-hand sides (yaw), in degrees (default `[16, 12]` for
screen type, which is drawn straight on, `[0, 0]` in the scene, where the camera shows them).

**Motion.** Lengths are in edges, times in clip seconds.

- The fall: a word's first block lands on the frame `floor(onset x fps + 0.4)` (as a lyric word does, never before `from`), its other
  letters `stagger` s apart (to the frame); a label's first at `land`, every next one `stagger` later. Each falls from rest `height`
  edges above its place (`"top"`, the default: from just out of the picture for screen type, just above the panel in the scene) under
  `g` edges/s^2, so it is on screen from the frame its fall begins; it is tilted by up to `spin` degrees about an axis of its own,
  turning at a steady rate to land flat, bounces with restitution `bounce` until its hops are too low to see, and squashes on each
  impact (by `squash` for a fall of 4 edges or more, less for a lower one), a damped spring of about 50 ms about its bottom.
- Hits: `hits = {lane, move, min_db, every, from, to, ...}` or a list of them. On every hit of a lane of the `[audio] hits` file
  (picked as `[[key]] lane` picks: `min_db`, `every`, `from` / `to`), on the hit's frame (the landing rule), every block that stands on
  its floor moves: `move = "jump"` hops `height` edges, `wave` s later per block pitch to the right (a ripple), turning `turn` degrees
  about its x axis on the way (whole turns: 360 is a flip that lands on its letter again); `move = "slam"` is lifted `height` over the
  first 60 % of `lead` s before the hit and slammed down on it, squashing by `squash` with the whole pile (each block's floor sinks with
  the rows under it). A block still falling or bouncing lets a hit go by.
- The next line: with `next = "tumble"` the earlier line is knocked off when the next line's first block starts to fall, from where
  that block will land outward; `crash = t` knocks off every block still up at `t`, from the middle outward. A knocked block flies off
  sideways (`kick` edges/s, less the farther it is), up (`pop`) and toward the viewer, spinning (`spin` degrees/s), `gap` s later per
  block pitch away; with `floor` (default in the scene) it lands on the floor of the bottom row, its lowest corner on it, bounces,
  slides to a stop and lies flat on its nearest face until the cut; without it (screen type) it falls out of the bottom of the picture,
  or is gone `life` s after it was knocked.
- Gone: at `to` (the cut), as lyric words; a label at `blocks.to` (default: the end of the build).

**Faces.** The front shows the letter in `ink` on its paint, inside a rim (`rim`), raised `relief` edges (`letter = "raised"`) or
printed flat (`"printed"`). `sides = "mixed"` (default): the back the letter in the other case, the right side the next letter of the
alphabet, the bottom a digit, the top and the left side a picture; `"letters"` letters and a digit all round; `"pictures"` five
pictures; `"plain"` paint only. A face whose glyph the font lacks is left plain; a front letter it lacks is an error that names its
place (`letter 2 of word (1, 3)`). The pictures are printed in palette colours on faces painted `ground`: carrot, tomato, peapod,
eggplant, radish, broccoli, corn, mushroom, apple, sun, star and heart (`pictures` picks, cycled by the letter).

| Key (in `blocks`) | Meaning |
|---|---|
| `paint` | a colour or a list cycled over the blocks, one per face as well (default `["love", "gold", "foam", "iris", "rose", "pine"]`) |
| `ink`, `wood`, `ground` | the letters and rims (`surface`), the rounded wooden edges (`"gold:base:0.6"`) and the faces with pictures (`surface`) |
| `letter`, `relief`, `round`, `rim` | `raised` (default) or `printed`; the raised height (0.05 edges); the radius of the rounded edges (0.12, 0.02 .. 0.3); a rim round each lettered face (true) |
| `sides`, `pictures` | what the other faces show: `mixed` (default), `letters`, `pictures` or `plain`; the pictures to use (default all) |
| `layout`, `rows`, `gap`, `space` | `flow` (default) or `stack`; rows per line (`"auto"`); edges between the blocks of a word (0.06) and between words (0.6) |
| `view` | `[pitch, yaw]` degrees the blocks turn to show their tops and right sides (screen type `[16, 12]`, the scene `[0, 0]`) |
| `fall` | `{height, g, bounce, stagger, spin, squash}`: `"top"` or edges; edges/s^2 (196.2: a 5 cm block); restitution (0.35); s between letters (0.03); degrees of tilt (25); squash (0.12) |
| `hits` | a table or a list: `{lane, move, min_db, every, from, to}` and the move's own: `jump` `{height 0.6, wave 0.012, turn 0}`, `slam` `{height 0.35, lead 0.09, squash 0.28, wave 0}` |
| `next`, `crash` | `stack` (default) or `tumble`; a clip second at which every block still up tumbles away |
| `tumble` | `{kick, pop, spin, gap, life, floor}`: 5 and 7 edges/s, 420 degrees/s, 0.02 s per pitch, 1.2 s, true in the scene |
| `land`, `to` | a label's: clip seconds its first block lands (0) and it is gone (the end of the build) |

```toml
[[text]]                              # the first words crash in over the picture, hop on the kicks and are slammed on the sub
name = "hook"
screen = { anchor = "bottom", height = 0.34 }
font = "fraunces_black"
[text.lyrics]
timeline = "audio/timeline.json"
line = 1
words = [1, 4]
case = "upper"
style = "blocks"
to = 5.9
[text.blocks]
hits = [{ lane = "kick", move = "jump", min_db = -10 }, { lane = "sub", move = "slam", min_db = -2 }]

[[text]]                              # a tower on the floor of the set, word on word, that crashes before the cut
name = "tower"
at = [1.3, -0.5, 0.75]
facing = [0.0, -1.0, 0.0]
box = [1.4, 1.5]
valign = "bottom"                     # the panel's bottom edge on the floor
size = 0.2                            # 20 cm blocks
lyrics = { timeline = "audio/timeline.json", line = 5, case = "upper", style = "blocks" }
blocks = { layout = "stack", fall = { height = 22 }, crash = 23.98 }

[[text]]                              # a label on a wall, letter by letter from 0.5 s, flipping on the snare
name = "abc"
on = "studio:wall"
size = 0.22
text = "ABC 123"
blocks = { land = 0.5, sides = "pictures", hits = { lane = "snare", move = "jump", turn = 360 } }
```

The stage report has per entry `{style, label, blocks, words, lines, layout, rows, next, edge, view, fits, skipped, first_frame,
last_frame, cut_frame, hits, moves, tumbles, crash, meshes, on, font, panel, object, per_word}`: Blender frames (`first_frame` the first
landing), `hits` the hits picked per move and `moves` the blocks that made them, `tumbles` `{t, blocks}`, `per_word` each word's
`lyric`, `blocks`, `land` and `frames` [on, off); never a letter. Look at a landing with `mk look --frames F --sheet` at the words'
`land` frames, and at a hit on its frame.

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
`[look.breathe]` makes the palette breathe into another one over clip time ([Lights and the look](#lights-and-the-look)).

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

With `engine = "mkr"` the final preset's own 64 samples are drawn as 36 of twice the scene's shadow rays ([mkr](#mkr): its samples).

`[render]` (all optional) overrides the preset: `samples`, `shutter` (the motion blur shutter, 0.35) and `engine` (`eevee`, the
default (EEVEE Next), `cycles`, `workbench` or `mk`, below). A fisheye shot ([Fisheye shots](#fisheye-shots)) needs `cycles`:
under another engine the render stops before it starts, naming the frame and the camera. `--samples`, `--percent` and `--engine` override both for one run; `--output NAME` renders some
outputs, `--frames SPEC` some frames (default the whole clip), `--jobs N` runs N Blender processes per output that share the frames.
Frames are claimed with an empty file first, so a stopped render resumes; frames and layers on disk that are older than the built
scene show the scene of an earlier build, so they are drawn again (the report's `stale` counts them; `--keep` takes them as they
are, for a scene built again without a change); a disk check refuses to start when the frames would not fit
(it keeps 1.5 GB spare). Shots with a render-time look ([Shots](#looks-silhouette-and-reflection): silhouette, reflection) are finished
flat frames on disk, composed inside the render job; `--no-styles` renders them as lit, and leaves out the cut effects and the screen type too (they are part of the looks). `[[transition]]` and `[[insert]]`
([Shots](#transitions-and-inserts)) add the layers they need next to the frames (`plate/`, `matte/`, `back/`, `point/`; `--no-transitions`
leaves them out). Screen type ([Text](#screen-type)) is `screen/<frame>.png`, straight RGBA, for the frames that have any (with
`negative = true` type also `negative/<frame>.png`, whose alpha is the coverage that inverts the picture, written before it). With
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

#### mkr

mkr (`mkr/mkr`, Rust and wgpu on Vulkan) draws EEVEE Next's lit frames from the bake, deferred the way EEVEE draws them
(`mkr render --bake BAKE --output NAME --frames SPEC --out DIR`, with `--samples`, `--percent`, `--motion-blur`,
`--shutter`, `--params FILE.json`; `mkr/mkr/src/render.rs`). Per frame: the scene posed on the frame's own time (with
motion blur on the shutter's middle, and on its open and close too: skinning, morphs, the bend normals) and the shadow
maps of the lights that cast
shadows, drawn once (`shadows.rs`: a cube of faces around each light, five for area and spot lights, or for a sun four
orthographic maps, as EEVEE's sun reads the clipmap level around the camera that holds the receiver: three over slices
of the view from the camera out to 3, 24 and 192 m, one over every caster, each receiver reading the finest that holds
it and its shadow ray running eight times that map's width, or across the last one's depth). EEVEE draws each face at
the level its receivers on screen ask for (their pixel
footprint, bounded by the light's `shadow_maximum_resolution` and scaled by the scene's `shadow_resolution_scale`; 8192
texels across a face at the finest; `shaders/shadow_lod.wgsl` has the rules measured): mkr keeps every face whole at a
coarse level (`shadow_face_size`, 1024 texels, halved while the faces would take more than 1 GB) and lays a pool of
finer windows (`shadow_windows`, 8 of `shadow_map_size`, 2048 texels) over the receivers that ask for more, placed after
the first sample's depth prepass (`shaders/shadow_setup.wgsl`). Then per sample (the camera jittered by EEVEE's film
filter): a depth prepass, where dithered materials test a
hashed threshold (a hash of the surface's position on grids a quarter of a pixel wide, turned by the sample's offset:
coplanar layers cover together and the noise left is EEVEE's); the G-buffer of the opaque and dithered surfaces, each
material its own small module from `mkr-shade`; one lighting compute pass over it, compiled once (`shaders/light.wgsl`,
`ltc.wgsl`, `shadow_map.wgsl`: each light's lobes integrated over its shape with linearly transformed cosines and
EEVEE's sphere approximation of the horizon, faded out at EEVEE's influence radius (the light's own when it has one:
`use_custom_distance`, `cutoff_distance`), and shadowed by tracing a ray toward
a random point of the light through its maps in the scene's shadow steps, by the hit rule measured on EEVEE's frames:
the steps near the receiver read the level it asks for, the farther ones the face's, and the ray's offsets off the
surface and its depth tolerance are counted in that level's texels, as EEVEE's are; the ray is lifted along the shading
normal and its point of the light kept above that normal's horizon, as EEVEE's developer docs say its biases are: with
the geometric normal Crimson's figure was 0.03 levels and lab `mmd_model` 0.01-0.04 further from EEVEE's frames); the
blended surfaces lit forward, back to front; with motion blur, EEVEE's post process smearing the sample's picture
(below); and the film taking the
samples four at a time. Then the view transform's table, EEVEE's dither and 8
bits, read back and written while the next frame draws. Dithered materials whose shadows follow their alpha cast
EEVEE's hashed shadow, a texel stopping the light with the probability of the caster's opacity per sample: the maps hold
their nearest partly transparent layer over the opaque casters and the product of the layers' transparencies.
`shadow_mode: "rays"` traces ray queries toward the light instead of the maps: converged soft shadows, as EEVEE's
jittered shadows and Cycles draw them.

Keyed and driven colours (a [breathing palette](#lights-and-the-look), a material morph's tint, a keyed lamp) are
mkr's like any frame: every driven node socket value and every light's colour and strength come per frame from the
bake. What is keyed where the bake holds one value (a colour ramp's stops, a node's setting, a material's or a world's own
colour when it draws without nodes: bake.json's `keyed`) makes the frames that show it Blender's, and `mkr info` lists
the material (or the world, for the whole scene) with those paths.

Motion blur (`--motion-blur`, `motion_blur.rs`, `shaders/motion_blur.wgsl`) is EEVEE Next's at `motion_blur_steps` 1:
every sample drawn at the shutter's middle (the frame itself, a half shutter later when the scene's
`motion_blur_position` is START, earlier when END), the G-buffer keeping each pixel's velocity to where the shutter's open
and its close show it (the bake's poses on those times: bones, object matrices, shape keys, frame-by-frame meshes, the
camera; where nothing is drawn, the camera's turn alone), then per sample, before the film takes it, the post process:
the longest velocity toward each end per 32 px tile, spread along its line over the tiles it reaches (a tile taking the
other end's longest too: on the lab that halves the error), and per pixel 6 taps along its tile's longest velocity and 6
along its own for each half of the shutter (`mb_taps`; EEVEE takes 8; over a frame's 64 samples 6 leave mkr's frames as
close to its, and final frames' 36 samples take 8: below), the tile picked a random quarter tile off and the taps a
random offset per pixel and sample, each counted as background or foreground by a soft depth test (the scene's
`motion_blur_depth_scale`)
and by whether its blur reaches, the background's mean standing for what the taps leave uncovered (the weights
Jimenez 2014 describes, after McGuire et al. 2012 and Guertin et al. 2014). EEVEE Next has no `motion_blur_max`: it
smears a 300 px move over 300 px. `motion_blur: "quality"` draws each sample at its own time in the shutter instead
(true motion blur, as Cycles draws it); a scene of more than one step gets that too until mkr draws EEVEE's segments.
Over the moving pixels (where Blender's frame differs from the one before by more than 8 levels, dilated 4 px; in the
lab, where the blur changes the frame), mean |error| with the post process against the time samples (Vega Tables
measured with 8 taps):

| Frames | Moving | Post process | Time samples |
|---|---|---|---|
| lab `mb_translate`, `mb_start`, `mb_end`, `mb_shutter1` | 11-15 % | 1.56, 1.36, 1.50, 1.78 | 2.74, 2.50, 2.65, 3.90 |
| lab `mb_spin`, `mb_arm`, `mb_pan` | 9-71 % | 3.07, 2.13, 0.65 | 3.31, 3.18, 2.39 |
| lab `mb_steps3` (3 steps: both on time samples) | 11 % | 2.37 | 2.37 |
| Playing Possum final, 16:9 and 9:16 (20 frames each) | 14, 32 % | 2.72, 2.85 | 3.05, 3.36 |
| Vega Tables final, 16:9 and 9:16 (20 each) | 13, 25 % | 1.60, 1.68 | 1.96, 2.48 |
| Crimson and Clover final, 16:9 and 9:16 (21 each) | 11, 22 % | 3.77, 3.47 | 4.58, 4.13 |

The post process costs 7-24 ms of a 64-sample 1080p frame's GPU time (Crimson 9.9, 7.0 and 15.4 ms for 16:9, 1:1
and 9:16, 1-2 ms less than with 8 taps; Playing Possum 15 and 24 for 16:9 and 9:16). Blurring the film of a group of
samples at a time instead (the blur is linear in colour) would cost a tenth of that (once on the whole film, averaging
8 draws of its noise: Crimson 1.2, 0.9 and 2.0 ms), but the depth and velocity it blurs by are one sample's, which
leaves the film's soft edges of what moves partly sharp: 0.07-0.15 levels more error over the moving pixels (Crimson
3.88 and 3.57, Playing Possum 2.79 and 2.96, the lab's mb cases 1.95 on average against 1.80), so mkr blurs each
sample as EEVEE does.

Every constant that matches Blender only by measurement is a tunable: `mkr params` lists them with their defaults (the
best values measured), `--params FILE.json` changes some, and `mk parity tune` climbs them (below). Each material's
pipelines compile in parallel threads (at most one per 1.5 GB of free memory, 4 by default, `MKR_COMPILE_THREADS`)
through the driver's pipeline cache, kept per adapter and driver in `$XDG_CACHE_HOME/mkr` (else `~/.cache/mkr`): Crimson
and Clover's 79 materials compile in about 3.5 s from cold (1.2 GB peak) and in 0.3 s warm. The report on stdout times
every frame: `gpu_ms_per_frame` and `gpu_ms_phases` (pose, shadow maps, prepass, gbuffer, lighting, blended, motion
blur, film, dof,
resolve; GPU timestamps), `cpu_ms_encode` (recording and submitting a frame, of which `cpu_ms_stage` poses it and
`cpu_ms_submit` is wgpu finishing and submitting its commands), `cpu_ms_wait` (blocked on the GPU), `cpu_ms_collect`
(the readback and the PNG writers' queue), `setup_seconds`, `compile_seconds` and `peak_rss_mb`.

On Crimson and Clover (64 samples, motion blur, 100 frames across the clip on an RX 7700 XT) a 1920 x 1080 frame takes
70 ms (GPU 66: prepass 8.8, G-buffer 14.3, lighting 17.4, blended 4.9, motion blur 9.9, film 6.2, shadow maps 4.0)
against Blender's 1.9 s, a 1080 x 1080 one 60 ms (GPU 56) and a 1080 x 1920 one 97 ms (GPU 93, lighting 31: the
portrait frame is mostly her); the CPU's 16 ms a frame overlap the GPU's. The lighting is the shadow tracing (each
step about 3 ms of the portrait frame's); the
geometry passes are bound by vertex work, run twice per sample, and the swarm's 144 butterflies (88 % of the triangles)
take about a third of them. Per-draw overhead, varyings, rasterisation and the map lookups' memory are not what costs
(measured: doubling the draws, halving the varyings or culling back faces leave the time within noise, and testing
coarse blocks of the maps before their texels gains 2 % at most). A G-buffer without the depth prepass would save
7 ms, but its extras are storage writes, which need the single surface per pixel the prepass leaves. With the extras
in two more colour targets, the butterflies drawn straight into the G-buffer save 3-5 ms of the geometry passes, but
they add to what makes frames differ from run to run. RADV's colour compression (DCC) of the G-buffer's colour targets
now and then gives one pixel of one sample another value: a 1-2 level speck in one or two of every 300 9:16 frames, in
today's frames as in the earliest build measured (none in 680 frames with those targets uncompressed, none with
`RADV_DEBUG=nodcc`, which leaves every other frame byte for byte as it is); the two extra targets triple it.
Uncompressed G-buffer targets would cost 3 ms of a 1:1 frame and 9 ms of a 9:16 one, so mkr keeps them compressed, and
the prepass; byte-identity checks between two builds run both with `RADV_DEBUG=nodcc`.

Measured against EEVEE's frames: Crimson's 90 test frames (30 per output, no motion blur) at a pooled mean |error| of
0.82 levels (0.66, 0.80 and 1.00 for 16:9, 1:1 and 9:16; 2.78, 2.60 and 2.36 over her figure alone, 11.9, 10.8 and 9.4 %
of its pixels off by more than 8), from 5.1 (18.0, 14.8, 13.9) before; the lab's cases between 0.03 and 1.75 (`mk parity
lab`). The shadow tunables' defaults are `mk parity tune`'s best over Crimson's figure and the lab's cases together, the
lab weighted 3 to 1 so that no setting buys the figure with the lab. EEVEE's glossy light is an LTC table fitted
the way Heitz et al. fit theirs (normalised over the whole sphere, so the part of a rough lobe under the horizon is
lost), times the split-sum albedo, faded out as the light's bounding sphere sinks under the lobe's horizon
(`ltc_glossy: "eevee"`, measured on probes; `"mkr"` integrates true GGX). What still differs: EEVEE's shadow tracing is
known by measurement only (`shadow_rule`: the default rule matches Crimson's frames and the MMD lab cases best; `"v2"`,
fitted on EEVEE's probes, the lab's plates and square or disk lights), and its dithered alpha leaves per-pixel noise mkr
reproduces only statistically. Her eyes are the largest share of what is left in the close-ups (5.0 levels over the
eyes, mkr darker by 3.9): the eye white (白目) is the floor of a socket 2-3.5 cm deep, its walls the same material, the
iris a lens floating 2 cm in front of it and the highlight 1-2 mm in front of that, all opaque but the iris's rim. Under
the key light Cycles leaves that floor 4 % of its light, EEVEE 27 % and mkr 23 % (lab `mmd_model` face: the eye white
0.83 times EEVEE's under the key, 0.91 under the fill, its unshadowed light within 0.3 %); jwin's eyes, the same
model's sockets, are 0.86 times EEVEE's. Neither the casters' alpha, the normals, the levels the steps read nor the
biases move it; the steps' placement does (12 steps: 0.26, step power 2.5: 0.26), but every such setting costs the
lab's area, penumbra and `shadow_close` cases more than it gains. EEVEE's own frames at other `shadow_step_count`s show
where the trace parts from its rule: its eye white keeps 0.40, 0.62, 0.47, 0.27, 0.19, 0.14 and 0.13 of its light at
2, 3, 4, 6, 8, 12 and 16 steps, mkr's (`shadow_steps`) 0.19, 0.23, 0.24, 0.23, 0.23, 0.26 and 0.28; on
`shadow_close` mkr's leak falls with the steps as EEVEE's does (the 5 mm card at 35 degrees, 6 to 16 steps: 0.34,
0.26, 0.17, 0.14 against 0.48, 0.37, 0.21, 0.12) but not below 6 (2 steps: 0.45 against 0.58; overhead 0.67 against
0.38). mkr meets the scene's 6 steps by its tuning, not by EEVEE's rule. Fitted apart from the hit rule, the steps
spread over the whole ray (`shadow_trace` 1, fine reads along it: `shadow_lod_reach` 1000) follow EEVEE's curves on
`shadow_close` best (mean |V - EEVEE| over 2-16 steps 0.094 -> 0.059; at 6 steps the 35-degree cards 0.45, 0.54,
0.68, 0.86 against EEVEE's 0.48, 0.54, 0.66, 0.83), and `shadow_trace` 1 alone takes Crimson's figure from 2.78, 2.60
and 2.36 to 2.61, 2.46 and 2.28 (her eyes 5.0 -> 4.0 levels); but it doubles the plain area cases' error (lab
`area_square` 1.64 -> 2.85, the lab's mean 0.85 -> 1.06), so the default stays. No placement makes the eye white's
curve peak and fall as EEVEE's does (0.16-0.28 at every step count): that part is the hit rule's.

Its samples. Fewer samples leave more of mkr's own noise in its frames, so their error against EEVEE's grows (Crimson's
90 frames at 64, 48 and 32 samples: 0.833, 0.855 and 0.889 levels; the figure 2.56, 2.63 and 2.73). Split by giving one
source of noise its own sample count (the others repeating each draw twice), the shadows' rays are most of it: alone at
32 they cost 0.025 levels of the 0.055 (the figure 0.076 of 0.170), the film's jitter 0.003 (0.008), hashed alpha
0.0002, the materials' closure picks nothing (no depth of field, ray tracing or motion blur in those frames). So a
sample can trace several shadow rays per light (`shadow_rays`, a multiple of the scene's `shadow_ray_count`, EEVEE's
own rays per sample: ray r of sample s draws point s K + r of the sequences, so two rays at 32 samples are bit for bit
64 samples whose pairs share a G-buffer), and draw them from a better sequence (`shadow_sequence: "sobol"`: one point
of Sobol's sequence in five dimensions per ray, its index shuffled among the frame's rays and its numbers
Owen-scrambled per pixel and light, where the default R2 draws the light's point and the filter disk's from one
sequence turned two ways; Sobol's alone takes 64 samples from 0.833 to 0.828). `mk render --preset final` with mkr
draws 36 samples of twice the scene's shadow rays from Sobol's sequence with 8 motion blur taps (`mkmmd/render/mkr.py`
FINAL, when the preset's own samples apply: no `--samples`, no `[render] samples`; `mk parity frames --preset final`
draws the same; the defaults stay the scene's rays from R2 and 6 taps). Against EEVEE's frames (Crimson without motion
blur; Vega Tables and Playing Possum final, 20 frames per output, whole frame and moving pixels; the lab's 51 case
outputs, mean of their means, scored as the shadow tune scores: the figure plus 3 times the lab) and GPU ms per frame
with motion blur (Crimson 100 frames, both in one hold):

| | 64 samples | final: 36 x 2 rays, Sobol, 8 taps |
|---|---|---|
| Crimson whole: 16:9, 1:1, 9:16 (pooled) | 0.664, 0.804, 1.019 (0.8334, 3.376 % over 8) | 0.661, 0.801, 1.012 (0.8290, 3.373 %) |
| Crimson figure (pooled) | 2.815, 2.626, 2.397 (2.5624, 10.384 %) | 2.803, 2.616, 2.381 (2.5489, 10.373 %) |
| Vega whole / moving, 16:9 and 9:16 | 0.564 / 1.598, 0.701 / 1.684 | 0.563 / 1.598, 0.700 / 1.682 |
| Possum whole / moving, 16:9 and 9:16 | 0.535 / 2.715, 1.216 / 2.853 | 0.534 / 2.712, 1.213 / 2.849 |
| lab mean (over 8), shadow tune score | 1.2478 (4.45 %), 6.306 | 1.2456 (4.50 %), 6.286 |
| GPU ms: Crimson, Vega, Possum | 67.8 / 57.2 / 96.1, 331 / 322, 100 / 124 | 49.8 / 42.3 / 74.9, 283 / 277, 75 / 101 |

The lab's areas, penumbra and ggx gain 0.03-0.11, `dof_psf` 0.17; `alpha_hashed` loses 0.27 (11.1 % over 8 against
8.2: hashed alpha's own noise, which smaller hash cells only raise), `dof` 0.19 (36 points of the aperture), `mb_steps3`
0.04 (time samples). Vega's 16:9 moving pixels are 0.0004 levels worse (1.5982 against 1.5978), its 9:16 ones 0.002
better; with 6 taps they were 0.003 worse on both. Scenes whose lighting is most of a frame gain least (Vega's 14
shadowed lights: 14 %); jwin's, its lighting mostly lobes and ray tracing per sample, 38 % (419 -> 259 ms). Its 36
depth-of-field frames per output (final settings, against Blender's): whole frame 4.586 -> 4.586 and 3.681 -> 3.676,
the error depth of field adds (DepthOfField's measure) +0.665 -> +0.668 and +0.498 -> +0.498, but the blur itself
against Blender's (both frames less their DOF-off ones) 1.508 -> 1.527 and 1.117 -> 1.127: 36 aperture points are a
little noisier, which the shadows' gain covers on the whole frame; the lab's `dof` loses 0.19, `dof_psf` gains 0.17.

The world (`mkr/mkr/src/world.rs`, `shaders/world*.wgsl`): mkr-shade compiles the world's node tree like a material
(`mat_world`, its animated values and images in the scene's buffers); the camera sees it where no surface is (a pass
after the lighting, per sample, through the sample's jittered camera; not with the film transparent), and surfaces are
lit by its probe as EEVEE Next lights them without ray tracing (measured on the lab's world_* cases): the world drawn
once a frame (again only when its animated values change) into a cube map as half the World's `probe_resolution`
across, the Light Path seeing no camera ray there (world_lightpath), its mips box-averaged; glossy lobes read it along
the reflection at the mip whose texels are `world_glossy_blur` (4) GGX alphas across, blending into the harmonics'
light from roughness 0.7 to 0.9; diffuse and translucent closures read its L1 spherical harmonics, projected from the
32-texel mip, band 1 at `world_band1` (0.675) of the cosine lobe's weight: EEVEE's diffuse world light fits that to
0.2-0.5 % on a lab ball (band 1 at full weight is 10-20 % off), band 0 at 0.991. EEVEE leaves world light unshadowed
without ray tracing (no contact darkening at all in world_uniform) and ignores the world's ray visibility flags
(Cycles' own: world_visibility), so mkr does too. Against EEVEE (64 samples), mean |error| in levels: world_uniform
0.20, world_lightpath 0.15, world_transparent 1.13, world_visibility 1.20, world_sky 1.49, world_stripes 2.79 (the
rough balls' blur: EEVEE's probe filter is not a box pyramid). A scene without a world draws byte for byte as before
(no pass runs, the lighting adds nothing). What mkr still leaves to Blender, every lit frame and plate (`mkr info`:
`unsupported.scene`): EEVEE's ray tracing by light probes (`ray_tracing_method` PROBE), a world mkr-shade cannot draw
(a volume, node types it lacks), a bake older than the world's settings, and a world over a non-zero sun threshold
(EEVEE takes that light out into a sun light: world_sun) unless it is one colour under it; mk's own worlds
(night_sky, photo_world) set the threshold to 0. The look passes, drawn with a black world and no ray tracing, stay
mkr's.

EEVEE's ray tracing (`use_raytracing`, method SCREEN: `mkr/mkr/src/raytrace.rs`, `shaders/raytrace.wgsl`), measured
on the lab's rt_* cases against EEVEE's frames, its frames without ray tracing (`noraytrace`) and Cycles with bounces
(`cycles_gi`): each glossy lobe (reflection, coat) up to the scene's `trace_max_roughness` sends a ray per
sample, drawn from its GGX distribution of visible normals, through the sample's depth buffer (`trace_steps` 32,
denser near the surface, then a bisection; a ray going away from the camera hits a surface it crossed since its last
step, with `trace_cross` (0.25) of the scene's `screen_trace_thickness` to spare past it, one coming back toward the
camera a surface it is less than that thickness behind: below). A ray that finds a surface takes the light the
previous sample left there (the lighting pass keeps its lit picture, so light bounces from sample to sample as EEVEE's
radiance feedback does; the first sample's comes from a lighting pass of its own before it); one that finds nothing
takes the world's probe along the ray, blurred as a lobe half as rough (`trace_miss` 0.5: EEVEE's traced rough floors
are darker where they mirror the sky than its probe alone, by 1-5 levels at roughness 0.25-0.4; read along the mirror
direction mkr's were 2-3 levels too bright). The film's samples average what EEVEE's temporal accumulation leaves of
the rays' light (`rt_temporal`, below). Lobes rougher than `trace_max_roughness` fade from their ray into fast GI as
EEVEE's do (progressively), wholly fast GI's
from roughness `gi_rough` (1): EEVEE's open rough floor (lab `gi_rough`) keeps 0.75-0.81 of its probe-lit light at
roughness 0.5-0.6, 0.82-0.96 at 0.7-0.8 and 0.97 at 0.9-1. Blended surfaces are lit forward with the probe, as EEVEE
lights them. Fast GI (`fast_gi`, on: as the scene sets it; `raytrace.wgsl` `rt_scan`) is EEVEE's horizon scan: per
sample `fast_gi_ray_count` / `fast_gi_resolution`² slices of the directions round the view vector (at least one; a
sample's screen direction the same for every pixel, stratified over the samples, so neighbours read neighbouring
texels), each scanned both ways in `fast_gi_step_count` depth samples out to the screen's edge (or
`fast_gi_distance`), denser near the pixel; every sample hides the directions from its front to its back
(`fast_gi_thickness_near` farther along its camera ray, and at least `fast_gi_thickness_far` of angle past its front),
in 32 sectors of equal cosine-weighted measure (a visibility bitmask, after Therrien et al. 2023), the normal leaning
`fast_gi_bias` toward the camera. What stays open passes the world's diffuse light; the GLOBAL_ILLUMINATION method adds
what a sample hides first, lit as the previous sample left it, `gi_bounce` of it (1: the scene's method): one bounce of
each sample's direct and reflected light (the lighting pass keeps its picture without that bounce for the next sample's
rays; fed back, a closed space lit itself over the samples), and only off the side of a sample the camera sees when
that side faces the pixel (the G-buffer's true normal there), its light clamped by `clamp_surface_indirect` as a traced
ray's is (below). EEVEE draws both: the wall behind thin red strands gets none of their lit fronts' light (lab
`gi_strands`: 0.002 where mkr's bounce gave 0.057 without the facing rule, 0.005 with it), and the floor round
`rt_clamp`'s strips glows 6.5 times as much without the clamp, as the strips' 2.5, 10, 40 and 160 against 2.5, 10, 10
and 10 add up. Measured on the lab's white-world cases with their fastgi_ao, noraytrace and cycles_gi variants: a floor
2.5, 5.5 and 10.5 m from a 2.5 m wall keeps 0.78, 0.88 and 0.96 of its light under EEVEE's occlusion alone, its bounce
brings it to 1.05, 1.02 and 0.99 (Cycles' bounces: 0.89, 0.98, 1.0), a body 10 cm off the wall up to 1.4 times Cycles'
light; in the open cabin EEVEE's occlusion blackens the seat back yet leaves 0.19-0.27 on the walls and roof, where
Cycles leaves 0.04-0.09 (the screen does not show what closes the cabin). How EEVEE's fast GI depends on its settings
and the scene (the lit cases `gi_seat` and `gi_strands` and their `fastgi_*` variants, scene-linear, the bounce being
GLOBAL_ILLUMINATION's frame less AMBIENT_OCCLUSION_ONLY's): its picture's resolution moves neither the occlusion
anywhere nor the bounce of large surfaces (the seats' within 5 % at full, half and quarter resolution), but takes thin
strands' bounce off the wall to 0.35 of full resolution's at half and 0.28 at a quarter; the near thickness at 2 cm
(mk's 25) or the far one at 0.1 rad (pi / 4) move it by less than 10 %; the lights' shadows, the camera's clip range,
the picture's size, motion and the material (Diffuse, Principled, MMD's Diffuse-Glossy mix, a Transparent mix) not at
all. A strand's bounce falls with its distance from the wall (0.040, 0.032, 0.020 and 0.008 at 1, 3, 10 and 30 cm).
The seats gather 36 % of their own light near (1.4 times Cycles' bounce), and EEVEE's bounce off large surfaces is
1.1-2.1 times mkr's, more as the camera nears (the seats 1.8 at 1.3 m and 1.15 at 3.5 m, gi_wall 1.4-1.6, gi_cabin
1.1-1.6, rt_corner 2.1); yet on jwin's own frames (the same frames with and without the bounce, at 1080p and 540p, the
car's interior lights alone too) EEVEE's bounce is about 1 % of the picture (median 0.2-1.4 %, 95th percentile 4-9 %),
on the faces, hair and seats alike: what keeps it that low there is not found (not the settings above, which jwin
shares; EEVEE's full-resolution frames of jwin draw squares of other light and are no measure). mkr's bounce against
EEVEE's (mean over the pixels it lights), before the two rules -> with them: the lab's boxes 0.5-0.9 -> 0.4-0.55, the
strands 5-7 -> 1.5-2, the wall behind them 30 -> 3-4, two of jwin's interior frames 4.6 and 14 -> 0.9 and 2.1. Mean
\|error\| against EEVEE (levels, % of pixels over 8), the occlusion alone -> the bounce before -> the bounce now:
gi_seat 5.43 (31.7) -> 4.20 -> 2.59 (16.2), gi_strands 0.41 -> 0.72 -> 0.40, gi_wall 8.21 -> 3.23 -> 3.67 (11.2),
gi_cabin 17.04 -> 6.02 -> 7.66 (35.6), gi_rough 7.52 -> 5.68 -> 5.40 (31.3), rt_corner 14.44 -> 6.01 -> 6.54 (49.8),
rt_corner_ao 0.38, rt_world 3.29 -> 3.41 -> 2.41 (8.3), rt_rough 2.95 -> 2.86 -> 2.58 (8.8), rt_mirror 1.76 -> 1.33 ->
1.37 (3.7), rt_clamp 4.63 -> 13.70 -> 3.81 (18.0): the eleven's mean 6.00 -> 4.32 -> 3.35 (fast GI off, before the
scan: gi_wall 3.58, gi_cabin 22.69, gi_rough 10.18, rt_corner 13.75, rt_world 6.03, rt_rough 2.72, rt_mirror 1.44).
EEVEE itself is 2.6, 8.8, 7.4, 2.3 levels from Cycles on rt_mirror, rt_rough, rt_world, rt_corner; mkr still misses
EEVEE's 2-7 % darker open floor under the sky (rt_world) and half of its bounce off large surfaces. On jwin's frames
(26 per output, two a lit shot, 64 samples, motion blur), the same three: 16:9 2.53 (8.1 % over 8) -> 4.43 (20.5) ->
2.55 (7.6), over the figure 2.57 -> 4.49 -> 2.59; 9:16 2.06 (6.4) -> 3.88 (19.0) -> 2.15 (6.5), figure 2.09 -> 3.93 ->
2.18. Against the occlusion alone on the same tree the bounce costs jwin 0.02 and 0.09 levels (16:9's pixels over 8
fall from 8.1 to 7.6 %, 9:16's rise from 6.4 to 6.5 %), which the lab's gain outweighs: `gi_bounce` is 1 (the bounce
before was measured on the tree just before EEVEE's temporal accumulation came in, whose occlusion alone was 2.75 and
2.31). The bounce costs 10 and 7 ms of a jwin frame (GPU 489 -> 499 and 473 -> 480 ms); the scan itself 70 and 61 ms
of a frame of 64 samples (GPU 416 -> 486 and 408 -> 469 ms when it came in; two slices a sample were 107 and 95), the
rough lobes' rays included. A scene without ray tracing compiles none of it (`pipelines.rs`
`RT_TRACED`): Crimson's frames are the same bytes as before (10 frames, `RADV_DEBUG=nodcc`). A ray's light, found or
the probe's, is clamped as the scene's `clamp_surface_indirect` says (10 by Blender's default, jwin's too): scaled down
until its largest channel is the clamp (lab `rt_clamp`, strips glowing at 2.5-160 over a mirror and a wet asphalt: 7.70
levels unclamped, 4.61 with that rule, 4.99 by luminance, 4.84 per channel; no other rt_* or gi_* case moves). A point
or spot light is lit as EEVEE lights it, as a disk as wide as its sphere through its centre facing the point (on its
axis P / (4 pi (d² + r²)), where the sphere's own silhouette gives P / (4 pi d²): lab `point_near`, lights of radius
5 cm 0.1-0.8 m over a diffuse and a glossy floor, EEVEE within 0.2-2 % of Cycles there, mkr 22 % too bright under
the nearest before and within 1 % after; `point` 0.263 -> 0.246, `spot` 0.292 -> 0.288); one narrower than 1e-4 of
its distance (`shadow_soft_size` 0: jwin's headlights) is a disk that wide with its radiance spread to match, so it
falls off with the square of the distance (it lit jwin's whole road before: 14.7 levels on 16:9, 4.6 after).

jwin split by EEVEE's own frames per component (6 frames per output, no motion blur: the world, the moon with and
without its shadow, each light group, the emission alone, all rendered by EEVEE and by mkr from the same bake, both
scene-linear, per surface class from mkr's material indices): EEVEE's components add up to its frame within 0.2 %.
Before these two changes the frames were 2.70 levels off (bias +1.5): the wet asphalt 6.4 (+6.3, 8 % of the pixels),
the car's interior 3.6 (+2.8, 16 %), the seats 2.8 (+2.3), car paint 2.5 (+1.1), the eyes -2.4; skin, hair and clothes
within 1.3. The interior's excess was the dash glow (a point light of radius 5 cm about 0.2 m from the wheel's hub:
x2.65 there, x1.81 over the frame, the four interior area lights within 1 %); the asphalt's the wet coat's traced
reflections of the lamps', cars' and signs' emission (x1.5-2.6, in the emission alone; EEVEE's clamp takes its own
down by up to x1.58 there). Still open then: mkr's traced reflections of bright emitters 1.25-1.4 times EEVEE's on
`rt_clamp` (split below), the world on the seats x1.2-2.6 (fast GI's occlusion), the eyes x0.86, and the mixed
pixels (edges, depth of field, hashed alpha).
Over jwin's 26 frames per output (two a lit shot, 64 samples, motion blur): 16:9 3.05 (10.2 % over 8) -> 2.96 (10.0)
with the clamp -> 2.75 (9.3) with the disk, over the figure 3.09 -> 3.01 -> 2.80; 9:16 2.64 (8.7) -> 2.50 (8.5) ->
2.31 (7.7), figure 2.67 -> 2.54 -> 2.34; GPU 484 and 467 ms a frame either way. Crimson's frames are the same bytes.

Reflections of emitters, measured on the lab's `rt_emit` (one width of white card a frame, 2, 6, 20 and 60 cm across,
under the clamp, over a black dielectric floor and a Glossy one at roughness 0, 0.1, 0.2 and 0.4) with EEVEE's
denoiser stages and trace settings changed one at a time (variants `nodenoise`, `notemporal`, `nobilateral`,
`trace_thin`, `trace_thick`, `trace_coarse`, `trace_fine`) and Cycles with bounces, each card's light summed over the
floor. Three things make EEVEE's darker than mkr's were. The hit: EEVEE draws the same reflections at thicknesses 0.05
and 0.2 (more at 1), on the mirror within 5 % of Cycles' for the 20 and 60 cm cards; mkr counted every ray whose
stretch since its last step overlapped the whole thickness, so rays passing under and behind a card took its light,
11-15 % more than EEVEE's undenoised frames on the 20 and 60 cm cards and up to 1.6 times on the thin ones. With
`trace_cross` (rays going away from the camera keep a quarter of the thickness past a surface they crossed; rays
coming back toward it, a ball's lower half mirroring the floor, the whole thickness: a quarter took `rt_world` from
3.23 to 3.51) the wide cards' light is within 1.5 % of EEVEE's undenoised frames up to roughness 0.2. The denoiser:
its temporal accumulation (off, the frames are the undenoised ones, the bilateral filter then changing nothing) and
its bilateral filter take light away where few rays find the emitter, the edges and tails of a reflection and narrow
ones: at roughness 0.1-0.2, 6-17 % of the 20 and 60 cm cards' light and 26-47 % of the 2 and 6 cm cards', 1-8 % on the
mirror. The trace of rough lobes misses thin emitters: at roughness 0.4 EEVEE's undenoised frames find none of the
2 cm card's light and a third to a half of the 6 cm card's (Cycles all of it, mkr most of it), more at precision 1
(`trace_fine`) and less at 0 (`trace_coarse`).
On `rt_clamp`'s mirror the white strips' reflections were within 2 % of EEVEE's; the excess was the red strips hidden
behind them (mkr's rays took the white strips' light through the thickness) and, on the rough floors, the denoiser's
loss; the glow round the strips is fast GI's bounce on the floor's diffuse base (none with `fastgi_ao`, 6 times
brighter with `noclamp`: EEVEE clamps the bounce too). Mean \|error\| before -> after `trace_cross`: rt_emit 1.50 ->
1.31, rt_clamp 4.61 -> 4.29, rt_mirror 1.96 -> 1.73, rt_rough 3.29 -> 3.19, gi_rough 7.61 -> 7.59, rt_world 3.23 ->
3.25, the other rt_* and gi_* cases the same; jwin's 26 frames per output 16:9 2.75 (9.3 % over 8) -> 2.70 (9.0),
figure 2.80 -> 2.74, 9:16 2.31 (7.7) -> 2.26 (7.5), figure 2.34 -> 2.29, GPU 483 -> 487 and 466 -> 469 ms a frame;
Crimson's frames the same bytes. EEVEE's temporal accumulation (`raytrace.wgsl` `rt_temporal`, when the scene's
`use_denoise` and `denoise_temporal` are on, as Blender sets them): the lighting pass keeps the traced lobes' light
apart and blends it per pixel into a history over the frame's samples (two more pictures beside the traced light's:
the result and the raw light's luminance), the history clamped in luminance to the range of the previous sample's raw
light over the 3 x 3 pixels round it and of this sample's own, at 1 / (n + 1) on sample n; the next sample's rays still
find the raw light. Where few rays find an emitter that range is often dark and the history loses its light, as
EEVEE's does: rt_emit's 6-60 cm cards' light within 2 % of EEVEE's frames without its bilateral filter (`nobilateral`)
at roughness 0.2, the 2 and 6 cm cards 8-15 % brighter at 0.1, the thin cards at 0.4 mostly gone as in EEVEE's. Mean
\|error\| before -> after: rt_emit 1.31 -> 0.55, rt_rough 3.19 -> 2.95, rt_mirror 1.73 -> 1.76 and rt_world 3.25 ->
3.29 (a little darker where the mirror and the balls reflect the box and the sky), rt_clamp 4.29 -> 4.63: its floor
lacked fast GI's bounce glow (`gi_bounce` was 0 then), which mkr's brighter reflections hid: against EEVEE's frames
without the bounce (`fastgi_ao`) it goes 1.57 -> 1.03, and with the clamped bounce on since (`gi_bounce` 1) it is 3.81
against EEVEE's own; gi_rough 7.59 -> 7.52, the other rt_* and gi_* cases the same.
jwin's 26 frames per output: 16:9 2.70 (9.0 % over 8) -> 2.53 (8.1), figure 2.74 -> 2.57, 9:16 2.26 (7.5) -> 2.06
(6.4), figure 2.29 -> 2.09, GPU 487 -> 489-491 and 469 -> 471-474 ms a frame (two runs); Crimson's frames the same
bytes (`RADV_DEBUG=nodcc`). Still open: the bilateral filter's further loss (EEVEE's frames 3-22 % darker than
`nobilateral` at roughness 0.1-0.2) and the rough lobes' missed thin emitters in the trace itself.

Many lights (jwin_mv: 74, a road lamp every 50 m, tunnel lights, cars' beams): a frame of more than eight
(`pipelines.rs` `TILED_FROM`) culls them per 8 x 8 tile first (`shaders/deferred.wgsl` `light_pass_tiled`): the tile
bounds its pixels' surface points and keeps, a bit per light, the lights that may reach that box (inside the influence
radius, in front of a one-sided area light, inside a spot's cone: `light.wgsl` `light_reaches_box`, each test with a
margin over its rounding); its pixels run only those, in the order every light runs, so the picture is the same to the
bit. Closures no light lights (emission and transparency alone: jwin's glows and halos) skip the lights altogether, in
the lighting pass and the blended one. On jwin (7 frames per output, 64 samples, motion blur) the lighting pass went
from 272 to 250 ms a frame on 16:9 and from 265 to 246 on 9:16, the blended pass from 55 to 14 and 52 to 12.5 (GPU 466
-> 404 and 448 -> 390 ms); what is left of the lighting is the lights each pixel does reach (their lobes over a 32-gon
for spot and point lights) and the per-pixel rest (the moon's shadow, the world, the traced reflections: 134 ms with
the moon alone). Crimson's six lights keep the plain pass: its frames are the same bytes, its lighting the same 21 and
35 ms. With the lights' own distances (the road lamps reach 90 m, not the threshold's 66; the tunnel lights 45, not 29)
jwin's tunnel shots on 16:9 went from 4.7-4.8 to 3.2-3.4 levels; over 26 frames, two a lit shot: 16:9 4.69 -> 4.53
(21.1 -> 20.2 % over 8; over the figure 4.75 -> 4.59), 9:16 3.95 -> 3.96 (18.1 %; figure 3.99 -> 4.01). Split by drawing
the frames again without each group of lights, the world's diffuse light or the shadows, mkr's frames were 3.3-3.6
levels too bright on average, and the error fit (regressed on each group's share of the picture) as: the moon (the
scene's sun) where EEVEE shows none of it, 1.2 levels: its one map was fitted to every caster, a 24 km ground plane
among them (17 m texels), and its shadow ray sampled that map's depth span of kilometres, its nearest steps hundreds of
metres from the receiver, so the car kept no shadow of its own (`shadow_mode: "rays"`: 16:9 4.53 -> 4.09, 9:16 3.96 ->
3.53). The sun's clipmaps above bring it to 4.16 and 3.60 (18.1 and 15.5 % over 8; over the figure 4.21 and 3.64) for
24 ms of GPU a frame (lighting 250 -> 273 ms, shadow maps 0.3 -> 1.7); the lab's `sun` 0.28 -> 0.26, `rt_mirror` and
`rt_rough` unchanged, a scene without a shadowed sun compiles the code of one map (Crimson: the same bytes and times).
What is left: the world's diffuse light where it alone lights a pixel, a quarter too bright (EEVEE's horizon scan
occludes it in the car: fast GI, above, now draws that scan; its occlusion alone takes these frames to 3.05 and 2.64,
the scene's GLOBAL_ILLUMINATION bounce on top first to 4.83 and 4.30, then, off camera-facing sides and clamped, within
0.02 and 0.09 of the occlusion alone),
and 1.4-1.7 levels over every pixel that no group explains: not the view, film, render or compositor settings (jwin's are
Crimson's but for ray tracing, and its compositor tree has no nodes), not the glows' emission (scaling it only adds
error). The car's interior lights, the road lamps, the tunnel lights and the beams fit within 2-25 % of their own
light.

Depth of field (a shot's `dof`; `mkr/mkr/src/dof.rs`, `camera.rs`): by default each sample looks from its own point of
the aperture through the focus plane (the camera moved there and its frame sheared so the focus plane keeps its place:
a thin lens, what EEVEE's jittered camera and Cycles converge on), the aperture's points a sunflower over the disk
mapped area-evenly onto the blades' polygon (turned by the rotation, narrowed across by the ratio), ordered over the
samples apart from their jitter and shutter time; the rigid-mesh cull widens each frustum side by the aperture.
`--params` `dof: "eevee"` follows the scene's EEVEE scheme instead: its post process (an emulation, `shaders/dof.wgsl`:
the half-resolution copy that favours the widest blur and dark colours, gathered layers, slight focus, run once on the
film), or with `use_bokeh_jittered` the jittered camera over most of the aperture and the post process over the rest;
`dof: "off"` draws the pinhole. An orthographic camera's depth of field stays Blender's (`mkr info`), and so do a
look's EEVEE passes through a camera with depth of field. EEVEE's own post process is far from a lens: on the lab's
`dof` case it is 3.66 levels mean \|error\| (11.0 % of pixels over 8) from Cycles' and 4.19 (17.3 %) from its own
jittered camera; it drops small bright highlights and thin details beside a wider blur and darkens what it blurs.
mkr's lens is 1.03 (1.6 %) from Cycles and 3.79 (11.1 %) from EEVEE's default frames, the emulation 3.41 (11.5 %);
on `dof_psf` the lens 3.06 (8.5 %), the emulation 3.70 (9.4 %). On
a lit MV's DOF shots (36 frames per output, 16:9 and 9:16, final settings; Blender's frames again with every camera's
depth of field off isolate it from the scene's other gaps) the error the depth of field adds is 0.43 and 0.27 levels
with the lens, 0.51 and 0.31 with the emulation, 0.98 and 0.80 drawing none: the lens is the default. It costs nothing
measurable per frame (the samples are there anyway); the emulation 1.2-1.3 ms of GPU at 1080p (`gpu_ms_phases` "dof").

#### mkr's layers

With `engine = "mkr"` mkr (`mkr/`, mk-mmd's renderer of the lit scene from the bake) draws the layers beside its own
frames too, so a frame needs no Blender at all: `mk render` resolves the layers' object patterns on the bake's records
(`mkmmd/render/mkr.py`), `mkr info --layers` sorts every layer of the frames between mkr and Blender, and `mkr render
--layers` draws mkr's share into the same files Blender would write (`mkr/mkr/src/layers.rs`, `layers_gpu.rs`):

| Layer | How mkr draws it |
|---|---|
| `plate/<shot>/`, `bare/<key>/` | a lit frame like the cut's, through the other shot's camera, or with the glitch's objects hidden from the camera and from the shadows |
| `figure/<key>/` | the objects' coverage as Workbench antialiases it: its 16 sample points and their weights, measured on probe scenes Blender rendered (a sample counts for a pixel by the Blackman-Harris weight of twice its offset), and the curve its film bends the weighted coverage through (an edge loses about 0.13 px) |
| `point/<key>/` | the bake's world point (`lit.points`) through the shot's camera, `world_to_camera_view`'s numbers |
| `keep/` | EEVEE's flat emission pass: kept materials white, the others black, an MMD material transparent by its own alpha (hashed or blended as in the frame), everything else opaque |
| `screen/` | the screen type's meshes through the screen camera, lights off, the Standard view's LUT on the straight colour; the GPU holds only the words on screen on the frame being drawn (a project's whole screen type can run to tens of millions of vertices) |

Blender keeps what mkr cannot draw: anything seen through a reflection, a fisheye, and what the bake could not hold
(screen type over the bake's cap: `lit.screen.left_out`); `mkr info` names each such layer and why (`unsupported.layers`
lists the kinds). A Blender job starts only when something is left for it: frames mkr cannot draw, layers it cannot
draw, the screen type and keep masks of its frames it leaves (`mk_screen`, `mk_keep` tell the job which mkr draws). The
report adds per output `blender_layers` (items and screen layers left to Blender) and `blender_jobs`, and mkr's own
report counts its `layers` and `looks`.

On Crimson and Clover (900 frames, three outputs, with a `[post] keep` and a glitch added for the test) `mk render
--engine mkr --preset final` left Blender one job per output, for the eight silhouette frames and their own layers, and
wrote every file Blender would have (900 frames, 1921 transition and effect layers, 900 keep masks, 861 screen layers per
output) in 409, 311 and 376 s for 16:9, 1:1 and 9:16 (the draft: 119, 137 and 214 s). Against Blender's own files, all
three outputs: figures (3702) 0.006 to 0.010 levels mean |error|, 0.009 to 0.014 % of pixels off by more than 8; points
(1635) the same numbers to the files' six decimals; screen type (2583) 0.04 to 0.09 levels, 0.18 to 0.34 % over 8 (the
edges of the letters); plates (408) 0.10 to 0.18 levels, 0.31 to 0.61 % over 8, and bare frames (6 of 16:9, against a
Blender render of the same glitch) 0.18 levels and 0.78 %, as close as mkr's lit frames; keep masks (15 frames of 16:9)
0.002 levels, 0.006 % over 8. Each pass draws its samples into the layers of a texture array and one compute pass spreads
them with the pass's filter, in order, and resolves them to the file's bytes (`shaders/layer_film.wgsl`; a grey byte per
pixel for a figure or keep mask, widened to the file's channels by the PNG writers), read back through a ring of eight
buffers while the GPU draws the next passes; the screen type's words of a frame are rebuilt in parallel. On Crimson
and Clover's whole clip (1230 figures and 853 screen layers per output) the flat passes take 7.5, 6.0 and 7.6 s for 16:9,
1:1 and 9:16 (28.1, 20.9 and 27.9 s when each sample had a film pass of its own), every file the same pixels; a keep mask
takes about 9 ms.

#### mkr's looks

A frame or layer in a silhouette or vector look (a shot's or a [[medium]] window's: the cut's frames, plates and bare
frames seen through one, mattes and their backs) is mkr's too. `mk render` sorts each look's objects on the bake's
records as styles.Looks sorts the scene's (`mkmmd/render/mkr.py`: `look_jobs`, a figure seen through a look loses what
the look hides) and hands mkr one job per picture; mkr draws the passes Blender's job renders (`mkr/mkr/src/looks.rs`,
`looks_gpu.rs`) into a spool in memory, and worker processes compose them with the same `compose_silhouette` /
`compose_vector`, rings and type as Blender's frames (`mkmmd/render/compose.py`: `finish_look`) while mkr draws the next:

| Pass | How mkr draws it |
|---|---|
| `flat`, `matte` | Workbench's flat light: the subject and hard accents in their colours (linear, as the look sets `Object.color`), depth-tested; a matte the subject's coverage at twice the frame's size (in a vector look too). Workbench's antialiasing exactly: its 16 sample points (BLI_jitter_init), the Blackman-Harris weight of twice each one's offset, summed in log2 in half floats rounded toward zero (what gives a whole pixel 254 of 255 and a flat #1f1d2e 32 30 47), through the Standard view the bake captures (`looks.lut`) |
| `soft`, `type`, `knock` | EEVEE with the lights off: the soft accents' coverage and the opaque share of their Mix Shader (each soft material compiled once more with styles.add_weight_aov's AOV), the type's emission held out by the subject and hard accents, the knock-out objects; the look's samples and EEVEE's film filter |
| `screen_knock` | a silhouette frame's knock-out screen type, reversed out of its figure: drawn as the screen layer draws the rest of the screen type (through the screen camera, lights off), at the screen layer's samples |
| `id`, `depth`, `shade`, `tex` | the vector look at twice the frame's size, as the Chromium engine draws them: the material ids and planar depth without antialiasing, the studio light with the stencil shadow volumes of the look's light, the texture colour, the last two through Workbench's antialiasing |

The bake adds what the looks read: each object's `soft` flag, the image of each material's Workbench texture, the tint
holders' values on every frame (`looks.tints`) and the Standard view (`looks_lut.bin`). Blender keeps a reflection, a
silhouette frame whose look's samples differ from the screen layer's while knock-out screen type is on it, screen type the
bake could not hold, a soft accent drawn BLENDED, instances in a silhouette's flat colours, and a look's EEVEE passes
through a camera with depth of field. Against Blender's own frames: Crimson's two silhouettes (24 final frames, three
outputs) 0.003 to 0.004 levels mean |error|, under 0.01 % of pixels off by more than 8; Samurai's draft (60 frames, three
outputs) silhouettes 0.01 to 0.02 levels, vector close-ups with drawn tones 0.08 to 0.11 levels (0.11 to 0.14 % over 8),
medium windows (sketch and cel) 0.20 to 0.34 levels (0.10 to 0.30 %); Playing Possum's final 16:9 silhouettes and
sketches 0.002 levels. A silhouette frame costs about 0.3 s of wall time with eight workers composing.

### Post

`mk post` composites the cut effects over the cut's frames, draws the lens streaks (`streaks`, below) from the result, lays the screen
type over it, turns the frames of the loop's wrap over frame 0 as finished, then grades the frames and encodes
`<project>/out/<name>_<output>[_<preset>].mp4` (the preset is left out of the name for `final`) with `[audio]` (`file`, `start` = song
seconds at clip time 0): H.264 (x264, `--crf` 15, yuv420p, even size) and AAC 320k with a 60 ms fade-out. A clip of one frame
(`[project] duration` of one frame) is a still: each output's graded frame is written as `<name>_<output>[_<preset>].png`, at
its own size, with no sound (the sim and cloth reports then measure its frame and the two before). A missing frame or layer is an error
unless `--allow-gaps` (the previous frame repeats; the plain cut shows where a layer is missing); `--to DIR` also copies the videos;
`mk look` composes the same way for the images it shows. The report has each video's frames, size, `screen_type_frames`, and the darkest luma
after the grade (`min_luma`), the encoder, the processes (`jobs`) and whether the GPU finished frames (`gpu`: true, or why not).

With `[look] breathe` ([Lights and the look](#lights-and-the-look)) every colour post takes from the palette is resolved
with both palettes and mixed by each frame's mix (`mkmmd.post.Breath`): the cut's plan is made with each palette and on
every frame each of its entries (a transition, an insert, a glitch, an `[[fx]]` entry, the loop's wrap) is the first's
with whatever differs mixed that far toward the second's (`mkmmd.core.breathe.Pair`: colours as display-space floats,
as `"a:b:t"` mixes them in sRGB; the same object at mix 0 and 1, copied only where it differs between them), and the
grade's colours with it: the floor mixed between the two palettes' floors (their darkest slots), the split toning's
colours, the halation's and the streaks' tints (`mkmmd.post.Mixed`). Footage and photographs keep their own colours.
What an effect keeps from frame to frame is kept by the colours it was drawn in; mkr-post gets each frame's grade in its
job, and a mandala's shape tiles in colours strictly between the two palettes' are written for their frame alone and
removed by mkr-post once read (`FrameFx.passing`, Tile::File `once`), so however long a breath, the scratch folder holds
the two ends' tiles and those of the frames in flight. An entry that is not the same thing with both palettes (a
string, a count or a key only one of them has) is a usage error that names it, before anything is drawn. `mk look`
previews the effects in the same colours.

The frames are finished by worker processes over chunks of about a dozen consecutive frames and handed to the encoder in
order (`mkmmd/postframes.py`): `--jobs N` or `[post] jobs` (default: up to 4, fewer when half the memory available would
not hold them at a worker's peak, about 1.7 GB at 1920 x 1080; one process for a little work). They are the frames one
process makes in order: a missing frame repeats the last one rendered before it in any chunk, the wrap's frame 0 as
finished is finished afresh by a worker that needs it, and a chunk only starts on a frame where no effect holds a picture
of the frames before (a film's frame over its step, a booth's captures, an eraser's frame, a mosh with a flow:
`FrameFx.safe`). Two things remain that can move a level here and there: OpenCV and numpy round a pixel differently when
a picture's memory is aligned otherwise (the same pass started at another frame shows it too), and footage relies on
OpenCV seeking to the exact frame (checked on every clip frame Crimson and Clover shows). On Crimson and Clover's final
16:9 cut (900 frames, four workers) 883 frames are byte for byte today's; of the 17 others, those checked differ in one
pixel by one level.

The encoder is `[post] encoder` or `--encoder`: `x264` (the default: libx264 at `[post] x264_preset` or `--x264-preset`,
`slow` by default, CRF `--crf`) or `vaapi` (the GPU's H.264 encoder on the config's `vaapi_device`, constant QP `--crf`:
some 250 frames a second at 1080p against x264 slow's 14, larger files).

`--gpu` or `[post] gpu = true` hands what mkr-post can do of each frame to the GPU (`mkr/mkr-post`, driven by
`mkmmd/postgpu.py`): the frame effects it draws, the screen type (the ordinary layer over the picture, then a frame's negative
layer, `negative/<frame>.png`, inverting it: the job's `layer` and `negative`, kernels `composite` and `negate`), the keep mask, the grade and the bytes for the encoder,
the rest of a frame (cut effects, the other kinds, entries before the last one it
cannot draw) worked out by the workers and handed on as a picture; the loop's wrap, `halation` and `streaks` stay on the
CPU (with either, the whole output does). Each kind it draws has a module in `mkmmd/gpufx/` that works the frame's
numbers out with the same `FrameFx` methods the CPU path uses, and `mkr/mkr-post/src/fx/<kind>.rs` with
`shaders/fx_<kind>.wgsl` draws the pixels; what an effect keeps from frame to frame (a mosh's picture, a film's print)
stays on the GPU, and a frame whose keeping started on the CPU stays there. A picture or layer that is not an 8-bit PNG
of the output's size (the CPU path resamples it) leaves its entry to the CPU. mkr-post writes each finished frame
straight into the encoder's pipe (`serve --stats FD`: its index and darkest luma come back on a pipe of their own), a
frame finished on the CPU handed through it in order (`finished` jobs). The worker processes (`--jobs`, default up to
3 on this path) only plan the frames, up to 32 ahead of mkr-post, and plan a heavy frame (one whose picture they work
out whole: the entries before a pixel sort, some fifty times a plain frame's work) long before its turn, from the
output's start, on one worker fewer than `--jobs` at most and with 512 MB of such pictures waiting at most
(`postframes.EARLY_BYTES`), so the frames round it never wait for it; a group of consecutive heavy frames is cut into
one run for each of those workers, so each reads its clips on from frame to frame rather than seeking them afresh. A
thread of mk post's own sends mkr-post the jobs in order, a footage op's clip frame decoded and fitted there first (one
decoder a clip, read in order, as the CPU path reads them), while the main thread takes the planned frames from the
workers and mkr-post's answers. A clip frame decoded for one output is fitted on a thread of its own for the outputs
after it whose entry shows the same clip on the same frames (once their first frame says their size), into rgb24 files
under `.mk/cache/post/` (on disk, a quarter of its free space at most) that mkr-post reads and leaves until the run
ends, so a run decodes each clip frame once, not once an output (`gpufx.footage.Kept`). A scope's stems are read once a
process, not once an output. mkr-post decodes the PNGs a frame reads on four threads, each row packed into the words
the GPU takes as it comes out of the decoder (a figure layer's luma as one byte a pixel, widened on the GPU), and those
threads write the frame, its screen layer and its keep mask into buffers the GPU copies from, so the thread that hands
the frames' work to the GPU copies none of them (`engine::inputs`). The outputs go one after another (one mkr-post on
the card at a time: `mkmmd/gpulock.py`), sharing the workers; each output's setup (its pipeline built, its files
checked) is worked out on a thread while the output before it is finished (the first one's while the GPU is waited
for), and each encoder writes its last frames while the next output starts. The kinds, and what else of them stays on
the CPU:

| Kinds | On the CPU |
|---|---|
| `zoom`, `lens`, `push`, `shock`, `split`, `slab`, `melt`, `kaleido`, `tunnel` (with a `mask` too: the hole filled, the figure laid back as a sticker) | |
| `echo`, `slitscan`, `trails`, `scope` (its trace drawn on the CPU) | a slit scan whose ages do not run one way |
| `mosh` (the cut's motion, or a flow's moves, worked out on the CPU) | a frame the cut has not rendered; a flow whose frame before went the CPU's way |
| `pixelsort` | lines over 4096 px; and it is always the first op of its frame (below) |
| `footage` (the clip decoded and fitted by OpenCV as the CPU path reads it, in the planning process, and handed over as rgb24: an enlarging fit's bicubic is IPP's, which no kernel would match) | a pattern round a contained clip |
| `liquid`, `flood`, `spiral`, `tint`, `invert`, `halftone` | |
| `shapes` (the cells, their deal and each pixel's cell worked out by the workers, the box blur at each cell's middle and the shapes drawn by mkr-post; the layout before too while a new one pops in), `key` (a `burn`'s frames read from the cut's PNGs and keyed again at their own tolerance, a `clip` back decoded and fitted as `footage`'s is, a `delay` frame read as an echo's) | a figure layer, or a cut frame a `burn` or `delay` reads, that is not an 8-bit PNG of the output's size |
| `mandala` (sprite, shape and figure rings, the centre figure, pattern backs, googly eyes and smiles) | a figure moved by `keys`, `at` or `height`; a silkscreen fill |
| `paintapp` that is only stamps (`ui = false`, `cursor = false`, no `bucket`, `eraser` or `undo`: each stamp a sticker in the order it landed, popping, turning, drifting and fading, with its die-cut edge, shadow, googly eyes and smile; the pupils' and sparkles' coverages drawn by the workers, as the CPU path draws them; an `aspect` table's own `at` too) | a paint app with the chrome, a cursor, a `bucket`, an `eraser` or an `undo` (the entry stays on the CPU, every frame of it) |
| `grid` (colour and picture backs, silkscreen fills) | pattern backs |
| `stack` | wallpaper bands, era looks, squash, silkscreen and copies under 0.75 |
| `doodle`, `label`, `camcorder` (strokes, text, leaders, the tape's random parts and its stamp drawn by the workers), `film` (in a `rect` too: its box printed as a picture of its own) | |

An entry limited to a `rect` goes to mkr-post whatever its kind: its ops are bracketed by two generic `rect` ops
(`mkr/mkr-post/src/fx/rect.rs`, `shaders/fx_rect.wgsl`; `mkmmd.postgpu.ops`): `begin` keeps the picture as it reaches the entry,
`end` puts it back outside the rect, `before + (after - before) * m` with `m` the rect's ramps (`picturefx.rect_ramps`, rows
then columns over the whole frame, written once to a file named by their bytes and kept), as `FrameFx._apply` keeps it. A
`film` still prints the rect's box itself. The frames' clips (a `footage` entry's, a `key`'s) are decoded and fitted by OpenCV
in the process that sends the jobs, one thread, as the CPU path reads them; `mkmmd.footage.source` keeps as many decoders
open as fit 600 MB (`OPEN_BYTES`: a clip counts 36 bytes a pixel, 75 MB at 1080p, 300 MB at 4K; the last two always), so that
four clips shown together (a peak of four keys, each over a clip of its own) are not reopened, and seeked from a key frame,
on every frame: that thrash was 1.5 s a frame, the whole of a 243 s post of bros' 773 draft frames, which now takes 20 s.

Its kernels follow the numpy and OpenCV code they replace: warps and remaps sample at float coordinates with fused
multiply-adds as OpenCV 5 does (bit for bit), sorts pack their keys as numpy does, the vignette, the grain and the 1/255
table come from numpy. Each kind matches the CPU path within a level on every frame of the projects that use it (Crimson
and Clover, Samurai, Vega, Playing Possum), and `tests/test_gpu_*.py` hold every path to it on synthetic pictures. Over
Crimson and Clover's 900 final 16:9 frames, every entry but those before its pixel sort drawn by mkr-post, no pixel is
off by more than one level from the CPU path (at most 16,609 pixels a frame off by one, 671 on the median frame). A pixel
sort is chaotic in its input: a key a float's last bits higher can land in the next of its 16-bit steps, or a pixel cross
a run's threshold, and pixels move along their runs (1e-6 of noise in the picture moved thousands of pixels of a Crimson
frame by up to 99 levels). Effects drawn on the GPU before it, each within a level, would hand it another picture, so
on a frame it sorts the entries before it are worked out by the workers and mkr-post sorts exactly the CPU path's
picture (`postgpu.first`). Each frame's bytes are taken back and written out on a thread of their own while the GPU goes
on with the next (three frames in flight). Nothing of a frame is read back while it is drawn but the box of a figure
mask's sticker (the figure as the remap left it, which only the GPU has) and the support of a stack's figure warped by a
map that turns: a figure layer's coverage is summarised as it is read (each row's and column's largest byte, a bit per
covered pixel: `fx::Cover`, on the reader threads), and the boxes a masked remap fills, a footage lays its figure over
and a stack cuts its layers from come from it, through a map that keeps the axes exactly as the warp kernel samples
(`figure::carried_support`); a fill's pushpull runs only when the GPU finds a hole pixel no ground mirrors into (its
passes dispatched indirectly, all or none of their workgroups), a masked melt that moves nothing is put back on the
GPU, and a film's grain spread is folded in 64-bit floats on the GPU where the device has them (the row sums read back
and folded on the CPU otherwise). Coverages, rings and shadows are single-float planes (`fx/plane.rs`), worked out over
the box they reach with the frame's coordinates and borders (a pasted figure's: every pixel as over the whole frame),
and a Gaussian blur reads each run of a row into a workgroup's memory once, the same taps summed in the same order.
With `MKR_POST_PROFILE=1` mkr-post logs each frame's time waiting for its files, drawing and handing it on, each op's
CPU time and each file read, and GPU timestamps after every op and its stages (`profile gpu frame K step=ms ...`,
without holding the frames back), as `profile ...` lines on its stderr, which mk post keeps in `mkr-post.log` in the
run's scratch folder until it ends; `MKR_POST_PROFILE=passes` adds a timestamp after every pass, `k_<kernel>`.

On Crimson and Clover's three outputs (900 final frames each, VAAPI encode) `mk post --gpu` took 76 to 82 s on this
machine while other jobs kept its load at 6 to 21 (31 to 33, 16 and 24 to 26 s for 16:9, 1:1 and 9:16; every frame
within a level of the build before, 883 to 894 of each output's 900 identical), against 90 and 102 s for that build in
the same GPU turns, once each clip frame was decoded once a run, consecutive heavy frames planned in one run a worker,
figure layers sent to the card at a byte a pixel, the frames' files staged for the card by the threads that read them
and each output's setup worked out while the output before it was finished; 187 s with the first mkr-post on a quiet
machine. The CPU sets the pace, not the card (busy 49 to 52 %): about 200 s in mkr-post (most of it decoding PNGs: some
3,100 an output, and its one thread that hands each frame's work to the card, some 20 ms a frame), 80 s in mk post's
own process (most of it decoding the clips' frames, 698 an output, once for the three), 150 s in the workers (about
half of it the 18 heavy frames an output) and 46 s in ffmpeg (the encoder's YUV made from the RGB frames). With the
final sampling ([mkr](#mkr): its samples), `mk render --engine mkr --preset final --post --post-args "--gpu --encoder
vaapi"` finishes the three videos in 227 s on a quiet machine, against 5508 s in Blender and numpy (3292 s of render,
2216 of post); run one after the other, the render takes 197 s (60, 50 and 83 s for 16:9, 1:1 and 9:16; 47, 39 and
68 ms of GPU a lit frame) and `mk post --gpu` 64 s (30, 13 and 21 s).

`--follow` takes the frames as a running `mk render` writes them. Before it draws, mk render writes each output folder's
manifest, `render.json` (`mkmmd/core/progress.py`): its process, the clip frames it leaves there and, when the engine can
say (mkr's can), the files every frame gets: the frame, its layers (the cut effects' and the collages' items), its screen
type where the frame has any, its keep mask. It marks the manifest done when it stops, with the error that stopped it.
mk post then plans each frame once every file promised for it and for the frames before it is all there: a PNG once it
ends with its IEND chunk (Blender writes a frame over its claim in place, mkr renames a finished file into place), a
point once it parses (a heavy frame is planned before its turn only once its files are there). A cut effect that
settles its scale over its window (an expand, a collapse, a slice) waits for the whole window first. A render that
stops without a promised file is an error naming it, and a manifest whose process is gone counts as stopped. Without a
manifest (another engine, no render running) it waits a minute for one, then takes the folder as it is. `mk render
--post` starts `mk post --preset P --follow` for the same outputs alongside the render, inside the render's own hold of
the GPU lock (the two are one job: mkr and one mkr-post on the card; a `mk post --follow` started on its own waits for
the render's hold to end), `--post-args "..."` adding its flags (e.g. `--gpu --encoder vaapi`), and adds its report
under `post`. mkr draws an output 48 frames at a time (`mkr/mkr/src/main.rs`): those frames' lit frames, then their
flat layers and their looks' passes (`Renderer::draw_flats` with the layer kit made once), so a frame's files land a
chunk after it is drawn, the files the same bytes as when every lit frame came first.

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
Negative screen type (`negative = true`, [Screen type](#screen-type)) goes the same way: it inverts the picture under the grade (the
inverted picture is graded) or, with `"over"`, the finished, graded one.

`[post]` (all optional) is otherwise the grade, applied after the screen type in this order:

| Key | Meaning |
|---|---|
| `halation` | `{strength 0.22, radius 0.012, threshold 0.72, tint "rose"}`: glow around highlights (radius as a fraction of the frame height), tinted like film halation; off unless the table has a key |
| `saturation` | 1.06 |
| `keep` | `{cast, materials, saturation 1.0}`: those materials of a cast member (PMX names; her irises: `["瞳"]`) keep their colour where they are seen, at `saturation` of the frame's own instead of the grade's (a black-and-white picture with only her eyes in colour: `saturation = 0.0` beside it). `mk render` draws the mask beside each frame, `keep/<frame>.png`: an EEVEE pass with every material a flat emission, white for the kept ones and black for the rest, so what stands in front hides them as in the picture (MMD materials see-through by their own alpha, texture included; anything else opaque), linear, antialiased. Screen type laid on under the grade is graded as type; inside a cut effect's window or an `[[fx]]` the frame is graded whole, and at the loop's wrap the masks of the page and of frame 0 under it turn with the page; a missing mask is an error unless `--allow-gaps`; the report counts `keep_frames`. An unknown member, key or material is an error that lists what there is |
| `contrast`, `pivot` | 1.10 around the pivot 0.60: midtone contrast |
| `split` | `{shadows "iris", highlights "gold", amount 0.06}`: split toning toward palette slots; off unless the table has a key |
| `vignette` | 0.12: corner darkening (a fraction) |
| `floor`, `toe` | the palette's darkest slot (`base` in the dark Rosé Pines, `text` in Dawn), 0.12: the darkest colour allowed (a palette slot or `#hex`; `false` for none): luminance below the floor plus the toe is lifted smoothly into that band and tinted toward the floor colour, so nothing in the frame is black |
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

### Parity

`mk parity` measures how close mkr's frames are to Blender's (`mkmmd/render/parity.py`; the lab in `mkmmd/render/lab.py` and
the Blender ops `lab_build` and `lab_variant`, `mkmmd/blender/ops_lab.py`). Every number is in 8-bit levels over the pixels
measured (a 16-bit PNG counts as levels with fractions, value x 255 / 65535):

| Metric | Meaning |
|---|---|
| `mean_abs` | mean \|test - reference\| over pixels and channels |
| `bias` | per channel, the mean of test - reference (positive: the test is brighter) |
| `over8`, `over32` | % of pixels whose largest channel is off by more than 8 (32) levels |
| `p99`, `max` | the 99th percentile and the largest per-pixel error (its largest channel's), in 1/8-level steps |
| `rmse`, `psnr` | root mean square error over pixels and channels; 10 log10(255² / MSE) dB (null when identical) |

Over several frames the pixels pool (a frame weighs its pixel count). "1:1" is a pooled `mean_abs` <= 1.0 and `over8` <=
0.5 % (`pass` in every report; `--gate` exits 1 without it). Two 8-bit frames are measured in integers, exactly. Contact
sheets (`sheet_NN.png`, six frames each) put Blender's frame, the test's and the error as heat side by side (`--gain`, 8:
32 levels saturate; `--signed` adds the signed luma error, red where the test is brighter), for at most 60 frames: the
worst half and an even spread of the rest. `--mask FILE|DIR` (white counts; a folder holds one per frame) and `--region
X0,Y0,X1,Y1` (fractions of the frame when all are <= 1, else pixels) limit the pixels, and `--figure [LEVELS]` keeps only
the figure: the pixels where either frame has a channel above LEVELS (4), so the black void around a character does not
dilute the numbers (`tune --figure` scores the same way). Frames are read and measured in parallel threads: 30 frames of
1080p in about 1.4 s, 3 s more for their sheets.

| Subcommand | Does |
|---|---|
| `compare REF_DIR TEST_DIR` | any two folders of frames, matched by name (`00091.png` and `91.png` are frame 91; an empty claim file is no frame); `--frames` limits them; `summary.json` (with every frame's metrics and the frames only one folder has) and the sheets go to `--out` (default `TEST_DIR-compare` beside it) |
| `frames` | renders frames with `mkr render` from the project's bake into `.mk/parity/<run>/<output>/frames/` and compares them with Blender's (`renders/<preset>/<output>`, or `--refs` DIR: a folder of frames, or of `<output>/` folders) in `compare/` beside them, every output unless `--output`; by default every frame of the clip the references have. Frames whose picture Blender does not draw lit are skipped and listed by reason: a shot with a render-time look for that output (the bake's shot table), a [[medium]] window's look, a fisheye camera. mkr gets the references' resolution percentage (`--percent` overrides), `--samples` (64; with `--preset final` and neither `--samples` nor `[render] samples`, the final sampling `mk render` gives mkr, its tunables under `--params`': [mkr](#mkr)), motion blur as the preset has it (`--motion-blur` / `--no-motion-blur`, `--shutter`, else `[render] shutter`, 0.35), `--params` and whatever follows `--mkr-args`; the report is also `run.json` |
| `lab [CASES]` | mkr on the lab's cases (below), a table on stderr (`--per-frame`: a row per frame, with its label); `--variant` renders Blender's variants of them, `--against` measures mkr against one, `--separate` splits the error with them |
| `tune PARAMS.json` | hill climbing over mkr's tunables (below) |

#### The lab

Small scenes that each isolate one thing EEVEE does, rendered by Blender once and by mkr as often as it changes, from the
same bake. A case is a tiny project in `<lab>/<case>/` (`--root`, else `$MK_LAB`, else `~/.cache/mk/lab`): `mk.toml` (30
fps, `frame0` 1, its outputs, `blend = "scene.blend"`), `case.json` (the case as built: its spec, the Blender version, the
EEVEE settings, times), `scene.blend`, `scene.bake/` (the build's own bake stage), Blender's frames in three forms from one
render each (`ref/<output>/` 8-bit as `mk render` writes them, the target; `ref16/` the same picture as 16-bit PNG,
before 8-bit quantisation and its dither; `linear/` scene-linear OpenEXR, half float, no view transform), then
`mkr/<output>/` and `compare/<output>/` of the last run. Every case has one shot, `lab`, over its frames, a camera
`lab@<output>` per output; outputs are `main` (960x540) unless named. EEVEE Next, 64 samples, no world, filter 1.5, AgX
unless the case says, dither 1, shadows on, no ray tracing.

| Case | Isolates |
|---|---|
| `area_square`, `area_rect`, `area_disk`, `area_ellipse`, `point`, `spot`, `sun` | one light over a diffuse plane, a diffuse sphere and a glossy one (roughness 0.25): frame 1 from above in front, frame 2 low from behind (the light partly under the spheres' horizon) |
| `contact_shadow` | a box, a ball and a 5 cm pole on the plane under a 10 cm area light, high then low |
| `penumbra` | a ball and a slab floating over the plane under a disk light 2 m, then 0.6 m wide |
| `ggx` | the Glossy BSDF, one roughness a frame: multiscatter GGX at 0.05-0.8, then GGX at 0.1, 0.35, 0.8 |
| `principled` | the Principled BSDF, one sphere a frame: dielectric, metallic, specular, coat, sheen, transmission |
| `view_agx`, `view_standard` | emission strips of white, the primaries, their mixes and a skin tone from 2^-10 to 2^8 across, through AgX and Standard |
| `film_edge` | the film filter: a slanted white edge, a grey one, lines 0.3 and 1 px wide, a 0.4 px dot (Standard, 1 px = 1 cm) |
| `alpha_hashed`, `alpha_blended` | an alpha ramp and a half-transparent plane over a checkerboard, DITHERED and BLENDED (MMD's Transparent BSDF mix) |
| `skinned_bend` | a tube skinned to two bones bending 0-105 degrees |
| `motion_blur` | shutter 0.35: a ball moving 1 m and a bar turning 30 degrees a frame (frames 2-4), then the camera panning (7-9) |
| `mb_translate` | motion blur (shutter 0.35, centred, 1 step): rows of an emission card, a sphere and a box crossing the frame at 6, 24, 60, 150 and 300 px of blur over a checkered wall, behind a still pole (frames 2-4) |
| `mb_start`, `mb_end`, `mb_shutter1`, `mb_steps3` | the same rows with the shutter opening on the frame (START), closing on it (END), a whole frame long, and in 3 steps |
| `mb_spin` | a bar at 30, a disc of 24 sectors at 45 and a propeller at 90 degrees a frame, facing the camera |
| `mb_arm` | a skinned arm swinging 30 degrees a frame at the shoulder and bending 20 at the elbow |
| `mb_pan` | the camera panning and tracking sideways over a still pole, ball, box and wall |
| `dof` | depth of field through an 85 mm lens: emission checkerboards at 1, 2, 2.3 and 4 m before a checkered wall at 8 m, small bright spheres for bokeh; f/1.4, 2.8, 5.6, 8 (round), 6 blades, 6 turned 15 degrees, 5 blades, ratio 2, focus 1 m, focus 8 m |
| `dof_psf` | the same lens over black, focus 2 m: lines 0.6 px wide at 16 depths (the blur's profile across), 2 x 2 px squares dim, above EEVEE's sprite threshold and far above, an in-focus card over a blurred one and a blurred one over an in-focus one; f/1.4, 2.8, 5.6, 6 blades, turned, 5 blades, ratio 2 and 0.5 |
| `mmd_model` | a cast member at rest under one area light a frame (a warm 260 W key, a neutral 90 W fill, a red 180 W rim), then all three; `full` (1080x1920) and `face` (1920x1080) |
| `mmd_sphere` | a sphere in each of the member's three most used MMD materials, the key light |
| `shadow_bias` | 16 plates 0.1-8 mm over the plane, hidden from the camera, under a light 1 cm wide from 0, 30 and 60 degrees off the normal: EEVEE's normal offset and depth tolerance; `near` (960x540, 0.5 m away: EEVEE's finest shadow texels) and `far` (480x270, 2.4 m: coarser ones) |
| `shadow_thin` | strands 0.5-2 mm thick, a 2 mm card and lash-like slivers 5 mm to 8 cm over the plane under a key light (1.6 m, 260 W) from the front, the side and behind; `near` and `far` |
| `shadow_layers` | cloth layers: plates 2, 4 and 8 mm off a block's front and top, a sleeve over a hand, the same key light and views |
| `shadow_hashed` | hashed (DITHERED) see-through casters: plates of alpha 0.25, 0.5, 0.75 and strands of 0.5, the same key light and views |
| `shadow_close` | a close-up (`near` only, 0.36 m away) of cards 2.5 cm wide 5, 10, 20 and 50 mm over the plane and strands 1 mm thick 5 and 10 mm over it, hidden from the camera, under Crimson's key light (1.6 m square, 260 W, 3.2 m away, 35 degrees off the normal), then overhead, then a 1.6 x 0.25 m rectangle long along the strands and long across them (the rays toward its ends spread out of the plane of incidence, then in it). EEVEE lets about half the light into the cards' umbrae (visibility 0.49 under the 5 mm card, where Cycles has 0.001); mkr matches that overhead and is darker at 35 degrees, most for the rays spread in the plane of incidence (0.27 against 0.43): mean \|error\| 0.96. Reading the receiver's level along the whole ray (`shadow_lod_reach` 1000) brings it to 0.43 but darkens Crimson's close-ups (key light alone: figure 4.72 to 5.50) |
| `world_uniform`, `world_sky`, `world_stripes`, `world_sun` | a grey floor, a diffuse ball and glossy balls (roughness 0.05, 0.2, 0.4, 0.7) under a world: uniform (grey, then blue x2.5, keyed), a sky (ground, horizon, zenith, a soft sun), 16 sharp bands round the sky, a sky with a sun disc over the sun threshold (EEVEE's extracted sun); from the front (the sky over them) and from above |
| `world_transparent`, `world_lightpath`, `world_visibility` | the sky with the film transparent; a world warm to camera rays and cool to the rest (Light Path); the sky with its camera and glossy ray visibility off |
| `rt_world`, `rt_mirror`, `rt_rough`, `rt_corner`, `rt_corner_ao` | EEVEE's ray tracing as mk's scenes set it (screen traced to roughness 0.5, denoised, fast GI at half resolution): the balls under the sky; a glossy floor (0.02, then 0.1-0.7 keyed) reflecting a box, a ball and a lamp above the frame; a white floor in a red and green corner with a block (fast GI's bounce, then its ambient occlusion alone) |
| `gi_wall`, `gi_cabin`, `gi_rough` | fast GI alone (mk's ray tracing settings, lit by the world, no light): under a white world a white floor and wall, a slab 0.3 m over the floor (a car's underside), a body 10 cm off the wall and one in the open, from 3, 6 and 12 m; a body seated in a car's cabin open at the front, seen through it from 3.6 and 2.2 m; a glossy floor of roughness 0.5-1 (keyed) under the sky by a white wall, a box and a ball |
| `gi_seat`, `gi_strands` | fast GI with a light (mk's ray tracing settings, a dim uniform world of 0.1, one area light), the bounce of a surface's own object's light: two orange car seats (an L of cushion and back, one with side bolsters as a bucket seat's) from 1.3 m in front, 1.5 m from the side and 3.5 m; red strands 6 mm thick 1, 3, 10 and 30 cm off a white wall and a curtain of 24 of them 7 mm apart 2 cm off it, from 1.2 and 2.3 m |
| `rt_clamp` | EEVEE's clamp of the light reflected through ray tracing (`clamp_surface_indirect`, 10 as Blender sets it): white and saturated red emissive strips glowing at 2.5, 10, 40 and 160 over a floor that mirrors them, then a wet asphalt (a base of roughness 0.175 under a coat of 0.1, as mk's highway sets it), the base alone, the coat alone |
| `rt_emit` | the traced reflection of an emitter by its width and the floor's roughness: one width of white emissive card a frame (2, 6, 20 and 60 cm across, 40 cm tall, 10 cm over the floor, strength 4, under the clamp) over a black dielectric floor and a Glossy one, at roughness 0, 0.1, 0.2 and 0.4 |
| `point_near` | the near field of a sphere light (jwin's dash glow on the wheel): point lights of radius 5 cm, no shadows, 0.1, 0.2, 0.4 and 0.8 m over a diffuse grey floor and a dark glossy one (roughness 0.5), their power growing as the height squared |

`--build` builds the cases not built yet with Blender (seconds each, the mmd ones minutes), `--rebuild` all named again,
`--rebake` bakes the built ones again (`mk build --bake` on each case project: after the bake changes), `--refs-only` stops
there. `mmd_model` and `mmd_sphere` take the member from a built scene: `--source` a project folder (`--cast` one of its
[[cast]] names) or a .blend (`--cast` its armature); the scene is stripped to the member (unanimated, its rest pose, at the
origin) before the lights and cameras go in, and a rebuild remembers the source. Over the lab, the 8-bit frames are the
16-bit ones 0.41 levels brighter on average (mean \|difference\| 0.45, at most 1.75 levels): Blender's dither and rounding.

`--variant NAME` renders a variant of every named built case with Blender, again from its saved scene (the Blender op
`lab_variant` changes the scene in memory and writes nothing of the case: `scene.blend`, `ref/` and the bake stay), into
`<case>/variants/<name>/`: `<output>/` 8-bit PNG with the scene-linear EXR beside it, `16/<output>/` 16-bit PNG, and
`variant.json` (the variant, the engine's settings, the lights, times). Each run renders it again; a rebuild of the
case deletes its variants (they showed the old scene).

| Variant | Is |
|---|---|
| `noshadow` | the case's EEVEE frames with every light's shadow off |
| `cycles` | Cycles with direct light only: the light EEVEE approximates, computed exactly. No bounce (EEVEE has no world, light probes or ray tracing here) but through transparency (64 layers, no Russian roulette), 2048 samples (`mmd_*` 4096; `--variant-samples`) from seed 0 (`--seed`), no adaptive sampling, denoiser, clamping or light threshold, a Blackman-Harris filter as wide as EEVEE's, the lights unseen by the camera as EEVEE has them, the case's view transform and dither; on the GPU when Blender finds one |
| `cycles_noshadow` | the same with every light's shadow off (Cycles 4.2 reads the lights' `use_shadow`) |
| `jittered` | EEVEE as the case has it with `use_bokeh_jittered` on (a variant's `eevee` table sets scene.eevee properties): the camera jittered over the aperture each sample, the post process blurring the rest |
| `raytrace`, `fastgi_ao`, `noraytrace` | EEVEE as the case has it with mk's ray tracing on, with fast GI as ambient occlusion alone, or with ray tracing off (a nested `eevee` table sets `ray_tracing_options`) |
| `fastgi_res1`, `fastgi_ao_res1`, `fastgi_res4`, `fastgi_thin`, `fastgi_far`, `fastgi_noshadow`, `fastgi_ao_noshadow` | EEVEE with mk's ray tracing, one fast GI setting changed: the GI picture at full resolution (with the bounce, then as ambient occlusion alone), at a quarter, the near thickness 2 cm (mk's 25), the far (angular) thickness 0.1 rad (mk's pi / 4), every light's shadow off (with the bounce, then the occlusion alone) |
| `noclamp` | EEVEE as the case has it without its clamp of the indirect light (`clamp_surface_indirect` 0) |
| `nodenoise`, `notemporal`, `nobilateral` | EEVEE as the case has it with its ray tracing's denoiser off, or its temporal accumulation or its bilateral filter alone off |
| `trace_thin`, `trace_thick`, `trace_coarse`, `trace_fine` | EEVEE's ray tracing undenoised with the screen trace's thickness 0.05 or 1, or its precision 0 or 1 (mk's: 0.2 and 0.5) |
| `cycles_gi` | `cycles` with 4 diffuse and 4 glossy bounces: the indirect light a world, ray tracing and fast GI approximate, computed exactly |

Cycles' noise at those counts, from two renders with seeds 0 and 1 (the noise of one render: their difference over
sqrt 2): 0.002-0.024 levels mean, the 99th percentile of a pixel's largest channel 0.09-0.54 (area_rect, penumbra, ggx,
alpha_hashed at 2048); on mmd_model's face at 4096, 0.023-0.028 mean, over the figure 0.07-0.09 and p99 1.4-1.5. A
960x540 frame takes Cycles 35-60 s at 2048 samples on an RX 7700 XT (HIP), a 1920x1080 mmd_model frame 2-7 min at 4096.

`--against NAME` measures mkr against a variant's frames instead of `ref/` (sheets and summary in
`variants/<name>/compare/<output>/`). `--separate` draws mkr as it is (`mkr/`) and with `{"shadows": false}` over its
`--params` (`mkr_noshadow/`), then splits each case's error (every comparison is test minus reference; the table on
stderr, `--per-frame` a row per frame, the numbers as JSON on stdout, sheets in `<case>/separate/<comparison>/<output>/`;
a comparison whose frames are not all there is listed as missing):

| Column | Compares |
|---|---|
| `E-C`, `M-C`, `M-E` | EEVEE against Cycles, mkr against Cycles, mkr against EEVEE: mean \|error\| and the bias over the channels |
| `En-Cn`, `Mn-Cn`, `Mn-En` | the same with every shadow off: the light integral alone (the BSDF over the light's shape, falloff, film, view) |
| shadow terms | a renderer's frame minus its frame without shadows, over the pixels any of them darkens by more than 4 levels (`shad %` of the frame): how deep each one's shadows are (`dE dC dM`, luma), the terms against each other (`sE-C sM-C sM-E`, mean \|difference\|: a difference of the unshadowed light counts too), the light leaking into Cycles' umbra (`leakE leakM`: the renderer's luma minus Cycles' where Cycles' term is below -4 and its shadowed luma at most 5 % of its unshadowed one) and, in the JSON, `acne`: the shadow where Cycles has none (mean term where Cycles' is within half a level of 0, % of those pixels below -4) |
| probes | per case (`PROBES` in lab.py: the penumbra's and contact_shadow's shadow edges and the darkening by the ball's, slab's and cube's contacts; on `mmd_model`'s face under the key, the fill and all lights, the hair's shadow between the bangs and by a side lock, the eye white under the lid and the head's shadow on the neck; `--probes FILE`, `{case: {"lines": [...], "boxes": [...]}}`, replaces them): a shadow edge's 10-90 % width in pixels along a scanline (`y` and `x: [x0, x1]`, or `x` and `y: [y0, y1]`; 5 rows or columns averaged) on each renderer's shadowed frame, and the mean luma in a box with each renderer's darkening there (its unshadowed frame minus it) |

#### Tuning

`mk parity tune PARAMS.json --cases CASE... | --project DIR` climbs mkr's `--params` tunables (a flat JSON object, `mkr
params` prints them with their defaults) on the mean, over every frame of the cases (or of the project's frames, chosen as
for `frames`: `--output`, `--frames`, `--preset`, `--refs`), of one metric. With `--cases` and `--project` together the
score is the project's mean plus the cases' mean, so that no setting wins one by losing the other whatever their frame
counts, each times its `weights` entry (1 unless named: `{"lab": 3}` keeps a setting that helps the project from
costing the lab); `--pool` (or `"pool": true`) scores each output's (or case's) frames pooled over their pixels, as
`compare` reports them, then takes the mean over the outputs (for `mean_abs`, `over8`, `over32`, `bias_abs`):

```json
{"params": {"area_scale": {"start": 1.0, "step": 0.05, "min": 0.5, "max": 2.0},
            "shadow_steps": {"step": 2, "min": 1, "int": true},
            "filter": {"choices": ["gauss", "box"]}},
 "fixed": {"other": 1}, "metric": "mean_abs", "method": "coord", "iterations": 60, "min_step": 0.001, "pool": false,
 "weights": {"project": 1, "lab": 1}}
```

A param without `start` starts at mkr's default. `metric` is `mean_abs`, `rmse`, `over8`, `over32`, `p99`, `max`,
`bias_abs` (the mean of the channels' \|bias\|) or `psnr` (maximised); `--metric`, `--method` and `--iterations` override the
file. `coord` is coordinate descent: each number a step up and down, going on while it pays, each choice tried; a sweep
that moves nothing halves the steps (whole numbers down to 1, others down to `min_step`, default a 64th of the step).
`nelder-mead` is scipy's, from a simplex one step along each number, inside the bounds. A point already measured costs
nothing and counts no evaluation; a render mkr refuses scores infinity. Every evaluation is a line of `OUT/tune.jsonl`
(params, the metric, score, the score per case and output, with both the project's and the cases' too, seconds), the
best so far `OUT/best.json` and `OUT/best_params.json` (mkr's `--params` file: `fixed` and the best), and at the end the
best is drawn once more with sheets under `OUT/scratch/` (`OUT/scratch/project/` and `lab/` with both; `--no-sheets`:
not; `OUT`: `--out`, else `<project>/.mk/parity/tune-<time>/` with a project, else `<lab>/_tune/<time>/`).

`--resume LOG` goes on from an earlier run's `OUT/tune.jsonl` (the same tune file, options and mkr), one stopped or cut
short: its points are known, so the same search walks through them again for nothing, its best is the one to beat, and
`--iterations` counts the new evaluations. Each evaluation, its renders and their comparisons, holds the GPU lock and
lets it go before the next: a tune waits for the GPU once per evaluation, not once per render.

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

**When CTC picks the wrong place.** Forced alignment of a known lyric finds the best fit of the words to the whole
window: a line sung twice in it can land on the earlier repeat, and a line can end up in a vocal gap. Whisper hears
the words where they are sung, so the token aligner (`mkmmd/timeline/tokens.py`, numpy only) places the known words on
Whisper's own: a semi-global Needleman-Wunsch matches Whisper's tokens to the words by text similarity (tokens before
the first and after the last word are free, so the occurrence that holds every line in order wins over a partial
repeat; a token whose span is not sung scores less, so a repeat heard in silence loses), then a monotone dynamic
programme puts each word on a sung onset of the vocal stem (a spectral-flux peak, or the voice starting again after a
stop; a Whisper word boundary where the voice sings as a weak one): near its token's start and, within a line, near
the previous token's end (Whisper's word ends sit on the next sung onset within tens of ms, its starts drift, by up to
0.6 s at a phrase start, so a line's first word leans on its token least), strong onsets preferred, at least 0.12 s
apart, and no word of a line after a pause of 0.35 s or more since the one before. "Sung" is voiced and within 24 dB
of the vocal's peak, less a delay's echo: a run that comes back after a stop on the pitch of the previous run's last
second and at least 6 dB under it. A word no token was matched to falls where the onsets and its neighbours put it;
when the onsets cannot hold every word a 50 ms grid of sung frames is added. Each word ends where the next starts, a
line's last word when the singing stops. `analyze --align tokens` uses it instead of CTC on the transcription it has
just made. `mk timeline words [--timeline FILE] [--lyrics FILE] [--tokens FILE] [--out FILE] [--context 20]`
re-times the words of an existing timeline and changes nothing else (default in place; write a separate file to keep
the analysis): the lines are `--lyrics`, else the timeline's own; the tokens are those of `--tokens`, a timeline that
`analyze` wrote without `--lyrics` (`words_source` `"whisper"`) over any span of the same song (its times are moved by
the difference of the two `audio.start`), else a fresh Whisper run on the clip. The stems are read as `onsets` reads
them. The words get `ctc_start` null and `whisper` their token's span; `word_stats` counts the words matched exactly
(`anchored`), loosely (`fuzzy`) and not at all (`unanchored`) and the median distance of an anchored word from its
token's start. It prints numbers only.

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
`stems-<key>.npz`), `screen/` (each screen word's fitted encoding for the bake, by its modifiers and node trees with every
value, the keys on them, on the object and its parents, the fonts' bytes, the frames it shows on, what its materials read
and the code that bakes it: a word that reads the scene elsewhere, through drivers, constraints, time or another object, is
baked every time). The code is `mkmmd.cache.code`'s fingerprint, not a module's whole source: the functions, classes and
constants that make the result and everything they reach in the package, compared as syntax trees without docstrings, so
an edit to a comment or to code that does not make the word keeps the cache (the screen words: `_ScreenWord`, the `Lit`
methods that drive it, `_geom` and the fitter they reach in `mkmmd.core.fitgeo`). Changing hair settings re-runs only the
hair; nothing else is recomputed; editing a solver invalidates its results.
`<project>/.mk/look/<time>/` is where `mk look` writes by default. The user cache `~/.cache/mk/` (`MK_CACHE`) holds the job folders, serve
sockets, `empty.blend`, `samples/` and `look/` outside a project, and the reference sets.

Reference clips (`mk ref`) live outside the project: `~/.cache/mk/ref/<project name | default>/<set>/` holds
`clips.json`, the capped downloads, `track/<id>.npz` and the contact sheet; `mk ref clean` deletes everything but
`clips.json`. Only the small measurement JSON is kept with the project (`<project>/ref/<set>.json`). Reference photos for modelling
(`mk ref photos`) go to `<project>/refs/<set>/` (outside a project `~/.cache/mk/ref/photos/<set>/`) with `SOURCES.json` and a contact sheet.
Stock footage that a project shows (`mk ref footage`, [Footage](#footage-stickers-and-figure-moves)) goes to `<project>/footage/`
and is registered in the asset registry, not cached: the project's videos are made of it.

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
