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
- `mk serve SCENE.blend` keeps one Blender alive on a Unix socket. The bridge uses it automatically for the ops that
  leave the scene as they found it (`ping`, `list`, `q`, `sample`, `visibility`), which matters for big files. Ops
  that change scene state (`look` hides objects, removes markers, adds a camera) always get a fresh Blender.
- `sample` is how numbers leave Blender: one pass over a set of frames writes posed bone and object matrices, rest
  matrices, expression values, the active camera and resolved colliders to an `.npz`; checks and solvers work on that.

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
Checks refer to channels as `track:<name>.<channel>`.

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
credits from the project's cast and `[credits] assets` and exits 1 while any of them is unreviewed.

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
| `camera_inside` | frames with the camera inside a closed mesh | |
| `flicker` | worst frame's 99th percentile of temporal luma noise | on rendered frames |
| `palette` | near-black share (or median distance to a palette) | on rendered frames; `rose-pine-moon`, `rose-pine`, `rose-pine-dawn` built in |

`mk check --list` prints every metric's arguments.

## Looking

`mk look` renders views without touching the file: the cut (scene camera with its timeline markers) per output
aspect, named cameras, or orbit presets around any target expression relative to a cast member's facing. It writes
one JPEG per view, aspect and frame, plus optional contact sheets, strips, A/B pairs against another scene and
framing guides.

## Building

`mk build` assembles the scene from `mk.toml` in one Blender session and saves it to `[project] blend`. Stages run
in order and each reads its own sections:

| Stage | Sections | Does |
|---|---|---|
| scene | `[scene]` (`start`, `end`, `settle_frames`) | empty scene, fps, frame range including the pre-roll before `frame0` |
| sets | `[[set]]` (`name`, `kind`, `at`, `yaw`, builder keys) | library set builders (sky, roads, tunnels, skylines, rooms) with their paths, surfaces and lights |
| props | `[[prop]]` (`name`, `card`, `at`, `yaw`, `parent`, `slots`) | library props (`library:car_mockup`, `library:chair`) or card files; their use points and colliders |
| vehicles | `[[vehicle]]` (`prop`, `path`, `lane`, `speed`, `at`, `roll`, `pitch`, `wheelbase`, `steer_ratio`) | a prop drives a set path's lane: position and heading per frame, body roll and pitch, wheels spinning, the steering wheel turning with the curvature |
| cast | `[[cast]]` (`name`, `asset` or `pmx`, `armature`, `at`, `yaw`, `parent`, `physics`) | models imported without Bullet (`physics = "mk"`), named, placed |
| pose | `[pose.<cast>]` | sit on a prop's seat, feet on targets (leg IK), lean / turn / head, arm IK to points, edges and moving keys (targets can ride a prop part such as a steering wheel), finger presets; `[[prop]] attach = "cast:bone"` puts props on bones |
| motion | `[[motion.<cast>]]` | VMDs on NLA strips: source range, scale or `retime = "beats"`, body masks, blends |
| perform | `[perform.<cast>]` | gaze events over an idle target, breathing, sway, nod, beat bob, startles, blinks, lids, expressions, lip sync, twitches |
| shots | `[[shot]]` | the cut, see Shots |
| lights | `[[light]]`, `[look]` | lights in palette colours (mounted, aimed, keyed); view transform, contrast look, exposure |
| sim | `[sim.<cast>]` | secondary motion solved outside Blender and baked to keys |
| save | | the `.blend` |

The character eases from its rest pose (at `start`) into the base pose over `settle_frames`; secondary motion settles
in the same pre-roll. Rotations are composed in the armature's axes (the model faces -Y): a bone's posed rotation
relative to rest is `D_bone = D_parent . q`, and the bone-local key is `R^-1 q R` with `R` its armature-space rest
rotation, so characters riding a moving vehicle keep correct keys. Gaze blends unit directions from the eyes, so
targets at any distance (a mirror 0.5 m away, a road 30 m ahead) mix evenly.

Targets anywhere in the build spec are `[x, y, z]` (world), `{prop = "car", point = [..]}` (a prop's local frame),
`"car:road"` (a prop use point), `"cast:rin"` (eyes) or `"cast:rin.head"` (a bone), or `"camera"`.

Solvers run as `python -m <module> IN.npz OUT.npz` on the CLI's Python and are cached in `<project>/.mk/cache/`
by a hash of their inputs.

## Shots

`[[shot]]` (`name`, `from`, `to` in clip seconds) makes one camera per output aspect (`<shot>@<output>`), keyed on
every frame: `mount` (a prop, set or object the camera rides), `at` (in the mount's frame), `look` (any target),
`lens`, `roll`, `lag` (operator lag in seconds, applied in the mount's frame so a camera in a car lags the subject,
not the road), `shake` (handheld, degrees), `keys = [{t, at, look, lens}]` for moves, `dof = {focus, fstop}`, and
`frame = {subject, fill, solve}` to solve the lens (or dolly) per aspect so the subject fills that share of the
frame height. `[shot.aspect.<output>]` overrides any key for one aspect. Timeline markers cut between shots; the
scene keeps the shot table in `scene["mk_shots"]`, and `mk look` / `mk render` point the markers at each aspect's
cameras before rendering it.

## Palettes

`[look] palette` picks a named palette (`rose-pine-moon`, `rose-pine`, `rose-pine-dawn`), `[look.slots]` overrides
slots. Slots: base, surface, overlay, muted, subtle, text, love, gold, rose, pine, foam, iris, hl_low, hl_med,
hl_high. Sets, props, lights and the grade colour by slot, never by hard-coded values.

## Rendering and post

`mk render --preset draft|preview|final` renders each output's cut to `<project>/renders/<preset>/<output>/
<frame>.png`. Frames are claimed with an empty file first, so a stopped render resumes and `--jobs N` Blender
processes share the frames; a disk check refuses to start when the frames would not fit.

`mk post` grades the frames (`[post]`: contrast around a pivot, saturation, split toning, a palette floor with a
soft toe so nothing is black, halation from blurred highlights, vignette, grain) and encodes
`<project>/out/<name>_<output>[_<preset>].mp4` with `[audio]` (`file`, `start` = song seconds at clip time 0).

## Timeline

`mk timeline analyze` turns the clip's song span into `audio/timeline.json`: `bpm`, `beat_s`, `beats`, `downbeats`,
`grid` (a constant-tempo fit when the beats are that steady), per-frame `vocal_db`, `energy_db`, `drums_db`, and
`lines[].words[]` with `start`, `end`, `voiced_end`, `ctc_start`, `whisper` (seconds from clip time 0). Beats come
from the kick and snare bands of the Demucs drum stem (hi-hats would double the tempo) with a dynamic-programming
tracker; downbeats from kick accents. Words come from a lyrics file (one sung line per line) or from Whisper's
transcription, aligned with wav2vec2 CTC on the vocal stem and snapped to sung onsets. Lines and words are numbered
from 1; tools refer to words as `(line, word)` and never print their text.

## Cache

`<project>/.mk/cache/<op>/<key>.{json,npz}` where `key` is a SHA-256 of the op's inputs (arguments, input-file
fingerprints, solver version). Changing hair settings re-runs only the hair; nothing else is recomputed.
