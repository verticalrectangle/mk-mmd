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
- `mk serve SCENE.blend` keeps one Blender alive on a Unix socket. The bridge uses it automatically for read-only
  ops (`q`, `list`, `look`, `sample`) on that scene, which matters for big files.

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

[[check]]               # see Checks
name = "back hair is calm"
metric = "jitter"
args = { family = "hair" }
max = 0.5
```

## Model description: `rig.json` (schema 1)

Written by `mk inspect MODEL.pmx`, stored next to the asset registry (`<assets>/rigs/<slug>.rig.json`). Bone names
are the Blender names mmd_tools gives on import (left/right prefixes become `.L`/`.R` suffixes); `jp` keeps the
original PMX name.

```jsonc
{
  "schema": 1,
  "source": {"path": "...pmx", "sha1": "...", "name_j": "...", "name_e": "..."},
  "scale": 0.08,
  "map": {"head": "頭", "wrist.R": "手首.R", "index2.L": "人指２.L", ...},   // semantic -> Blender bone
  "missing": ["upper_body2", "arm_twist.L", ...],                           // standard bones not found
  "bones": {"頭": {"jp": "頭", "parent": "首", "head": [x, y, z], "tail": [x, y, z], "deform": true}, ...},
  "fingers": {"index.L": ["人指１.L", "人指２.L", "人指３.L"], ...},
  "chains": [                                                               // physics-driven bone chains
    {"family": "hair", "root": "後髪1", "anchor": "頭", "bones": ["後髪1", ...], "body_radius": [0.072, ...]}
  ],
  "bodies": [{"bone": "頭", "shape": "sphere", "size": [..], "group": 0, "dynamic": false}, ...],
  "morphs": {"blink": "まばたき", "a": "あ", "i": "い", "u": "う", "e": "え", "o": "お", "smile": "笑い", ...},
  "morph_list": [{"name": "まばたき", "panel": "eye"}, ...],
  "measure": {"height": 1.62, "eye_height": 1.48, "arm": 0.52, "forearm": 0.24, "hand": 0.17, "leg": 0.78,
              "hip_height": 0.83, "shoulder_width": 0.33},
  "quirks": ["vertex groups that are not bones: mmd_edge_scale, mmd_vertex_order", ...]
}
```

Chain families are inferred from bone names (`hair`, `bangs`, `side_hair`, `back_hair`, `ears`, `tail`, `skirt`,
`breasts`, `ribbon`, `sleeve`, `accessory`, `other`). Projects tune the secondary-motion solver per family.

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

`<assets>/registry.json`: a list of entries, one per model, motion, prop, audio file or reference clip.

```jsonc
{"slug": "miy_reisen", "kind": "model", "path": "/abs/path/model.pmx", "name": "Reisen Udongein Inaba",
 "author": "Miy", "source_url": "...", "license": "...", "restrictions": "...", "credit": "Model: Miy",
 "rig": "rigs/miy_reisen.rig.json", "tags": ["touhou"], "extra": {...}}
```

`mk assets credits` builds a CREDITS file from the entries a project uses.

## Checks

A check is a named metric with a threshold. Metrics compute from sampled data or renders and return numbers; `mk
check` prints every result and exits 1 if any fails.

```jsonc
{"name": "back hair is calm", "metric": "jitter", "value": 0.38, "max": 0.5, "ok": true, "detail": {...}}
```

Metrics: `jitter` (secondary motion shake vs motion, per family), `penetration` (points inside shapes), `contact`
(distance between tracked points), `foot_slide`, `joint_limits`, `flicker` (rendered frame strips), `palette`
(near-black share, distance to palette), `framing` (subject inside each output's safe area), `occlusion`,
`camera_inside`.

## Cache

`<project>/.mk/cache/<op>/<key>.{json,npz}` where `key` is a SHA-256 of the op's inputs (arguments, input-file
fingerprints, solver version). Changing hair settings re-runs only the hair; nothing else is recomputed.
