# mk-mmd

`mk` is a command-line toolkit for making music videos with MMD models in Blender. A project is one `mk.toml`: `mk` builds the
scene from it (sets, props, vehicles, characters, poses, grips, performance, secondary motion, lights, shots, cut effects, type in
the world and over the picture), measures it with checks, renders every output and grades the result into MP4s with the song. It also
analyses songs (beats, bars, sung words), measures how real people move in reference clips, and builds original MMD characters in
code.

It is built for AI agents first (JSON out, checks that fail loudly, fast reruns) and works fine for people too.

## Install

Requirements: Linux, [uv](https://docs.astral.sh/uv/), Blender 4.2 with the
[mmd_tools](https://extensions.blender.org/add-ons/mmd-tools/) extension, ffmpeg. Optional: espeak-ng (lip sync from sung words),
`secret-tool` (the Pexels key for `mk ref`).

```sh
git clone https://github.com/verticalrectangle/mk-mmd.git
cd mk-mmd
uv tool install --python 3.12 --editable '.[fast,dev]'          # add ,ref for mk ref and ,timeline for mk timeline
mk doctor
```

`mk doctor` tells you what is missing (and `mk doctor --fix` repairs mmd_tools' bundled wheels). Point mk at your Blender with
`MK_BLENDER=/path/to/blender` or `~/.config/mk/config.toml` ([Configuration](docs/design.md#configuration)).

## A small project

A folder with an `mk.toml` is a project. This one puts a registered character in front of a night sky and checks that she stays in
frame (register a model first: `mk assets add model.pmx --kind model --slug my_model --author NAME`):

```toml
[project]
name = "hello"
fps = 30
frame0 = 61                   # clip time 0; the frames before it are the pre-roll where the character settles
duration = 6.0                # seconds
blend = "build/hello.blend"

[[output]]
name = "16x9"
size = [1920, 1080]

[look]
palette = "rose-pine-moon"    # sets, props, lights and the grade colour by palette slot

[[set]]
name = "sky"
kind = "night_sky"

[[cast]]
name = "rin"
asset = "my_model"            # a model registered with `mk assets add`

[perform.rin]
blink = { per_min = 15 }
breath = { per_min = 16 }

[[shot]]
name = "front"
from = 0.0
to = 6.0
at = [0.0, -3.0, 1.3]         # world metres; she faces -Y
look = "cast:rin.head"
lens = 50

[[check]]
name = "she stays in frame"
metric = "framing"
args = { cast = "rin" }
min = 0.0
```

```sh
mk build                      # mk.toml -> build/hello.blend
mk check                      # exit 1 if a check fails
mk look --frames 100          # a picture of the cut to look at
```

## How a video is made

```sh
mk doctor                                               # is everything installed?
mk assets add model.pmx --kind model --slug my_model --author NAME   # register a model (its rig.json comes with it)
mk assets show my_model --terms                         # read its terms of use, then record licence and credit
mk timeline analyze song.flac --start 15.59 --duration 30.6 --out audio/timeline.json
mk build                                                # mk.toml -> build/<name>.blend
mk check                                                # every check of the project, exit 1 if one fails
mk look --frames 200,400 --sheet                        # look at the cut in every output
mk render --preset draft --jobs 2                       # frames per output, resumable
mk post --preset draft                                  # grade + MP4 with the song
mk assets credits --out CREDITS.md
```

The playbook ([docs/AGENTS.md](docs/AGENTS.md)) goes from the brief to the credits: brief, timeline, storyboard, cast, sets and props,
pose, perform and sim, shots, effects and type, checks, look sheets, render and post, credits.

## A fuller project

The keys of every section are in [docs/design.md](docs/design.md). This one drives a convertible along a night highway, with the
character at the wheel, her hair and skirt simulated in the moving air, two shots and a sign on a gantry:

```toml
[project]
name = "night_drive"
fps = 30
frame0 = 61                  # the clip starts here; the frames before are the pre-roll where poses and hair settle
duration = 8.0
blend = "build/night_drive.blend"

[[output]]
name = "16x9"
size = [1920, 1080]

[[output]]
name = "9x16"
size = [1080, 1920]

[audio]
file = "~/Music/song.flac"
start = 15.59                # song seconds at clip time 0

[look]
palette = "rose-pine-moon"   # sets, props, lights and the grade colour by palette slot

[[set]]
name = "sky"
kind = "night_sky"

[[set]]
name = "road"
kind = "highway"
points = [[0, 0, 0], [0, -400, 0], [30, -900, 0], [70, -1400, 0]]
gantries = [{ s = 330, panels = 2 }]

[[prop]]
name = "car"
card = "library:convertible_80s"

[[vehicle]]                  # the car drives the road's lane; wheels spin, the steering wheel follows the curves
prop = "car"
path = "road:road"
lane = "fwd1"
speed = 26.0
at = 120.0

[[cast]]
name = "rin"
asset = "my_model"

[pose.rin]
sit = "car:driver"
lean = 20                    # the build warns when a hand cannot reach its grip: lean further or move the seat

[pose.rin.hands.L]           # the fingers are solved on the model's own skin around the rim; the hand rides the wheel
grip = "car:wheel"
clock = 10
ride = "car_wheel"

[pose.rin.hands.R]
grip = "car:wheel"
clock = 2
ride = "car_wheel"

[perform.rin]
look = "car:road"
blink = { per_min = 15 }
sing = { timeline = "audio/timeline.json", lines = [1, 2] }

[sim.rin]                    # hair, ears, tails and the skirt in the air of the moving car
families = ["back_hair", "bangs", "side_hair", "ears", "tail", "skirt"]
wind = { carrier = "car", exposure = 0.3, gust = 1.5 }

[[shot]]
name = "hood"
from = 0.0
to = 5.0
mount = "car"
at = [0.0, -1.6, 1.3]
look = "cast:rin.head"
lens = 35

[[shot]]
name = "road"
from = 5.0
to = 8.0
mount = "car"
at = [-0.38, 0.1, 1.15]      # from the passenger seat, looking down the road
look = "car:road"
lens = 28

[[text]]
name = "sign"
on = "road:gantry1_panel1"
text = "NORTH\nEXIT 12"
fit = 0.8

[[check]]
name = "hair stays out of the body and the car"
metric = "penetration"
args = { cast = "rin" }
max = 8.0
```

## Commands

| Command | What it does |
|---|---|
| `mk doctor` | checks Blender, mmd_tools, ffmpeg, Python packages, disk and keys (`--fix` repairs mmd_tools' wheels) |
| `mk inspect` | describes a model (rig.json: semantic bones, chains, bodies, expressions, quirks) or a motion (tempo, energy, travel) |
| `mk assets` | the local asset registry: `add`, `scan`, `list`, `show` (terms of use), `set` (licence and credit fields), `rm`, `credits` for a project |
| `mk model` | builds an original MMD character from a spec in code and exports a PMX: `build`, `info`, `studio` (see Characters in design.md) |
| `mk timeline` | analyses a song span: `analyze` (tempo, beats, bars, loudness, note onsets, sung words by `(line, word)`), `show`, `onsets` |
| `mk ref` | measures blinks, gaze, nods, sway and mouth in reference clips of real people (`search`, `add`, `track`, `measure`, `sheet`, `clean`; Pexels + MediaPipe); `mk ref photos` fetches licensed Wikimedia photos to model props from |
| `mk build` | builds the scene from `mk.toml`, stage by stage (`--until`, `--skip`) |
| `mk q`, `mk serve` | ask a scene anything, frame by frame (`mk serve` keeps a big scene loaded for `q`, `check` and `grip`) |
| `mk grip` | solves a hand grip (pen, wheel, pinch, rest, neck) on a model's hand |
| `mk check` | runs the project's checks: jitter, contact, penetration, foot slide, joint limits, framing, occlusion, camera inside, flicker, palette, form (prop boxiness), strum, prop in body |
| `mk look` | renders views without touching the file: the cut per output, cameras, orbit views of any target, sheets, A/B, reference photos side by side (`--ref`) |
| `mk render` | renders the cut of every output to frames (presets draft / preview / final, parallel, resumable) |
| `mk post` | composites the cut effects and screen type, grades the frames (contrast, split tone, halation, vignette, grain) and encodes MP4s with the song |

Every command prints JSON; exit codes are 0 ok, 1 a check, gate or verification failed, 2 usage error, 3 Blender or runtime error.
`mk <command> -h` has the flags and examples.

## Documentation

- [docs/AGENTS.md](docs/AGENTS.md): the playbook for making a video with mk, brief to credits, with the pitfalls (also for people)
- [docs/design.md](docs/design.md): the reference: architecture, the project file, every build stage and its keys, checks, shots, effects, text, post, caches, characters
- [docs/modelling.md](docs/modelling.md): how library props and sets are modelled (references first, forms from profiles, no blocky shapes)
- [docs/head_part.md](docs/head_part.md), [docs/hair_part.md](docs/hair_part.md): the head and hair parts of `mk model`

## What is not in this repository

Licensed or copyrighted material (models, motions, music, lyrics, reference clips and photos, renders) never goes into this repository;
projects keep it in their own folders and in the local asset library (`~/mk-assets`), with each asset's terms of use recorded in the
registry. The repository carries no licence file.

## Credits

mk drives [Blender](https://www.blender.org/) and the [mmd_tools](https://extensions.blender.org/add-ons/mmd-tools/) extension, and uses
ffmpeg, numpy, scipy, OpenCV and Pillow; optionally numba, MediaPipe, Demucs, faster-whisper and espeak-ng. Reference clips come from
Pexels and reference photos from Wikimedia Commons, with their licences kept. The built-in palettes are the Rosé Pine colours.
