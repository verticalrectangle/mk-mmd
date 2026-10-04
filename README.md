# mk-mmd

`mk` is a command-line toolkit for making music videos with MMD models in Blender. A project is one `mk.toml`: `mk`
builds the scene from it (sets, props, vehicles, characters, poses, grips, performance, secondary motion, lights,
shots, in-world type), measures it with checks, renders every output aspect and grades the result into MP4s with the
song. It also analyses songs (beats, bars, sung words), measures how real people move in reference clips, and builds
original MMD characters in code.

It is built for AI agents first (JSON out, checks that fail loudly, fast reruns) and works fine for people too.

## Install

Requirements: Linux, [uv](https://docs.astral.sh/uv/), Blender 4.2 with the
[mmd_tools](https://extensions.blender.org/add-ons/mmd-tools/) extension, ffmpeg.

```sh
git clone https://github.com/verticalrectangle/mk-mmd.git
cd mk-mmd
uv tool install --python 3.12 --editable '.[fast,dev]'          # add ,ref for mk ref and ,timeline for mk timeline
mk doctor
```

`mk doctor` tells you what is missing (and `mk doctor --fix` repairs mmd_tools' bundled wheels). Point mk at your
Blender with `MK_BLENDER=/path/to/blender` or `~/.config/mk/config.toml`.

## How a video is made

```sh
mk assets add model.pmx --kind model --author NAME     # register a model (its rig.json comes with it)
mk assets show my_model --terms                         # read its terms of use, then record licence and credit
mk timeline analyze song.flac --start 15.59 --duration 30.6 --out audio/timeline.json
mk build                                                # mk.toml -> build/<name>.blend
mk check                                                # every check of the project, exit 1 if one fails
mk look --frames 200,400 --sheet                        # look at the cut in every output aspect
mk render --preset draft --jobs 2                       # frames per output, resumable
mk post --preset draft                                  # grade + MP4 with the song
mk assets credits --out CREDITS.md
```

A small project (the full keys of every section are in [docs/design.md](docs/design.md)):

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
gantries = [{ s = 500, panels = 2 }]

[[prop]]
name = "car"
card = "library:car_mockup"

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
to = 8.0
mount = "car"
at = [0.0, -1.6, 1.3]
look = "cast:rin.head"
lens = 35

[[text]]
name = "sign"
on = "road:gantry1_panel1"
text = "OLD HELL\nEXIT 12"
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
| `mk doctor` | checks Blender, mmd_tools, ffmpeg, Python packages, disk and keys |
| `mk inspect` | describes a model (rig.json: semantic bones, chains, bodies, expressions, quirks) or a motion |
| `mk assets` | the local asset registry: add, terms of use, licence and credit fields, credits for a project |
| `mk model` | builds an original MMD character from a spec in code and exports a PMX (see Characters in design.md) |
| `mk timeline` | analyses a song span: tempo, beats, bars, loudness, sung words by `(line, word)` |
| `mk ref` | measures blinks, gaze, nods, sway and mouth in reference clips of real people (Pexels + MediaPipe); `mk ref photos` fetches licensed Wikimedia photos to model props from |
| `mk build` | builds the scene from `mk.toml`, stage by stage (`--until`, `--skip`) |
| `mk q`, `mk serve` | ask a scene anything, frame by frame (`mk serve` keeps a big scene loaded) |
| `mk grip` | solves a hand grip (pen, wheel, pinch, rest) on a model's hand |
| `mk check` | runs the project's checks: penetration, contact, joint limits, jitter, framing, occlusion, palette, prop form (boxiness) and more |
| `mk look` | renders views without touching the file: the cut per aspect, orbit views of any target, sheets, A/B, reference photos side by side (`--ref`) |
| `mk render` | renders the cut of every output to frames (presets draft / preview / final, parallel, resumable) |
| `mk post` | grades the frames (contrast, split tone, halation, vignette, grain) and encodes MP4s with the song |

Every command prints JSON; exit codes are 0 ok, 1 a check failed, 2 usage, 3 Blender or runtime error.

## Documentation

- [docs/AGENTS.md](docs/AGENTS.md): the playbook for making a video with mk (also for people)
- [docs/design.md](docs/design.md): architecture, the project file, every build stage and its keys, checks, caches
- [docs/modelling.md](docs/modelling.md): how library props and sets are modelled (references first, no blocky forms)

## What is not in this repository

Licensed or copyrighted material (models, motions, music, lyrics, reference clips and photos, renders) never goes
into this repository; projects keep it in their own folders and in the local asset library (`~/mk-assets`), with
each asset's terms of use recorded in the registry.
