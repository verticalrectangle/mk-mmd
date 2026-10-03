# mk-mmd

`mk` is a command-line toolkit for making music videos with MMD models in Blender. It understands a model's rig,
poses it, puts things in its hands, performs, simulates hair, builds sets, frames shots, renders and checks the
result. It is built for AI agents first (JSON in and out, checks that fail loudly, fast retries) and works fine for
people too.

Status: under construction. See [docs/design.md](docs/design.md) for the architecture and contracts.

## Install

Requirements: Linux, [uv](https://docs.astral.sh/uv/), Blender 4.2 with the
[mmd_tools](https://extensions.blender.org/add-ons/mmd-tools/) extension, ffmpeg.

```sh
git clone https://github.com/verticalrectangle/mk-mmd.git
cd mk-mmd
uv tool install --python 3.12 --editable '.[fast,dev]'
mk doctor
```

`mk doctor` tells you what is missing. Point mk at your Blender with `MK_BLENDER=/path/to/blender` or
`~/.config/mk/config.toml`.

## First commands

```sh
mk q scene.blend --list bones                      # every bone, with its PMX and semantic names
mk q scene.blend 'bone("head").head' --frames 1:100   # where the head is, frame by frame
```

Licensed or copyrighted material (models, motions, music, lyrics, reference clips, renders) never goes into this
repository; projects keep it in their own folders and in the local asset library.
