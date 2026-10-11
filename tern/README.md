# mk in Tern

The [Tern](https://stencil.so/tern) plugin of mk-mmd: mk's reviews and PMX models beside your work.

- **Review** (`*.review.toml`): decisions put to a person by an agent. Opened from Tern's Files pane (or `tern open`),
  the file becomes the review page mk serves (docs/design.md: Reviews, The site) in place of the pane: each question
  with its pictures to draw on, its models to turn in mk's 3D viewer, its options and a notes field; **Send** hands the
  answers to the agent.
- **PMX model** (`*.pmx`): any MMD model with its names and numbers, chips for the poses its bones allow (T-pose, rest,
  arms down, sit, hands) and for its morphs by panel, the model posed in the 3D block beside it, and a lab sheet of the
  pose (toon views, numbers in mm).

## Install

From the mk-mmd checkout (the plugin runs the `mk` of the checkout it sits in, through `uv run`):

```sh
tern plugin link tern/              # use it where it is; edits reload it
```

or, with mk installed as a tool (`uv tool install` in the checkout, so `mk` is on PATH):

```sh
tern plugin install github.com/verticalrectangle/mk-mmd/tern
```

## Reviews

An agent writes `NAME.review.toml` beside its pictures (the format is in `mkmmd/review.py` and docs/design.md: Reviews)
and opens it beside its own pane; the page is a browser block docked to it:

```sh
mk review check looks.review.toml         # every problem at once
mk review open looks.review.toml --wait   # beside this pane; returns the answers when Send is pressed
mk review answers looks.review.toml       # the answers, the marks (in mm on lab sheets) and the views kept in 3D
```

The page saves choices and notes to `NAME.answers.json` as they change, so nothing is lost before Send, and the marks
in `NAME.marks.json`.

## PMX models

A `.pmx` opened in Tern (Files pane, palette, a drop, `tern open`) opens as a PMX block, posed in T-pose in the 3D block
beside it (`mk model glb`, no Blender, about a second for a 50,000-vertex model).

| Key | |
|---|---|
| `t` `r` `a` `s` | T-pose, rest, arms down, sit |
| `e` `c` `f` `p` | relaxed, curled, fists, spread hands |
| `0` | no morph |
| `o` | show it in 3D again |

## How it works

- `window.luau` routes `*.review.toml` and `*.pmx` opens to the blocks, and claims the one `mk://` link a block opens
  with `cx:open` (`tern.route.link` sees every one): `preview`, a .glb in the tab's 3D preview.
- `review.luau` (the host half) is a launcher: it runs `mk review open --replace PANE`, so the review page takes the
  block's place, and says why when it cannot. `pmx.luau` runs `mk` with `tern.process.run` (`mk model glb`, `mk model
  lab`) and sends pictures with `cx:blob`. Models are cached in the plugin's data folder (`tern plugin dir`,
  `plugin-data/mk/`).

## Development

```sh
tern plugin types tern/          # tern.d.luau for luau-lsp (not committed)
tern plugin reload               # exits 1 when the plugin fails to load
```

A headless Tern runs the plugin from a fixtures folder: put a link to `tern/` at `ROOT/crates/plugins/fixtures/mk`, set
`STENCIL_FIXTURE_ROOT=ROOT/crates/tern`, start `tern serve --control SOCK`, and drive it with `tern ctl --control SOCK`
(`plugins fixtures`, `run "tern open FILE"`, `click "[data-id='…']"`, `state`, `shot NAME`). Shots don't wait for a
block's next frame: pause between commands.
