# mk in Tern

The [Tern](https://stencil.so/tern) plugin of mk-mmd: two blocks that put mk's pictures and models beside your work.

- **Review** (`*.review.toml`): decisions put to a person by an agent. Each question is a card with its pictures (a
  click zooms), its models (shown in Tern's 3D block beside it), its options with a radio each and a notes field.
  **Mark up** puts a picture on a whiteboard to draw on and **Read marks** reads back what was drawn; **Send** posts the
  answers into the chat of the agent that opened the review.
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
and opens it beside its own pane:

```sh
mk review check looks.review.toml         # every problem at once
mk review open looks.review.toml          # beside this pane; Send posts the answers here
mk review answers looks.review.toml       # the answers, and the marks (in mm on lab sheets)
```

Choices and notes are written to `NAME.answers.json` as they change, so nothing is lost before Send. Marks are kept in
`NAME.marks.json` in each picture's own pixels: lines and arrows by their two ends, boxes, ellipses and ink by their
boxes, text with its words. On a lab sheet (`mk model lab`, its `.json` beside it) `mk review answers` turns them into
model space: how long a line is, how far inside or outside the outline it runs, a box's size in mm.

| Key | |
|---|---|
| `j` `k`, ↑ ↓ | the question the keys answer |
| `1`–`9` | choose that option |
| `n` | type in the notes (Esc leaves) |

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

- `window.luau` routes `*.review.toml` and `*.pmx` opens to the blocks, and claims the `mk://` links the blocks open with
  `cx:open` (`tern.route.link` sees every one): `preview` (a .glb in the tab's 3D preview), `send` (`cx.agents:ask`),
  `markup` (a whiteboard with the picture, locked) and `marks` (reads the whiteboards). Window calls have a 50 ms
  budget, so this half reads and writes no files: a link carries what it needs, each whiteboard is read in a timer tick
  of its own, and results come back to the block as actions on its dock (`cx.session:event`).
- `review.luau` and `pmx.luau` (the host half) run `mk` with `tern.process.run` (`mk review show`, `mk model glb`, `mk
  model lab`), send pictures with `cx:blob`, and write the answers and marks files. Models are cached in the plugin's data
  folder (`tern plugin dir`, `plugin-data/mk/`).

## Development

```sh
tern plugin types tern/          # tern.d.luau for luau-lsp (not committed)
tern plugin reload               # exits 1 when the plugin fails to load
```

A headless Tern runs the plugin from a fixtures folder: put a link to `tern/` at `ROOT/crates/plugins/fixtures/mk`, set
`STENCIL_FIXTURE_ROOT=ROOT/crates/tern`, start `tern serve --control SOCK`, and drive it with `tern ctl --control SOCK`
(`plugins fixtures`, `run "tern open FILE"`, `click "[data-id='…']"`, `state`, `shot NAME`). Shots don't wait for a
block's next frame: pause between commands.
