# Model bases: making your own character

A model base is a complete character that mk builds from spec files alone: `mkmmd/model/bases/<name>/`. A character starts
from one and changes only what it wants; when the base improves, every character gets it on its next build.

| Base | What it is |
|---|---|
| `girl` | the neutral base: generated body and face, a chin-length chestnut bob, brown eyes, a plain navy dress with blue frills and a calf-length skirt, black Mary-Janes, CC0 hands |
| `rin` | the worked example: `girl` plus Rin's red hair, twin braids, cat ears and tails, red eyes, leaf-print dress with green frills and a calf ribbon (Rin Kaenbyou of Touhou) |

## Start

```sh
mk model new mika                          # ./mika/model.toml: [model] name, out and include = ["base:girl"]
mk model build mika/model.toml             # PMX, textures, rig.json and a review .blend in [model] out
mk look ~/mk-assets/models/mika/mika.blend --frames 1 --view=0:0,30:10,90:0,180:0 \
    --target "bone('lower_body').head" --dist 3.2 --lens 70
```

`mk model new NAME --from rin` starts from the example instead; `--dir` and `--out` place the spec and the build. `mk model
build base:girl --out DIR` builds a base on its own.

## How a character is put together

`include = ["base:girl"]` merges the base's files first, then the character's own tables on top: tables merge key by key,
any other value (lists too) is replaced, the character wins. So a character file holds only what differs. Read
`mkmmd/model/bases/rin/`: four short files that turn the girl into Rin. A path value `"base:girl/hand.npz"` names a file
inside a base.

| Table | In the base | What it changes |
|---|---|---|
| `[proportions]` | proportions.toml | every size: the landmarks (bone heads, absolute metres), limb and torso sections, foot and shoe, the head shape, the face layout `[proportions.face]` (eyes, brows, mouth), the outfit's guide heights |
| `[colors.*]` | colors.toml | `skin`, `hair`, `eyes`, `mouth`, `ears` (cat ears and tails), `black` (ribbons, shoes), `accent` (red nails) |
| `[body]` | body.toml | the skin look, nails, the neck shadow band, ring resolution, skeleton options; `[body.hand]` the hand (a mesh, or the designed hand's keys); every key in `mkmmd/model/parts/body.py` |
| `[head]` | head.toml | the face: `[head.shape]` (outline), `[head.face]`, eyes, lids, brows, mouth ([head_part.md](head_part.md)) |
| `[hair]` | hair.toml | volume, bangs, side and back hair; `[hair.braids]`, `[hair.ears]`, `[hair.tails]` switch the braids, cat ears and tails ([hair_part.md](hair_part.md)) |
| `[outfit]` | outfit.toml | `print` (the leaf print), skirt chains, sleeve and collar physics, `[outfit.legs.ribbon] enabled` (the calf ribbon), `[colors.outfit]` the cloth and frill colours (`outfit_tex.DEFAULT_COLORS`) |

Some changes, each a few lines in the character's model.toml (or a file it includes):

```toml
[colors.hair]                 # a whole colour family: keep the tonal structure (shadow ~0.45 of the base's lightness)
base = "#2b2f4a"
shadow = "#161828"
deep = "#0f101a"
light = "#3f4570"

[hair.braids]                 # Rin's braids and cat ears on the girl
enabled = true
[hair.ears]
enabled = true

[outfit]
print = true                  # the leaf and flower print
[colors.outfit]
dress_base = "#3a1d24"        # wine-red cloth
frill = "#e9e2d6"             # cream frills
frill_inner = "#ffffff"

[body]
nails = "red"                 # [colors.accent] nail_red
```

Proportions are absolute numbers. Change a few landmarks or sections at a time and look: the body, the outfit (it fits
itself to the skin) and the hair follow, but seats, steering wheels and grips in mk projects are made for the base's size.

## The loop

1. Change one table.
2. Build: `mk model build model.toml --only hair` (that part and what it needs, into `<out>/only_hair/`) while iterating;
   `--no-export` runs the builders without Blender; `--set hair.bangs.count=9` tries a value without editing.
3. Look: `mk model lab model.toml --region hand --poses rest,fist,spread` (seconds, no Blender: views and poses with the
   model's own weights, numbers under them; give it the base or a reference .pmx too and they stand side by side at one
   scale), then `mk look` sheets in Blender's toon shading, and `--ab OTHER.blend` next to the base or the previous
   build. Never call it good before looking at the images.
4. Read the build log: `WARNING` lines are findings; `verify` must pass.
5. Repeat.

## Parts from other artists: what the hands taught

The base's hands were designed in code three times, and each looked worse than the next try with a CC0 glove from an
artist's model. Organic shapes (hands, bodies, faces) come out better from an artist's mesh; code fits it, rigs it,
welds it and checks it.

- **Licence on the author's page.** Read the terms where the author publishes them, not in a search summary. CC0 (public
  domain) needs no credit. Record the source and licence in the asset (make_hand writes `license` and `source` into
  hand.npz) and, in a project, in `mk assets`. Downloads that need an account are the user's to make.
- **Garments are anatomy.** A glove is a hand (cut off its flared cuff), a leotard covers a midriff, a boot shaft is a
  calf (its foot is a boot, not a foot).
- **Use the donor's rig.** Take the joints from its bones and the weights from its groups, summed onto mk's semantic
  bones (`GROUP` in make_hand). Re-pose to a relaxed rest (`FAN`, `CURL`, `THUMB_IN`); move each tip joint out to the end
  of the skin its last bone drives, because rigs often put tips inside the fingertips.
- **Place by joints, scale by a design measurement.** The hand is turned into the hand frame by its own joints and scaled
  so the straight middle finger reaches `[body.hand] length`. A bounding box makes curled fingers shrink the hand.
- **Seams.** Build the neighbour with the donor's ring count (the forearm takes the hand's 56 points), pair the rings by
  angle round the axis, ease the donor onto the ring over a blend length (`BLEND`, 3 cm), and fail loudly when they are
  far apart. A test checks the crease angle across the seam.
- **Reproducible.** A maker beside the asset, `make_<name>.py` with `make(out)`, downloads the source, exports it with
  Blender and prepares it: `uv run python -m mkmmd.model.bases.girl.make_hand`. A missing asset is rebuilt by it on
  first use.
- **Check it the way it will be used.** Views from every side; posed with its own weights (rest, relaxed, curled, fist,
  spread); in Blender with the outfit, `--ab` against the previous version; then a real use: the steering-wheel grip
  (`mk grip`) still has to reach its contacts without penetration.

The format of a hand mesh is in `mkmmd/model/parts/body_hand_mesh.py`.

## Measuring and judging

- Measure before answering a question about shape ("is the palm too long": wrist to web against web to tip, beside a
  reference and real anthropometry), and say what was measured and how.
- Use exact cross-sections of the mesh. A "14 mm waist" came from sparse vertex slices of overlapping shells.
- Distrust numbers that are too good: grip results identical to the last run meant the scene still loaded the old build.
  Check what is loaded.
- When the user marks up a screenshot of a lab sheet in red, `mk model trace MARKED.png --sheet SHEET.png` turns the
  marks into millimetres (how far inside or outside the outline, where along the part) before anything changes. Show
  previews of the options side by side, and let the user choose on taste.

## Where the base comes from

- Proportions and the face layout: Reisen's measured numbers (Miy's model, measured and never copied), scaled x0.95 for
  the body and x0.985 for the head. Keep this note when the base is shared (proportions.toml says it too).
- Hands: the left glove of B-chan by AmarilloArts, CC0 1.0.
- Rin's hair, ears, tails and dress are her character design (Touhou); the neutral girl carries none of it.

## Rules

- A character is done when it builds and verifies with no `WARNING`, and its sheets have been looked at from all sides.
- Library changes come with tests that fail on the bug they guard against, and the full suite passes.
- Never commit built models, renders or downloaded sources. The one committed mesh is the base's CC0 hand,
  `mkmmd/model/bases/girl/hand.npz`.
- Library code names no character; characters live in `mkmmd/model/bases/` and in project specs.
