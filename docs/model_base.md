# Model bases: making your own character

A model base is a complete character that mk builds from spec files alone: `mkmmd/model/bases/<name>/`. A character starts
from one and changes only what it wants; when the base improves, every character gets it on its next build. The spec
format, the builders and every command are in [design.md: Characters](design.md#characters-mk-model); this page is the
playbook.

| Base | What it is |
|---|---|
| `girl` | the neutral base: Blender Studio's stylized body (CC0) fitted to her skeleton, a generated face, a chin-length chestnut bob, brown eyes drawn soft (a short low wing, a thin band), a plain navy dress with blue frills and a calf-length skirt, black Mary-Janes, CC0 hands |
| `rin` | the worked example: `girl` plus Rin's red hair, twin braids, cat ears and tails, red eyes in her cat-like drawing, leaf-print dress with green frills and a calf ribbon (Rin Kaenbyou of Touhou) |

## Start

```sh
mk model new mika                          # ./mika/model.toml: [model] name, out and include = ["base:girl"]
mk model build mika/model.toml             # PMX, textures, rig.json and a review .blend in [model] out (~40 s)
mk look ~/mk-assets/models/mika/mika.blend --frames 1 --view=0:0,30:10,90:0,180:0 \
    --target "(0, 0, 0.78)" --dist 3.6 --lens 70 --out /tmp/mika_look
```

The fixed target frames the whole figure, shoes included, and keeps two builds at one scale for `--ab OTHER.blend`: a
bone target (`--target "bone('head').head"`) frames each model on itself, so a change of height does not show. Without
`--out` the frames go to `~/.cache/mk/look/<time>`; one jpg per view reads better than a `--sheet` strip.

`mk model new NAME --from rin` starts from the example instead; `--dir` and `--out` place the spec and the build. `mk model
build base:girl --out DIR` builds a base on its own.

## How a character is put together

`include = ["base:girl"]` merges the base's files first, then the character's own tables on top: tables merge key by key,
any other value (lists too) is replaced, the character wins. So a character file holds only what differs. Read
`mkmmd/model/bases/rin/`: five short files that turn the girl into Rin. A path value `"base:girl/hand.npz"` names a file
inside a base.

| Table | In the base | What it changes |
|---|---|---|
| `[proportions]` | proportions.toml | every size: the landmarks (bone heads, absolute metres), limb and torso sections, foot and shoe, the head outline `[proportions.head]`, the face layout `[proportions.face]` (where the eyes, brows, nose and mouth sit), the guide heights of the hair and the outfit; `leg_extra` lengthens the legs |
| `[colors.*]` | colors.toml | `skin`, `hair`, `eyes`, `mouth`, `ears` (cat ears and tails), `black` (ribbons, shoes), `accent` (red nails); the base's colors.toml lists every key of each family |
| `[body]` | body.toml | the skin look and the body's skin colours (`skin`), nails, the neck shadow band, ring resolution, skeleton options; `[body.hand]` the hand (a mesh, or the designed hand's keys); every key in `mkmmd/model/parts/body.py` |
| `[head]` | head.toml | the face: `[head.shape]` (outline), `[head.face]`, eyes, lids, brows, mouth, the lash drawing `[head.lash]`, `[head.lashwing]`, `[head.lashfork]` ([head_part.md](head_part.md)); the eyes' colours, the lashes' and the brows' are `[colors.eyes]`, where eyes, brows and mouth sit is `[proportions.face]` |
| `[hair]` | hair.toml | volume, bangs, side and back hair; `[hair.braids]`, `[hair.ears]`, `[hair.tails]` switch the braids, cat ears and tails ([hair_part.md](hair_part.md)) |
| `[outfit]` | outfit.toml | `print` (the leaf print), `shading` (`"auto"` in the girl: shadows, outlines and the ruffle's tint follow `[colors.outfit]`; `"tuned"`, the default, keeps the ones tuned for Rin's green), skirt chains, sleeve and collar physics, `[outfit.legs.ribbon] enabled` (the calf ribbon), `[colors.outfit]` the cloth and frill colours (`outfit_tex.DEFAULT_COLORS`), `[outfit.materials.<key>]` any material field (keys `dress frill frill_inner ruffle satin leg shoe sole`) |

Where the keys and their defaults are: `DEFAULTS` at the top of `hair_head.py` (`[hair]`: volume, ring, bangs, side,
back), `hair_braids.py`, `ears.py` and `tails.py`; `outfit.py` (its pieces read more keys where they use them:
`outfit_dress.py`, `outfit_skirt.py`) and `outfit_legs.py` (the calf ribbon and the shoes, listed in its docstring);
`head_shape.FACE` (`[head.face]`) and `head_eye.DEFAULTS` (`[head.eye]`); `body.py`'s docstring for the body. A key no
part reads is reported by the build (`[spec] WARNING`, the loop's step 4), so a guess never passes for a key that works.

Some changes, each a few lines in the character's model.toml (or a file it includes):

```toml
[colors.hair]                 # a whole family: in HSV value shadow about 0.6 of the base, deep 0.4, light 1.3, highlight 1.8
base = "#2b2f4a"              # dark blue; these are the girl's chestnut family turned to it (hue turned, saturation and
shadow = "#181b2b"            # value scaled: mkmmd.model.colour.follow); set every key, or the ones left out stay chestnut
deep = "#0e101b"
light = "#373b5e"
highlight = "#606189"
highlight_core = "#83829a"
rim = "#4f517a"               # toon_shadow_multiplier stays: a cool shadow suits any hair; the crown sheen follows itself

[hair.braids]                 # Rin's braids and cat ears on the girl
enabled = true
clearance = 0.020             # keep the braids 2 cm off the skin: the dress stands up to 18 mm off it where they hang
[hair.ears]
enabled = true

[colors.outfit]
dress_base = "#3a1d24"        # wine-red cloth (plain: with [outfit] print = true also set dress_leaf, dress_leaf_hi,
frill = "#e9e2d6"             # dress_leaf_dark and dress_accent, or the leaves stay Rin's green); cream frills, which
frill_inner = "#ffffff"       # the girl's shading = "auto" shades warm grey, not Rin's green

[body]
nails = "red"                 # [colors.accent] nail_red
```

Proportions are absolute numbers. `leg_extra = 0.03` in `[proportions]` makes her 3 cm taller in the legs: everything from
the hip joint up rises 3 cm, the thigh and shin stretch alike and the feet stay. Any other change of height moves the
numbers that hold heights together: the landmarks' z, `[proportions.sections.torso]` z and `[proportions.sections.neck]`
z, `[proportions.head]` (crown_skull, chin_point, forehead_front), `[proportions.face]` (eye_z, brow_z, nose_tip,
mouth_z), `[proportions.hair_guides]` and `[proportions.outfit_guides]`. Change a few at a time and look: the body, the
outfit (it fits itself to the skin) and the hair follow, but seats, steering wheels and grips in mk projects are made for
the base's size.

## The loop

1. Change one table.
2. Build: `mk model build model.toml --only hair` (that part and what it needs, into `<out>/only_hair/`) while iterating;
   `--no-export` runs the builders without Blender (into `<out>/no_export/`, so the exported model stays as it was);
   `--set hair.bangs.count=9` tries a value without editing. A full build takes about 40 s the first time; after that a
   part whose inputs did not change comes from the part cache (the report says `cached` per part), so a hair change
   builds only the hair and the outfit after it. `--no-cache` builds every part again.
3. Look: `mk model lab model.toml --region hand --poses rest,fist,spread` (seconds, no Blender: views and poses with the
   model's own weights, numbers under them; give it the base or a reference .pmx too and they stand side by side at one
   scale). A spec builds only the body by default (the head for `--region head`): `--parts all` adds the hair and the
   dress, as a .pmx always has them, so its head height, width and depth include its hair and ears; girths are cut on
   the skin either way. Then `mk look` sheets in Blender's toon shading, and `--ab OTHER.blend` next to the base or the previous
   build. In Tern, `mk model glb model.toml --out x.glb` (T-pose by default; `--region head` for the face) opens in its
   3D block to turn around, and any built `.pmx` opens as a PMX block (numbers, poses, morphs, in 3D). Never call it
   good before looking at the images.
4. Read the build report (the JSON it prints, also `<out>/build.json`): its `warnings` list must be empty and `verify`
   must pass. `[hair] WARNING ...` lines are a part's findings (a chain inside a collider, an eye off its landmark);
   `[spec] WARNING hair.back.lenght: never read by any part ...` names keys of yours that no part reads: misspelt, in the
   wrong table, or for a feature that is switched off. Keys that come from the base are left out.
5. When a choice is the person's (a look, a route), ask with pictures: write a `NAME.review.toml` (questions, options,
   images, models) and `mk review open --wait` it beside your chat; it returns the answers when they press Send, and
   `mk review answers` reads them, with lines drawn on lab sheets in mm (docs/design.md: Reviews).
6. Repeat.

## Rules

- A character is done when it builds and verifies with an empty `warnings` list, and its sheets have been looked at from
  all sides.
- Library changes come with tests that fail on the bug they guard against, and the full suite passes.
- Never commit built models, renders or downloaded sources. The committed meshes are the base's CC0 hand and body,
  `mkmmd/model/bases/girl/hand.npz` and `body.npz` (`make_hand.py` and `make_body.py` rebuild them from their sources).
- Library code names no character; characters live in `mkmmd/model/bases/` and in project specs.

## Measuring and judging

- Measure before answering a question about shape ("is the palm too long": wrist to web against web to tip, beside a
  reference and real anthropometry), and say what was measured and how.
- Use exact cross-sections of the mesh. A "14 mm waist" came from sparse vertex slices of overlapping shells.
- Distrust numbers that are too good: grip results identical to the last run meant the scene still loaded the old build.
  Check what is loaded.
- When the user marks up a screenshot of a lab sheet in red, `mk model trace MARKED.png --sheet SHEET.png` turns the
  marks into millimetres (how far inside or outside the outline, where along the part) before anything changes. Show
  previews of the options side by side, and let the user choose on taste.

## Parts from other artists: what the hands and the body taught

The base's hands were designed in code three times, and each looked worse than the next try with a CC0 glove from an
artist's model. The generated body's boxy shoulders, hip ledge and thin legs went the same way: the girl wears Blender
Studio's stylized body (CC0), fitted to her skeleton (`[body] source = "mesh"`; `"procedural"` still draws the generated
one). Organic shapes (hands, bodies, faces) come out better from an artist's mesh; code fits it, rigs it, welds it and
checks it.

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
- **No rig? Measure the joints.** A base mesh without bones gets its joints from its sections: the elbow at its section's
  centre (an axis through a bent arm misses it by a centimetre), the knee at the narrowest section, the hip above the
  crotch, all written down with how they were found (`JOINTS` in make_body). Which region owns each vertex (torso, arm,
  leg) is a harmonic field between seed regions, so it follows the surface: the side of the chest under the arm stays
  torso. Fit with long hand-overs, weight with short ones (`SHARE`): the upper arm is all arm 7 cm past the joint.
- **Fit to her skeleton, keep its flesh.** Each limb segment turns and stretches onto her bone, the girth scaled by her
  size only (her neck seam's height over the donor's, `leg_extra` left out: longer legs make no wider body); the torso
  warps by height through her hip joints, shoulder joints and neck seam; the foot keeps its flat stance on the shoe's
  inner floor. Her rig, heights and grips stay what they were, so nothing downstream moves.
- **The neighbours meet new flesh.** Build every part on it and read the warnings. A real neck widens into the shoulders
  sooner than the generated one: the back hair now keeps 4 mm off the body's skin (`hair_head.off_body`), and colliders
  fitted to the skin a bone owns took the trapezius with the shoulder (a ball of 6 cm radius reaching 4 cm above the
  skin, the hair's chains inside it), so the shoulder's are as wide as its top (`body_donor.shoulder_tops`). Sharper
  creases (the armpit) need distances signed by the edge's normal, not one face's (`outfit_fit.Skin.outward`).
- **Cut where the neighbour starts.** The body's head comes off at a level plane just under the jaw and its neck ring is
  laid onto the seam ellipse the head builds on; each forearm is cut at an edge loop just before the hand's seam and a
  strip joins it to a ring of the hand's 56 points sampled from the fitted skin.

The format of a hand mesh is in `mkmmd/model/parts/body_hand_mesh.py`, of a body mesh in `body_donor.py`.

## Where the base comes from

- Proportions and the face layout: Reisen's measured numbers (Miy's model, measured and never copied), scaled x0.95 for
  the body and x0.985 for the head. Keep this note when the base is shared (proportions.toml says it too).
- Hands: the left glove of B-chan by AmarilloArts, CC0 1.0.
- Body: the stylized female body of Blender Studio's Human Base Meshes v1.4.1 ("by Blender Studio and community
  contributions", CC0 1.0, blender.org's demo files), fitted to her skeleton; she keeps her proportions, it gives the flesh.
- Rin's hair, ears, tails and dress are her character design (Touhou); the neutral girl carries none of it.
