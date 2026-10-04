# The hair part (`mkmmd/model/parts/hair*.py`)

One part, `hair`, built by `mk model build` after the body and head (`needs = ["body", "head"]`). It makes the scalp cap,
bangs, side locks and back hair, twin braids with bows and tufts, black cat ears and two cat tails, with their bones, dynamic
chains, rigid bodies and joints. Everything is bpy-free (numpy only) and driven by `[hair]` in `hair.toml`.

| Module | What |
|---|---|
| `hair.py` | the builder: `HeadFit` from the head part's info, palette from `[colors.hair]`, head hair, then the slices below, merged into one `Part` (materials de-duplicated, display frames 髪 三つ編 猫耳 尻尾) |
| `hair_fit.py` | `Skull` (radial skin surface about the head centre, ray-cast from the head part's closed `info["skin"]` shell), `HeadFit` (hairline, lower cap boundary `cap_phi(theta)`, eyes, ears, cat-ear anchors), `Volume` (how far the hair stands off the skin) |
| `hair_geo.py` | swept clumps (`sweep`: lens-shaped closed shells with pointed / round / blunt tips), polylines, `MeshAccum`, `strip_normals`, `Piece` |
| `hair_rig.py` | `Rig.chain`: connected bones, one dynamic capsule per bone, joints, collision groups; `Chain.weights` (smooth partition of unity along a chain) |
| `hair_tex.py` | hair atlas (height gradient + fibre tiles), warm toon ramp, additive sphere ring, palette |
| `hair_head.py` | scalp cap, crown clumps, bangs (two staggered layers + long strands), side locks, back hair |
| `hair_braids*.py` | braids: three-strand weave, bows (band, knot, wings, notched tails), tuft, gather clumps (slice by RinHairBraids) |
| `ears.py`, `tails.py`, `cat_*.py` | cat ears (twitch bone + chain) and tails (slice by RinHairCats) |

Conventions: model space metres, Z up, the character faces -Y, her left is +X; `theta` = azimuth from the front towards +X, `phi`
= polar angle from +Z. Hair surfaces are the skin plus `fit.vol(theta, phi)` (about 28 mm at the crown, 34 mm at the temples,
27 mm at the back) plus a layer delta: cap -14 mm, crown clumps -7 mm, main clumps 0, front clumps +4..6 mm. Head-hair vertices
take the texture's v from their height, so one ring of highlight crosses every clump. mmd_tools indexes the toon texture by the
view-space normal (top row lit, bottom row shaded) and adds the sphere map, so `hair_tex.toon_ramp` and `hair_tex.sphere_ring`
follow that.

## Look (polish round)

Head hair is a few big rounded clumps (elliptical cross-section, 7 vertices across, `[hair.*] thick`, `bulge`, `k_outer`) in two
layers: outer clumps over inner ones that take the darker texture tiles (`hair_tex.clump_u(dark=True)`); the crown clumps swirl out of a
soft whorl (`[hair.crown] swirl`); every bang ends in a small V-shaped point 2-6 mm longer than its shoulders, with two longer thin
strands; the nape edge is the irregular row of pointed back clumps (the inner layer hangs lower). Normals are blended towards the head-radial
direction (`normal_mix`, 0.9) so the toon shading and the additive sphere ring (`hair_tex.sphere_ring`: one soft curved coral band with a pale
line) read as one form. The atlas is a smooth root-to-tip gradient with a convex-ribbon profile per tile and one or two sparse strand lines.
The toon ramp uses `[colors.hair] toon_shadow_multiplier` with a clear but soft step (`[hair.toon] edge, soft`).
Tails: `[hair.tails] radius_tip` 6 mm, a soft S-curve (`path`), and `spikes` small pointed clumps fanning from the last 5 cm.

Last round: the head hair no longer uses a sphere ring (a camera-relative additive band crosses the bangs like a headband, in
any light). The angel ring is baked into the atlas instead (`hair_tex.make_atlas(..., ring_f, ring_w)`, `[hair.ring] drop, width`): one
thin soft light coral-gold patch per clump tile on the upper crown, different heights and lengths per tile and a small random shift per
clump, so it reads as a broken curved dash line that follows the head, is dimmed by the lighting and fades long before the fringe. The
crown rises by `[hair.volume] lift` (a bump about `lift_deg` from the crown, stronger at the back, `lift_back`) and the crown, side and
back clumps vary in width (up to 1.25x), outward offset (a few mm) and length, so the outer clumps overlap and the outline is slightly
irregular. The braids still use `hair_tex.sphere_ring`. The bangs end in shallow rounded points (3-8 mm longer than their shoulders, varied), the back hair now reaches
the collar (`end_above_chin = 0`), and the tails rest low: their apex stays below the shoulder line (`[hair.tails] path.elev`), so they
do not frame the face from the front.

Round 3 (against Reisen's refs): the fringe is `[hair.bangs] count` (11) separate pointed wedges (width 28 mm, 16-26 mm tapers, tips
within +-`var` = 5 mm of one cut line, a little overlap) over a darker back layer of count - 1 clumps half a pitch aside, plus three thin
loose strands (`strands`) in front that use only the middle of the clump tile. The shadow of the fringe on the forehead is two
cel-shading steps (`hair_shadow` mesh, materials 髪影淡 and 髪影): strips 0.6 mm above the skin shaped like the fringe with rounded teeth
(`[hair.bangs.shadow] tooth, valley, round, step, tint, tint2`), drawn with the head part's own skin material (same texture, toon, UVs and
custom normals, darker and warmer, opaque), bound to the `face` mesh so it carries its morph offsets (HeadFit.on_face / face_morphs /
face_normals). The crown has two layers of clumps (the inner one a half pitch aside, darker tiles, `inner_delta`). The atlas keeps three
tones: outer tiles dark root -> shadow red -> base -> lighter tips, inner tiles a shade of the shadow red, the cap darkest; clump edges
tint towards the shadow red; the toon ramp steps early (`[hair.toon] edge 0.42, soft 0.16`) into `[colors.hair] toon_shadow_multiplier`.
The baked ring is a crisp coral-gold band per outer tile (wavy upper edge, zig-zag lower edge, different heights and lengths, a pale core
line), faint on the inner tiles, none on the cap or lower than 5 cm under the top.

Round 4 (smoothness): Miy's hair is smooth and soft, so everything hard was removed. The hair is shaded like one smooth form: the
normals of every clump (and of the scalp cap) are blended 88 % (78 % at a clump's edges, so a clump has no rim of its own) towards
a smooth head-shaped proxy (`hair_head.shade_proxy`: the normal of a vertical axis through the head, `[hair] shade z_hi z_lo`; the
crown and fringe tilt up and are lit, the sides are horizontal, the lower hair tilts down), which gives ONE terminator across the
head under mmd_tools' view-normal toon and under any light, with only a subtle per-clump roundness left. The toon step is a soft slope
(`[hair.toon] edge 0.52, soft 0.20`). The atlas has soft gradients only: dark-ish root -> base -> lighter tip over the whole height,
inner tiles x 0.90 / 0.88 and the cap x 0.80 of the outer tone, clump edges 16 % darker with a smooth falloff, no strand lines. The
crown highlight is one wide feathered band of light pink-coral (a lighter, less saturated version of the base) with a slightly
brighter soft core, baked per clump tile (`hair_tex.sheen_alpha`, centre `[hair.ring] drop` below the top of the hair, half width
`width`), wavy and interrupted at clump edges, faint on the inner tiles. Locks hang in soft S-curves (`HeadHair.sway`, `[hair.side|back]
sway`), the nape line is a smooth layered wave (not a saw) ending in moderately pointed locks (8 + 7 clumps), the side locks are
longer with a gentler taper, and the braids' gather clumps now start buried in the hair (`gather_bury`), so no plate edges show.

Refit to the Reisen-style face: the head part's `face` mesh is now an imported face shell with a generated skull cap, so the
fringe sits at `[hair.bangs] above_eye = 0.027` (the new forehead is steeper and its lids and brows are higher), the cap's lower
boundary (`HeadFit._cap_curve`) ramps smoothly from each temple to the jaw end of the sideburn line instead of stepping (a step made
cap quads cut chords through the head), and tests/test_model_hair.py builds the real head part of the Rin project (skipped when the
project is missing) to check scalp coverage, hidden human ears, fringe vs lids, the cap boundary and that the forehead shadow strips lie
on the `face` mesh with its morphs.

Refit to the Reisen-style body (the user's Reisen body scaled to Rin's height, no hair or clothes): the hair needed only one change
for it, in the spec, not in code: `[hair.braids] clearance = 0.020` (default 0.008). The hair part is built before the outfit and
cannot see the dress, and the real blouse stands 8-18 mm off the skin around the chest where the braids hang, so with the bare-body
default 28 vertices per braid ended up 8 mm under the blouse (a bow wing and the plait's edge in the front view); at 0.020 no braid
vertex is under `outfit_dress`/`outfit_frills` and the nearest is 6 mm away. Re-measure when the dress changes
(`tests/test_model_hair_real.py::test_braids_hang_clear_of_the_blouse`). The tails anchor on the body's `tail_root.L/R` and
`tail_normal` as published (the first free bone starts 5 cm behind the root on 下半身, the first 2-4 cm of the tube are buried in the
back and hidden by the dress, everything further than 8 cm from the roots is above the skin and the dress); all chains keep >= 5 mm
from the 25 `col_*` colliders (closest: 後髪7_4 vs col_shoulder_R 5.0 mm, braids 7.2 mm vs col_upper_body2, fringe 14 mm vs the new
col_head, tails 16 mm). The old col_head sphere stuck 21 mm out of the face at brow height and put the fringe 5.5 mm inside it; RinBody
refitted it to the largest sphere inside the head shell.

## Chains (what `mk strands`/`sim` simulates)

| Family | Bones | Notes |
|---|---|---|
| bangs | `前髪i_1..2` x 7 | each front-layer clump owns a chain; the back-layer clump beside it rides it |
| side_hair | `横髪左1_1..4`, `横髪右1_1..4` | three clumps per side ride one chain |
| back_hair | `後髪i_1..4` x 8 | outer-layer chains; inner-layer clumps ride the nearest |
| braid | `三つ編左1..N`, `三つ編右1..N` | bow to bow plus the tuft; bow tails are branches `リボン*` (family of the chain root: braid) |
| ears | `猫耳左`/`猫耳右` (static twitch bone, tiny static body) + `猫耳左1..3` | `perform.twitch = [{family = "ears", ...}]` rotates exactly the twitch bones |
| tail | `尻尾1_1..9` (left), `尻尾2_1..9` (right) | parent 下半身 |

Chain bodies sit in groups 4 (hair), 5 (ears), 6 (tails), 7 (ribbons); the first body of each chain is in group 8 and ignores
group 0 (the body part's static colliders), like the solver's `anchor_free` region. Every chain capsule is at least 5 mm clear of
the body part's `col_*` bodies at rest (a body overlapped at rest is ignored by the solver for good).

## Spec (`hair.toml`)

`[hair] seed, dynamic`, `[hair.volume] top front side back`, `[hair.bangs] count above_eye long_strands ...`, `[hair.side]`,
`[hair.back]`, `[hair.crown]`, `[hair.cap]`, `[hair.braids] enabled ...`, `[hair.ears] enabled ...`, `[hair.tails] enabled ...`
(`false` removes the slice). Every key has a default in `DEFAULTS` of the module that reads it (`hair_head.py`,
`hair_braids.py`, `ears.py`, `tails.py`). Colours: `[colors.hair]`, `[colors.ears]`, `[colors.black]`.

## Run and test

```
mk model build ~/Projects/mk-tests/rin_model/model.toml --only hair --out /tmp/rin_hair   # PMX + rig.json + review .blend
mk look /tmp/rin_hair/rin.blend --view front,3q,left,back --frames 1 --target "bone('head').head" --dist 1.15 --sheet
pytest -q tests/test_model_hair.py tests/test_model_hair_braids.py tests/test_model_hair_cats.py tests/test_model_hair_real.py
```

Tests run on stand-in head and body parts (`tests/hair_fixtures.py`, built from the proportions study) and cover chain families
(`mkmmd.core.families.classify`), weight normalisation, scalp coverage (rays from the head centre meet the hair in the whole
hair region), bangs above the eyes, human ears hidden, clearance from the static colliders and determinism.
`tests/test_model_hair_real.py` (and the last three tests of `test_model_hair.py`) build the REAL body, head and outfit of the Rin
project instead (about 30 s, skipped when `~/Projects/mk-tests/rin_model/model.toml` is missing): braids vs the blouse and the skin,
tail roots and free length vs the dress, every chain capsule vs the real `col_*` bodies, nape vs the neck.
