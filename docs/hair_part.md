# The hair part (`mkmmd/model/parts/hair*.py`)

One part, `hair`, built by `mk model build` after the body and head (`needs = ["body", "head"]`). It makes the scalp cap, bangs, side
locks and back hair, twin braids with bows and tufts, black cat ears and two cat tails, with their bones, dynamic chains, rigid bodies and
joints. Everything is bpy-free (numpy only) and driven by `[hair]` in `hair.toml` ([Characters in design.md](design.md#characters-mk-model) has
the spec and the builder framework).

| Module | What |
|---|---|
| `hair.py` | the builder: `HeadFit` from the head part's info, palette from `[colors.hair]`, head hair, then the slices below, merged into one `Part` (materials de-duplicated, display frames 髪 三つ編 猫耳 尻尾) |
| `hair_fit.py` | `Skull` (radial skin surface about the head centre, ray-cast from the head part's closed `info["skin"]` shell), `HeadFit` (hairline, lower cap boundary `cap_phi(theta)`, eyes, ears, cat-ear anchors), `Volume` (how far the hair stands off the skin) |
| `hair_geo.py` | swept clumps (`sweep`: lens-shaped closed shells with pointed / round / blunt tips), polylines, `MeshAccum`, `strip_normals`, `Piece` |
| `hair_rig.py` | `Rig.chain`: connected bones, one dynamic capsule per bone, joints, collision groups; `Chain.weights` (smooth partition of unity along a chain) |
| `hair_tex.py` | hair atlas (height gradient + fibre tiles), warm toon ramp, additive sphere ring, palette |
| `hair_head.py` | scalp cap, crown clumps, bangs (two staggered layers + long strands), side locks, back hair |
| `hair_braids*.py` | braids: three-strand weave, bows (band, knot, wings, notched tails), tuft, gather clumps |
| `ears.py`, `tails.py`, `cat_*.py` | cat ears (twitch bone + chain) and tails |

Conventions: model space metres, Z up, the character faces -Y, her left is +X; `theta` = azimuth from the front towards +X, `phi` = polar
angle from +Z. Hair surfaces are the skin plus `fit.vol(theta, phi)` (about 28 mm at the crown, 34 mm at the temples, 27 mm at the back) plus a
layer delta: cap -14 mm, crown clumps -7 mm, main clumps 0, front clumps +4..6 mm. Head-hair vertices take the texture's v from their height, so
one ring of highlight crosses every clump. mmd_tools indexes the toon texture by the view-space normal (top row lit, bottom row shaded) and adds
the sphere map, so `hair_tex.toon_ramp` and `hair_tex.sphere_ring` follow that.

## Look

Head hair is a few big rounded clumps (elliptical cross-section, 7 vertices across; `[hair.*] thick`, `bulge`, `k_outer`) in layers: outer
clumps over inner ones that take the darker texture tiles (`hair_tex.clump_u(dark=True)`). The crown clumps swirl out of a soft whorl
(`[hair.crown] swirl`) in two layers (the inner one half a pitch aside, `inner_delta`) and rise by `[hair.volume] lift` (a bump about `lift_deg`
from the crown, stronger at the back, `lift_back`); crown, side and back clumps vary in width (up to 1.25x), outward offset (a few mm) and
length, so the outline is slightly irregular.

- **Fringe**: `[hair.bangs] count` (11) separate pointed wedges (width 28 mm, 16-26 mm tapers, tips within +-`var` = 5 mm of one cut line, a little
  overlap) over a darker back layer of count - 1 clumps half a pitch aside, plus three thin loose strands (`strands`) in front. It sits at
  `[hair.bangs] above_eye` (0.027) over the lids. Its shadow on the forehead is two cel-shading steps (`hair_shadow` mesh, materials 髪影淡 and 髪影):
  strips 0.6 mm above the skin shaped like the fringe with rounded teeth (`[hair.bangs.shadow] tooth, valley, round, step, tint, tint2`), drawn with
  the head part's own skin material (same texture, toon, UVs and custom normals, darker and warmer, opaque) and bound to the `face` mesh so it carries
  its morph offsets (`HeadFit.on_face`, `face_morphs`, `face_normals`).
- **Back hair and nape**: `[hair.back]` (`hair_head.DEFAULTS`): `end_above_chin` (0) is where the locks end, metres above the chin's height (0: about the
  collar; negative lengthens them, -0.13 reaches the shoulder blades); `count` (8) darker inner clumps hang 2 cm lower under `count - 1` outer ones; `delta`
  (0) moves the hanging part out from the hair volume: a longer fall needs it (about 0.06 to 0.08 for -0.13), or the locks tuck into the back and the
  build warns that their chains are inside the body's colliders; `width` (0.076), `thick` (0.018), `sway` (0.008, the soft S-curve, also `[hair.side] sway`).
  The nape edge is a smooth layered wave ending in moderately pointed locks.
- **Shading**: the normals of every clump (and of the scalp cap) are blended 88 % (78 % at a clump's edges, so a clump has no rim of its own) towards a
  smooth head-shaped proxy (`hair_head.shade_proxy`: the normal of a vertical axis through the head, `[hair] shade z_hi z_lo`), which gives one terminator
  across the head under mmd_tools' view-normal toon and under any light. The toon step is a soft slope (`[hair.toon] edge 0.52, soft 0.20`) into
  `[colors.hair] toon_shadow_multiplier`.
- **Atlas**: soft gradients only: a dark-ish root to the base to a lighter tip over the whole height, inner tiles x 0.90 / 0.88 and the cap x 0.80 of the
  outer tone, clump edges 16 % darker with a smooth falloff, no strand lines. The crown highlight ("angel ring") is baked into the atlas
  (`hair_tex.make_atlas(..., ring_f, ring_w)`, `hair_tex.sheen_alpha`; `[hair.ring] enabled, drop, width`): one wide feathered band with a slightly
  brighter soft core per outer clump tile, wavy and interrupted at clump edges, faint on the inner tiles, none on the cap or lower than 5 cm under the top.
  Its colours are `[colors.hair] ring` and `ring_core`; by default they follow the family: Rin's light pink-coral sheen turned to the hue of the
  family's `highlight` (`highlight_core`) and scaled by its saturation and value, so chestnut hair gets a dusty rose-beige sheen and dark blue a steel
  blue one (`hair_tex.follow`); `[hair.ring] enabled = false` bakes none.
  The braids keep `hair_tex.sphere_ring`.
- **Braids and bows**: the hair part is built before the outfit and cannot see the dress, so `[hair.braids] clearance` (the gap kept to what the outfit will
  be, 0.008 by default) is a spec number: raise it (0.020) for a garment that stands 8-18 mm off the skin around the chest where the braids hang, and re-measure
  when the dress changes (`tests/test_model_hair_real.py::test_braids_hang_clear_of_the_blouse`). Bow tails are branches (`リボン*`).
- **Tails**: `[hair.tails] radius_tip` 6 mm, a soft S-curve (`path`), `spikes` small pointed clumps fanning from the last 5 cm; they rest low (`path.elev`): their apex
  stays below the shoulder line, so they do not frame the face from the front. They anchor on the body's `tail_root.L/R` and `tail_normal` as published (the first
  free bone starts 5 cm behind the root on 下半身; the first 2-4 cm of the tube are buried in the back and hidden by the dress).

## Chains (what `mk model` hands the sim stage)

| Family | Bones | Notes |
|---|---|---|
| bangs | `前髪i_1..2` x 11 (`[hair.bangs] count`) | each front-layer clump owns a chain; the back-layer clump beside it rides it |
| side_hair | `横髪左1_1..4`, `横髪右1_1..4` | three clumps per side ride one chain |
| back_hair | `後髪i_1..4` x 7 (`[hair.back] count - 1`) | the outer layer's chains; the inner layer's clumps ride the nearest |
| braid | `三つ編左1..N`, `三つ編右1..N` | bow to bow plus the tuft; bow tails are branches `リボン*` (family of the chain root: braid) |
| ears | `猫耳左`/`猫耳右` (static twitch bone, tiny static body) + `猫耳左1..3` | `perform.twitch = [{family = "ears", ...}]` rotates exactly the twitch bones |
| tail | `尻尾1_1..9` (left), `尻尾2_1..9` (right) | parent 下半身 |

Chain bodies sit in groups 4 (hair), 5 (ears), 6 (tails), 7 (ribbons); the first body of each chain is in group 8 and ignores group 0 (the body part's static
colliders), like the solver's `anchor_free` region. At rest every other chain capsule is clear of the body part's `col_*` bodies (on the bases the closest
are a fringe capsule 3.7 mm from `col_head` and a back-hair capsule 4.5 mm from a shoulder); a capsule that overlaps one is ignored by the solver for good and
thrown out by MMD's physics, so the build logs `WARNING hair chain <name>: <bone> is N mm inside <collider> at rest ... ([hair.<table>])` for it.

## Spec (`hair.toml`)

`[hair] seed, dynamic`, `[hair.volume] top front side back`, `[hair.bangs] count above_eye long_strands ...`, `[hair.side]`, `[hair.back] count width thick delta
end_above_chin sway`, `[hair.ring] enabled drop width`, `[hair.crown]`, `[hair.cap]`, `[hair.braids] enabled ...`, `[hair.ears] enabled ...`, `[hair.tails]
enabled ...` (`false` removes the slice). Every key has a default in `DEFAULTS` of the module that reads it (`hair_head.py`, `hair_braids.py`, `ears.py`,
`tails.py`). Colours: `[colors.hair]` (base shadow deep light highlight highlight_core rim toon_shadow_multiplier, and ring ring_core), `[colors.ears]`,
`[colors.black]`.

## Run and test

```sh
mk model build model.toml --only hair --out /tmp/hair      # PMX + rig.json + review .blend
mk look /tmp/hair/<name>.blend --view front,3q,left,back --frames 1 --target "bone('head').head" --dist 1.15 --sheet
pytest -q tests/test_model_hair.py tests/test_model_hair_braids.py tests/test_model_hair_cats.py tests/test_model_hair_real.py
```

Tests run on stand-in head and body parts (`tests/hair_fixtures.py`) and cover chain families (`mkmmd.core.families.classify`), weight normalisation, scalp
coverage (rays from the head centre meet the hair in the whole hair region), bangs above the eyes, human ears hidden, clearance from the static colliders and
determinism. `tests/test_model_hair_real.py` (and the last tests of `test_model_hair.py`) build a complete model from a local spec instead (about 30 s, skipped
when that spec is missing): braids against the blouse and the skin, tail roots and free length against the dress, every chain capsule against the real `col_*`
bodies, the nape against the neck.
