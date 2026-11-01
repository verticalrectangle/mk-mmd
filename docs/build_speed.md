# Making `mk build` fast

Rendering a finished video went from about 90 minutes to under 4 (mkr and mkr-post, docs/design.md: mkr, Post), so
`mk build` and its bake are now the slowest part of a change. This is the plan to make them fast, in order. Update each
step's status and its numbers as the work lands.

## Where the time goes (2026-11-01)

The bake reports its parts (`mk build --bake`'s JSON: `setup_seconds`, `loop_seconds`, `export_seconds`, and
`lit.seconds` with `frames_by_part`, `lit.screen.seconds`, `lit.screen.cache`). Seconds:

| | Crimson and Clover, cold | Crimson, warm | jwin_mv, cold (3 runs, 10-31) |
|---|---|---|---|
| whole bake | 220 | 28 | 4,746-6,775 |
| Blender evaluating each frame (`frame_set`) | 16.2 | 3.8 | to measure |
| screen words, reading meshes out of Blender | 84.5 | 0 | 1,334-1,779 |
| screen words, fitting (`fitgeo`) | 91.2 | 0 | 3,117-4,660 |
| lit meshes by frame | 6.6 | 6.5 | |
| capture, export, write, check | 10.6 | 6.7 | 81-89 |
| everything else | 11 | 11 | |

jwin_mv's 918 frames carry 180 screen words, and nearly all its bake is theirs: fitting is two thirds of it. With the
cache (step 2) a rebake that changes no word takes the rest, about 300 s. The scene's own stages (`sim`, `clip`,
`shots`, ...) take 92 s on Crimson.

**Cabin Essence (2026-11-01, warm screen cache): 405 s → 180 s.** A profile of its whole build showed most of it was
Blender evaluating every frame again and again: the sampling passes of `perform`, `shots` and `sim` each change frame
~900 times, and each change cost 13 ms because ~1,000 driver-animated props (confetti, a track) were evaluated too.
Those passes now evaluate only what they read (`mkmmd.blender.scope`, 13 → 4 ms a frame); the bake reads every object's
matrix and visibility with one `foreach_get` a frame instead of one object at a time (its transforms 37.6 → 1.9 s, the
lit tracks 8.8 → 0.1 s); the clip ranges keep static boxes and take the objects as arrays. The bake's files are the same
bytes as before (but for mmd_tools' run-specific node names).

| stage | before | after |
|---|---|---|
| sim | 88-147 | 33 |
| clip | 56 | 42 |
| bake (warm) | 91-125 | 47 |
| shots | 46 | 7 |
| perform | 38-40 | 3 |
| cast | 20 | 20 |
| everything else | ~38 | ~28 |
| **total** | **405-427** | **180** |

Found on the way: a bake whose screen words came from the cache, or a stretch with no word shown, froze the breathing
colours of the words' materials (a material no shown object wears is not evaluated); `Lit` now shows an empty mesh
wearing them for the loop, and the colours match the drivers exactly on every frame.

What is left on Cabin Essence: `bake` 47 (Blender's frame changes 13, the glTF export 8, capture 8, check 4), `clip` 42
(Blender's frame changes and the depth search of ~1,400 objects a frame), `sim` 33 (five levels of chains, each sampled
over every frame). `cast` 20 -> 4 s: each model's mmd_tools import is kept in `<project>/.mk/cache/cast/` as a
.blend (keyed by the model file, the import's settings, mmd_tools' version and the code) and appended on later builds,
its shared node groups and images and its materials' mmd ids as a fresh import leaves them: the bake the same bytes.
The whole build: 165 s.

## Steps

1. **Measure.** Done: the bake's report has the parts above (jwin_mv's from its earlier reports; its `frame_set`
   share shows on its next bake).
2. **A cache key that follows the code that makes the word.** Done. A screen word's key holds what makes its geometry
   and the fingerprint of the code that bakes it (`mkmmd.cache.code`: `_ScreenWord`, the `Lit` methods that drive it
   and all they reach, `fitgeo` included, as syntax trees without docstrings) instead of two modules' whole source.
   On Crimson, an edit to a comment and to code no word uses kept all 297 words cached (28 s instead of 220); an edit
   inside `_geom` rebaked them all; the bake's files were byte-identical throughout. Tests: `tests/test_cache_code.py`.
3. **Several Blender processes per bake**, each over a share of the frames (as `mk render --jobs`): reading meshes out
   of Blender is single-threaded Python, and the bake does not need the GPU. The limit is memory: each process holds
   the whole scene. Status: open, after 1.
4. **The fitting in compiled code** (numba in `mkmmd/solvers`, or Rust beside mkr): two thirds of jwin_mv's cold bake
   (3,100-4,700 s), 90 s of Crimson's. The next step. Status: open.
5. **Read less from Blender.** mkr skins and morphs from the rest pose, the bones and the shape keys itself; only the
   meshes modifiers or geometry nodes make need baking frame by frame. List them per project and draw what mkr can;
   skip frames that do not change; read with `foreach_get`. Status: open, after 1.
6. **Rebuild only what changed.** Each build stage and each part of the bake keyed by a hash of its inputs, so a change
   to one shot or one light does not redo the rest (`--until` and `--skip` do it by hand today). Status: open.
7. **Bake and render together.** mkr draws the frames the bake has written, as `mk post --follow` finishes the video
   while the render draws. Worth it once the bake is the slowest part of a run. Status: later.
8. **No Blender in the build for MMD models.** The rig, the physics and the shape keys evaluated in Rust from the PMX and
   VMD files, with parity tests against Blender's evaluation as mkr has against its frames. The largest gain and the
   largest job. Status: later.
