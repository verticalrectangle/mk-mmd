# Playbook: making a video with mk

For agents (and people) driving `mk`. Every command prints JSON; exit codes are 0 ok, 1 a check, gate or verification failed, 2 usage,
3 Blender or runtime error. [design.md](design.md) is the reference for the contracts and every `mk.toml` key behind what follows;
[modelling.md](modelling.md) is how props are modelled. Times in `mk.toml` are clip seconds, frames are Blender frame numbers.

## Rules that never bend

- Never print or quote lyric text: not in code, logs, briefs, commit messages or agent transcripts. Refer to words as `(line, word)`.
- Never commit models, motions, audio, lyrics, reference clips, reference photos or renders; projects keep them in their own folders.
- Never guess a licence. Read the readme lines (`mk assets show SLUG --terms`), record what they say, and say so when a readme only links a page.
- Never loosen a threshold to make a check pass without saying why. Never call a shot good before you have looked at the images.
- Read the build log. `WARNING` lines are findings (a hand short of its goal, a prop over its form limit, a pattern matching nothing).

## The loop

| Phase | Do | Commands |
|---|---|---|
| 1. Brief | the person's wishes part by part and for the video as a whole; your treatment and questions back | `mk brief new`, `mk brief show`, `mk review open` |
| 2. Timeline | beats, downbeats, loudness, note onsets, word timings | `mk timeline analyze`, `show`, `onsets` |
| 3. Storyboard | shots on downbeats, the effect at each cut, what each output shows | |
| 4. Cast, sets, props | register and clear assets, model what is missing, place everything | `mk assets`, `mk inspect`, `mk ref photos`, `mk build --until props` |
| 5. Pose, perform, sim | seats, grips, gaze, blinks, lip sync, hair in the wind | `mk build`, `mk grip`, `mk ref measure` |
| 6. Shots, effects, type | cameras per output, looks, transitions, inserts, glitches, lyrics and signs | `mk build --until text` |
| 7. Checks | numbers that define done | `mk check` |
| 8. Look sheets | the cut in every output, details from all sides | `mk look` |
| 9. Render and post | draft, preview, final; grade and encode; review | `mk render`, `mk post`, `mk play` |
| 10. Credits | every asset named, nothing unreviewed | `mk assets credits` |

Go back as often as needed: build, check and look are cheap (seconds); render and post are not. Repeat 5-8 until the checks pass and
the sheets look right, then render a draft, post it, and render the final once the draft is approved. A person judges motion from
a playing video, not from your sheets and not from the checks: after a fix, render the changed shot in draft with mkr (`mk render
--engine mkr --preset draft --frames A:B`: seconds; a whole 22 s show in three sizes drew in two minutes where its final took
nineteen), post that, show it, and render the final only once it is approved. A render after a rebuild draws what the change
reaches and keeps the rest (the report's `reused`), so the final after a fix is the fix's frames, not the clip's.

## 1. Brief

A video starts from what the person wants. Given a song, `mk brief new DIR --song FILE [--lyrics FILE]` makes the project
and opens its Brief tab beside your pane ([Briefs](design.md#briefs-mk-brief)): they choose the part of the song, paste its
lyrics, time the words, and write what they want for each line or group of lines and for the video as a whole. Send types
it into your pane; `mk brief show` reads all of it: each part on the song and on the clip with its lines, words and bars,
and `open`, the stretches no part covers. Someone who tells you in chat instead gets the same: put their wishes into the
brief's parts yourself (`brief/brief.json`). Nothing you write about it quotes the lyrics: refer to `(line, word)`.

Then pitch a **treatment**, as a music video director pitches an artist: a review (`review/treatment.review.toml` with
`[review] preselect = true`, [Reviews](design.md#reviews-mk-review)) that answers the brief part by part.

- **Their wishes first.** A question per part: their note in your words, the shot you would make of it as the recommended
  option and one or two others, `music = { project = "..", from, to }` over the part (clip seconds), and test frames of
  their own scene as pictures (`mk look` renders in seconds), not stock references.
- **Go further.** The music suggests what the brief did not say: a freeze-frame orbit on the biggest hit, glitches on a
  snare fill, the light turning at the chorus, the camera holding still through a long note, type that hits on the words
  that matter, a chorus that grows each time it comes back. Each extra is a question with `tag = "my idea"`, Keep
  (recommended) and Cut. When an idea needs something mk cannot do yet, say so, and build it.
- **Ask deeper**, only where the brief left room: who the camera is (someone in the crowd, a film crew), whether a part
  that comes back repeats its shots or grows, what lands on the drop, the frame people will screenshot, lyrics on screen
  or not, the look of each part. Each with pictures and your pick as the recommended option.

`preselect` starts every question on your pick, so Send alone means "go with your taste" and the person changes only what
they want changed. Their answers, with the brief, are the plan: the song span (the clip the brief set), the outputs (the
brief's sizes: `16x9`, `9x16`, `1x1`, each gets its own camera for every shot), the palette ([Palettes](design.md#palettes)),
who is in the scene, where, with which props, and the effects. `mk brief new` starts at 30 fps with `frame0` 91: at least
3 seconds (90 frames) come before clip time 0, where poses ease in and hair settles (`[scene] settle_frames`, default 24).
A whole song is many parts: build it and show it section by section (a draft of each for review) rather than all at once;
a final of a whole song takes hours to render.

## 2. Timeline

`mk timeline analyze song.flac --start 15.59 --duration 30.6 --out audio/timeline.json` (or just `mk timeline analyze` in a project with
`[audio] file` and `start`) turns the song span into beats, downbeats, per-frame loudness, note onsets and word timings by
`(line, word)`; `mk timeline show` prints its numbers. Pick shot boundaries on downbeats from it. A lyrics file (`--lyrics`, one sung line
per line) beats Whisper's line breaks. For a strummed or plucked part, `mk timeline onsets` recomputes the onsets of a Demucs stem. If
the transcription is wrong the words move, not the beats: check `words_source` and `word_stats` in the summary. The Brief tab's
Time the words runs this on the clip already (with the lyrics when the person pasted them); run it again when the part changes.
`mk timeline hits` then writes the moments to key things on (`audio/hits.json`, `[audio] hits`): kick, snare and hat hits, the stops
(the mix falling silent), the bass's silences and the notes with their pitch; a video that moves on the drums starts there. Check a lane
against the music before leaning on it (`mk site --tab music` shows them): a lane is a detector's guess, and `db` sorts the real hits
from the leakage.

## 3. Storyboard

List the shots as a table (name, `from`, `to` in clip seconds on downbeats, subject, camera, what happens at the cut). Decide at each
cut whether it is a plain cut or a [transition or insert](design.md#transitions-and-inserts), and what each output shows: a 9:16 output
crops differently from a 16:9 one (`[shot.aspect.<output>]`, `frame = {...}` to solve the lens per output, or a lens shift for an exact
square crop of a vertical master). Say whose each idea is (the brief's part, or yours): the person should always see which
ideas were theirs. Put the plan in comments of `mk.toml` or a local file, never in the repository's tracked docs.

## 4. Cast, sets and props

1. **Know the model.** `mk inspect MODEL.pmx` (or `mk assets add MODEL.pmx --kind model`) and read the summary: `kind`, `quirks`, chain
   `families`, `expressions`, `measure`. Quirks are things that will bite later (missing bones, non-bone vertex groups, locked physics
   joints, missing textures). A character that does not exist yet is made from a model base: `mk model new NAME`, then
   [model_base.md](model_base.md); `mk model build` writes the PMX and the rig.json a `[[cast]]` takes, and her VRM beside them
   (`<name>.vrm`, mk's own model file for VRM apps: docs/design.md, VRM).
2. **Register and clear licences.** `mk assets add` for everything you use, then `mk assets show SLUG --terms`, read the readme lines, and fill
   in `license`, `restrictions` and `credit` with `mk assets set`. `mk assets credits` fails while anything used is unreviewed.
3. **Set up the project.** An `mk.toml` with `[project]` (fps, frame0, duration, blend), one `[[output]]` per output, the `[[cast]]`,
   collider sets and the checks that define "done" (see the example below). The README has a project that builds.
4. **Build the places.** Library sets and props ([Sets](design.md#sets), [Props](design.md#props)), placed by rules instead of coordinates
   ([Placement](design.md#placement)): `place = { on = "room:floor", align = "edge:back" }`. A placement that cannot be met fails with the
   numbers and the prop in the way. Props from MMD accessory models need no card ([PMX props](design.md#pmx-props)). Run `mk build --until props` and
   look at the room with `mk look` (an orbit camera clips at 100 m: look through a shot camera to see a far skyline). A toy paper theatre
   round a marionette is `kind = "paper_theatre"`: give it `arrive` (its proscenium drops in, its clouds, stars and moon fall in on their
   threads) and `step = 12` with `step_from` your `[[stepped]]` window's `from`, so the set and the cast change pose on the same frames.
   Its pond and machinery (`pond`, `panorama`, `rise`, `night`, `moon`, `loose`, `sun`, `day`, `house`) give it a wave machine,
   lily pads and lilies that pop open on a pulled tab, drift-paper ripples off the pad she lies on (which bobs with the
   marionette's rest box under `pond.ride`), scenery that rolls past a hopping puppet, ground that sinks away into the stage's
   cellar as she is flown, a cranked day-to-night backdrop with constellations drawn on the notes, a croissant moon and a sun
   that pops up, and `house` takes everything but the water away for a camera straight above her; key them on the words and
   notes they land on, and give the hands `obj('<set>_tab').loc` and `obj('<set>_crank_knob').loc`. With `cafe` instead it is a
   little café of paper round her chair (`library:paper_cafe_chair`, her `sit = "<chair>:seat"`, and `library:paper_cafe_table`
   beside her with a cup that `rattle`s and a `scissors` anchor): a tall window right behind her head onto a blue paper sky
   and a low skyline, pendant lamps, bunting, a shelf of cups, a chalkboard, and a border of sky over the room whose old
   taped rip over her bursts open on `cafe.hole` and is taped across on `cafe.slap` (a hand rides `obj('<set>_slap').loc`).
5. **Model what is missing** with the loop under [Modelling props](#modelling-props), and measure it with `form`.
6. **Measure real people (optional, for performances).** `mk ref search "woman singing in car"`, `mk ref add ID...`, `mk ref track`, `mk ref measure`
   give blink, gaze, nod, sway and mouth numbers shaped like `[perform.<cast>]`, each with a confidence; `mk ref sheet` shows what was
   tracked; `mk ref clean` frees the disk. Prefer front-on or three-quarter clips (side-view faces are rarely found), read `qc.warnings`, and
   treat values with low `n` as hints.

## 5. Pose, perform and sim

- **Pose** ([Posing](design.md#posing)): `sit = "car:driver"` puts the hips on a seat use point; a standing pose writes its feet as
  `{cast = "rin", point = [x, y, 0]}` (in the character's own frame, so it survives moving her). Hands that hold things are built, not placed
  (below). The pose stage reports each arm IK's miss in mm and warns past 5 mm: lean the character (`lean`), move the seat (`sit_offset`) or bring the prop closer.
  Look at the pose with `mk build --until pose --out build/pose_test.blend` and `mk look --view front,3q,left,back --frames N --sheet`.
- **Perform** ([Perform](design.md#perform)): gaze events over an idle target, eye-only glances, breathing, sway, nods, beat bob, blinks,
  expressions, lip sync from the timeline (`sing`; `words = [[2, -1]]` mouths only chosen words, a partner shouting the line endings), twitches,
  strumming (`strum`: down strokes on the downbeats, the pick meeting the string at each strike).
  A body that stands and plays, sings or waits gets a `groove`: her weight goes from foot to foot in time (hips over the standing
  leg, that hip up, the chest back so the head keeps still, the free heel off the floor) and her knees let her down onto each
  beat. It is ONE motion; a `rock` with a `bounce` over it is a pendulum and a pulse that know nothing of each other, and she
  bobs like a buoy. At a mic keep `still` high (0.85) and the sway a bar a leg (`every = 4`); off it let her head go
  (`still = 0.5`) and change leg twice as fast. Window it out of her steps, turns and spins (`from`, `to`), read the report's
  `knee_deg` (8 to about 50) and look at it in a front strip a beat long and as curves, not in one still: a sway is right or
  wrong in time. A strum's rhythm comes from the guitar's stem, not from taste: measure the onset energy on each eighth of
  the bar; where every eighth sounds, strum them all and `accent` the loud ones, and where only some do, play those (`hits`)
  and let the hand keep time over the strings on the rest (ghost strokes).
  A character that turns round uses `step_turn` (the head, chest and hips in turn, the feet stepping), and one that twirls (in the
  air or on the ground) `spin` (a wind-up, the head spotting the camera and whipping round ahead of the chest, the hips trailing,
  an overshoot settling back), not a `[[key]]` on its root's rotation: that spins it like a figure on a turntable. Look at a spin
  in a front and a top strip every 0.1 s (`mk look --view front,0:89 --frames t=A:B:0.1 --strip`). A character in the air (a
  jump, a blast; its height keyed on the root) gets `air`, so its legs tuck and reach down to land instead of hanging straight, and
  a `crouch` before it to brace. One that floats for seconds (`rise`) gets `air` with `kicks` too (`every = 2` lands them on the
  beats), so its legs tread lazily in turns instead of holding the tuck like a statue, and a `point` past the tucked shin's slope
  (about 55) so its feet hang pointed, not flexed.
  A walk walks with its whole body: `walk = {steps, arms = 0.11}` swings the arms from the shoulders (the hands need a `rest`
  move), turns the pelvis with the legs and the chest against it; a strut is `cross = 1, sway = 0.035, twist = 8, list = 4,
  arms = 0.13` (the feet on one line, the hips over the foot that is down and swinging). Do not play a walk VMD's arms on her
  instead: a VMD turns each bone from its own model's rest pose, and on a model whose arms rest another way they come out
  splayed with the wrists kinked (it happened: forearms 30 to 44 degrees out all through a walk). Look at a walk from the
  front, every third frame, before it is done, and give it an `arm_sweep` check (under 20): arms that fold in across the
  body and out again at every step are a stiff sideways sway, whatever their splay and their wrists measure (that
  happened too, 28 degrees, with the elbows out to the sides: the walk now keeps them behind her).
  Keyed curves (`lean`, `tilt`, `turn`, `head_*`, `crouch`) pass the keys they meet on their way and stop only where they
  turn round or hold, so key a body the way one moves: nothing arrives at once (the hips, then the chest a few frames later,
  then the head; the side lean after the forward one), a move runs a little past its end and comes back (a key beyond, then
  the end), a held pose drifts (two keys a degree apart) instead of standing dead still, and a landing gives and pushes back
  before it rises. Then measure it: a `stillness` check on each shot where she stands or sits (the share of frames her chest
  does not turn: under 40 %; a statue with busy hands scores 60 to 80), and `mk q` on the head for the frames where its speed
  jumps.
  A zero-g moment is `[sim.<cast>] gravity` eased out over a beat or two (`[[t, 1], [t + 0.4,
  0]]`), not cut to 0 on one frame: a sudden cut flings the hair. A character who flies in a plate stays where she is and the
  air moves past her ([Sim](design.md#sim)): `gravity` 0 while she flies free, the wind's `speed` and `direction` keyed against
  her flight (shooting up: the air down past her, slowing as she does; a dive: from below and ahead), `tumble` turning her heels
  over head, sideways or over onto her front (`axis = "twist"`), and a take that starts on a jump is a cast member of its own
  (`shows`) keyed from well before it, so her hair is already flying on its first frame.
  Group and material morphs (vowels with a tongue bone, a blush) are keyed like any other: the cast stage binds the model's morph sliders
  (`bound_morphs` in its report).
  Take the numbers from `mk ref measure` or a timeline, not from taste.
- **Moves** ([Moves](design.md#moves)): `[[move.<cast>]]` puts named moves on the beats (`chest_pat`, `paws`, `point`, `heart_wink`,
  `hands_up`, `stank` ...) or on clip seconds of their own (`hits = [...]`: a `bonk` on the snares; `flung` throws both arms up on its
  time, as a reaction does, where other moves set off early to arrive on it), and `{name = "rest", hand = "R", place = "mic"}` keeps a mic hand at the mouth between them (a two-hand move then
  plays with the free hand only; `place = "dainty"` rests a hand lightly on the front of a skirt, elbows in); they compile to the pose's hand keys (riding the chest) and perform's lean, twitches and faces.
  The gag moves of a kids' show (`mkmmd/core/gags.py`): `bounce` (an EDM bounce, knees pumping on `hits = { lane = "sub" }`, fists up
  and a hop on `accents`), `run_in_place` (knees up, fists pumping; on a treadmill prop stand her on its `run` feet), `yawn` (a
  stretch, or hugging a pillow with `prop`), `chomp` (a held carrot bitten on its hits, its `bite<k>` fired), `brush` (a toothbrush on
  the teeth), `wag`, `wink_point` and mimed `chores` (`chore = "sweep" | "stir" | "water"`); a held prop is placed by the move (no
  `attach` of its own: key its `scale` to show it). `[perform.<cast>] squash = { lane = "sub" }` squashes and stretches the whole body
  on a lane's hits, keeping its volume.
  The mosh moves of a pit (`mkmmd/core/mosh.py`): `pound` (a mallet, `library:spike_mallet`, in both fists, swung down on the
  hits onto `target`, a stake or a bowl; start it a little before the first hit and key the mallet's `scale` to show it; a
  target out of reach is refused with where the face can land), `headbang` (`depth`, `hands = "hips"`; list the hair in
  `[sim.<cast>]` so it whips), `pogo` (a jump on every hit, as high as the time allows; `bounce` only hops on accents),
  `wotagei` (`props = { L = "pen_l", R = "pen_r" }`, two `library:penlight`s, `routine = "arcs" | "eights" | "punch" |
  "circle"`; key the penlights' `color` on a lane with `values`), `whistle` (a conductor's cord pulled on the hits) and
  `bass_face` (`style = "scowl" | "pout"` with a nod on the hits; it keys only the head's nod, so it plays over a `pound` or a
  `wotagei`). Check `hand_room`, `wrist_bend` and `joint_limits` after every change to the pit's moves.
  The poses `hair_flip`, `over_shoulder`, `hand_hip` and `finger_gun` hold a body and head attitude too (mirrored for the right hand): put
  one on a dead stop of the music, between two walks, and shoot it from the side it reads from. A `hair_flip`'s hand runs
  through the hair: with long simulated hair watch it in motion before keeping it (a lock can fly out like a whip). The hands
  come in from outside and stop at the body and clothes (simulated hair and a simulated skirt make way),
  and wrists stay within 60 degrees of the forearm: the pose report's
  `moves.places` says how far each place was kept out and how much its wrist bends. A hand travels between places at a human pace
  and the build refuses a move it cannot reach in time (it says how long the trip takes): a few gestures, each held, read better
  than one on every word. Draw a pose sheet of every move, front and 3/4, and look at it closely before building the shots, then
  watch the moves in motion: a sheet does not show how a hand gets from one place to the next.
- **Sim** ([Sim](design.md#sim)): hair, ears, tails, skirts and the chains under a garment's sleeves (`sleeve`) are solved outside Blender
  and baked. List the families, give the colliders (the seat, the car, the fingers, the floor), and `wind = { carrier = "car", ... }` in a
  moving vehicle. A simulated skirt with a hand resting on it (`place = "dainty"`) wants `bend_max = 30`, and one on a dancer who whips
  round faster than cloth could follow wants `inertia = 0.5` (`[sim.<cast>.params.skirt]`). An mk-built sweater's sleeves want
  `[sim.<cast>.params.sleeve] radius = 1`: their chains' capsules are as thick as the bag they carry, and the sweater wears colliders the
  size of its knit, so at that radius a bag lies on the knit when an arm comes to her side instead of hanging in it. `mk build --skip sim`
  is the fast loop for poses and performance.
- **Folds** ([Folds](design.md#folds)): a model whose rig.json has fold shapes (an mk-built garment's crook of the elbow, the knit bunching
  as she leans or twists) gets them keyed from her pose with no table; `[folds.<cast>] depth` scales them, `enabled = false` turns them off.
- **Cloth** ([Cloth](design.md#cloth)): a model whose rig.json has a cloth copy (an mk-built garment: the sweater's knit) is simulated with
  Blender's cloth when the scene says `[cloth.<cast>]` (an empty table will do), and then its fold shapes and the chains it replaces are left
  to it (take `sleeve` out of the sim's families). It costs about 0.1 s a frame for the sweater, once: a rebuild of the same motion reuses
  it. Read the report's `pen_mm` (how deep the cloth went into her and on which frame) and `lag_mm`, and look at that frame; `hold` keeps
  the cloth closer to its shape when a move is too fast for it, `mk build --skip cloth` is the fast loop for everything else.
- **Worn props** (a guitar on its strap, [Playing a worn guitar](design.md#playing-a-worn-guitar)): `wear = "rin"` on the prop, `grip = "guitar:neck"` and
  `grip = "guitar:strum"` on the hands, `strum = {...}` in `perform`; check `strum`, `string_play`, `prop_body` and the arm `contact`. The
  strap lays itself on her (nothing to tune) and the sim stage makes it a soft band: look at it from the side and from behind
  in motion; `strap = { slack }` in the wear loosens or tightens its free runs.
  An `offset_guitar` (the shoegaze one) adds `anchor_to = "rin"`: its tremolo arm goes in her picking hand and swings with the
  strum; lay it into her fingers with `slots = { arm = { swing, lift, length } }` and look at the hand. A pick hand's
  `hold = "across"` (the offset guitar's default) is a real plectrum grip, `clear` keeps its skin off the strings: look at the
  hand from the thumb side, the index side and along the strings before showing a close-up. The
  strings the hand plays vibrate on their own ([Strings that ring](design.md#strings-that-ring): only those strings, from the fret
  held, a strum's one after another): look at a close-up of the neck and of the picking hand frame by frame before trusting a take.
  A pick hand that leaves the strings (lifted off after the last chord, a peace sign by her face) takes `keys` on its strum
  grip, `{t, grip = true}` where it leaves and comes back ([Letting go](design.md#playing-a-worn-guitar)): end the strum first.
- **Marionettes** ([Marionette](design.md#marionette)): a `library:marionette` prop and `[perform.<cast>] marionette = {prop,
  from, ...}` hang her on threads from a wooden cross over her head: key the cross (`tilt`, `sway`, `rock` on the beats, `knees`
  in turn, `hops`) and she follows late and loose, then `cuts` let her go a thread at a time and she takes over with a
  `groove` of her own. Step through each arrival and each cut at full size in the shot's camera, and through a beat of
  the dance from the front: a thread through the guitar or her face, a hand through the guitar after its cut. She can
  lie and fly too (design.md: Marionette, the world's keys): `lie = [[t, -90]]` hangs her flat on her back and `raise`
  lowers her onto her surface (she touches it when the control's last drop meets it, a few hundredths before), `float`
  bobs the water under her, `kicks` are her own, a jerk of `lie` and `raise` pulls her up onto her feet, `hops = {land
  = [...]}` land ON their times, `swim` while flown flat, `cuts` in the air let a limb drop and swing, `release` lets the
  control go after her as she falls (give her `air`), `land` sets her on a point (a ball's top), `yank` snatches her up by
  her head thread, `knot` hangs her back on every thread; `slots = { control = "stirrers" }` makes the control two coffee
  stirrers. Give her hands an `at` (the threads move them), the sim the prop's `<prop>_rest` box with `floor = false`,
  and for a loop the keys before 0 that are the clip's last ones less its length, over a pre-roll long enough to replay
  her last tip back (`knot.loop` replays the knot's swing), so her hair comes to frame 0 as it is on the last frame;
  `over` keys hold the control out of a plate camera's short lens, `width` keys the threads thin for a close plate.
  Watch the touch, the jerk, a hop, each cut and the landing frame by frame in the shot's camera, and the portrait from
  above: her face clear of the bars and the threads. Set free (design.md: Marionette, set free): `seat` sits her in a
  chair on her threads, `gather` takes them into one fist over her head, `scissors` brings a `library:paper_scissors`
  from the table to the bundle and `snip` shuts the blades ON its time (the control dropped past her, the pieces on her
  her tails), `ascend` floats her up out of the chair, `haul` has a hand catch her tails and haul her down in tugs ON
  their times and lets her fall into the chair ON the `knot`, `crossed` crosses her arms; `loop` replays the end in
  the pre-roll. `tails` alone (no `from`) gives a member who is never on the control six free threads streaming in
  her sim's wind (a flight's, a seat's take). Step through the gather, the snip and the landing at full size.
- **Flights** ([Flights](design.md#flights)): `[[flight]]` flies a prop relative to a target, a bone of hers or a point,
  as a hand flies a rod puppet: `in`, `dart` and `glide` land ON their time (put darts on the words and the notes),
  `circle` goes round her head, `perch` lands on it, and between moves it hovers with small darts. `library:dragonfly`
  is the paper one on a steel wire: give `hand` (its far end) a point off the frame's edge, its slots the set's `step`
  and `step_from` so it changes pose with the theatre, and a `gaze` at `"<prop>:head"` makes her follow it. Watch a
  dart and a circle frame by frame: the wire bends round her head and must never cross her face. `library:paper_planet`
  is a paper ball on a skewer with a toothpick flag: `up` brings it into place on its time, `land` makes it sink, bob and
  settle (key what lands on it to `mkmmd.core.paper_planet.top`). `library:paper_scissors` is a child's card scissors on
  a brass split pin: `attach` it to her hand (it then steps with her `[[stepped]]` window), key its `open` (0 shut .. 1 a
  V of 60 degrees) so the blades shut ON the snip, run the thread through `"<prop>:cut"` (the blades close over it below
  open 0.27), and step through the snip frame by frame in the shot's camera.
  `library:paper_plane` is a big folded paper dart she rides side-saddle (`sit = "<prop>:seat"`, her legs over its
  `side` wing): its `bank`, `pitch` and a landing's dip (`land` ON her landing) are keyed on its root on the theatre's
  `step` grid, so she rides them; keep it level on the scene's first frame (where she is seated), and draw trails from
  `"<prop>:tip_L"` and `"<prop>:tip_R"`, which ride the wing tips. Her sim takes its colliders (her hair lies on the wing
  behind her); give her skirt `[sim.<cast>.params.skirt] bend_max = 90` so its back folds onto the paper.
- **Gag props** (their rows in [Props](design.md#props)): `library:pillow_cloud` (key `puff` 0 -> 1: the pillow puffs into a cloud; `hug`
  anchor, `hold_L` / `hold_R` palms), `giant_carrot` (key `bite1`..`bite4` CONSTANT at the chomps, in order; `use.grip` `hold`, `use.bite`
  points; the chunks fly and fall on their own), `record_treadmill` (stand the cast at [-0.45, 0, 0.16] with `feet = "<name>:run"`, key
  `rpm`), `parent_cutout` (`figure` father / mother; key the father's `wag` or `wagging`), `sandbox_piano` (`sit = "<name>:bench"`, hands on
  `"<name>:keys"`), `chalkboard` (`[[text]] on = "<name>:board"`, `drawing = "noughts"`), `carrot_plush`, `dinner_plate` (`veggies =
  false` empties it) and `colorbar_card` (`style`: the bars of `mkmmd.core.colorbars`). Light a cut-out against a backdrop: its drop
  shadow is a printed layer behind it.

Hands. `grip = "car:wheel"` (or `"rest"` with `rest = "car:sill_R"`) in `[pose.<cast>.hands.L]` solves the fingers on the model's own skin
([Grips](design.md#grips)), where `at` plus a finger preset only guesses. A `WARNING ... short of its goal` means the seat is out of the arm's
reach; a grip digest past 3 / 1 / 1 mm (gap, penetration, clash) means the prop does not fit that hand. Pin it with a `contact` check between the
fingertip bone's `.tail` and the prop.

- **Ribbons** ([Ribbons](design.md#ribbons)): `[[ribbon]]` draws 3D paint strokes the camera moves through: a `crayon` `orbit` round a
  bone, a striped `toothpaste` squeezed `from = "prop:tube.nozzle"` (`library:toothpaste_tube`; it keys the tube's `squeeze`) and
  wrapped round her, a `neon` `swoop` hoop for the lens to fly through; `draw = [t0, t1]`, `retract` or `chase`, `pulse = { lane =
  "kick" }`. Each is pushed clear of the cast's bodies; read `clear.least_gap_mm` in the report and add a `ribbon_clear` check (it
  measures the scene after the sim). `library:toothbrush` is the giant brush (`use.grip` `hold`, `use.anchor` `brush`).

## 6. Shots, effects and type

- **Shots** ([Shots](design.md#shots)): one `[[shot]]` per cut, a camera per output is made for you. Mount the camera on a car or a set, aim it at a target
  (`"cast:rin.head"`), give lens and lag; `frame = {subject, fill, solve}` solves the lens or dolly per output. A shot that two outputs frame differently
  overrides keys in `[shot.aspect.<output>]`. A side-scroller is `follow = {target = "cast:rin.center", axes = "x"}` with a fixed `at` and `look` (her hops
  rise in the frame, a tumble never spins the camera); `solve = "dolly"` with a keyed lens is a dolly zoom; `keys[].roll` turns the frame on the beat.
- **Looks**: `style = "silhouette"` (flat background, the scene in one colour, accents in another), `style = "vector"` (a flat-vector drawing: every material a tone
  of the project's `[vector]` table, two-tone shading, outlines) and `reflection` (her image in a pane) are composed at render time; `mk look` draws them as
  `mk render` will, `--no-styles` draws the shot as lit. A vector look starts with the model's material names: `mk q BLEND --list materials` lists them with
  the name a `[vector] materials` rule matches, and the build warns about a rule that names none. `[[ring]]` entries flip a vector shot to the look's
  `opposite` palette inside discs and bands that sweep out from a point on a beat. A vector tone `{texture = true, shade}` is cel (the model's own
  colours in two tones); a shot's own `materials = [{match = "*", tone = "cel"}]` and `lines` restyle everything in it; `sketch = {...}` draws the
  look as boiling ink or pencil line art, `sketch = {pen = "crayon", inks = "tones", fill}` as waxy crayon on paper (each part's lines in its
  own crayon, coloured in with `fill`), and `cutout = true` cuts the look out of construction paper (rough edges, a white rim, layered shadows).
- **Medium shifts** ([Medium shifts](design.md#medium-shifts)): `[[medium]] {from, to, style, ...}` draws the whole picture in another look for a
  window, whatever shot is in the cut: a beat or two of cel, a flat cartoon (the vector close-ups' tones on a wide shot) or a sketch, cut on a snare,
  a stop or a big hit, and back. Pair a cartoon or a sketch with a `[[stepped]]` window over the same span (she moves on twos, the camera on
  every frame; [Stepped motion](design.md#stepped-motion)), and print or glitch flashes with `[[fx]]` `halftone`, `split` and `mosh`. Keep them
  short and on the hits, vary the medium, and look at each one: a dark lit frame gives a halftone nothing to screen, and a tint over a stop
  washes a medium out unless its `strength` comes down.
- **Freezes**: `[[freeze]] {from, to}` holds the world while the shot's camera goes on moving (a hit-stop orbit); `mk look` and `mk render` show it, `mk q`
  reads the timeline as it is.
- **Transitions and inserts**: composited by `mk post` from layers `mk render` draws next to the frames. Windows may not overlap and must lie inside one shot of the
  cut (a `spin` straddles its cut: the frame spun up into it, the next one down to rest after it); the build refuses a window that starts before `[scene] start`.
  A `slice` cuts the frame through the figure (`center = "figure"`) and parts it onto the next shot, white `edge`s like a sticker's; a split screen is
  `[[insert]] shape = "split"` (another shot or a video beside the host, sliding in, trading sides on a hit, `aspect.9x16` top and bottom); an
  iris is `shape = "iris"` (a round window opening from a scene point, a toy's eyehole, onto another shot or a video, growing into it on the cut
  with `out = "expand"`); `ring = "vinyl" | "plate" | "lens"` makes it a portal (a spinning record round it, a dinner plate with googly-eyed veggie
  `sprites` on its rim, TikTok chrome lens rings). `kind = "page"` turns the frame over like a picture book's page onto the next shot; put one at
  the clip's end (`at = [project] duration`) and it is the loop's wrap onto frame 0 as it finally looks, its `[[fx]]` and type on (nothing needs
  keeping off frame 0). The physical cuts: `kind = "peel"` pulls the picture off the next one as a sticker from a corner (a hand's pinch, pull
  and flick; `mkmmd.core.peel.corner` is where the pinched corner is, for a hand to ride), `"shatter"` breaks it like plastic (`material =
  "paper" | "glass"`) and throws the shards at the lens, `"tear"` rips it down a jagged line and swings the halves off (`tear.grips` for two
  hands). Their first window frame is the impact, the pinch or the rip: put `at` a `dur` after the beat. Preview a frame inside one with `mk look`.
  A transition carries off only the cut's picture: mark the `[[fx]]` that make the picture (a sheet, the smears on it) `under = true` and
  they are drawn before it and go with the outgoing shot, while the rest (a hand riding the corner) stays on top. Mark what makes the next
  picture (the sky the paper tears open onto) `under = "incoming"`, from the window's first frame: it is laid on the incoming shot's plate,
  so the transition uncovers it as the frames after the cut show it.
- **Glitches** ([Glitches](design.md#glitches)): `[[glitch]] {from, to, objects}` makes a character (its meshes, `"Rin_*"`, and what it holds) break up in
  jumping slices and colour ghosts and vanish by `to`; the build keeps it hidden after that. Start it on a hit; `seed` gives another take.
- **Frame effects** ([Frame effects](design.md#frame-effects)): `[[fx]]` bends, smears and draws over the finished pictures: `lens` (a black hole's
  lensing about a scene point), `echo` (figure trails), `slitscan` (time smeared across the frame), `push` (zoom into a pupil that shows the
  first frame as the effects before the push finish it: the seam of a loop, so put it last in the table), `scope` (a stem as an
  oscilloscope trace), `zoom` (a punch on every kick), `shock` (a pressure ring
  running out from a point), `slab` (Persona's slanted panels slamming in and flung off), `tint` (the Dark Hour's green over a stop),
  `halftone` (comic print: Ben-Day dots in the shadows, the plates off register), `split` (a broken signal: torn bands, RGB apart) and
  `mosh` (a datamosh: the frame before a cut, graded as it was, dragged along the new shot's motion; with a `flow`, a drift, zoom, swirl or noise,
  a still's own picture dragged in blocks, and `mask = "ground"` keeps her clean). The collages cut a figure (`objects = "Rin_*"`) out
  onto flat colour and step on the hits: `grid` (a chessboard of clones, a knight hopping across it), `orbit` (clones on a ring round
  her), `mirror` (folded about a line: with `place` beside it, two of her face each other), `panels` (slanted split screen), `stack`
  (a banded echo; `mirror = true` adds its mirror image about `center`, so one figure flanks her on both sides; its bands may be
  wallpapers of googly-eyed sprites, `{color, sprites = {tag = [...]}, eyes}`, its copies drawn in eras' `looks` (halftone, flat,
  vhs, pixel, chrome), falling into an endless `recede` tunnel, `delay = [...]` busy clones that `peel` off to places of their
  own, and a `squash` on a lane about their feet: design.md Rainbow stacks) and `footage` (her as
  a sticker, a white `edge` and a drop `shadow`, over a video `mk ref footage` fetches from Pexels); `back = "picture"` lays any of
  them over the frame as it is instead of flat colour, a list of `back` or `fill` colours flips on every hit, `cells = [1, [2, 1],
  [2, 2], [4, 2], [4, 4]]` with `divide` makes the grid divide like cells on the hits, `fill = "silkscreen"` prints her as a Warhol,
  `keys` move her (a stand-up tipping over flat) and `[fx.aspect.<output>]` gives an output its own keys. A figure that should not
  share her set (a guitarist beside a dancer, nobody in front of anybody) comes from a `plate = true` shot of its own: `shot = "..."`
  takes the collage's figure from that shot's render, `at` and `height` set it into the frame. Give a collage a medium or close
  framing (a far one leaves tiny clones; an orbit's `scale` brings her near) and put it first in the table, before a zoom or a
  shock that should move it. A `clip` turns footage round: the video itself as a die-cut sticker over the picture (a rounded
  `rect`, a `circle`, a `heart` or a `star`, moved by `keys`; `blend = "screen"` for night light trails, the clip's black gone).
  A `cutout` lays real hands on: a clip cut out by its own matte (`hand.mp4` beside `hand.matte.mp4`), stepped at 12 poses a
  second, `sync = [video second, clip second]` landing a slap, a snip or a tear on a beat, its `anchor` (the fingertip) at
  `at` (a frame point or `{point = ...}`), with a white `edge`, a `shadow` and a light `grade` (design.md: Cutouts).
  A hand has to be seen to do its thing, in every output: look at it frame by frame. It holds what it holds only if its
  fingertip is `anchor`ed there (a hand that sways in its own clip takes a track, `anchor = [[video second, x, y], ...]`);
  it takes a sticker a sheet peels with `at = {sticker = <the peel's t>}`, and a peel transition's corner with `{peel =
  ...}`: a frame place keyed by eye fits one output and is off in the others. Blades go across what they cut (`turn`).
  Its arm must leave the picture inside the clip's own frame (turn it, size it, bring it in from the side the arm comes
  from): past the clip's border the arm is only that border's line drawn on. A `keys` row keeps what it leaves out, so an
  arrival names its zeros (`{t, dx = 0, dy = 0}`).
  A `rewind` winds the clip back as a finger scrubs a video: a paper scrub bar is slapped on, a cutout riding its
  playhead (`at = {rewind = <its from>}`, after it in the table) drags it left, and the picture is the clip's own at
  the seconds of its time `map` (a few `{t, from, to}` moments on the words, each slow enough to read, hard jumps and a
  tape's tear between; no type on them, as type is laid after the frame effects): step through every jump at full size
  (design.md: Rewinds).
  A `zap` makes a sheet's live sticker somewhere else ON a time instead of travelling: it tears into slices with its
  colours apart over the frames before, lies at the new place on the time (a `step` key of its sheet, or the next sheet:
  another plate, pose or size), streaks along the line, and leaves a ghost that is dragged along the footage and breaks
  up in blocks, bursting where a finger goes through it (`touch`); `fail = true` tears and comes back, twice. Put the
  zap's time on the beat, name every sheet she is on round it, and step through four frames before to twelve after at
  full size (design.md: Zaps). A `trails` with `sheet = "<name>"` (before the sheet in the table) leaves after-images
  of that sticker as its 2D keys fling it about: `train = {}` drops a copy each sixth of her height she travels (a long
  train when she is fast, none when she is still), `spiral` winds a nautilus of her round her middle; look at every
  second frame of a fast and a slow crossing, and at a spiral's whole window in motion.
  They overlap freely and stack in table order; key `strength` on the hits, and fire `zoom` and `shock` on a `lane` of `[audio] hits`
  (`mk timeline hits`). `mk look` previews them, rendering the frames an echo or a slitscan reads beside the ones asked for. A `whip`
  transition is a whip pan between two shots. The space kit's `floor` (mirror, chessboard, Tron grid), `starfield` and `disc` (round
  a `hole`) build a world in deep space; a `burst` of `shard`s on a `[[key]] lane = "kick"` bursts glass on every kick. Real 3D
  butterflies: `library:butterfly` (a swallowtail, monarch, morpho or small blue, 9 cm across, patterned in palette roles,
  flapping at `flap` beats a second, or one beat on every hit with `flap = 0` and `[[key]] prop = "hit", lane = "kick"`) and
  `library:butterflies` (a swarm of them in rings, an orbit, a drift or a burst; see their rows in design.md). A `mandala`
  ([Mandalas and trails](design.md#mandalas-and-trails)) rings her with sticker copies of herself, of registry sprites (`{tag =
  "butterfly"}`) or of drawn clovers, flowers, hearts, stars and butterflies, and of kits drawn with a pen's line (a moon, a rabbit,
  a carrot, dango; a daisy, a cassette, a pedal, cherries; a pond's lily, lily pad, bud, reed, dragonfly, drop; a café's croissant,
  cup, sugar cube, orange slice, coffee bean; a sky's paper plane, bird on a wire, bird flying (`bird_up`), cumulus, sun, kite,
  balloon, shooting star, feather: `pen = true` draws the flat ones as they are), the rings turning both ways, blooming on the hits and
  popping in one by one (`grow`); any sprite source takes `eyes = true` (googly eyes whose pupils jiggle on the hits and when
  the sprite moves, `{size, wobble, look, seed}`) and `mouth = true` (a doodled smile; design.md: Googly eyes), and the
  registry's cut-out vegetables are `{tag = "veggie"}`, one colour family with `{tag = ["veggie", "veg_green"]}` (`veg_red`,
  `veg_orange`, `veg_yellow`, `veg_green`, `veg_purple`, `veg_white`); `trails` leaves rainbow after-images behind her while
  she moves. The colour fields
  ([Colour fields](design.md#colour-fields-liquid-light-op-art-floods)) paint the picture, `mask = "ground"` behind her: `liquid` (a
  60s oil-and-water light show in the palette's `inks`, swelling on a `lane`; `mask = "figure"` with `blend = "multiply"` projects it
  on her), `spiral` (op-art: a spiral, rings, a moire, a sunburst or a bulging checkerboard, turning and stepping on the hits, or
  `pattern = "drift"`: still rings or rows of rail ties or wheels whose luminance steps make them seem to turn by themselves, every
  other ring the other way, the same way through a palette breath) and
  `flood` (ink dropped from a point, her head's `{point}`, filling the frame on a downbeat and drawing back); any collage's `back` may
  be such a pattern (`{pattern = "spiral", colors = [...]}`: a grid of hypnotic spirals). The remaps
  ([Remaps](design.md#remaps-kaleidoscope-tunnel-melt)) move the picture itself: `kaleido` (a kaleidoscope of `segments` mirrored
  wedges, its `source` aimed at the ink or the petals, turning, stepping on the hits; `mask = "ground"` keeps her clean on top,
  `"figure"` makes a mandala of her alone), `tunnel` (a Droste tunnel falling inward a level a bar, `frame` lines in the palette
  so the nesting reads) and `melt` (`breathe`, `ripple`, `swirl`, `drip`, `noise`, swelling on a `lane`); with a footage sticker
  under them give them its `edge` and `shadow`. The paper world ([Activity book and labels](design.md#activity-book-and-labels)):
  `activity` lays the frame on a workbook page (a crayon sketch of her drawn on it) and does its exercises on the hits, a dotted outline
  traced (her own silhouette with `shape = "figure"`), numbered dots joined until a sprite appears, a colour by number filled a region a
  hit; `label` sets tiny editorial type ("1967", "SIDE A") with a dotted leader to a scene point. A sticker sheet
  ([Sticker sheets](design.md#sticker-sheets-paper-live-stickers-hollows-doodles)): `sheet` lays her as a live die-cut sticker
  on drifting paper (`slap` it down with a squash on frame 0, `peel` it off at the end), with `hollows` (the outline a peeled
  sticker leaves: her own rim frozen at a second, paper inside; with its own `peel` the blank sticker is still there, fades
  and is taken in the span, so the hollow means something when it shows) and pen-kit `doodles` that `jump` out from under the slap, `hop`
  on the beats, are `slap`ped down one by one on their times and peel off; `fling` throws such stickers from a `{point}` (a guitar's
  body) at the lens on the hits, or `away` from it into the point (stars falling with her into a cup), and a `thread` ties a
  streaming thread to her live sticker (`at = {point = "bone('head').tail"}` keeps it on her crown as she moves in her plate).
  With a `clip` the sheet is a table ([Tables](design.md#tables-a-sheet-over-a-clip-seen-through-a-view)):
  a café table from above, the video regraded through a gradient map (`grade`: its colours at chosen `lumas`, the luma
  `soften`ed first; the coffee through a `cup` map of its own), a `cup` of coffee with paper `pads`, drift `rings` and `drops`
  from a `touch`, or holding the cut's own frame (`inside`: a shot straight down on a scene, its lens and shift keyed with the
  view, one picture with the full frame), her sticker from a plate whose camera zooms with the table's own camera (`view`, keyed
  close and wide, stepped jumps with `ease = "step"`; `track` keeps her where she lies), and a `punch` into the cup onto the shot
  cut in under the table; a hand cutout rides it with `at = {table = [x, y]}` and leaves the frame as the view closes in. A
  table's figure sits on a seat its clip's camera moves with `pin` ([Pinned
  stickers](design.md#pinned-stickers-a-figure-sitting-on-a-seat-the-clips-camera-moves)): the seat's track from `python -m
  mkmmd.pintrack` (look at its `--sheet`), contact `shadow`s cut to the seat's outline, `front` pieces of the clip laid over
  her (a luma `key` for a wire); step through her contact frame by frame at full size. A `kaleido` or
  `tunnel` after a table remaps all of it, without a mask: to keep her clean, lay her after it as a sheet of her own (`paper = false`,
  her plate `shot`, moved by `keys`). A background of real footage behind her is a `sky` ([Skies](design.md#skies-clip-after-clip-smeared-melting)):
  `[[fx.clips]]` segments on their times, each a clip through its own `view` and a grade by warmth (`grade = {lean}`: a
  city warm, another cool; a table takes it too), the clear sky's `paper`, `layers` (a city hanging upside down over
  another, birds that `keep = "dark"`) or a seat table's window (`table`); each `smear`s in from the one before ON its
  time (the old picture dragged in blocks along the new clip's motion, the new one breaking through; a table smears in
  `from` a sky by name), and `melt`s streak towers up into the sky or the whole sky past the lens. Step through every
  change at full size from the frame before it. The toy paint app
  ([Toy paint app](design.md#toy-paint-app)): `paintapp` lays a kids' paint program's chrome over the frames (its toolbar on the left,
  at the bottom of a 9:16 frame) and a cursor that picks its tools: `stamps` pop googly-eyed sprites on the hits, `bucket` floods a
  colour region round a point (pin it with `{point}` on the set when the camera moves), `undo` takes the last one back, and an `eraser`
  rubs out (`wipe`) or blows into confetti the outgoing picture across a cut. It is flash-safe by itself (at most three fills or
  erasers a second); keep `ui = false` for the paint alone. The era looks ([Era looks](design.md#era-looks-digicam-chrome-gradmap-invert-test-card-cards))
  hop the picture through pop's eras, one key each: `film` with `era = "1967"` (a Technicolor print), `camcorder` with `era = "1985"`
  (VHS), `digicam` (2005: noise, oversharpening, JPEG blocks, purple fringes, an orange date stamp) and `chrome` (now: iridescent chrome on
  her with `mask = "figure"`, sparkles, bubbles); `gradmap` regrades through a list of maps, the next on every cut (give an odd number when
  two shots ping-pong), `invert` flips the world for a window or on hits (eased, three flashes a second at most; `hue = 180` keeps Dawn's
  hues), `testcard` frames the scene in a 1967 test card's circle (`center = {point}` on her face, `open` to blow it open into the next
  shot) and `cards` tumbles TV colour-bar cards through the frame on a lane. Look at each on a light and a dark shot: a gradmap's lights go
  to its cream, so a bright set reads as its light colour. `lasers` ([Lasers](design.md#lasers)) lays a festival's laser fans over
  the picture, from frame points or scene points (`origin = {point = "obj('car_headlamp_L').loc"}`: they ride the headlamps),
  swinging over, thrown open and lit on a `lane` (three times a second at most), glowing in haze, `mask = "ground"` behind her.
  A photo `booth` with `look = "purikura"` ([Picture effects](design.md#picture-effects-pixel-sort-film-camcorder-doodles-photo-booth)) is a
  purikura strip: captures on a `lane` through a beauty filter, pastel frames on dotted paper, sticker stamps (the doodle shapes, now with
  `bow`, `flower`, `twinkle`, `bubble`, `note`, `bunny_ears`, `whiskers`, `blush`), `purikura.marks` (ears on her head) and a date or a word
  per photo written by a gel pen (`purikura.write`, never lyrics); its flash fires three times a second at most. A `key`
  ([Keys pushed too far](design.md#keys-pushed-too-far)) keys a colour out (the silhouette's flat ground, a sky, a leaf green) and pushes
  it past clean: a clip, a colour or the frame before burns through in fringes (`split`: each plane its own tolerance) and crushed
  steps (`bands`), leaving a trail (`burn`), widening while a stem sings (`follow`) or on the hits (`pump`), opening out from a point
  such as her head (`spread`, `center`), `mask = "ground"` off her.
  `shapes` ([Shapes](design.md#shapes)) redraws the picture as flat circles, squares, triangles and diamonds, a cell each in the colour
  under it (or `inks`, one per shape kind with `pick = "shape"`); on a flat `back` the ground stays empty, so a silhouette comes
  apart, and a `cell` keyed down on a `lane` assembles her out of shapes, finer on every hit.
  An `inset` ([Insets](design.md#insets-the-picture-on-a-screen-in-a-recording)) sets the picture into a screen in a recording (a
  player's pane on a recorded desktop: `clip`, `into`), and its `view` keyed from 0 to 1 drives into that screen or pulls back out of
  it; the effects after it work on the recording too (a `key` burning out of the pane over the desktop), and a `stack` or a `mitosis`
  with a `shot` (a plate), `at` and `height` lets her step out of the screen. A `clip` sticker with a `crop` lifts a piece of the
  recording over the picture (a notification), and a `push` that falls into the recording takes `border = "extend"` (its wallpaper
  goes on round it; mirrored, a second window shows). Convert a recorder's AV1 take with `ffmpeg` first.
- **Type**: text on a sign, a screen or a page (`[[text]] on = "road:gantry1_panel1"`), lyric type (one text per sung word on its onset), handwriting that appears behind a pen's
  nib, and screen type laid over the picture (`negative = true` makes it a photographic negative of the picture under its glyphs). Fonts are registered assets (`mk assets add FILE --kind font`); use static fonts. Words come from the timeline and never reach a log.
  A word or line is in place when it is sung: whatever travels (a `drop`, `slide` or `slam` word, a craft unit's slap, a pen) plays before its onset frame.
  Delete `renders/<preset>` when type changes: a stale `screen/` or `negative/` folder stays with the frames it belongs to. Words a character has to dodge are lyric type with
  `bullets = {at = "cast:rin.center", offsets = [...]}` ([Lyrics](design.md#lyrics)): each flies past where she is on the frame it lands; key her hop or duck on
  the same onsets and prove it with a `clearance` check. Persona's kinetic type is screen type with `lyrics = {style = "slam", layout = "same",
  recycle = {fade = 4, slide = 7}, backing = {color = "#E3263A", torn = 0, slant = 14}}` in a heavy italic font. A phrase sung again and again
  accumulates as rings of words turning about a point instead of each line replacing the last: one entry with `lines = [a, b]` and
  `layout = {kind = "ring", rings = "line" | "fill", center, turn}` (a ring per repeat, or each ring filled before the next; `center` may be a
  target such as her head), and `wobble = {}` makes the letters of a sung word bob on a wave with the voice ([Lyrics](design.md#lyrics): Rings, Wobble).
  A motion-design reel's huge type is `layout = {kind = "bleed", rows, yaw, pitch, swing}` with `style = "slam"`, `recycle = true` and a heavy
  grotesk: each word off the frame's edges, tilted in depth (kinetic `yaw`, `pitch`, `persp`).
  Toy alphabet blocks are `lyrics = {style = "blocks"}` (or a label: `text` with `blocks = {...}`) ([Block type](design.md#block-type)): a 3D wooden
  block per letter that drops in and lands on the onset, `blocks.hits` hop (`jump`) or `slam` them on a lane, `next = "tumble"` knocks a line off
  as the next comes down and `crash` scatters a pile; as screen type over the picture or standing in the scene (`size` is a block's edge).
  Lines that arrive whole or in two or three chunks, made by hand, are craft type ([Craft type](design.md#craft-type)): `craft = {kind}`
  (`emboss` label tape, `masking` tape written on, a die-cut `sticker`, `cutpaper` labels, `brush` letters) with `screen` and `lyrics =
  {line, units = [{words = [i, j], t}], leave = "peel"}`, drawn by `mk post` (no rebuild); `under = true` lays it under the `[[fx]]` so a
  smear drags its ink. Give 9:16 two rows (`aspect.9x16 = {craft = {rows = 2}}`) and look at every line at full size in every output.

## 7. Checks

A project's `[[check]]` entries define done; `mk check` runs them all in one Blender pass and exits 1 if one fails. Worth having in every project:

| Want | Check |
|---|---|
| Hair, ears, tails do not shake | `jitter` per family, `max = 0.5` (side locks that swing: 0.8) |
| Hair stays out of the body and props | `penetration` with the project's collider set, `max = 5` (mm) |
| Feet do not skate | `foot_slide`, `max = 1.5` (mm per frame) |
| Feet stand on the floor, not in it (a step back into a sweep's cove, a slope, a stage) | `ground` with `ground = "<set or prop>"`, `max = 10` (mm; a few millimetres is a sole pressed on the floor) |
| No broken elbows, knees, necks; no arm turned past a human one | `joint_limits`, `max = 0` (a dance motion that turns the arms' twist bones further: widen `forearm_twist` and `arm_twist` in `limits` and keep `skin_collapse`) |
| No sleeve or skin collapsing round a twisted arm or leg | `skin_collapse` (default `max` 0.15) |
| A hand really holds or touches something | `contact` between two points or a point and a track (see Hands above) |
| A worn prop (a guitar) is played in time and stays out of the body | `strum` (the pick meets the strings at every down stroke, `max = 10` mm; `detail.timing` is the strike error in ms), `string_play` (every note: its string rings from its fret, the fretting fingertip on it behind the wire, the pick or fingertip crossing it on time; `max = 0`, `detail.misses` names what each missed note lacks), `prop_body` (`max = 8` mm), and `contact` of `bone("wrist.R").tail` against `obj("<cast>_hand.R").loc` (the arm reached every goal, `max = 5`) |
| The subject is in frame in every output | `framing` per shot (`frames` of that shot), `min = 0` |
| Nothing blocks the subject | `occlusion`, `max = 0.3` |
| The camera is not inside a car body or a wall | `camera_inside`, `max = 0` |
| No near-black, stays in palette | `palette` on rendered frames |
| No shimmer | `flicker` on rendered frames, compared shot with shot |
| A hero prop is not a pile of boxes | `form`, default `max` 0.25 (see Modelling props) |
| Words that fly past miss her (or hit her on purpose) | `clearance` with the entry's words, `objects = "<entry>_l*"`, `min` a margin above 0 (or `max = 0` for a hit) |

Reading results:

- `jitter` with a `static` note: the chain barely moves and the ratio is float32 noise; judge by `jerk_p95_mm`.
- `penetration` is measured on the baked bones (what renders), with each chain's own radius. It reports the frame, bone and the shape it is in (`body:<bone>` for the model's own
  bodies). Body capsules are fatter than the mesh, so a few millimetres rarely show; look before fixing.
- `clearance` measures the rig's collision bodies, which often stop at the ankles and wrists: a word that clears the ankle capsules by less than a shoe's height
  goes through the shoe. `detail.at_frame` and `detail.body` say where to look.
- `joint_limits` does not flag sideways elbow bends: MMD rigs treat the elbow as a ball joint and professional motions put upper-arm rotation there. It flags fold-through, in-plane
  hyperextension, knees bending the wrong way or sideways, wrists past 110 degrees, necks and spines past what dances do, and forearms and upper arms turned about
  themselves past a human arm's range (`forearm_twist`, `arm_twist`: the series' `spans` say when).
- `skin_collapse` names the frame, the limb (`region`), the bone and the mesh object where a cut through the skin lost the most of its rest size: look at that
  frame from the side the sleeve faces (`mk look --view ... --target 'bone("elbow.L").tail'`).
- `framing` margins are fractions of the frame; negative means outside the safe area. Configure it per shot: a page close-up is not supposed to contain the head.

## 8. Look sheets

Numbers miss things. `mk look --frames 200,400,600 --sheet` shows the cut in every output; `mk look --view front,left,back --target 'bone("wrist.R").head' --dist 0.4 --frames 300 --sheet`
inspects a detail from all sides; `--ab OTHER.blend` pairs two scenes; `--ref` puts reference photos beside the render. Look at the images before claiming anything about how a shot
looks: silhouettes and proportions first, surface detail second. Frames inside a transition, an insert or a glitch and frames with screen type are previewed composited; `--no-transitions` and `--no-styles` leave effects and looks out.

### Measuring a render and showing a plan

When the person asks for a change they judge by eye (a collar's depth, how a sleeve meets the body), measure the render
and mark the plan on it before building, then measure the result against the same marks:

```sh
mk look build/pose.blend --frames 61 --view front --target '(0, 0, 1.07)' --dist 1.1 --lens 60 --size 1100 --map --out look/plan
mk mark look/plan/front_sq_00061.jpg --grid                     # read positions in the image's pixels (also photos, drawings)
mk mark look/plan/front_sq_00061.jpg --grid 25 --crop 250,175,600,675 --zoom 2      # a close look, same coordinates
mk measure look/plan/front_sq_00061.map.npz                     # the materials seen, their pixels and boxes
mk measure look/plan/front_sq_00061.map.npz --edges セーターリブ --x 300:900:20 --y 150:500   # a band's top and bottom, cm
mk measure look/plan/front_sq_00061.map.npz --gaps --y 430:600:10 --x 150:480                 # the space under an arm
mk mark look/plan/front_sq_00061.jpg --marks plan.toml --ref photo.png --out plan.png         # the plan beside the photo
```

Draw the plan from the measurements (`mk measure` gives centimetres as seen, `seen_cm`, and between surface points, `cm`, in
the world and at rest), state sizes in centimetres on it, and show it in a review; after the build, render the same view with
`--map` and draw the same marks file on the new image (`mk mark NEW.jpg --marks plan.toml`), then measure the result: the
plan and the result should agree to a few pixels. A point in a marks file can be a world point (`[x, y, z]`, from `mk q` or
`mk measure --at`); `mk measure --at X,Y` gives a pixel's point on the model at rest, which is where a builder's numbers live.

### Asking questions of a scene

```sh
mk q scene.blend --list bones                         # names: Blender, PMX and semantic
mk q 'bone("head").head' --frames 181:280             # any expression, frame by frame
mk q 'dist(bone("index_tip.R").head, obj("Pen").loc)' --frames all --summary
mk serve scene.blend &                                # big scene: keep it loaded, q / check / grip answer in < 1 s
```

Bones can be named by their semantic name (`head`, `wrist.R`, `index2.L`, see `mkmmd/core/bonemap.py`), their Blender name (`手首.R`) or their PMX name (`右手首`).
The expression language is listed under [Expressions](design.md#expressions).

## 9. Render and post

`mk render --preset draft --jobs 2` renders every output (frames are claimed, so a stopped render resumes and `--jobs` share the work); `--output 9x16 --frames t=0:5` renders a
part. A project whose shots are vector looks renders several times faster with `[render] engine = "mk"` (or `--engine mk` for one run): mk draws those shots on the GPU
from the bake (`mk build --bake` after saving the .blend by hand), as Blender would to a few pixels in a thousand, and Blender the rest; the report's `shots` says which
engine drew each and why. `mk post --preset draft` grades and encodes `out/<name>_<output>_draft.mp4` with the song; `mk render ... --post` runs it beside the render, each video finished as its frames come (its minutes inside the render's, not after them: a 22 s show's final took six to post on its own). Review the draft (`mk play` opens it in the player of the `player` setting, with the shots as chapters), fix, render `--preset preview` for motion blur, then `final`. A missing
layer or frame is an error unless `--allow-gaps`; after changing text, delete the preset's frames. `mk post --to DIR` copies the videos. A loop for a feed or a
teaser is an `[[excerpt]] {name, from, to, outputs, loop}` ([Excerpts](design.md#excerpts)): `mk post` writes that span beside each output as it encodes it. The grade is `[post]`
([Rendering and post](design.md#rendering-and-post)): check `min_luma` in the report if you care about black. A clip of two scenes that want
different pictures (a room at night, then a bright collage) gives a part of it its own: `[[look.span]] {from, to, exposure, contrast, gamma}`
for what the renderer sees it through ([Lights and the look](design.md#lights-and-the-look)) and `[[post.span]] {from, to, fade, ...}` for its
grade; both switch on the frame a cut written with the same seconds does, and a project with post spans is graded on the CPU. A video watched on phones
wants `[audio.master]` when its beat lives in the bass (808s, kicks: a phone's speakers play little below 250 Hz): it gives the
booms of a hits lane overtones a phone can play, and the report's `boom_phone_db` says how far they stand out there, before and after.

## 10. Credits

`mk assets credits --out CREDITS.md` builds the credits from the cast, the footage and split- and iris-insert videos, the photos of `photo_world` sets, the sprites of mandalas and `[credits] assets`; put what only the project knows (the song, a print made for it, the palette, tools) in
`[credits] lines`. It exits 1 while any asset is unreviewed or has no author or credit line.

To publish the result, `mk export web` writes `out/web/`: the scene and its timeline as a static page, with
`CREDITS.md` and each model's readme. It exits 1 under the same rule, and also when an asset's restrictions forbid
redistribution.

## Modelling props

A prop assembled from boxes reads as a toy even when every dimension is right. For anything built in code (a library prop, a vehicle, furniture, a set element) follow this loop, and do
not call a prop done before step 6:

1. **References first.** `mk ref photos 'Dodge 600 convertible' --set dodge600 --n 12` fetches licensed photographs from Wikimedia Commons into `refs/dodge600/` (outside a project:
   `~/.cache/mk/ref/photos/dodge600/`) with `SOURCES.json` (licence, author, page) and `contact_sheet.jpg`. LOOK at the sheet and pick the angles that matter (front 3/4, side, rear 3/4,
   top, interior, the detail you will model). Ask again for what is missing (`'Dodge 600 dashboard'`, `... wheel`): the same set takes more. The photos are visual references only: never
   textures, never in a render, never traced.
2. **Real dimensions.** Write down the true size and the proportions that matter (length, width, height, wheelbase, seat and sill heights; a mug's height against its width) before the first
   vertex. A data sheet or the Wikipedia page beats a guess; the photo decides what the data sheet cannot say (the slope of the hood, the radius of a fender).
3. **Build forms from profiles, lofts and subdivision with creases.** Draw a section, pull it along a path or between stations, let the Subdivision Surface round it, and put a crease only
   where the real part has a line. The toolkit is `mkmmd/core/shell.py` (`loft`, `cage`, `sweep`, `lathe`, `rounded_panel`, ...); the rules and recipes are in [modelling.md](modelling.md).
4. **Bevel every visible hard edge; never stack cuboids for visible forms.** Real surfaces have radii, and the highlights live on them. A box, a slab or a plate with a 1 mm bevel is still a
   box: round it, crown it, or build it as a loft. Parts that really are flat graphic layers (a label, a display) are tagged `mk_form_exempt` (`shell.exempt(obj)`).
5. **Measure with `form`.** A `[[check]]` per hero prop, run after every build:

   ```toml
   [[check]]
   name = "car form"
   metric = "form"
   args = { prop = "car", exclude = ["car_underbody"] }     # exclude: objects nobody sees; hidden colliders never count
   max = 0.25                                               # the default for hero props
   ```

   `value` is the boxiness (0 smooth .. 1 boxes). Read `detail.worst_parts` first: each entry names the object, the part inside it, its area, its share of the prop, `why` (`a box: ...` or
   `flat panels with sharp edges`), its size and where it is in the prop's frame; `sharp_panels` lists the biggest flat panels with unbevelled rims. Fix the top entry, rebuild, check again: a
   prop usually needs three or four rounds. Furniture and gadgets that are boxes by nature (a boombox, a nightstand) may be judged at `max` 0.5-0.7 once you have looked at them; walls, gantries and
   barriers are `exempt = ["wall*"]` (measured and listed, not scored) or `exclude`d. Do not raise `max` to make a hero prop pass. `mk build` runs the same maths on every library prop it places
   and logs `WARNING prop 'car': form 0.51 > 0.25 (car_body part 25: a box: ...)` when one is over its limit. A prop's card says what is fair for it: `form_max` (its limit; a boombox says 0.7; 1
   switches the guard off) and `form_exempt` (object names or patterns that are architecture or graphic layers), from the builder's card or the project's `[[prop]] card_extra`.
6. **Compare side by side and look.** `mk look build/scene.blend --frames 1 --view 35:12,90:8 --target 'obj("car").matrix @ Vector((0, 0, 0.6))' --dist 7.5 --ref refs/dodge600/front_left.jpg,refs/dodge600/side.jpg`
   writes `ref.jpg`: photo i beside view i at the same height. Choose `yaw:elev` to match the photo's angle (the photo's horizon and perspective tell you), then compare silhouette and
   proportion first (wheelbase, hood length, roofline, stance), surface detail second. Open the image. A passing `form` number means no boxes, not the right car.

## Debugging

- `MK_KEEP_JOBS=1 mk ...` keeps `~/.cache/mk/jobs/<id>/` (job, result, Blender log) even on success.
- `mk check --keep-sample` keeps the sampled `.npz` and its JSON reply in `~/.cache/mk/samples/` for your own numpy.
- A failing Blender job prints `error`, `log` (path) and `trace`; read the log before retrying.
- `mk build --until STAGE` and `--skip STAGE` bisect a build; `mk doctor` checks the environment (`--fix` restores mmd_tools' wheels).

## Pitfalls

Models and rigs:

- mmd_tools gives bones without a tail target a 0.08 m tail; never use bone tails as physical lengths.
- mmd_tools adds `_dummy_` / `_shadow_` helper bones and non-bone vertex groups (`mmd_edge_scale`, `mmd_vertex_order`); rig.json marks the former and quirks list the latter.
- Many PMX hair joints are fully locked (zero angular range): Bullet swings such chains as rigid sticks. That is why a cast member's chains are solved by the sim stage instead (`physics = "mk"`).
- A Blender started with `--factory-startup` can delete mmd_tools' bundled wheels and break PMX import everywhere; run `mk doctor --fix`.
- MMD models hide their drawn expressions (star and heart eyes, a blush, tears, `><`) a hair behind the skin and bring them out with a morph: a camera that starts at 2 cm shows them all at once through the face from a couple of metres away. The clip stage starts every shot camera at half the depth of the nearest thing it sees on each frame, and `mk look`'s orbit views do the same ([Clip ranges](design.md#clip-ranges)); `mk build --skip clip` brings the bug back.

The build:

- Stage order matters: a target that names a cast member (`cast:rin`) or the camera does not exist while sets and props are built, and the camera does not exist while poses and performance are
  built (`look = "camera"` in `[perform.<cast>]` fails). Aim at a bone or a point instead.
- An empty `[pose.rin]` table does nothing, and a standing pose needs explicit feet targets (`{cast = "rin", point = [x, y, 0]}`): `feet = "floor"` or `"seat"` without a `sit` fails.
- `mk look` removes the timeline markers for debug views (they switch cameras every frame) and keeps them for the cut; its orbit camera clips at 100 m and cuts far skylines away.
- A silhouette, vector or reflection shot is a finished flat frame on disk (composed inside the render job): `--no-styles` shows the lit scene instead, in `mk look` and `mk render`.
- After editing a `[[transition]]` or `[[insert]]` run `mk build` again (the cameras of the shots that lend frames are keyed at build; `mk render` tells you when they are not).
- A scripted driver is an animation that needs no Python only while its expression is a Blender *simple expression*, and Blender stores at most 255 characters of it (a longer one is silently cut there and then fails).
  `%`, `**`, `sign` and `copysign` are not simple (`fmod`, `pow`, `floor`, `clamp`, `smoothstep`, `lerp`, `min`, `max`, `abs`, `sin`, `cos`, `radians`, `sqrt`, `if ... else`, `and`, `or` are); check
  `driver.is_simple_expression` and `is_valid`, and split a long formula over driven custom properties of the object (the hearts prop's `age`, `u` and `pin`).
- A prop over 600 000 triangles is not measured by the form guard; a pmx or appended `.blend` prop is never measured.
- `mk serve` keeps the scene it loaded: re-run it after the file changes on disk. Only `q`, `check` and `grip` use it.

Renders and post:

- Frames and layers are claimed with an empty file, so a stopped render resumes. After a `mk build`, `mk render` draws again every frame and layer that is older than the scene (the report's `stale`; `--keep` takes them as they are): a video is never old frames under a new sound track. An mkr render keeps the frames the build left as they were (the report's `reused`: each frame's key in `keys.json` says what it shows): after a change to one character's motion it draws her frames from the change on, not the scenes without her; a change to a material, a mesh or the cut is part of every frame. So fix a shot, build, render, and only that much is drawn. A change that does not go through the build is not seen that way: delete `renders/<preset>` then.
- When one clip holds several scenes (takes), each with its own cast, show each member only during her own scene: `[[cast]] shows = [[from, to]]`, her scene's clip seconds. Then a change to her motion draws her own scene's frames again and none of the others'; without it her hair keeps moving in the scenes she is not in, and all of them are drawn again. Out of view she only ever reached the other scenes through their lights' shadows (on a two-scene show 60 m apart, hiding her changed no pixel). Placing the scenes far apart does not do it: a character out of view and beyond the lights' reach still counts in their shadows.
- Use static fonts: Blender reads a variable font's default instance (`python -m mkmmd.fontinst` pins the axes).
- A fast move (a spin, a windmill, a kick, a step) is judged from every one of its frames at full size, in the shot's own camera, before anyone is shown it: a sheet of small tiles hides an arm across a face, a hand flipped over for three frames, a foot in the floor. A draft has no motion blur, so anything that travels more than a hand's width a frame strobes.
- `strum`'s `windmill` keeps the hand in the plane of the strings: its circle lies in front of the player, not beside her as a real one does. On a small player or a guitar worn high it crosses her chest and her face; the build warns (`head_cm` in the report) and it is better left out there.

## Example `mk.toml`

```toml
[project]
name = "cafe"
fps = 30
frame0 = 181
duration = 24.6
blend = "build/cafe.blend"

[[output]]
name = "9x16"
size = [1080, 1920]

[[cast]]
name = "rin"
armature = "Rin_arm"
asset = "my_model"

[colliders]
cafe = [{ type = "box", object = "ChairBack", rnd = 0.012 }, { type = "fingers" }, { type = "floor", z = 0.0 }]

[[check]]
name = "back hair is calm"
metric = "jitter"
args = { family = "back_hair" }
max = 0.5

[[check]]
name = "hair stays out of the body"
metric = "penetration"
args = { colliders = "cafe", radius = { scale = 0.5, max = 0.02 } }
max = 5.0
```
