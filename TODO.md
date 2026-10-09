## Todo

- [ ] Fix the body's weak spots #body #you
    The ledge where the torso meets the thighs, the boxy shoulders and the thin legs: every character inherits them. Two routes: fix the procedural body, or take a CC0 donor body the way the hands came from B-chan's glove. Your call, after seeing both side by side.
    - [ ] today's body next to the candidates, same scale
    - [ ] choose the route
    - [ ] build it, refit the outfit, run the checks and the grip
- [ ] Give the girl her own eyes? #face #you
    She has Rin's eye drawing (the cat-like wing and flick) in brown. Keep it, or give the neutral base a softer look of her own: a few numbers in her head.toml.
- [ ] Prove the playbook with a fresh agent #base
    A new agent makes a character with `mk model new` and docs/model_base.md alone; fix whatever it trips on.

## Doing

## Done

- [x] Wheel grip with the CC0 hand #hands
    The wrist bent 44° on the wheel (18° with the designed hand), past the 35° line where the build warns, and each hand took about 3.5 min to solve. Now 18° on both hands, in 142 s and 112 s.
    - [x] why: the solver only saw the rim and tilted the long-palmed, short-fingered hand 25° to wrap it; the arm came after and had to bend the wrist
    - [x] the solver is given the arm and keeps the wrist within 15° of the forearm: 44° → 18° (designed hand 18° → 16°)
    - [x] the skin searched on one vertex per 2 mm of the hand's own frame: 277 s → 142 s (left), 166 s → 112 s (right)
    - [x] tests (the allowed elbows, the solver given an arm), each shown failing on its bug; docs
    - [x] both hands on the wheel, rendered before and after (review/next_steps/33)
    - [x] commit
- [x] Fix the face's known flaws #face
    Every character shares the face code, so each fix lands in all of them.
    - [x] the skin dived into the inner eye corners (4.0 mm over the last 6 mm; now 1.9, Miy's 1.8): a recess on the nose side of the socket
    - [x] ω mouth: the blush patches (18 vertices up to 1.2 mm in front of the skin) now follow the skin through every mouth shape
    - [x] the crease in the middle of the upper lip in 口角上げ / 口角下げ: the midline was treated as a mouth corner (np.sign(0))
- [x] Register B-chan's .blend in mk assets as `bchan` #tools
    CC0 1.0 (OpenGameArt's licence field and notice, the author's itch.io page), credit appreciated not required, tagged with its recipe (make_hand.py).
- [x] Bring the lab into the repo: `mk model lab` and `mk model trace` #tools
    Sheets in seconds, no Blender: specs and PMX files side by side at one scale, posed with their own weights (fingers, arms down, sit, any morph), numbers cut across the skin. A red line on a screenshot of a sheet reads back in mm.
    - [x] views and pose sheets with the part's own weights
    - [x] side-by-side sheets at the same scale, with a numbers summary
    - [x] red lines drawn on a render turned into millimetres
    - [x] a faster renderer for dense meshes (16 hand cells in 1.3 s)
    - [x] keep the study images somewhere that lasts: mk-tests/rin_model/review/hands_and_bases (with the lab scripts)
- [x] Face: drawn eye lines, face planes, folded mouth #face
- [x] Hands: the designed hand, then B-chan's CC0 glove #hands
- [x] Model bases: the girl and the rin example, `mk model new` #base
- [x] Commit the work: three commits on main
- [x] Push the three commits (and this board) to GitHub
