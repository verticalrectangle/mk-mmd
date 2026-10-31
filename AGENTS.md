# Notes for agents working on or with mk-mmd

- Read [docs/design.md](docs/design.md) first: layers, the Blender bridge, conventions, the data contracts and every `mk.toml` key.
  [docs/AGENTS.md](docs/AGENTS.md) is the playbook for making a video, [docs/modelling.md](docs/modelling.md) the rules for modelling props,
  [docs/model_base.md](docs/model_base.md) how to make a character from a model base (`mk model new`).
- `mkmmd/core` and `mkmmd/blender` run inside Blender 4.2 (Python 3.11, numpy 1.24): no 3.12-only syntax, no
  numpy-2-only API, no scipy there. scipy and numba belong in `mkmmd/solvers`.
- No model, prop or project names in library code; use semantic bone names and prop cards. Characters live as spec files in
  `mkmmd/model/bases/` (the girl base and the rin example) and in projects.
- Never print or quote lyric text, in code, logs, commit messages or briefs. Refer to words by `(line, word)`.
- Never commit models, motions, audio, lyrics, reference clips or renders (`.gitignore` enforces most of it). The committed
  meshes are the girl base's CC0 hand and body, `mkmmd/model/bases/girl/hand.npz` and `body.npz` (their `make_hand.py` and
  `make_body.py` rebuild them).
- One GPU job at a time. mk's own GPU work (mk render, mk post, Blender render jobs, mkr, mkr-post) takes the GPU lock;
  run anything else that touches the GPU (a Blender script of your own, GPU tests, a scratch build of mkr) as
  `mk gpu run -- CMD`, and check `mk gpu status` before a big job. Several jobs at once fill the card and spill into RAM.
- Keep test outputs on disk (a work folder of your own), not in `/tmp` or `/dev/shm`: both are RAM here. Delete your big
  scratch when you are done with it.
- Run `uv run --with pytest pytest -q` (or `pytest -q` in the tool env) before pushing.
