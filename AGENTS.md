# Notes for agents working on or with mk-mmd

- Read [docs/design.md](docs/design.md) first: layers, the Blender bridge, conventions and the data contracts.
- `mkmmd/core` and `mkmmd/blender` run inside Blender 4.2 (Python 3.11, numpy 1.24): no 3.12-only syntax, no
  numpy-2-only API, no scipy there. scipy and numba belong in `mkmmd/solvers`.
- No model, prop or project names in library code; use semantic bone names and prop cards.
- Never print or quote lyric text, in code, logs, commit messages or briefs. Refer to words by `(line, word)`.
- Never commit models, motions, audio, lyrics, reference clips or renders (`.gitignore` enforces most of it).
- Run `uv run --with pytest pytest -q` (or `pytest -q` in the tool env) before pushing.
