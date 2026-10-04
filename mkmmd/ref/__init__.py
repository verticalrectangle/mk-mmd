"""Reference measurement: how real people move in reference clips, turned into performance parameters.

Pipeline (CLI: `mk ref`): search Pexels -> add clips to a reference set -> track (download + MediaPipe) -> measure
(per-clip and pooled statistics, recommended `[perform.<cast>]` values) -> sheet (landmarks drawn on frames) -> clean.

Modules
  store      cache layout, reference sets (clips.json), cleaning
  pexels     Pexels API client: search, clip metadata, rendition choice, capped download (key from the keyring)
  models     MediaPipe model bundles (downloaded into the cache)
  track      per-clip tracking pass -> <cache>/<set>/track/<id>.npz (face / hands / pose / camera drift)
  dsp        numerics on gap-bearing series (NaN = not tracked): smoothing, runs, spectra, event helpers
  face       head pose, blinks, eyelid rest level, gaze shifts, mouth activity, head oscillations
  body       shoulders, breathing proxy, hands
  measure    per-clip analysis, pooling over clips
  recommend  pooled statistics -> perform parameters with a `why` per field
  sheet      contact sheet

Only `store`, `pexels`, `models` touch the network; only `track` and `sheet` import MediaPipe / decode video. The
analysis modules are pure numpy on the tracked arrays, so they run (and are tested) without MediaPipe.

Downloads and tracking data are large and copyrighted: they live under the cache (`~/.cache/mk/ref/<project or
default>/<set>/`), never in a repository, and `mk ref clean` deletes them."""


class RefError(Exception):
    """A runtime failure: network, missing API key, tracking."""


class RefUsage(Exception):
    """Arguments or inputs the caller can fix (unknown set, nothing tracked yet, bad id)."""
