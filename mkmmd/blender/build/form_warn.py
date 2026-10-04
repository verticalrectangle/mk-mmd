"""form_warn: the props stage's guard against blocky props (docs/AGENTS.md: Modelling props).

After a library prop is built and placed, what a render shows of it goes through mkmmd.core.form (the maths of the `form`
check) and a prop over its limit logs one line, so that a box-built prop is noticed in the build log and not by a person:

    WARNING prop 'car': form 0.51 > 0.25 (car_body part 25: a box: flat faces on three axes make up 84% of its surface,
    31% of the prop)

The card says what is fair for this prop (a builder's own card, or the project's `card_extra`):
  form_max     the limit (default 0.25, for hero props); a thing that is a box by nature (a boombox) says 0.7; 1 or more
               switches the guard off for that prop
  form_exempt  object names or fnmatch patterns that are architecture or graphic layers (as the check's `exempt`); objects
               tagged `mk_form_exempt` are exempt too
Props from PMX models and appended .blend files are not measured (someone else's modelling), nor are sets. Results are
cached in <project>/.mk/cache/form/ by a hash of the evaluated geometry (in the prop's own frame, so placement does not
matter) and of core/form.py: an unchanged prop costs a mesh evaluation, no analysis. The numbers are the check's: `mk check
SCENE form --args '{"prop": "car"}'` lists the parts to fix."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from ...core import form as FM

MAX_TRIANGLES = 600_000         # a bigger prop is no prop (a scene): skipped, said in the log


def _source_key():
    return hashlib.sha256(Path(FM.__file__).read_bytes()).hexdigest()[:12]


def _cache_path(ctx, key):
    return os.path.join(ctx.cache, "form", key + ".json")


def guard(ctx, prop, ref):
    """Measure `prop` (a build `Prop`: name, root, card) and log a WARNING when its boxiness is over its limit. Returns
    {"form", "limit", "worst"} (None when the prop is not measured). Advisory: a problem while measuring is logged as
    `form: prop ... not measured (...)`, it never fails the build."""
    card = prop.card
    limit = FM.card_limit(card)
    if limit >= 1.0 or not FM.is_library(ref, card):
        return None
    try:
        res = _measure(ctx, prop, card)
    except Exception as e:                                   # noqa: BLE001 - advisory tooling inside a build
        ctx.log("form", f"prop {prop.name!r}: not measured ({type(e).__name__}: {e})")
        return None
    if res is None:
        return None
    res["limit"] = limit
    if res["form"] > limit:
        ctx.log("WARNING", f"prop {prop.name!r}: form {res['form']:.2f} > {limit:g} ({res['worst']})")
    return res


def _measure(ctx, prop, card):
    """{"form", "worst"} of a prop (from the cache when its geometry is unchanged), None when there is nothing to judge."""
    from .. import ops_sample as SMP                         # needs bpy: imported when the guard runs, not at module load
    out = {}
    meta = SMP._sample_meshes([{"prop": prop.name, "objects": [], "exclude": [], "frame": None}], out)[0]
    V, T, owner = out["mesh_v_0"], out["mesh_t_0"], out["mesh_o_0"]
    if not len(T):
        return None
    if len(T) > MAX_TRIANGLES:
        ctx.log("form", f"prop {prop.name!r}: not measured, {len(T)} triangles")
        return None
    V = FM.to_frame(V, np.array(meta["roots"][prop.name]))
    names = meta["objects"]
    exempt = FM.card_exempt(names, meta["exempt"], card)
    if len(exempt) == len(names):                            # nothing but architecture or graphic layers
        return None
    key = FM.fingerprint(V, T, owner, names, _source_key(), sorted(exempt))
    path = _cache_path(ctx, key)
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            res = json.load(fh)
    else:
        r = FM.analyse(V, T, owner, names, exempt=exempt)
        res = {"form": round(r["score"], 4), "worst": FM.headline(r)}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(res, fh)
    return res
