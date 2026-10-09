"""Part builders: the registry, the build context and the runner.

A part builder is a plain function `build(ctx) -> Part` registered with `@builder("hair")` in
`mkmmd/model/parts/<name>.py` (numpy and PIL only, no bpy). `run(spec, only=None)` calls the builders of
`spec["model"]["parts"]` in spec order, checks every Part (`part.check` plus cross-part references) and returns them.

    from mkmmd.model.build import builder

    @builder("hair", needs=("body", "head"))          # `--only hair` builds these first; default needs = ("body",)
    def build(ctx):
        cfg = ctx.cfg                                  # spec["hair"] (a dict, {} when absent)
        head = ctx.parts["head"].info                  # what earlier parts published
        top = ctx.land["head_tip"]                     # landmarks published by the body part
        png = ctx.save_png("hair_diffuse", rgba)       # -> "hair_hair_diffuse.png" (prefixed by the part name)
        return Part("hair", meshes=[...], ...)

`BuildCtx` (what builders rely on):
  spec        the merged spec (dict, with the [proportions] changes made, `proportions.apply`: leg_extra; watched: after
              the build the author's keys no builder read are logged as warnings, `[spec] WARNING hair.back.x: never read
              by any part ...`, one line per top-level table); `cfg` is the table named like the part being built,
              `section(name)` any table. Merge defaults under a table with `spec.merge(DEFAULTS, cfg)` and read keys by
              name (not by listing the table) so the reads stay noted.
  tex_dir     Path where PNGs are written (create it yourself never: `save_png` does)
  save_png(name, rgba) -> file name (str, relative to tex_dir; the part name is prefixed unless present)
  parts       dict name -> Part built so far, in spec order (only the dependencies when built with `--only`)
  land        semantic name -> np.array(3): landmarks the body published in `part.info["landmarks"]`
  rng         np.random.Generator seeded from (model seed, part name): identical with or without `--only`
  log(*a)     progress and warnings (collected in `ctx.logs`; `WARNING` prefix counts as a warning)
  need(name)  the Part `name` or a BuildError explaining which `needs` to declare
  find_body(bone), find_bone(name), bones()  look things up across the parts built so far
"""
import importlib
import time
import zlib
from pathlib import Path

import numpy as np

from . import part as P
from . import proportions as PR
from . import spec as SP

REGISTRY = {}
DEFAULT_NEEDS = ("body",)


class BuildError(RuntimeError):
    pass


class Builder:
    def __init__(self, name, fn, needs):
        self.name, self.fn, self.needs = name, fn, tuple(needs)


def builder(name, needs=None):
    """Register `build(ctx) -> Part` for the part `name`. `needs`: parts that must exist before this one when only a
    subset is built (default `("body",)`, none for the body itself)."""
    def deco(fn):
        n = needs if needs is not None else (() if name == "body" else DEFAULT_NEEDS)
        REGISTRY[name] = Builder(name, fn, n)
        return fn
    return deco


def get_builder(name, spec=None):
    """The registered builder for a part, importing `mkmmd.model.parts.<name>` (or the module named in
    `[model.builders]`) on first use."""
    if name not in REGISTRY:
        from . import parts as PARTS
        mod = PARTS.module_for(name, (SP.model_cfg(spec)["builders"] if spec is not None else {}))
        try:
            importlib.import_module(mod)
        except ModuleNotFoundError as e:
            if e.name != mod:
                raise
            raise BuildError(f"no builder for part {name!r}: module {mod} does not exist "
                             f"(known: {', '.join(PARTS.KNOWN)})") from None
    if name not in REGISTRY:
        raise BuildError(f"module for part {name!r} imported but it did not register @builder({name!r})")
    return REGISTRY[name]


class BuildCtx:
    def __init__(self, spec, tex_dir, seed=1, log=None):
        self.spec = SP.watch(spec)
        PR.apply(self.spec)                  # [proportions] leg_extra, before any part reads a height
        self.tex_dir = Path(tex_dir)
        self.seed = int(seed)
        self.parts = {}
        self.land = {}
        self.logs = []
        self.part = ""                       # name of the part being built
        self.timings = {}
        self.textures = []                   # file names written, in order
        self.cached = []                     # parts reused from the part cache (`run(cache=...)`)
        self._log = log

    # ---- spec access
    @property
    def cfg(self):
        """The spec table named like the part being built ({} when absent)."""
        return self.section(self.part)

    def section(self, name):
        v = self.spec.get(name, {})
        return v if isinstance(v, dict) else {}

    def get(self, dotted, default=None):
        """Nested spec lookup with a dotted key: `ctx.get("proportions.height", 1.6)`."""
        return SP.dig(self.spec, dotted, default)

    @property
    def rng(self):
        """Generator seeded by (model seed, part name): the same stream whatever else is built."""
        return self.rng_for(self.part)

    def rng_for(self, key):
        return np.random.default_rng(np.random.SeedSequence([self.seed, zlib.crc32(str(key).encode("utf-8"))]))

    # ---- logging
    def log(self, *a):
        msg = " ".join(str(x) for x in a)
        self.logs.append(f"[{self.part}] {msg}" if self.part else msg)
        if self._log:
            self._log(self.logs[-1])

    @property
    def warnings(self):
        return [m for m in self.logs if "WARNING" in m]

    # ---- textures
    def save_png(self, name, rgba):
        """Write an image into `tex_dir` and return its file name. `rgba`: (h, w, 3|4) uint8 or float 0..1 (straight
        alpha), a PIL image, or a path-less array from `mkmmd.model.tex`. The part name is prefixed to `name` unless it
        already starts with it; `.png` is added. Names stay ASCII."""
        from PIL import Image
        stem = Path(str(name)).stem if str(name).lower().endswith(".png") else str(name)
        if self.part and not stem.startswith(self.part + "_"):
            stem = f"{self.part}_{stem}"
        if not stem.isascii() or any(c in stem for c in '/\\:*?"<>| '):
            raise BuildError(f"texture name {stem!r}: use plain ASCII without spaces or slashes")
        if isinstance(rgba, Image.Image):
            img = rgba.convert("RGBA")
        else:
            a = np.asarray(rgba)
            if a.ndim == 2:
                a = a[..., None]
            if a.dtype != np.uint8:
                a = np.clip(np.rint(np.clip(a.astype(np.float64), 0.0, 1.0) * 255.0), 0, 255).astype(np.uint8)
            if a.shape[-1] == 1:
                a = np.repeat(a, 3, axis=-1)
            if a.shape[-1] == 3:
                a = np.concatenate([a, np.full(a.shape[:2] + (1,), 255, np.uint8)], axis=-1)
            if a.ndim != 3 or a.shape[-1] != 4:
                raise BuildError(f"texture {stem}: expected (h, w, 3|4) pixels, got shape {a.shape}")
            img = Image.fromarray(np.ascontiguousarray(a), "RGBA")
        self.tex_dir.mkdir(parents=True, exist_ok=True)
        fname = stem + ".png"
        img.save(self.tex_dir / fname, format="PNG", compress_level=6)
        if fname not in self.textures:
            self.textures.append(fname)
        return fname

    # ---- lookups across the parts built so far
    def need(self, name):
        if name not in self.parts:
            raise BuildError(f"part {self.part!r} needs part {name!r}, which was not built: declare "
                             f"@builder({self.part!r}, needs=(..., {name!r})) or add it to [model.needs] / --only")
        return self.parts[name]

    def bones(self):
        """name -> Bone over every part built so far."""
        return {b.name: b for p in self.parts.values() for b in p.bones}

    def find_bone(self, name):
        for p in self.parts.values():
            for b in p.bones:
                if b.name == name:
                    return b
        return None

    def find_body(self, bone, static_only=True):
        """The first rigid body attached to `bone` in any part built so far (static ones by default: what a chain
        joint partner usually is), else None."""
        for p in self.parts.values():
            for rb in p.bodies:
                if rb.bone == bone and (not static_only or rb.mode == "static"):
                    return rb
        return None

    def land_point(self, name):
        if name not in self.land:
            raise BuildError(f"landmark {name!r} was not published by the body part "
                             f"(have: {', '.join(sorted(self.land)[:12])}...)")
        return self.land[name]


# ---- cross-part checks

def check_refs(parts, strict=True):
    """Cross-part consistency: unique names, parents, weights, bodies, joints, materials, textures. Returns warnings
    (list of str); raises BuildError for hard errors when `strict`."""
    errors, warns = [], []
    bones, mats, bodies, morph_decl = {}, {}, {}, {}
    for p in parts:
        for b in p.bones:
            if b.name in bones:
                errors.append(f"bone {b.name!r} defined by both {bones[b.name]!r} and {p.name!r}")
            bones[b.name] = p.name
        for m in p.materials:
            if m.name in mats:
                errors.append(f"material {m.name!r} defined by both {mats[m.name]!r} and {p.name!r}")
            mats[m.name] = p.name
        for rb in p.bodies:
            if rb.name in bodies:
                errors.append(f"rigid body {rb.name!r} defined by both {bodies[rb.name]!r} and {p.name!r}")
            bodies[rb.name] = p.name
        for mo in p.morphs:
            morph_decl.setdefault(mo.name, mo)
    seen_joint = set()
    for p in parts:
        for b in p.bones:
            if b.parent and b.parent not in bones:
                errors.append(f"{p.name}: bone {b.name!r} has unknown parent {b.parent!r}")
            if not b.parent and b.name != "全ての親":
                warns.append(f"{p.name}: bone {b.name!r} has no parent")
            if b.tail_bone and b.tail_bone not in bones:
                errors.append(f"{p.name}: bone {b.name!r} tail_bone {b.tail_bone!r} is not a bone")
            if b.grant and b.grant.get("parent") not in bones:
                errors.append(f"{p.name}: bone {b.name!r} grant parent {b.grant.get('parent')!r} is not a bone")
            if b.ik:
                for ref in [b.ik.get("target")] + [c.get("bone") for c in b.ik.get("chain", [])]:
                    if ref not in bones:
                        errors.append(f"{p.name}: IK of {b.name!r} refers to unknown bone {ref!r}")
        for m in p.meshes:
            for bn in m.weights:
                if bn not in bones:
                    errors.append(f"{p.name}/{m.name}: weights for unknown bone {bn!r}")
            for mn in m.mats:
                if mn not in mats:
                    errors.append(f"{p.name}/{m.name}: unknown material {mn!r}")
            for k in m.morphs:
                if k not in morph_decl:
                    warns.append(f"{p.name}/{m.name}: morph {k!r} is not declared in any part's `morphs` "
                                 f"(it will go to the 'other' panel)")
        for rb in p.bodies:
            if rb.bone and rb.bone not in bones:
                errors.append(f"{p.name}: rigid body {rb.name!r} is attached to unknown bone {rb.bone!r}")
        for j in p.joints:
            for ref in (j.a, j.b):
                if ref not in bodies:
                    errors.append(f"{p.name}: joint {j.name!r} refers to unknown rigid body {ref!r}")
            if j.name in seen_joint:
                errors.append(f"{p.name}: duplicate joint name {j.name!r}")
            seen_joint.add(j.name)
    if errors and strict:
        raise BuildError("model check failed:\n  " + "\n  ".join(errors[:30]) +
                         (f"\n  ... and {len(errors) - 30} more" if len(errors) > 30 else ""))
    return errors + warns if not strict else warns


def check_textures(parts, tex_dir):
    """Names of textures referenced by materials but missing from `tex_dir`."""
    missing = []
    for p in parts:
        for m in p.materials:
            for f in (m.texture, m.toon, m.sphere):
                if f and not (Path(tex_dir) / f).exists():
                    missing.append(f"{p.name}/{m.name}: {f}")
    return missing


def lint(parts):
    """Quality warnings (never errors): unweighted vertices, degenerate faces, missing UVs, bad normals."""
    out = []
    for p in parts:
        for m in p.meshes:
            v = np.asarray(m.verts, float)
            n = len(v)
            if n == 0:
                out.append(f"{p.name}/{m.name}: empty mesh")
                continue
            if m.weights:
                tot = np.zeros(n)
                for w in m.weights.values():
                    tot += np.asarray(w, float)
                zero = int((tot <= 1e-6).sum())
                if zero:
                    out.append(f"{p.name}/{m.name}: {zero} of {n} vertices have no bone weight")
            else:
                out.append(f"{p.name}/{m.name}: no weights (the whole mesh will follow the fallback bone)")
            if m.uv is None:
                out.append(f"{p.name}/{m.name}: no UV")
            used = np.zeros(n, bool)
            for f in m.faces:
                used[list(f)] = True
            if not used.all():
                out.append(f"{p.name}/{m.name}: {int((~used).sum())} vertices are not used by any face")
            if m.normals is not None and np.asarray(m.normals).shape != (n, 3):
                out.append(f"{p.name}/{m.name}: normals shape {np.asarray(m.normals).shape} != ({n}, 3)")
    return out


# ---- planning and running

def plan(spec, only=None):
    """Part names to build, in build order. With `only`, those parts plus their `needs` (transitively, only parts
    that are in the spec's `parts` list), in spec order."""
    cfg = SP.model_cfg(spec)
    names = cfg["parts"]
    if only is None:
        return list(names)
    only = [only] if isinstance(only, str) else list(only)
    unknown = [o for o in only if o not in names]
    if unknown:
        raise BuildError(f"--only {unknown}: not in [model] parts {names}")
    want, stack = set(), list(only)
    while stack:
        n = stack.pop()
        if n in want:
            continue
        want.add(n)
        needs = cfg["needs"].get(n)
        if needs is None:
            needs = get_builder(n, spec).needs
        for d in needs:
            if d in names and d not in want:
                stack.append(d)
    return [n for n in names if n in want]


def run(spec, only=None, tex_dir=None, log=None, ctx=None, cache=None):
    """Build the parts of `spec` (a `Spec` or dict) and return them as a list, in spec order. `only`: a part name
    or list (their `needs` are built first). Textures go to `tex_dir` (default `<model.out>/tex`). `cache`
    (`partcache.PartCache`): reuse the parts whose inputs did not change (their names in `ctx.cached`)."""
    if not isinstance(spec, SP.Spec):
        spec = SP.from_dict(spec)
    cfg = SP.model_cfg(spec)
    ctx = ctx or BuildCtx(spec, tex_dir or (cfg["out"] / "tex"), seed=cfg["seed"], log=log)
    out, keys = [], []                         # keys: the parts' cache keys so far, each chained on the ones before
    names = plan(spec, only)
    for name in names:
        entry = get_builder(name, spec)
        ctx.part = name
        t0 = time.time()
        hit = cache.find(ctx.spec, name, ctx.seed, keys) if cache is not None else None
        if hit is not None:
            key, got = hit
            part = got["part"]
            _reuse(ctx, got)
            ctx.cached.append(name)
        else:
            n_logs, n_tex, snap = len(ctx.logs), len(ctx.textures), ctx.spec._reads.snapshot()
            try:
                part = entry.fn(ctx)
            except BuildError:
                raise
            except Exception as e:
                raise BuildError(f"part {name!r} failed: {type(e).__name__}: {e}") from e
            if not isinstance(part, P.Part):
                raise BuildError(f"builder {name!r} returned {type(part).__name__}, expected Part")
            if part.name != name:
                raise BuildError(f"builder {name!r} returned a Part named {part.name!r}")
            try:
                P.check(part)
            except ValueError as e:
                raise BuildError(str(e)) from e
            if cache is not None:
                key = cache.store(ctx.spec, name, ctx.seed, keys, ctx.spec._reads.since(snap), part,
                                  {f: (ctx.tex_dir / f).read_bytes() for f in ctx.textures[n_tex:]}, ctx.logs[n_logs:])
        if cache is not None:
            keys.append(key)
        ctx.timings[name] = round(time.time() - t0, 3)
        ctx.parts[name] = part
        out.append(part)
        lm = part.info.get("landmarks")
        if lm:
            for k, v in lm.items():
                ctx.land[k] = np.asarray(v, float)
        check_refs(out)
        ctx.part = ""
    report_unread(ctx, None if only is None else names)
    return out


def _reuse(ctx, got):
    """Put a cached build's textures, log lines and spec reads back as if its part had just been built."""
    ctx.tex_dir.mkdir(parents=True, exist_ok=True)
    for f, data in got["textures"].items():
        (ctx.tex_dir / f).write_bytes(data)
        if f not in ctx.textures:
            ctx.textures.append(f)
    for line in got["logs"]:
        ctx.logs.append(line)
        if ctx._log:
            ctx._log(line)
    ctx.spec._reads.add(*got["reads"])


def report_unread(ctx, tables=None):
    """Warn, one line per top-level table, about the author's spec keys no builder read (`spec.unread`): a misspelt key,
    one in the wrong table or one for a feature that is off must not pass for one that works. `tables`: only these
    top-level tables (a partial build: the parts it built)."""
    groups = {}
    for path, n in SP.unread(ctx.spec, tables):
        groups.setdefault(path[0], []).append(SP.dotted(path) + (f" (table, {n} keys)" if n else ""))
    ctx.part = "spec"
    for items in groups.values():
        ctx.log(f"WARNING {', '.join(items)}: never read by any part (misspelt, in the wrong table, or for something "
                f"switched off?)")
    ctx.part = ""
