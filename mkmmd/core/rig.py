"""The rig description (rig.json): everything mk knows about a model's rig besides its meshes, as one data model that
every maker writes and every reader reads (docs/design.md: Model description). numpy-free, Blender-safe.

It is organised in layers. `describe` (mkmmd.blender.ops_inspect) reads the first five off any PMX it imports (an
mk-built character or anyone's); `mk model` adds the last two, which a PMX cannot hold, from what its part builders
declare (`Part.rig`):

    layer       rig.json keys                      what it is                                   read by
    identity    schema source scale comment kind   the file, its import scale, what it is      registry, reports
                quirks stats
    skeleton    map missing missing_required      bones (Blender names, `jp` the PMX name),    cast, pose, motion,
                bones fingers measure             semantic names (core.bonemap); a bone's     perform, checks, lab
                                                  `grant`: the helper bone takes `ratio` of
                                                  another bone's rotation (or move)
    colliders   bodies                            rigid bodies, static and dynamic, their     sim, checks, cables
                                                  geometry in the bone's rest frame
    dynamics    chains                            the bones dynamic bodies move, one chain    sim (solvers.strands),
                                                  per root hanging from a non-simulated bone; checks (jitter,
                                                  `ties`: the chains joints between their     penetration)
                                                  bodies tie to it sideways (a sheet)
    morphs      morphs morph_list                 shape keys and morph panels, semantic names perform, sing, moves
    folds       folds                             mk model: shapes a joint angle drives        the folds stage
                                                  (`driver`, core.folds)
    cloth       cloth                             mk model: a garment's simulation copy       the cloth stage
                                                  (arrays in <model>.cloth.npz beside the PMX)

Bone names in rig.json are Blender's (mmd_tools turns the 左/右 prefixes into .L/.R suffixes); a part builder names
bones by their PMX names, and `merge` translates its layers when the build writes rig.json. SCHEMA counts the
description's versions: 2 added `grant`, `ties`, `folds` and `cloth`. Readers take what is there: a key a model
lacks means the model has none of it."""

SCHEMA = 2
MODEL_LAYERS = ("folds", "cloth")       # the layers `mk model` adds (a PMX cannot hold them)


def blender_names(rig):
    """{PMX name: Blender name} of the described model's bones."""
    return {b["jp"]: name for name, b in rig.get("bones", {}).items() if b.get("jp")}


def _bone(names, pmx, where):
    if pmx not in names:
        raise ValueError(f"rig layer {where}: no bone {pmx!r} in the model")
    return names[pmx]


def _fold(entry, names):
    d = dict(entry["driver"])
    for k in ("bone", "ref"):
        if d.get(k):
            d[k] = _bone(names, d[k], f"folds[{entry['morph']}].driver.{k}")
    return dict(entry, driver=d)


def _cloth(entry, names):
    return dict(entry, bones=[_bone(names, b, f"cloth[{entry['name']}].bones") for b in entry.get("bones", [])])


RENAME = {"folds": _fold, "cloth": _cloth}


def merge(rig, layers):
    """`rig` (as describe wrote it) with the model layers `layers` ({layer: [entry, ...]}, bone names PMX as the part
    builders give them) added in Blender bone names. Raises ValueError for an unknown layer or a bone the model lacks."""
    out = dict(rig, schema=SCHEMA)
    names = blender_names(rig)
    for layer, entries in (layers or {}).items():
        if layer not in RENAME:
            raise ValueError(f"unknown rig layer {layer!r} (layers mk model adds: {', '.join(MODEL_LAYERS)})")
        if entries:
            out[layer] = [RENAME[layer](e, names) for e in entries]
    return out


def collect(parts):
    """{layer: [entry, ...]} of every part's `rig` layers, in part order. Raises ValueError for an unknown layer or a
    name two entries share (a fold's morph, a cloth's name)."""
    out, seen = {}, set()
    for p in parts:
        for layer, entries in (getattr(p, "rig", None) or {}).items():
            if layer not in MODEL_LAYERS:
                raise ValueError(f"part {p.name}: unknown rig layer {layer!r} (have {', '.join(MODEL_LAYERS)})")
            for e in entries:
                key = (layer, e.get("morph") if layer == "folds" else e.get("name"))
                if key in seen:
                    raise ValueError(f"part {p.name}: {layer} {key[1]!r} is declared twice")
                seen.add(key)
                out.setdefault(layer, []).append(e)
    return out
