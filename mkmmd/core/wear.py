"""Wearing a prop on a standing character: where a worn prop (a guitar on its strap) sits against the body, and the strap
that holds it (docs/design.md: Prop card `use.wear`, `[[prop]] wear`). numpy only, so the pose stage and the tests share it.

Character axes are the armature's at rest: x the character's left, y behind her, z up (she faces -y). A card's `use.wear`
entry (one of the prop's ways of being worn, found by `name`) says

    bone     the semantic bone the prop rides on (the chest: `upper_body2`)
    pivot    the point of the prop (its frame, metres) that is put at `at`: the back of a guitar at its saddle line
    ref      the reference body the numbers were made for: {measure name: value} (rig.json `measure`)
    at       [x, y, z] metres from the bone's head in character axes for the reference body; every coordinate is multiplied by
             the wearer's measure / the reference's measure named in `scale` (one measure name per coordinate)
    neck_deg, yaw_deg, roll_deg   how the prop is turned from its upright pose: its +z axis (a guitar's neck) is brought
             to the character's left and raised by neck_deg above horizontal, swung toward the front by yaw_deg, and the
             prop is rolled about that axis by roll_deg (positive turns the prop's -y face up toward the sky)
    strap, cable   see `strap_path` and mkmmd.core.cable

The prop's own frame needs the convention of a guitar: z along the neck, its face looking along -y, x across it. The
project overrides any of the numbers with `[[prop]] wear = {neck_deg = 20, at = [..], ...}`."""
import math

import numpy as np

NUMBERS = ("at", "neck_deg", "yaw_deg", "roll_deg", "pivot", "scale")
TABLES = {"strap": ("top", "bottom", "over", "width", "thickness", "material", "shoulder_radius"),
          "cable": ("object", "anchor", "radius", "trail", "out", "sway", "reach", "tail_len", "follow", "sim")}


class WearError(ValueError):
    pass


def rot(axis, deg):
    """3x3 right-handed rotation about a coordinate axis ('x', 'y', 'z') or a vector, by deg."""
    a = np.asarray({"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}.get(axis, axis), float)
    a = a / np.linalg.norm(a)
    th = math.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def entry(card_use, name=None):
    """The `use.wear` entry called `name` (the only one when None)."""
    items = (card_use or {}).get("wear") or []
    if not items:
        raise WearError("the card has no use.wear entry")
    if name is None:
        if len(items) > 1:
            raise WearError(f"the card has several wear entries ({[i['name'] for i in items]}): name one")
        return items[0]
    for it in items:
        if it["name"] == name:
            return it
    raise WearError(f"the card has no wear entry {name!r} ({[i['name'] for i in items]})")


def params(e, override=None):
    """The entry's numbers with the project's overrides on top (`at`, `pivot`, `scale`, `neck_deg`, `yaw_deg`, `roll_deg`); the
    `strap` and `cable` tables of an override are `table`'s."""
    out = {k: e[k] for k in NUMBERS if k in e}
    for k, v in (override or {}).items():
        if k in TABLES:
            continue
        if k not in NUMBERS:
            raise WearError(f"wear: unknown key {k!r} (have {', '.join(NUMBERS + tuple(TABLES))}, and use / cast)")
        out[k] = v
    out.setdefault("roll_deg", 0.0)
    out.setdefault("yaw_deg", 0.0)
    out.setdefault("neck_deg", 0.0)
    return out


def table(e, override, key):
    """The entry's `strap` or `cable` table with the project's `key = {...}` laid over it, key by key (None when neither the card
    nor the project has one). The project's keys must be ones the table has (TABLES), and the card's entry must have the table:
    a project changes where a strap or a cord goes, it does not invent one."""
    base, over = e.get(key), (override or {}).get(key)
    if over is None:
        return base
    if not isinstance(over, dict):
        raise WearError(f"wear: `{key}` is a table of {', '.join(TABLES[key])}")
    unknown = sorted(set(over) - set(TABLES[key]))
    if unknown:
        raise WearError(f"wear {key}: unknown key{'s' if len(unknown) > 1 else ''} {', '.join(repr(k) for k in unknown)} "
                        f"(known: {', '.join(TABLES[key])})")
    if not base:
        raise WearError(f"wear: the card's entry {e.get('name')!r} has no `{key}` to change")
    return {**base, **over}


def orientation(neck_deg, yaw_deg=0.0, roll_deg=0.0):
    """3x3 whose columns are the prop's x, y, z axes in character axes: upright prop -> neck (+z) to the character's left and
    raised by neck_deg, swung toward her front by yaw_deg, then rolled by roll_deg about the neck."""
    base = np.array([[0.0, 0.0, 1.0],          # prop x -> down, prop y -> back, prop z -> left (columns: images of x, y, z)
                     [0.0, 1.0, 0.0],
                     [-1.0, 0.0, 0.0]])
    roll = rot((0.0, 0.0, 1.0), -roll_deg)      # about the prop's own neck axis: positive turns -y (the face) up
    return rot("z", -yaw_deg) @ rot("y", -neck_deg) @ base @ roll


def scaled_at(p, measure, ref):
    """`at` for this wearer: each coordinate times measure[name] / ref[name] for its entry of `scale` (no scale: as given)."""
    at = np.asarray(p["at"], float)
    names = p.get("scale") or [None, None, None]
    out = at.copy()
    for i, nm in enumerate(names):
        if nm is None:
            continue
        if nm not in measure or nm not in ref:
            raise WearError(f"wear scale {nm!r}: the wearer's measure has {sorted(measure)}, the reference {sorted(ref)}")
        out[i] = at[i] * float(measure[nm]) / float(ref[nm])
    return out


def placement(e, measure, override=None):
    """(R, pivot_offset, root_offset): the prop's axes in character axes, where its pivot sits (metres from the bone's head,
    character axes) and where its origin sits (the pivot minus R @ pivot): the prop root's matrix relative to the bone's head
    frame `[R | root_offset]` in character axes."""
    p = params(e, override)
    ref = e.get("ref") or {}
    at = scaled_at(p, measure, ref)
    R = orientation(p["neck_deg"], p["yaw_deg"], p["roll_deg"])
    piv = np.asarray(p.get("pivot", (0.0, 0.0, 0.0)), float)
    return R, at, at - R @ piv


def matrix(e, measure, bone_head, override=None):
    """4x4 of the prop root in the armature's frame (the character axes at rest): bone_head + the placement."""
    R, at, root = placement(e, measure, override)
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = np.asarray(bone_head, float) + root
    return M


# =============================================================================================== the strap
def spine_axis(points):
    """Interpolator y(z), x(z) of the torso's centre line from spine bone heads (rows [x, y, z], any order): the strap goes
    round the torso about it."""
    P = np.asarray(sorted(map(tuple, points), key=lambda r: r[2]), float)

    def at(z):
        return np.array([np.interp(z, P[:, 2], P[:, 0]), np.interp(z, P[:, 2], P[:, 1])])
    return at


def strap_path(top, bottom, shoulder, spine, radius, over_side=1.0, standoff=0.007, shoulder_r=0.045, n=48):
    """The centre line of a strap from the prop's upper button `top` over the wearer's `shoulder` (the shoulder joint, character
    axes) down her back to the hip on the other side and to the lower button `bottom`: (n, 3) points, character axes.

    `top` / `bottom` are (point, direction) pairs: where the strap leaves each button and the way it goes first.
    `spine` is the interpolator from `spine_axis`, `radius` the torso's half depth (m) at the chest (the strap rides
    `standoff` outside it), `over_side` +1 when the shoulder is on the character's left (+x)."""
    (p0, d0), (p1, d1) = [(np.asarray(a, float), np.asarray(b, float) / np.linalg.norm(b)) for a, b in (top, bottom)]
    s = np.asarray(shoulder, float)
    R = float(radius) + float(standoff)
    z_top = s[2] + float(shoulder_r)

    def on_torso(z, x, front):
        """A point at height z and lateral x on the torso's front or back surface (an ellipse about its axis)."""
        ax = spine(z)
        a = R * (1.15 if z < s[2] - 0.2 else 1.0)                   # a little wider at the hips
        y = ax[1] + (-1.0 if front else 1.0) * a * math.sqrt(max(1.0 - (x / (1.6 * a)) ** 2, 0.15))
        return np.array([x, y, z])

    xs = over_side * abs(s[0])
    zh = p1[2]                                                       # the lower button's height: the strap meets the hip there
    pts = [p0, p0 + d0 * 0.03,
           on_torso(s[2] - 0.035, xs * 0.85, True),                   # across the chest, just inside the shoulder joint
           np.array([xs * 0.95, spine(z_top)[1] - 0.004, z_top]),     # over the top of the shoulder
           on_torso(s[2] - 0.035, xs * 0.85, False),                  # down behind it
           on_torso(0.5 * (s[2] + zh) + 0.05, xs * 0.25, False),      # diagonally across the back ...
           on_torso(zh + 0.12, -over_side * abs(s[0]) * 0.55, False),
           np.array([-over_side * (1.55 * R), spine(zh + 0.04)[1] + 0.02, zh + 0.04]),    # ... round the far hip ...
           on_torso(zh + 0.02, -over_side * abs(s[0]) * 0.45, True),
           p1 + d1 * 0.03, p1]
    P = np.array(pts, float)
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    t = d / d[-1]
    q = np.linspace(0.0, 1.0, int(n))
    return np.stack([_cr(P[:, i], t, q) for i in range(3)], 1)


def _cr(v, t, q):
    """Monotone-in-t cubic (Catmull-Rom style, centripetal-free) through (t, v) sampled at q."""
    h = np.diff(t)
    m = np.zeros_like(v)
    m[1:-1] = (v[2:] - v[:-2]) / (t[2:] - t[:-2])
    m[0], m[-1] = (v[1] - v[0]) / h[0], (v[-1] - v[-2]) / h[-1]
    k = np.clip(np.searchsorted(t, q, side="right") - 1, 0, len(h) - 1)
    s = (q - t[k]) / h[k]
    s2, s3 = s * s, s * s * s
    return (2 * s3 - 3 * s2 + 1) * v[k] + (s3 - 2 * s2 + s) * h[k] * m[k] + (-2 * s3 + 3 * s2) * v[k + 1] + (s3 - s2) * h[k] * m[k + 1]
