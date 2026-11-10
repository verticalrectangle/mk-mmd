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
TABLES = {"strap": ("top", "bottom", "over", "width", "thickness", "material", "standoff", "slack", "sim"),
          "cable": ("object", "anchor", "radius", "trail", "out", "sway", "reach", "tail_len", "slack", "follow", "sim")}


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


STRAP_LEAD = 0.03             # m: a band that would turn back over the prop first runs this far across its button's pin
STRAP_REACH = 0.12            # m: the wearer's surface is looked for this far outside the strap's first line
STRAP_SIDE = (0.005, 0.08)    # m: the grid across the first line her surface is mapped on; how far the band may move to a side
STRAP_TILT = 1.0              # the band rolls with what it lies on up to this slope across it (45 degrees)


def strap_path(top, bottom, shoulder, spine, radius, surface, over_side=1.0, standoff=0.007, width=0.05, n=96):
    """(points, normals, kind): points and normals (n, 3) in character axes, the line of a strap's inner face from the prop's
    upper button `top` over the wearer's `shoulder` (the shoulder joint), across her back and round her other side to the
    lower button `bottom`, and the way the band faces along it; kind (n,) of each point: 0 the prop's (a button, a run
    across its pin), 1 lying on her, 2 clear of her.

    `top` / `bottom` are (point, direction) pairs: a button and the way its pin points. `spine` is the interpolator from
    `spine_axis`, `radius` the torso's half depth (m) at the chest, `over_side` +1 when the shoulder is on the character's
    left (+x). `surface(point, direction, far)` is the wearer herself: None, or (distance, facing) where a ray first
    meets her, facing False when it meets her from within (her skin and clothes without what hangs or swings, her arms
    and her head).

    The band is pulled taut, as the prop's weight does: straight where nothing is under it, on her where she is in the
    way, `standoff` off her surface under its whole `width` and rolled to the slope of what it lies on. A first line is
    drawn without her surface (straight from the upper button to the top of the shoulder, then round the torso's ellipse
    seen from above, falling evenly, to where it runs straight to the lower button: a straight line on the unrolled
    torso); her surface is mapped under it and to either side (STRAP_SIDE); then every point of the line takes the place
    across the line and the height off her that make the band shortest, so it lies against the neck's foot, passes
    beside what stands out of her back and bridges her hollows. Over the shoulder it never moves toward the arm (a taut
    string would slide down a sloping shoulder; a strap's grip holds it). A band that would turn back over the prop from
    a button (the tail of a body worn across her) first runs STRAP_LEAD across the pin: a tab lies flat round its pin,
    and the band comes to the button from behind the prop."""
    flip = np.array([1.0 if over_side >= 0 else -1.0, 1.0, 1.0])         # a right shoulder: solved as its mirror image
    (p0, d0), (p1, d1) = [(np.asarray(a, float) * flip, np.asarray(b, float) * flip / np.linalg.norm(b)) for a, b in (top, bottom)]
    s = np.asarray(shoulder, float) * flip

    def axis(z):
        return np.asarray(spine(z), float) * flip[:2]

    def hit(p, d, far):
        return surface(p * flip, d * flip, far)

    def inside(p):
        """A point within her that the band at p faces away from: on the torso's axis at p's height; over the shoulder,
        under p (the band faces up there, forward before it and backward after)."""
        k = min(max((p[2] - (s[2] - 0.10)) / 0.08, 0.0), 1.0)
        k = k * k * (3.0 - 2.0 * k)
        z = p[2] - 0.07 * k
        x, y = axis(z)
        return np.array([x + k * (p[0] - x), y, z])

    R = float(radius) + float(standoff)
    crest = np.array([abs(s[0]) * 0.95, axis(s[2])[1] - 0.004, s[2] + 0.05])
    cx, cy = axis(0.5 * (s[2] + p1[2]))
    th = np.radians(np.arange(math.degrees(math.acos(min(0.9 * crest[0] / (1.6 * R), 1.0))), 270.0, 1.0))
    E = np.stack([cx + 1.6 * R * np.cos(th), cy + R * np.sin(th)], 1)    # the torso seen from above, from behind the shoulder
    dE = np.stack([-1.6 * R * np.sin(th), R * np.cos(th)], 1)            # round her back to her other side and her front

    def round_her(end):
        """The first line from behind the shoulder round the torso to where it runs straight to `end`: (k, 3)."""
        side = dE[:, 0] * (end[1] - E[:, 1]) - dE[:, 1] * (end[0] - E[:, 0])
        turn = np.nonzero(np.sign(side) != np.sign(side[0]))[0]
        k = int(turn[0]) if len(turn) else len(th) - 1
        xy = np.vstack([E[np.unique(np.r_[np.arange(0, k, 12), k])], end[None, :2]])
        run = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        return np.column_stack([xy, (s[2] - 0.035) + (end[2] - (s[2] - 0.035)) * run / run[-1]])

    def lead(p, d, toward):
        """Where the band turns after running across the pin at p, or None when it leaves the button freely."""
        v = toward - p
        if v @ d >= 0.0:
            return None
        across = v - (v @ d) * d
        return p + STRAP_LEAD * across / max(np.linalg.norm(across), 1e-9)

    first, last = lead(p0, d0, crest), lead(p1, d1, round_her(p1)[-2])
    a, b = (p0 if first is None else first), (p1 if last is None else last)
    way = np.vstack([a, a + (crest - a) / 3.0, a + (crest - a) * 2.0 / 3.0, crest, round_her(b)])
    t = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(way, axis=0), axis=1))]
    q = np.linspace(0.0, 1.0, 2 * int(n))
    G, _ = _resample(np.stack([_cr(way[:, i], t / t[-1], q) for i in range(3)], 1), n)
    T = np.gradient(G, axis=0)
    T /= np.linalg.norm(T, axis=1)[:, None]
    N = G - np.array([inside(p) for p in G])
    N -= np.einsum("ij,ij->i", N, T)[:, None] * T
    N /= np.linalg.norm(N, axis=1)[:, None]
    B = np.cross(T, N)                                                   # across the band: toward the arm over the shoulder
    F, S, step = _strap_floor(G, N, B, hit, float(standoff), float(width))
    side = _strap_side(G, N, B, F, step, np.where(G[:, 2] > s[2] - 0.06, 0.0, np.inf))
    G = G + side[:, None] * B
    floor = _at(F, side / step)
    h = _taut(G, N, floor)
    on = np.nonzero(h - floor < 0.002)[0]                                # where it lies on her it rolls with her slope,
    roll = np.interp(np.arange(len(G)), on, _at(S, side / step)[on]) if len(on) else np.zeros(len(G))
    for _ in range(3):                                                   # and keeps that roll where it hangs free
        roll[1:-1] = 0.25 * roll[:-2] + 0.5 * roll[1:-1] + 0.25 * roll[2:]
    lies = np.zeros(len(G))
    lies[on] = 1.0
    P, N = G + h[:, None] * N, np.c_[N - roll[:, None] * B, lies]
    lead = [0.0, 0.0]                                                    # the band's length held by the prop at each end
    if first is not None:
        P, N, lead[0] = np.vstack([p0, P]), np.vstack([N[:1], N]), STRAP_LEAD
    if last is not None:
        P, N, lead[1] = np.vstack([P, p1]), np.vstack([N, N[-1:]]), STRAP_LEAD
    along = np.linspace(0.0, float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()), int(n))
    P, N = _resample(P, n, N)
    kind = np.where((along <= lead[0] + 1e-9) | (along >= along[-1] - lead[1] - 1e-9), 0, np.where(N[:, 3] > 0.5, 1, 2))
    return P * flip, N[:, :3] / np.linalg.norm(N[:, :3], axis=1)[:, None] * flip, kind


def _resample(P, n, also=None):
    """(n points evenly along the polyline P, the rows of `also` read at the same places)."""
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    q = np.linspace(0.0, d[-1], int(n))
    return (np.stack([np.interp(q, d, P[:, i]) for i in range(3)], 1),
            None if also is None else np.stack([np.interp(q, d, also[:, i]) for i in range(also.shape[1])], 1))


def _strap_floor(G, N, B, hit, standoff, width):
    """(F, S, step), F and S (m, 2K + 1): for the band centred at each point of the first line G and each offset k * step
    across it (along B), the slope S across it of what lies under its width (a line fitted through her surface there,
    STRAP_TILT at most) and the height F along its normal N its centre line must keep for the band, rolled to that slope,
    to clear her by `standoff`: -1 where nothing is under it, STRAP_REACH where the ray meets her from within (the foot
    of her neck: a wall). The ends are left free."""
    step, reach = STRAP_SIDE
    wide, K = max(1, int(round(0.4 * width / step))), int(round(reach / step))
    off = np.arange(-K - wide, K + wide + 1) * step
    M = np.full((len(G), len(off)), np.nan)
    for i in range(1, len(G) - 1):
        for k, b in enumerate(off):
            met = hit(G[i] + b * B[i] + STRAP_REACH * N[i], -N[i], 3.0)
            if met is not None and not met[1]:
                M[i, k] = STRAP_REACH
            elif met is not None and met[0] < STRAP_REACH + 0.25:
                M[i, k] = STRAP_REACH - met[0]
    u = np.arange(-wide, wide + 1) * step
    F, S = np.full((len(G), 2 * K + 1), -1.0), np.zeros((len(G), 2 * K + 1))
    for k in range(2 * K + 1):
        W = M[:, k:k + 2 * wide + 1]
        ok = np.isfinite(W)
        cnt = np.maximum(ok.sum(1), 1)
        Wz = np.where(ok, W, 0.0)
        du, dw = (u - (ok * u).sum(1, keepdims=True) / cnt[:, None]) * ok, (Wz - Wz.sum(1, keepdims=True) / cnt[:, None]) * ok
        var = (du * du).sum(1)
        S[:, k] = np.where((ok.sum(1) >= 3) & (var > 1e-12), np.clip((du * dw).sum(1) / np.maximum(var, 1e-12), -STRAP_TILT, STRAP_TILT), 0.0)
        F[:, k] = np.where(ok.any(1), np.where(ok, W - S[:, k:k + 1] * u, -np.inf).max(1) + standoff, -1.0)
    return F, S, step


def _at(F, f):
    """The map F read at the offsets f (m,), in steps from its middle column: one value per row."""
    K = F.shape[1] // 2
    f = np.clip(np.asarray(f, float) + K, 0.0, 2 * K - 1e-9)
    k = np.floor(f).astype(int)
    r = np.arange(len(F))
    return F[r, k] * (1.0 - (f - k)) + F[r, k + 1] * (f - k)


def _strap_side(G, N, B, F, step, most, sweeps=4000, tol=5e-6):
    """The offset across the first line (m,), `most` (m,) at most, at which each point of the band lies when the band is
    at its shortest over the floor map F: each point in turn takes the best place between its neighbours (every offset of
    the map tried with the best height over it, refined between the grid's lines), until none moves; the offsets are then
    smoothed."""
    m, K = len(G), F.shape[1] // 2
    grid = np.arange(-K, K + 1) * step
    side = np.zeros(m)
    h = np.maximum(F[:, K], 0.0)
    h[0] = h[-1] = 0.0
    for _ in range(sweeps):
        moved = 0.0
        for idx in (np.arange(1, m - 1, 2), np.arange(2, m - 1, 2)):
            P = G + side[:, None] * B + h[:, None] * N
            base = G[idx][:, None, :] + grid[None, :, None] * B[idx][:, None, :]
            _h, cost = _best_height(P[idx - 1][:, None, :] - base, P[idx + 1][:, None, :] - base, N[idx][:, None, :], F[idx])
            k = np.clip(cost.argmin(1), 1, 2 * K - 1)
            r = np.arange(len(idx))
            c0, c1, c2 = cost[r, k - 1], cost[r, k], cost[r, k + 1]
            bend = c0 - 2.0 * c1 + c2
            sub = np.where(bend > 1e-12, np.clip(0.5 * (c0 - c2) / np.maximum(bend, 1e-12), -0.5, 0.5), 0.0)
            new = np.minimum(side[idx] + 0.7 * ((k - K + sub) * step - side[idx]), most[idx])
            base = G[idx] + new[:, None] * B[idx]
            nh, _cost = _best_height(P[idx - 1] - base, P[idx + 1] - base, N[idx], _at(F[idx], new / step))
            moved = max(moved, float(np.abs(nh - h[idx]).max()), float(np.abs(new - side[idx]).max()))
            side[idx], h[idx] = new, nh
        if moved < tol:
            break
    for _ in range(2):
        side[1:-1] = 0.25 * side[:-2] + 0.5 * side[1:-1] + 0.25 * side[2:]
    return np.minimum(side, most)


def _best_height(a, b, N, floor):
    """(h, length): the height h >= floor along the unit normal N, from a base point whose neighbours lie at a and b
    from it, at which the way from one neighbour to the other through the point is shortest, and that way's length."""
    aa, ab = (a * N).sum(-1), (b * N).sum(-1)
    ra = np.sqrt(np.maximum((a * a).sum(-1) - aa * aa, 0.0))
    rb = np.sqrt(np.maximum((b * b).sum(-1) - ab * ab, 0.0))
    h = np.maximum(floor, (aa * rb + ab * ra) / np.maximum(ra + rb, 1e-12))
    return h, np.sqrt((h - aa) ** 2 + ra * ra) + np.sqrt((h - ab) ** 2 + rb * rb)


def _taut(G, N, floor, sweeps=12000, tol=2e-6):
    """Heights h (m,) >= floor along the unit normals N, zero at both ends, at which the line G + h N is shortest (a
    convex problem: each point in turn takes its best height between its neighbours, until none moves)."""
    m = len(G)
    h = np.maximum(floor, 0.0)
    h[0] = h[-1] = 0.0
    for _ in range(sweeps):
        moved = 0.0
        for idx in (np.arange(1, m - 1, 2), np.arange(2, m - 1, 2)):
            P = G + h[:, None] * N
            new, _cost = _best_height(P[idx - 1] - G[idx], P[idx + 1] - G[idx], N[idx], floor[idx])
            moved = max(moved, float(np.abs(new - h[idx]).max()))
            h[idx] = new
        if moved < tol:
            break
    return h


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
