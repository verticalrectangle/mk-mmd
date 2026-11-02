"""Layout of the railroad track (pure numpy + mkmmd.core.shell, no bpy: importable from tests): the options, the colour
roles, the meshes of its parts and the card. The plan and the laying front are mkmmd/core/rail_track.py.

An 1860s standard-gauge track: hewn ties 8.5 ft long (flat top and bottom faces, the log's rounded sides, wane and adze
marks; six variants), 56 lb/yd iron T rails in 30 ft lengths joined by fish plates with four bolts, cut spikes hooked over
the rail foot (two a rail on every tie, staggered), optional tie plates (`plates`: a later era's), on a ballast bed with
sloped shoulders. No lettering.

Frames (metres; the ground at z = 0, the track from the root along +X):
  bed piece   in its own frame, from its start (x = 0) along the track one tie spacing (following the curve); the ends are
              1 : 1 ramps (a piece's start ramp lies inside the last one's end ramp; the first piece's leans the other
              way: the track's start), the bottom 2 cm under the ground
  tie         its top face's middle at the origin, its length along Y (across the track), x along the track, z up; it
              lies SINK down in the bed (its top at TIE_TOP)
  spikes      the four spikes of a tie in the tie's frame, the origin SPIKE_DROP under the tie's top (a flattened set
              hides in the tie)
  rail span   both rails from s = 0 to the span's length, their seat (the foot's underside) at z = 0, and from the second
              span on the joint bars and bolts at its start; straight: in the track's frame; curved: about the circle's
              centre (the builder turns the object about it to slide the span along the arc)

Colour roles (`slots` of the prop: a palette slot, `paper` / `ink`, a hex or a mix `a:b:t`): ballast and ballast_dark
(the bed's stones), tie and grain (weathered wood and its dark grain and checks), rail, railhead (the bright running
surface), rust, spike, plate (fish plates, bolts, tie plates)."""
import math

import numpy as np

from ....core import place as PL
from ....core import rail_track as RT
from ....core import shell as S

ROLES = {"ballast": "subtle:gold:0.25", "ballast_dark": "muted:ink:0.45", "tie": "gold:muted:0.6", "grain": "gold:ink:0.7",
         "rail": "muted:ink:0.35", "railhead": "subtle:paper:0.35", "rust": "love:gold:0.45", "spike": "ink:muted:0.35",
         "plate": "muted:ink:0.5"}
MATS = ("ballast", "tie", "rail", "railhead", "spike", "plate")       # material index -> material (roles mix inside)
OPTIONS = {"length": 30.0, "radius": 0.0, "plates": False, "seed": 1, "laid": None}
VARIANTS = 6

TIE_TOP = RT.BED_H - RT.SINK + RT.TIE_T              # the ties' top faces over the ground
PLATE_T, PLATE_L, PLATE_W = 0.014, 0.19, 0.26        # a tie plate: thick, along the track, across
RAIL_Y = RT.GAUGE / 2 + RT.HEAD_W / 2                # the rails' centre lines either side of the track's
WEB_T, FOOT_EDGE, HEAD_D = 0.0135, 0.0085, 0.030     # the rail's web thickness, foot edge thickness, head depth
SHANK, HEAD = 0.0159, (0.027, 0.0349, 0.0159)        # a cut spike: 5/8 in shank; head along x, toward the rail, high
SPIKE_STAGGER = 0.045                                # spikes off the tie's middle along the track, one either way
SPIKE_DEEP = 0.07                                    # how far a spike's shank is modelled into the tie
SPIKE_DROP = 0.003                                   # the spikes' origin under the tie's top
BAR_L, BAR_T, BAR_H = 0.508, 0.016, 0.040            # a fish plate: 20 in long, thick, high
BOLTS = (-0.19, -0.065, 0.065, 0.19)                 # the bolts along a joint, from the joint
BED_TOP_Y = RT.TIE_L / 2 + RT.BED_SHOULDER          # the bed's top half width
BED_TOE_Y = BED_TOP_Y + RT.SLOPE * RT.BED_H          # its toe's
BED_FLOOR = -0.02                                    # its underside, under the ground
BED_DOME = 0.06                                      # a piece's end ramp bulges this far forward at its middle


class LayoutError(ValueError):
    """Bad options for the track."""


def options(slots=None):
    """The build options of the `slots` of a [[prop]] (the colour roles are read by the builder): length (m), radius (m;
    0 straight, > 0 left), plates (tie plates under the rails), seed (the ties' variety), laid (the starting `laid`;
    default the whole track: `done`)."""
    s = dict(slots or {})
    out = dict(OPTIONS)
    for k in OPTIONS:
        if k in s:
            out[k] = s[k]
    for k in ("length", "radius"):
        if isinstance(out[k], bool) or not isinstance(out[k], (int, float)):
            raise LayoutError(f"rail_track {k} = {out[k]!r}: a number (metres)")
        out[k] = float(out[k])
    if not isinstance(out["plates"], bool):
        raise LayoutError(f"rail_track plates = {out['plates']!r}: true or false")
    if isinstance(out["seed"], bool) or not isinstance(out["seed"], int):
        raise LayoutError(f"rail_track seed = {out['seed']!r}: a whole number")
    if out["laid"] is not None and (isinstance(out["laid"], bool) or not isinstance(out["laid"], (int, float))):
        raise LayoutError(f"rail_track laid = {out['laid']!r}: metres along the track")
    try:
        plan = RT.Plan(out["length"], out["radius"])
    except RT.TrackError as e:
        raise LayoutError(f"rail_track: {e}") from None
    out["laid"] = plan.done if out["laid"] is None else float(out["laid"])
    return out, plan


def seat_z(plates):
    """The rails' seat (the underside of the foot) over the ground."""
    return TIE_TOP + (PLATE_T if plates else 0.0)


# ======================================================================================================== the bed
def bed_profile():
    """(y, z) of the bed's cross-section, an open polyline from the left toe (its slope carried on 2 cm into the ground)
    over the top, its shoulders rounded and crowned 4 mm, down to the right toe: counter-clockwise when closed along the
    bottom. The underside is never built: nobody sees it."""
    under = RT.SLOPE * -BED_FLOOR
    pts = np.array([(BED_TOE_Y + under, BED_FLOOR), (BED_TOE_Y, 0.0), (BED_TOP_Y, RT.BED_H), (0.0, RT.BED_H + 0.004),
                    (-BED_TOP_Y, RT.BED_H), (-BED_TOE_Y, 0.0), (-BED_TOE_Y - under, BED_FLOOR)])
    return S.fillet(pts, [0.0, 0.0, 0.22, 0.0, 0.22, 0.0, 0.0], n=4)


def _arc_point(k, u, y):
    """(x, y) in a frame whose x runs along the track from its origin, curving with curvature k, at u along it and y to
    its left."""
    u, y = np.asarray(u, float), np.asarray(y, float)
    if k == 0.0:
        return u, y
    a = k * u
    return np.sin(a) / k - np.sin(a) * y, (1.0 - np.cos(a)) / k + np.cos(a) * y


def bed_mesh(k, first=False, rings=5):
    """One bed piece in its own frame (curvature k): the profile carried along one SPACING, the ends 1 : 1 ramps (the
    toe at the piece's ends, the top a bed's height short of them), each a cap of its own (so the sides' smooth normals
    carry on into the next piece without a seam) domed BED_DOME forward: a heap's rounded front at the laying front, and
    the same surface for the next piece's start. The first piece's start ramps up from the track's start instead."""
    prof = bed_profile()
    y, z = prof[:, 0], prof[:, 1]
    zc = np.clip(z, 0.0, RT.BED_H)                               # a point's height over the toe (the floor's: the toe's)
    u0 = zc if first else -zc                                    # the start: a ramp up (the track's start) or the last's
    u1 = RT.SPACING - zc                                         # the end: the toe at the piece's end, the top short of it
    t = np.linspace(0.0, 1.0, rings)
    U = u0[None, :] + t[:, None] * (u1 - u0)[None, :]
    X, Y = _arc_point(k, U, np.broadcast_to(y, U.shape))
    G = np.stack([X, Y, np.broadcast_to(z, U.shape)], -1)
    side = S.cage(G, closed=False, flip=False, mat=MATS.index("ballast"))     # (y, z, along) is right-handed: outward
    caps = []
    for end, ring_u in ((0, u0), (-1, u1)):
        ring = G[end]
        uc = float(ring_u.mean())
        tangent = np.array([math.cos(k * uc), math.sin(k * uc), 0.0])
        out = -1.0 if end == 0 else 1.0                          # away from the piece
        c = ring.mean(0) + tangent * BED_DOME * (out if first and end == 0 else 1.0)
        m = len(ring)
        j = np.arange(m)
        cap = S.Mesh(np.vstack([ring, c]), T=np.stack([j, (j + 1) % m, np.full(m, m)], 1), mat=MATS.index("ballast"))
        if cap.face_vectors()[1].sum(0) @ (tangent * out) < 0:
            cap = cap.flip()
        caps.append(cap)
    return S.merge([side] + caps)


def layers(plan, plates=False, step=0.5):
    """The card's height layers (core.place.layers_of_points: the plan box of each height slice) from the track's
    outline: the bed narrowing up its shoulders, the ties' ends, the rails. What the props stage would measure, without
    sampling every triangle."""
    s = np.linspace(0.0, plan.length, max(2, math.ceil(plan.length / step) + 1))
    prof = bed_profile()
    pts = [plan.point(s, y=yy, z=zz) for yy, zz in prof if zz >= 0.0]
    for zz in (RT.BED_H - RT.SINK, TIE_TOP):
        pts += [plan.point(plan.ties, y=side * RT.TIE_L / 2, z=zz) for side in (1.0, -1.0)]
    for zz in (seat_z(plates), seat_z(plates) + RT.RAIL_H):
        pts += [plan.point(s, y=side * (RAIL_Y + RT.FOOT_W / 2), z=zz) for side in (1.0, -1.0)]
    return [[round(float(v), 4) for v in layer] for layer in PL.layers_of_points(np.concatenate(pts), 0.05)]


# ======================================================================================================== ties
def _wobble(rng, y, scale, waves=3):
    """A smooth random curve along y in -1..1-ish: a few sines of random wavelength and phase."""
    out = np.zeros_like(y)
    for _ in range(waves):
        lam = rng.uniform(0.6, 2.4)
        out += np.sin(2.0 * math.pi * y / lam + rng.uniform(0.0, 2.0 * math.pi)) / waves
    return scale * out


def tie_section(wt, wm, wb, t, rc, dz=0.0, x0=0.0):
    """(x, z) of a hewn tie's section: the hewn top face (half width wt) at z = dz, the log's rounded sides bulging to wm
    at mid depth, the bottom face (half width wb) at z = -t; the top corners rounded by rc (wider where the log's wane
    shows), a closed counter-clockwise loop."""
    pts = np.array([(wt, dz), (wm, -0.5 * t), (wb, -t), (-wb, -t), (-wm, -0.5 * t), (-wt, dz)])
    pts[:, 0] += x0
    return S.fillet(pts, [rc, 0.07, 0.012, 0.012, 0.07, rc], n=3, closed=True)


def tie_mesh(variant, seed=1, plates=False, step=0.07):
    """Tie `variant` (0 .. VARIANTS - 1): ONE cage of hewn sections along its length (y), the ends' edges rounded and
    capped; the top face at z = 0 at its highest. With `plates` the two tie plates lie on it under the rails."""
    rng = np.random.default_rng([int(seed), 13, int(variant)])
    half = RT.TIE_L / 2 * (1.0 + rng.uniform(-0.012, 0.012))
    inner = np.linspace(-half + 0.03, half - 0.03, max(4, round((2 * half - 0.06) / step) + 1))
    ys = np.concatenate([[-half, -half + 0.008, -half + 0.02], inner, [half - 0.02, half - 0.008, half]])
    shrink = np.ones(len(ys))
    shrink[[0, -1]], shrink[[1, -2]] = 0.93, 0.985          # the end edges rolled over
    wt = RT.TIE_W / 2 + _wobble(rng, ys, 0.008)
    wm = wt + 0.016 + _wobble(rng, ys, 0.006)
    wb = wt - 0.006 + _wobble(rng, ys, 0.006)
    t = RT.TIE_T + _wobble(rng, ys, 0.005)
    rc = 0.014 + np.maximum(_wobble(rng, ys, 0.03, waves=2), 0.0)                       # wane where it grows
    adze = -0.0009 * (1.0 - np.cos(2.0 * math.pi * ys / rng.uniform(0.11, 0.16) + rng.uniform(0, 6.3)))
    bow = _wobble(rng, ys, 0.004, waves=2)
    rings = []
    for i, y in enumerate(ys):
        sec = tie_section(wt[i], wm[i], wb[i], t[i], rc[i], adze[i], bow[i])
        c = np.array([bow[i], -0.5 * t[i]])
        sec = c + (sec - c) * shrink[i]
        rings.append(np.stack([sec[:, 0], np.full(len(sec), y), sec[:, 1]], 1))
    body = S.cage(np.array(rings), closed=True, caps=("fan", "fan"), cap_bulge=(0.002, 0.002), mat=MATS.index("tie"))
    body.V[:, 2] -= body.V[:, 2].max()                     # the top's high points carry the rails at z = 0
    if not plates:
        return body
    return S.merge([body] + [plate_mesh().moved((0.0, side * RAIL_Y, -0.0005)) for side in (1.0, -1.0)])


def plate_mesh():
    """A tie plate under a rail (its middle at the origin, on z = 0): a rounded plate with two shoulders along the rail's
    foot."""
    plate = S.rounded_panel(S.rrect(PLATE_L, PLATE_W, 0.012, 3), PLATE_T, r_edge=0.003, mat=MATS.index("plate"))
    ridges = [S.rounded_box((PLATE_L - 0.02, 0.012, 0.009), (0.0, side * (RT.FOOT_W / 2 + 0.007), PLATE_T + 0.0025),
                            r=0.004, k=2, mat=MATS.index("plate")) for side in (1.0, -1.0)]
    return S.merge([plate] + ridges)


# ======================================================================================================== spikes
def spike_mesh(toward):
    """One cut spike standing at the origin on its shank's axis, its head's underside at z = 0 and the head hooked
    toward `toward` (+1: +y, -1: -y) over the rail foot."""
    shank = S.rounded_box((SHANK, SHANK, SPIKE_DEEP + 0.002), (0.0, 0.0, -SPIKE_DEEP / 2 + 0.001), r=0.002, k=1,
                          mat=MATS.index("spike"))
    hx, hy, hz = HEAD
    head = S.rounded_box((hx, hy, hz), (0.0, toward * (hy - SHANK) / 2, hz / 2), r=0.0045, k=2, mat=MATS.index("spike"))
    return S.merge([shank, head])


def spikes_mesh(plates=False):
    """The four spikes of a tie in the tie's frame, raised SPIKE_DROP (the object's origin is that far under the tie's
    top): two a rail, gauge side and field side, staggered along the track, their heads on the foot's edge."""
    z = SPIKE_DROP + (PLATE_T if plates else 0.0) + FOOT_EDGE
    gap = 0.0005 + (0.012 if plates else 0.0)
    out = []
    for side in (1.0, -1.0):
        for inward in (True, False):
            y = side * (RAIL_Y + (-1.0 if inward else 1.0) * (RT.FOOT_W / 2 + SHANK / 2 + gap))
            x = SPIKE_STAGGER * side * (1.0 if inward else -1.0)
            toward = side * (1.0 if inward else -1.0)             # the head hooks over the foot, toward the rail
            out.append(spike_mesh(toward).moved((x, y, z)))
    return S.merge(out)


# ======================================================================================================== rails
def rail_section(n=3):
    """(u, v) of the rail's section, u across (left +), v up from the seat, a closed counter-clockwise loop: the foot
    with its sloped top, the web, the head with its sloped underside and a crowned running surface."""
    fw, ww, hw, H = RT.FOOT_W / 2, WEB_T / 2, RT.HEAD_W / 2, RT.RAIL_H
    tan = math.tan(math.radians(13.0))
    foot_top = FOOT_EDGE + (fw - ww) * tan
    head_low = H - HEAD_D
    under = head_low - (hw - ww) * tan
    right = [(fw, 0.0), (fw, FOOT_EDGE), (ww, foot_top), (ww, under), (hw, head_low), (hw - 0.001, H - 0.0015)]
    pts = [(0.0, 0.0)] + right + [(0.0, H)] + [(-u, v) for u, v in reversed(right)]
    rad = [0.0, 0.0025, 0.002, 0.009, 0.008, 0.003, 0.008, 0.0, 0.008, 0.003, 0.008, 0.009, 0.002, 0.0025]
    return S.fillet(np.array(pts), rad, n=n, closed=True)


def ear_clip(P):
    """Triangles (k, 3) of a simple counter-clockwise polygon P (m, 2), by ear clipping (the caps of a section that is
    not star-shaped: a T rail's end)."""
    idx = list(range(len(P)))
    out = []

    def cross(a, b, c):
        return (P[b, 0] - P[a, 0]) * (P[c, 1] - P[a, 1]) - (P[b, 1] - P[a, 1]) * (P[c, 0] - P[a, 0])

    guard = 0
    while len(idx) > 3 and guard < 10 * len(P) * len(P):
        guard += 1
        n = len(idx)
        for k in range(n):
            a, b, c = idx[k - 1], idx[k], idx[(k + 1) % n]
            if cross(a, b, c) <= 1e-14:
                continue                                            # a reflex (or flat) corner is no ear
            inside = False
            for q in idx:
                if q in (a, b, c):
                    continue
                if cross(a, b, q) >= 0 and cross(b, c, q) >= 0 and cross(c, a, q) >= 0:
                    inside = True
                    break
            if not inside:
                out.append((a, b, c))
                idx.pop(k)
                break
        else:
            raise ValueError("ear_clip: no ear (the polygon crosses itself or runs clockwise)")
    out.append(tuple(idx))
    return np.array(out, np.int64)


def along(plan, s, section, y0=0.0, z0=0.0, mat=0):
    """A solid of the 2D `section` (u across to the left, v up; counter-clockwise) carried along the track at stations
    s, `y0` left of the centre line and `z0` up: every ring exactly square to the track (on a curve: radial), the ends
    capped flat by ear clipping."""
    s = np.asarray(s, float)
    sec = np.asarray(section, float)
    m = len(sec)
    rings = np.stack([plan.point(np.full(m, sk), y=y0 + sec[:, 0], z=z0 + sec[:, 1]) for sk in s])
    body = S.cage(rings, closed=True, caps=(None, None), flip=False, mat=mat)     # (u, v, forward) is right-handed:
    tri = ear_clip(sec)                                                            # the quads face out, as do the caps
    T = np.concatenate([tri[:, ::-1], tri + (len(s) - 1) * m])
    mesh = S.Mesh(body.V, body.Q, T, body.Qm, np.full(len(T), mat, np.int32))
    return mesh if mesh.volume() >= 0 else mesh.flip()


def _stations(plan, s0, s1, step):
    return np.linspace(s0, s1, 2 if plan.k == 0.0 else max(2, math.ceil((s1 - s0) / step) + 1))


def rail_mesh(plan, length, joint=False):
    """A rail span from station 0 to `length` (span 0's place; the builder moves it): both rails, the running surface's
    faces in the railhead material, and with `joint` the fish plates and bolts at its start (they reach back over the
    last rail). Curved: about the circle's centre."""
    sec = rail_section()
    top = sec[:, 1] > RT.RAIL_H - 0.004
    m = len(sec)
    parts = []
    for side in (1.0, -1.0):
        r = along(plan, _stations(plan, 0.0, length, 0.5), sec, side * RAIL_Y, mat=MATS.index("rail"))
        j = np.arange(len(r.Q)) % m
        r.Qm[top[j] & top[(j + 1) % m]] = MATS.index("railhead")
        parts.append(r)
        if joint:
            parts += joint_meshes(plan, side)
    out = S.merge(parts)
    if plan.k != 0.0:
        out = out.moved((-plan.centre[0], -plan.centre[1], 0.0))
    return out


def joint_meshes(plan, side):
    """The two fish plates of the rail at `side` (+1 left, -1 right) across the joint at station 0, and their four bolts."""
    sec = S.rrect(BAR_T, BAR_H, 0.004, 3)
    v = (FOOT_EDGE + (RT.FOOT_W / 2 - WEB_T / 2) * math.tan(math.radians(13.0)) + RT.RAIL_H - HEAD_D) / 2
    out = []
    bolt = S.lathe(S.fillet(np.array([(0.0, 0.0), (0.0125, 0.0), (0.0125, 0.007), (0.0, 0.0105)]), [0, 0.0015, 0.004, 0],
                            n=2), seg=6, mat=MATS.index("plate"))
    for face in (1.0, -1.0):                                   # field and gauge side of the web
        u = side * RAIL_Y + face * (WEB_T / 2 + BAR_T / 2 + 0.0004)
        out.append(along(plan, _stations(plan, -BAR_L / 2, BAR_L / 2, 0.1), sec, u, v, MATS.index("plate")))
        for ds in BOLTS:
            x, y, h = (float(a) for a in plan.pose(float(ds)))
            n = np.array([-math.sin(h), math.cos(h), 0.0])
            base = np.array([x, y, v]) + n * (u + face * (BAR_T / 2 + 0.0003))
            out.append(S.place(bolt, base, n * face, up=(math.cos(h), math.sin(h), 0.0)))
    return out


# ======================================================================================================== card
def bounds(plan):
    """{min, max} of the whole track in the root's frame (the bed's toes along both edges, the rails' tops)."""
    s = np.linspace(0.0, plan.length, max(2, int(plan.length / 1.0) + 1))
    P = np.concatenate([plan.point(s, y=BED_TOE_Y), plan.point(s, y=-BED_TOE_Y)])
    top = seat_z(True) + RT.RAIL_H + 0.06
    lo = np.array([P[:, 0].min() - 0.05, P[:, 1].min(), BED_FLOOR])
    hi = np.array([P[:, 0].max() + 0.05, P[:, 1].max(), top])
    return {"min": [round(float(v), 4) for v in lo], "max": [round(float(v), 4) for v in hi]}


def card(name, plan, opts, roles):
    """The prop card (docs/design.md: Prop card): the front's anchor (an empty that rides `laid`), the last tie's top for
    a spike to stand in, the look points, the live `laid` and the track's numbers."""
    rail_top = seat_z(opts["plates"]) + RT.RAIL_H
    end = plan.point([plan.length], z=rail_top)[0]
    last = plan.point([plan.ties[-1]], z=TIE_TOP)[0]
    r = lambda v: [round(float(a), 4) for a in v]
    b = bounds(plan)
    return {"size": [round(hi - lo, 4) for lo, hi in zip(b["min"], b["max"])], "origin": "start", "front": "-Y",
            "bounds": b, "layers": layers(plan, opts["plates"]), "slots": dict(roles),
            "params": {"laid": round(opts["laid"], 4)},
            "use": {"anchor": [{"name": "front", "point": [0.0, 0.0, round(rail_top, 4)], "object": f"{name}_front",
                                "dir": [1.0, 0.0, 0.0]},
                               {"name": "last_tie", "point": r(last), "dir": [0.0, 0.0, 1.0]}],
                    "look": [{"name": "start", "point": [0.0, 0.0, round(rail_top, 4)]},
                             {"name": "end", "point": r(end)},
                             {"name": "front", "point": [0.0, 0.0, round(rail_top, 4)], "object": f"{name}_front"}]},
            "track": {"length": round(plan.length, 4), "radius": plan.radius, "gauge": RT.GAUGE,
                      "spacing": RT.SPACING, "rail": round(RT.RAIL, 4), "done": round(plan.done, 4),
                      "ties": plan.n, "rails": len(plan.spans), "plates": bool(opts["plates"]),
                      "tie_top": round(TIE_TOP, 4), "rail_top": round(rail_top, 4),
                      "objects": {"bed": f"{name}_bed<k>", "ties": f"{name}_tie<i>", "spikes": f"{name}_spikes<i>",
                                  "rails": f"{name}_rail<j>", "front": f"{name}_front"}}}
