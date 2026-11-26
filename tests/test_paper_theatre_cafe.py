"""The paper theatre's café (mkmmd.core.paper_theatre_cafe and its meshes in paper_theatre_parts, no bpy): a theatre
without it is left alone, the room round her stays clear of her strings, the window behind her head shows sky and an
old taped rip to a close camera, the hole is a ragged rip that bursts open and the tape lands across it on their times,
seen whole, and every piece is a closed solid."""
import math

import numpy as np
import pytest

from mkmmd.blender.library.props import parent_cutout_layout as PCL
from mkmmd.blender.library.sets import paper_theatre_parts as P
from mkmmd.core import palette as PAL
from mkmmd.core import paper_theatre as PT
from mkmmd.core import paper_theatre_cafe as PC

DAWN = PAL.PALETTES["rose-pine-dawn"]
SPEC = {"name": "theatre", "kind": "paper_theatre", "step": 12, "cafe": {"hole": 29.4, "slap": 30.3}}


def _scene(**kw):
    o = PT.options(dict(SPEC, **kw))
    lay, cafe = PT.scene(o)
    return o, lay, cafe


def _held(o, t):
    """The clip second of the set's held step that shows clip second t (60 fps, frame0 361)."""
    _keys, times, _ = PT.key_times(1, 2300, 361, 60.0, o["step"], o["step_from"])
    shown = (np.round(times * 60.0) + 361).astype(int)
    return float(times[np.searchsorted(shown, 361 + round(t * 60.0), side="right") - 1])


def test_options_and_a_theatre_without_the_cafe_is_left_alone():
    with pytest.raises(PT.TheatreError) as e:
        PT.options(dict(SPEC, cafe={"hole": 29.4, "tape": 30.3}))
    assert "tape" in str(e.value) and "slap" in str(e.value)
    for bad in ({"cafe": {"hole": 30.0, "slap": 29.0}}, {"cafe": {"hole": True}}, {"pond": True},
                {"colors": {"wallpaper": "rose"}}):
        with pytest.raises(PT.TheatreError):
            PT.options(dict(SPEC, **bad))
    plain = PT.options({"name": "t", "kind": "paper_theatre"})
    lay, extra = PT.scene(plain)
    assert extra is None and lay is PT.layout(2.2, 3.2, 2.35, 1) and set(PT.colours(plain, DAWN)) == set(PT.ROLES)
    o, lay, _c = _scene(colors={"wall": "rose"})
    c = PT.colours(o, DAWN)
    assert c["wall"] == pytest.approx(PAL.srgb(DAWN["rose"])) and c["sky"] != PT.colours(plain, DAWN)["sky"]
    assert lay.sun is None and not {"sun", "sun_rays", "hills_back", "grass"} & {p.name for p in lay.statics}


def _points(part):
    pts = []
    for p in part.plates:
        V = np.stack([p.outline[:, 0], np.full(len(p.outline), p.y), p.outline[:, 1]], 1)
        if p.frame is not None:
            R, t = p.frame
            V = V @ np.asarray(R, float).T + np.asarray(t, float)
        pts.append(V)
    pts += [np.array([a, b]) for a, b, _r, _role in part.tubes]
    return np.concatenate(pts)


def test_the_room_round_her_strings_stays_clear():
    """Nothing of the café (but the floor) comes into |x| < 0.85, |y| < 0.6 under the border, the border is over 3 m
    above her, and the lamps' round shades stay out of it however they swing (the theatre's keep-out, its pieces
    treated as flat; a lamp is round)."""
    o, lay, cafe = _scene(swing=PT.RANGES["swing"][1], twist=PT.RANGES["twist"][1])
    kx, ky = PT.KEEP_OUT
    for part in lay.statics:
        if part.name in ("floor", "base", "sky", "border", "clear", "flies"):      # under her, behind, over her
            continue
        V = _points(part)
        inside = (np.abs(V[:, 0]) < kx) & (np.abs(V[:, 1]) < ky) & (V[:, 2] > 0.01)
        assert not inside.any(), part.name
    assert PC.border_point(*PC.border_uv(0.0, 0.0))[2] > 3.0
    t = np.arange(0.0, 31.2, 1.0 / 60.0)
    ch, _hidden, _flips = PC.motion(o, lay, cafe, t)
    val = {(c.object, c.path, c.index): c.values for c in ch}
    for h in lay.hangs:                                     # at the widest swing and twist the options allow
        r = 0.5 * h.size[0]
        reach = (h.length + h.size[1]) * np.sin(np.abs(val[(h.name + "_hang", "rotation_euler", 0)])).max()
        side = np.abs(val[(h.name + "_hang", "rotation_euler", 1)]).max()
        assert abs(h.anchor[1]) - r - reach >= ky or abs(h.anchor[0]) - r - h.size[1] * math.sin(side) >= kx
    o, lay, cafe = _scene()                                 # and by default they swing a little
    ch, _hidden, _flips = PC.motion(o, lay, cafe, t)
    for c in ch:
        if c.object.endswith("_hang") and c.path == "rotation_euler":
            assert 0.3 < math.degrees(np.abs(c.values).max()) < 3.0 or c.index == 0


def test_a_close_camera_sees_sky_behind_her_head_and_an_old_taped_rip_beside_it():
    """From 1.1 m in front of her face, every line of sight past her head and her raised fists (|x| < 0.15, from her
    chin at 1.0 to 1.7) goes through the window, over the skyline where it crosses it; the old rip taped with an X on
    the window's sky is seen well inside the window, over the skyline, upper right of her head and clear of that
    corridor."""
    _o, lay, _c = _scene()
    city = [p.outline for p in [s for s in lay.statics if s.name == "skyline"][0].plates if p.frame is None]
    hw, sill, spring, rise = PC.WINDOW
    win = PC.window_outline(hw, sill, spring, rise)
    cam = np.array(PC.CLOSE_EYE)

    def over_city(at_city):
        under = [P_[np.abs(P_[:, 0] - at_city[0]) < 0.03, 1] for P_ in city]
        return at_city[2] > max([float(u.max()) for u in under if len(u)], default=0.0)
    for x in np.linspace(-0.15, 0.15, 5):
        for z in np.linspace(1.0, 1.7, 8):
            d = np.array([x, 0.0, z]) - cam
            at_wall = cam + d * (PC.WALL_Y - cam[1]) / d[1]
            assert PT.inside(np.array([[at_wall[0], at_wall[2]]]), win)[0], (x, z)
            assert over_city(cam + d * (PC.SKYLINE_Y - cam[1]) / d[1]), (x, z)
    rip = [s for s in lay.statics if s.name == "sky_rip"][0]
    assert sorted(p.role for p in rip.plates) == ["crack", "tape", "tape", "torn"]          # an X of two strips
    X = np.concatenate([np.stack([p.outline[:, 0], np.full(len(p.outline), p.y), p.outline[:, 1]], 1)
                        for p in rip.plates])
    assert (X[:, 1] > PC.SKYLINE_Y + 0.5).all() and (X[:, 1] < 2.35).all()        # in front of the backdrop
    d = X - cam
    at_wall = (cam + d * ((PC.WALL_Y - cam[1]) / d[:, 1])[:, None])[:, [0, 2]]
    assert PT.inside(at_wall, win).all() and PT.edge_distance(at_wall, win).min() > 0.04
    at_her = cam + d * (-cam[1] / d[:, 1])[:, None]
    assert at_her[:, 0].min() > 0.15 + 0.02 and (at_her[:, 2] > 1.35).all() and (at_her[:, 2] < 1.7).all()
    assert all(over_city(cam + dd * (PC.SKYLINE_Y - cam[1]) / dd[1]) for dd in d[::7])


def test_the_hole_bursts_open_and_the_tape_lands_on_their_times():
    """On the step held on `hole`'s frame the patch is gone and the flaps are open (and not a step before); the tape
    strip is flat on the border on the step held on `slap`'s, in the hand a moment before, and nowhere earlier."""
    o, lay, cafe = _scene()
    t = np.array([_held(o, 29.4) - 2.0 / 12.0, _held(o, 29.4), _held(o, 30.3) - 0.3, _held(o, 30.3) - 0.1,
                  _held(o, 30.3), 31.1])
    _ch, hidden, flips = PC.motion(o, lay, cafe, t)
    opened = PC.hole_open(o, t)
    assert not hidden["patch"][0] and hidden["flaps"][0] and opened[0] == 0.0
    assert hidden["patch"][1] and not hidden["flaps"][1] and opened[1] >= 1.0
    p, gone = PC.slap_place(o, cafe, t)
    assert gone[2] and not gone[3] and not gone[4]
    assert np.linalg.norm(p[4] - np.array(cafe.slap_rest)) < 1e-9 and np.linalg.norm(p[3] - p[4]) > 0.05
    _o2, lay2, cafe2 = _scene(cafe=True)
    _ch, hidden2, flips2 = PC.motion(_o2, lay2, cafe2, t)
    assert not hidden2["patch"].any() and hidden2["flaps"].all() and hidden2["slap"].all()
    assert len(flips2["flaps"][0]) == 1


def test_the_hole_is_over_her_large_enough_and_shows_clear_sky():
    """The hole is above her and a little behind, wider than her shoulders with her arms crossed and long enough to
    drop through on the border's slope, and from straight under it the clear card fills it."""
    _o, lay, cafe = _scene()
    u0, v0 = PC.border_uv(*PC.HOLE[:2])
    H = cafe.hole
    c = PC.border_point(u0, v0)
    assert abs(c[0]) < 0.15 and 0.0 < c[1] < 0.4 and c[2] > 2.9
    assert np.ptp(H[:, 0]) > 0.75 and np.ptp(H[:, 1]) * math.cos(PC.border_angle()) > 0.6
    clear = [p for p in lay.statics if p.name == "clear"][0].plates[0]
    assert PT.inside(H, clear.outline).all()


def test_every_cafe_piece_is_a_closed_outward_solid():
    """Cut, filled and rolled, the flaps at every openness too: an open mesh shows its inside in a render. The hole's
    pieces for other seeds too (its tear is drawn anew for each)."""
    _o, lay, cafe = _scene()
    for part in [p for p in lay.statics if p.name not in ("floor", "base", "sky")] + [cafe.patch, cafe.slap] + \
            [h.part for h in lay.hangs]:
        m = P.part_mesh(part, 3)
        assert m.is_closed() and m.volume() > 0, part.name
    m, positions = P.flaps_mesh(cafe, 3)
    for o in (0.0, 0.5, 1.12):
        mo = P.S.Mesh(positions(o), m.Q, m.T, m.Qm, m.Tm)
        assert mo.is_closed() and mo.volume() > 0, o
    shut, opened = positions(0.0), positions(1.0)
    assert np.allclose(shut[-200:], opened[-200:])                             # the rim stays
    assert opened[:, 2].min() < shut[:, 2].min() - 0.12                        # the flaps hang down into the room
    for seed in (2, 4):
        _o, lay, cafe = _scene(seed=seed)
        for part in [p for p in lay.statics if p.name in ("border", "sky_rip")] + [cafe.patch, cafe.slap]:
            m = P.part_mesh(part, seed)
            assert m.is_closed() and m.volume() > 0, (seed, part.name)
        m, positions = P.flaps_mesh(cafe, seed)
        for o in (0.0, 1.12):
            mo = P.S.Mesh(positions(o), m.Q, m.T, m.Qm, m.Tm)
            assert mo.is_closed() and mo.volume() > 0, (seed, o)


def test_the_hole_is_a_ragged_rip_with_flaps_of_different_sizes():
    """For several seeds: the torn outline, the white rim's edge and every flap are simple polygons (a crossing folds
    the mesh); the outline is ragged (a sixth longer than the ellipse it is torn round) with two or more bites running
    4 cm or more on into the paper; the rim is 1 to 2 cm of white fibres all round outside it; the flaps lie inside
    the hole and apart, long tongues over 0.17 m and short flaps under 0.12 m, on hinges of different widths; folded
    down into the room the long tongues hang within 20 degrees of as plumb as their hinge (on the sloping border) lets
    them, the short ones stick out stiffer."""
    c = np.array(PC.border_uv(*PC.HOLE[:2]))
    a, b = PC.HOLE[2], PC.HOLE[3]
    ellipse = math.pi * (3.0 * (a + b) - math.sqrt((3.0 * a + b) * (a + 3.0 * b)))
    for seed in (1, 2, 3, 4):
        outline, flaps = PC.hole_drawing(seed)
        rim = PC.rim_drawing(seed)
        assert not PCL.crossings(outline) and not PCL.crossings(rim), seed
        assert np.linalg.norm(np.diff(np.vstack([outline, outline[:1]]), axis=0), axis=1).sum() > 1.16 * ellipse
        d = outline - c
        r = np.hypot(*d.T)[np.argsort(np.arctan2(d[:, 1] / b, d[:, 0] / a))]
        local = np.array([np.median(np.take(r, range(i - 40, i + 40), mode="wrap")) for i in range(len(r))])
        bite = r - local > 0.04
        assert np.sum(bite & ~np.roll(bite, 1)) >= 2, seed
        assert not PT.inside(rim, outline).any()
        gap = PT.edge_distance(rim, outline)
        assert gap.min() > 0.009 and gap.max() < 0.021, seed
        lengths, widths = [], []
        for k, (F, (A, B), deg) in enumerate(flaps):
            assert not PCL.crossings(F), (seed, k)
            assert PT.inside(F + 0.004 * (c - F) / np.linalg.norm(c - F, axis=1)[:, None], outline).all(), (seed, k)
            for j, (G, _h, _d) in enumerate(flaps[:k]):
                assert not PT.inside(F, G).any() and not PT.inside(G, F).any(), (seed, j, k)
            e = (B - A) / np.linalg.norm(B - A)
            nin = np.array([-e[1], e[0]]) if (c - A) @ np.array([-e[1], e[0]]) > 0.0 else np.array([e[1], -e[0]])
            lengths.append(np.abs((F - A) @ nin).max())
            widths.append(np.linalg.norm(B - A))
            R = np.asarray(PC.border_frame()[0])
            way, axis = R @ np.array([nin[0], 0.0, nin[1]]), R @ np.array([e[0], 0.0, e[1]])
            hang = way * math.cos(math.radians(deg)) + PC.border_normal() * math.sin(math.radians(deg))
            off_plumb = math.degrees(math.acos(min(1.0, -hang[2]))) - math.degrees(math.asin(abs(axis[2])))
            assert 25.0 < deg < 120.0 and (off_plumb < 20.0 or lengths[-1] < 0.12), (seed, k, deg, off_plumb)
        assert sum(v > 0.17 for v in lengths) >= 2 and sum(v < 0.12 for v in lengths) >= 2, (seed, lengths)
        assert len(flaps) >= 6 and np.ptp(widths) > 0.05, seed


def _first_hits(eye, targets, surfaces):
    """Which of `surfaces` ({name: (vertices, Mesh)}) each ray from `eye` through a target point meets first."""
    eye = np.asarray(eye, float)
    tris = {}
    for name, (V, m) in surfaces.items():
        T = np.concatenate([m.T, m.Q[:, [0, 1, 2]], m.Q[:, [0, 2, 3]]])
        tris[name] = (V[T[:, 0]], V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    out = []
    for q in np.asarray(targets, float):
        ray = (q - eye) / np.linalg.norm(q - eye)
        best, who = np.inf, None
        for name, (A, e1, e2) in tris.items():
            p = np.cross(ray, e2)
            det = (e1 * p).sum(1)
            inv = 1.0 / np.where(np.abs(det) > 1e-12, det, np.inf)
            s = eye - A
            u = (s * p).sum(1) * inv
            qq = np.cross(s, e1)
            v = (qq @ ray) * inv
            t = (e2 * qq).sum(1) * inv
            hit = (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t > 1e-6)
            if hit.any() and t[hit].min() < best:
                best, who = t[hit].min(), name
        out.append(who)
    return out


def test_the_landed_tape_lies_across_the_rip_seen_whole():
    """The strip of masking tape slapped across the hole lands parallel to the border 1 cm under its face (its world
    matrix on_border's frame at its middle), reaches a good 20 cm past the torn rim on both sides, no flap hinged in
    its way; and from the camera inside under the hole and from the one in front of the stage every point of it is
    seen, nothing in front of it (no flap, no rim, no sky paper)."""
    o, lay, cafe = _scene()
    t = np.array([_held(o, 30.3) + 0.02])
    p, gone = PC.slap_place(o, cafe, t)
    assert not gone[0] and cafe.slap.plates[0].role == "tape"
    um, vm = np.array(PC.border_uv(*PC.HOLE[:2])) + np.array(PC.SLAP_AT[:2])
    assert np.allclose(cafe.slap_R, PC.on_border(um, vm, PC.SLAP[2])[0])
    assert np.allclose(p[0], PC.border_point(um, vm, PC.SLAP_AT[2]))
    tape = P.part_mesh(cafe.slap, 3)
    R = np.asarray(cafe.slap_R)
    V = tape.V @ R.T + p[0]
    below = (V - np.asarray(PC.border_frame()[1])) @ PC.border_normal()
    assert below.min() > 0.0095 and below.max() < 0.0115
    b = math.radians(PC.SLAP[2])
    along, across = np.array([math.cos(b), math.sin(b)]), np.array([-math.sin(b), math.cos(b)])
    rim = PC.rim_drawing(1) - (um, vm)
    reach = rim[np.abs(rim @ across) < 0.5 * PC.SLAP[1]] @ along
    assert 0.5 * PC.SLAP[0] - reach.max() > 0.2 and 0.5 * PC.SLAP[0] + reach.min() > 0.2
    for f in cafe.flaps:
        assert (np.abs((np.array(f.hinge) - (um, vm)) @ across) > 0.5 * PC.SLAP[1] + 0.05).all()
    fm, positions = P.flaps_mesh(cafe, 3)
    border = P.part_mesh([s for s in lay.statics if s.name == "border"][0], 3)
    surfaces = {"tape": (V, tape), "flaps": (positions(float(PC.hole_open(o, t)[0])), fm), "border": (border.V, border)}
    n = R @ np.array([0.0, -1.0, 0.0])
    s, w = np.meshgrid(np.linspace(-0.5 * PC.SLAP[0] + 0.03, 0.5 * PC.SLAP[0] - 0.03, 33), (-0.04, 0.0, 0.04))
    face = p[0] + s.reshape(-1, 1) * (R @ (1.0, 0.0, 0.0)) + w.reshape(-1, 1) * (R @ (0.0, 0.0, 1.0)) + 0.0013 * n
    for eye in ([0.25, -1.25, 0.95], [0.0, -3.6, 1.5]):
        first = _first_hits(eye, face, surfaces)
        assert all(f == "tape" for f in first), (eye, [i for i, f in enumerate(first) if f != "tape"])
