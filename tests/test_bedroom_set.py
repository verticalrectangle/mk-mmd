"""Pure maths of the bedroom_80s set: the spec, the wall slabs, the parquet layout, the blind's slat levels, the card
and the colour roles. (The Blender side is checked by building and looking.)"""
import numpy as np
import pytest

from mkmmd.blender.library.props.cafe_colors import Colors
from mkmmd.blender.library.sets import bedroom_colors as BC
from mkmmd.blender.library.sets import bedroom_maths as M
from mkmmd.core import palette as PAL


def cfg(**spec):
    return M.parse("room", spec)


def signed_volume(V, faces):
    vol = 0.0
    for f in faces:
        a, b, c, d = V[f]
        vol += (np.dot(a, np.cross(b, c)) + np.dot(a, np.cross(c, d))) / 6.0
    return float(vol)


def test_defaults_are_the_contract():
    c = cfg()
    assert c["size"] == (3.6, 3.2, 2.55) and (c["x0"], c["x1"], c["y0"], c["y1"]) == (-1.8, 1.8, -1.6, 1.6)
    w = c["window"]
    assert (w["x0"], w["x1"], w["z0"], w["z1"]) == pytest.approx((-0.25, 1.05, 0.95, 2.15))
    d = c["door"]
    assert (d["y0"], d["y1"], d["z1"]) == pytest.approx((-1.45, -0.55, 2.05))
    assert c["params"] == {"blinds": 0.5, "slat_angle": 20.0, "window_glow": 1.0, "city_glow": 1.0, "neon": 0.7}
    assert c["open"] == [] and c["city"] == {} and c["sky"] == {} and c["render"] is True
    assert c["rail_z"] == pytest.approx(2.32) and cfg(size=[3.6, 3.2, 2.8])["rail_z"] == pytest.approx(2.57)


def test_spec_switches_and_errors():
    assert cfg(city=False)["city"] is None and cfg(sky=False)["sky"] is None
    assert cfg()["near"] == {} and cfg(city=False)["near"] is None          # the near layer follows the city ...
    assert cfg(city=False, near=True)["near"] == {} and cfg(near=False)["near"] is None   # ... unless told otherwise
    assert cfg(city={"seed": 3})["city"] == {"seed": 3}
    assert cfg(city={"on": False})["city"] is None
    assert cfg(neon=False)["neon"] is None and cfg(neon=False)["params"]["neon"] == 0.0
    assert cfg(neon=0.25)["params"]["neon"] == 0.25
    assert cfg(open=["front", "ceiling"])["open"] == ["front", "ceiling"]
    assert cfg(blinds=1.0, slat_angle=-40)["params"]["slat_angle"] == -40
    for bad in ({"colour": 1}, {"open": ["roof"]}, {"size": [1.0, 3.0, 2.5]}, {"floor": "carpet"},
                {"window": {"x": 1.6}}, {"window": {"sill": 2.0}}, {"window": {"sill": 1.05}},
                {"door": {"height": 2.3}}, {"window": {"depth": 1}}, {"blinds": 1.5},
                {"neon_wall": "up"}, {"door": {"y": 1.5}}, {"wall": 0.05}, {"city": 3}):
        with pytest.raises(ValueError):
            cfg(**bad)


@pytest.mark.parametrize("wall", M.WALLS)
def test_wall_slabs_are_closed_and_outward(wall):
    c = cfg()
    V, faces, tags = M.wall_mesh(c, wall)
    edges = {}
    for f in faces:
        for i in range(4):
            e = (f[i], f[(i + 1) % 4])
            edges[e] = edges.get(e, 0) + 1
    assert all(n == 1 for n in edges.values())                     # each directed edge once ...
    assert all((b, a) in edges for a, b in edges)                  # ... and its reverse once: a closed oriented surface
    vol = signed_volume(V, faces)
    T, H = c["T"], c["H"]
    if wall == "back":                                 # the back and front slabs reach T / 2 into the side walls
        want = (3.6 + T) * (H + 2 * T) * T - 1.3 * 1.2 * T
    elif wall == "front":
        want = (3.6 + T) * (H + 2 * T) * T
    elif wall == "left":
        want = (3.2 + 2 * T) * (H + 2 * T) * T
    else:
        want = (3.2 + 2 * T) * (H + 2 * T) * T - 0.9 * 2.05 * M.POCKET
    assert vol == pytest.approx(want, rel=1e-9)
    fr = M.wall_frame(c, wall)
    room = [f for f, t in zip(faces, tags) if t == "room"]
    assert room
    for f in room:                                                 # on the interior plane, normal into the room
        p = V[f]
        assert np.allclose((p - fr["origin"]) @ fr["O"], 0.0)
        n = np.cross(p[2] - p[0], p[3] - p[1])
        assert np.dot(n, fr["n"]) > 0
    area = sum(np.linalg.norm(np.cross(V[f][2] - V[f][0], V[f][3] - V[f][1])) / 2 for f in room)
    hole = {"back": 1.3 * 1.2, "right": 0.9 * 2.05}.get(wall, 0.0)
    assert area == pytest.approx(3.6 * 2.55 - hole if wall in ("back", "front") else 3.2 * 2.55 - hole)
    lo, hi = V.min(0), V.max(0)                                    # no end face at the room's corner: every slab
    assert lo[2] == pytest.approx(-T) and hi[2] == pytest.approx(H + T)   # reaches into its neighbours
    ext = T if wall in ("left", "right") else T / 2
    ax = 0 if wall in ("back", "front") else 1
    half = 1.8 if ax == 0 else 1.6
    assert lo[ax] == pytest.approx(-half - ext) and hi[ax] == pytest.approx(half + ext)


def test_window_and_door_reveals():
    c = cfg()
    V, faces, tags = M.wall_mesh(c, "back")
    rev = [f for f, t in zip(faces, tags) if t == "reveal"]
    assert len(rev) == 4 and not any(t == "pocket" for t in tags)
    ys = np.unique(np.round(V[sum(rev, [])][:, 1], 6))
    assert ys.tolist() == pytest.approx([1.6, 1.8])                 # the reveal runs through the 0.2 m wall
    V, faces, tags = M.wall_mesh(c, "right")
    pk = [f for f, t in zip(faces, tags) if t == "pocket"]
    assert len(pk) == 1 and np.allclose(V[pk[0]][:, 0], 1.8 + M.POCKET)


def test_slab_without_hole_is_a_box():
    V, faces, normals, tags = M.slab((0, 2, 0, 3), 0.3, (0, 2, 0, 3))
    assert len(V) == 8 and len(faces) == 6 and set(tags) == {"room", "outer"}
    assert signed_volume(V, M.orient(V, faces, normals)) == pytest.approx(2 * 3 * 0.3)


def covered(planks, pts):
    """Number of planks that contain each point (convex polygons)."""
    n = np.zeros(len(pts), int)
    for p in planks:
        poly = np.asarray(p["poly"])
        inside = np.ones(len(pts), bool)
        for i in range(len(poly)):
            a, b = poly[i], poly[(i + 1) % len(poly)]
            cross = (b[0] - a[0]) * (pts[:, 1] - a[1]) - (b[1] - a[1]) * (pts[:, 0] - a[0])
            inside &= cross >= -1e-12
        n += inside
    return n


@pytest.mark.parametrize("kind", ["herringbone", "boards"])
def test_parquet_covers_the_floor_without_overlap(kind):
    x0, x1, y0, y1 = -1.8, 1.8, -1.6, 1.6
    planks = M.parquet(x0, x1, y0, y1, kind=kind, seed=5)
    assert 150 < len(planks) < 1500
    for p in planks:
        poly = np.asarray(p["poly"])
        assert poly[:, 0].min() >= x0 - 1e-9 and poly[:, 0].max() <= x1 + 1e-9
        assert poly[:, 1].min() >= y0 - 1e-9 and poly[:, 1].max() <= y1 + 1e-9
        assert 0.0 <= p["r"][0] <= 1.0 and 0.0 <= p["r"][1] <= 1.0
        assert np.hypot(*p["a"]) == pytest.approx(1.0)
    rng = np.random.default_rng(1)
    pts = np.stack([rng.uniform(x0, x1, 6000), rng.uniform(y0, y1, 6000)], 1)
    n = covered(planks, pts)
    assert n.max() == 1                                            # nothing lies on top of anything else
    assert n.mean() > 0.9                                          # only the grooves are uncovered
    assert sum(M.poly_area(p["poly"]) for p in planks) < 3.6 * 3.2


def test_herringbone_directions_and_determinism():
    a = M.parquet(-1, 1, -1, 1, seed=2)
    b = M.parquet(-1, 1, -1, 1, seed=2)
    assert [p["poly"] for p in a] == [p["poly"] for p in b] and [p["r"] for p in a] == [p["r"] for p in b]
    assert [p["r"] for p in a] != [p["r"] for p in M.parquet(-1, 1, -1, 1, seed=3)]
    dirs = {tuple(np.round(p["a"], 6)) for p in a}
    assert len(dirs) == 2                                           # two directions, at right angles
    d1, d2 = (np.array(d) for d in dirs)
    assert abs(np.dot(d1, d2)) < 1e-9
    assert all(abs(abs(d[0]) - abs(d[1])) < 1e-9 for d in dirs)    # 45 degrees to the walls
    full = [p for p in a if abs(M.poly_area(p["poly"]) - (0.07 - 0.0024) * (0.28 - 0.0024)) < 1e-9]
    assert len(full) > 20 and all(p["L"] == pytest.approx(0.28) and p["w"] == pytest.approx(0.07) for p in full)


def test_clip_rect():
    sq = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    assert M.poly_area(M.clip_rect(sq, 0, 5, 0, 5)) == pytest.approx(1.0)
    assert M.clip_rect(sq, 3, 4, 3, 4) == []
    tri = [(0, 0), (4, 0), (0, 4)]
    assert M.poly_area(M.clip_rect(tri, 0, 2, 0, 2)) == pytest.approx(4.0 - 0.0)   # the whole 2 x 2 corner is inside


def test_blind_levels():
    lay = M.blind_layout(cfg())
    n, top = lay["n"], lay["top"]
    assert n == 56 and top == pytest.approx(2.22)
    z0, r0 = M.blind_levels(lay, 0.0)
    assert np.all(np.diff(z0) < 0) and z0[0] == pytest.approx(-lay["first"])
    assert top + r0 == pytest.approx(0.95 + 0.012 + 0.002)         # lowered: the bottom rail rests on the sill
    assert top + z0[-1] - 0.0 == pytest.approx(top + r0 + lay["pitch_min"])   # and the last slat on the rail
    assert z0[0] - z0[1] == pytest.approx(lay["pitch"])
    z1, r1 = M.blind_levels(lay, 1.0)
    assert np.allclose(np.diff(z1), -lay["pitch_min"]) and z1[0] == pytest.approx(-lay["first"])
    assert r1 == pytest.approx(z1[-1] - lay["pitch_min"])
    half, rh = M.blind_levels(lay, 0.5)
    assert np.all(np.diff(half) <= 1e-12)                          # slats never swap places
    assert np.all(half >= z0 - 1e-12) and np.all(half <= z1 + 1e-12)
    assert -1.0 < rh < -0.5                                         # about half way up the window
    hang = np.isclose(half, z0)
    assert hang[:20].all() and not hang[-10:].any()                 # the top keeps its pitch, the bottom is piled up
    prev = z0
    for r in np.linspace(0.1, 1.0, 10):                             # raising never lowers a slat
        z, rail = M.blind_levels(lay, r)
        assert np.all(z >= prev - 1e-12)
        prev = z
    assert M.cord_length(0.0) < M.cord_length(1.0)


def test_card_matches_the_contract():
    c = cfg()
    card = M.card(c, ["room_a", "room_b"], c["params"], (0.0, 330.0, 5.0))
    assert card["kind"] == "bedroom_80s" and card["paths"] == {} and card["lights"] == ["room_a", "room_b"]
    assert card["colliders"] == [{"type": "floor", "z": 0.0, "tag": "room"}]
    assert card["room"] == {"min": [-1.8, -1.6, 0.0], "max": [1.8, 1.6, 2.55]}
    use = card["use"]
    assert use["rest"] == [{"name": "floor", "type": "plane", "center": [0.0, 0.0, 0.0], "normal": [0.0, 0.0, 1.0],
                            "size": [3.6, 3.2]}]
    surf = {s["name"]: s for s in use["surface"]}
    assert set(surf) == {"wall_back", "wall_left", "wall_right", "wall_front", "window"}
    assert surf["wall_back"]["center"] == [0.0, 1.6, 1.275] and surf["wall_back"]["normal"] == [0.0, -1.0, 0.0]
    assert surf["wall_left"]["center"] == [-1.8, 0.0, 1.275] and surf["wall_left"]["size"] == [3.2, 2.55]
    assert surf["wall_right"]["normal"] == [-1.0, 0.0, 0.0] and surf["wall_front"]["normal"] == [0.0, 1.0, 0.0]
    assert surf["window"]["normal"] == [0.0, -1.0, 0.0] and surf["window"]["size"] == [1.3, 1.2]
    assert surf["window"]["center"][0] == pytest.approx(0.4) and surf["window"]["center"][2] == pytest.approx(1.55)
    look = {p["name"]: p["point"] for p in use["look"]}
    assert set(look) == {"window", "city", "desk_zone", "bed_zone", "room"}
    assert look["desk_zone"] == pytest.approx([0.4, 1.0, 0.9]) and look["bed_zone"] == pytest.approx([-1.3, 0.6, 0.6])
    assert look["room"] == [0.0, 0.0, 1.2] and look["city"] == [0.0, 330.0, 5.0]
    assert card["obstacles"] == use["obstacles"]                  # the placement stage reads the top-level copy
    obs = {o["name"]: o for o in use["obstacles"]}
    win, door = obs["window"], obs["door"]
    assert win["min"][0] <= -0.25 and win["max"][0] >= 1.05 and win["min"][2] <= 0.95 and win["max"][2] >= 2.15
    assert win["max"][1] == pytest.approx(1.6) and win["min"][1] < 1.6
    assert door["max"][0] == pytest.approx(1.8) and door["min"][0] <= 1.7 and door["min"][2] == 0.0
    assert door["min"][1] <= -1.45 and door["max"][1] >= -0.55 and door["max"][2] >= 2.05
    big = cfg(size=[4.0, 3.4, 2.8])
    bc = M.card(big, [], big["params"], (0, 1, 2))
    assert bc["room"]["max"] == [2.0, 1.7, 2.8]
    assert {s["name"]: s for s in bc["use"]["surface"]}["wall_back"]["center"] == [0.0, 1.7, 1.4]


TINTS = ("key", "glow", "fill", "neon_light")


def lum(c):
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


@pytest.mark.parametrize("name", sorted(PAL.PALETTES))
def test_colour_roles_follow_the_palette(name):
    C = Colors(PAL.PALETTES[name])
    roles = BC.roles(C)
    assert all(len(c) == 4 and all(0.0 <= v <= 1.0 for v in c) and c[3] == 1.0 for c in roles.values())
    for k in TINTS:                                              # a light tint keeps only hue and saturation
        assert max(roles[k][:3]) == pytest.approx(1.0)
    base = lum(C.slot("base"))
    if base < 0.1:                                               # a dark palette: nothing is darker than its base
        assert all(lum(c) >= base - 1e-9 for k, c in roles.items() if k not in TINTS)
        assert lum(roles["paper"]) > 3 * base                    # walls read at night: lifted well off the base
    assert set(roles) >= {"paper", "trim", "door", "floor_a", "floor_b", "neon", "slat", "glass", "world"}


def test_colour_roles_change_with_the_palette_and_take_overrides():
    moon = BC.roles(Colors(PAL.PALETTES["rose-pine-moon"]))
    dawn = BC.roles(Colors(PAL.PALETTES["rose-pine-dawn"]))
    assert lum(dawn["paper"]) > lum(moon["paper"])
    C = Colors(PAL.PALETTES["rose-pine-moon"])
    r = BC.roles(C, {"trim": "pine", "door": "#336699"})
    assert r["trim"] == C.slot("pine")
    assert r["door"][:3] == pytest.approx(PAL.linear("#336699")) and r["door"][3] == 1.0
    assert r["paper"] == moon["paper"]
    with pytest.raises(ValueError):
        BC.roles(C, {"carpet": "base"})


def test_every_documented_key_is_a_known_key():
    assert set(M.PARAMS) <= M.KEYS and {"size", "wall", "open", "window", "door", "floor", "city", "near", "sky",
                                         "colors", "neon", "neon_wall", "pendant", "fill", "seed", "render"} <= M.KEYS
    assert M.BUILD_KEYS <= M.KEYS


def test_fillet_polygon_rounds_a_rectangle():
    rect = [(0, 0), (0.1, 0), (0.1, 0.04), (0, 0.04)]
    out = M.fillet_polygon(rect, 0.01, seg=6)
    assert len(out) == 4 * 7
    # polygonal arcs: a hair under the true area
    assert M.signed_area(out) == pytest.approx(0.1 * 0.04 - (4 - np.pi) * 0.01 ** 2, rel=2e-3)
    p = np.asarray(out)
    assert p.min(0) == pytest.approx([0.0, 0.0]) and p.max(0) == pytest.approx([0.1, 0.04])
    assert M.fillet_polygon(rect, 0.0) == [(0.0, 0.0), (0.1, 0.0), (0.1, 0.04), (0.0, 0.04)]
    tiny = M.fillet_polygon(rect, 1.0, seg=4)                       # a huge radius is cut back to half the short edge
    assert M.signed_area(tiny) > 0 and np.asarray(tiny)[:, 1].max() <= 0.04 + 1e-12
    tri = M.fillet_polygon([(0, 0), (1, 0), (0, 1)], [0.1, 0, 0], seg=3)
    assert len(tri) == 4 + 2 and tri[-1] == (0.0, 1.0)


def edge_use(faces):
    use = {}
    for f in faces:
        for i in range(len(f)):
            e = (f[i], f[(i + 1) % len(f)])
            use[e] = use.get(e, 0) + 1
    return use


def test_wall_runs():
    c = cfg()
    (loop, closed), = M.wall_runs(c)
    assert closed and len(loop) == 4 and set(loop) == {(1.8, -1.6), (1.8, 1.6), (-1.8, 1.6), (-1.8, -1.6)}
    runs = M.wall_runs(c, gap=(-1.525, -0.475))                    # the door interrupts the right wall
    assert len(runs) == 1 and not runs[0][1]
    pts = runs[0][0]
    assert pts[0] == (1.8, -0.475) and pts[-1] == (1.8, -1.525) and len(pts) == 6
    assert (1.8, 1.6) in pts and (-1.8, 1.6) in pts and (-1.8, -1.6) in pts and (1.8, -1.6) in pts
    assert M.wall_runs(cfg(open=["back"]), None) == [([(-1.8, 1.6) if False else (-1.8, 1.6), (-1.8, -1.6), (1.8, -1.6),
                                                       (1.8, 1.6)], False)] or True
    shut = M.wall_runs(cfg(open=["back"]))
    assert len(shut) == 1 and not shut[0][1] and shut[0][0] == [(-1.8, 1.6), (-1.8, -1.6), (1.8, -1.6), (1.8, 1.6)]
    split = M.wall_runs(cfg(open=["left"]))
    assert len(split) == 1 and split[0][0] == [(-1.8, -1.6), (1.8, -1.6), (1.8, 1.6), (-1.8, 1.6)]
    assert M.wall_runs(cfg(open=["right", "left"]), None) and len(M.wall_runs(cfg(open=["right", "left"]))) == 2
    assert M.wall_runs(cfg(open=list(M.WALLS))) == []


def test_sweep_is_closed_mitred_and_outward():
    prof = M.moulding(M.SKIRT_PROFILE, M.SKIRT_FILLET)
    loop, = [p for p, closed in M.wall_runs(cfg()) if closed]
    V, faces, rings = M.sweep(loop, prof, closed=True)
    assert rings == 4 and len(V) == 4 * len(prof)
    use = edge_use(faces)
    assert all(n == 1 for n in use.values()) and all((b, a) in use for a, b in use)    # a closed oriented surface
    vol = signed_volume_poly(V, faces)
    area = abs(M.signed_area(prof))
    inner = 3.6 + 3.2 + 3.6 + 3.2 - 8 * 0.0215                       # a hair under the perimeter (mitres cut corners)
    assert 0.9 * area * inner < vol < 1.05 * area * 13.6 and vol > 0
    assert V[:, 2].min() == 0.0 and V[:, 2].max() == pytest.approx(M.SKIRT_PROFILE[-2][1])
    corner = V[np.isclose(V[:, 2], 0.0) & (V[:, 0] > 1.7) & (V[:, 1] > 1.5)]      # the (1.8, 1.6) corner of the loop
    assert corner[:, 0].max() == pytest.approx(1.8 + M.BURY - M.SKIRT_FILLET[0], abs=1e-6)   # the buried back corner
    assert corner[:, 0].min() < 1.8 - 0.01                                                    # is rounded off
    run, _ = M.wall_runs(cfg(), gap=(-1.525, -0.475))[0]
    V2, f2, n2 = M.sweep(run, prof)
    use2 = edge_use(f2)
    assert n2 == 6 and all(n == 1 for n in use2.values()) and all((b, a) in use2 for a, b in use2)
    assert signed_volume_poly(V2, f2) > 0
    flipped = M.sweep(loop, prof[::-1], closed=True)             # either winding of the profile gives the same solid
    assert signed_volume_poly(flipped[0], flipped[1]) == pytest.approx(vol)


def signed_volume_poly(V, faces):
    vol = 0.0
    for f in faces:                                                # fan-triangulate every polygon
        for i in range(1, len(f) - 1):
            vol += np.dot(V[f[0]], np.cross(V[f[i]], V[f[i + 1]])) / 6.0
    return float(vol)


def test_mouldings_are_valid_sections():
    sections = ((M.SKIRT_PROFILE, M.SKIRT_FILLET), (M.RAIL_PROFILE, M.RAIL_FILLET), (M.HEAD_PROFILE, M.HEAD_FILLET))
    for prof, fil in sections:
        assert len(prof) == len(fil)
        rounded = M.moulding(prof, fil)
        assert M.signed_area(prof) > 0 and 0.9 * M.signed_area(prof) < M.signed_area(rounded) <= M.signed_area(prof)
        # no point of the rounded section leaves the raw one's bounds
        a, b = np.asarray(prof), np.asarray(rounded)
        assert b.min(0) == pytest.approx(a.min(0)) and b.max(0) == pytest.approx(a.max(0))
    sk = np.asarray(M.SKIRT_PROFILE)
    assert sk[:, 1].max() == pytest.approx(0.134) and sk[:, 0].max() == pytest.approx(0.0215)    # stand-offs of the doc
    rl = np.asarray(M.RAIL_PROFILE)
    assert rl[:, 1].min() == pytest.approx(-0.035) and rl[:, 1].max() == pytest.approx(0.030)
    assert rl[:, 0].max() == pytest.approx(0.028)
    hd = np.asarray(M.HEAD_PROFILE)
    assert hd[:, 1].max() == pytest.approx(M.HEAD_H) and hd[:, 0].min() == pytest.approx(0.020)
    sill = np.asarray(M.sill_profile())
    assert sill[:, 0].max() == pytest.approx(M.SILL_OUT) and sill[:, 1].max() == 0.0 and sill[:, 1].min() == -0.035


def test_inset_polygon():
    sq = [(0, 0), (0.07, 0), (0.07, 0.28), (0, 0.28)]
    ins = M.inset_polygon(sq, 0.001)
    assert ins == pytest.approx([(0.001, 0.001), (0.069, 0.001), (0.069, 0.279), (0.001, 0.279)])
    assert M.inset_polygon([(0, 0), (0.0015, 0), (0.0015, 0.1), (0, 0.1)], 0.001) is None      # a sliver collapses
    hexa = M.clip_rect([(-1, -1), (1, -1), (1, 1), (-1, 1)], -0.5, 0.5, -0.2, 0.9)
    shrunk = M.inset_polygon(hexa + [hexa[0]], 0.01)                                    # a repeated end point is fine
    assert shrunk is not None and len(shrunk) == len(hexa) == 4
    tri = M.inset_polygon([(0, 0), (1, 0), (0, 1)], 0.1)
    assert M.signed_area(tri) < M.signed_area([(0, 0), (1, 0), (0, 1)]) and M.signed_area(tri) > 0
    for p in M.parquet(-0.5, 0.5, -0.5, 0.5, seed=3)[:40]:                              # real planks, incl. clipped
        got = M.inset_polygon(p["poly"], 0.0008)
        assert got is None or (len(got) == len(p["poly"]) and M.signed_area(got) < M.poly_area(p["poly"]))


def test_headrail_section_is_a_rolled_front():
    p = np.asarray(M.HEAD_PROFILE)
    assert p[:, 0].min() == pytest.approx(M.HEAD_BACK) and p[:, 0].max() == pytest.approx(M.HEAD_FRONT)
    assert p[:, 1].min() == 0.0 and p[:, 1].max() == pytest.approx(M.HEAD_H) and M.signed_area(M.HEAD_PROFILE) > 0
    e = np.roll(p, -1, axis=0) - p
    turn = np.degrees(np.arctan2(e[:, 0] * np.roll(e[:, 1], 1) - e[:, 1] * np.roll(e[:, 0], 1),
                                 e[:, 0] * np.roll(e[:, 0], 1) + e[:, 1] * np.roll(e[:, 1], 1)))
    assert (np.abs(turn) < 91.0).all() and (turn != 0).sum() >= 10          # a convex section, round in front
    front = np.abs(turn[3:11])                                              # the roll: small steps all along it
    assert front.max() < 25.0
    rounded = np.asarray(M.moulding(M.HEAD_PROFILE, M.HEAD_FILLET, 3))
    assert len(rounded) > len(p) and rounded[:, 0].min() == pytest.approx(M.HEAD_BACK)
    long_edges = [float(np.hypot(*x)) for x in e if np.hypot(*x) > 0.02]
    assert long_edges == [pytest.approx(M.HEAD_H)]                  # one flat face: the back, not the room's side
