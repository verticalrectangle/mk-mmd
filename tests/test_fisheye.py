"""Fisheye shot cameras (mkmmd/core/fisheye.py), bpy-free: the projection agrees with Cycles 4.2's own fisheye code (ported
below from kernel/camera/projection.h and the Blender sync's sensor fit, which renders of markers at known directions
confirmed to half a pixel), the lens puts the frame's corners or the circle's edge at fov / 2 in every output aspect, the
`fisheye` table and what it refuses are checked (mkmmd/core/shotspec.py), a fisheye's clip start counts distances in its
cone (mkmmd/core/clip.py), and the checks (mkmmd/checks/camera.py) and pixel maps (mkmmd/pixmap.py) see through it."""
import math

import numpy as np
import pytest

from mkmmd import pixmap as PM
from mkmmd.checks import CheckError
from mkmmd.checks import camera as CC
from mkmmd.core import clip as CL
from mkmmd.core import fisheye as FE
from mkmmd.core import shotspec as SP

SIZES = [(2048, 2048), (1920, 1080), (1080, 1920)]
SPECS = [{"fov": fov, "projection": p, "fill": fill} for fov in (120.0, 180.0, 220.0) for p in FE.PROJECTIONS
         for fill in FE.FILLS]


def cycles_direction(px, py, cam, size):
    """Cycles' ray for the centres of pixels (px, py) (from the top left) of a fisheye camera (`cam`: the camera data's
    fields): the panoramic viewplane [0, 1] moved by the shift, the sensor fitted as blender/camera.cpp does for
    FISHEYE_EQUISOLID and FISHEYE_LENS_POLYNOMIAL, fisheye_equisolid_to_direction / fisheye_lens_polynomial_to_direction,
    and the panoramic camera's axes (its x the view axis) turned to Blender's camera axes. Zero where the picture is
    black."""
    w, h = size
    u = (np.asarray(px, float) + 0.5) / w + cam["shift_x"]
    v = (h - np.asarray(py, float) - 0.5) / h + cam["shift_y"]
    fit, S = cam["sensor_fit"], cam["sensor_width"]
    horizontal = w > h if fit == "AUTO" else fit == "HORIZONTAL"
    if fit == "VERTICAL":
        S = cam["sensor_height"]
    sw, sh = (S, S * h / w) if horizontal else (S * w / h, S)
    uu, vv = (u - 0.5) * sw, (v - 0.5) * sh
    r = np.hypot(uu, vv)
    if cam["panorama_type"] == "FISHEYE_EQUISOLID":
        f, fov = cam["fisheye_lens"], cam["fisheye_fov"]
        black = r > 2.0 * f * math.sin(fov / 4.0)
        theta = 2.0 * np.arcsin(np.clip(r / (2.0 * f), -1.0, 1.0))
    else:
        theta = -(cam["fisheye_polynomial_k0"] + cam["fisheye_polynomial_k1"] * r + cam["fisheye_polynomial_k2"] * r ** 2
                  + cam["fisheye_polynomial_k3"] * r ** 3 + cam["fisheye_polynomial_k4"] * r ** 4)
        black = np.abs(theta) > cam["fisheye_fov"] / 2.0
    phi = np.arccos(np.where(r > 0, uu / np.where(r > 0, r, 1.0), 0.0))
    phi = np.where(vv < 0, -phi, phi)
    pano = np.stack([np.cos(theta), -np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta)], 1)
    D = np.stack([-pano[:, 1], pano[:, 2], -pano[:, 0]], 1)            # blender_camera_matrix: x forward -> -z
    D[black] = 0.0
    return D


def camera_fields(fe, size, shift=(0.0, 0.0), fit="AUTO", sensor=FE.SENSOR, sensor_h=24.0):
    cam = {"sensor_width": sensor, "sensor_height": sensor_h, "sensor_fit": fit, "shift_x": shift[0], "shift_y": shift[1],
           "fisheye_lens": 10.5, "fisheye_polynomial_k0": -1e-5, "fisheye_polynomial_k1": -0.02,
           "fisheye_polynomial_k2": 0.0, "fisheye_polynomial_k3": 0.0, "fisheye_polynomial_k4": 0.0}
    cam.update(FE.blender(fe, size, sensor))
    return cam


# ---------------------------------------------------------------- the lens and Blender's settings
@pytest.mark.parametrize("fe", SPECS)
@pytest.mark.parametrize("size", SIZES)
def test_the_frame_corners_or_the_circle_edge_sit_at_half_the_field_in_every_aspect(fe, size):
    f = FE.lens(fe, size)
    w, h = size
    per = FE.SENSOR / max(w, h)                                          # sensor mm per pixel
    edge = 0.5 * per * (math.hypot(w, h) if fe["fill"] == "frame" else min(w, h))
    assert FE.angle(edge, f, fe["projection"]) == pytest.approx(math.radians(fe["fov"]) / 2, abs=1e-12)
    lens = FE.Lens.of(fe, size)
    corner = lens.rays([0], [0])[0]                                      # the top left pixel's centre
    half = math.radians(fe["fov"]) / 2

    def off_axis(ray):
        return math.atan2(math.hypot(ray[0], ray[1]), -ray[2])
    if fe["fill"] == "frame":                                            # the corner shows, half a pixel inside the field
        assert np.linalg.norm(corner) == pytest.approx(1.0)
        inside = math.hypot(w / 2 - 0.5, h / 2 - 0.5) / lens.k
        assert off_axis(corner) == pytest.approx(FE.angle(inside, f, fe["projection"]), abs=1e-9) and off_axis(corner) < half
        assert lens.reach() == pytest.approx(half, abs=1e-12)
    else:
        assert not corner.any()                                           # black outside the circle
        if w >= h:                                                        # the short side's edge: on the circle's rim
            rim, past = lens.rays([w / 2 - 0.5], [0])[0], lens.rays([w / 2 - 0.5], [-1])[0]
        else:
            rim, past = lens.rays([0], [h / 2 - 0.5])[0], lens.rays([-1], [h / 2 - 0.5])[0]
        inside = (min(w, h) / 2 - 0.5) / lens.k
        assert off_axis(rim) == pytest.approx(FE.angle(inside, f, fe["projection"]), abs=1e-9) and off_axis(rim) < half
        assert not past.any()                                             # a pixel beyond it: black
        assert lens.reach() == pytest.approx(half, abs=1e-12)
    assert FE.blender(fe, size)["fisheye_lens"] < 100.0                   # Blender's hard limit on fisheye_lens


def test_blender_settings_use_the_lens_polynomial_for_equidistant_and_the_field_cuts_past_the_corners():
    eq = FE.blender({"fov": 180.0, "projection": "equisolid", "fill": "circle"}, (1080, 1920))
    assert eq["type"] == "PANO" and eq["panorama_type"] == "FISHEYE_EQUISOLID" and eq["fisheye_fov"] == pytest.approx(math.pi)
    assert eq["fisheye_lens"] == pytest.approx(0.5 * 36.0 * 1080 / 1920 / (2 * math.sin(math.pi / 4)))
    ed = FE.blender({"fov": 200.0, "projection": "equidistant", "fill": "frame"}, (1920, 1080))
    assert ed["panorama_type"] == "FISHEYE_LENS_POLYNOMIAL" and ed["fisheye_polynomial_k1"] == pytest.approx(-1 / ed["fisheye_lens"])
    assert ed["fisheye_polynomial_k0"] == ed["fisheye_polynomial_k2"] == ed["fisheye_polynomial_k4"] == 0.0
    assert ed["fisheye_fov"] == pytest.approx(math.radians(200.0) * FE.PAST)        # equidistant: r scales with theta
    full = FE.blender({"fov": 360.0, "projection": "equisolid", "fill": "frame"}, (2048, 2048))
    assert full["fisheye_fov"] == pytest.approx(2 * math.pi)                        # never past all round


@pytest.mark.parametrize("fe", SPECS)
@pytest.mark.parametrize("size", SIZES)
def test_the_projection_puts_each_cycles_ray_back_on_its_pixel(fe, size):
    cam = camera_fields(fe, size)
    lens = FE.from_blender(dict(cam, type="PANO"), size)
    rng = np.random.default_rng(5)
    px, py = rng.uniform(0, size[0] - 1, 4000), rng.uniform(0, size[1] - 1, 4000)
    D = cycles_direction(px, py, cam, size)
    shown = np.linalg.norm(D, axis=1) > 0
    assert shown.sum() > 1000
    x, y, dist = lens.pixels(D[shown] * 3.0)                              # 3 m out along each ray
    assert np.abs(x - (px[shown] + 0.5)).max() < 1e-6 and np.abs(y - (py[shown] + 0.5)).max() < 1e-6
    assert dist == pytest.approx(np.full(shown.sum(), 3.0))
    R = lens.rays(px, py)
    assert np.abs(R - D).max() < 1e-9                                     # and the rays are Cycles' rays, black included


def test_shifts_and_sensor_fits_follow_cycles_too():
    fe = {"fov": 190.0, "projection": "equisolid", "fill": "circle"}
    for size, fit, shift in (((1920, 1080), "AUTO", (0.1, -0.05)), ((1920, 1080), "VERTICAL", (0.0, 0.0)),
                             ((1080, 1920), "HORIZONTAL", (-0.03, 0.02))):
        cam = camera_fields(fe, size, shift=shift, fit=fit, sensor=30.0, sensor_h=20.0)
        lens = FE.from_blender(dict(cam, type="PANO"), size)
        px, py = np.meshgrid(np.arange(0, size[0], 37.0), np.arange(0, size[1], 41.0))
        D = cycles_direction(px.ravel(), py.ravel(), cam, size)
        shown = np.linalg.norm(D, axis=1) > 0
        x, y, _ = lens.pixels(D[shown])
        assert np.abs(x - px.ravel()[shown] - 0.5).max() < 1e-6 and np.abs(y - py.ravel()[shown] - 0.5).max() < 1e-6


def test_a_point_past_the_field_has_a_negative_distance_and_the_rest_a_positive_one():
    lens = FE.Lens.of({"fov": 180.0, "projection": "equisolid", "fill": "circle"}, (1000, 1000))
    P = np.array([[0.0, 0.0, -2.0], [2.0, 0.0, -0.01], [2.0, 0.0, 0.01], [0.0, 0.0, 2.0], [0.0, 0.0, 0.0]])
    _, d = lens.project(P)
    assert d[0] == pytest.approx(2.0) and d[1] > 0 and d[2] < 0 and d[3] == pytest.approx(-2.0) and d[4] == 0.0
    uv, _ = lens.project(P[:2])
    theta = math.atan2(2.0, 0.01)                                        # a hair inside the rim (90 degrees off the axis)
    assert uv[0] == pytest.approx([0.5, 0.5]) and uv[1, 0] == pytest.approx(0.5 + 0.5 * math.sin(theta / 2) / math.sin(
        math.pi / 4), abs=1e-12)


def test_the_scale_is_the_patch_facing_the_camera_and_equisolid_keeps_areas():
    size = (1200, 800)
    for proj in FE.PROJECTIONS:
        lens = FE.Lens.of({"fov": 200.0, "projection": proj, "fill": "frame"}, size)
        for theta in (0.0, 0.4, 1.2, 1.6):
            c = 2.5 * np.array([math.sin(theta) * math.cos(0.7), math.sin(theta) * math.sin(0.7), -math.cos(theta)])
            n = c / np.linalg.norm(c)
            a = np.cross(n, [0.0, 0.0, 1.0] if abs(n[2]) < 0.9 else [1.0, 0.0, 0.0])
            a /= np.linalg.norm(a)
            b = np.cross(n, a)
            e = 1e-4                                                     # a tiny square facing the camera
            x, y, _ = lens.pixels(np.array([c, c + e * a, c + e * b]))
            area = abs((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0])) / e ** 2
            want = math.sqrt(area) / size[1]
            assert lens.scale(c[None])[0] == pytest.approx(want, rel=1e-3), (proj, theta)
            if proj == "equisolid":
                assert want == pytest.approx(lens.f * lens.k / 2.5 / size[1], rel=1e-3)
            px_x, px_y = x[0] - 0.5, y[0] - 0.5
            assert lens.m_per_px(2.5, px_x, px_y) == pytest.approx(1.0 / (want * size[1]), rel=1e-3)


def test_the_margin_is_the_circle_where_the_circle_is_nearer_than_the_frame():
    lens = FE.Lens.of({"fov": 180.0, "projection": "equidistant", "fill": "circle"}, (1920, 1080))
    rim = lens.rim
    on_circle = np.array([[(960 + rim * math.cos(0.6)) / 1920, (540 + rim * math.sin(0.6)) / 1080]])
    assert lens.margin(on_circle)[0] == pytest.approx(0.0, abs=1e-9)        # well inside the frame, but on the circle
    assert lens.margin([[0.5, 0.5]])[0] == pytest.approx(0.5)               # the centre: half the short side to the rim
    full = FE.Lens.of({"fov": 180.0, "projection": "equidistant", "fill": "frame"}, (1920, 1080))
    assert full.margin([[0.02, 0.5]])[0] == pytest.approx(0.02)             # a full frame: the frame's margins


def test_blender_cameras_mk_does_not_project_are_refused_and_others_are_not_fisheyes():
    cam = camera_fields({"fov": 180.0, "projection": "equidistant", "fill": "frame"}, (100, 100))
    assert FE.from_blender(dict(cam, type="PERSP"), (100, 100)) is None
    with pytest.raises(FE.FisheyeError, match="EQUIRECTANGULAR"):
        FE.from_blender(dict(cam, type="PANO", panorama_type="EQUIRECTANGULAR"), (100, 100))
    with pytest.raises(FE.FisheyeError, match="k1 alone"):
        FE.from_blender(dict(cam, type="PANO", fisheye_polynomial_k2=1e-4), (100, 100))
    with pytest.raises(FE.FisheyeError, match="FISHEYE_EQUISOLID"):         # Blender's own (stretched) equidistant
        FE.from_blender(dict(cam, type="PANO", panorama_type="FISHEYE_EQUIDISTANT"), (100, 100))


# ---------------------------------------------------------------- the `fisheye` table
def test_the_fisheye_table_takes_defaults_and_says_what_is_wrong():
    assert FE.spec({}, "s") == {"fov": 180.0, "projection": "equisolid", "fill": "frame"}
    assert FE.spec(None, "s") is None and FE.spec(False, "s") is None
    for bad, frag in (({"fov": 20}, "fov = 20 is not degrees from 30 to 360"), ({"fov": True}, "fov = True"),
                      ({"fov": "wide"}, "fov = 'wide'"), ({"projection": "stereographic"}, "equisolid, equidistant"),
                      ({"fill": "crop"}, "is none of frame, circle"), (True, "fisheye is a table")):
        with pytest.raises(FE.FisheyeError) as e:
            FE.spec(bad, "shot 'f'")
        assert frag in str(e.value)


SHOT = {"name": "low", "from": 0.0, "to": 1.0, "at": [0, -0.4, 0.1], "look": "cast:carly.head",
        "fisheye": {"fov": 180, "projection": "equisolid", "fill": "frame"}, "roll": 8, "lag": 0.2,
        "keys": [{"t": 0.0}, {"t": 1.0, "at": [0, -0.3, 0.1], "roll": 0}], "follow": {"target": "cast:carly", "axes": "x"},
        "dof": {"focus": "cast:carly", "fstop": 4.0}}


def test_a_fisheye_shot_takes_at_look_roll_keys_lag_follow_and_dof():
    SP.check(SHOT, ["1x1", "16x9"])
    assert SP.fisheye_spec(SHOT, "s") == {"fov": 180.0, "projection": "equisolid", "fill": "frame"}


@pytest.mark.parametrize("patch, frag", [
    ({"lens": 24}, "shot 'low': `lens` with `fisheye`: a fisheye's focal length comes from its `fov` and `fill`"),
    ({"frame": {"subject": "cast:carly.head"}}, "`frame` with `fisheye`: automatic framing solves a perspective lens"),
    ({"shift": [0, 0.1]}, "`shift` with `fisheye`: a fisheye's picture is centred on its axis"),
    ({"style": "silhouette"}, "`style` with `fisheye`: a look is drawn with Workbench and EEVEE"),
    ({"reflection": {"object": "glass"}}, "`reflection` with `fisheye`"),
    ({"keys": [{"t": 0}, {"t": 1, "lens": 50}]}, "shot 'low' keys[1]: `lens` with `fisheye`"),
    ({"keys": [{"t": 0, "shift": [0, 0.1]}]}, "shot 'low' keys[0]: `shift` with `fisheye`"),
    ({"fisheye": {"fov": 180, "fil": "frame"}}, "shot 'low' fisheye: unknown key 'fil' (known: fov, projection, fill)"),
    ({"fisheye": {"fov": 400}}, "shot 'low' fisheye: fov = 400"),
    ({"aspect": {"1x1": {"fisheye": {"fill": "round"}}}}, "shot 'low' aspect.1x1 fisheye: fill = 'round'"),
])
def test_what_a_fisheye_refuses_says_why(patch, frag):
    with pytest.raises(SP.ShotError) as e:
        SP.check(dict(SHOT, **patch), ["1x1", "16x9"])
    assert frag in str(e.value)


def test_an_output_can_drop_the_fisheye_for_a_lens_or_change_its_fill_and_keep_the_rest():
    shot = dict(SHOT, aspect={"16x9": {"fisheye": False, "lens": 35}, "1x1": {"fisheye": {"fill": "circle"}}})
    SP.check(shot, ["1x1", "16x9"])
    assert SP.fisheye_spec(SP.merged(shot, shot["aspect"]["16x9"]), "s") is None
    assert SP.fisheye_spec(SP.merged(shot, shot["aspect"]["1x1"]), "s") == {"fov": 180.0, "projection": "equisolid",
                                                                           "fill": "circle"}
    persp = {"name": "p", "from": 0, "to": 1, "lens": 35, "aspect": {"1x1": {"fisheye": {}}}}
    with pytest.raises(SP.ShotError, match=r"aspect.1x1: `lens` with `fisheye`.*own \[shot.aspect.<output>\] table"):
        SP.check(persp, ["1x1", "16x9"])                                   # the shot's lens would reach the fisheye


# ---------------------------------------------------------------- the clip start of a fisheye
def sampled_distance(tri, n=300):
    a, b = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    keep = a + b <= 1
    P = tri[0] + a[keep][:, None] * (tri[1] - tri[0]) + b[keep][:, None] * (tri[2] - tri[0])
    return float(np.linalg.norm(P, axis=1).min())


def test_a_triangles_distance_is_its_nearest_point():
    rng = np.random.default_rng(11)
    everywhere = CL.Cone(math.pi)
    for _ in range(120):
        tri = rng.normal(0, 1.5, 3) + rng.normal(0, rng.choice([0.05, 0.6, 3.0]), (3, 3))
        d, near = CL.tri_distance(tri[None], everywhere)[0], sampled_distance(tri)
        step = max(np.linalg.norm(tri[i] - tri[i - 1]) for i in range(3)) / 300
        assert d <= near + 1e-12 and near - d < 2 * step, (d, near, step)


def test_a_fisheye_counts_what_reaches_into_its_cone_and_leaves_what_is_behind_it():
    cone = CL.Cone(math.pi / 2)                                          # a 180 degree fisheye: 99 degrees each side
    assert cone.half == pytest.approx(math.radians(99.0))
    ahead = np.array([[-0.1, -0.1, -2.0], [0.1, -0.1, -2.0], [0.0, 0.1, -2.0]])
    beside = np.array([[0.5, -0.05, 0.08], [0.5, 0.05, 0.08], [0.6, 0.0, 0.08]])     # 9 degrees behind the lens's plane
    behind = np.array([[-0.1, -0.1, 0.5], [0.1, -0.1, 0.5], [0.0, 0.1, 0.5]])
    d = CL.tri_distance(np.stack([ahead, beside, behind]), cone)
    assert d[0] == pytest.approx(2.0, abs=0.01) and d[1] == pytest.approx(math.hypot(0.5, 0.08), abs=1e-6)
    assert np.isinf(d[2])
    assert CL.sphere_bounds([[-1, -1, -1]], [[1, 1, 1]], np.eye(4), cone)[0] == 0.0        # the camera in a box
    items = [((tri.min(0), tri.max(0)), np.eye(4),
              lambda tri=tri: (CL.Piece(tri, np.array([[0, 1, 2]])), tri)) for tri in (ahead, beside, behind)]
    assert CL.least_distance(items, cone) == pytest.approx(math.hypot(0.5, 0.08), abs=1e-6)
    assert CL.least_distance(items[::2], cone) == pytest.approx(2.0, abs=0.01)


def test_the_distance_search_finds_what_the_minimum_over_every_triangle_finds():
    rng = np.random.default_rng(2)
    cone = CL.Cone(math.radians(70.0))
    for _ in range(20):
        items, every = [], []
        for _o in range(4):
            n = int(rng.integers(3, 500))
            V = rng.uniform(-4, 4, 3) + rng.normal(0, 0.5, (3 * n, 3))
            F = np.arange(3 * n).reshape(-1, 3)
            M = np.eye(4)
            M[:3, 3] = rng.normal(0, 0.3, 3)
            local = V - M[:3, 3]
            piece = CL.Piece(local, F, size=40)
            items.append(((local.min(0), local.max(0)), M, lambda piece=piece, local=local: (piece, local)))
            every.append(V[F])
        assert CL.least_distance(items, cone) == pytest.approx(CL.tri_distance(np.concatenate(every), cone).min(),
                                                              abs=1e-12)


# ---------------------------------------------------------------- the checks and the pixel maps see through it
def sampled_camera(fe, size, shift=(0.0, 0.0)):
    lens = FE.Lens.of(fe, size)
    return {"names": ["Fish"], "world": np.eye(4)[None], "lens": np.zeros(1), "sensor": np.array([[36.0, 24.0]]),
            "shift": np.array([shift]), "sensor_fit": {"Fish": "AUTO"},
            "fisheye": np.array([[FE.CODES[fe["projection"]], lens.f, lens.field]]), "fisheye_errors": {}}


def test_framing_projects_through_a_fisheye_and_refuses_a_panorama_it_cannot():
    fe = {"fov": 180.0, "projection": "equisolid", "fill": "frame"}
    cam = sampled_camera(fe, (1920, 1080))
    d = np.array([1920.0, 1080.0, 0.0]) / math.hypot(1920, 1080)          # towards the top right corner, 90 degrees off
    uv, depth = CC.project(np.array([[0, 0, -3.0], 2.0 * d]), cam, 0, (1920, 1080))
    assert uv[0] == pytest.approx([0.5, 0.5]) and depth[0] == pytest.approx(3.0)
    assert uv[1] == pytest.approx([1.0, 1.0], abs=1e-9) and depth[1] == pytest.approx(2.0)
    assert CC.fisheye_lens(cam, 0, (1920, 1080)).margin(uv)[1] == pytest.approx(0.0, abs=1e-9)
    cam["fisheye"][0, 0], cam["fisheye_errors"] = -1, {"Fish": "a panoramic camera of type MIRRORBALL"}
    with pytest.raises(CheckError, match="camera 'Fish': a panoramic camera of type MIRRORBALL"):
        CC.project(np.zeros((1, 3)), cam, 0, (1920, 1080))


def test_a_pixel_map_through_a_fisheye_finds_what_its_rays_hit():
    size = (120, 80)
    fe = {"fov": 200.0, "projection": "equidistant", "fill": "frame"}
    fields = dict(camera_fields(fe, size), type="PANO")
    info = {"matrix": np.eye(4).tolist(), "type": "PANO", "lens": 50.0, "sensor": [36.0, 24.0], "fit": "AUTO",
            "shift": [0.0, 0.0], "ortho_scale": 7.0, "clip_start": 0.01, "size": list(size), "fisheye": fields}
    cam = PM.Camera(info)
    # a wall 1 m ahead (wider than the view: its triangles are bent most) and a small tile nearer, off the axis
    wall = np.array([[-6.0, -4.0, -1.0], [6.0, -4.0, -1.0], [6.0, 4.0, -1.0], [-6.0, 4.0, -1.0]])
    tile = np.array([[0.20, 0.05, -0.5], [0.30, 0.05, -0.5], [0.30, 0.12, -0.5], [0.20, 0.12, -0.5]])
    P = np.concatenate([wall, tile])
    T = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])
    tri, _, depth = PM.rasterise(P, T, cam)
    px, py = np.meshgrid(np.arange(size[0]), np.arange(size[1]))
    D = cam.lens.rays(px.ravel(), py.ravel())
    with np.errstate(divide="ignore", invalid="ignore"):
        tw, tt = -1.0 / D[:, 2], -0.5 / D[:, 2]                          # along each ray to the wall, to the tile
        Xw, Xt = D * tw[:, None], D * tt[:, None]
    on_wall = (D[:, 2] < 0) & (np.abs(Xw[:, 0]) <= 6) & (np.abs(Xw[:, 1]) <= 4)
    on_tile = (D[:, 2] < 0) & (Xt[:, 0] >= 0.2) & (Xt[:, 0] <= 0.3) & (Xt[:, 1] >= 0.05) & (Xt[:, 1] <= 0.12)
    want = np.where(on_tile, 2, np.where(on_wall, 0, -1))
    got = np.where(tri.ravel() >= 2, 2, np.where(tri.ravel() >= 0, 0, -1))
    edge = np.abs(np.abs(Xt[:, 0] - 0.25) - 0.05) < 2e-3                  # a ray grazing the tile's side may go either way
    assert (got == want)[~edge].all() and on_tile.sum() > 10 and on_wall.sum() > 2000
    hit = tri.ravel() >= 0
    assert np.abs(depth.ravel()[hit] - np.where(on_tile, tt, tw)[hit]).max() < 1e-6        # the distance along the ray
    m = PM.Map(tri, np.zeros(size[::-1] + (2,), np.float32), depth,
               {"materials": ["a"], "objects": ["o"], "posed": P, "rest": P, "tris": T, "tri_object": np.zeros(4, int),
                "tri_material": np.zeros(4, int), "tri_face": np.zeros(4, int)}, cam)
    w = m.where([0.25, 0.085, -0.5])
    assert w["on_picture"] and not w["hidden"] and w["depth"] == pytest.approx(np.linalg.norm([0.25, 0.085, -0.5]), abs=1e-4)
    assert m.where([0.0, 0.0, -1.0])["hidden"] is False and m.where([0.0, 0.0, -1.0])["x"] == pytest.approx(60.0)
