"""Fisheye shot cameras, bpy-free (docs/design.md: Shots: Fisheye shots): the `fisheye` table of a [[shot]], the lens that
puts the image circle where `fill` says for an output's picture, the Blender settings that make it, and the projection
between the camera's axes and the picture that the checks, `mk q`'s `screen()`, the scene points of the cut effects, the
pixel maps and the clip stage use for a fisheye camera.

A fisheye images the ray `theta` radians off its axis at `r` sensor millimetres from the picture's centre, along the
ray's own azimuth: equisolid r = 2 f sin(theta / 2) (most real fisheyes: equal solid angles cover equal areas),
equidistant r = f theta. Blender 4.2 draws them with Cycles only (EEVEE and Workbench draw a panoramic camera as a
perspective one) from a camera of type PANO; probed in Blender 4.2.3 by rendering markers at known directions, every one
within half a pixel of these formulas, in 1:1, 16:9 and 9:16 pictures:
  equisolid    panorama_type FISHEYE_EQUISOLID, fisheye_lens = f (mm); black past fisheye_fov (the field, radians)
  equidistant  panorama_type FISHEYE_LENS_POLYNOMIAL, theta = -(k0 + k1 r + k2 r^2 + ...) with fisheye_polynomial_k1 =
               -1 / f (radians per mm) and the other coefficients 0; black past fisheye_fov. Blender's own
               FISHEYE_EQUIDISTANT ignores the sensor and stretches its circle to both sides of the picture (an ellipse in
               a 16:9 frame), so it is not used
The sensor is fitted as a perspective camera's: with fit AUTO its width spans the picture's longer side (36 mm for every
shot camera), HORIZONTAL its width the picture's width, VERTICAL its height the height. A panoramic camera's lens shift
moves the picture by shift_x of its width and shift_y of its height (a perspective camera's moves it by fractions of the
longer side). Cycles clips a panoramic camera's rays at clip_start along each ray, so a fisheye's near limit is a
distance from the camera, not a depth (all three probed too).

Camera axes: x right, y up, looking down -z. Picture coordinates as Blender's world_to_camera_view gives them: (x, y) in
0..1 from the bottom left; pixels from the top left where a function says so."""
import math

import numpy as np

PROJECTIONS = ("equisolid", "equidistant")
FILLS = ("frame", "circle")
KEYS = ("fov", "projection", "fill")
DEFAULT = {"fov": 180.0, "projection": "equisolid", "fill": "frame"}
FOV = (30.0, 360.0)          # degrees: a fisheye's field (a narrower one would want a lens past Blender's 100 mm)
SENSOR = 36.0                # mm across the picture's longer side: every shot camera's sensor (fit AUTO)
PAST = 1.01                  # a frame-filling lens turns black this far past the picture's corners (so no sample of a
                             # corner pixel's filter falls outside its field)
CODES = {"equisolid": 1, "equidistant": 2}    # a projection as a number (sampled camera data, mkmmd.blender.ops_sample)
PANORAMA = {"equisolid": "FISHEYE_EQUISOLID", "equidistant": "FISHEYE_LENS_POLYNOMIAL"}
ZERO = 1e-9                  # a polynomial coefficient this small is 0


class FisheyeError(ValueError):
    """A `fisheye` table the shots stage cannot build, or a panoramic camera mk does not project."""


def spec(value, where):
    """The `fisheye` of a shot as one output sees it, checked: None when it has none (or `fisheye = false`), else
    {"fov": degrees, "projection", "fill"} with the defaults filled in (180, "equisolid", "frame"). Unknown keys are the
    shot table's to report (mkmmd.core.shotspec)."""
    if value is None or value is False:
        return None
    if not isinstance(value, dict):
        raise FisheyeError(f"{where}: fisheye is a table {{fov, projection, fill}} (or false in an aspect table)")
    fov = value.get("fov", DEFAULT["fov"])
    if isinstance(fov, bool) or not isinstance(fov, (int, float)) or not FOV[0] <= fov <= FOV[1]:
        raise FisheyeError(f"{where} fisheye: fov = {fov!r} is not degrees from {FOV[0]:g} to {FOV[1]:g} (the field "
                           f"across the frame's diagonal, or across the circle)")
    projection = value.get("projection", DEFAULT["projection"])
    if projection not in PROJECTIONS:
        raise FisheyeError(f"{where} fisheye: projection = {projection!r} is none of {', '.join(PROJECTIONS)}")
    fill = value.get("fill", DEFAULT["fill"])
    if fill not in FILLS:
        raise FisheyeError(f"{where} fisheye: fill = {fill!r} is none of {', '.join(FILLS)} (frame: the image circle "
                           f"round the frame, its diagonal spanning fov; circle: the whole circle inside the short side)")
    return {"fov": float(fov), "projection": projection, "fill": fill}


def radius(theta, f, projection):
    """Sensor mm from the picture's centre where a lens of `f` mm images the rays `theta` radians off its axis."""
    theta = np.asarray(theta, float)
    return 2.0 * f * np.sin(theta / 2.0) if projection == "equisolid" else f * theta


def angle(r, f, projection):
    """The ray (radians off the axis) a lens of `f` mm images `r` sensor mm from the centre: `radius` turned round; NaN
    past an equisolid lens's rim (r > 2 f, the ray straight behind it)."""
    r = np.asarray(r, float)
    if projection == "equisolid":
        with np.errstate(invalid="ignore"):
            return 2.0 * np.arcsin(r / (2.0 * f))
    return r / f


def _swing(theta, f, projection):
    """(along, across): sensor mm per radian a lens images a turn of the ray `theta` with, along the picture's radius
    (dr / dtheta) and across it (r / sin theta, f on the axis)."""
    theta = np.asarray(theta, float)
    if projection == "equisolid":
        c = np.maximum(np.cos(theta / 2.0), 1e-12)
        return f * c, f / c
    s = np.sin(theta)
    with np.errstate(invalid="ignore", divide="ignore"):
        across = np.where(theta > 1e-6, f * theta / np.maximum(s, 1e-12), f)
    return np.full(theta.shape, float(f)), across


def reach_mm(size, fill, sensor=SENSOR):
    """Sensor mm from the centre that `fill` puts at the edge of the field in a picture `size` (w, h) px, its sensor
    `sensor` mm across the longer side: the half-diagonal ("frame") or half the shorter side ("circle")."""
    w, h = float(size[0]), float(size[1])
    per = sensor / max(w, h)                                    # mm per px
    return 0.5 * per * (math.hypot(w, h) if fill == "frame" else min(w, h))


def lens(fe, size, sensor=SENSOR):
    """The focal length (mm) that puts `fe`'s field (fov degrees) across the frame's diagonal (`fill` "frame") or its
    circle ("circle") in a picture `size` (w, h) px."""
    R = reach_mm(size, fe["fill"], sensor)
    half = math.radians(fe["fov"]) / 2.0
    return R / (2.0 * math.sin(half / 2.0)) if fe["projection"] == "equisolid" else R / half


def cut(fe, size, sensor=SENSOR):
    """Blender's fisheye_fov (radians) for `fe` in a picture `size`: the picture is black past it. A circle ends at the
    field; a frame-filling lens turns black PAST times the half-diagonal from the centre (at most all round, 2 pi)."""
    if fe["fill"] == "circle":
        return math.radians(fe["fov"])
    f = lens(fe, size, sensor)
    r = PAST * reach_mm(size, "frame", sensor)
    if fe["projection"] == "equisolid":
        return 2.0 * math.pi if r >= 2.0 * f else 2.0 * float(angle(r, f, "equisolid"))
    return min(2.0 * math.pi, 2.0 * r / f)


def blender(fe, size, sensor=SENSOR):
    """{attribute: value} of bpy.types.Camera that make `fe` for a picture `size` (w, h) px with the sensor `sensor` mm
    across its longer side (fit AUTO): type PANO, the panorama type and its lens and field."""
    f = lens(fe, size, sensor)
    out = {"type": "PANO", "panorama_type": PANORAMA[fe["projection"]], "fisheye_lens": f,
           "fisheye_fov": cut(fe, size, sensor)}
    if fe["projection"] == "equidistant":
        out.update(fisheye_polynomial_k0=0.0, fisheye_polynomial_k1=-1.0 / f, fisheye_polynomial_k2=0.0,
                   fisheye_polynomial_k3=0.0, fisheye_polynomial_k4=0.0)
    return out


def params(cam):
    """(projection, f mm, field radians) of a Blender camera from its data's fields (a dict: type, panorama_type,
    fisheye_lens, fisheye_fov, fisheye_polynomial_k0 .. k4): None for a camera that is not panoramic, a FisheyeError for
    a panoramic one mk does not project (another panorama, or a lens polynomial that is not k1 alone)."""
    if cam["type"] != "PANO":
        return None
    kind = cam["panorama_type"]
    if kind == "FISHEYE_EQUISOLID":
        return "equisolid", float(cam["fisheye_lens"]), float(cam["fisheye_fov"])
    if kind == "FISHEYE_LENS_POLYNOMIAL":
        k = [float(cam[f"fisheye_polynomial_k{i}"]) for i in range(5)]
        if k[1] >= 0 or any(abs(k[i]) > ZERO for i in (0, 2, 3, 4)):
            raise FisheyeError(f"a lens polynomial of k0..k4 = {k}: mk projects k1 alone (k1 = -1 / f: an equidistant "
                               f"fisheye)")
        return "equidistant", -1.0 / k[1], float(cam["fisheye_fov"])
    raise FisheyeError(f"a panoramic camera of type {kind}: mk projects perspective cameras and fisheyes "
                       f"(FISHEYE_EQUISOLID, or FISHEYE_LENS_POLYNOMIAL with k1 alone: equidistant)")


def from_blender(cam, size):
    """The Lens of a Blender camera in a picture `size` (w, h) px, from its data's fields (`params`' and sensor_width,
    sensor_height, sensor_fit, shift_x, shift_y): None for a camera that is not panoramic, a FisheyeError for a panoramic
    one mk does not project."""
    got = params(cam)
    if got is None:
        return None
    projection, f, field = got
    return Lens(projection, f, field, size, sensor=float(cam["sensor_width"]), fit=cam.get("sensor_fit", "AUTO"),
                shift=(float(cam.get("shift_x", 0.0)), float(cam.get("shift_y", 0.0))),
                sensor_h=float(cam.get("sensor_height", 24.0)))


class Lens:
    """A fisheye lens as a picture of `size` (w, h) px shows it: `projection`, `f` (mm), `field` (radians: Blender's
    fisheye_fov, black past it), the sensor (`sensor` mm across the side `fit` names; `sensor_h`, its height, for
    VERTICAL) and the lens `shift`. `k` is pixels per sensor mm, (cx, cy) where the axis meets the picture (px from the
    left and from the BOTTOM), `rim` how far from there (px) the picture turns black."""

    def __init__(self, projection, f, field, size, sensor=SENSOR, fit="AUTO", shift=(0.0, 0.0), sensor_h=None):
        if projection not in PROJECTIONS:
            raise FisheyeError(f"projection {projection!r} is none of {', '.join(PROJECTIONS)}")
        self.projection, self.f, self.field = projection, float(f), float(field)
        self.w, self.h = float(size[0]), float(size[1])
        if fit == "VERTICAL":
            self.k = self.h / float(sensor_h or sensor)
        elif fit == "HORIZONTAL":
            self.k = self.w / float(sensor)
        else:
            self.k = max(self.w, self.h) / float(sensor)
        self.cx = self.w / 2.0 - float(shift[0]) * self.w
        self.cy = self.h / 2.0 - float(shift[1]) * self.h
        self.half = min(self.field / 2.0, math.pi)                 # the field's half-angle
        self.rim = float(radius(self.half, self.f, projection)) * self.k

    @classmethod
    def of(cls, fe, size, sensor=SENSOR):
        """The lens a shot's checked `fisheye` table (`spec`) makes for a picture `size` (w, h) px."""
        return cls(fe["projection"], lens(fe, size, sensor), cut(fe, size, sensor), size, sensor)

    def _polar(self, P):
        P = np.asarray(P, float).reshape(-1, 3)
        rho = np.hypot(P[:, 0], P[:, 1])
        theta = np.arctan2(rho, -P[:, 2])
        with np.errstate(invalid="ignore", divide="ignore"):
            ux = np.where(rho > 0, P[:, 0] / np.where(rho > 0, rho, 1.0), 0.0)
            uy = np.where(rho > 0, P[:, 1] / np.where(rho > 0, rho, 1.0), 0.0)
        return theta, ux, uy, np.linalg.norm(P, axis=1)

    def project(self, P):
        """(uv (n, 2), dist (n,)) of points P (n, 3) in the camera's axes: their place in the picture, (x, y) in 0..1 from
        the bottom left, and their distance from the camera, negative where the lens images nothing of them (past its
        field, or at the camera itself)."""
        theta, ux, uy, dist = self._polar(P)
        r = radius(theta, self.f, self.projection) * self.k
        uv = np.stack([(self.cx + r * ux) / self.w, (self.cy + r * uy) / self.h], 1)
        seen = (theta <= self.half + 1e-12) & (dist > 0)
        return uv, np.where(seen, dist, -dist)

    def pixels(self, P):
        """(x, y, dist) of points P (n, 3) in the camera's axes: pixel coordinates from the TOP left (pixel (x, y) covers
        [x, x + 1) x [y, y + 1)) and the signed distance of `project`."""
        uv, d = self.project(P)
        return uv[:, 0] * self.w, (1.0 - uv[:, 1]) * self.h, d

    def rays(self, px, py):
        """Unit directions (k, 3) in the camera's axes of the rays through the centres of pixels (px, py) (from the top
        left), zero where the picture is black (past the field)."""
        x = np.asarray(px, float).reshape(-1) + 0.5 - self.cx
        y = self.h - (np.asarray(py, float).reshape(-1) + 0.5) - self.cy
        r = np.hypot(x, y)
        theta = angle(r / self.k, self.f, self.projection)
        ok = np.isfinite(theta) & (theta <= self.half)
        theta = np.where(ok, theta, 0.0)
        s = np.sin(theta)
        with np.errstate(invalid="ignore", divide="ignore"):
            ux = np.where(r > 0, x / np.where(r > 0, r, 1.0), 0.0)
            uy = np.where(r > 0, y / np.where(r > 0, r, 1.0), 0.0)
        D = np.stack([s * ux, s * uy, -np.cos(theta)], 1)
        D[~ok] = 0.0
        return D

    def px_per_m(self, theta, dist):
        """Pixels a metre spans at distance `dist` on the ray `theta` off the axis, across the line of sight: the side of
        the square a small patch facing the camera there covers, as a share of its own side (a fisheye stretches along
        the picture's radius and across it by different amounts; an equisolid lens keeps areas, so it is k f / dist)."""
        along, across = _swing(theta, self.f, self.projection)
        return self.k * np.sqrt(along * across) / np.asarray(dist, float)

    def scale(self, P):
        """Picture heights a metre spans at points P (n, 3) in the camera's axes (`px_per_m` over the height): what turns
        a size in metres into a size in the picture there."""
        theta, _, _, dist = self._polar(P)
        with np.errstate(divide="ignore"):
            return self.px_per_m(theta, dist) / self.h

    def m_per_px(self, dist, px, py):
        """Metres a pixel spans at distance `dist` (m along its ray) at pixels (px, py) from the top left."""
        x = np.asarray(px, float) + 0.5 - self.cx
        y = self.h - (np.asarray(py, float) + 0.5) - self.cy
        theta = angle(np.hypot(x, y) / self.k, self.f, self.projection)
        theta = np.where(np.isfinite(theta), theta, math.pi)
        return 1.0 / self.px_per_m(theta, np.asarray(dist, float))

    def reach(self):
        """The largest angle off the axis (radians) the picture shows: its farthest corner's, or the field's edge where
        that is nearer."""
        r = max(math.hypot(x - self.cx, y - self.cy) for x in (0.0, self.w) for y in (0.0, self.h))
        theta = float(angle(r / self.k, self.f, self.projection))
        return min(theta if math.isfinite(theta) else math.pi, self.half)

    def margin(self, uv):
        """(n,) how far points at `uv` (from `project`) lie inside what the picture shows, as a share of the frame: the
        least of the frame's margins (x, 1 - x, y, 1 - y) and, where the image circle is nearer, the circle's (its
        radius less theirs, over the picture's shorter side); negative outside."""
        uv = np.asarray(uv, float).reshape(-1, 2)
        frame = np.minimum.reduce([uv[:, 0], 1.0 - uv[:, 0], uv[:, 1], 1.0 - uv[:, 1]])
        r = np.hypot(uv[:, 0] * self.w - self.cx, uv[:, 1] * self.h - self.cy)
        return np.minimum(frame, (self.rim - r) / min(self.w, self.h))
