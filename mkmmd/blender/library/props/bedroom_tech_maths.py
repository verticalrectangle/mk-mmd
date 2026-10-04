"""Pure maths of the bedroom tech props (`desk_lamp`, `cassette_player`, `cassette_tape` in bedroom_tech.py): the lamp's
arm kinematics, a coil-spring path, the stable pick of a tape's label colour, handwriting-like scribble strokes and small
layout helpers. No bpy: importable from tests (tests/test_bedroom_tech_maths.py). Metres, the prop frames of the props.

Lamp plane. The lamp stands in the plane x = 0 of its frame (origin = centre of the base's underside, +Z up, the shade
looks toward -Y). Points of that plane are written (f, z) with f = -y the distance forward; `lamp_joints` returns them as
(x, y, z) triples."""
import hashlib
import math
import random
import re

# ================================================================= label colours
LABEL_SLOTS = ("love", "gold", "foam", "iris", "rose", "pine")      # the palette slots a tape label is picked from


def label_index(name, n=len(LABEL_SLOTS)):
    """Stable index into LABEL_SLOTS for an instance name. A trailing number steps through the list (tape1, tape2, ...
    get consecutive colours, so a scatter of up to six tapes never repeats one); the rest of the name picks the start
    with an md5 digest (not Python's randomised hash), so a name always gets the same colour in every run."""
    m = re.search(r"(\d+)$", name)
    base = name[:m.start()] if m else name
    h = int.from_bytes(hashlib.md5(base.encode("utf-8")).digest()[:4], "little")
    return (h + (int(m.group(1)) if m else 0)) % n


def seed_of(name):
    """Stable 32-bit seed from a name (for random.Random)."""
    return int.from_bytes(hashlib.md5(name.encode("utf-8")).digest()[4:8], "little")


# ================================================================= lamp arms
LAMP = dict(
    pivot_f=-0.012, pivot_z=0.056,          # base pivot (arm 1's lower end): a little behind the base centre
    lower_len=0.380, lower_back_deg=12.0,   # lower arm: pivot-to-pivot length, tilt back from vertical (up and back)
    upper_len=0.360, upper_up_deg=-3.0,     # upper arm: length, angle above horizontal (negative = drooping forward)
    shade_down_deg=35.0,                    # the shade looks this far below the horizon, toward -Y
    neck_back=0.072,                        # head pivot -> bulb centre, along the shade axis (the neck of the shade)
)


def lamp_joints(p=None):
    """Joints of the lamp in its frame -> dict of (x, y, z) triples and unit vectors:
      pivot      the base pivot (lower end of arm 1)
      elbow      the middle joint (arm 1's upper end = arm 2's lower end)
      head       the head pivot (arm 2's upper end), on the shade axis behind the bulb
      bulb       the bulb centre = the shade's origin
      out        unit vector of the light's direction (out of the shade)
      lower_dir, upper_dir   unit vectors along the arms (pivot -> elbow, elbow -> head)
    The arms fold up and back (arm 1) and forward (arm 2)."""
    q = dict(LAMP, **(p or {}))
    a1 = math.radians(q["lower_back_deg"])
    a2 = math.radians(q["upper_up_deg"])
    d1 = (-math.sin(a1), math.cos(a1))                    # (f, z): backward and up
    d2 = (math.cos(a2), math.sin(a2))
    f0, z0 = q["pivot_f"], q["pivot_z"]
    f1, z1 = f0 + q["lower_len"] * d1[0], z0 + q["lower_len"] * d1[1]
    f2, z2 = f1 + q["upper_len"] * d2[0], z1 + q["upper_len"] * d2[1]
    s = math.radians(q["shade_down_deg"])
    out = (math.cos(s), -math.sin(s))                     # (f, z)
    fb, zb = f2 + q["neck_back"] * out[0], z2 + q["neck_back"] * out[1]

    def xyz(f, z):
        return (0.0, -f, z)
    return {"pivot": xyz(f0, z0), "elbow": xyz(f1, z1), "head": xyz(f2, z2), "bulb": xyz(fb, zb),
            "out": (0.0, -out[0], out[1]), "lower_dir": (0.0, -d1[0], d1[1]), "upper_dir": (0.0, -d2[0], d2[1])}


def rot_x_for(direction):
    """Rotation about X (rad) that turns +Z onto `direction` (a unit vector in the plane x = 0)."""
    return math.atan2(-direction[1], direction[2])


# ================================================================= coil spring
def coil_path(a, b, radius, pitch, per_turn=8, hook=0.0035):
    """Centre line of a close-wound tension spring between the anchors a and b (3-tuples): a short straight hook from
    each anchor, then a helix of `radius` about the segment a-b with `pitch` between turns. -> list of (x, y, z).
    The number of turns is rounded so the coil ends where it starts in phase (it closes on the hooks)."""
    ax, ay, az = a
    d = (b[0] - ax, b[1] - ay, b[2] - az)
    length = math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
    u = (d[0] / length, d[1] / length, d[2] / length)
    e1 = _perp(u)
    e2 = (u[1] * e1[2] - u[2] * e1[1], u[2] * e1[0] - u[0] * e1[2], u[0] * e1[1] - u[1] * e1[0])
    body = max(length - 2.0 * hook, pitch)
    turns = max(round(body / pitch), 1)
    n = turns * per_turn
    pts = [tuple(a)]
    for i in range(n + 1):
        t = i / n
        ang = 2.0 * math.pi * turns * t
        s = hook + body * t
        c, sn = radius * math.cos(ang), radius * math.sin(ang)
        pts.append((ax + u[0] * s + e1[0] * c + e2[0] * sn, ay + u[1] * s + e1[1] * c + e2[1] * sn,
                    az + u[2] * s + e1[2] * c + e2[2] * sn))
    pts.append(tuple(b))
    return pts


def _perp(u):
    ref = (0.0, 0.0, 1.0) if abs(u[2]) < 0.9 else (1.0, 0.0, 0.0)
    d = ref[0] * u[0] + ref[1] * u[1] + ref[2] * u[2]
    p = (ref[0] - u[0] * d, ref[1] - u[1] * d, ref[2] - u[2] * d)
    n = math.sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2)
    return (p[0] / n, p[1] / n, p[2] / n)


# ================================================================= small layouts
def spread(n, total, gap):
    """n equal cells of a row `total` wide with `gap` between them, centred on 0 -> (centre, width) per cell."""
    w = (total - gap * (n - 1)) / n
    x0 = -total / 2.0 + w / 2.0
    return [(x0 + i * (w + gap), w) for i in range(n)]


def ticks(n, span):
    """n tick positions across a dial of width `span` centred on 0 -> [(x, kind)], kind 2 every 10th, 1 every 5th."""
    out = []
    for i in range(n):
        k = 2 if i % 10 == 0 else 1 if i % 5 == 0 else 0
        out.append((-span / 2.0 + span * i / (n - 1), k))
    return out


def rib_radii(r_in, r_out, n):
    """n radii evenly spread over a grille from r_in to r_out (inclusive)."""
    return [r_in + (r_out - r_in) * i / (n - 1) for i in range(n)]


# ================================================================= handwriting-like scribble
def scribble(rng, width, height, lines=2, size=0.0030, step=0.0005):
    """Strokes that look like a line or two of cursive handwriting without being text: words of looped, wavy curves
    on baselines. Returns a list of polylines [(x, y), ...] inside [0, width] x [0, height] (metres, y up). `size`
    is the x-height; the strokes lean right and loop on the ascenders."""
    out = []
    pitch = height / max(lines, 1)
    for ln in range(lines):
        base = pitch * (ln + 0.28)
        x = rng.uniform(0.0, 0.08) * width
        limit = width * (0.97 if ln == 0 else rng.uniform(0.55, 0.9))      # the last line stops short
        while x < limit - 0.012:
            n_letters = rng.randint(3, 8)
            wlen = min(n_letters * size * rng.uniform(0.85, 1.15), limit - x)
            if wlen < 0.006:
                break
            pts = []
            steps = max(int(wlen / step), 8)
            om = 2.0 * math.pi * n_letters / wlen                 # loops per metre
            ph = rng.uniform(0.0, 2.0 * math.pi)
            asc = [rng.random() < 0.28 for _ in range(n_letters + 1)]
            for i in range(steps + 1):
                s = i / steps
                xx = x + wlen * s
                k = min(int(s * n_letters), n_letters)
                loop = size * 0.30 * math.cos(om * wlen * s + ph)           # backward loop of the stroke: x wiggle
                yy = base + size * (0.5 + 0.5 * math.sin(om * wlen * s + ph + 1.1)) * (1.9 if asc[k] else 1.0)
                pts.append((min(max(xx + loop, 0.0), width), min(max(yy, 0.0), height)))
            out.append(pts)
            x += wlen + size * rng.uniform(1.2, 2.2)                # a space
    return out


def new_rng(name, salt=""):
    return random.Random(seed_of(name + salt))
