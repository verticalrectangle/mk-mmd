"""Generated textures of the body part: the skin atlas (warm skin with soft pink at knees, elbows, knuckles, joints,
fingertips, heels and ankle bones, painted per atlas tile) and the warm toon ramp. Pure numpy; colours come from the spec.

Atlas tiles are tubes: u runs around the tube (u = angle / 2 pi), v along it. Which u is which side:
  arm     0 = front (-Y, the thumb side), 0.25 = palm side (down), 0.5 = back (+Y), 0.75 = top of the arm
  leg     0 = front (knee cap), 0.25 = inner side of the left leg, 0.5 = back, 0.75 = outer side
  palm    the back of the hand from the thumb side (0) to the little finger side (~0.5), then the palm from the little
          finger side back to the thumb side (~1); v along the hand. The designed hand (body_hand) takes u = k / 16 of its
          16-point rings, a mesh hand (body_hand_mesh) spreads each side evenly over the palm's width; the hand shell's
          info `palm_u` gives each finger's knuckle column (back u, palm u)
  fingers u round the finger, v = s / length along the finger's chain: the u of the back and of the palm side per
          finger in the hand shell's info `tiles` (0.125 and 0.625 for the four fingers)"""
import numpy as np

from .body_geom import smoothstep
from .body_mesh import ATLAS

DEFAULT = dict(base="#fffefc", blush="#f0a09a", blush_knee="#eba79c", nail="#f5c4bc", crease="#e29a92",
               neck_shadow="#e3aa95")
DEFAULT_TOON = (0.96, 0.84, 0.81)                  # light warm-peach shadow half (lit half = white): never grey


def hex_rgb(h):
    h = str(h).lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], float) / 255.0


def rgb_hex(c):
    return "#" + "".join(f"{int(round(float(np.clip(x, 0, 1)) * 255)):02x}" for x in c)


def palette(cfg=None):
    p = dict(DEFAULT)
    for k in p:
        if cfg and cfg.get(k):
            p[k] = cfg[k]
    return {k: hex_rgb(v) for k, v in p.items()}


FLUSH = 0.0                       # the texture is near white; colour comes from the marks and the studio's light


def base_tone(pal, flush=FLUSH):
    """The skin colour of the atlas away from any mark: the base with a faint overall flush (what the face should match)."""
    return pal["base"] * (1 - flush) + pal["blush_knee"] * flush


def _bump(u, v, cu, cv, su, sv):
    """A soft elliptical blob on the unit square; u wraps around the tube."""
    du = np.abs(u - cu)
    du = np.minimum(du, 1.0 - du)
    return np.exp(-((du / su) ** 2 + ((v - cv) / sv) ** 2))


def _v(sh, s):
    a, b = sh.s_range
    return float(np.clip((s - a) / max(b - a, 1e-9), 0.0, 1.0))


def hand_marks(info):
    """Marks of the hand's tiles from the hand shell's info, all soft and on the palm side or at the tips (the back of
    the hand and of the fingers stays clean): the heel of the palm, the ball of the thumb, pads under the fingers; on
    every finger faint joint creases, a pink pad and a pink tip."""
    out = {}
    pv = info["palm_v"]
    v_palm = lambda A: float(np.clip((A - pv["seam"]) / (pv["end"] - pv["seam"]), 0.0, 1.0))
    m = [(0.72, v_palm(0.25 * info["a_mid"]), 0.16, 0.10, 0.30, "blush"),                     # heel of the palm
         (0.90, v_palm(0.40 * info["a_mid"]), 0.06, 0.12, 0.22, "blush")]                     # ball of the thumb
    for part in ("index", "middle", "ring", "little"):
        A_k = float(np.dot(info["chains"][part][0] - info["wrist"], info["frame"][0]))
        m.append((info["palm_u"][part][1], v_palm(A_k + 0.004), 0.035, 0.040, 0.40, "blush"))  # pad under the finger
    out["palm"] = m
    for part, t in info["tiles"].items():
        L1, L2, _ = info["lengths"][part]
        L = t["length"]
        b, p = t["back_u"], t["front_u"]
        out[part] = [(p, L1 / L, 0.20, 0.012, 0.35, "crease"), (p, (L1 + L2) / L, 0.20, 0.011, 0.28, "crease"),
                     (p, 0.95, 0.40, 0.070, 0.60, "blush"), (b, 0.995, 0.50, 0.050, 0.20, "blush")]
    return out


def marks(shells):
    """Colour marks per atlas tile: {tile: [(cu, cv, su, sv, strength, colour key)]} from the shells' joints (see the module
    docstring for what u means). Strength 1 mixes half of the colour in at the centre of the mark."""
    out = {}
    for sh in shells:
        if sh.name.endswith("_R") or sh.info.get("kind") == "nail":
            continue
        k = sh.info.get("kind")
        i = sh.info
        if k == "leg":
            out["leg"] = [(0.0, _v(sh, i["sK"]), 0.23, 0.042, 1.25, "blush_knee"),         # knee cap
                          (0.5, _v(sh, i["sK"]), 0.12, 0.022, 0.55, "blush_knee"),         # back of the knee
                          (0.5, _v(sh, i["sC"] - 0.004), 0.17, 0.034, 0.85, "blush_knee"),  # heel
                          (0.0, _v(sh, i["sB"]), 0.22, 0.032, 0.55, "blush_knee"),         # ball of the foot
                          (0.0, 0.985, 0.50, 0.060, 0.80, "blush"),                        # toe tips
                          (0.25, _v(sh, i["sA"] + 0.009), 0.07, 0.018, 0.55, "blush_knee"),  # ankle bones
                          (0.75, _v(sh, i["sA"] + 0.007), 0.07, 0.018, 0.55, "blush_knee")]
        elif k == "arm":
            out["arm"] = [(0.5, _v(sh, i["la"]), 0.18, 0.030, 1.00, "blush_knee"),        # elbow
                          (0.25, _v(sh, i["la"] - 0.02), 0.20, 0.060, 0.30, "blush")]      # inner elbow
        elif k == "hand":
            out.update(hand_marks(i))
    return out


def neck_weight(u, front=1.0, back=0.35):
    """Share of the neck shadow around the neck: u = 0 front (under the chin), 0.25 her left, 0.5 the nape."""
    return back + (front - back) * (0.5 + 0.5 * np.cos(2 * np.pi * u)) ** 1.5


def skin_texture(mk, pal, size=1024, flush=FLUSH, neck=None):
    """RGBA uint8 skin atlas. `mk`: the dict returned by `marks`; `flush` mixes that share of `blush_knee` into the base
    everywhere (one living skin tone for torso, arms and legs)."""
    base = pal["base"] * (1 - flush) + pal["blush_knee"] * flush
    img = np.zeros((size, size, 3), float)
    img[:] = base
    yy, xx = np.mgrid[0:size, 0:size]
    U, V = (xx + 0.5) / size, 1.0 - (yy + 0.5) / size
    for tile, (u0, v0, u1, v1) in ATLAS.items():
        inside = (U >= u0) & (U < u1) & (V >= v0) & (V < v1)
        lu, lv = (U - u0) / (u1 - u0), (V - v0) / (v1 - v0)
        col = np.broadcast_to(base, (size, size, 3)).copy()
        for cu, cv, su, sv, k, key in mk.get(tile, []):
            b = np.clip(k * 0.5 * _bump(lu, lv, cu, cv, su, sv), 0.0, 0.85)
            col = col * (1 - b[..., None]) + pal[key] * b[..., None]
        if neck and tile == "torso" and neck["strength"] > 0:
            # soft peach-brown band under the jaw: darkest at the seam ring (v = 1), fading over `v_h` towards the chest
            wv = np.where(lv <= 1.0, 1.0 - smoothstep((1.0 - lv) / neck["v_h"]), 0.0)
            b = np.clip(neck["strength"] * wv * neck_weight(lu), 0.0, 0.9)
            col = col * (1 - b[..., None]) + pal["neck_shadow"] * b[..., None]
        img = np.where(inside[..., None], col, img)
    return np.dstack([np.clip(img * 255 + 0.5, 0, 255), np.full((size, size), 255.0)]).astype(np.uint8)


def toon_ramp(pal, mult, width=32, height=32, edge=0.5, soft=0.10):
    """Toon ramp (rows from the lit white at the top to the shadow colour `mult` at the bottom, as MMD reads it)."""
    v = np.linspace(1.0, 0.0, height)                        # row 0 = lit
    t = smoothstep((v - (edge - soft / 2)) / soft)
    col = np.asarray(mult, float)[None] * (1 - t[:, None]) + np.ones((1, 3)) * t[:, None]
    row = np.clip(col * 255 + 0.5, 0, 255)
    img = np.repeat(row[:, None, :], width, axis=1)
    return np.dstack([img, np.full((height, width), 255.0)]).astype(np.uint8)
