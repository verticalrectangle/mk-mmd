"""The parity lab (docs/design.md: Parity): small scenes that each isolate one thing EEVEE does, rendered by Blender
once and by mkr as often as it changes, from the same bake.

A case is a tiny mk project in <lab>/<case>/ (the lab folder: --root, else $MK_LAB, else <user cache>/lab):
  mk.toml           [project] (30 fps, frame0 1, the case's frames) and its [[output]]s; blend = "scene.blend"
  case.json         the case as built: its spec (CASES), the build's report (Blender version, EEVEE settings, times)
  scene.blend       the scene, built by the Blender op lab_build (mkmmd/blender/ops_lab.py)
  scene.bake/       its bake: the build's bake stage (`mk build --bake --project <case>` bakes it again)
  ref/<output>/     Blender's frames (EEVEE, 64 samples), 8-bit as `mk render` writes them
  ref16/<output>/   the same frames as 16-bit PNG (before the 8-bit quantisation and its dither)
  linear/<output>/  the same frames scene-linear (OpenEXR half, no view transform)
  mkr/<output>/     mkr's frames of the last `mk parity lab`; compare/<output>/ their metrics and sheet
  variants/<name>/  Blender's frames of a variant of the case (VARIANTS: shadows off, Cycles), rendered from the saved
                    scene by the Blender op lab_variant: <output>/ 8-bit PNG with the scene-linear EXR beside it,
                    16/<output>/ 16-bit, variant.json, and compare/<output>/ of the last `mk parity lab --against <name>`
  mkr_noshadow/<output>/  mkr's frames with every shadow off ({"shadows": false}) of the last `--separate`;
                    separate/<comparison>/<output>/ that report's sheets

`separate` splits a case's error into its parts: EEVEE and mkr each against Cycles (the light integral EEVEE
approximates, computed exactly), with and without shadows, and each renderer's shadow term (its frame minus the same
frame without shadows) against the others', plus the case's probes (a penumbra's width along a scanline, the mean in a
box). The cases that show a cast member (mmd_model, mmd_sphere) are built from a built project's scene (`--source`), the
member at rest; the others from nothing. No case names a model or a project."""
import fnmatch
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np

from .. import bridge
from .. import config as CFG
from ..core import cast as CN
from ..project import Project, ProjectError
from . import mkr as MKR
from . import parity as PY

FPS = 30
MAIN = [{"name": "main", "size": [960, 540]}]
KEY = [1.0, 0.921582, 0.822786]                  # a warm key, a neutral fill and a red rim (linear), as a lit MV has them
FILL = [0.887923, 0.814847, 0.752942]
RIM = [0.679542, 0.223228, 0.208637]


def _area(shape, size, energy, size_y=None, **kw):
    return {"type": "AREA", "shape": shape, "size": size, **({"size_y": size_y} if size_y else {}), "energy": energy, **kw}


def _light_case(doc, light):
    """One light over a diffuse plane, a diffuse sphere and a glossy one: frame 1 from above and in front, frame 2
    from low behind on the right (the light partly under the spheres' horizon)."""
    return {"kind": "light", "doc": doc, "frames": 2, "labels": ["high", "grazing"], "outputs": MAIN,
            "params": {"light": light, "camera": [[0.0, -5.0, 2.6], [0.0, 0.0, 0.45]],
                       "places": [[[-2.2, -2.0, 3.0], [0.0, 0.0, 0.3]], [[2.8, 1.2, 0.9], [0.0, 0.0, 0.4]]]}}


def _spheres_case(doc, materials, light, place, labels):
    return {"kind": "spheres", "doc": doc, "frames": len(materials), "labels": labels, "outputs": MAIN,
            "params": {"materials": materials, "light": light, "place": place,
                       "camera": [[0.0, -3.4, 1.3], [0.0, 0.0, 0.6]]}}


def _glossy(distribution, r):
    return {"name": f"{'multi' if distribution == 'MULTI_GGX' else 'ggx'}_{r:g}", "kind": "glossy",
            "distribution": distribution, "inputs": {"Color": [0.8, 0.8, 0.8], "Roughness": r}}


def _principled(name, **inputs):
    return {"name": name, "kind": "principled", "inputs": {k.replace("_", " "): v for k, v in inputs.items()}}


def _alpha_case(doc, method):
    return {"kind": "alpha", "doc": doc, "frames": 1, "outputs": MAIN,
            "params": {"method": method, "width": 16.0, "checker": [16, 9, 1.0],
                       "planes": [{"name": "ramp", "color": [1.0, 0.1, 0.05], "alpha": "u", "w": 14.0, "h": 4.0,
                                   "at": [0.0, 1.0], "y": 0.0},
                                  {"name": "half", "color": [0.05, 0.2, 1.0], "alpha": 0.5, "w": 8.0, "h": 5.0,
                                   "at": [1.0, -1.0], "y": -0.5}]}}


GGX = [0.05, 0.1, 0.2, 0.35, 0.5, 0.8]
GGX1 = [0.1, 0.35, 0.8]
RAMP = {"width": 16.0, "height": 9.0, "lo": -10.0, "hi": 8.0,
        "rows": [[1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1], [0.8, 0.45, 0.3]]}
MEMBER_LIGHTS = [_area("SQUARE", 1.6, 260.0, name="key", color=KEY, offset=[-1.6, -2.0, 1.0]),
                 _area("SQUARE", 2.0, 90.0, name="fill", color=FILL, offset=[2.2, -1.8, 0.2]),
                 _area("SQUARE", 1.5, 180.0, name="rim", color=RIM, offset=[1.2, 2.2, 1.0])]


def _shadow_case(doc, objects, light, places, cameras, labels, eevee=None, floor=0.6):
    """A shadow case seen by two cameras: `near` (960 x 540) close enough for EEVEE's finest shadow texels, `far`
    (480 x 270) far enough for coarser ones (cameras: {output: [at, target, lens]})."""
    c = {"kind": "shadow", "doc": doc, "frames": len(places), "labels": labels,
         "outputs": [{"name": "near", "size": [960, 540]}, {"name": "far", "size": [480, 270]}],
         "params": {"light": light, "objects": objects, "places": places, "cameras": cameras, "floor": floor}}
    if eevee:
        c["eevee"] = eevee
    return c


# a key light as a lit MV has it (1.6 m square, 260 W, 3.3 m away), aimed at a spot `at` from three sides: high in
# front, low from the side, steep from behind
def _key_places(at):
    x, y, z = at
    return [[[x - 1.4, y - 2.6, z + 2.0], at], [[x + 3.0, y - 1.0, z + 0.6], at], [[x - 0.8, y + 1.8, z + 2.6], at]]


KEY_LIGHT = _area("SQUARE", 1.6, 260.0, color=KEY)
# plates a known height over the floor (mm), hidden from the camera, under a light 1 cm wide: EEVEE's normal offset
# and depth tolerance at three angles, at its finest level (near) and a coarser one (far)
BIAS_HEIGHTS = [0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5, 3.2, 4.0, 5.0, 6.4, 8.0]
BIAS_PLATES = [{"kind": "plate", "name": f"plate_{i:02d}", "size": [0.04, 0.04], "camera": False,
                "at": [0.2 + ((i % 4) - 1.5) * 0.06, ((i // 4) - 1.5) * 0.06, h / 1000.0]}
               for i, h in enumerate(BIAS_HEIGHTS)]
# thin occluders over the floor: strands 0.5, 1 and 2 mm thick, a 2 mm card on edge, a fan of lash-like slivers,
# at 5 mm, 2 cm and 8 cm
THIN = ([{"kind": "tube", "name": f"strand_{d:g}mm_{h:g}", "r": d / 2000.0, "h": 0.3, "at": [-0.15, y, h],
          "rot": [0.0, 90.0, 0.0]}
         for d, y in ((0.5, -0.12), (1.0, -0.06), (2.0, 0.0)) for h in (0.005, 0.02, 0.08)]
        + [{"kind": "box", "name": "card", "size": [0.002, 0.2, 0.06], "at": [0.0, 0.12, 0.05]},
           *[{"kind": "box", "name": f"lash_{k}", "size": [0.0004, 0.05, 0.0002], "at": [0.12 + 0.01 * k, 0.1, 0.006],
              "yaw": 8.0 * (k - 3)} for k in range(7)]])
# cloth layers: a body block with plates 2, 4 and 8 mm off its front and top, a sleeve over a hand-like slab
LAYERS = [{"kind": "box", "name": "body", "size": [0.4, 0.25, 0.5], "at": [0.0, 0.0, 0.25]},
          *[{"kind": "box", "name": f"lapel_{g}mm", "size": [0.08, 0.003, 0.3], "at": [-0.12 + 0.12 * i, -0.125 - g / 1000.0 - 0.0015, 0.25]}
            for i, g in enumerate((2, 4, 8))],
          *[{"kind": "box", "name": f"collar_{g}mm", "size": [0.1, 0.06, 0.003], "at": [-0.12 + 0.12 * i, -0.06, 0.5 + g / 1000.0 + 0.0015]}
            for i, g in enumerate((2, 4, 8))],
          {"kind": "box", "name": "hand", "size": [0.12, 0.08, 0.02], "at": [0.32, -0.1, 0.12]},
          {"kind": "box", "name": "sleeve", "size": [0.08, 0.1, 0.06], "at": [0.29, -0.1, 0.161], "yaw": 5.0}]
# hashed see-through occluders: plates of alpha 0.25, 0.5, 0.75 and strands of alpha 0.5
HASHED = ([{"kind": "plate", "name": f"veil_{a:g}", "size": [0.12, 0.12], "at": [-0.15 + 0.15 * i, 0.0, 0.06], "alpha": a}
           for i, a in enumerate((0.25, 0.5, 0.75))]
          + [{"kind": "tube", "name": f"hair_{k}", "r": 0.0008, "h": 0.25, "at": [-0.15, 0.12 + 0.01 * k, 0.02],
              "rot": [0.0, 90.0, 0.0], "alpha": 0.5} for k in range(5)])
# a close-up under Crimson's key light (1.6 m, 260 W, 3.2 m away, 35 degrees off the receiver's normal), occluders
# hidden from the camera: 2.5 cm cards 5, 10, 20 and 50 mm over the floor, strands 1 mm thick 5 and 10 mm over it
# along x and one 10 mm over it along y. Per frame the light is that square, then overhead, then a 1.6 x 0.25 m
# rectangle long along the strands (rays toward its ends run along them) and long across them (rays toward its ends
# cross them away from the ray toward its centre)
CLOSE = ([{"kind": "plate", "name": f"card_{h}mm", "size": [0.025, 0.025], "camera": False, "at": [x, -0.045, h / 1000.0]}
          for h, x in ((5, -0.09), (10, -0.045), (20, 0.0), (50, 0.05))]
         + [{"kind": "tube", "name": f"strand_{h}mm", "r": 0.0005, "h": 0.22, "at": [-0.11, y, h / 1000.0],
             "rot": [0.0, 90.0, 0.0], "camera": False} for h, y in ((5, 0.02), (10, 0.035))]
         + [{"kind": "tube", "name": "strand_y_10mm", "r": 0.0005, "h": 0.14, "at": [0.1, -0.07, 0.01],
             "rot": [-90.0, 0.0, 0.0], "camera": False}])
CLOSE_KEY = [0.0, -3.2 * math.sin(math.radians(35.0)), 3.2 * math.cos(math.radians(35.0))]
# depth of field through an 85 mm lens (960 x 540, the camera 2 m before the focus card): emission checkerboards at 1,
# 2, 2.3 and 4 m before a checkered wall at 8 m (the front one a slanted bar over the focus card's corner, the 2.3 m one
# slightly out of focus behind the focus card's), and small bright spheres before the wall, by the middle card and in
# front of the focus card for bokeh. Per frame: [f-stop, blades, rotation (degrees), ratio, focus distance (m)]
DOF_KEYS = [[1.4, 0, 0.0, 1.0, 2.0], [2.8, 0, 0.0, 1.0, 2.0], [5.6, 0, 0.0, 1.0, 2.0], [8.0, 0, 0.0, 1.0, 2.0],
            [2.0, 6, 0.0, 1.0, 2.0], [2.0, 6, 15.0, 1.0, 2.0], [2.0, 5, 0.0, 1.0, 2.0], [2.0, 0, 0.0, 2.0, 2.0],
            [1.4, 0, 0.0, 1.0, 1.0], [2.8, 0, 0.0, 1.0, 8.0]]
DOF_CARDS = [{"name": "wall", "at": [0.0, 6.0, 1.0], "cells": [26, 16], "cell": 0.2,
              "colors": [[0.85, 0.8, 0.7], [0.03, 0.04, 0.08]]},
             {"name": "mid", "at": [0.45, 2.0, 1.1], "cells": [5, 5], "cell": 0.1,
              "colors": [[0.9, 0.12, 0.08], [0.9, 0.9, 0.9]]},
             {"name": "focus", "at": [-0.1, 0.0, 0.98], "cells": [12, 12], "cell": 0.02,
              "colors": [[0.1, 0.25, 0.9], [0.95, 0.8, 0.1]]},
             {"name": "near", "at": [-0.32, 0.3, 0.82], "cells": [16, 12], "cell": 0.01,
              "colors": [[0.95, 0.95, 0.9], [0.05, 0.3, 0.1]]},
             {"name": "front", "at": [0.0, -1.0, 1.0], "cells": [1, 12], "cell": 0.03, "spin": 10.0,
              "colors": [[0.1, 0.8, 0.2], [0.02, 0.02, 0.02]]}]
DOF_DOTS = ([{"at": [x, 5.9, 1.55], "r": 0.006, "strength": 150.0, "color": c}
             for x, c in ((-1.2, [1.0, 0.85, 0.6]), (-0.6, [0.6, 0.8, 1.0]), (0.0, [1.0, 0.5, 0.4]))]
            + [{"at": [x, 1.9, z], "r": 0.004, "strength": 150.0, "color": [1.0, 1.0, 1.0]} for x, z in ((0.25, 0.72),
                                                                                                      (0.7, 0.78))]
            + [{"at": [-0.12, -0.8, 0.92], "r": 0.002, "strength": 150.0, "color": [1.0, 0.9, 0.7]}])
DOF_PX = 960 * 85.0 / 36.0                          # pixels a metre spans one metre away through their camera


def _dof_card(name, u, v, d, w, h, value):
    """A uniform emission card `d` metres from the dof cases' camera, seen w x h pixels wide and high at pixel (u, v)."""
    s = d / DOF_PX
    side = min(w, h)
    return {"name": name, "at": [(u - 480) * s, d - 2.0, 1.0 + (270 - v) * s],
            "cells": [round(w / side), round(h / side)], "cell": side * s, "colors": [[value] * 3] * 2}


# the psf probe over black, focus 2 m: a row of lines 0.6 px wide at 16 depths (their spread is the blur's profile
# across), rows of 2 x 2 px squares at five depths, dim (0.5), above EEVEE's sprite threshold (8) and far above (100),
# and two edges: an in-focus card over a blurred one behind, a blurred one in front over an in-focus one
DOF_PSF_DEPTHS = [1.0, 1.2, 1.4, 1.6, 1.8, 1.9, 2.0, 2.1, 2.2, 2.4, 2.7, 3.0, 3.5, 4.5, 6.0, 9.0]
DOF_PSF_CARDS = ([_dof_card(f"line_{d:g}", 30 + 60 * i, 70, d, 0.6, 60, 0.5) for i, d in enumerate(DOF_PSF_DEPTHS)]
                 + [_dof_card(f"square_{value:g}_{d:g}", 100 + 180 * i, v, d, 2, 2, value)
                    for v, value in ((200, 0.5), (300, 8.0), (400, 100.0)) for i, d in enumerate((1.3, 1.6, 2.6, 4.0, 8.0))]
                 + [_dof_card("edge_far", 650, 490, 6.0, 300, 60, 0.25), _dof_card("edge_focus", 520, 490, 2.0, 240, 60, 0.6),
                    _dof_card("edge_near", 140, 490, 1.3, 180, 60, 0.6), _dof_card("edge_behind", 270, 490, 2.0, 180, 60, 0.25)])
DOF_PSF_KEYS = [[1.4, 0, 0.0, 1.0, 2.0], [2.8, 0, 0.0, 1.0, 2.0], [5.6, 0, 0.0, 1.0, 2.0], [1.4, 6, 0.0, 1.0, 2.0],
                [1.4, 6, 30.0, 1.0, 2.0], [1.4, 5, 0.0, 1.0, 2.0], [1.4, 0, 0.0, 2.0, 2.0], [1.4, 0, 0.0, 0.5, 2.0]]


# The world, lit by EEVEE through a sphere probe of it (glossy lobes read it by roughness) and its spherical harmonics
# (diffuse), and ray tracing. A grey floor, a diffuse ball and glossy balls of roughness 0.05, 0.2, 0.4 and 0.7 in a
# row, seen from the front (the sky over them) and from above (no sky)
def _shade(kind, color, **inputs):
    return {"kind": kind, "inputs": {"Color": color, **{k.replace("_", " "): v for k, v in inputs.items()}}}


def _plate(name, size, at, material, **kw):
    return {"kind": "plate", "name": name, "size": size, "at": at, "material": material, **kw}


WORLD_ROUGH = [0.05, 0.2, 0.4, 0.7]
WORLD_BALLS = ([_plate("floor", [14.0, 14.0], [0.0, 0.0, 0.0], _shade("diffuse", [0.5] * 3)),
                {"kind": "sphere", "name": "diffuse", "r": 0.35, "at": [-1.6, 0.0, 0.35],
                 "material": _shade("diffuse", [0.8] * 3)}]
               + [{"kind": "sphere", "name": f"glossy_{r:g}", "r": 0.35, "at": [-0.8 + 0.8 * i, 0.0, 0.35],
                   "material": _shade("glossy", [0.8] * 3, Roughness=r)} for i, r in enumerate(WORLD_ROUGH)])
WORLD_FRONT = [[0.0, -4.6, 1.1], [0.0, 0.0, 0.55]]
WORLD_ABOVE = [[0.0, -1.6, 4.4], [0.0, 0.0, 0.2]]
SKY = {"kind": "sky", "zenith": [0.2, 0.35, 1.0], "horizon": [0.8, 0.8, 0.9], "ground": [0.1, 0.08, 0.06],
       "sun": {"dir": [-0.5, 0.6, 0.6], "color": [1.0, 0.9, 0.7], "strength": 4.0, "size": 12.0}}
# EEVEE takes what of the world is brighter than its sun threshold (10) out of the probe into a sun light
SKY_SUN = dict(SKY, sun={"dir": [-0.5, 0.6, 0.6], "color": [1.0, 0.9, 0.7], "strength": 400.0, "size": 2.0},
               props={"sun_threshold": 10.0, "sun_angle": math.radians(2.0), "use_sun_shadow": True})
STRIPES = {"kind": "stripes", "a": [0.05, 0.05, 0.06], "b": [1.6, 1.4, 1.2], "bands": 16}


def _world_case(doc, world, views, labels, **kw):
    # mk's own worlds (the night_sky and photo_world sets) take no sun out of the probe: sun threshold 0 unless the
    # case says
    world = dict(world, props={"sun_threshold": 0.0, **(world.get("props") or {})})
    return {"kind": "scene", "doc": doc, "frames": len(views), "labels": labels, "outputs": MAIN, "world": world,
            "params": {"objects": WORLD_BALLS, "camera": views, "lens": 35.0}, **kw}


# EEVEE's ray tracing as mk's lit scenes have it (screen traced below roughness 0.5, denoised; fast GI above it: the
# horizon scan with bounce light, at half resolution)
RT = {"use_raytracing": True, "ray_tracing_method": "SCREEN",
      "ray_tracing_options": {"resolution_scale": "1", "trace_max_roughness": 0.5, "screen_trace_quality": 0.5,
                              "screen_trace_thickness": 0.2, "use_denoise": True, "denoise_spatial": True,
                              "denoise_temporal": True, "denoise_bilateral": True},
      "fast_gi_method": "GLOBAL_ILLUMINATION", "fast_gi_resolution": "2", "fast_gi_ray_count": 2,
      "fast_gi_step_count": 8, "fast_gi_quality": 0.25, "fast_gi_distance": 0.0, "fast_gi_thickness_near": 0.25,
      "fast_gi_thickness_far": math.pi / 4, "fast_gi_bias": 0.05}
RT_ROUGH = [0.1, 0.25, 0.4, 0.55, 0.7]
# a glossy floor seen from a camera looking down: a red box and a blue ball on it, a lamp above the frame (only its
# reflection is on screen: screen tracing cannot find it), a sun from the side, the sky
RT_FLOOR = [{"kind": "box", "name": "box", "size": [0.6, 0.6, 0.6], "at": [-0.9, 1.2, 0.3], "yaw": 20.0,
             "material": _shade("diffuse", [0.8, 0.15, 0.1])},
            {"kind": "sphere", "name": "ball", "r": 0.4, "at": [0.9, 1.8, 0.4], "material": _shade("diffuse", [0.1, 0.25, 0.8])},
            {"kind": "sphere", "name": "lamp", "r": 0.15, "at": [0.6, 2.5, 2.2],
             "material": {"kind": "emission", "inputs": {"Color": [1.0, 0.6, 0.25], "Strength": 20.0}}}]
RT_SUN = {"type": "SUN", "energy": 3.0, "angle": 2.0, "place": [[-3.0, -1.0, 3.0], [0.0, 1.5, 0.0]]}


def _floor_case(doc, roughness, labels):
    floor = _plate("floor", [14.0, 14.0], [0.0, 0.0, 0.0],
                   {"kind": "glossy", "inputs": {"Color": [0.5] * 3, "Roughness": roughness[0]},
                    "keys": {"Roughness": roughness} if len(roughness) > 1 else {}})
    return {"kind": "scene", "doc": doc, "frames": len(roughness), "labels": labels, "outputs": MAIN, "world": SKY,
            "eevee": RT, "params": {"objects": [floor] + RT_FLOOR, "lights": [RT_SUN],
                                    "camera": [[0.0, -4.0, 1.6], [0.0, 1.5, 0.0]], "lens": 35.0}}


# a corner of white floor, a red wall and a green one with a white block in it, under an area light and a dim world:
# the bounce light fast GI gathers (or, its AMBIENT_OCCLUSION_ONLY method, the occlusion alone)
CORNER = [_plate("floor", [4.0, 4.0], [0.0, 0.0, 0.0], _shade("diffuse", [0.8] * 3)),
          _plate("wall_red", [4.0, 2.5], [-1.5, 0.5, 1.25], _shade("diffuse", [0.8, 0.1, 0.08]), rot=[90.0, 0.0, 90.0]),
          _plate("wall_green", [4.0, 2.5], [0.5, 1.5, 1.25], _shade("diffuse", [0.1, 0.7, 0.15]), rot=[90.0, 0.0, 0.0]),
          {"kind": "box", "name": "block", "size": [0.5, 0.5, 0.8], "at": [-0.9, 0.9, 0.4], "yaw": 15.0,
           "material": _shade("diffuse", [0.8] * 3)}]


def _corner_case(doc, method):
    return {"kind": "scene", "doc": doc, "frames": 1, "outputs": MAIN, "world": {"kind": "uniform", "color": [0.03] * 3},
            "eevee": dict(RT, fast_gi_method=method),
            "params": {"objects": CORNER, "lights": [_area("SQUARE", 1.0, 120.0, place=[[1.4, -1.2, 2.2], [-0.6, 0.6, 0.3]])],
                       "camera": [[1.6, -2.6, 1.5], [-0.6, 0.7, 0.6]], "lens": 30.0}}


# strips glowing at 2.5-160 (white, then a saturated red: its largest channel over the clamp, its luminance under it up
# to 40) over a floor that reflects them: a mirror, a wet asphalt (rough base under a coat, as mk's highway sets it),
# the base alone, the coat alone (one a frame)
RT_CLAMP_FLOOR = _plate("floor", [14.0, 14.0], [0.0, 0.0, 0.0],
                        {"kind": "principled", "inputs": {"Base Color": [0.03, 0.03, 0.05], "Roughness": 0.02},
                         "keys": {"Roughness": [0.02, 0.175, 0.175, 0.175], "Coat Weight": [0.0, 1.0, 0.0, 1.0],
                                  "Coat Roughness": [0.1, 0.1, 0.1, 0.1], "Specular IOR Level": [0.5, 0.4, 0.4, 0.0]}})
RT_CLAMP_STRIPS = [{"kind": "box", "name": f"strip_{name}_{i}", "size": [0.3, 0.05, 0.5], "at": [-1.35 + 0.9 * i, y, 0.4],
                    "material": {"kind": "emission", "inputs": {"Color": color, "Strength": s}}}
                   for name, color, y in (("white", [1.0, 1.0, 1.0], 2.0), ("red", [1.0, 0.05, 0.02], 3.2))
                   for i, s in enumerate((2.5, 10.0, 40.0, 160.0))]
# one width of white emissive card a frame (2, 6, 20 and 60 cm across, 40 cm tall, 10 cm over the floor, strength 4:
# under the clamp) over a black dielectric floor (left) and a Glossy one (right), at roughness 0, 0.1, 0.2 and 0.4: how
# much of a card's light EEVEE's traced reflections find by its width and the floor's roughness (its variants: the
# denoiser's stages, the trace's thickness and quality)
RT_EMIT_WIDTHS = [0.02, 0.06, 0.2, 0.6]
RT_EMIT_FRAMES = [(w, r) for r in (0.0, 0.1, 0.2, 0.4) for w in RT_EMIT_WIDTHS]
RT_EMIT_FLOORS = [
    _plate("diel", [4.0, 16.0], [-2.0, 4.0, 0.0],
           {"kind": "principled", "inputs": {"Base Color": [0.0, 0.0, 0.0], "Roughness": 0.0, "Specular IOR Level": 0.5},
            "keys": {"Roughness": [r for _, r in RT_EMIT_FRAMES]}}),
    _plate("glossy", [4.0, 16.0], [2.0, 4.0, 0.0],
           {"kind": "glossy", "inputs": {"Color": [0.8, 0.8, 0.8], "Roughness": 0.0},
            "keys": {"Roughness": [r for _, r in RT_EMIT_FRAMES]}})]
RT_EMIT_CARDS = [{"kind": "box", "name": f"e_{side}_{round(w * 100)}", "size": [w, 0.02, 0.4],
                  "at": [sgn * x, 2.0, 0.3],
                  "material": {"kind": "emission", "inputs": {"Color": [1.0, 1.0, 1.0], "Strength": 4.0},
                               "keys": {"Strength": [4.0 if fw == w else 0.0 for fw, _ in RT_EMIT_FRAMES]}}}
                 for side, sgn in (("d", -1.0), ("g", 1.0))
                 for w, x in zip(RT_EMIT_WIDTHS, (0.6, 1.2, 1.8, 2.6))]

# point lights of radius 5 cm (jwin's dash glow: 2 W about 0.2 m from the wheel's hub) at 0.1, 0.2, 0.4 and 0.8 m over
# a diffuse grey floor (left) and a dark glossy one (right, roughness 0.5: the wheel's), 2 W times the height squared
# over 0.1 m squared, no shadows (as mk's car lights)
POINT_NEAR_FLOORS = [_plate("diffuse", [4.0, 12.0], [-2.0, 3.0, 0.0], _shade("diffuse", [0.5] * 3)),
                     _plate("glossy", [4.0, 12.0], [2.0, 3.0, 0.0],
                            {"kind": "principled", "inputs": {"Base Color": [0.027, 0.024, 0.056], "Roughness": 0.5,
                                                              "Specular IOR Level": 0.4}})]
POINT_NEAR_LIGHTS = [{"type": "POINT", "energy": 2.0 * (h / 0.1) ** 2, "radius": 0.05, "props": {"use_shadow": False},
                      "place": [[x, 1.8 * i, h], [x, 1.8 * i, 0.0]]}
                     for i, h in enumerate((0.1, 0.2, 0.4, 0.8)) for x in (-1.0, 1.0)]


# Fast GI alone (mk's ray tracing settings, RT) lit by the world and no light, so that a diffuse surface's light is the
# world's times what EEVEE's horizon scan leaves open (and, GLOBAL_ILLUMINATION, plus what it gathers); its variants
# fastgi_ao, noraytrace and cycles_gi split that into occlusion, bounce and the exact answer
def _gi_case(doc, world, objects, views, labels, frames=None):
    return {"kind": "scene", "doc": doc, "frames": frames or len(views), "labels": labels, "outputs": MAIN,
            "world": dict(world, props={"sun_threshold": 0.0}), "eevee": RT,
            "params": {"objects": objects, "camera": views, "lens": 35.0}}


GI_WHITE = {"kind": "uniform", "color": [1.0] * 3}
GI_GREY = _shade("diffuse", [0.8] * 3)
# a white floor and wall, a slab 0.3 m over the floor (a car's underside), a body 10 cm off the wall and one in the open
GI_WALL = ([_plate("floor", [14.0, 14.0], [0.0, 0.0, 0.0], GI_GREY),
            _plate("wall", [5.0, 2.5], [0.0, 1.5, 1.25], GI_GREY, rot=[90.0, 0.0, 0.0]),
            {"kind": "box", "name": "slab", "size": [1.8, 1.0, 0.5], "at": [-1.5, -0.3, 0.55], "material": GI_GREY}]
           + [o for x, y, name in ((0.5, 1.25, "wall"), (1.9, -0.2, "open"))
              for o in ({"kind": "tube", "name": f"body_{name}", "r": 0.15, "h": 1.3, "at": [x, y, 0.0],
                         "material": GI_GREY},
                        {"kind": "sphere", "name": f"head_{name}", "r": 0.12, "at": [x, y, 1.45], "material": GI_GREY})])
# a car's cabin open at the front (its floor 0.3 m up, a roof, a back, sides, a dashboard, a seat), a body seated in it
GI_CABIN = ([_plate("floor", [14.0, 14.0], [0.0, 0.0, 0.0], GI_GREY)]
            + [{"kind": "box", "name": n, "size": s, "at": a, "material": _shade("diffuse", [v] * 3)} for n, s, a, v in (
                ("cabin_floor", [1.6, 2.4, 0.1], [0.0, 0.0, 0.35], 0.3), ("roof", [1.6, 2.4, 0.06], [0.0, 0.0, 1.6], 0.6),
                ("back", [1.6, 0.06, 1.2], [0.0, 1.2, 0.97], 0.6), ("side_l", [0.06, 2.4, 1.2], [-0.8, 0.0, 0.97], 0.6),
                ("side_r", [0.06, 2.4, 1.2], [0.8, 0.0, 0.97], 0.6), ("dash", [1.6, 0.3, 0.3], [0.0, -0.9, 0.85], 0.3),
                ("seat", [1.4, 0.5, 0.45], [0.0, 0.6, 0.62], 0.5), ("seat_back", [1.4, 0.12, 0.6], [0.0, 0.9, 1.15], 0.5))]
            + [{"kind": "tube", "name": "torso", "r": 0.15, "h": 0.5, "at": [0.3, 0.6, 0.85], "material": GI_GREY},
               {"kind": "sphere", "name": "head", "r": 0.1, "at": [0.3, 0.6, 1.45], "material": GI_GREY}])
# a glossy floor rougher than the trace's limit (0.5) under the sky by a white wall, the box and the ball, no light
GI_ROUGH = [0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0]
GI_ROUGH_FLOOR = ([_plate("floor", [14.0, 14.0], [0.0, 0.0, 0.0],
                          {"kind": "glossy", "inputs": {"Color": [0.5] * 3, "Roughness": GI_ROUGH[0]},
                           "keys": {"Roughness": GI_ROUGH}}),
                   _plate("wall", [5.0, 2.5], [0.0, 3.0, 1.25], GI_GREY, rot=[90.0, 0.0, 0.0])] + RT_FLOOR[:2])

# fast GI with lights (the lit scenes' case): a dim world and an area light, so a surface's bounce is mostly its own
# object's light. An L of an orange cushion and back (jwin's car seats), and one with side bolsters as a bucket seat's
GI_DIM = {"kind": "uniform", "color": [0.1] * 3}
GI_ORANGE = _shade("diffuse", [0.8, 0.32, 0.1])
GI_SEAT = ([_plate("floor", [8.0, 8.0], [0.0, 0.0, 0.0], _shade("diffuse", [0.15] * 3))]
           + [{"kind": "box", "name": n, "size": s, "at": a, "material": GI_ORANGE} for n, s, a in (
               ("cushion_l", [0.5, 0.5, 0.12], [-0.35, 0.0, 0.45]), ("back_l", [0.5, 0.1, 0.65], [-0.35, 0.3, 0.8]),
               ("cushion_r", [0.5, 0.5, 0.12], [0.35, 0.0, 0.45]), ("back_r", [0.5, 0.1, 0.65], [0.35, 0.3, 0.8]),
               ("bolster_a", [0.08, 0.14, 0.55], [0.14, 0.18, 0.8]), ("bolster_b", [0.08, 0.14, 0.55], [0.56, 0.18, 0.8]))])
GI_SEAT_LIGHT = _area("SQUARE", 0.5, 40.0, place=[[0.0, -1.0, 1.9], [0.0, 0.15, 0.7]])
# thin red strands 6 mm thick (jwin's red hair) hanging 1, 3, 10 and 30 cm off a white wall, and a curtain of 24 of
# them 7 mm apart 2 cm off it, the key light from the front left
GI_RED = _shade("diffuse", [0.7, 0.06, 0.04])
GI_STRANDS = ([_plate("floor", [8.0, 8.0], [0.0, 0.0, 0.0], _shade("diffuse", [0.15] * 3)),
               _plate("wall", [4.0, 2.5], [0.0, 0.5, 1.25], GI_GREY, rot=[90.0, 0.0, 0.0])]
              + [{"kind": "tube", "name": f"strand_{i}", "r": 0.003, "h": 0.6, "at": [x, 0.497 - g, 0.5],
                  "material": GI_RED} for i, (x, g) in enumerate(((-0.4, 0.01), (-0.28, 0.03), (-0.16, 0.1), (-0.04, 0.3)))]
              + [{"kind": "tube", "name": f"curtain_{i}_{j}", "r": 0.003, "h": 0.6,
                  "at": [0.1 + 0.007 * i, 0.477 - 0.007 * j, 0.5], "material": GI_RED} for i in range(12) for j in range(2)])
GI_STRANDS_LIGHT = _area("SQUARE", 1.0, 120.0, place=[[-0.8, -1.6, 1.7], [0.0, 0.45, 0.8]])


def _gi_lit_case(doc, objects, light, views, labels):
    case = _gi_case(doc, GI_DIM, objects, views, labels)
    case["params"]["lights"] = [light]
    return case



# Motion blur (EEVEE Next: steps 1 renders every sample at the shutter's middle and blurs it along the pixels' motion
# to the shutter's ends; steps N renders N such segments). Rows of things crossing the frame at blur lengths of 6 to
# 300 px (shutter 0.35, 40 mm, 6 m away: MB_PX a metre), over a checkered wall, behind a still pole; frames 2-4 of 5.
MB_PX = 960 * 40.0 / 36.0 / 6.0
MB_ROWS = [("card", 2.2, 6.0), ("sphere", 1.75, 24.0), ("box", 1.3, 60.0), ("card", 0.85, 150.0), ("sphere", 0.4, 300.0)]
MB_COLORS = {"card": [1.0, 1.0, 1.0], "sphere": [0.85, 0.3, 0.2], "box": [0.2, 0.45, 0.85]}


def _mb_row(i, kind, z, blur):
    v = blur / (0.35 * MB_PX)                       # metres a frame for `blur` px at shutter 0.35
    x = -0.9 + 0.45 * i
    o = {"kind": kind, "name": f"{kind}_{blur:g}px", "color": MB_COLORS[kind],
         "keys": {"location": [[1, [x - 2 * v, 0.0, z]], [5, [x + 2 * v, 0.0, z]]]}, "at": [x, 0.0, z]}
    return dict(o, **({"size": 0.2} if kind == "card" else {"r": 0.2} if kind == "sphere" else {"size": [0.3, 0.3, 0.3]}))


MB_SCENE = {"wall": [2.5, 0.4], "light": _area("SQUARE", 2.0, 600.0), "place": [[-2.5, -4.0, 4.0], [0.0, 0.0, 1.0]],
            "camera": [[1, [0.0, -6.0, 1.2], [0.0, 0.0, 1.2]]], "lens": 40.0,
            "objects": [_mb_row(i, *r) for i, r in enumerate(MB_ROWS)]
            + [{"kind": "tube", "name": "pole", "r": 0.05, "h": 3.0, "at": [0.5, -1.5, 0.0], "color": [0.75, 0.75, 0.7]}]}


def _mb_case(doc, blur, params=None, labels=("left", "middle", "right")):
    return {"kind": "motion", "doc": doc, "frames": 5, "refs": [2, 3, 4], "labels": list(labels), "outputs": MAIN,
            "motion_blur": blur, "params": params or MB_SCENE}


MB_SPIN = {"wall": [2.5, 0.4], "light": _area("SQUARE", 2.0, 500.0), "place": [[-2.0, -3.5, 3.5], [0.0, 0.0, 1.0]],
           "camera": [[1, [0.0, -4.0, 1.1], [0.0, 0.0, 1.1]]], "lens": 40.0,
           "objects": [{"kind": "box", "name": "bar", "size": [1.2, 0.04, 0.12], "at": [-1.35, 0.0, 1.1],
                        "color": [0.2, 0.45, 0.85], "keys": {"rotation": [[1, [0.0, 0.0, 0.0]], [5, [0.0, 120.0, 0.0]]]}},
                       {"kind": "sectors", "name": "platter", "r": 0.5, "n": 24, "at": [0.0, 0.0, 1.1],
                        "color": [0.8, 0.15, 0.1], "keys": {"rotation": [[1, [90.0, 0.0, 0.0]], [5, [90.0, 180.0, 0.0]]]}},
                       {"kind": "box", "name": "propeller", "size": [1.0, 0.04, 0.1], "at": [1.35, 0.0, 1.1],
                        "color": [0.9, 0.75, 0.2], "keys": {"rotation": [[1, [0.0, 0.0, 0.0]], [5, [0.0, 360.0, 0.0]]]}}]}
MB_ARM = {"wall": [2.5, 0.4], "light": _area("SQUARE", 2.0, 500.0), "place": [[-2.0, -3.5, 3.5], [0.0, 0.0, 1.0]],
          "camera": [[1, [0.0, -4.0, 1.0], [0.0, 0.0, 1.0]]], "lens": 40.0,
          "arm": {"at": [0.0, 0.0, 0.2], "h": 1.4, "r": 0.08, "lower": [[1, -60.0], [5, 60.0]], "upper": [[1, 0.0], [5, 80.0]]}}
MB_PAN = {"wall": [4.0, 0.4], "light": _area("SQUARE", 2.0, 600.0), "place": [[-2.5, -4.0, 4.0], [0.0, 0.0, 1.0]],
          "camera": [[1, [-0.6, -4.5, 1.2], [-1.5, 0.0, 0.8]], [5, [0.6, -4.5, 1.2], [1.5, 0.0, 0.8]]], "lens": 40.0,
          "objects": [{"kind": "tube", "name": "pole", "r": 0.05, "h": 2.5, "at": [-0.3, -2.0, 0.0], "color": [0.75, 0.75, 0.7]},
                      {"kind": "sphere", "name": "ball", "r": 0.4, "at": [0.6, 0.0, 0.4], "color": [0.85, 0.3, 0.2]},
                      {"kind": "box", "name": "box", "size": [0.6, 0.6, 0.8], "at": [-1.0, 1.5, 0.4], "color": [0.2, 0.45, 0.85]}]}


CASES = {
    "area_square": _light_case("a square area light (a warm key colour)", _area("SQUARE", 1.0, 150.0, color=KEY)),
    "area_rect": _light_case("a rectangular area light", _area("RECTANGLE", 1.6, 150.0, size_y=0.4)),
    "area_disk": _light_case("a disk area light", _area("DISK", 1.0, 150.0)),
    "area_ellipse": _light_case("an elliptic area light", _area("ELLIPSE", 1.6, 150.0, size_y=0.5)),
    "point": _light_case("a point light (10 cm radius)", {"type": "POINT", "energy": 500.0, "radius": 0.1}),
    "spot": _light_case("a spot light (40 degrees, blend 0.25)",
                        {"type": "SPOT", "energy": 800.0, "radius": 0.1, "spot": 40.0, "blend": 0.25}),
    "sun": _light_case("a sun light (2 degrees)", {"type": "SUN", "energy": 3.0, "angle": 2.0}),
    "contact_shadow": {
        "kind": "shadow", "doc": "a box, a ball and a thin pole on the plane, a small area light: shadows where they touch",
        "frames": 2, "labels": ["high", "low"], "outputs": MAIN,
        "params": {"light": _area("SQUARE", 0.1, 60.0),
                   "objects": [{"kind": "box", "name": "cube", "size": [0.6, 0.6, 0.6], "at": [-0.45, 0.25, 0.3],
                                "yaw": 25.0},
                               {"kind": "sphere", "name": "ball", "r": 0.35, "at": [0.45, -0.1, 0.35]},
                               {"kind": "tube", "name": "pole", "r": 0.025, "h": 1.0, "at": [0.15, 0.7, 0.0]}],
                   "places": [[[-1.8, -1.6, 2.6], [0.0, 0.0, 0.0]], [[2.6, -0.8, 0.9], [0.0, 0.2, 0.0]]],
                   "camera": [[1.0, -2.8, 1.5], [0.0, 0.15, 0.25]]}},
    "penumbra": {
        "kind": "shadow", "doc": "a ball and a slab floating over the plane under a big disk light (2 m, then 0.6 m)",
        "frames": 2, "labels": ["2 m", "0.6 m"], "outputs": MAIN,
        "params": {"light": _area("DISK", 2.0, 300.0), "light_keys": {"size": [2.0, 0.6], "size_y": [2.0, 0.6]},
                   "objects": [{"kind": "sphere", "name": "ball", "r": 0.3, "at": [-0.4, 0.0, 0.9]},
                               {"kind": "box", "name": "slab", "size": [0.8, 0.8, 0.04], "at": [0.5, 0.2, 0.5],
                                "yaw": 15.0}],
                   "places": [[[-1.0, -1.2, 3.2], [0.0, 0.0, 0.0]], [[-1.0, -1.2, 3.2], [0.0, 0.0, 0.0]]],
                   "camera": [[0.6, -4.2, 2.6], [0.0, 0.1, 0.3]]}},
    "ggx": _spheres_case(
        "GGX highlights (the Glossy BSDF): multiscatter at six roughnesses, then single-scatter at three",
        [_glossy("MULTI_GGX", r) for r in GGX] + [_glossy("GGX", r) for r in GGX1],
        _area("RECTANGLE", 1.0, 200.0, size_y=0.3), [[-1.5, -1.6, 2.2], [0.0, 0.0, 0.6]],
        [f"multi {r:g}" for r in GGX] + [f"ggx {r:g}" for r in GGX1]),
    "principled": _spheres_case(
        "the Principled BSDF: dielectric, metallic, specular, coat, sheen, transmission",
        [_principled("base", Base_Color=[0.8, 0.3, 0.2], Roughness=0.4),
         _principled("metallic", Base_Color=[0.95, 0.64, 0.37], Metallic=1.0, Roughness=0.25),
         _principled("specular", Base_Color=[0.05, 0.05, 0.05], Roughness=0.2, Specular_IOR_Level=1.0,
                     Specular_Tint=[1.0, 0.6, 0.6]),
         _principled("coat", Base_Color=[0.1, 0.3, 0.8], Roughness=0.5, Coat_Weight=1.0, Coat_Roughness=0.03),
         _principled("sheen", Base_Color=[0.25, 0.05, 0.3], Roughness=0.8, Sheen_Weight=1.0, Sheen_Roughness=0.5),
         _principled("transmission", Base_Color=[0.9, 0.9, 0.9], Roughness=0.05, Transmission_Weight=1.0)],
        _area("SQUARE", 1.0, 200.0), [[-1.6, -1.8, 2.4], [0.0, 0.0, 0.6]],
        ["base", "metallic", "specular", "coat", "sheen", "transmission"]),
    "view_agx": {"kind": "ramp", "frames": 1, "outputs": MAIN, "render": {"view": "AgX"}, "params": RAMP,
                 "doc": "emission strips 2^-10..2^8 through AgX (white, the primaries, their mixes, a skin tone)"},
    "view_standard": {"kind": "ramp", "doc": "the same strips through the Standard view transform",
                      "frames": 1, "outputs": MAIN, "render": {"view": "Standard"}, "params": RAMP},
    "film_edge": {
        "kind": "edge", "doc": "the film filter: a slanted white edge, a grey one, lines 0.3 and 1 px wide, a 0.4 px dot "
                               "(Standard view, 1 px = 1 cm)",
        "frames": 1, "outputs": MAIN, "render": {"view": "Standard"},
        "params": {"width": 9.6, "shapes": [
            {"name": "half", "w": 4.0, "h": 6.0, "at": [-2.6, 0.0], "spin": 4.0},
            {"name": "grey", "w": 1.8, "h": 1.6, "at": [2.0, -1.2], "spin": -3.0, "color": [0.18, 0.18, 0.18]},
            {"name": "vline", "w": 0.003, "h": 3.0, "at": [0.4, 0.5]},
            {"name": "wline", "w": 0.01, "h": 3.0, "at": [0.9, 0.5]},
            {"name": "hline", "w": 2.0, "h": 0.003, "at": [2.2, 1.6]},
            {"name": "dot", "w": 0.004, "h": 0.004, "at": [3.8, 0.4], "strength": 10.0}]}},
    "alpha_hashed": _alpha_case("hashed alpha (DITHERED): an alpha ramp and a half-transparent plane over a checkerboard",
                                "DITHERED"),
    "alpha_blended": _alpha_case("blended alpha (BLENDED): the same planes", "BLENDED"),
    "skinned_bend": {
        "kind": "bend", "doc": "a tube skinned to two bones bending 0-105 degrees (normals of a deformed mesh)",
        "frames": 4, "labels": ["rest", "35", "70", "105"], "outputs": MAIN,
        "params": {"at": [0.0, 0.0, 0.0], "r": 0.15, "h": 1.6, "rings": 32, "blend": 0.15,
                   "lower": [[0, 0, 0], [0, 10, 0], [0, 20, 0], [0, 30, 0]],
                   "upper": [[0, 0, 0], [35, 0, 0], [70, 0, 0], [105, 0, 0]],
                   "light": _area("SQUARE", 1.0, 150.0), "place": [[2.2, -2.4, 2.8], [0.0, 0.0, 0.8]],
                   "camera": [[4.0, -2.2, 1.3], [0.0, 0.0, 0.9]], "lens": 40.0}},
    "motion_blur": {
        "kind": "blur", "doc": "motion blur (shutter 0.35): a ball moving 1 m and a bar turning 30 degrees a frame, then "
                               "the camera panning over the still scene",
        "frames": 10, "refs": [2, 3, 4, 7, 8, 9], "labels": ["object"] * 3 + ["camera"] * 3,
        "outputs": MAIN, "motion_blur": 0.35,
        "params": {"ball": [[1, [-2.5, 0.0, 0.4]], [5, [1.5, 0.0, 0.4]]], "bar_at": [0.0, 1.2, 1.4],
                   "bar": [[1, 0.0], [5, 120.0]], "light": _area("SQUARE", 1.5, 200.0),
                   "place": [[-2.0, -2.5, 3.0], [0.0, 0.0, 0.4]],
                   "camera": [[1, [0.0, -6.0, 1.6], [0.0, 0.0, 0.6]], [6, [0.0, -6.0, 1.6], [0.0, 0.0, 0.6]],
                              [10, [1.2, -5.8, 1.6], [0.9, 0.0, 0.6]]]}},
    "mb_translate": _mb_case("motion blur, shutter 0.35, centred, 1 step: rows crossing the frame at 6, 24, 60, 150 and "
                             "300 px of blur over a checkered wall, behind a still pole", 0.35),
    "mb_start": _mb_case("the rows of mb_translate with the shutter opening on the frame (position START)",
                         {"shutter": 0.35, "position": "START"}),
    "mb_end": _mb_case("the rows of mb_translate with the shutter closing on the frame (position END)",
                       {"shutter": 0.35, "position": "END"}),
    "mb_shutter1": _mb_case("the rows of mb_translate with a shutter of a whole frame", 1.0),
    "mb_steps3": _mb_case("the rows of mb_translate in 3 steps (3 segments of the shutter, each blurred)",
                          {"shutter": 0.35, "steps": 3}),
    "mb_spin": _mb_case("motion blur of turning things: a bar at 30, a disc of 24 sectors at 45 and a propeller at 90 "
                        "degrees a frame, facing the camera", 0.35, MB_SPIN),
    "mb_arm": _mb_case("motion blur of a skinned arm swinging 30 degrees a frame at the shoulder, bending 20 at the elbow",
                       0.35, MB_ARM),
    "mb_pan": _mb_case("motion blur of the camera panning and tracking sideways over a still pole, ball, box and wall",
                       0.35, MB_PAN),
    "dof": {
        "kind": "dof", "doc": "depth of field (85 mm): emission cards at 1, 2, 2.3 and 4 m before a checkered wall at 8 m, "
                              "bright dots for bokeh; f/1.4-8 round, 6 blades, turned, 5 blades, anamorphic, focus "
                              "near and far",
        "frames": len(DOF_KEYS), "outputs": MAIN,
        "labels": ["f/1.4", "f/2.8", "f/5.6", "f/8", "6 blades", "6 blades 15 deg", "5 blades", "ratio 2",
                   "focus 1 m", "focus 8 m"],
        "params": {"camera": [[0.0, -2.0, 1.0], [0.0, 0.0, 1.0]], "lens": 85.0, "keys": DOF_KEYS, "cards": DOF_CARDS,
                   "dots": DOF_DOTS}},
    "dof_psf": {
        "kind": "dof", "doc": "depth of field over black (85 mm, focus 2 m): lines 0.6 px wide at 16 depths, 2 x 2 px "
                              "squares dim, above the sprite threshold and far above, an in-focus edge over a blurred "
                              "one and a blurred one over an in-focus one; f/1.4, 2.8, 5.6, 6 and 5 blades, ratio 2, 0.5",
        "frames": len(DOF_PSF_KEYS), "outputs": MAIN,
        "labels": ["f/1.4", "f/2.8", "f/5.6", "6 blades", "6 blades 30 deg", "5 blades", "ratio 2", "ratio 0.5"],
        "params": {"camera": [[0.0, -2.0, 1.0], [0.0, 0.0, 1.0]], "lens": 85.0, "keys": DOF_PSF_KEYS,
                   "cards": DOF_PSF_CARDS, "dots": []}},
    "mmd_model": {
        "kind": "mmd_model", "doc": "a cast member at rest, its MMD materials under one light a frame (key, fill, rim), "
                                    "then all three; full body and face", "source": True,
        "frames": 4, "labels": ["key", "fill", "rim", "all"], "cycles_samples": 4096,
        "outputs": [{"name": "full", "size": [1080, 1920]}, {"name": "face", "size": [1920, 1080]}],
        "params": {"views": {"full": "full", "face": "face"}, "eye": 0.12, "face_distance": 1.0, "lights": MEMBER_LIGHTS}},
    "mmd_sphere": {
        "kind": "mmd_sphere", "doc": "a sphere in each of a cast member's three most used MMD materials, the key light",
        "source": True, "frames": 3, "outputs": MAIN, "cycles_samples": 4096,
        "params": {"lights": MEMBER_LIGHTS[:1], "camera": [[0.0, -2.6, 1.15], [0.0, 0.0, 1.0]]}},
    "shadow_bias": _shadow_case(
        "plates 0.1-8 mm over the floor under a light 1 cm wide, 0, 30 and 60 degrees off the normal: the normal offset "
        "and depth tolerance of EEVEE's shadow maps at their finest level and a coarser one",
        BIAS_PLATES, _area("SQUARE", 0.01, 30.0),
        [[[0.2 + 3.0 * math.sin(math.radians(a)), 0.0, 3.0 * math.cos(math.radians(a))], [0.2, 0.0, 0.0]] for a in (0, 30, 60)],
        {"near": [[0.2, 0.0, 0.5], [0.2, 0.0, 0.0], 50.0], "far": [[0.2, 0.0, 2.4], [0.2, 0.0, 0.0], 50.0]},
        ["0 deg", "30 deg", "60 deg"]),
    "shadow_thin": _shadow_case(
        "strands 0.5-2 mm thick, a 2 mm card and lash-like slivers 5 mm to 8 cm over the floor under a key light",
        THIN, KEY_LIGHT, _key_places([0.0, 0.0, 0.0]),
        {"near": [[0.02, -0.25, 0.32], [0.02, 0.0, 0.0], 50.0], "far": [[0.4, -2.4, 1.8], [0.0, 0.0, 0.0], 50.0]},
        ["key", "side", "back"]),
    "shadow_layers": _shadow_case(
        "cloth layers: plates 2, 4 and 8 mm off a body block's front and top, a sleeve over a hand, under a key light",
        LAYERS, KEY_LIGHT, _key_places([0.0, 0.0, 0.3]),
        {"near": [[0.15, -0.9, 0.55], [0.05, 0.0, 0.3], 50.0], "far": [[0.6, -4.5, 1.6], [0.05, 0.0, 0.3], 50.0]},
        ["key", "side", "back"]),
    "shadow_hashed": _shadow_case(
        "hashed (DITHERED) see-through occluders: plates of alpha 0.25, 0.5, 0.75 and strands of 0.5 over the floor",
        HASHED, KEY_LIGHT, _key_places([0.0, 0.05, 0.0]),
        {"near": [[0.0, -0.45, 0.45], [0.0, 0.05, 0.0], 50.0], "far": [[0.4, -2.6, 1.8], [0.0, 0.05, 0.0], 50.0]},
        ["key", "side", "back"]),
    "shadow_close": {
        "kind": "shadow", "frames": 4, "labels": ["square", "overhead", "along", "across"],
        "doc": "a close-up of cards 5-50 mm and strands 5-10 mm over the floor under Crimson's key light (1.6 m square, "
               "3.2 m away, 35 degrees off the normal), then overhead, then a 1.6 x 0.25 m rectangle along the strands "
               "and across them",
        "outputs": [{"name": "near", "size": [960, 540]}],
        "params": {"light": _area("RECTANGLE", 1.6, 260.0, size_y=1.6, color=KEY),
                   "light_keys": {"size": [1.6, 1.6, 1.6, 0.25], "size_y": [1.6, 1.6, 0.25, 1.6]},
                   "objects": CLOSE, "floor": 0.6,
                   "places": [[CLOSE_KEY, [0.0, 0.0, 0.0]], [[0.0, 0.0, 3.2], [0.0, 0.0, 0.0]],
                              [CLOSE_KEY, [0.0, 0.0, 0.0]], [CLOSE_KEY, [0.0, 0.0, 0.0]]],
                   "cameras": {"near": [[0.0, -0.2, 0.3], [0.0, 0.0, 0.0], 50.0]}}},
    "world_uniform": _world_case(
        "a uniform world (grey 0.5, then blue at strength 2.5, keyed) over a floor, a diffuse ball and glossy ones of "
        "roughness 0.05-0.7: the background, the probe's and the harmonics' light",
        dict({"kind": "uniform", "color": [0.5] * 3}, keys={"color": [[0.5] * 3, [0.5] * 3, [0.3, 0.5, 1.0]],
                                                            "strength": [1.0, 1.0, 2.5]}),
        [WORLD_FRONT, WORLD_ABOVE, WORLD_FRONT], ["grey", "grey above", "blue x2.5"]),
    "world_sky": _world_case("a sky (ground, horizon, zenith, a soft sun under the extraction threshold) over the balls",
                             SKY, [WORLD_FRONT, WORLD_ABOVE], ["front", "above"]),
    "world_stripes": _world_case("16 sharp bands round the sky: the probe's blur per roughness on the glossy balls",
                                 STRIPES, [WORLD_FRONT, WORLD_ABOVE], ["front", "above"]),
    "world_sun": _world_case("a sky with a sun disc far over the threshold (10): EEVEE's extracted sun light and shadow",
                             SKY_SUN, [WORLD_FRONT, WORLD_ABOVE], ["front", "above"]),
    "world_transparent": _world_case("the sky with the film transparent: unseen, still lighting", SKY, [WORLD_FRONT],
                                     ["front"], render={"film_transparent": True}),
    "world_lightpath": _world_case("a world warm to camera rays (Light Path) and cool to the rest",
                                   {"kind": "lightpath", "camera": [1.0, 0.6, 0.3], "light": [0.2, 0.4, 1.0]},
                                   [WORLD_FRONT], ["front"]),
    "world_visibility": _world_case("the sky with its ray visibility off for the camera and glossy rays",
                                    dict(SKY, visibility={"camera": False, "glossy": False}), [WORLD_FRONT], ["front"]),
    "rt_world": _world_case("the balls under the sky with EEVEE's ray tracing (mk's settings)", SKY,
                            [WORLD_FRONT, WORLD_ABOVE], ["front", "above"], eevee=RT),
    "rt_mirror": _floor_case("a glossy floor (roughness 0.02) reflecting a box, a ball and a lamp above the frame, "
                             "ray traced", [0.02], ["mirror"]),
    "rt_rough": _floor_case("the glossy floor at roughness 0.1-0.7 (screen traced up to 0.5, fast GI over it)",
                            RT_ROUGH, [f"rough {r:g}" for r in RT_ROUGH]),
    "rt_corner": _corner_case("a white floor in a red and green corner with a block, an area light: fast GI's bounce",
                              "GLOBAL_ILLUMINATION"),
    "rt_corner_ao": _corner_case("the corner with fast GI's ambient occlusion alone", "AMBIENT_OCCLUSION_ONLY"),
    "gi_wall": _gi_case("fast GI alone under a white world: a wall, a slab 0.3 m over the floor, a body 10 cm off the "
                        "wall and one in the open, from 3, 6 and 12 m", GI_WHITE, GI_WALL,
                        [[[0.3, -2.4, 1.3], [0.0, 0.5, 0.5]], [[0.6, -5.4, 2.2], [0.0, 0.5, 0.5]],
                         [[1.2, -11.4, 3.8], [0.0, 0.5, 0.5]]], ["3 m", "6 m", "12 m"]),
    "gi_cabin": _gi_case("fast GI alone under a white world: a body seated in a car's cabin open at the front, seen "
                         "through it from 3.6 and 2.2 m", GI_WHITE, GI_CABIN,
                         [[[0.3, -3.0, 1.15], [0.3, 0.6, 1.1]], [[0.3, -1.6, 1.2], [0.3, 0.6, 1.15]]], ["far", "near"]),
    "gi_rough": _gi_case("fast GI on a glossy floor of roughness 0.5-1 under the sky by a white wall, a box and a ball",
                         SKY, GI_ROUGH_FLOOR, [[0.0, -4.0, 1.6], [0.0, 1.5, 0.0]], [f"rough {r:g}" for r in GI_ROUGH],
                         frames=len(GI_ROUGH)),
    "gi_seat": _gi_lit_case("fast GI on two orange car seats (an L of cushion and back, one with side bolsters) under an "
                            "area light and a dim world, from 1.3 m in front, 1.5 m from the side and 3.5 m",
                            GI_SEAT, GI_SEAT_LIGHT,
                            [[[0.0, -1.1, 1.0], [0.0, 0.15, 0.72]], [[-1.0, -0.9, 1.15], [0.1, 0.15, 0.72]],
                             [[0.3, -3.2, 1.3], [0.0, 0.15, 0.7]]], ["near", "side", "far"]),
    "gi_strands": _gi_lit_case("fast GI on thin red strands (6 mm) 1-30 cm off a white wall and a curtain of them 2 cm off "
                               "it, a key light from the front left and a dim world, from 1.2 and 2.3 m",
                               GI_STRANDS, GI_STRANDS_LIGHT,
                               [[[0.0, -0.7, 0.85], [0.0, 0.5, 0.8]], [[0.2, -1.8, 1.0], [0.0, 0.5, 0.8]]],
                               ["near", "far"]),
    "rt_clamp": {"kind": "scene", "doc": "EEVEE's clamp of the light reflected through ray tracing "
                 "(clamp_surface_indirect, 10 as Blender sets it): white and red emissive strips of 2.5-160 over a "
                 "floor, a mirror, then a wet asphalt (rough base under a coat), the base alone, the coat alone",
                 "frames": 4, "labels": ["mirror", "wet", "base", "coat"], "outputs": MAIN, "eevee": RT,
                 "params": {"objects": [RT_CLAMP_FLOOR] + RT_CLAMP_STRIPS, "camera": [[0.0, -4.0, 1.6], [0.0, 1.5, 0.0]],
                            "lens": 35.0}},
    "rt_emit": {"kind": "scene", "doc": "one width of white emissive card a frame (2, 6, 20, 60 cm, strength 4) over a "
                "black dielectric floor and a Glossy one at roughness 0, 0.1, 0.2, 0.4: the traced reflection of an "
                "emitter by its width and the floor's roughness", "frames": len(RT_EMIT_FRAMES),
                "labels": [f"w{round(w * 100)} r{r:g}" for w, r in RT_EMIT_FRAMES], "outputs": MAIN, "eevee": RT,
                "params": {"objects": RT_EMIT_FLOORS + RT_EMIT_CARDS, "camera": [[0.0, -4.0, 1.0], [0.0, 2.0, 0.0]],
                           "lens": 35.0}},
    "point_near": {"kind": "scene", "doc": "point lights 5 cm across, 0.1-0.8 m over a diffuse floor and a dark glossy "
                   "one (roughness 0.5, a steering wheel's), no shadows, their power growing as the height squared: the "
                   "near field of a sphere light (jwin's dash glow on the wheel)", "frames": 1, "labels": ["near"],
                   "outputs": MAIN, "params": {"objects": POINT_NEAR_FLOORS, "lights": POINT_NEAR_LIGHTS,
                                               "camera": [[0.0, -4.5, 4.0], [0.0, 2.6, 0.0]], "lens": 35.0}},
}

# Cycles' samples: its noise (two seeds) at 2048 is 0.03 levels mean, p99 0.8 in the widest penumbra; the cast member
# (hashed hair, fine edges) is noisier and takes `cycles_samples` of its case
CYCLES_SAMPLES = 2048
VARIANTS = {
    "noshadow": {"doc": "EEVEE as the case has it, every light's shadow off", "engine": "EEVEE", "shadows": False},
    "cycles": {"doc": "Cycles, direct light only (no bounce but through transparency), fixed samples, no denoiser",
               "engine": "CYCLES", "shadows": True, "samples": CYCLES_SAMPLES, "seed": 0, "transparent": 64},
    "cycles_noshadow": {"doc": "the same Cycles render, every light's shadow off", "engine": "CYCLES",
                        "shadows": False, "samples": CYCLES_SAMPLES, "seed": 0, "transparent": 64},
    "jittered": {"doc": "EEVEE as the case has it with use_bokeh_jittered on: the camera jittered over the aperture each "
                        "sample (the post-process blur only fills between the samples)",
                 "engine": "EEVEE", "shadows": True, "eevee": {"use_bokeh_jittered": True}},
    "raytrace": {"doc": "EEVEE as the case has it with mk's ray tracing (RT: screen traced, fast GI)", "engine": "EEVEE",
                 "shadows": True, "eevee": RT},
    "fastgi_ao": {"doc": "EEVEE with mk's ray tracing, fast GI as ambient occlusion only", "engine": "EEVEE",
                  "shadows": True, "eevee": dict(RT, fast_gi_method="AMBIENT_OCCLUSION_ONLY")},
    "noraytrace": {"doc": "EEVEE as the case has it without ray tracing (the probes alone)", "engine": "EEVEE",
                   "shadows": True, "eevee": {"use_raytracing": False}},
    "noclamp": {"doc": "EEVEE as the case has it without its clamp of the light reflected through ray tracing and "
                       "probes (clamp_surface_indirect 0)", "engine": "EEVEE", "shadows": True,
                "eevee": {"clamp_surface_indirect": 0.0}},
    "cycles_gi": {"doc": "Cycles with bounces (4 diffuse, 4 glossy): the indirect light ray tracing and fast GI "
                         "approximate, and what a world lights, computed exactly", "engine": "CYCLES", "shadows": True,
                  "samples": CYCLES_SAMPLES, "seed": 0, "transparent": 64, "bounces": 4},
    "fastgi_res1": {"doc": "EEVEE with mk's ray tracing, fast GI at full resolution (fast_gi_resolution 1)",
                    "engine": "EEVEE", "shadows": True, "eevee": dict(RT, fast_gi_resolution="1")},
    "fastgi_ao_res1": {"doc": "EEVEE with mk's ray tracing, fast GI as ambient occlusion only at full resolution",
                       "engine": "EEVEE", "shadows": True,
                       "eevee": dict(RT, fast_gi_method="AMBIENT_OCCLUSION_ONLY", fast_gi_resolution="1")},
    "fastgi_res4": {"doc": "EEVEE with mk's ray tracing, fast GI at quarter resolution (fast_gi_resolution 4)",
                    "engine": "EEVEE", "shadows": True, "eevee": dict(RT, fast_gi_resolution="4")},
    "fastgi_thin": {"doc": "EEVEE with mk's ray tracing, fast GI's near thickness 2 cm (mk's: 25 cm)", "engine": "EEVEE",
                    "shadows": True, "eevee": dict(RT, fast_gi_thickness_near=0.02)},
    "fastgi_far": {"doc": "EEVEE with mk's ray tracing, fast GI's far (angular) thickness 0.1 rad (mk's: pi / 4)",
                   "engine": "EEVEE", "shadows": True, "eevee": dict(RT, fast_gi_thickness_far=0.1)},
    "fastgi_noshadow": {"doc": "EEVEE with mk's ray tracing (fast GI's bounce), every light's shadow off (as jwin's car "
                               "lights)", "engine": "EEVEE", "shadows": False, "eevee": RT},
    "fastgi_ao_noshadow": {"doc": "EEVEE with mk's ray tracing, fast GI as ambient occlusion only, every light's shadow "
                                  "off", "engine": "EEVEE", "shadows": False,
                           "eevee": dict(RT, fast_gi_method="AMBIENT_OCCLUSION_ONLY")},
    "nodenoise": {"doc": "EEVEE with its ray tracing's denoiser off (use_denoise)", "engine": "EEVEE", "shadows": True,
                  "eevee": {"ray_tracing_options": {"use_denoise": False}}},
    "notemporal": {"doc": "EEVEE with the denoiser's temporal accumulation off", "engine": "EEVEE", "shadows": True,
                   "eevee": {"ray_tracing_options": {"denoise_temporal": False}}},
    "nobilateral": {"doc": "EEVEE with the denoiser's bilateral filter off", "engine": "EEVEE", "shadows": True,
                    "eevee": {"ray_tracing_options": {"denoise_bilateral": False}}},
    "trace_thin": {"doc": "EEVEE's ray tracing undenoised, screen trace thickness 0.05 (mk's: 0.2)", "engine": "EEVEE",
                   "shadows": True,
                   "eevee": {"ray_tracing_options": {"use_denoise": False, "screen_trace_thickness": 0.05}}},
    "trace_thick": {"doc": "EEVEE's ray tracing undenoised, screen trace thickness 1 (mk's: 0.2)", "engine": "EEVEE",
                    "shadows": True,
                    "eevee": {"ray_tracing_options": {"use_denoise": False, "screen_trace_thickness": 1.0}}},
    "trace_coarse": {"doc": "EEVEE's ray tracing undenoised, screen trace precision 0 (mk's: 0.5)", "engine": "EEVEE",
                     "shadows": True,
                     "eevee": {"ray_tracing_options": {"use_denoise": False, "screen_trace_quality": 0.0}}},
    "trace_fine": {"doc": "EEVEE's ray tracing undenoised, screen trace precision 1 (mk's: 0.5)", "engine": "EEVEE",
                   "shadows": True,
                   "eevee": {"ray_tracing_options": {"use_denoise": False, "screen_trace_quality": 1.0}}},
}
# The error split of `separate`: (key, its column in the table: E EEVEE, C Cycles, M mkr, n shadows off, test - reference;
# the reference folder, the test folder), each folder of the case holding <output>/
SEPARATION = (("eevee_cycles", "E-C", "variants/cycles", "ref"),
              ("mkr_cycles", "M-C", "variants/cycles", "mkr"),
              ("mkr_eevee", "M-E", "ref", "mkr"),
              ("eevee_cycles_ns", "En-Cn", "variants/cycles_noshadow", "variants/noshadow"),
              ("mkr_cycles_ns", "Mn-Cn", "variants/cycles_noshadow", "mkr_noshadow"),
              ("mkr_eevee_ns", "Mn-En", "variants/noshadow", "mkr_noshadow"))
TERMS = {"eevee": ("ref", "variants/noshadow"), "cycles": ("variants/cycles", "variants/cycles_noshadow"),
         "mkr": ("mkr", "mkr_noshadow")}             # a shadow term: the frame minus the same frame without shadows
TERM_PAIRS = (("eevee_cycles", "eevee", "cycles"), ("mkr_cycles", "mkr", "cycles"), ("mkr_eevee", "mkr", "eevee"))
SOURCES = {"eevee": "ref", "eevee_ns": "variants/noshadow", "cycles": "variants/cycles",
           "cycles_ns": "variants/cycles_noshadow", "mkr": "mkr", "mkr_ns": "mkr_noshadow"}
SHADOW_LEVELS = 4.0                                   # a pixel is in shadow when a term darkens a channel by more
# Pixels of a case's frames the separation measures (probe): a shadow edge's 10-90 % width along a scanline (a row
# `y` from x0 to x1, or a column `x` from y0 to y1, `rows` of them averaged), the mean luma in a box


def _line(name, frame, output="main", **at):
    return {"name": name, "output": output, "frame": frame, "rows": 5, **at}


def _box(name, frame, box, output="main"):
    return {"name": name, "output": output, "frame": frame, "box": box}


PROBES = {
    "penumbra": {"lines": [_line("slab shadow, near edge", 1, x=680, y=[296, 404]),
                           _line("slab shadow, right edge", 1, y=300, x=[700, 860]),
                           _line("slab shadow, near edge", 2, x=680, y=[330, 390]),
                           _line("slab shadow, right edge", 2, y=300, x=[730, 800]),
                           _line("slab shadow, left edge", 2, y=330, x=[544, 612]),
                           _line("ball shadow, left edge", 2, y=290, x=[330, 452])],
                 "boxes": [_box("slab umbra", 1, [640, 290, 720, 320]), _box("ball shadow", 1, [420, 270, 480, 310]),
                           _box("slab umbra", 2, [640, 290, 720, 340]), _box("ball umbra", 2, [430, 280, 470, 300])]},
    "contact_shadow": {"lines": [_line("ball shadow, far edge", 1, x=850, y=[452, 490]),
                                 _line("cube shadow by its base, edge", 1, y=300, x=[466, 494]),
                                 _line("pole shadow, across", 1, x=860, y=[204, 219]),
                                 _line("ball shadow, near edge", 2, x=500, y=[420, 446]),
                                 _line("cube shadow, near edge", 2, x=350, y=[396, 426]),
                                 _line("cube shadow, far near edge", 2, x=120, y=[340, 376])],
                       "boxes": [_box("floor behind the ball", 1, [620, 436, 680, 456]),
                                 _box("by the cube's base", 1, [460, 300, 474, 340]),
                                 _box("ball shadow by its contact", 2, [470, 405, 550, 430]),
                                 _box("cube shadow by its base", 2, [300, 366, 400, 400])]},
    "mmd_model": {"boxes": [_box(f"{what} ({light})", f, box, "face")
                            for f, light in ((1, "key"), (2, "fill"), (4, "all"))
                            for what, box in (("forehead between the bangs: hair's shadow", [905, 575, 985, 635]),
                                              ("forehead centre", [930, 640, 1010, 680]),
                                              ("eye white under the lid", [850, 745, 900, 790]),
                                              ("cheek by the side lock: hair's shadow", [740, 760, 790, 880]),
                                              ("neck under the chin: the head's shadow", [1060, 920, 1160, 990]))]},
}


class LabError(ValueError):
    """A lab case that cannot be built or run."""


def root(arg=None):
    """The lab folder: `arg`, else $MK_LAB, else <user cache>/lab."""
    p = Path(arg or os.environ.get("MK_LAB") or CFG.cache_dir() / "lab").expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p.resolve()


def names(patterns):
    """Case names from names or shell patterns (`area_*`); none: every case."""
    if not patterns:
        return list(CASES)
    out = []
    for pat in patterns:
        hits = [n for n in CASES if fnmatch.fnmatchcase(n, pat)]
        if not hits:
            raise LabError(f"no lab case {pat!r} (cases: {', '.join(CASES)})")
        out += [h for h in hits if h not in out]
    return out


def spec(name):
    """A case as the Blender op and mkr get it: CASES[name] with its name, refs (default every frame), samples and fps."""
    c = dict(CASES[name], name=name)
    c.setdefault("refs", list(range(1, c["frames"] + 1)))
    c.setdefault("samples", 64)
    c["fps"] = FPS
    return c


def write_project(d, case):
    """The case's mk.toml: the project the bake stage and `mk parity frames` read."""
    d.mkdir(parents=True, exist_ok=True)
    lines = [f"# mk parity lab case {case['name']} (mkmmd/render/lab.py): {case['doc']}", "[project]",
             f'name = "lab_{case["name"]}"', f"fps = {FPS}", "frame0 = 1", f"duration = {case['frames'] / FPS!r}",
             'blend = "scene.blend"', ""]
    for o in case["outputs"]:
        lines += ["[[output]]", f'name = "{o["name"]}"', f"size = [{o['size'][0]}, {o['size'][1]}]", ""]
    (d / "mk.toml").write_text("\n".join(lines), encoding="utf-8")
    return Project.load(d)


def source(arg, cast=None):
    """(scene .blend, armature) of the member the MMD cases show: `arg` a project folder (its scene; `cast` one of its
    [[cast]] names, default its only member) or a .blend (`cast` its armature's name, or a part of it)."""
    if arg is None:
        raise LabError("the mmd cases are built from a built scene: pass --source PROJECT_DIR|SCENE.blend [--cast NAME]")
    p = Path(arg).expanduser().resolve()
    if p.suffix == ".blend":
        if not p.is_file():
            raise LabError(f"no scene {p}")
        if not cast:
            raise LabError("--source SCENE.blend needs --cast ARMATURE")
        return p, cast
    try:
        proj = Project.load(p)
    except ProjectError as e:
        raise LabError(str(e)) from None
    if not proj.blend or not proj.blend.is_file():
        raise LabError(f"{p}: no built scene (run `mk build`)")
    members = [c for c in proj.cast if cast is None or c.get("name") == cast]
    if len(members) != 1:
        raise LabError(f"{p}: name one cast member with --cast (it has {', '.join(c.get('name', '?') for c in proj.cast)})")
    return proj.blend, CN.armature_name(members[0])


def is_built(d):
    return (d / "case.json").is_file() and (d / "scene.blend").is_file()


def load_case(d):
    try:
        return json.loads((d / "case.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise LabError(f"no built lab case in {d} ({e}): `mk parity lab --build`") from None


def build(name, lab, src=None, cast=None):
    """Build a case with Blender (scene, Blender's frames, bake) into <lab>/<name>/; its case.json."""
    case = spec(name)
    d = lab / name
    for sub in ("ref", "ref16", "linear", "mkr", "compare", "scene.bake", "variants", "mkr_noshadow", "separate"):
        shutil.rmtree(d / sub, ignore_errors=True)
    proj = write_project(d, case)
    args = {"case": case, "dir": str(d)}
    blend = None
    if case.get("source"):
        blend, args["armature"] = source(src, cast)
    t0 = time.time()
    rep = bridge.run("lab_build", args, blend=blend, project=proj, timeout=3600 * 2, use_serve=False)
    rec = {"case": case, "built": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - t0, 1),
           "source": {"blend": str(blend), "armature": args["armature"]} if blend else None, "report": rep}
    (d / "case.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
    return rec


def rebake(name, lab):
    """Bake a built case's saved scene again with the build's bake stage (`mk build --bake` on the case project)."""
    d = lab / name
    rec = load_case(d)
    proj = Project.load(d)
    t0 = time.time()
    try:
        res = bridge.run("build", {"stages": ["bake"], "out": str(proj.blend), "cache": str(proj.mk_dir / "cache")},
                         blend=proj.blend, project=proj, timeout=3600, use_serve=False)
        bake = dict(res["stages"]["bake"])
        bake["version"] = json.loads((d / "scene.bake" / "bake.json").read_text(encoding="utf-8")).get("version")
    except bridge.BlenderError as e:
        bake = {"error": str(e), "log": e.log}
    bake["seconds"] = round(time.time() - t0, 1)
    rec["report"]["bake"] = bake
    rec["rebaked"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (d / "case.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
    return bake


def variant_dir(d, which):
    return d / "variants" / which


def has_variant(d, which):
    return (variant_dir(d, which) / "variant.json").is_file()


def render_variant(name, lab, which, samples=None, seed=None, device=None):
    """Blender's frames of a variant (VARIANTS) of a built case, rendered from its saved scene by the Blender op
    lab_variant into <case>/variants/<which>/ (<output>/ 8-bit PNG and scene-linear EXR, 16/<output>/ 16-bit PNG) with
    variant.json (the variant and Blender's report); `samples` (else the case's `cycles_samples` for Cycles, else the
    variant's), `seed` (Cycles) and `device` (GPU, CPU) override the variant's. The scene, the references and the bake
    stay as they are."""
    d = lab / name
    rec = load_case(d)
    v = {k: val for k, val in VARIANTS[which].items() if k != "doc"}
    v["name"] = which
    if v["engine"] == "CYCLES" and name in CASES:
        v["samples"] = CASES[name].get("cycles_samples", v["samples"])
    for k, val in (("samples", samples), ("seed", seed), ("device", device)):
        if val is not None:
            v[k] = val
    out = variant_dir(d, which)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    t0 = time.time()
    rep = bridge.run("lab_variant", {"case": rec["case"], "variant": v, "out": str(out)}, blend=d / "scene.blend",
                     project=Project.load(d), timeout=3600 * 6, use_serve=False)
    vrec = {"case": name, "variant": v, "rendered": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seconds": round(time.time() - t0, 1), "report": rep}
    (out / "variant.json").write_text(json.dumps(vrec, indent=1, ensure_ascii=False), encoding="utf-8")
    return vrec


def render(name, lab, out=None, extra=()):
    """mkr's frames of a built case, every output, into out/<output>/ (default the case's mkr/): {output: report}."""
    d = lab / name
    case = load_case(d)["case"]
    out = Path(out) if out is not None else d / "mkr"
    blur = case.get("motion_blur")
    blur = blur if blur is None or isinstance(blur, dict) else {"shutter": blur}
    cfg = {"samples": case.get("samples", 64), "percent": 100, "motion_blur": blur is not None,
           "shutter": (blur or {}).get("shutter", 0.35)}
    reps = {}
    for o in case["outputs"]:
        od = out / o["name"]
        shutil.rmtree(od, ignore_errors=True)
        od.mkdir(parents=True)
        reps[o["name"]] = MKR.draw(d / "scene.bake", o["name"], case["refs"], od, cfg, timeout=3600, extra=extra)
    return reps


def measure(name, lab, test=None, draw=True, out=None, ref=None, **kw):
    """Compare a case's frames (test: a folder holding <output>/, default the case's mkr/) with Blender's (ref: a
    folder holding <output>/, default the case's ref/; a variant's folder): {output: summary}, the sheets and
    summary.json in out/<output>/ (default the case's compare/) when `draw`."""
    d = lab / name
    case = load_case(d)["case"]
    test = Path(test) if test is not None else d / "mkr"
    ref = Path(ref) if ref is not None else d / "ref"
    out = Path(out) if out is not None else d / "compare"
    return {o["name"]: PY.compare(ref / o["name"], test / o["name"], out / o["name"] if draw else None,
                                  frames=case["refs"], draw=draw, **kw) for o in case["outputs"]}


def evaluator(names_, lab, scratch, extra=(), **kw):
    """evaluate(params_file, draw) for `mk parity tune` on lab cases: each case rendered by mkr with those params into
    scratch/<case>/ and measured (`kw`: compare's mask, figure; sheets in scratch/<case>/compare/ when `draw`):
    [(case/output, frame, metrics)]."""
    def evaluate(params_file, draw=False):
        out = []
        for n in names_:
            render(n, lab, out=Path(scratch) / n, extra=["--params", str(params_file), *extra])
            for o, s in measure(n, lab, test=Path(scratch) / n, draw=draw, out=Path(scratch) / n / "compare",
                                **kw).items():
                if s["errors"] or s["only_ref"]:
                    raise PY.ParityError(f"{n}/{o}: mkr left frames out or they cannot be read "
                                         f"({s['only_ref'] or s['errors']})")
                out += [(f"{n}/{o}", f, m) for f, m in s["per_frame"].items()]
        return out
    return evaluate


def labels(rec):
    """{frame: label} of a built case (its spec's labels, or the materials an mmd_sphere case was built with)."""
    case = rec["case"]
    built = (rec.get("report") or {}).get("built") or {}
    return dict(zip(case["refs"], case.get("labels") or list(built.get("materials") or {})))


def rows(name, rec, summaries, seconds=None):
    """Table rows of a case: one per output (pooled over its frames) and one per frame with its label."""
    lab_ = labels(rec)
    out = []
    for o, s in summaries.items():
        out.append({"case": name, "output": o, "frame": None, "label": None, "frames": s["frames"], **_short(s["overall"]),
                    "pass": s["pass"], **({"mkr_seconds": seconds[o]} if seconds and o in seconds else {})})
        for f, fm in s["per_frame"].items():
            out.append({"case": name, "output": o, "frame": f, "label": lab_.get(f), **_short(fm)})
    return out


def _short(m):
    if not m.get("pixels"):
        return {"mean_abs": None}
    return {"mean_abs": round(m["mean_abs"], 3), "bias": [round(b, 2) for b in m["bias"]],
            "over8": round(m["over8"], 3), "over32": round(m["over32"], 3), "p99": m["p99"],
            "psnr": round(m["psnr"], 2) if m["psnr"] is not None else None}


def table(rows_, per_frame=False):
    """The rows as an aligned text table (per-frame rows only with `per_frame`)."""
    head = (f"{'case':<16} {'output':<6} {'frame':<12} {'mean':>7} {'>8 %':>7} {'>32 %':>7} {'p99':>5} {'psnr':>6}"
            "  bias r/g/b")
    lines = [head, "-" * len(head)]
    for r in rows_:
        if r["frame"] is not None and not per_frame:
            continue
        if r.get("mean_abs") is None:
            lines.append(f"{r['case']:<16} {r['output']:<6} {'':<12} {'no frames':>7}")
            continue
        fr = "all" if r["frame"] is None else f"{r['frame']}" + (f" {r['label']}" if r.get("label") else "")
        psnr = f"{r['psnr']:6.2f}" if r["psnr"] is not None else "   inf"
        b = r["bias"]
        lines.append(f"{r['case']:<16} {r['output']:<6} {fr[:12]:<12} {r['mean_abs']:7.3f} {r['over8']:7.3f} "
                     f"{r['over32']:7.3f} {r['p99']:5g} {psnr}  {b[0]:+.2f} {b[1]:+.2f} {b[2]:+.2f}")
    return "\n".join(lines)


# ================================================================================================= separating the error
def _complete(folder, case):
    """Whether a folder of a case holds every reference frame of every output."""
    return all((folder / o["name"] / f"{int(f):05d}.png").is_file() for o in case["outputs"] for f in case["refs"])


def _frame(folder, output, f):
    return PY.read(folder / output / f"{int(f):05d}.png").astype(np.float32)


UMBRA = 0.05                                         # Cycles' umbra: its shadowed luma at most this of its unshadowed


class _Terms:
    """Sums over the shadow terms of frames that pool over frames (shadow_terms)."""

    def __init__(self):
        self.n, self.total, self.depth, self.pairs, self.leak, self.acne = 0, 0, {}, {}, {}, {}

    def add(self, frames, levels):
        """frames: {renderer: (its frame, the same frame without shadows)} (levels, (h, w, 3))."""
        terms = {r: s - u for r, (s, u) in frames.items()}
        shadowed = np.zeros(next(iter(terms.values())).shape[:2], bool)
        for t in terms.values():
            shadowed |= t.min(axis=2) < -levels
        self.n += int(np.count_nonzero(shadowed))
        self.total += shadowed.size
        for r, t in terms.items():
            self.depth[r] = self.depth.get(r, 0.0) + t[shadowed].sum(axis=0, dtype=np.float64)
        for key, test, ref in TERM_PAIRS:
            if test in terms and ref in terms:
                self.pairs.setdefault(key, PY.Stats()).add(PY.stats(terms[ref], terms[test], shadowed))
        if "cycles" in terms:
            s_c, u_c = (x @ PY.LUMA for x in frames["cycles"])
            truth = s_c - u_c
            umbra, lit = (truth < -levels) & (s_c <= UMBRA * u_c), np.abs(truth) <= 0.5
            for r, t in terms.items():
                if r != "cycles":
                    s = frames[r][0] @ PY.LUMA
                    self._sum(self.leak, r, (s - s_c)[umbra].sum(dtype=np.float64), np.count_nonzero(umbra))
                    self._sum(self.acne, r, (t @ PY.LUMA)[lit].sum(dtype=np.float64), np.count_nonzero(lit),
                              np.count_nonzero(t[lit].min(axis=1) < -levels))
        return self

    @staticmethod
    def _sum(table, key, *values):
        table[key] = tuple(a + float(b) for a, b in zip(table.get(key, (0.0,) * len(values)), values))

    def merge(self, other):
        self.n += other.n
        self.total += other.total
        for r, s in other.depth.items():
            self.depth[r] = self.depth.get(r, 0.0) + s
        for k, st in other.pairs.items():
            self.pairs.setdefault(k, PY.Stats()).add(st)
        for table, theirs in ((self.leak, other.leak), (self.acne, other.acne)):
            for r, v in theirs.items():
                self._sum(table, r, *v)
        return self

    def result(self):
        n = self.n
        return {"pixels": n, "share": round(100.0 * n / self.total, 3) if self.total else 0.0,
                "depth": {r: [round(float(x) / n, 3) for x in s] for r, s in self.depth.items()} if n else {},
                "pairs": {k: _short(PY.metrics(st)) for k, st in self.pairs.items()},
                "leak": {r: round(s / c, 3) for r, (s, c) in self.leak.items() if c},
                "acne": {r: {"mean": round(s / c, 3), "over": round(100.0 * k / c, 3)}
                         for r, (s, c, k) in self.acne.items() if c}}


def shadow_terms(d, case, levels=SHADOW_LEVELS):
    """Each renderer's shadow term (TERMS: its frame minus the same frame with shadows off, levels, <= 0 where a
    shadow falls) for the renderers whose frames are there, measured over the shadowed pixels (where any term darkens
    a channel by more than `levels`), per output pooled over the frames ("all") and per frame:
      pixels   the shadowed pixels
      depth    {renderer: [r, g, b]} its mean term there (how much its shadows darken; negative)
      pairs    {key: metrics} (TERM_PAIRS) the test's term against the reference's (bias > 0: lighter shadows)
      leak     {renderer: levels} in Cycles' umbra (its term below -`levels` and its shadowed luma at most UMBRA of
               its unshadowed one) the mean luma of the renderer's frame minus Cycles' (> 0: light leaking into it)
      acne     {renderer: {mean, over}} where Cycles' term is within half a level of 0 (lit): the mean of its term's
               luma (< 0: shadow where Cycles has none) and the % of those pixels it darkens by more than `levels`"""
    have = {r: (d / a, d / b) for r, (a, b) in TERMS.items() if _complete(d / a, case) and _complete(d / b, case)}
    outs = {}
    for o in case["outputs"] if have else ():
        total, per = _Terms(), {}
        for f in case["refs"]:
            acc = _Terms().add({r: (_frame(a, o["name"], f), _frame(b, o["name"], f)) for r, (a, b) in have.items()},
                               levels)
            per[int(f)] = acc.result()
            total.merge(acc)
        outs[o["name"]] = {"all": total.result(), "per_frame": per}
    return {"renderers": sorted(have), "levels": levels, "outputs": outs}


def step_width(profile, lo=0.1, hi=0.9):
    """Walking a profile from its darkest sample toward its brightest: the width (samples, interpolated) over which it
    climbs from `lo` to `hi` of the way between them (a shadow's edge along a scanline: the penumbra), and the two
    values; None when it is flat."""
    p = np.asarray(profile, np.float64)
    a, b = int(np.argmin(p)), int(np.argmax(p))
    dark, bright = float(p[a]), float(p[b])
    if bright <= dark:
        return None
    seg = p[a:b + 1] if a < b else p[b:a + 1][::-1]

    def cross(frac):
        t = dark + frac * (bright - dark)
        i = int(np.argmax(seg >= t))
        return float(i) if i == 0 else i - 1 + (t - seg[i - 1]) / (seg[i] - seg[i - 1])
    return {"width": round(cross(hi) - cross(lo), 2), "dark": round(dark, 2), "bright": round(bright, 2)}


def probe(d, case, probes):
    """Probes measured on the frames each SOURCES folder of a case has (pixels; `probes`: {"lines": [{name, output,
    frame, y, x: [x0, x1] | x, y: [y0, y1], rows}], "boxes": [{name, output, frame, box: [x0, y0, x1, y1]}]}): a
    line's step_width of the mean luma of `rows` rows (5) round row y from x0 to x1 (or of as many columns round
    column x from y0 to y1), on each renderer's shadowed frames; a box's mean luma on every source, and each
    renderer's darkening there (its frame without shadows minus its frame)."""
    res = {"lines": [], "boxes": []}

    def luma(src, p):
        path = d / SOURCES[src] / p["output"] / f"{int(p['frame']):05d}.png"
        return PY.read(path).astype(np.float32) @ PY.LUMA if path.is_file() else None
    for p in (probes or {}).get("lines", []):
        n, across = int(p.get("rows", 5)), isinstance(p["x"], (list, tuple))
        if across:
            (x0, x1), y0 = (int(v) for v in p["x"]), int(p["y"]) - n // 2
            x0, y0, x1, y1 = _inside(case, p, (x0, y0, x1, y0 + n))
        else:
            (y0, y1), x0 = (int(v) for v in p["y"]), int(p["x"]) - n // 2
            x0, y0, x1, y1 = _inside(case, p, (x0, y0, x0 + n, y1))
        got = {}
        for src in TERMS:
            im = luma(src, p)
            if im is not None:
                got[src] = step_width(im[y0:y1, x0:x1].mean(axis=0 if across else 1))
        res["lines"].append(dict(p, measured=got))
    for p in (probes or {}).get("boxes", []):
        x0, y0, x1, y1 = _inside(case, p, p["box"])
        got = {}
        for src in SOURCES:
            im = luma(src, p)
            if im is not None:
                got[src] = round(float(im[y0:y1, x0:x1].mean()), 2)
        dark = {r: round(got[r + "_ns"] - got[r], 2) for r in TERMS if r in got and r + "_ns" in got}
        res["boxes"].append(dict(p, luma=got, darkening=dark))
    return res


def _inside(case, p, box):
    """A probe's box (x0, y0, x1, y1, pixels) when it lies in its output's frame and its frame is a reference."""
    size = {o["name"]: o["size"] for o in case["outputs"]}
    x0, y0, x1, y1 = (int(v) for v in box)
    if p.get("output") not in size or int(p.get("frame", -1)) not in case["refs"]:
        raise LabError(f"probe {p.get('name')!r}: no output {p.get('output')!r} or frame {p.get('frame')} in the case")
    w, h = size[p["output"]]
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise LabError(f"probe {p.get('name')!r}: {x0},{y0},{x1},{y1} is not inside the {w}x{h} frame")
    return x0, y0, x1, y1


def separate(name, lab, draw=True, figure=None, probes=None, levels=SHADOW_LEVELS):
    """A case's error split into its parts, from the frames on disk (ref/, variants/, mkr/, mkr_noshadow/):
    {"comparisons": {key: {output: compare's summary}} for each SEPARATION entry whose folders are complete (sheets in
    separate/<key>/<output>/ when `draw`; `figure` as compare's), "missing": {key: [the folders it lacks]}, "shadow":
    shadow_terms, "probes": probe (`probes`, default PROBES[name])}."""
    d = lab / name
    case = load_case(d)["case"]
    res = {"comparisons": {}, "missing": {}}
    for key, _, ref, test in SEPARATION:
        lack = [x for x in (ref, test) if not _complete(d / x, case)]
        if lack:
            res["missing"][key] = lack
            continue
        res["comparisons"][key] = {
            o["name"]: PY.compare(d / ref / o["name"], d / test / o["name"],
                                  d / "separate" / key / o["name"] if draw else None, frames=case["refs"], draw=draw,
                                  figure=figure) for o in case["outputs"]}
    res["shadow"] = shadow_terms(d, case, levels)
    res["probes"] = probe(d, case, PROBES.get(name) if probes is None else probes)
    return res


def separation_brief(sep):
    """A separation without the per-frame tables of its comparisons (what the command prints with the rows)."""
    return {"comparisons": {k: {o: PY.brief(s) for o, s in sums.items()} for k, sums in sep["comparisons"].items()},
            "missing": sep["missing"], "shadow": sep["shadow"], "probes": sep["probes"]}


def separation_rows(name, rec, sep):
    """Rows of a case's separation: per output one pooled over its frames and one per frame (with its label)."""
    lab_, case, out = labels(rec), rec["case"], []
    for o in (x["name"] for x in case["outputs"]):
        sh = sep["shadow"]["outputs"].get(o) or {}
        for f in [None, *case["refs"]]:
            comps = {}
            for key, sums in sep["comparisons"].items():
                m = sums[o]["overall"] if f is None else sums[o]["per_frame"].get(int(f))
                comps[key] = _short(m) if m else {"mean_abs": None}
            out.append({"case": name, "output": o, "frame": f, "label": None if f is None else lab_.get(f),
                        "comparisons": comps, "shadow": sh.get("all") if f is None else sh.get("per_frame", {}).get(int(f))})
    return out



def _cell(m):
    """mean |error| and the bias averaged over the channels (13 characters), or a dash."""
    if not m or m.get("mean_abs") is None:
        return f"{'-':>13}"
    return f"{m['mean_abs']:6.2f} {sum(m['bias']) / 3:+6.2f}"


def _num(x, width, sign="+"):
    return f"{'-':>{width}}" if x is None else format(x, f">{sign}{width}.1f")


def separation_table(rows_, per_frame=False):
    """Separation rows as an aligned text table (per-frame rows only with `per_frame`): for each comparison (its
    SEPARATION column) the mean |error| and the bias averaged over the channels; then over the shadowed pixels (their %
    of the frame) each renderer's mean shadow darkening (luma, dE dC dM), the mean |difference| of the terms (sE-C sM-C
    sM-E) and the light leaking into Cycles' umbra (shadow_terms' leak: luma, > 0 lighter)."""
    keys = [(k, col) for k, col, *_ in SEPARATION]
    head = (f"{'case':<16} {'output':<6} {'frame':<12} " + " ".join(f"{col:>13}" for _, col in keys) +
            f"  | {'shad %':>6} {'dE':>6} {'dC':>6} {'dM':>6} {'sE-C':>5} {'sM-C':>5} {'sM-E':>5} {'leakE':>6} {'leakM':>6}")
    lines = [head, "-" * len(head)]
    for r in rows_:
        if r["frame"] is not None and not per_frame:
            continue
        fr = "all" if r["frame"] is None else f"{r['frame']}" + (f" {r['label']}" if r.get("label") else "")
        cells = " ".join(_cell(r["comparisons"].get(k)) for k, _ in keys)
        sh, tail = r.get("shadow") or {}, ""
        if sh.get("pixels"):
            depth = {k: float(np.dot(v, PY.LUMA)) for k, v in sh["depth"].items()}
            pairs = {k: v.get("mean_abs") for k, v in sh["pairs"].items()}
            tail = (f"  | {sh['share']:6.2f} " + " ".join(_num(depth.get(k), 6) for k in ("eevee", "cycles", "mkr")) +
                    " " + " ".join(_num(pairs.get(k), 5, "") for k in ("eevee_cycles", "mkr_cycles", "mkr_eevee")) +
                    " " + " ".join(_num(sh["leak"].get(k), 6) for k in ("eevee", "mkr")))
        lines.append(f"{r['case']:<16} {r['output']:<6} {fr[:12]:<12} {cells}{tail}")
    return "\n".join(lines)


def probe_table(name, probes):
    """A case's probes as text lines: a line's penumbra width (px) and a box's mean luma and darkening per source."""
    lines = []
    for p in probes.get("lines", []):
        got = "  ".join(f"{k} {v['width']:.1f}" for k, v in p["measured"].items() if v)
        lines.append(f"{name:<16} {p['output']:<6} {p['frame']:<3} {p['name'][:48]:<48} 10-90 % width px  {got}")
    for p in probes.get("boxes", []):
        got = "  ".join(f"{k} {v:.1f}" for k, v in p["luma"].items())
        dark = "  ".join(f"{k} {v:+.1f}" for k, v in p["darkening"].items())
        lines.append(f"{name:<16} {p['output']:<6} {p['frame']:<3} {p['name'][:48]:<48} luma  {got}"
                     + (f"  | darkening  {dark}" if dark else ""))
    return lines
