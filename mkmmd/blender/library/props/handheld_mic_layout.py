"""Layout of the handheld vocal microphone (pure Python + numpy, no bpy: importable from tests): its profiles, the colour
roles and the card.

Frame (metres): the mic stands on its tail along +Z, the cable connector's rim at z = 0 and the top of the ball grille at
z = LENGTH; it is round about Z. The card's anchor `jack` is where the cord leaves the connector (under the tail, `dir`
-Z: `attach` with `cable` hangs the cord from it, docs/design.md: Posing); the look points are `grille` (the ball's
centre: put it at a mouth) and `grip` (the middle of the handle: where a fist closes). A dynamic vocal mic of the classic
shape in proportion: 162 mm long, a 51 mm ball over a collar and a handle tapering from 43 to 31 mm. No logo, badge or
lettering.

Colour roles (`slots` of the prop: a palette slot or a hex): body (base: the handle and collar), grille (text: the ball),
hardware (subtle: the connector), cable (text: the cord)."""
import math

import numpy as np

LENGTH = 0.162
TAIL_H, TAIL_R = 0.013, 0.0124          # the connector shell: z 0 .. TAIL_H
HANDLE = (0.0155, 0.0215)               # the handle's radius just above the connector and under the collar
NECK = 0.104                            # where the handle meets the collar
COLLAR = (0.0234, 0.1165)               # the collar under the ball: radius, top
BALL_R = 0.0255
BALL_C = LENGTH - BALL_R                # the ball's centre
GRIP_Z = 0.060
CABLE_R = 0.0030
ROLES = {"body": "base", "grille": "text", "hardware": "subtle", "cable": "text"}


def connector_profile():
    """(r, z) of the connector shell, axis to axis: a short metal sleeve with rounded rims."""
    return np.array([(0.0, 0.0), (TAIL_R - 0.0012, 0.0), (TAIL_R, 0.0012), (TAIL_R, TAIL_H), (0.0, TAIL_H)], float)


def body_profile(n=10):
    """(r, z) of the handle and the collar, axis to axis: the handle swells from HANDLE[0] to HANDLE[1] (an eased taper,
    fuller toward the top as a hand holds it), steps out to the collar and rounds over to the axis under the ball."""
    t = np.linspace(0.0, 1.0, n)
    z = TAIL_H - 0.0005 + (NECK - TAIL_H + 0.0005) * t
    r = HANDLE[0] + (HANDLE[1] - HANDLE[0]) * np.sin(0.5 * math.pi * t) ** 1.2
    pts = [(0.0, TAIL_H - 0.0005)] + list(zip(r, z))
    pts += [(COLLAR[0] - 0.0006, NECK + 0.0008), (COLLAR[0], NECK + 0.0024), (COLLAR[0], COLLAR[1] - 0.0016),
            (COLLAR[0] - 0.0010, COLLAR[1]), (0.0, COLLAR[1])]
    return np.array(pts, float)


def grille_profile(n=28):
    """(r, z) of the ball grille, from where it sits in the collar up over the top to the axis."""
    z0 = COLLAR[1] - 0.0030
    a0 = math.asin((z0 - BALL_C) / BALL_R)
    a = np.linspace(a0, 0.5 * math.pi, n)
    pts = [(0.0, z0)] + [(BALL_R * math.cos(x), BALL_C + BALL_R * math.sin(x)) for x in a]
    pts[-1] = (0.0, LENGTH)
    return np.array(pts, float)


def card(name, slots=None, size=None):
    """The prop card (docs/design.md: Prop card): the jack anchor the cord hangs from, the grille and grip look points,
    a hidden cylinder collider round the handle and the ball."""
    jack = {"name": "jack", "point": [0.0, 0.0, -0.002], "dir": [0.0, 0.0, -1.0], "object": f"{name}_jack"}
    looks = [{"name": "grille", "point": [0.0, 0.0, round(BALL_C, 5)]}, {"name": "grip", "point": [0.0, 0.0, GRIP_Z]}]
    return {"size": [round(float(v), 4) for v in (size if size is not None else (2 * BALL_R, 2 * BALL_R, LENGTH))],
            "origin": "tail", "front": "+Z", "slots": dict(slots or {}), "use": {"anchor": [jack], "look": looks},
            "colliders": [{"type": "cylinder", "object": f"{name}_col_body", "tag": name}]}
